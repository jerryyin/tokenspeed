# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""EP route finalize of the opt-in A4W4 EP registration (no GPU needed).

``TOKENSPEED_A4W4_EP_FUSED_FINALIZE=1`` makes
``gluon_mxfp4_precomputed_ep_moe_apply`` finalize the deferred direct decode's
routes (rows 3-8) with ``ep_route_finalize`` (one launch) instead of six eager
tensor ops. Without a GPU the kernel runs in Triton's interpreter on CPU
tensors. The GPU tests beside the kernel are in test_gluon_mxfp4_amd.py
(``test_precomputed_ep_apply_shards_sum_to_tp_apply[...-fused]``).
"""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

_FINALIZE = "tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.ep_route_finalize"


@pytest.fixture
def finalize_module(monkeypatch):
    """The finalize module, interpreted on CPU when no GPU is visible."""
    on_gpu = torch.cuda.is_available()
    if not on_gpu:
        monkeypatch.setenv("TRITON_INTERPRET", "1")
    module = importlib.import_module(_FINALIZE)
    if not on_gpu:
        module = importlib.reload(module)
    yield module, ("cuda" if on_gpu else "cpu")
    if not on_gpu:
        monkeypatch.delenv("TRITON_INTERPRET")
        importlib.reload(sys.modules[_FINALIZE])


def _routes(
    rows,
    top_k,
    hidden,
    rank,
    ep_size,
    num_experts,
    device,
    seed,
    weight_dtype=torch.float32,
):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand((rows, num_experts), generator=generator)
    weights, ids = torch.topk(scores, top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(weight_dtype)
    num_local = num_experts // ep_size
    owned = (ids // num_local) == rank
    route_map = torch.where(
        owned.reshape(-1), torch.arange(rows * top_k), torch.tensor(-1)
    ).to(torch.int32)
    routes = torch.randn((rows * top_k, hidden), generator=generator).to(
        torch.bfloat16
    )
    # Rows of other ranks' routes are undefined: poison them.
    routes[~owned.reshape(-1)] = float("nan")
    return (
        routes.to(device),
        route_map.to(device),
        weights.to(device),
        owned.to(device),
    )


def _slot_order_reference(routes, route_map, weights):
    """FP32 products rounded, then summed in slot order, then BF16."""
    rows, top_k = weights.shape
    acc = torch.zeros((rows, routes.shape[1]), dtype=torch.float32)
    routes, route_map, weights = routes.cpu(), route_map.cpu(), weights.cpu()
    for slot in range(top_k):
        row = torch.arange(rows) * top_k + slot
        owned = (route_map[row] >= 0)[:, None]
        value = torch.where(owned, routes[row].float(), 0.0)
        acc = acc + value * weights[:, slot].float()[:, None]
    return acc.to(torch.bfloat16)


def _eager_epilogue(routes, route_map, weights, out_dtype=torch.bfloat16):
    """The registration's eager finalize (switch off), verbatim."""
    rows, top_k = weights.shape
    local = (route_map >= 0).view(rows, top_k, 1)
    routes = torch.where(local, routes.view(rows, top_k, -1).float(), 0.0)
    combined = (routes * weights.view(rows, top_k, 1).float()).sum(dim=1)
    return combined.to(out_dtype)


@pytest.mark.parametrize(
    "rows,top_k,hidden,num_experts",
    [
        (3, 6, 5120, 384),  # target, 3 rows
        (6, 6, 5120, 384),  # target, one request's 6 verify rows
        (8, 6, 5120, 384),
        (6, 3, 5120, 128),  # DSpark draft shape (128 experts, top-3)
        (5, 3, 300, 16),  # width not a multiple of the 1024-column block
    ],
)
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_ep_route_finalize_matches_slot_order_reference(
    finalize_module, rows, top_k, hidden, num_experts, rank, weight_dtype
):
    module, device = finalize_module
    routes, route_map, weights, owned = _routes(
        rows,
        top_k,
        hidden,
        rank,
        4,
        num_experts,
        device,
        20261122 + rows,
        weight_dtype,
    )
    out = module.ep_route_finalize(routes, route_map, weights)
    assert out.dtype == torch.bfloat16 and out.shape == (rows, hidden)
    assert not torch.isnan(out).any(), "read a row of another rank's route"
    expected = _slot_order_reference(routes, route_map, weights)
    # Bit-exact: the same FP32 products and slot-order accumulation.
    assert torch.equal(out.cpu(), expected)
    # Tokens with no route on this rank are exactly zero.
    assert torch.all(out.cpu()[~owned.cpu().any(dim=1)] == 0)
    # Against the eager epilogue (whose sum order torch does not fix): equal
    # within one BF16 rounding of the FP32 sum.
    clean = torch.where(
        (route_map >= 0)[:, None], routes, torch.zeros_like(routes)
    )
    eager = _eager_epilogue(clean.cpu(), route_map.cpu(), weights.cpu())
    torch.testing.assert_close(
        out.cpu().float(), eager.float(), atol=1e-6, rtol=2**-8
    )


def test_ep_route_finalize_writes_out_and_handles_zero_rows(finalize_module):
    module, device = finalize_module
    routes, route_map, weights, _ = _routes(6, 6, 5120, 1, 4, 384, device, 7)
    out = torch.full((6, 5120), 3.0, dtype=torch.bfloat16, device=device)
    result = module.ep_route_finalize(routes, route_map, weights, out=out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out.cpu(), _slot_order_reference(routes, route_map, weights))
    empty = module.ep_route_finalize(
        torch.empty((0, 64), dtype=torch.bfloat16, device=device),
        torch.empty((0,), dtype=torch.int32, device=device),
        torch.empty((0, 6), dtype=torch.float32, device=device),
    )
    assert empty.shape == (0, 64)
    with pytest.raises(ValueError, match="route_map"):
        module.ep_route_finalize(routes, route_map.long(), weights)
    with pytest.raises(ValueError, match="out must be"):
        module.ep_route_finalize(
            routes, route_map, weights, out=torch.empty((6, 64), device=device)
        )


def _gluon_module():
    gluon = importlib.import_module("tokenspeed_kernel.ops.moe.gluon")
    if not hasattr(gluon, "gluon_mxfp4_precomputed_ep_moe_apply"):
        pytest.skip("the AMD MoE registrations are not defined on this platform")
    return gluon


def test_fused_finalize_switch_follows_environment():
    """Off unless TOKENSPEED_A4W4_EP_FUSED_FINALIZE=1 at import (the default
    switch set runs with it unset, so this pins the default off)."""
    import os

    gluon = _gluon_module()
    expected = os.environ.get("TOKENSPEED_A4W4_EP_FUSED_FINALIZE", "0") == "1"
    assert gluon._A4W4_EP_FUSED_FINALIZE is expected


@pytest.mark.parametrize("rows", [2, 3, 6, 8, 9])
@pytest.mark.parametrize("fused", [False, True])
def test_ep_registration_finalize_paths(finalize_module, monkeypatch, rows, fused):
    """The registration hands the deferred routes (rows 3-8) to the fused
    finalize only with the switch; other rows and the switch-off path are
    unchanged. The fused MoE itself is replaced by a recorder."""
    module, device = finalize_module
    gluon = _gluon_module()
    fused_moe = importlib.import_module(
        "tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.moe"
    )
    top_k, hidden, num_experts, ep_size, rank = 6, 5120, 384, 4, 2
    routes, route_map, weights, _ = _routes(
        rows, top_k, hidden, rank, ep_size, num_experts, device, 11 + rows
    )
    ids = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
    calls = []
    finalized = torch.full((rows, hidden), 5.0, dtype=torch.bfloat16, device=device)

    def recorder(x, topk_weights, topk_ids, *args, do_finalize, out, **kwargs):
        calls.append(
            (do_finalize, kwargs["expert_start"], kwargs["global_num_experts"])
        )
        if do_finalize:
            return finalized
        return routes, None, route_map

    monkeypatch.setattr(gluon, "gluon_mxfp_precomputed_mxfp4_fused_moe", recorder)
    monkeypatch.setattr(gluon, "_A4W4_EP_FUSED_FINALIZE", fused)
    w = SimpleNamespace(
        num_experts=num_experts,
        num_local_experts=num_experts // ep_size,
        ep_rank=rank,
        ep_size=ep_size,
        w13_precision_config=SimpleNamespace(b_mx_scale=None),
        w2_precision_config=SimpleNamespace(b_mx_scale=None, out_dtype=None),
        w13_weight_triton_tensor=None,
        w2_weight_triton_tensor=None,
    )
    x = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=device)
    result = gluon.gluon_mxfp4_precomputed_ep_moe_apply(
        {}, x, w, None, topk_weights=weights, topk_ids=ids
    )
    deferred = (
        fused_moe._DIRECT_DECODE_MAX_M < rows < fused_moe._PACKAGE_PREFILL_MIN_M
    )
    assert calls == [(not deferred, rank * (num_experts // ep_size), num_experts)]
    if not deferred:
        assert result is finalized
        return
    assert not torch.isnan(result).any()
    expected = _slot_order_reference(routes, route_map, weights)
    if fused:
        assert torch.equal(result.cpu(), expected)
    else:
        torch.testing.assert_close(
            result.cpu().float(), expected.float(), atol=1e-6, rtol=2**-8
        )
