# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT
"""Distributed Gluon Petit numerics for DeepSeek V4.1 Flash at EP4 and EP8.

DRAFT (CPU-authored, not yet run on GPUs). Launch with the venv interpreter;
plain /usr/local/bin/torchrun uses the system python and fails::

    WS=8   # or 4 with HIP_VISIBLE_DEVICES=0,1,2,3
    $VENV/bin/python -m torch.distributed.run --standalone --nproc_per_node=$WS \
        -m pytest --import-mode=importlib -p no:cacheprovider -rA <this file>

The world size is taken from ``WORLD_SIZE`` (must be 4 or 8). Each rank owns
``num_experts // WORLD_SIZE`` contiguous experts (V4.1 Flash 384 -> 96 / 48,
DSpark 128 -> 32 / 16).

The oracle is independent of Petit: it decodes the same packed MXFP4 bytes
(E2M1 nibbles + E8M0 scales) with a pure-PyTorch table, runs every routed
expert densely in FP32, applies ``silu(min(gate, 10)) * clamp(up, -10, 10)``,
simulates the MXFP4 input and intermediate quantisation, and rounds each
weighted route to BF16 before the FP32 combine (as main's
test_gluon_petit_distributed.py does). Weights have one nonzero per row, so
both matmuls are exact in FP32 and the tolerance can stay tight.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
import tokenspeed_kernel
import torch
import torch.distributed as dist
from utils import is_cdna4

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_SUPPORTED_WORLD_SIZES = (4, 8)
_CAPACITY = 1024
_CLAMP = 10.0
_SCALE_CODE = 123  # E8M0 2**-4
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class _V41Profile:
    name: str
    experts: int
    top_k: int
    hidden: int = 5120
    logical_intermediate: int = 2304  # zero-padded to 2560 by the adapter
    compute_intermediate: int = 2560


_PROFILES = {
    "dsv41_flash": _V41Profile("dsv41_flash", experts=384, top_k=6),
    "dsv41_flash_dspark": _V41Profile("dsv41_flash_dspark", experts=128, top_k=3),
}


def _parameter(shape, device, dtype=torch.uint8):
    return torch.nn.Parameter(
        torch.zeros(shape, dtype=dtype, device=device), requires_grad=False
    )


class _V41MoeLayer(torch.nn.Module):
    """The attributes the Petit adapter reads from a V4.1 MoELayer."""

    def __init__(self, profile: _V41Profile, world_size: int, device) -> None:
        super().__init__()
        local = profile.experts // world_size
        inter, hidden = profile.logical_intermediate, profile.hidden
        self.num_experts = profile.experts
        self.top_k = profile.top_k
        self.hidden_size = hidden
        self.intermediate_size = inter
        self.num_local_experts = local
        self.ep_size = world_size
        self.tp_size = 1
        self.activation = "swiglu"
        self.swiglu_beta = None
        self.swiglu_arg = argparse.Namespace(alpha=None, limit=_CLAMP)
        self.w13_input_layout = "concatenated"
        self.register_parameter("w13_weight", _parameter((local, 2 * inter, hidden // 2), device))
        self.register_parameter("w13_weight_scale", _parameter((local, 2 * inter, hidden // 32), device))
        self.register_parameter("w2_weight", _parameter((local, hidden, inter // 2), device))
        self.register_parameter("w2_weight_scale", _parameter((local, hidden, inter // 32), device))
        self.register_parameter("w13_weight_bias", None)
        self.register_parameter("w2_weight_bias", None)


def _make_plan(profile: _V41Profile, world_size: int) -> dict:
    # Same keywords MoELayer passes for V4.1 (expert.py: swiglu_form "standard",
    # activation_clamped because swiglu_arg.limit is set).
    return tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="gluon_petit",
        ep_size=world_size,
        ispp=profile.logical_intermediate,
        fp8_scale_block_shape=None,
        internal_activation_dtype="mxfp4",
        with_bias=False,
        process_group=None,
        hidden=profile.hidden,
        swiglu_form="standard",
        activation_clamped=True,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        solution="gluon",
    )


# --------------------------------------------------------------------------
# Deterministic packed MXFP4 expert weights (a function of the global expert id)
# --------------------------------------------------------------------------


def _expert_packed(profile: _V41Profile, expert: int, device, *, tail: bool):
    """Packed (w13, w13_scale, w2, w2_scale) for one global expert.

    One nonzero E2M1 code per row. Gate rows are positive; up rows alternate
    sign so the up clamp is exercised at both +10 and -10. ``tail`` moves the
    w2 reads onto the last logical intermediate block (2272..2303), right next
    to the 2304 -> 2560 zero padding.
    """
    inter, hidden = profile.logical_intermediate, profile.hidden
    rows = torch.arange(2 * inter, device=device)
    is_gate = rows < inter
    channel = rows % inter
    cols = (channel * 7 + torch.where(is_gate, 0, 3)) % hidden
    magnitude = torch.where(is_gate, 2 + expert % 3, 2 + (expert + 1) % 3)
    negative = (~is_gate) & (channel % 2 == 1)
    codes = (magnitude + torch.where(negative, 8, 0)).to(torch.uint8)
    w13 = torch.zeros((2 * inter, hidden // 2), dtype=torch.uint8, device=device)
    w13[rows, cols // 2] = (codes << ((cols % 2) * 4)).to(torch.uint8)
    out_rows = torch.arange(hidden, device=device)
    if tail:
        out_cols = inter - 32 + (out_rows * 5 + expert) % 32
    else:
        out_cols = (out_rows * 11 + expert) % inter
    w2 = torch.zeros((hidden, inter // 2), dtype=torch.uint8, device=device)
    w2[out_rows, out_cols // 2] = ((2 + expert % 3) << ((out_cols % 2) * 4)).to(torch.uint8)
    w13_scale = torch.full((2 * inter, hidden // 32), _SCALE_CODE, dtype=torch.uint8, device=device)
    w2_scale = torch.full((hidden, inter // 32), _SCALE_CODE, dtype=torch.uint8, device=device)
    return w13, w13_scale, w2, w2_scale


def _fill_layer(layer, profile, rank, world_size, *, tail: bool) -> None:
    local = profile.experts // world_size
    for i in range(local):
        w13, s13, w2, s2 = _expert_packed(profile, rank * local + i, layer.w13_weight.device, tail=tail)
        layer.w13_weight.data[i].copy_(w13)
        layer.w13_weight_scale.data[i].copy_(s13)
        layer.w2_weight.data[i].copy_(w2)
        layer.w2_weight_scale.data[i].copy_(s2)


# --------------------------------------------------------------------------
# Independent pure-PyTorch reference
# --------------------------------------------------------------------------


def _dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode [rows, cols/2] E2M1 bytes (low nibble = even column) with E8M0 scales."""
    table = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(packed.shape[0], -1).long()
    values = table[nibbles & 7] * torch.where(nibbles >= 8, -1.0, 1.0)
    factor = torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=1)
    return values * factor


def _quantized_reference(value: torch.Tensor, *, input_quantization: bool) -> torch.Tensor:
    """MXFP4 block-32 scales and nearest-even E2M1 values (main's oracle)."""
    shape = value.shape
    blocks = value.float().reshape(-1, 32)
    maximum = blocks.abs().amax(dim=1)
    if input_quantization:
        bits = (maximum * (1.0 / 6.0)).contiguous().view(torch.int32)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    else:
        bits = maximum.contiguous().view(torch.int32)
        exponent = (((bits + 0x00400000) & 0x7F800000) >> 23).clamp(min=2) - 2
    scale = (exponent << 23).view(torch.float32)[:, None]
    scale = torch.where(scale == 0, 1.0, scale)
    normalized = blocks.abs() / scale
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0], device=value.device)
    closest = (normalized[..., None] - levels).abs().argmin(dim=-1)
    return (levels[closest] * blocks.sign() * scale).reshape(shape)


def _reference(x, ids, weights, profile, *, tail: bool, pad_reference: bool = False):
    """Route-by-route FP32 MoE over every expert any local token selects."""
    inter = profile.logical_intermediate
    xq = _quantized_reference(x, input_quantization=True)
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in torch.unique(ids).tolist():
        token, slot = torch.where(ids == expert)
        w13, s13, w2, s2 = _expert_packed(profile, expert, x.device, tail=tail)
        w13f = _dequant_mxfp4(w13, s13)  # [2*inter, hidden]
        w2f = _dequant_mxfp4(w2, s2)  # [hidden, inter]
        gate_up = xq[token] @ w13f.t()
        gate, up = gate_up[:, :inter], gate_up[:, inter:]
        hidden = torch.nn.functional.silu(gate.clamp(max=_CLAMP)) * up.clamp(-_CLAMP, _CLAMP)
        if pad_reference:
            # Zero-pad to the compute intermediate; the padded blocks are all
            # zero so they must not change the quantised intermediate.
            pad = profile.compute_intermediate - inter
            hidden = torch.nn.functional.pad(hidden, (0, pad))
            w2f = torch.nn.functional.pad(w2f, (0, pad))
        hidden = _quantized_reference(hidden, input_quantization=False)
        result = hidden @ w2f.t()
        routed = (result * weights[token, slot, None]).to(torch.bfloat16).float()
        output.index_add_(0, token, routed)
    return output.to(torch.bfloat16)


# --------------------------------------------------------------------------
# Routing and cases
# --------------------------------------------------------------------------


def _routing(tokens, rank, profile, world_size, device, step=0):
    """Distinct experts per token spread over all destinations; ``step``
    changes destinations so successive calls reuse buffers with new routes."""
    local = profile.experts // world_size
    token = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
    route = torch.arange(profile.top_k, dtype=torch.int64, device=device)[None, :]
    if step == 0:
        destination = (rank + token + token // 16 + route) % world_size
        local_id = (token * profile.top_k + route) % local
    else:
        # Concentrate on one destination (top_k <= local, so experts are distinct).
        destination = torch.full_like(token + route, step % world_size)
        local_id = (token + route + step) % local
    ids = (destination * local + local_id).to(torch.int32).contiguous()
    raw = 1.0 + ((token + 2 * route + step) % 7).to(torch.float32)
    weights = (raw / raw.sum(dim=1, keepdim=True)).contiguous()
    return ids, weights


def _cases(world_size: int):
    ws = world_size
    uneven8 = [0, 1, 15, 16, 31, 32, 63, 64]
    return {
        "balanced": ([16] * ws, 1.0 / 32, False),
        "uneven_empty": (uneven8[:ws] if ws == 8 else [0, 1, 31, 64], 1.0 / 32, False),
        "mostly_empty": ([0] * (ws - 1) + [7], 1.0 / 32, False),
        "all_empty": ([0] * ws, 1.0 / 32, False),
        # Last logical intermediate block feeds w2: padding must be inert.
        "padding_tail": ([33] * ws, 1.0 / 32, True),
        # |x| up to 252 drives gate > 10 and |up| > 10 through E2M1 2.0 * 2**-4.
        "clamp_saturation": ([64] * ws, 4.0, False),
        "partial_scales": ([256] * ws, 1.0 / 32, False),
        "capacity_one_rank": ([_CAPACITY] + [1] * (ws - 1), 1.0 / 32, False),
        "capacity_all": ([_CAPACITY] * ws, 1.0 / 32, False),
    }


_CASE_NAMES = tuple(_cases(8))


def _init():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    assert dist.get_world_size() == _WORLD_SIZE
    return dist.get_rank(), torch.device("cuda", local_rank)


_layer_cache: dict = {}


def _prepared(profile, tail, device, rank):
    key = (profile.name, tail)
    if key not in _layer_cache:
        layer = _V41MoeLayer(profile, _WORLD_SIZE, device)
        _fill_layer(layer, profile, rank, _WORLD_SIZE, tail=tail)
        plan = _make_plan(profile, _WORLD_SIZE)
        tokenspeed_kernel.moe_process_weights(plan, layer)
        _layer_cache.clear()  # keep one prepared layer resident
        _layer_cache[key] = (layer, plan)
    return _layer_cache[key]


pytestmark = pytest.mark.skipif(
    _WORLD_SIZE not in _SUPPORTED_WORLD_SIZES or not is_cdna4(),
    reason="requires python -m torch.distributed.run with 4 or 8 gfx950 GPUs",
)


@pytest.mark.parametrize("case", _CASE_NAMES)
@pytest.mark.parametrize("profile_name", tuple(_PROFILES))
def test_v41_petit_matches_reference(profile_name, case):
    expected_root = os.environ.get("PETIT_EXPECTED_RUNTIME_ROOT")
    if expected_root is not None:
        assert Path(tokenspeed_kernel.__file__).resolve().is_relative_to(
            Path(expected_root).resolve()
        ), "pytest imported a different runtime checkout"
    rank, device = _init()
    ws = _WORLD_SIZE
    profile = _PROFILES[profile_name]
    counts, input_scale, tail = _cases(ws)[case]
    assert len(counts) == ws and max(counts) <= _CAPACITY
    layer, plan = _prepared(profile, tail, device, rank)
    assert layer.num_local_experts == profile.experts // ws

    count = counts[rank]
    x = torch.empty((count, profile.hidden), device=device, dtype=torch.bfloat16)
    ids, weights = _routing(count, rank, profile, ws, device)

    def refresh(step):
        values = torch.arange(x.numel(), device=device).reshape(x.shape)
        x.copy_(((((values * 13 + rank * 7 + step * 17) % 127) - 63).float()) * input_scale)
        new_ids, new_weights = _routing(count, rank + step, profile, ws, device, step)
        ids.copy_(new_ids)
        weights.copy_(new_weights)

    def run():
        return tokenspeed_kernel.moe_apply(
            plan,
            x,
            layer,
            x.new_empty((count, 0)),
            topk_weights=weights,
            topk_ids=ids,
            num_tokens_global=sum(counts),
            max_num_tokens_per_gpu=max(max(counts), 1),
            do_finalize=True,
            low_latency=None,
            overlap_fn=None,
            shared_input=None,
            shared_weight=None,
            shared_out=None,
        )

    refresh(0)
    eager = run().clone()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for step in (0, 1, 2):  # routing changes between calls on the same buffers
        refresh(step)
        graph.replay()
        torch.cuda.synchronize()
        actual = captured.clone()
        expected = _reference(x, ids, weights, profile, tail=tail)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=8e-3)
        # Eager on the same inputs is bit-identical to graph replay.
        torch.testing.assert_close(actual, eager if step == 0 else run(), atol=0, rtol=0)
        assert torch.isfinite(actual).all()
        if case == "padding_tail" and step == 0 and count:
            padded = _reference(x, ids, weights, profile, tail=tail, pad_reference=True)
            torch.testing.assert_close(padded, expected, atol=0, rtol=0)
        if case == "clamp_saturation" and step == 0 and count:
            _assert_clamp_engaged(x, ids, profile, tail)
        dist.barrier()
    del graph


def _assert_clamp_engaged(x, ids, profile, tail):
    """The case is only meaningful if pre-activation values cross the clamp."""
    inter = profile.logical_intermediate
    xq = _quantized_reference(x, input_quantization=True)
    expert = int(ids[0, 0])
    w13, s13, _, _ = _expert_packed(profile, expert, x.device, tail=tail)
    gate_up = xq @ _dequant_mxfp4(w13, s13).t()
    assert (gate_up[:, :inter] > _CLAMP).any(), "gate never exceeds 10"
    assert (gate_up[:, inter:] > _CLAMP).any() and (gate_up[:, inter:] < -_CLAMP).any()
