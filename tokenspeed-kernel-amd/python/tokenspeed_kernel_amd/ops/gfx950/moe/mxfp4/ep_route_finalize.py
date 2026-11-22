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

"""One-launch finalize of an EP shard's deferred A4W4 decode routes.

``gluon_mxfp_precomputed_mxfp4_fused_moe`` finalizes an expert-parallel shard
only on its direct decode (rows <= 2) and package prefill (rows >= 9) paths;
between them the opt-in EP registration ``gluon_mxfp4_precomputed_ep_moe_apply``
runs the deferred direct decode, which returns one BF16 row per route plus a
route map (``-1`` marks routes of other ranks, whose rows are undefined). This
kernel weights and sums the rank's routes in one launch:

    out[t, :] = bf16( sum_{s < top_k, map[t*top_k+s] >= 0}
                      fp32(routes[t*top_k+s, :]) * fp32(w[t, s]) )

accumulated in FP32 in slot order, products rounded before the add (FP
contraction off). It replaces the registration's six eager tensor ops (compare,
cast, where, multiply, sum, cast) with the same arithmetic; rows of other
ranks' routes are never read.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import tl, triton

EP_ROUTE_FINALIZE_BLOCK_N = 1024


@triton.jit
def _ep_route_finalize_kernel(
    routes,
    route_map,
    weights,
    out,
    hidden,
    stride_route,
    stride_weight_token,
    stride_weight_slot,
    stride_out,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    in_width = cols < hidden
    acc = tl.full((BLOCK_N,), 0.0, tl.float32)
    for slot in tl.static_range(TOPK):
        row = token * TOPK + slot
        owned = tl.load(route_map + row) >= 0
        weight = tl.load(
            weights + token * stride_weight_token + slot * stride_weight_slot
        ).to(tl.float32)
        value = tl.load(
            routes + row.to(tl.int64) * stride_route + cols,
            mask=in_width & owned,
            other=0.0,
        ).to(tl.float32)
        acc += value * weight
    tl.store(
        out + token.to(tl.int64) * stride_out + cols,
        acc.to(out.dtype.element_ty),
        mask=in_width,
    )


def ep_route_finalize(
    routes: torch.Tensor,
    route_map: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weight and sum an EP shard's deferred routes into ``[tokens, hidden]``.

    Args:
        routes: ``[tokens * top_k, hidden]`` BF16 route rows (row-contiguous).
        route_map: ``[tokens * top_k]`` int32; ``-1`` marks routes of other
            ranks (their rows are not read).
        topk_weights: ``[tokens, top_k]`` FP32 or BF16 routing weights.
        out_dtype: Output dtype when ``out`` is None.
        out: Optional contiguous ``[tokens, hidden]`` destination.
    """
    if topk_weights.ndim != 2:
        raise ValueError(
            f"topk_weights must be rank-2, got {tuple(topk_weights.shape)}"
        )
    tokens, top_k = (int(v) for v in topk_weights.shape)
    if routes.ndim != 2 or int(routes.shape[0]) != tokens * top_k:
        raise ValueError(
            f"routes must be [tokens * top_k, hidden] = [{tokens * top_k}, hidden], "
            f"got {tuple(routes.shape)}"
        )
    if routes.stride(1) != 1:
        raise ValueError("routes must be row-contiguous")
    if tuple(route_map.shape) != (tokens * top_k,) or route_map.dtype != torch.int32:
        raise ValueError(
            f"route_map must be int32 [{tokens * top_k}], got "
            f"{route_map.dtype} {tuple(route_map.shape)}"
        )
    hidden = int(routes.shape[1])
    if out is None:
        out = torch.empty((tokens, hidden), dtype=out_dtype, device=routes.device)
    elif tuple(out.shape) != (tokens, hidden) or not out.is_contiguous():
        raise ValueError(
            f"out must be a contiguous [{tokens}, {hidden}] tensor, got "
            f"{tuple(out.shape)}"
        )
    if tokens == 0:
        return out
    grid = (tokens, triton.cdiv(hidden, EP_ROUTE_FINALIZE_BLOCK_N))
    _ep_route_finalize_kernel[grid](
        routes,
        route_map,
        topk_weights,
        out,
        hidden,
        routes.stride(0),
        topk_weights.stride(0),
        topk_weights.stride(1),
        out.stride(0),
        TOPK=top_k,
        BLOCK_N=EP_ROUTE_FINALIZE_BLOCK_N,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["EP_ROUTE_FINALIZE_BLOCK_N", "ep_route_finalize"]
