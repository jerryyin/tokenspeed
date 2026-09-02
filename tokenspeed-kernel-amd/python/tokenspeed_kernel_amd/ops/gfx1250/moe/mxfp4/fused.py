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


from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
from tokenspeed_kernel_amd._triton import (
    aggregate,
    gl,
    gluon,
    tl,
    triton,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe._common import (
    FP4,
    FnSpecs,
    FusedActivation,
    RaggedTensorMetadata,
    Storage,
    Tensor,
    make_ragged_tensor_metadata,
    wrap_torch_tensor,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._common import (
    MoEConfig,
    MoEPipelinedProgram,
    MoEProgramBase,
    _situ_gfx1250,
    _swiglu_gfx1250,
    composition,
    compute_offsets,
    compute_pids,
    create_descriptor,
    get_bitwidth,
    get_blocked_layout,
    get_scaled_dot_format_string,
    get_tdm_gather_scatter_idx_layout,
    ragged_metadata_fields,
    situ_activation_fn,
    swiglu_beta_fn,
    swiglu_fn,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._indexing import (
    select_tdm_index_width_bits,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4._specialize import (
    ClosureArg,
    SpecializationModule,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.decode import _matmul_decode


@dataclass
class PrecisionConfig:
    """Minimal precision config accepted by the gfx1250 MoE wrapper."""

    a_mx_scale: torch.Tensor | Tensor | None = None
    b_mx_scale: torch.Tensor | Tensor | None = None
    out_dtype: torch.dtype | None = None


@dataclass
class _OptFlags:
    block_m: int
    block_n: int
    block_k: int
    group_m: int = 8
    xcd_swizzle: int = 1
    split_k: int = 1


@dataclass(frozen=True)
class _NamedScaleLayout:
    name: str


def _parse_amdgcn_metric(amdgcn: str, key: str) -> int | None:
    m = re.search(rf"\.{key}:\s+(\d+)", amdgcn)
    if m is not None:
        return int(m.group(1))
    m = re.search(rf";\s+{key}\s*[:=]?\s+(\d+)", amdgcn)
    return int(m.group(1)) if m is not None else None


def static_profile(kernel: Any, *, label: str = "") -> dict:
    """Return basic AMDGCN resource metrics from a compiled kernel object."""

    amdgcn = kernel.asm.get("amdgcn", "")
    profile = {
        "sgpr_count": _parse_amdgcn_metric(amdgcn, "sgpr_count"),
        "sgpr_spill_count": _parse_amdgcn_metric(amdgcn, "sgpr_spill_count"),
        "vgpr_count": _parse_amdgcn_metric(amdgcn, "vgpr_count"),
        "vgpr_spill_count": _parse_amdgcn_metric(amdgcn, "vgpr_spill_count"),
        "scratch_size": _parse_amdgcn_metric(amdgcn, "ScratchSize"),
        "code_len_in_byte": _parse_amdgcn_metric(amdgcn, "codeLenInByte"),
        "occupancy": _parse_amdgcn_metric(amdgcn, "Occupancy"),
    }
    if label:
        profile["label"] = label
    return profile


@gluon.jit
def create_split_descriptor(cfg: MoEConfig, x_ptr, w_ptr, off_m, w_offs, M, N, K, stride_xm, stride_xk, stride_wk,
                            stride_wn, GatherIndx, start_m):
    X_SUBTILE_M: gl.constexpr = cfg.BLOCK_M // cfg.NUM_SUBTILES[0]
    W_SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
    BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

    if cfg.USE_GATHER:
        IDX_BASE_LAYOUT: gl.constexpr = get_tdm_gather_scatter_idx_layout(X_SUBTILE_M, cfg.NUM_WARPS)
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(0, IDX_BASE_LAYOUT)
        GatherIndx_ptr = GatherIndx + start_m
        offs_m0 = off_m + gl.arange(0, X_SUBTILE_M, IDX_LAYOUT)
        mask_m0 = start_m + offs_m0 < M
        gm0 = gl.load(GatherIndx_ptr + offs_m0, mask=mask_m0, other=0).to(cfg.index_type)
        if cfg.NUM_SUBTILES[0] == 1:
            gm1 = gl.constexpr(0)
        else:
            offs_m1 = off_m + X_SUBTILE_M + gl.arange(0, X_SUBTILE_M, IDX_LAYOUT)
            mask_m1 = start_m + offs_m1 < M
            gm1 = gl.load(GatherIndx_ptr + offs_m1, mask=mask_m1, other=0).to(cfg.index_type)

        x0_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=x_ptr, shape=(M, K // cfg.DIV_FACTOR_X),
                                             strides=(stride_xm, stride_xk),
                                             block_shape=(X_SUBTILE_M, BLOCK_K_PACKED_X), layout=cfg.shared_layout_x)
        x1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=x_ptr, shape=(M, K // cfg.DIV_FACTOR_X),
                                             strides=(stride_xm, stride_xk),
                                             block_shape=(X_SUBTILE_M, BLOCK_K_PACKED_X), layout=cfg.shared_layout_x)
    else:
        gm0 = gl.constexpr(0)
        gm1 = gl.constexpr(0)
        x_offs = off_m * stride_xm
        x0_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=x_ptr + x_offs, shape=(M, K // cfg.DIV_FACTOR_X),
                                             strides=(stride_xm, stride_xk),
                                             block_shape=(X_SUBTILE_M, BLOCK_K_PACKED_X), layout=cfg.shared_layout_x)
        x1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=x_ptr + x_offs, shape=(M, K // cfg.DIV_FACTOR_X),
                                             strides=(stride_xm, stride_xk),
                                             block_shape=(X_SUBTILE_M, BLOCK_K_PACKED_X), layout=cfg.shared_layout_x)

    if cfg.W_TRANSPOSE:
        w0_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=w_ptr + w_offs, shape=(N, K // cfg.DIV_FACTOR_W),
                                             strides=(stride_wn, stride_wk),
                                             block_shape=(W_SUBTILE_N, BLOCK_K_PACKED_W), layout=cfg.shared_layout_w)
        w1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=w_ptr + w_offs, shape=(N, K // cfg.DIV_FACTOR_W),
                                             strides=(stride_wn, stride_wk),
                                             block_shape=(W_SUBTILE_N, BLOCK_K_PACKED_W), layout=cfg.shared_layout_w)
    else:
        w0_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=w_ptr + w_offs, shape=(K // cfg.DIV_FACTOR_W, N),
                                             strides=(stride_wk, stride_wn),
                                             block_shape=(BLOCK_K_PACKED_W, W_SUBTILE_N), layout=cfg.shared_layout_w)
        w1_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(base=w_ptr + w_offs, shape=(K // cfg.DIV_FACTOR_W, N),
                                             strides=(stride_wk, stride_wn),
                                             block_shape=(BLOCK_K_PACKED_W, W_SUBTILE_N), layout=cfg.shared_layout_w)

    return x0_desc, x1_desc, w0_desc, w1_desc, gm0, gm1


@composition
@aggregate
class MoESliceKProgram:
    base: MoEProgramBase

    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr

    x_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    w_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    x_scale_desc: gl.amd.gfx1250.tdm.tensor_descriptor | gl.constexpr
    w_scale_desc: gl.amd.gfx1250.tdm.tensor_descriptor | gl.constexpr

    gathered_m: gl.tensor | gl.constexpr
    off_k_x: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg: MoEConfig, x_buffer, w_buffer, x_scale_buffer, w_scale_buffer, x_desc, w_desc, x_scale_desc,
                 w_scale_desc, gathered_m, off_k_x):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer
        self.x_scale_buffer = x_scale_buffer if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc = w_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.gathered_m = gathered_m
        self.off_k_x = off_k_x

        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(cfg: MoEConfig, x_desc, w_desc, x_scale_desc, w_scale_desc, gathered_m, off_k_x):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(x_desc.dtype, shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
                                             layout=cfg.shared_layout_x)
        w_buffer = gl.allocate_shared_memory(
            w_desc.dtype, shape=[NUM_BUFFERS, cfg.BLOCK_N, BLOCK_K_PACKED_W]
            if cfg.W_TRANSPOSE else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N], layout=cfg.shared_layout_w)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_M_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_x_scale)
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_N_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_w_scale)
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceKProgram(cfg, x_buffer, w_buffer, x_scale_buffer, w_scale_buffer, x_desc, w_desc, x_scale_desc,
                                w_scale_desc, gathered_m, off_k_x)

    @gluon.jit
    def issue_subtile_local_loads(self, wmma_idx, subtile_start_idx: gl.constexpr):
        cfg = self.cfg
        SUBTILE_LEN: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES[2]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        SUBTILE_LEN_SCALE: gl.constexpr = SUBTILE_LEN // cfg.SCALE_BLOCK
        subtile_start: gl.constexpr = subtile_start_idx * SUBTILE_LEN

        x = self.x_buffer.index(wmma_idx % cfg.NUM_BUFFERS).slice(subtile_start // cfg.DIV_FACTOR_X,
                                                                  SUBTILE_LEN // cfg.DIV_FACTOR_X,
                                                                  1).load(layout=cfg.dot_layout_x)

        if cfg.W_TRANSPOSE:
            w = self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS).slice(subtile_start // cfg.DIV_FACTOR_W,
                                                                      SUBTILE_LEN // cfg.DIV_FACTOR_W,
                                                                      1).permute([1, 0]).load(layout=cfg.dot_layout_w)
        else:
            w = self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS).slice(subtile_start // cfg.DIV_FACTOR_W,
                                                                      SUBTILE_LEN // cfg.DIV_FACTOR_W,
                                                                      0).load(layout=cfg.dot_layout_w)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer_slice = self.x_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            if cfg.SCALE_PRESHUFFLE:
                x_scale_buffer_slice = x_scale_buffer_slice.reshape(
                    (cfg.BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                     cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_M, BLOCK_K_SCALE))
            x_scale_buffer_slice = x_scale_buffer_slice.slice(subtile_start // cfg.SCALE_BLOCK, SUBTILE_LEN_SCALE, 1)
            scale_x = x_scale_buffer_slice.load(layout=cfg.layout_x_scale)
        else:
            scale_x = 0
            scale_x = scale_x.to(gl.uint8)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer_slice = self.w_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            if cfg.SCALE_PRESHUFFLE:
                w_scale_buffer_slice = w_scale_buffer_slice.reshape(
                    (cfg.BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                     cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
            w_scale_buffer_slice = w_scale_buffer_slice.slice(subtile_start // cfg.SCALE_BLOCK, SUBTILE_LEN_SCALE, 1)
            scale_w = w_scale_buffer_slice.load(layout=cfg.layout_w_scale)
        else:
            scale_w = 0
            scale_w = scale_w.to(gl.uint8)

        return x, w, scale_x, scale_w

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0

        # prologue
        # iter 0
        load_idx = self.issue_global_loads(load_idx)

        # iter 1
        load_idx = self.issue_global_loads(load_idx)
        self.async_wait(cfg.NUM_BUFFERS - 1)

        # iter 0
        x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)

        accumulator = gl.zeros((cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout)
        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K) - 1
        for _ in range(0, loop_ub - 1):
            # iter i
            accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
            # iter i
            x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
            wmma_idx += 1
            # iter i + 2
            load_idx = self.issue_global_loads(load_idx)
            # iter i
            accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)
            # iter i + 1
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)

        # epilogue
        accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
        x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
        wmma_idx += 1
        accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)

        self.async_wait(0)
        x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)
        accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
        x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
        wmma_idx += 1
        accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)

        return accumulator

    @gluon.jit
    def warp_pipeline(self, loop_k):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0
        gl.static_assert(cfg.NUM_BUFFERS == 3)

        # prologue
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx)

        accumulator = gl.zeros((cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout)
        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K) - (cfg.NUM_BUFFERS - 1)
        gl.assume(loop_ub >= 0)
        self.async_wait(cfg.NUM_BUFFERS - 2)
        for _ in range(0, loop_ub):
            with gl.amd.warp_pipeline_stage("lds0", priority=1):
                x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)

            self.async_wait(cfg.NUM_BUFFERS - 3)
            with gl.amd.warp_pipeline_stage("tdm+wmma+lds1", priority=0):
                load_idx = self.issue_global_loads(load_idx)
                accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
                x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
                wmma_idx += 1
                accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)

        # epilogue
        for i in gl.static_range(cfg.NUM_BUFFERS - 1):
            self.async_wait(cfg.NUM_BUFFERS - 1 - i)
            x0, w0, scale_x0, scale_w0 = self.issue_subtile_local_loads(wmma_idx, 0)
            accumulator = self.wmma(x0, scale_x0, w0, scale_w0, accumulator)
            x1, w1, scale_x1, scale_w1 = self.issue_subtile_local_loads(wmma_idx, 1)
            accumulator = self.wmma(x1, scale_x1, w1, scale_w1, accumulator)
            wmma_idx += 1

        return accumulator


@composition
@aggregate
class MoESliceNKProgram:
    base: MoEProgramBase

    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr

    x_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    w_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    x_scale_desc: gl.amd.gfx1250.tdm.tensor_descriptor | gl.constexpr
    w_scale_desc: gl.amd.gfx1250.tdm.tensor_descriptor | gl.constexpr

    gathered_m: gl.tensor | gl.constexpr
    off_k_x: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg: MoEConfig, x_buffer, w_buffer, x_scale_buffer, w_scale_buffer, x_desc, w_desc, x_scale_desc,
                 w_scale_desc, gathered_m, off_k_x):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer
        self.x_scale_buffer = x_scale_buffer if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc = w_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.gathered_m = gathered_m
        self.off_k_x = off_k_x

        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(cfg: MoEConfig, x_desc, w_desc, x_scale_desc, w_scale_desc, gathered_m, off_k_x):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(x_desc.dtype, shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
                                             layout=cfg.shared_layout_x)
        w_buffer = gl.allocate_shared_memory(
            w_desc.dtype, shape=[NUM_BUFFERS, cfg.BLOCK_N, BLOCK_K_PACKED_W]
            if cfg.W_TRANSPOSE else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N], layout=cfg.shared_layout_w)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_M_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_x_scale)
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_N_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_w_scale)
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceNKProgram(cfg, x_buffer, w_buffer, x_scale_buffer, w_scale_buffer, x_desc, w_desc, x_scale_desc,
                                 w_scale_desc, gathered_m, off_k_x)

    @gluon.jit
    def issue_global_load_x_data(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X

        if cfg.USE_GATHER:
            col_offset_x = self.off_k_x + load_idx * BLOCK_K_PACKED_X
            x_desc_g = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x_desc, add_offsets=[0, col_offset_x], pred=pred,
                                                    clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_gather(x_desc_g, self.gathered_m, self.x_buffer.index(load_idx % cfg.NUM_BUFFERS))
        else:
            x_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x_desc, add_offsets=[0, load_idx * BLOCK_K_PACKED_X],
                                                       pred=pred, clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_load(x_load_desc, dest=self.x_buffer.index(load_idx % cfg.NUM_BUFFERS),
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_X)

    @gluon.jit
    def issue_global_load_x_scale(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        if cfg.WITH_X_MX_SCALE:
            if cfg.USE_GATHER:
                col_offset_x_scale = self.off_k_x * cfg.DIV_FACTOR_X // cfg.SCALE_BLOCK + load_idx * BLOCK_K_SCALE
                xs_desc_g = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x_scale_desc, add_offsets=[0, col_offset_x_scale],
                                                         pred=pred, clamp_bounds=True)
                gl.amd.gfx1250.tdm.async_gather(xs_desc_g, self.gathered_m, self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS))
            else:
                x_scale_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    self.x_scale_desc, add_offsets=[0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED], pred=pred,
                    clamp_bounds=True)
                gl.amd.gfx1250.tdm.async_load(x_scale_load_desc, dest=self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS),
                               warp_used_hint=cfg.TDM_WARP_USED_HINT_X)

    @gluon.jit
    def issue_global_load_w_scale(self, load_idx, pred=1):
        cfg = self.cfg
        if cfg.WITH_W_MX_SCALE:
            w_scale_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_scale_desc,
                                                             add_offsets=[0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                                                             pred=pred, clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_load(w_scale_load_desc, dest=self.w_scale_buffer.index(load_idx % cfg.NUM_BUFFERS),
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W_SCALE)

    @gluon.jit
    def issue_global_load_w_data(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        if cfg.W_TRANSPOSE:
            w_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_desc, add_offsets=[0, load_idx * BLOCK_K_PACKED_W],
                                                       pred=pred, clamp_bounds=True)
        else:
            w_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_desc, add_offsets=[load_idx * BLOCK_K_PACKED_W, 0],
                                                       pred=pred, clamp_bounds=True)

        gl.amd.gfx1250.tdm.async_load(w_load_desc, dest=self.w_buffer.index(load_idx % cfg.NUM_BUFFERS),
                       warp_used_hint=cfg.TDM_WARP_USED_HINT_W)
        return load_idx + 1

    @gluon.jit
    def async_wait(self, waitcnt):
        cfg = self.cfg
        NUM_LOADS_IN_BATCH: gl.constexpr = 2 + (1 if cfg.WITH_X_MX_SCALE else 0) + \
            (1 if cfg.WITH_W_MX_SCALE else 0)
        gl.amd.gfx1250.tdm.async_wait(waitcnt * NUM_LOADS_IN_BATCH)

    @gluon.jit
    def issue_local_load_x(self, wmma_idx, subtile_start_idx: gl.constexpr):
        cfg = self.cfg
        SUBTILE_LEN: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES[2]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        subtile_start: gl.constexpr = subtile_start_idx * SUBTILE_LEN

        x = self.x_buffer.index(wmma_idx % cfg.NUM_BUFFERS).slice(subtile_start // cfg.DIV_FACTOR_X,
                                                                  SUBTILE_LEN // cfg.DIV_FACTOR_X,
                                                                  1).load(layout=cfg.dot_layout_x)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer_slice = self.x_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            if cfg.SCALE_PRESHUFFLE:
                x_scale_buffer_slice = x_scale_buffer_slice.reshape(
                    (cfg.BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                     cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_M, BLOCK_K_SCALE))
            x_scale_buffer_slice = x_scale_buffer_slice.slice(subtile_start // cfg.SCALE_BLOCK,
                                                              SUBTILE_LEN // cfg.SCALE_BLOCK, 1)
            scale_x = x_scale_buffer_slice.load(layout=cfg.layout_x_scale)
        else:
            scale_x = 0
            scale_x = scale_x.to(gl.uint8)
        return x, scale_x

    @gluon.jit
    def issue_local_load_w(self, wmma_idx, subtile_start_idx_k: gl.constexpr, subtile_start_idx_n: gl.constexpr):
        cfg = self.cfg
        SUBTILE_LEN_K: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES[2]
        SUBTILE_LEN_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        subtile_start_k: gl.constexpr = subtile_start_idx_k * SUBTILE_LEN_K
        subtile_start_n: gl.constexpr = subtile_start_idx_n * SUBTILE_LEN_N

        if cfg.W_TRANSPOSE:
            w = self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS) \
                .slice(subtile_start_n, SUBTILE_LEN_N, 0) \
                .slice(subtile_start_k // cfg.DIV_FACTOR_W, SUBTILE_LEN_K // cfg.DIV_FACTOR_W, 1) \
                .permute([1, 0]).load(layout=cfg.dot_layout_w)
        else:
            w = self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS) \
                .slice(subtile_start_k // cfg.DIV_FACTOR_W, SUBTILE_LEN_K // cfg.DIV_FACTOR_W, 0) \
                .slice(subtile_start_n, SUBTILE_LEN_N, 1) \
                .load(layout=cfg.dot_layout_w)

        w_scale_buffer_slice = self.w_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        if cfg.SCALE_PRESHUFFLE:
            w_scale_buffer_slice = w_scale_buffer_slice.reshape(
                (cfg.BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                 cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
        w_scale_buffer_slice = w_scale_buffer_slice \
            .slice(subtile_start_n, SUBTILE_LEN_N, 0) \
            .slice(subtile_start_k // cfg.SCALE_BLOCK, SUBTILE_LEN_K // cfg.SCALE_BLOCK, 1)
        scale_w = w_scale_buffer_slice.load(layout=cfg.layout_w_scale)
        return w, scale_w

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        load_idx = 0
        wmma_idx = 0

        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            self.issue_global_load_x_data(load_idx)
            self.issue_global_load_x_scale(load_idx)
            self.issue_global_load_w_scale(load_idx)
            load_idx = self.issue_global_load_w_data(load_idx)

        self.async_wait(cfg.NUM_BUFFERS - 2)
        x0, scale_x0 = self.issue_local_load_x(wmma_idx, 0)
        w00, scale_w00 = self.issue_local_load_w(wmma_idx, 0, 0)

        # issue the NUM_BUFFERS-th global load batch
        self.issue_global_load_x_data(load_idx)
        self.issue_global_load_x_scale(load_idx)
        self.issue_global_load_w_scale(load_idx)
        load_idx = self.issue_global_load_w_data(load_idx)

        c0 = gl.zeros((cfg.BLOCK_M // cfg.NUM_SUBTILES[0], cfg.BLOCK_N // cfg.NUM_SUBTILES[1]), dtype=gl.float32,
                      layout=cfg.acc_layout)
        c1 = gl.zeros((cfg.BLOCK_M // cfg.NUM_SUBTILES[0], cfg.BLOCK_N // cfg.NUM_SUBTILES[1]), dtype=gl.float32,
                      layout=cfg.acc_layout)

        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K)
        epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)
        gl.assume(loop_ub >= cfg.NUM_BUFFERS)

        for i in range(0, loop_ub):
            # iter i
            c0 = self.wmma(x0, scale_x0, w00, scale_w00, c0)
            w01, scale_w01 = self.issue_local_load_w(wmma_idx, 0, 1)

            c1 = self.wmma(x0, scale_x0, w01, scale_w01, c1)
            x1, scale_x1 = self.issue_local_load_x(wmma_idx, 1)
            w10, scale_w10 = self.issue_local_load_w(wmma_idx, 1, 0)

            c0 = self.wmma(x1, scale_x1, w10, scale_w10, c0)
            w11, scale_w11 = self.issue_local_load_w(wmma_idx, 1, 1)

            wmma_idx += 1
            c1 = self.wmma(x1, scale_x1, w11, scale_w11, c1)

            # iter i + NUM_BUFFERS - 1: prefetch the next global load batch
            pred = i + 1 - epilogue_lb
            pred = (pred >> 31) & 1
            self.async_wait(cfg.NUM_BUFFERS - 2)
            self.issue_global_load_x_data(load_idx, pred=pred)
            self.issue_global_load_x_scale(load_idx, pred=pred)
            self.issue_global_load_w_scale(load_idx, pred=pred)
            load_idx = self.issue_global_load_w_data(load_idx, pred=pred)

            x0, scale_x0 = self.issue_local_load_x(wmma_idx, 0)
            w00, scale_w00 = self.issue_local_load_w(wmma_idx, 0, 0)

        accumulator = gl.join(c0, c1)
        accumulator = accumulator.permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        accumulator = gl.convert_layout(accumulator, cfg.acc_layout, assert_trivial=True)

        return accumulator


@composition
@aggregate
class MoESliceMNKProgram:
    """sliceMNK schedule: subtile the tile according to cfg.NUM_SUBTILES.

    The compute pipeline mirrors MXFPGEMMSliceMNKProgram: each K-block walks the
    M/N accumulator subtile grid and accumulates across the K subtiles.  x is
    loaded as a single per-buffer tile and sliced along M/K at local-load time; w
    along N/K.  The accumulator is returned to ``_matmul`` which applies the
    shared bias/activation/store epilogue (incl. scatter for MoE combine).
    """
    base: MoEProgramBase

    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr

    x_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    w_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    x_scale_desc: gl.amd.gfx1250.tdm.tensor_descriptor | gl.constexpr
    w_scale_desc: gl.amd.gfx1250.tdm.tensor_descriptor | gl.constexpr

    gathered_m: gl.tensor | gl.constexpr
    off_k_x: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg: MoEConfig, x_buffer, w_buffer, x_scale_buffer, w_scale_buffer, x_desc, w_desc, x_scale_desc,
                 w_scale_desc, gathered_m, off_k_x):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer
        self.x_scale_buffer = x_scale_buffer if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc = w_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.gathered_m = gathered_m
        self.off_k_x = off_k_x

        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(cfg: MoEConfig, x_desc, w_desc, x_scale_desc, w_scale_desc, gathered_m, off_k_x):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(x_desc.dtype, shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
                                             layout=cfg.shared_layout_x)
        w_buffer = gl.allocate_shared_memory(
            w_desc.dtype, shape=[NUM_BUFFERS, cfg.BLOCK_N, BLOCK_K_PACKED_W]
            if cfg.W_TRANSPOSE else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N], layout=cfg.shared_layout_w)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_M_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_x_scale)
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_N_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_w_scale)
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceMNKProgram(cfg, x_buffer, w_buffer, x_scale_buffer, w_scale_buffer, x_desc, w_desc, x_scale_desc,
                                  w_scale_desc, gathered_m, off_k_x)

    @gluon.jit
    def issue_global_load_x_scale(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if cfg.WITH_X_MX_SCALE:
            if cfg.USE_GATHER:
                col_offset_x_scale = self.off_k_x * cfg.DIV_FACTOR_X // cfg.SCALE_BLOCK + load_idx * BLOCK_K_SCALE
                xs_desc_g = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x_scale_desc, add_offsets=[0, col_offset_x_scale],
                                                         pred=pred, clamp_bounds=True)
                gl.amd.gfx1250.tdm.async_gather(xs_desc_g, self.gathered_m, self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS))
            else:
                gl.amd.gfx1250.tdm.async_load(self.x_scale_desc, [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                               self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                               warp_used_hint=cfg.TDM_WARP_USED_HINT_X)

    @gluon.jit
    def issue_global_load_x_data_maybe_w_data(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        if cfg.USE_GATHER:
            col_offset_x = self.off_k_x + load_idx * BLOCK_K_PACKED_X
            x_desc_g = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x_desc, add_offsets=[0, col_offset_x], pred=pred,
                                                    clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_gather(x_desc_g, self.gathered_m, self.x_buffer.index(load_idx % cfg.NUM_BUFFERS))
        elif cfg.FUSE_X_W:
            x_dest = self.x_buffer.index(load_idx % cfg.NUM_BUFFERS)
            w_dest = self.w_buffer.index(load_idx % cfg.NUM_BUFFERS)
            x_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x_desc, add_offsets=[0, load_idx * BLOCK_K_PACKED_X],
                                                       pred=pred, clamp_bounds=True)
            if cfg.W_TRANSPOSE:
                w_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_desc, add_offsets=[0, load_idx * BLOCK_K_PACKED_W],
                                                           pred=pred, clamp_bounds=True)
            else:
                w_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_desc, add_offsets=[load_idx * BLOCK_K_PACKED_W, 0],
                                                           pred=pred, clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_load_fused([(x_load_desc, x_dest, cfg.TDM_WARP_USED_HINT_X),
                                  (w_load_desc, w_dest, cfg.TDM_WARP_USED_HINT_W)])
        else:
            gl.amd.gfx1250.tdm.async_load(self.x_desc, [0, load_idx * BLOCK_K_PACKED_X],
                           self.x_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_X)

        return load_idx + 1

    @gluon.jit
    def issue_global_load_w_scale(self, load_idx, pred=1):
        cfg = self.cfg

        if cfg.WITH_W_MX_SCALE:
            gl.amd.gfx1250.tdm.async_load(self.w_scale_desc, [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                           self.w_scale_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W_SCALE)

    @gluon.jit
    def issue_global_load_w_data(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        if cfg.W_TRANSPOSE:
            gl.amd.gfx1250.tdm.async_load(self.w_desc, [0, load_idx * BLOCK_K_PACKED_W],
                           self.w_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W)
        else:
            gl.amd.gfx1250.tdm.async_load(self.w_desc, [load_idx * BLOCK_K_PACKED_W, 0],
                           self.w_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W)

        return load_idx + 1

    @gluon.jit
    def issue_global_load_w_data_and_scale(self, load_idx, pred=1):
        cfg = self.cfg

        if cfg.FUSE_W_W_SCALE:
            BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W
            w_dest = self.w_buffer.index(load_idx % cfg.NUM_BUFFERS)
            w_scale_dest = self.w_scale_buffer.index(load_idx % cfg.NUM_BUFFERS)
            if cfg.W_TRANSPOSE:
                w_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_desc, add_offsets=[0, load_idx * BLOCK_K_PACKED_W],
                                                           pred=pred, clamp_bounds=True)
            else:
                w_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_desc, add_offsets=[load_idx * BLOCK_K_PACKED_W, 0],
                                                           pred=pred, clamp_bounds=True)
            w_scale_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_scale_desc,
                                                             add_offsets=[0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                                                             pred=pred, clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_load_fused([(w_load_desc, w_dest, cfg.TDM_WARP_USED_HINT_W),
                                  (w_scale_load_desc, w_scale_dest, cfg.TDM_WARP_USED_HINT_W_SCALE)])
            return load_idx + 1

        if cfg.FUSE_X_W:
            self.issue_global_load_w_scale(load_idx, pred=pred)
            return load_idx + 1

        self.issue_global_load_w_scale(load_idx, pred=pred)
        return self.issue_global_load_w_data(load_idx, pred=pred)

    @gluon.jit
    def issue_local_load_x(self, wmma_idx, subtile_start_idx_m: gl.constexpr, subtile_start_idx_k: gl.constexpr):
        cfg = self.cfg
        SUBTILE_LEN_M: gl.constexpr = cfg.BLOCK_M // cfg.NUM_SUBTILES[0]
        SUBTILE_LEN_K: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES[2]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        subtile_start_m: gl.constexpr = subtile_start_idx_m * SUBTILE_LEN_M
        subtile_start_k: gl.constexpr = subtile_start_idx_k * SUBTILE_LEN_K

        x = self.x_buffer.index(wmma_idx % cfg.NUM_BUFFERS).slice(subtile_start_m, SUBTILE_LEN_M, 0) \
            .slice(subtile_start_k // cfg.DIV_FACTOR_X, SUBTILE_LEN_K // cfg.DIV_FACTOR_X, 1) \
            .load(layout=cfg.dot_layout_x)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer_slice = self.x_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            if cfg.SCALE_PRESHUFFLE:
                x_scale_buffer_slice = x_scale_buffer_slice.reshape(
                    (cfg.BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                     cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_M, BLOCK_K_SCALE))
            x_scale_buffer_slice = x_scale_buffer_slice.slice(subtile_start_m, SUBTILE_LEN_M, 0) \
                .slice(subtile_start_k // cfg.SCALE_BLOCK, SUBTILE_LEN_K // cfg.SCALE_BLOCK, 1)
            scale_x = x_scale_buffer_slice.load(layout=cfg.layout_x_scale)
        else:
            scale_x = 0
            scale_x = scale_x.to(gl.uint8)
        return x, scale_x

    @gluon.jit
    def issue_local_load_w(self, wmma_idx, subtile_start_idx_k: gl.constexpr, subtile_start_idx_n: gl.constexpr):
        cfg = self.cfg
        SUBTILE_LEN_K: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES[2]
        SUBTILE_LEN_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        subtile_start_k: gl.constexpr = subtile_start_idx_k * SUBTILE_LEN_K
        subtile_start_n: gl.constexpr = subtile_start_idx_n * SUBTILE_LEN_N

        if cfg.W_TRANSPOSE:
            w = self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS) \
                .slice(subtile_start_n, SUBTILE_LEN_N, 0) \
                .slice(subtile_start_k // cfg.DIV_FACTOR_W, SUBTILE_LEN_K // cfg.DIV_FACTOR_W, 1) \
                .permute([1, 0]).load(layout=cfg.dot_layout_w)
        else:
            w = self.w_buffer.index(wmma_idx % cfg.NUM_BUFFERS) \
                .slice(subtile_start_k // cfg.DIV_FACTOR_W, SUBTILE_LEN_K // cfg.DIV_FACTOR_W, 0) \
                .slice(subtile_start_n, SUBTILE_LEN_N, 1) \
                .load(layout=cfg.dot_layout_w)

        w_scale_buffer_slice = self.w_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        if cfg.SCALE_PRESHUFFLE:
            w_scale_buffer_slice = w_scale_buffer_slice.reshape(
                (cfg.BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                 cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
        w_scale_buffer_slice = w_scale_buffer_slice \
            .slice(subtile_start_n, SUBTILE_LEN_N, 0) \
            .slice(subtile_start_k // cfg.SCALE_BLOCK, SUBTILE_LEN_K // cfg.SCALE_BLOCK, 1)
        scale_w = w_scale_buffer_slice.load(layout=cfg.layout_w_scale)
        return w, scale_w

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        load_x_idx = 0
        load_w_idx = 0
        wmma_idx = 0

        # prologue: NUM_BUFFERS - 1 global load batches
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            self.issue_global_load_x_scale(load_x_idx)
            load_w_idx = self.issue_global_load_w_data_and_scale(load_w_idx)
            load_x_idx = self.issue_global_load_x_data_maybe_w_data(load_x_idx)

        self.issue_l2_prefetches_prologue(load_x_idx)

        self.async_wait(cfg.NUM_BUFFERS - 2)
        a00, scale_a00 = self.issue_local_load_x(wmma_idx, 0, 0)
        b00, scale_b00 = self.issue_local_load_w(wmma_idx, 0, 0)

        # issue the NUM_BUFFERS-th global load batch
        self.issue_global_load_x_scale(load_x_idx)
        load_w_idx = self.issue_global_load_w_data_and_scale(load_w_idx)
        load_x_idx = self.issue_global_load_x_data_maybe_w_data(load_x_idx)

        SUBTILE_M: gl.constexpr = cfg.BLOCK_M // cfg.NUM_SUBTILES[0]
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        c00 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c01 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c10 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c11 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K)
        epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)
        gl.assume(loop_ub >= cfg.NUM_BUFFERS)

        for i in range(0, loop_ub):
            c00 = self.wmma(a00, scale_a00, b00, scale_b00, c00)

            pred_prefetch = i - epilogue_lb
            pred_prefetch = (pred_prefetch >> 31) & 1

            b01, scale_b01 = self.issue_local_load_w(wmma_idx, 0, 1)

            c01 = self.wmma(a00, scale_a00, b01, scale_b01, c01)
            a10, scale_a10 = self.issue_local_load_x(wmma_idx, 1, 0)

            if cfg.L2_PREFETCH_DISTANCE >= 0:
                self.issue_l2_prefetches(cfg.L2_PREFETCH_DISTANCE, load_x_idx, pred=pred_prefetch)

            c10 = self.wmma(a10, scale_a10, b00, scale_b00, c10)
            b10, scale_b10 = self.issue_local_load_w(wmma_idx, 1, 0)

            c11 = self.wmma(a10, scale_a10, b01, scale_b01, c11)
            a01, scale_a01 = self.issue_local_load_x(wmma_idx, 0, 1)

            c00 = self.wmma(a01, scale_a01, b10, scale_b10, c00)
            b11, scale_b11 = self.issue_local_load_w(wmma_idx, 1, 1)

            c01 = self.wmma(a01, scale_a01, b11, scale_b11, c01)
            a11, scale_a11 = self.issue_local_load_x(wmma_idx, 1, 1)
            wmma_idx += 1

            # iter i + NUM_BUFFERS - 1: prefetch the next global load batch
            pred_load = i + 1 - epilogue_lb
            pred_load = (pred_load >> 31) & 1

            self.async_wait(cfg.NUM_BUFFERS - 2)

            self.issue_global_load_x_scale(load_x_idx, pred=pred_load)
            load_w_idx = self.issue_global_load_w_data_and_scale(load_w_idx, pred=pred_load)
            load_x_idx = self.issue_global_load_x_data_maybe_w_data(load_x_idx, pred=pred_load)

            c10 = self.wmma(a11, scale_a11, b10, scale_b10, c10)

            c11 = self.wmma(a11, scale_a11, b11, scale_b11, c11)

            a00, scale_a00 = self.issue_local_load_x(wmma_idx, 0, 0)
            b00, scale_b00 = self.issue_local_load_w(wmma_idx, 0, 0)

        acc_top = gl.join(c00, c01).permute(0, 2, 1).reshape((SUBTILE_M, cfg.BLOCK_N))
        acc_bot = gl.join(c10, c11).permute(0, 2, 1).reshape((SUBTILE_M, cfg.BLOCK_N))
        accumulator = gl.join(acc_top, acc_bot).permute(2, 0, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        accumulator = gl.convert_layout(accumulator, cfg.acc_layout)

        return accumulator


@composition
@aggregate
class MoESliceMNKTDMSplitProgram:
    base: MoEProgramBase

    cfg: MoEConfig
    x0_buffer: gl.shared_memory_descriptor
    x1_buffer: gl.shared_memory_descriptor
    w0_buffer: gl.shared_memory_descriptor
    w1_buffer: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr

    x0_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    x1_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    w0_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    w1_desc: gl.amd.gfx1250.tdm.tensor_descriptor
    x_scale_desc: gl.amd.gfx1250.tdm.tensor_descriptor | gl.constexpr
    w_scale_desc: gl.amd.gfx1250.tdm.tensor_descriptor | gl.constexpr

    gathered_m: gl.tensor | gl.constexpr
    gathered_m0: gl.tensor | gl.constexpr
    gathered_m1: gl.tensor | gl.constexpr
    off_k_x: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cfg: MoEConfig, x0_buffer, x1_buffer, w0_buffer, w1_buffer, x_scale_buffer, w_scale_buffer,
                 x0_desc, x1_desc, w0_desc, w1_desc, x_scale_desc, w_scale_desc, gathered_m, gathered_m0, gathered_m1,
                 off_k_x):
        self.cfg = cfg
        self.x0_buffer = x0_buffer
        self.x1_buffer = x1_buffer
        self.w0_buffer = w0_buffer
        self.w1_buffer = w1_buffer
        self.x_scale_buffer = x_scale_buffer if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.x0_desc = x0_desc
        self.x1_desc = x1_desc
        self.w0_desc = w0_desc
        self.w1_desc = w1_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.gathered_m = gathered_m
        self.gathered_m0 = gathered_m0
        self.gathered_m1 = gathered_m1
        self.off_k_x = off_k_x

        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(cfg: MoEConfig, x0_desc, x1_desc, w0_desc, w1_desc, x_scale_desc, w_scale_desc, gathered_m,
                   gathered_m0, gathered_m1, off_k_x):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        X_SUBTILE_M: gl.constexpr = cfg.BLOCK_M // cfg.NUM_SUBTILES[0]
        W_SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x0_buffer = gl.allocate_shared_memory(x0_desc.dtype, shape=[NUM_BUFFERS, X_SUBTILE_M, BLOCK_K_PACKED_X],
                                              layout=cfg.shared_layout_x)
        x1_buffer = gl.allocate_shared_memory(x1_desc.dtype, shape=[NUM_BUFFERS, X_SUBTILE_M, BLOCK_K_PACKED_X],
                                              layout=cfg.shared_layout_x)
        w0_buffer = gl.allocate_shared_memory(
            w0_desc.dtype, shape=[NUM_BUFFERS, W_SUBTILE_N, BLOCK_K_PACKED_W]
            if cfg.W_TRANSPOSE else [NUM_BUFFERS, BLOCK_K_PACKED_W, W_SUBTILE_N], layout=cfg.shared_layout_w)
        w1_buffer = gl.allocate_shared_memory(
            w1_desc.dtype, shape=[NUM_BUFFERS, W_SUBTILE_N, BLOCK_K_PACKED_W]
            if cfg.W_TRANSPOSE else [NUM_BUFFERS, BLOCK_K_PACKED_W, W_SUBTILE_N], layout=cfg.shared_layout_w)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_M_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_x_scale)
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_N_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_w_scale)
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceMNKTDMSplitProgram(cfg, x0_buffer, x1_buffer, w0_buffer, w1_buffer, x_scale_buffer,
                                          w_scale_buffer, x0_desc, x1_desc, w0_desc, w1_desc, x_scale_desc,
                                          w_scale_desc, gathered_m, gathered_m0, gathered_m1, off_k_x)

    @gluon.jit
    def issue_global_load_x_scale(self, load_idx, pred=1):
        cfg = self.cfg
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if cfg.WITH_X_MX_SCALE:
            if cfg.USE_GATHER:
                col_offset_x_scale = self.off_k_x * cfg.DIV_FACTOR_X // cfg.SCALE_BLOCK + load_idx * BLOCK_K_SCALE
                xs_desc_g = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x_scale_desc, add_offsets=[0, col_offset_x_scale],
                                                         pred=pred, clamp_bounds=True)
                gl.amd.gfx1250.tdm.async_gather(xs_desc_g, self.gathered_m, self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS))
            else:
                gl.amd.gfx1250.tdm.async_load(self.x_scale_desc, [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                               self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                               warp_used_hint=cfg.TDM_WARP_USED_HINT_X)

    @gluon.jit
    def issue_global_load_x_data_maybe_w_data(self, load_idx, pred=1):
        cfg = self.cfg
        X_SUBTILE_M: gl.constexpr = cfg.BLOCK_M // cfg.NUM_SUBTILES[0]
        W_SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        if cfg.USE_GATHER:
            col_offset_x = self.off_k_x + load_idx * BLOCK_K_PACKED_X
            x0_desc_g = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x0_desc, add_offsets=[0, col_offset_x], pred=pred,
                                                     clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_gather(x0_desc_g, self.gathered_m0, self.x0_buffer.index(load_idx % cfg.NUM_BUFFERS))
            if cfg.NUM_SUBTILES[0] != 1:
                x1_desc_g = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x1_desc, add_offsets=[0, col_offset_x], pred=pred,
                                                         clamp_bounds=True)
                gl.amd.gfx1250.tdm.async_gather(x1_desc_g, self.gathered_m1, self.x1_buffer.index(load_idx % cfg.NUM_BUFFERS))
        elif cfg.FUSE_X_W:
            x0_dest = self.x0_buffer.index(load_idx % cfg.NUM_BUFFERS)
            x1_dest = self.x1_buffer.index(load_idx % cfg.NUM_BUFFERS)
            w0_dest = self.w0_buffer.index(load_idx % cfg.NUM_BUFFERS)
            w1_dest = self.w1_buffer.index(load_idx % cfg.NUM_BUFFERS)
            x0_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x0_desc, add_offsets=[0, load_idx * BLOCK_K_PACKED_X],
                                                        pred=pred, clamp_bounds=True)
            x1_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.x1_desc,
                                                        add_offsets=[X_SUBTILE_M, load_idx * BLOCK_K_PACKED_X],
                                                        pred=pred, clamp_bounds=True)
            if cfg.W_TRANSPOSE:
                w0_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w0_desc, add_offsets=[0, load_idx * BLOCK_K_PACKED_W],
                                                            pred=pred, clamp_bounds=True)
                w1_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w1_desc,
                                                            add_offsets=[W_SUBTILE_N, load_idx * BLOCK_K_PACKED_W],
                                                            pred=pred, clamp_bounds=True)
            else:
                w0_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w0_desc, add_offsets=[load_idx * BLOCK_K_PACKED_W, 0],
                                                            pred=pred, clamp_bounds=True)
                w1_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w1_desc,
                                                            add_offsets=[load_idx * BLOCK_K_PACKED_W,
                                                                         W_SUBTILE_N], pred=pred, clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_load_fused([(x0_load_desc, x0_dest, cfg.TDM_WARP_USED_HINT_X),
                                  (w0_load_desc, w0_dest, cfg.TDM_WARP_USED_HINT_W)])
            gl.amd.gfx1250.tdm.async_load_fused([(x1_load_desc, x1_dest, cfg.TDM_WARP_USED_HINT_X),
                                  (w1_load_desc, w1_dest, cfg.TDM_WARP_USED_HINT_W)])
        else:
            gl.amd.gfx1250.tdm.async_load(self.x0_desc, [0, load_idx * BLOCK_K_PACKED_X],
                           self.x0_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_X)
            gl.amd.gfx1250.tdm.async_load(self.x1_desc, [X_SUBTILE_M, load_idx * BLOCK_K_PACKED_X],
                           self.x1_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_X)

        return load_idx + 1

    @gluon.jit
    def issue_global_load_w_scale(self, load_idx, pred=1):
        cfg = self.cfg

        if cfg.WITH_W_MX_SCALE:
            gl.amd.gfx1250.tdm.async_load(self.w_scale_desc, [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                           self.w_scale_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W_SCALE)

    @gluon.jit
    def issue_global_load_w_data(self, load_idx, pred=1):
        cfg = self.cfg
        W_SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        if cfg.W_TRANSPOSE:
            gl.amd.gfx1250.tdm.async_load(self.w0_desc, [0, load_idx * BLOCK_K_PACKED_W],
                           self.w0_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W)
            gl.amd.gfx1250.tdm.async_load(self.w1_desc, [W_SUBTILE_N, load_idx * BLOCK_K_PACKED_W],
                           self.w1_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W)
        else:
            gl.amd.gfx1250.tdm.async_load(self.w0_desc, [load_idx * BLOCK_K_PACKED_W, 0],
                           self.w0_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W)
            gl.amd.gfx1250.tdm.async_load(self.w1_desc, [load_idx * BLOCK_K_PACKED_W, W_SUBTILE_N],
                           self.w1_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred,
                           warp_used_hint=cfg.TDM_WARP_USED_HINT_W)

        return load_idx + 1

    @gluon.jit
    def issue_global_load_w_data_and_scale(self, load_idx, pred=1):
        cfg = self.cfg

        if cfg.FUSE_W_W_SCALE:
            W_SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
            BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W
            w0_dest = self.w0_buffer.index(load_idx % cfg.NUM_BUFFERS)
            w1_dest = self.w1_buffer.index(load_idx % cfg.NUM_BUFFERS)
            w_scale_dest = self.w_scale_buffer.index(load_idx % cfg.NUM_BUFFERS)
            if cfg.W_TRANSPOSE:
                w0_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w0_desc, add_offsets=[0, load_idx * BLOCK_K_PACKED_W],
                                                            pred=pred, clamp_bounds=True)
                w1_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w1_desc,
                                                            add_offsets=[W_SUBTILE_N, load_idx * BLOCK_K_PACKED_W],
                                                            pred=pred, clamp_bounds=True)
            else:
                w0_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w0_desc, add_offsets=[load_idx * BLOCK_K_PACKED_W, 0],
                                                            pred=pred, clamp_bounds=True)
                w1_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w1_desc,
                                                            add_offsets=[load_idx * BLOCK_K_PACKED_W,
                                                                         W_SUBTILE_N], pred=pred, clamp_bounds=True)
            w_scale_load_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(self.w_scale_desc,
                                                             add_offsets=[0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
                                                             pred=pred, clamp_bounds=True)
            gl.amd.gfx1250.tdm.async_load_fused([(w0_load_desc, w0_dest, cfg.TDM_WARP_USED_HINT_W),
                                  (w_scale_load_desc, w_scale_dest, cfg.TDM_WARP_USED_HINT_W_SCALE)])
            gl.amd.gfx1250.tdm.async_load(w1_load_desc, dest=w1_dest, warp_used_hint=cfg.TDM_WARP_USED_HINT_W)
            return load_idx + 1

        if cfg.FUSE_X_W:
            self.issue_global_load_w_scale(load_idx, pred=pred)
            return load_idx + 1

        self.issue_global_load_w_scale(load_idx, pred=pred)
        return self.issue_global_load_w_data(load_idx, pred=pred)

    @gluon.jit
    def issue_l2_prefetch_x(self, distance: gl.constexpr, load_idx, pred=True):
        cfg = self.cfg
        if distance < 0:
            return
        if not cfg.USE_GATHER:
            X_SUBTILE_M: gl.constexpr = cfg.BLOCK_M // cfg.NUM_SUBTILES[0]
            BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
            it = load_idx + distance
            pred_i1 = pred != 0
            gl.amd.gfx1250.tdm.prefetch(self.x0_desc, [0, it * BLOCK_K_PACKED_X], pred=pred_i1)
            gl.amd.gfx1250.tdm.prefetch(self.x1_desc, [X_SUBTILE_M, it * BLOCK_K_PACKED_X], pred=pred_i1)
            if cfg.WITH_X_MX_SCALE:
                gl.amd.gfx1250.tdm.prefetch(self.x_scale_desc, [0, it * cfg.BLOCK_K_SCALE_PRESHUFFLED], pred=pred_i1)

    @gluon.jit
    def issue_l2_prefetch_w(self, distance: gl.constexpr, load_idx, pred=True):
        cfg = self.cfg
        if distance < 0:
            return
        W_SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W
        it = load_idx + distance
        pred_i1 = pred != 0
        if cfg.W_TRANSPOSE:
            gl.amd.gfx1250.tdm.prefetch(self.w0_desc, [0, it * BLOCK_K_PACKED_W], pred=pred_i1)
            gl.amd.gfx1250.tdm.prefetch(self.w1_desc, [W_SUBTILE_N, it * BLOCK_K_PACKED_W], pred=pred_i1)
        else:
            gl.amd.gfx1250.tdm.prefetch(self.w0_desc, [it * BLOCK_K_PACKED_W, 0], pred=pred_i1)
            gl.amd.gfx1250.tdm.prefetch(self.w1_desc, [it * BLOCK_K_PACKED_W, W_SUBTILE_N], pred=pred_i1)
        if cfg.WITH_W_MX_SCALE:
            gl.amd.gfx1250.tdm.prefetch(self.w_scale_desc, [0, it * cfg.BLOCK_K_SCALE_PRESHUFFLED], pred=pred_i1)

    @gluon.jit
    def issue_local_load_x(self, wmma_idx, subtile_start_idx_m: gl.constexpr, subtile_start_idx_k: gl.constexpr):
        cfg = self.cfg
        SUBTILE_LEN_M: gl.constexpr = cfg.BLOCK_M // cfg.NUM_SUBTILES[0]
        SUBTILE_LEN_K: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES[2]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        subtile_start_m: gl.constexpr = subtile_start_idx_m * SUBTILE_LEN_M
        subtile_start_k: gl.constexpr = subtile_start_idx_k * SUBTILE_LEN_K

        # The M subtile selects which split buffer; M offset within the subtile is 0.
        if subtile_start_idx_m == 0:
            x_subtile_buffer = self.x0_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        else:
            x_subtile_buffer = self.x1_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        x = x_subtile_buffer.slice(subtile_start_k // cfg.DIV_FACTOR_X, SUBTILE_LEN_K // cfg.DIV_FACTOR_X, 1) \
            .load(layout=cfg.dot_layout_x)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer_slice = self.x_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
            if cfg.SCALE_PRESHUFFLE:
                x_scale_buffer_slice = x_scale_buffer_slice.reshape(
                    (cfg.BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                     cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_M, BLOCK_K_SCALE))
            x_scale_buffer_slice = x_scale_buffer_slice.slice(subtile_start_m, SUBTILE_LEN_M, 0) \
                .slice(subtile_start_k // cfg.SCALE_BLOCK, SUBTILE_LEN_K // cfg.SCALE_BLOCK, 1)
            scale_x = x_scale_buffer_slice.load(layout=cfg.layout_x_scale)
        else:
            scale_x = 0
            scale_x = scale_x.to(gl.uint8)
        return x, scale_x

    @gluon.jit
    def issue_local_load_w(self, wmma_idx, subtile_start_idx_k: gl.constexpr, subtile_start_idx_n: gl.constexpr):
        cfg = self.cfg
        SUBTILE_LEN_K: gl.constexpr = cfg.BLOCK_K // cfg.NUM_SUBTILES[2]
        SUBTILE_LEN_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        subtile_start_k: gl.constexpr = subtile_start_idx_k * SUBTILE_LEN_K
        subtile_start_n: gl.constexpr = subtile_start_idx_n * SUBTILE_LEN_N

        # The N subtile selects which split buffer; N offset within the subtile is 0.
        if subtile_start_idx_n == 0:
            w_subtile_buffer = self.w0_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        else:
            w_subtile_buffer = self.w1_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        if cfg.W_TRANSPOSE:
            w = w_subtile_buffer.slice(subtile_start_k // cfg.DIV_FACTOR_W, SUBTILE_LEN_K // cfg.DIV_FACTOR_W, 1) \
                .permute([1, 0]).load(layout=cfg.dot_layout_w)
        else:
            w = w_subtile_buffer.slice(subtile_start_k // cfg.DIV_FACTOR_W, SUBTILE_LEN_K // cfg.DIV_FACTOR_W, 0) \
                .load(layout=cfg.dot_layout_w)

        w_scale_buffer_slice = self.w_scale_buffer.index(wmma_idx % cfg.NUM_BUFFERS)
        if cfg.SCALE_PRESHUFFLE:
            w_scale_buffer_slice = w_scale_buffer_slice.reshape(
                (cfg.BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                 cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, BLOCK_K_SCALE))
        w_scale_buffer_slice = w_scale_buffer_slice \
            .slice(subtile_start_n, SUBTILE_LEN_N, 0) \
            .slice(subtile_start_k // cfg.SCALE_BLOCK, SUBTILE_LEN_K // cfg.SCALE_BLOCK, 1)
        scale_w = w_scale_buffer_slice.load(layout=cfg.layout_w_scale)
        return w, scale_w

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        load_x_idx = 0
        load_w_idx = 0
        wmma_idx = 0

        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            self.issue_global_load_x_scale(load_x_idx)
            load_w_idx = self.issue_global_load_w_data_and_scale(load_w_idx)
            load_x_idx = self.issue_global_load_x_data_maybe_w_data(load_x_idx)

        self.issue_l2_prefetches_prologue(load_x_idx)

        self.async_wait(cfg.NUM_BUFFERS - 2)
        a00, scale_a00 = self.issue_local_load_x(wmma_idx, 0, 0)
        b00, scale_b00 = self.issue_local_load_w(wmma_idx, 0, 0)

        self.issue_global_load_x_scale(load_x_idx)
        load_w_idx = self.issue_global_load_w_data_and_scale(load_w_idx)
        load_x_idx = self.issue_global_load_x_data_maybe_w_data(load_x_idx)

        SUBTILE_M: gl.constexpr = cfg.BLOCK_M // cfg.NUM_SUBTILES[0]
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        if cfg.NUM_SUBTILES[0] == 1:
            c0 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
            c1 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

            loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K)
            epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)
            gl.assume(loop_ub >= cfg.NUM_BUFFERS)

            for i in range(0, loop_ub):
                c0 = self.wmma(a00, scale_a00, b00, scale_b00, c0)

                pred_prefetch = i - epilogue_lb
                pred_prefetch = (pred_prefetch >> 31) & 1

                b01, scale_b01 = self.issue_local_load_w(wmma_idx, 0, 1)

                c1 = self.wmma(a00, scale_a00, b01, scale_b01, c1)
                a01, scale_a01 = self.issue_local_load_x(wmma_idx, 0, 1)
                b10, scale_b10 = self.issue_local_load_w(wmma_idx, 1, 0)

                if cfg.L2_PREFETCH_DISTANCE >= 0:
                    self.issue_l2_prefetches(cfg.L2_PREFETCH_DISTANCE, load_x_idx, pred=pred_prefetch)

                c0 = self.wmma(a01, scale_a01, b10, scale_b10, c0)
                b11, scale_b11 = self.issue_local_load_w(wmma_idx, 1, 1)
                wmma_idx += 1

                pred_load = i + 1 - epilogue_lb
                pred_load = (pred_load >> 31) & 1
                self.issue_global_load_x_scale(load_x_idx, pred=pred_load)
                load_w_idx = self.issue_global_load_w_data_and_scale(load_w_idx, pred=pred_load)
                load_x_idx = self.issue_global_load_x_data_maybe_w_data(load_x_idx, pred=pred_load)

                self.async_wait(cfg.NUM_BUFFERS - 1)

                c1 = self.wmma(a01, scale_a01, b11, scale_b11, c1)

                a00, scale_a00 = self.issue_local_load_x(wmma_idx, 0, 0)
                b00, scale_b00 = self.issue_local_load_w(wmma_idx, 0, 0)

            accumulator = gl.join(c0, c1).permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
            accumulator = gl.convert_layout(accumulator, cfg.acc_layout)

            return accumulator

        else:
            c00 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
            c01 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
            c10 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
            c11 = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

            loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K)
            epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)
            gl.assume(loop_ub >= cfg.NUM_BUFFERS)

            for i in range(0, loop_ub):
                c00 = self.wmma(a00, scale_a00, b00, scale_b00, c00)

                pred_prefetch = i - epilogue_lb
                pred_prefetch = (pred_prefetch >> 31) & 1

                b01, scale_b01 = self.issue_local_load_w(wmma_idx, 0, 1)

                c01 = self.wmma(a00, scale_a00, b01, scale_b01, c01)
                a10, scale_a10 = self.issue_local_load_x(wmma_idx, 1, 0)

                if cfg.L2_PREFETCH_DISTANCE >= 0:
                    self.issue_l2_prefetches(cfg.L2_PREFETCH_DISTANCE, load_x_idx, pred=pred_prefetch)

                c10 = self.wmma(a10, scale_a10, b00, scale_b00, c10)
                b10, scale_b10 = self.issue_local_load_w(wmma_idx, 1, 0)

                c11 = self.wmma(a10, scale_a10, b01, scale_b01, c11)
                a01, scale_a01 = self.issue_local_load_x(wmma_idx, 0, 1)

                c00 = self.wmma(a01, scale_a01, b10, scale_b10, c00)
                b11, scale_b11 = self.issue_local_load_w(wmma_idx, 1, 1)

                c01 = self.wmma(a01, scale_a01, b11, scale_b11, c01)
                a11, scale_a11 = self.issue_local_load_x(wmma_idx, 1, 1)
                wmma_idx += 1

                pred_load = i + 1 - epilogue_lb
                pred_load = (pred_load >> 31) & 1
                self.issue_global_load_x_scale(load_x_idx, pred=pred_load)
                load_w_idx = self.issue_global_load_w_data_and_scale(load_w_idx, pred=pred_load)
                load_x_idx = self.issue_global_load_x_data_maybe_w_data(load_x_idx, pred=pred_load)

                c10 = self.wmma(a11, scale_a11, b10, scale_b10, c10)

                self.async_wait(cfg.NUM_BUFFERS - 1)

                c11 = self.wmma(a11, scale_a11, b11, scale_b11, c11)

                a00, scale_a00 = self.issue_local_load_x(wmma_idx, 0, 0)
                b00, scale_b00 = self.issue_local_load_w(wmma_idx, 0, 0)

            acc_top = gl.join(c00, c01).permute(0, 2, 1).reshape((SUBTILE_M, cfg.BLOCK_N))
            acc_bot = gl.join(c10, c11).permute(0, 2, 1).reshape((SUBTILE_M, cfg.BLOCK_N))
            accumulator = gl.join(acc_top, acc_bot).permute(2, 0, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
            accumulator = gl.convert_layout(accumulator, cfg.acc_layout)

            return accumulator


@gluon.jit
def _matmul(
    Y,
    stride_y_k,
    stride_y_z,
    stride_y_m,
    stride_y_n,
    XGlobalScale,
    X,
    stride_x_z,
    stride_x_m,
    stride_x_k,
    XMxScale,
    stride_x_mx_z,
    stride_x_mx_m,
    stride_x_mx_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    W_TRANSPOSE: gl.constexpr,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    B,
    stride_b_e,  # Bias
    M,
    N,
    K,
    KW,  # shapes
    GatherIndx,
    WriteBackIndx,
    writeback_size,
    RAGGED_DIMENSION: gl.constexpr,  #
    XSliceSizes,
    XSliceOffs,
    XBlockOffs,
    XBlockSchedule,
    X_EXPECTED_SLICE_SIZE: gl.constexpr,
    X_SLICE_SIZES_DIVISIBILITY: gl.constexpr,  #
    WSliceSizes,
    WSliceOffs,
    WBlockOffs,
    WBlockSchedule,
    W_EXPECTED_SLICE_SIZE: gl.constexpr,
    _W_SLICE_SIZES_DIVISIBILITY: gl.constexpr,  #
    # true grid size
    batch_size,
    grid_m,
    grid_n,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    DO_SITU: gl.constexpr,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    # MoE config
    N_EXPTS_TOT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    SWIZZLE_MX_SCALE: gl.constexpr,
    EVEN_K: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
    INDEX_TYPE: gl.constexpr = gl.int32,
    NUM_BUFFERS: gl.constexpr = 2,
    SCALE_BLOCK: gl.constexpr = 32,
    SCHEDULE: gl.constexpr = "baseline",
    PINGPONG: gl.constexpr = False,
    NUM_WARPS: gl.constexpr = 4,
    L2_PREFETCH_DISTANCE: gl.constexpr = -1,
    PARTIAL_TDM: gl.constexpr = False,
    RESOLVE_PARTITION_CONFLICTS: gl.constexpr = False,
    TDM_SPLIT: gl.constexpr = False,
):
    gl.static_assert(RAGGED_DIMENSION is None or RAGGED_DIMENSION == "M")
    SPLIT_K: gl.constexpr = 1

    DTYPE_X: gl.constexpr = get_scaled_dot_format_string(X.dtype.element_ty)
    DTYPE_W: gl.constexpr = get_scaled_dot_format_string(W.dtype.element_ty)

    address_index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    USE_GATHER: gl.constexpr = GatherIndx is not None

    SCALE_PRESHUFFLE: gl.constexpr = (
        SWIZZLE_MX_SCALE is not None and SWIZZLE_MX_SCALE != "STRIDED"
    )

    WITH_X_MX_SCALE: gl.constexpr = XMxScale is not None
    WITH_W_MX_SCALE: gl.constexpr = WMxScale is not None

    if SCHEDULE == "sliceMNK":
        NUM_SUBTILES: gl.constexpr = (
            (1, 2, 2) if TDM_SPLIT and USE_GATHER else (2, 2, 2)
        )
    elif SCHEDULE == "sliceNK":
        NUM_SUBTILES: gl.constexpr = (1, 2, 2)
    elif SCHEDULE == "sliceK":
        NUM_SUBTILES: gl.constexpr = (1, 1, 2)
    else:
        gl.static_assert(SCHEDULE == "baseline")
        NUM_SUBTILES: gl.constexpr = (1, 1, 1)

    EFFECTIVE_TDM_SPLIT: gl.constexpr = TDM_SPLIT and SCHEDULE == "sliceMNK"
    if PARTIAL_TDM:
        if NUM_WARPS == 8:
            TDM_WARP_USED_HINT: gl.constexpr = 0b01010101
        else:
            gl.static_assert(NUM_WARPS == 4)
            TDM_WARP_USED_HINT: gl.constexpr = 0b00000101
    else:
        TDM_WARP_USED_HINT: gl.constexpr = None

    cfg = MoEConfig(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        SCALE_BLOCK=SCALE_BLOCK,
        NUM_BUFFERS=NUM_BUFFERS,
        W_TRANSPOSE=W_TRANSPOSE,
        WITH_X_MX_SCALE=WITH_X_MX_SCALE,
        WITH_W_MX_SCALE=WITH_W_MX_SCALE,
        SCALE_PRESHUFFLE=SCALE_PRESHUFFLE,
        index_type=INDEX_TYPE,
        NUM_SUBTILES=NUM_SUBTILES,
        EVEN_K=EVEN_K,
        USE_GATHER=USE_GATHER,
        NUM_WARPS=NUM_WARPS,
        TDM_WARP_USED_HINT=TDM_WARP_USED_HINT,
        L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE,
        RESOLVE_PARTITION_CONFLICTS=RESOLVE_PARTITION_CONFLICTS,
        TDM_SPLIT=EFFECTIVE_TDM_SPLIT,
    )

    PACKED_BLOCK_K_W: gl.constexpr = BLOCK_K // cfg.DIV_FACTOR_W

    if _W_SLICE_SIZES_DIVISIBILITY is None:
        W_SLICE_SIZES_DIVISIBILITY: gl.constexpr = 1
    else:
        if PACKED_BLOCK_K_W > BLOCK_K:
            W_SLICE_SIZES_DIVISIBILITY: gl.constexpr = _W_SLICE_SIZES_DIVISIBILITY * (
                PACKED_BLOCK_K_W // BLOCK_K
            )
        else:
            W_SLICE_SIZES_DIVISIBILITY: gl.constexpr = _W_SLICE_SIZES_DIVISIBILITY // (
                BLOCK_K // PACKED_BLOCK_K_W
            )

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)
    if RAGGED_DIMENSION == "M":
        padding_m = grid_m - gl.load(XBlockOffs + N_EXPTS_TOT)
    else:
        padding_m: gl.constexpr = 0

    unpadded_m = grid_m - padding_m
    gl.assume(unpadded_m >= 0)
    total_actual_tiles = batch_size * unpadded_m * grid_n * SPLIT_K

    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_s, pid_m, pid_n, pid_k = compute_pids(
        pid, unpadded_m, grid_n, total_actual_tiles, XCD_SWIZZLE, GROUP_M, SPLIT_K
    )

    expt_id, start_z, start_z_out, start_m, _, off_m, off_k_x, off_k_w = (
        compute_offsets(
            pid_s,
            pid_m,
            pid_k,
            XBlockSchedule,
            XSliceOffs,
            XBlockOffs,
            X_SLICE_SIZES_DIVISIBILITY,
            WBlockSchedule,
            WSliceOffs,
            W_SLICE_SIZES_DIVISIBILITY,
            RAGGED_DIMENSION,
            BLOCK_M,
            BLOCK_K,
            PACKED_BLOCK_K_W,
            SPLIT_K,
        )
    )
    if X_SLICE_SIZES_DIVISIBILITY is not None:
        off_k_x = off_k_x // X_SLICE_SIZES_DIVISIBILITY * X_SLICE_SIZES_DIVISIBILITY
    if W_SLICE_SIZES_DIVISIBILITY is not None:
        off_k_w = off_k_w // W_SLICE_SIZES_DIVISIBILITY * W_SLICE_SIZES_DIVISIBILITY

    if RAGGED_DIMENSION == "M":
        eM = gl.multiple_of(gl.load(XSliceSizes + expt_id), X_SLICE_SIZES_DIVISIBILITY)
    else:
        eM = M

    expt_id, off_m = expt_id.to(address_index_type), off_m.to(address_index_type)
    start_m, start_z = start_m.to(address_index_type), start_z.to(address_index_type)
    pid_n, pid_k = pid_n.to(address_index_type), pid_k.to(address_index_type)

    X_ptr = X + start_z * stride_x_z
    if not cfg.USE_GATHER:
        X_ptr += start_m * stride_x_m

    W_ptr = W + expt_id * stride_w_e
    w_offs = pid_n * BLOCK_N * stride_w_n

    if cfg.WITH_X_MX_SCALE:
        XMxScale_ptr = XMxScale + start_z.to(address_index_type) * stride_x_mx_z
        if not cfg.USE_GATHER:
            XMxScale_ptr += start_m * stride_x_mx_m
    else:
        XMxScale_ptr = XMxScale

    if cfg.WITH_W_MX_SCALE:
        WMxScale_ptr = WMxScale + expt_id * stride_w_mx_e
        w_scale_offs = pid_n * cfg.BLOCK_N_PRESHUFFLED * stride_w_mx_n
    else:
        WMxScale_ptr = WMxScale
        w_scale_offs = 0

    descriptor_m = M
    if not cfg.USE_GATHER:
        # Keep wide expert base-pointer arithmetic separate from the 32-bit
        # expert-local extent required by the TDM descriptor.
        descriptor_m = (eM - off_m).to(gl.int32)
    x_desc, w_desc, x_scale_desc, w_scale_desc, gathered_m = create_descriptor(
        cfg,
        X_ptr,
        W_ptr,
        XMxScale_ptr,
        WMxScale_ptr,
        off_m,
        off_k_x,
        w_offs,
        w_scale_offs,
        descriptor_m,
        N,
        K,
        stride_x_m,
        stride_x_k,
        stride_w_k,
        stride_w_n,
        stride_x_mx_m,
        stride_x_mx_k,
        stride_w_mx_n,
        stride_w_mx_k,
        GatherIndx,
        start_m,
    )

    Y_ptr = Y + start_z_out.to(address_index_type) * stride_y_z

    if SCHEDULE == "sliceMNK":
        if cfg.TDM_SPLIT:
            x0_desc, x1_desc, w0_desc, w1_desc, gm0, gm1 = create_split_descriptor(
                cfg,
                X_ptr,
                W_ptr,
                off_m,
                w_offs,
                descriptor_m,
                N,
                K,
                stride_x_m,
                stride_x_k,
                stride_w_k,
                stride_w_n,
                GatherIndx,
                start_m,
            )
            pgm = MoESliceMNKTDMSplitProgram.initialize(
                cfg,
                x0_desc,
                x1_desc,
                w0_desc,
                w1_desc,
                x_scale_desc,
                w_scale_desc,
                gathered_m,
                gm0,
                gm1,
                off_k_x // cfg.DIV_FACTOR_X,
            )
        else:
            pgm = MoESliceMNKProgram.initialize(
                cfg,
                x_desc,
                w_desc,
                x_scale_desc,
                w_scale_desc,
                gathered_m,
                off_k_x // cfg.DIV_FACTOR_X,
            )
    elif SCHEDULE == "sliceNK":
        pgm = MoESliceNKProgram.initialize(
            cfg,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
            gathered_m,
            off_k_x // cfg.DIV_FACTOR_X,
        )
    elif SCHEDULE == "sliceK":
        pgm = MoESliceKProgram.initialize(
            cfg,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
            gathered_m,
            off_k_x // cfg.DIV_FACTOR_X,
        )
    else:
        pgm = MoEPipelinedProgram.initialize(
            cfg,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
            gathered_m,
            off_k_x // cfg.DIV_FACTOR_X,
        )

    loop_k = K - off_k_x
    if PINGPONG:
        acc = pgm.warp_pipeline(loop_k)
    else:
        acc = pgm.pipeline(loop_k)
    if XGlobalScale is not None and not cfg.WITH_X_MX_SCALE:
        acc *= gl.load(XGlobalScale).to(gl.float32)

    # bias
    b_dtype = B.dtype if B is not None else gl.float32
    BLOCKED_LAYOUT_BIAS: gl.constexpr = get_blocked_layout(
        [BLOCK_N], b_dtype, cfg.NUM_WARPS, 1
    )
    offs_bias_n = BLOCK_N * pid_n + gl.arange(0, BLOCK_N, BLOCKED_LAYOUT_BIAS)
    mask_bias_n = offs_bias_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_bias_n
        if pid_k == 0:
            bias = gl.load(BPtrs, mask=mask_bias_n, other=0)
        else:
            bias = gl.full([BLOCK_N], 0, dtype=gl.float32, layout=BLOCKED_LAYOUT_BIAS)
    else:
        bias = gl.full([BLOCK_N], 0, dtype=gl.float32, layout=BLOCKED_LAYOUT_BIAS)

    bias = gl.convert_layout(bias, gl.SliceLayout(0, cfg.acc_layout))
    acc += bias[None, :]

    gl.static_assert(
        not (DO_SWIGLU and DO_SITU),
        "SwiGLU and SiTU cannot both be enabled",
    )
    if DO_SITU:
        out = _situ_gfx1250(acc, SITU_BETA, SITU_LINEAR_BETA)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
    elif DO_SWIGLU:
        out = _swiglu_gfx1250(acc, SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
    else:
        out = acc
        gl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )

    BLOCKED_LAYOUT_Y: gl.constexpr = get_blocked_layout(
        [BLOCK_M, OUT_BLOCK_N], Y.dtype, cfg.NUM_WARPS
    )
    if cfg.RESOLVE_PARTITION_CONFLICTS and get_bitwidth(Y.dtype) == 8:
        output_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )
        out_f32_smem = gl.allocate_shared_memory(
            gl.float32, (BLOCK_M, OUT_BLOCK_N), output_shared_layout
        )
        out_f32_smem.store(out)
        out = out_f32_smem.load(BLOCKED_LAYOUT_Y).to(Y.dtype.element_ty)
    else:
        out = out.to(Y.dtype.element_ty)
        out = gl.convert_layout(out, BLOCKED_LAYOUT_Y)

    if WriteBackIndx is not None:
        WriteBackIndx += start_m

        SCATTER_SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )

        IDX_BASE_LAYOUT: gl.constexpr = get_tdm_gather_scatter_idx_layout(
            BLOCK_M, cfg.NUM_WARPS
        )
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(0, IDX_BASE_LAYOUT)

        idx_offs = gl.arange(0, BLOCK_M, IDX_LAYOUT)
        idx_mask = (off_m + idx_offs < eM) & (
            start_m + off_m + idx_offs < writeback_size
        )
        dst_row_indices = gl.load(
            WriteBackIndx + off_m + idx_offs, mask=idx_mask, other=writeback_size
        )
        dst_row_indices = dst_row_indices.to(cfg.index_type)

        out_smem = gl.allocate_shared_memory(
            Y.dtype.element_ty, (BLOCK_M, OUT_BLOCK_N), SCATTER_SHARED_LAYOUT
        )
        out_smem.store(out)

        y_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=Y_ptr,
            shape=(writeback_size, yN),
            strides=(stride_y_m, stride_y_n),
            block_shape=(BLOCK_M, OUT_BLOCK_N),
            layout=SCATTER_SHARED_LAYOUT,
        )

        col_offset = (OUT_BLOCK_N * pid_n).to(gl.int32)
        y_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            y_desc, add_offsets=[0, col_offset], clamp_bounds=True
        )
        gl.amd.gfx1250.tdm.async_scatter(y_desc, dst_row_indices, out_smem)
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, BLOCKED_LAYOUT_Y))
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(
            0, OUT_BLOCK_N, gl.SliceLayout(0, BLOCKED_LAYOUT_Y)
        )
        mask_m = offs_y_m < eM
        mask_n = offs_y_n < yN

        Y_ptr += start_m * stride_y_m

        # buffer_store accepts a 32-bit expert-local element offset. Keep the
        # potentially wide start_m/start_z base-pointer advances above, then
        # narrow only the bounded offset passed to the instruction.
        y_offs = (
            offs_y_m.to(address_index_type)[:, None] * stride_y_m
            + offs_y_n.to(address_index_type)[None, :] * stride_y_n
        ).to(gl.int32)
        y_mask = mask_m[:, None] & mask_n[None, :]
        gl.amd.gfx1250.buffer_store(out, Y_ptr, y_offs, mask=y_mask)


decode_specializations = SpecializationModule(
    "tokenspeed_gfx1250_moe_decode",
    kernels=[("_matmul_decode", _matmul_decode)],
    closure_args={
        "activation": ClosureArg("ACTIVATION_FN", "activation_fn_args"),
    },
)


def _can_overflow_int32(tensor: Any) -> bool:
    if tensor is None:
        return False
    data = tensor.storage.data if isinstance(tensor, Tensor) else tensor
    shape = data.shape
    strides = data.stride()
    offset = 0
    for dim, stride in zip(shape, strides):
        offset += (dim - 1) * stride
    return offset > (1 << 31) - 1


def should_upcast_indices(*args: Any) -> bool:
    return any(_can_overflow_int32(arg) for arg in args if arg is not None)


def _canonicalize_storage(storage: Storage, out_ndim: int):
    assert out_ndim >= storage.data.ndim
    new_shape = [1] * (out_ndim - storage.data.ndim) + list(storage.data.shape)
    new_stride = [0] * (out_ndim - storage.data.ndim) + list(storage.data.stride())
    data = storage.data.as_strided(new_shape, new_stride)
    return Storage(data, storage.layout)


def _as_tensor(
    obj: torch.Tensor | Tensor | None, *, dtype: Any | None = None
) -> Tensor | None:
    if obj is None or isinstance(obj, Tensor):
        return obj
    return wrap_torch_tensor(obj, dtype=dtype)


def _mark_scale_preshuffled(scale: Tensor | None, enabled: bool) -> Tensor | None:
    if scale is not None and enabled:
        scale.storage.layout = _NamedScaleLayout("GFX1250_SCALE")
    return scale


def _activation_config(fused_activation: FusedActivation | None):
    if fused_activation is None:
        return False, 0.0, 0.0, 0.0, False, 0.0, 0.0, 1
    specs = fused_activation.specs
    if specs.name == FnSpecs.default().name:
        return False, 0.0, 0.0, 0.0, False, 0.0, 0.0, 1
    if specs.name == "situ":
        if len(fused_activation.fn_args) < 2:
            raise ValueError("SiTU activation requires beta and linear_beta")
        situ_beta = float(fused_activation.fn_args[0])
        situ_linear_beta = float(fused_activation.fn_args[1])
        if situ_beta <= 0.0 or situ_linear_beta <= 0.0:
            raise ValueError("SiTU beta and linear_beta must be positive")
        return (
            False,
            0.0,
            0.0,
            0.0,
            True,
            situ_beta,
            situ_linear_beta,
            int(specs.reduction_n),
        )
    if specs.name not in {"swiglu", "swiglu_beta"}:
        raise NotImplementedError(
            "gfx1250 MoE only supports no activation, SwiGLU, or SiTU, "
            f"got {specs.name!r}"
        )
    if len(fused_activation.fn_args) < 2:
        raise ValueError("SwiGLU activation requires at least alpha and limit")
    alpha = float(fused_activation.fn_args[0])
    limit = float(fused_activation.fn_args[1])
    beta = (
        float(fused_activation.fn_args[2])
        if len(fused_activation.fn_args) >= 3
        else 1.0
    )
    return True, alpha, limit, beta, False, 0.0, 0.0, int(specs.reduction_n)


def _validate_schedule(
    *,
    schedule: str,
    pingpong: bool,
    num_buffers: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    tdm_split: bool,
) -> None:
    if schedule not in ("baseline", "sliceK", "sliceNK", "sliceMNK"):
        raise ValueError(
            "schedule must be 'baseline', 'sliceK', 'sliceNK', or 'sliceMNK', "
            f"got {schedule!r}"
        )
    if schedule == "sliceNK":
        if block_k < 256 or block_n < 256:
            raise ValueError("sliceNK requires block_k >= 256 and block_n >= 256")
        if pingpong or num_buffers != 2:
            raise ValueError("sliceNK supports only num_buffers=2 and pingpong=False")
    if schedule == "sliceK":
        if block_k < 256:
            raise ValueError("sliceK requires block_k >= 256")
        if num_buffers not in (2, 3):
            raise ValueError("sliceK supports only num_buffers 2 or 3")
    if tdm_split and schedule != "sliceMNK":
        raise ValueError("tdm_split is supported only by sliceMNK")
    if pingpong:
        if num_warps != 8:
            raise ValueError("pingpong requires num_warps=8")
        if schedule not in ("baseline", "sliceK"):
            raise ValueError("pingpong supports only baseline and sliceK schedules")
        if num_buffers != 3:
            raise ValueError("pingpong requires num_buffers=3")


def _resolve_block_m(
    decode: bool,
    m: int,
    num_experts: int | None,
    *,
    is_combine: bool,
) -> int:
    """Use stage defaults for prefill and expert occupancy for decode."""
    if not decode:
        return 256 if is_combine else 128
    rows_per_expert = max(1, m // num_experts)
    return max(16, min(triton.next_power_of_2(rows_per_expert), 128))


def get_index_type(a, gather_indx, scatter_indx):
    """Select a safe TDM index type independently of pointer arithmetic."""

    gather_input_rows = None
    if gather_indx is not None:
        gather_input_rows = (
            int(a.shape_max[-2]) if isinstance(a, Tensor) else int(a.shape[-2])
        )
    scatter_writeback_rows = (
        None if scatter_indx is None else int(scatter_indx.shape[0])
    )
    width_bits = select_tdm_index_width_bits(
        gather_input_rows=gather_input_rows,
        scatter_writeback_rows=scatter_writeback_rows,
    )
    return gl.int16 if width_bits == 16 else gl.int32


def matmul(
    a,
    b,
    bias,
    a_ragged_metadata: RaggedTensorMetadata | None = None,
    b_ragged_metadata: RaggedTensorMetadata | None = None,
    gather_indx: torch.Tensor | None = None,
    scatter_indx: torch.Tensor | None = None,
    precision_config: PrecisionConfig | None = None,
    fused_activation: FusedActivation | None = None,
    *,
    x_global_scale: torch.Tensor | float | None = None,
    num_buffers: int = 2,
    scale_block: int = 32,
    block_m: int,
    block_n: int = 128,
    block_k: int = 256,
    group_m: int = 4,
    xcd_swizzle: int = 8,
    w_transpose: bool = True,
    scale_preshuffle: bool | None = None,
    schedule: str = "baseline",
    pingpong: bool = False,
    num_warps: int = 4,
    l2_prefetch_distance: int = -1,
    partial_tdm: bool = False,
    resolve_partition_conflicts: bool = False,
    tdm_split: bool = False,
    decode: bool = False,
):
    """Run the gfx1250 Gluon MoE matmul kernel.

    Args:
        a: Dense or expert-routed activation tensor. FP8 tensors should use a
            torch float8 dtype; MXFP4 tensors should use packed ``torch.uint8``.
        b: Expert weight tensor in ``(E, K_packed, N)`` or dense ``(K_packed, N)``
            layout. Packed MXFP4 weights use ``torch.uint8`` storage.
        bias: Optional expert bias tensor with shape ``(E, N)`` or ``(N,)``.
        a_ragged_metadata: Expert-row metadata for routed dispatch/combine.
        b_ragged_metadata: Reserved for K-ragged weights; currently unsupported.
        gather_indx: Optional source row indices for dispatch.
        scatter_indx: Optional destination row indices for combine writeback.
        precision_config: MX scale/output dtype configuration.
        x_global_scale: Optional scalar activation dequantization scale.
        fused_activation: Optional SwiGLU or SiTU activation descriptor.
        block_m: Concrete row tile resolved by the caller.
        decode: Select the small-M, M-ragged decode kernel.

    Returns:
        ``(output, kernel)`` where ``kernel`` is the Triton/Gluon launch object.
    """
    if b_ragged_metadata is not None:
        raise NotImplementedError(
            "gfx1250 MoE matmul does not support K-ragged weights"
        )
    if decode:
        schedule = "baseline"
        pingpong = False
        l2_prefetch_distance = -1
        partial_tdm = False
        resolve_partition_conflicts = False
        tdm_split = False
    _validate_schedule(
        schedule=schedule,
        pingpong=pingpong,
        num_buffers=num_buffers,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        tdm_split=tdm_split,
    )

    if precision_config is None:
        precision_config = PrecisionConfig()
    fused_activation = fused_activation or FusedActivation(FnSpecs.default(), tuple())
    (
        do_swiglu,
        swiglu_alpha,
        swiglu_limit,
        swiglu_beta,
        do_situ,
        situ_beta,
        situ_linear_beta,
        activation_reduction_n,
    ) = _activation_config(fused_activation)

    a_torch = a.storage.data if isinstance(a, Tensor) else a
    b_torch = b.storage.data if isinstance(b, Tensor) else b
    is_input_batched = a_torch.ndim == 3
    has_scatter = scatter_indx is not None
    is_a_ragged = a_ragged_metadata is not None
    ragged_dimension = "M" if is_a_ragged else None
    if decode and is_input_batched:
        raise ValueError("decode kernel does not support dense-batched matmul")

    M = int(a_torch.shape[-2] if gather_indx is None else gather_indx.shape[0])
    K = int(a_torch.shape[-1])
    K_W, N = map(int, b_torch.shape[-2:])
    if a_torch.dtype == torch.uint8:
        K *= 2
    if b_torch.dtype == torch.uint8:
        K_W *= 2
    if K != K_W:
        raise ValueError(f"K mismatch: activation K={K} vs weight K={K_W}")
    index_type = get_index_type(a, gather_indx, scatter_indx)

    out_dtype = precision_config.out_dtype or (
        a_torch.dtype if a_torch.dtype.is_floating_point else torch.bfloat16
    )

    if not isinstance(a, Tensor):
        a = wrap_torch_tensor(a, dtype=a_torch.dtype)
    if not isinstance(b, Tensor):
        if b_torch.stride(-2) != 1:
            b_torch = b_torch.transpose(-1, -2).contiguous().transpose(-1, -2)
            b = b_torch
        b_dtype = FP4 if b_torch.dtype == torch.uint8 else b_torch.dtype
        b = wrap_torch_tensor(b, dtype=b_dtype)

    a_scale = _as_tensor(precision_config.a_mx_scale)
    b_scale = _as_tensor(precision_config.b_mx_scale)
    if a_scale is not None:
        a_scale.storage.data = a_scale.storage.data.view(torch.uint8)
        a_scale.dtype = torch.uint8
    if b_scale is not None:
        b_scale.storage.data = b_scale.storage.data.view(torch.uint8)
        b_scale.dtype = torch.uint8
    if scale_preshuffle is None:
        scale_preshuffle = False
    a_scale = _mark_scale_preshuffled(a_scale, bool(scale_preshuffle))
    b_scale = _mark_scale_preshuffled(b_scale, bool(scale_preshuffle))

    batch_size = b.shape[0] if ragged_dimension is None and b.ndim == 3 else 1
    opt_flags = _OptFlags(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
        xcd_swizzle=xcd_swizzle,
    )

    expt_data_w = (None,) * 6
    expt_data_x = (
        (None,) * 6
        if ragged_dimension is None
        else ragged_metadata_fields(a_ragged_metadata, block_m)
    )

    y_rows = int(scatter_indx.shape[0] if scatter_indx is not None else M)
    y_cols = N // activation_reduction_n
    out_base = torch.empty(
        (batch_size, y_rows, y_cols), device=a.device, dtype=out_dtype
    )
    out_matmul = out_base[None, :, :, :]
    if has_scatter:
        c_view = out_matmul.view(math.prod(out_matmul.shape[:-1]), out_matmul.shape[-1])
    else:
        c_view = out_matmul.view(
            math.prod(out_matmul.shape[:-2]), *out_matmul.shape[-2:]
        )
    c = wrap_torch_tensor(c_view)

    grid_m = triton.cdiv(M, opt_flags.block_m)
    if ragged_dimension == "M":
        grid_m = a_ragged_metadata.n_blocks(
            a_ragged_metadata.n_slices, M, opt_flags.block_m
        )
    grid_n = triton.cdiv(N, opt_flags.block_n)
    grid = grid_m * grid_n * batch_size

    n_valid_slices = (
        a_ragged_metadata.n_slices if ragged_dimension == "M" else batch_size
    )

    a_storage = _canonicalize_storage(a.storage, 3)
    b_storage = _canonicalize_storage(b.storage, 3)
    c_storage = _canonicalize_storage(c.storage, 3)

    a_strides = [0] * (3 - a_storage.data.ndim) + list(a_storage.data.stride())
    a_scale_strides = a_scale.stride() if a_scale is not None else (None, None, None)
    a_scale_strides = (0,) * (3 - len(a_scale_strides)) + tuple(a_scale_strides)
    b_scale_strides = b_scale.stride() if b_scale is not None else (None, None, None)
    b_scale_strides = (0,) * (3 - len(b_scale_strides)) + tuple(b_scale_strides)
    bias_stride = None if bias is None else bias.stride(0)

    swizzle_mx_scale = None if b_scale is None else b_scale.storage.layout.name
    if x_global_scale is not None:
        if isinstance(x_global_scale, torch.Tensor):
            if x_global_scale.numel() != 1:
                raise ValueError("x_global_scale must be scalar")
            x_global_scale = x_global_scale.to(
                device=a.device, dtype=torch.float32
            ).contiguous()
        else:
            x_global_scale = torch.tensor(
                [float(x_global_scale)], device=a.device, dtype=torch.float32
            )

    if decode:
        target_kernel = decode_specializations.get(
            activation=fused_activation.specs
        )._matmul_decode
        activation_launch_args = (
            *fused_activation.fn_args,
            activation_reduction_n,
        )
    else:
        target_kernel = _matmul
        activation_launch_args = (
            do_swiglu,
            swiglu_alpha,
            swiglu_limit,
            swiglu_beta,
            do_situ,
            situ_beta,
            situ_linear_beta,
            activation_reduction_n,
        )
    kernel = target_kernel[(grid,)](
        c_storage.data,
        *out_matmul.stride(),
        x_global_scale,
        a_storage.data,
        *a_strides,
        a_scale,
        *a_scale_strides,
        b_storage.data,
        *b_storage.data.stride(),
        w_transpose,
        b_scale,
        *b_scale_strides,
        bias,
        bias_stride,
        M,
        N,
        K,
        K_W,
        gather_indx,
        scatter_indx,
        None if scatter_indx is None else scatter_indx.shape[0],
        ragged_dimension,
        *expt_data_x,
        *expt_data_w,
        batch_size,
        grid_m,
        grid_n,
        *activation_launch_args,
        n_valid_slices,
        opt_flags.block_m,
        opt_flags.block_n,
        opt_flags.block_k,
        opt_flags.group_m,
        opt_flags.xcd_swizzle,
        SWIZZLE_MX_SCALE=swizzle_mx_scale,
        EVEN_K=(K % opt_flags.block_k == 0),
        UPCAST_INDICES=should_upcast_indices(a, b, out_matmul),
        INDEX_TYPE=index_type,
        NUM_BUFFERS=num_buffers,
        SCALE_BLOCK=scale_block,
        SCHEDULE=schedule,
        PINGPONG=pingpong,
        NUM_WARPS=num_warps,
        L2_PREFETCH_DISTANCE=l2_prefetch_distance,
        PARTIAL_TDM=partial_tdm,
        RESOLVE_PARTITION_CONFLICTS=resolve_partition_conflicts,
        TDM_SPLIT=tdm_split,
        num_warps=num_warps,
    )
    out_final = c_storage.data
    if not is_input_batched:
        out_final = out_final.squeeze(0)
    return out_final, kernel


def _adapt_index(obj: Any | None, attr: str) -> Any | None:
    if obj is None or hasattr(obj, attr):
        return obj
    if isinstance(obj, torch.Tensor):
        return type("IndxAdapter", (), {attr: obj})()
    return obj


def _index_tensor(obj: Any | None, attr: str) -> torch.Tensor | None:
    if obj is None:
        return None
    return getattr(obj, attr) if hasattr(obj, attr) else obj


def gluon_mxfp_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    x_format: str = "e2m1",
    x_global_scale: torch.Tensor | float = 1.0,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    scatter_indx,
    gate_scal: torch.Tensor | None = None,
    n_tokens: int | None = None,
    n_expts_act: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_m: int | None = None,
    block_n: int = 256,
    block_k: int = 256,
    num_warps: int = 4,
    num_buffers: int = 3,
    use_warp_pipeline: bool | None = None,
    use_slice_mn: bool | None = None,
    use_slice_n: bool | None = None,
    scale_load_mode: str = "transpose",
    w_transpose: bool = True,
    persistent: bool | None = None,
    num_ctas: int | None = None,
    w_preshuffle: bool = False,
    x_scale_ragged_padded: bool = False,
    decode: bool = False,
) -> torch.Tensor:
    """Combine GEMM using the gfx1250 Gluon MoE kernel."""
    del use_warp_pipeline, use_slice_mn, use_slice_n
    del persistent, num_ctas, w_preshuffle, x_scale_ragged_padded
    if gate_scal is not None:
        raise NotImplementedError(
            "gfx1250 source kernel does not apply route gate scaling"
        )
    if x_format == "e2m1" and x_scale is None:
        raise ValueError("x_scale is required for e2m1/MXFP4 activation input")
    if x_format != "e2m1" and x_scale is not None:
        raise ValueError("x_scale is only supported for e2m1/MXFP4 activation input")
    scatter_tensor = _index_tensor(scatter_indx, "dst_indx")
    num_experts = None if a_ragged_metadata is None else a_ragged_metadata.n_slices
    if block_m is None:
        block_m = _resolve_block_m(
            decode, int(x.shape[-2]), num_experts, is_combine=True
        )
    precision = PrecisionConfig(
        out_dtype=out_dtype,
        a_mx_scale=x_scale,
        b_mx_scale=w_scale,
    )
    out, _ = matmul(
        x,
        w,
        bias,
        a_ragged_metadata=a_ragged_metadata,
        scatter_indx=scatter_tensor,
        precision_config=precision,
        x_global_scale=x_global_scale,
        scale_preshuffle=(scale_load_mode == "swizzle"),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
        w_transpose=w_transpose,
        decode=decode,
    )
    if n_expts_act is not None and int(n_expts_act) > 1:
        if n_tokens is None:
            if out.shape[0] % int(n_expts_act) != 0:
                raise ValueError(
                    "n_tokens is required when output rows are not divisible by n_expts_act"
                )
            n_tokens = out.shape[0] // int(n_expts_act)
        out = out.view(int(n_tokens), int(n_expts_act), out.shape[-1]).sum(dim=1)
    return out


@triton.jit
def _fp8_quantize_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SCALE_TENSOR: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    if HAS_SCALE:
        if HAS_SCALE_TENSOR:
            scale = tl.load(scale)
        x = x * (1.0 / scale)
    tl.store(out_ptr + offsets, x.to(tl.float8e4nv), mask=mask)


def _quantize_fp8_activation(
    x: torch.Tensor,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if x.dtype is torch.float8_e4m3fn:
        return x.contiguous()
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"gfx1250 FP8 path expects bf16/fp16/fp8 input, got {x.dtype}")
    x = x.contiguous()
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("FP8 activation scale must be scalar")
        scale = scale.contiguous()
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    _fp8_quantize_kernel[(triton.cdiv(x.numel(), 256),)](
        x,
        out,
        1.0 if scale is None else scale,
        x.numel(),
        BLOCK_SIZE=256,
        HAS_SCALE=scale is not None,
        HAS_SCALE_TENSOR=isinstance(scale, torch.Tensor),
    )
    return out


def _route_from_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype | None = None,
) -> tuple[
    RaggedTensorMetadata,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    valid = flat_ids >= 0
    safe_ids = torch.where(valid, flat_ids, flat_ids.new_zeros(()))
    sort_order = torch.argsort(safe_ids, stable=True)

    top_k = topk_ids.shape[1]
    gather_indx = (sort_order // top_k).to(torch.int32)
    scatter_indx = sort_order.to(torch.int32)
    gate_scal = topk_weights.reshape(-1)[sort_order]
    gate_scal = torch.where(valid[sort_order], gate_scal, torch.zeros_like(gate_scal))
    if dtype is not None and gate_scal.dtype != dtype:
        gate_scal = gate_scal.to(dtype)

    col_sum = torch.zeros((num_experts,), dtype=torch.int32, device=safe_ids.device)
    col_sum.scatter_add_(0, safe_ids, valid.to(torch.int32))
    ragged_metadata = make_ragged_tensor_metadata(col_sum, int(sort_order.numel()))
    return ragged_metadata, gather_indx, scatter_indx, gate_scal


def _precomputed_topk_route(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
):
    return _route_from_topk(
        topk_weights,
        topk_ids,
        num_experts,
        dtype=topk_weights.dtype,
    )


def gluon_mxfp_precomputed_mxfp4_fused_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    w13_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    activation: str = "swiglu",
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    swiglu_beta: float = 1.0,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    decode: bool = False,
    block_m: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch + combine for gfx1250 MXFP4-weight MoE with precomputed top-k.

    Args:
        hidden_states: Token activations in bf16/fp16/fp8, shaped
            ``(n_tokens, hidden_size)``.
        topk_weights: Route weights, shaped ``(n_tokens, top_k)``.
        topk_ids: Expert ids, shaped ``(n_tokens, top_k)``.
        w13_weight: gfx1250-preprocessed interleaved gate/up expert weight.
        w2_weight: gfx1250-preprocessed down-projection expert weight.
        w13_mx_scale: gfx1250-swizzled MXFP4 scale for ``w13_weight``.
        w2_mx_scale: gfx1250-swizzled MXFP4 scale for ``w2_weight``.
        w13_bias: Optional expert bias for the gate/up projection.
        w2_bias: Optional expert bias for the down projection.
        out_dtype: Final output dtype.
        activation: Fused gate activation, either ``"swiglu"``/``"silu"`` or
            ``"situ"``.
        swiglu_alpha: SwiGLU gate scale.
        swiglu_limit: Optional SwiGLU clamp limit; ``0`` disables clamping.
        swiglu_beta: SwiGLU linear branch offset.
        situ_beta: SiTU gate clamp.
        situ_linear_beta: SiTU linear-branch clamp.
        decode: Select the small-M decode kernel for both MoE projections.
        block_m: Optional row-tile override; unset values resolve per projection.
        out: Optional destination tensor for the finalized expert output.

    Returns:
        Tensor shaped ``(n_tokens, hidden_size)``.
    """
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank-2, got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )

    w13_raw = w13_weight.storage.data if isinstance(w13_weight, Tensor) else w13_weight
    if not isinstance(w13_raw, torch.Tensor) or w13_raw.ndim != 3:
        raise ValueError("w13_weight must expose a rank-3 expert weight tensor")
    num_experts = int(w13_raw.shape[0])
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")

    topk_ids = topk_ids.to(device=hidden_states.device, dtype=torch.int32).contiguous()
    topk_weights = topk_weights.to(
        device=hidden_states.device, dtype=torch.float32
    ).contiguous()
    if not torch.cuda.is_current_stream_capturing() and bool(
        ((topk_ids < 0) | (topk_ids >= num_experts)).any().item()
    ):
        raise NotImplementedError(
            "gfx1250 Gluon MXFP4 combine does not support masked or EP-local top-k ids"
        )

    ragged_metadata, gather_indx, scatter_indx, _gate_scal = _precomputed_topk_route(
        topk_weights,
        topk_ids,
        num_experts,
    )

    x_fp8 = _quantize_fp8_activation(
        hidden_states,
        w13_weight.act_scale,
    )
    if activation == "situ":
        fused_activation = FusedActivation(
            FnSpecs(
                "situ",
                situ_activation_fn,
                ("beta", "linear_beta"),
                reduction_n=2,
            ),
            (float(situ_beta), float(situ_linear_beta)),
        )
    elif activation == "silu":
        fused_activation = FusedActivation(
            FnSpecs(
                "swiglu_beta",
                swiglu_beta_fn,
                ("alpha", "limit", "beta"),
                reduction_n=2,
            ),
            (1.0, 0.0, 0.0),
        )
    elif activation == "swiglu":
        if float(swiglu_beta) == 1.0:
            fused_activation = FusedActivation(
                FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
                (float(swiglu_alpha), float(swiglu_limit)),
            )
        else:
            fused_activation = FusedActivation(
                FnSpecs(
                    "swiglu_beta",
                    swiglu_beta_fn,
                    ("alpha", "limit", "beta"),
                    reduction_n=2,
                ),
                (float(swiglu_alpha), float(swiglu_limit), float(swiglu_beta)),
            )
    else:
        raise ValueError(
            "gfx1250 Gluon MXFP4 MoE supports activation 'silu', "
            f"'swiglu', or 'situ', got {activation!r}"
        )
    intermediate = gluon_mxfp_ragged_matmul(
        x_fp8,
        w13_weight,
        w13_bias,
        w_mx_scale=w13_mx_scale,
        x_format="e4m3",
        x_global_scale=w13_weight.act_scale,
        a_ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        out_dtype=out_dtype,
        fused_activation=fused_activation,
        scale_preshuffle=True,
        block_m=block_m,
        block_n=256,
        block_k=256,
        num_warps=4,
        num_buffers=3,
        decode=decode,
    )
    intermediate_fp8 = _quantize_fp8_activation(
        intermediate,
        w2_weight.act_scale,
    )
    flat = gluon_mxfp_combine(
        intermediate_fp8,
        w2_weight,
        w2_mx_scale,
        x_format="e4m3",
        x_global_scale=w2_weight.act_scale,
        bias=w2_bias,
        a_ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        out_dtype=out_dtype,
        block_m=block_m,
        block_n=256,
        block_k=256,
        num_warps=4,
        num_buffers=3,
        scale_load_mode="swizzle",
        decode=decode,
    )
    weighted = flat.float() * topk_weights.reshape(-1, 1)
    result = (
        weighted.view(hidden_states.shape[0], topk_ids.shape[1], flat.shape[-1])
        .sum(dim=1)
        .to(out_dtype)
    )
    if out is None:
        return result
    if (
        out.shape != result.shape
        or out.dtype != result.dtype
        or out.device != result.device
    ):
        raise ValueError(
            "gfx1250 Gluon MXFP4 MoE output buffer must match the result's "
            "shape, dtype, and device"
        )
    out.copy_(result)
    return out


def gluon_mxfp_ragged_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    w_mx_scale: torch.Tensor,
    x_global_scale: torch.Tensor | float | None = None,
    x_mx_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    x_format: str = "e4m3",
    a_ragged_metadata=None,
    gather_indx=None,
    scatter_indx=None,
    fused_activation=None,
    n_tokens=None,
    n_expts_act=None,
    **extra_kwargs,
) -> torch.Tensor:
    """Tokenspeed-style wrapper around ``matmul`` for routed MoE calls."""
    if out_dtype is None:
        out_dtype = x.dtype if x.dtype.is_floating_point else torch.bfloat16
    gather_indx = _adapt_index(gather_indx, "src_indx")
    scatter_indx = _adapt_index(scatter_indx, "dst_indx")
    scale_preshuffle = bool(extra_kwargs.pop("scale_preshuffle", False))
    scale_load_mode = "swizzle" if scale_preshuffle else "transpose"
    extra_kwargs.pop("scale_load_mode", None)
    w_transpose = bool(extra_kwargs.pop("w_transpose", True))
    extra_kwargs.pop("w_preshuffle", None)
    gate_scal = extra_kwargs.pop("gammas", None)
    gate_scal = extra_kwargs.pop("gate_scal", gate_scal)
    allowed = {
        "num_buffers",
        "scale_block",
        "block_m",
        "block_n",
        "block_k",
        "group_m",
        "xcd_swizzle",
        "schedule",
        "pingpong",
        "num_warps",
        "l2_prefetch_distance",
        "partial_tdm",
        "resolve_partition_conflicts",
        "tdm_split",
        "decode",
    }
    launch_kwargs = {k: extra_kwargs.pop(k) for k in list(extra_kwargs) if k in allowed}
    combine_launch_kwargs = {
        k: v
        for k, v in launch_kwargs.items()
        if k in {"block_m", "block_n", "block_k", "num_buffers", "num_warps", "decode"}
    }
    unsupported = sorted(extra_kwargs)
    if unsupported:
        raise TypeError(f"unsupported gfx1250 MoE keyword(s): {unsupported}")

    if scatter_indx is not None and gather_indx is None:
        return gluon_mxfp_combine(
            x,
            w,
            w_mx_scale,
            x_scale=x_mx_scale,
            x_format=x_format,
            x_global_scale=x_global_scale,
            bias=bias,
            a_ragged_metadata=a_ragged_metadata,
            scatter_indx=scatter_indx,
            gate_scal=gate_scal,
            n_tokens=n_tokens,
            n_expts_act=n_expts_act,
            out_dtype=out_dtype,
            scale_load_mode=scale_load_mode,
            w_transpose=w_transpose,
            **combine_launch_kwargs,
        )
    if fused_activation is not None:
        if x_format == "e2m1" and x_mx_scale is None:
            raise ValueError("x_mx_scale is required for e2m1/MXFP4 activation input")
        if x_format != "e2m1" and x_mx_scale is not None:
            raise ValueError(
                "x_mx_scale is only supported for e2m1/MXFP4 activation input"
            )
        launch_kwargs.setdefault("block_n", 256)
        launch_kwargs.setdefault("block_k", 256)
        launch_kwargs.setdefault("num_warps", 4)
        launch_kwargs.setdefault("num_buffers", 3)
    precision = PrecisionConfig(
        out_dtype=out_dtype, a_mx_scale=x_mx_scale, b_mx_scale=w_mx_scale
    )
    gather_tensor = _index_tensor(gather_indx, "src_indx")
    decode = bool(launch_kwargs.pop("decode", False))
    m = int(x.shape[-2] if gather_tensor is None else gather_tensor.shape[0])
    num_experts = None if a_ragged_metadata is None else a_ragged_metadata.n_slices
    block_m = launch_kwargs.pop("block_m", None)
    if block_m is None:
        block_m = _resolve_block_m(decode, m, num_experts, is_combine=False)
    out, _ = matmul(
        x,
        w,
        bias,
        a_ragged_metadata=a_ragged_metadata,
        gather_indx=gather_tensor,
        scatter_indx=_index_tensor(scatter_indx, "dst_indx"),
        precision_config=precision,
        fused_activation=fused_activation,
        block_m=block_m,
        x_global_scale=x_global_scale,
        scale_preshuffle=scale_preshuffle,
        w_transpose=w_transpose,
        decode=decode,
        **launch_kwargs,
    )
    return out


__all__ = [
    "PrecisionConfig",
    "gluon_mxfp_combine",
    "gluon_mxfp_precomputed_mxfp4_fused_moe",
    "gluon_mxfp_ragged_matmul",
]
