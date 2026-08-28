# TokenSpeed-Kernel-AMD

TokenSpeed-Kernel-AMD is a standalone collection of performance-oriented AMD GPU kernels for LLM inference. Its performance kernels are written in [Gluon](https://www.youtube.com/watch?v=KqeI23SpJx8). The package currently covers architecture-specific implementations for:

- Attention: MHA, MLA, DSA, and KDA
- MoE: BF16, MXFP4 weights
- GEMM

## Performance Numbers

The following measurements use GPT-OSS 120B workloads on one AMD Instinct MI355X GPU. They were collected at TokenSpeed commit [`1492030`](https://github.com/lightseekorg/tokenspeed/commit/1492030a2a02d32bc7011645a74d2d691e99c2e6), with AITER 0.1.13 and ROCm 7.2.1. See the [TokenSpeed-Kernel PyTorch blog](https://pytorch.org/blog/lightseek-tokenspeed-kernel/) for the complete methodology and analysis.

### Attention

![GPT-OSS BF16 GQA causal prefill throughput on MI355X](assets/attention-performance.png)

The benchmark uses BF16 Q/K/V, head dimension 64, 64 query heads, 8 KV heads, full causal attention, and attention sinks. It covers sequence lengths 1K, 4K, and 8K with batch sizes from 1 to 16. Bars report throughput in TFLOP/s; higher is better. The Gluon kernel is the fastest evaluated backend. It is 1.4-2.3x faster than the Triton baseline and 1.1-1.3x faster than AITER.

### MoE

![GPT-OSS 120B MoE latency on MI355X](assets/moe-performance.png)

This benchmark measures full-MoE latency, including routing, both GEMMs, clamped SwiGLU, and combine, for GPT-OSS 120B with 128 experts, top-4 routing, MXFP4 weights, FP8 activations, and `D = I = 2880`. For small decode batches (`M = 1-4`), Gluon is 1.7-2.1x faster than Triton and 1.1-1.6x faster than AITER. At `M = 8-16`, Gluon remains 1.3-1.4x faster than Triton. The prefill results show Gluon substantially ahead of Triton and competitive with AITER across `M = 512-8192`.

## Package Organization

Kernels are organized by AMD architecture and operator family. This keeps architecture-specific tuning local while giving each family a consistent place for its implementations and supporting utilities.

```text
tokenspeed-kernel-amd/
└── python/
    └── tokenspeed_kernel_amd/
        └── ops/
            ├── gfx950/
            │   ├── attention/
            │   │   ├── mha/   # Multi-head attention
            │   │   ├── rmha/  # Relative-bias multi-head attention
            │   │   ├── mla/   # Multi-head latent attention
            │   │   ├── dsa/   # DeepSeek sparse attention
            │   │   └── kda/   # Kimi delta attention
            │   ├── gemm/      # GEMM implementations
            │   ├── moe/       # MoE implementations
            │   └── sampling/  # Sampling implementations
            └── gfx1250/
                ├── attention/
                │   ├── mha/
                │   ├── mla/
                │   └── kda/
                └── moe/
```

Public entry points currently remain architecture-specific. Consumers should import the implementation matching the target GPU, or use TokenSpeed-Kernel to select a compatible implementation through its registry.

### gfx1250 MXFP4 gather and scatter indices

The gfx1250 MXFP4 kernels partition gather and scatter index rows across every
warp in the CTA. The shared index-layout helper uses the row-distributed
`BlockedLayout([1, NUM_INDICES // NUM_WARPS], [32, 1], [1, NUM_WARPS], [0, 1])`
and its consumers remove dimension 0 with `SliceLayout(0, ...)`.

Do not replace this with the superficially similar
`warpsPerCTA=[2, 4]`/`SliceLayout(1, ...)` shortcut. With 32-bit gather indices,
that layout's two producer waves arise from the compiler splitting 16 indices
into two eight-index instructions. It couples producer ownership to index
packing instead of distributing rows over all eight warps, and its behavior
therefore changes when index width is corrected independently.

Descriptor index width is selected separately from pointer-address width. The
kernel uses 16-bit TDM indices only when every possible gather row and the
scatter masked-off sentinel fit in unsigned 16 bits; otherwise it uses 32-bit
TDM indices. Pointer arithmetic remains 32- or 64-bit according to the tensor
extent. Hard-coding 16-bit indices is not a safe shortcut: a gather input with
more than 65,536 rows, or a scatter sentinel above 65,535, would truncate.

## Usage

Install a ROCm-compatible PyTorch build first, then install the package from PyPI:

```bash
pip install tokenspeed-kernel-amd
```

For development from this repository:

```bash
pip install -e ./tokenspeed-kernel-amd
```
