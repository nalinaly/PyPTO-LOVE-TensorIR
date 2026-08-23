# GOAL

**Goal ID:** `PYPTO-NVIDIA-QWEN35-V1`

**Execution status:** blocked on an external protected-workload/resource gate
as of `2026-08-24T05:42:23+08:00`; the full objective and acceptance criteria
remain active and unchanged. Resume with a fresh
`python tools/preflight.py --mode heavy` only after the protected zcode/gem5
lane exits naturally and memory recovers. Never signal those processes.

Build a usable, high-performance NVIDIA SM120 backend for the authorized PyPTO
source; internalize NVIDIA TensorIR/CUDA Tile behind the single public `pypto`
compiler product; provide a standalone `pypto-kernels` project for all custom
high-performance operators; integrate through zero-source-diff,
commit-pinned TorchInductor and SGLang plugins; and run Qwen3.5-0.8B followed by
Qwen3.5-9B text generation on the local RTX 5090 Laptop GPU with strict 100%
PyPTO model-forward compute coverage.

## Non-negotiable acceptance

- Target: one RTX 5090 Laptop GPU, SM120, BF16, TP=PP=DP=1.
- Qwen3.5-9B model-forward compute coverage is exactly 100% PyPTO.
- No Triton, FlashInfer, CuTeDSL, sgl-kernel, ATen eager CUDA, cuBLAS, or
  cuBLASLt compute fallback in final strict runs.
- CUDA runtime operations such as memcpy, memset, allocator, stream/event, and
  CUDA Graph management are not compute fallback.
- PyTorch and SGLang official source checkouts remain clean.
- All handwritten high-performance operators, schedules, references,
  benchmarks, and tuning data live in `projects/pypto-kernels`.
- Framework adapters contain no kernel algorithms.
- No CPU offload or quantization is used to make 9B fit. If minimal BF16 9B
  cannot start on 24 GB, preserve evidence and stop for user direction.
- No system/WSL restart without explicit user approval.
- `/home/zhaosiying/amdgpu-sim` and `/home/zhaosiying/zcode-lane` are read-only
  external scopes. Never modify their files or signal their processes.
