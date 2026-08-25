# GOAL

**Goal ID:** `PYPTO-NVIDIA-QWEN35-V1`

**Execution status:** active. CP-0033 accepts the CPU/fake-driver
`NvidiaExecutable` v1 lifecycle, prepared launch packet and product-isolation
contract at PyPTO `2842a1c...`, layered over CP-0032 ArtifactCache and the
strict private TensorIR/CUDA Tile SM120 Cubin producer. Real libcuda module
load and non-default current-stream execution remain unaccepted because the
exclusive GPU gate is red for the protected ZCode/gem5/SGLang lane. The next
narrow transaction is that exact-product RTX 5090 smoke when the gate is green;
frontend-HIR lowering, operators, framework routes and model execution remain
later. The exact
PyTorch-pinned Triton reference wheel stays audited/frozen and deliberately
uninstalled as baseline-only infrastructure. The full objective and acceptance
criteria remain unchanged. Never signal protected amdgpu-sim/zcode processes.

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
