# GOAL

**Goal ID:** `PYPTO-NVIDIA-QWEN35-V1`

**Execution status:** active. CP-0038 accepts the finalized minimal real-SM120
`NvidiaExecutable` correctness v1 result. CP-0039 through CP-0041 add the
compile-free HIR emitter, standalone schedule identity and compiler-owned
frontend specialization/ABI identity. CP-0042 advances PyPTO to `642ff5b` with
the public one-producer `compile_structured_strict` transaction: one concrete
private producer call returns a joined final `KernelBuildSpec` and immutable
`Artifact`. Exact backend-ON and backend-OFF native/Python gates pass. This
CP-0043 accepts the separately versioned, manifest-bound FP32/BF16 frontend
correctness-smoke controls. CP-0044 now accepts their finalized real-SM120
result: two HIR programs, two one-producer Artifacts and four non-default-stream
correctness lifetimes with no fallback. CP-0045 advances PyPTO to `b83fcd3` and
accepts the bounded generic fused-pointwise V2 source/identity/ABI plus
backend-OFF/ON TensorIR/CUDA Tile Cubin production. Fused-pointwise GPU
numerical correctness remains a separate gate. RowReductionV3 is source-review
GO at `17b2b3c` but unbuilt; structured matmul follows. CP-0046 accepts the
separately reviewed nine-case fused-pointwise GPU-smoke controller, CPU
finalizer and deterministic compiler anchors at root `c98f984` plus manifest
`438c25f`; no real fused-pointwise GPU result is accepted yet.
Generic operators, framework routes and model execution remain later. The exact
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
