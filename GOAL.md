# GOAL

**Goal ID:** `PYPTO-NVIDIA-QWEN35-V1`

**Execution status:** active. CP-0037 records the v3 real-SM120 child that
completed six static/dynamic/scalar non-default-stream lifetimes and published a
provisional result, but whose CPU finalizer failed closed before acceptance.
The failure was a root-control defect: `NvidiaTargetInfo` canonically serialized
compute dtypes as `[FP32,BF16]`, while the v3 finalizer fixture handwrote
`[BF16,FP32]`. PyPTO `206447c`, its DSO and Cubins are unchanged. Root
`5564008` plus manifest-only `7639d82` defines one producer-derived ordered
contract, an exact validator and a complete negative matrix through immutable
manifest v4. All v4 CPU/control tests pass. The v3 child remains diagnostic and
cannot be promoted across control versions. No finalized PyPTO GPU correctness
result is accepted yet; the next transaction is a fresh exact v4 RTX 5090
smoke and no-site finalization.
Frontend-HIR lowering, operators, framework routes and model execution remain
later. The exact PyTorch-pinned Triton reference wheel stays audited/frozen and
deliberately uninstalled as baseline-only infrastructure. The full objective
and acceptance criteria remain unchanged. Never signal protected
amdgpu-sim/zcode processes.

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
