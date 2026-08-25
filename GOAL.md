# GOAL

**Goal ID:** `PYPTO-NVIDIA-QWEN35-V1`

**Execution status:** active. CP-0035 accepts the committed correctness-only
SM120 smoke harness and its externally anchored root-control manifest at
`394b75a` plus `2b53f0a`. It binds the CP-0034 live observation seam, CP-0033
NvidiaExecutable and strict private TensorIR/CUDA Tile producer into one exact
`-E/-I -B -S` parent/child/finalizer route, with protected-lane NVIDIA mapping
and compute-PID audits, a pre-release barrier, six module lifetimes, semantic
Artifact replay and no performance/model claim. All control and replay gates
pass CPU-only; no production observation, libcuda module load, kernel launch or
CUDA numerical result has run. The next transaction is the exact authorized
RTX 5090 static/dynamic/scalar non-default-current-stream correctness smoke.
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
