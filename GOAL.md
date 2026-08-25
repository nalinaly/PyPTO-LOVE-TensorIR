# GOAL

**Goal ID:** `PYPTO-NVIDIA-QWEN35-V1`

**Execution status:** active. CP-0036 preserves the fail-closed first real-SM120
diagnostic and accepts the generic CUDA parameter-ABI/enumeration repair at
PyPTO `206447c`, plus the externally anchored v3 smoke controls at root
`c71f32b` and `3de4cf7`.
The first run reached real PyTorch Runtime observation and libcuda
module/function prewarm, then stopped in static parameter-ABI validation before
`prepare_launch`. The exact failed predicate was not persisted; source/Cubin
audit found a four-byte dynamic-metadata defect and a separate live Driver
enumeration-order assumption. The final repair validates bounded offset ranges
and the width multiset independent of enumeration order, keeps launch pointers
in signature order, and packs dynamic size/stride as four-byte `int32`. Fresh
ON/OFF products, exact-DSO suites and all root controls pass CPU-only. CUDA Tile
host block remains the required logical `[1,1,1]`; Cubin `REQNTID` is internal
worker metadata. No PyPTO kernel launch or CUDA numerical result is accepted
yet. The next transaction is the exact v3
authorized RTX 5090 static/dynamic/scalar non-default-current-stream smoke.
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
