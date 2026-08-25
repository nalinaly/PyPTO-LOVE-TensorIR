# CP-0038 finalized runtime-semantics review

- Review time (UTC): `2026-08-25T08:09:13Z`
- Reviewer canonical task: `/root/cuda_tile_semantics_review`
- Run ID: `pypto-20260825T080254Z-910620-c669d9`
- Final report:
  `reports/data/pypto-nvidia-executable-sm120-pypto-20260825T080254Z-910620-c669d9.json`
- Final report SHA-256:
  `727362d7879d58cbee07b11050b17ad149274e8087b0d1872b8f186a66a272a9`
- Root control commit/tree:
  `37c16b3902192a8da59a1d912d2ff3e06bec02fb` /
  `cc90c7d8ec7a58ccd0952c4af8756be89a69507e`
- PyPTO commit/tree:
  `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7` /
  `e0357daaefa74dbf676550015e60701996c400fb`
- Minimal runtime milestone result: **GO**
- Severity: **P0 = 0, P1 = 0, P2 = 0**

## Scope and method

This was a read-only semantics and evidence audit of the finalized v4 SM120
correctness smoke. It checked the final report, its exact provisional and
replay inputs, process/preflight/gate/barrier joins, selected product/control
identities, six execution records, CUDA Tile launch semantics, and explicit
non-claims. The audit did not rerun a GPU command, build a product, modify
source, or signal a process.

The accepted scope is deliberately narrow: the exact three-fixture
`NvidiaExecutable`/TensorIR/CUDA Tile launch path on the recorded RTX 5090 and
Driver. This is not frontend, operator-family, framework, model, coverage,
CUDA Graph, profiling or performance acceptance.

## Final-report identity and finalization

The 29,107-byte final report is a canonical, duplicate-key-free, read-only JSON
document. Its status is `accepted-real-sm120-correctness-smoke`. Its exact
scope is:

- `provider = pypto.tensorir`;
- `runtime_object = NvidiaExecutable`;
- fixed-fixture `operator_correctness = true`;
- `model_forward = false`;
- `strict_coverage_result = false`;
- `cuda_graph_result = false`;
- `performance_result = false`.

The finalizer identity is bound to
`tools/finalize_pypto_nvidia_executable_sm120.py`, SHA-256
`aad7faf215e2aef0dc626553c1f917e443df0f7ffce4d22425c8276ed23e2f55`.
The finalizer joined the exact v4 control manifest, child provisional, all
process-safety sidecars, the exact PyPTO DSO/libcudart/Python/runner files, and
an independent exact-DSO semantic replay before publishing without
replacement.

The final report's `result` equals the provisional's `runtime` object
field-for-field, and its scope equals the provisional scope. The v4 control
identity also agrees across the gate, child, provisional and final report.

## Six executions, current stream, context and unload

The final report contains exactly six execution records, two repetitions for
each fixed case:

| Case | Repetitions | Grid | Kernel arguments | Expected/actual logical SHA-256 |
| --- | ---: | --- | ---: | --- |
| static FP32 add | `0,1` | `[4,1,1]` | 3 | `866734bdeb1c286f7a08783e2350e7b53961991e1fc99205b75d8717ff3b207c` |
| dynamic-stride FP32 add | `0,1` | `[6,1,1]` | 12 | `98f2270fa9656f025dfabfda29c8ca20035b98dcf93562b5b46d24a91dde24e6` |
| FP16 tensor plus FP32 scalar | `0,1` | `[4,1,1]` | 3 | `f11f9b04a58e9f148d364b14f8fa0f4e3630a1845dcb18fc4d55b7e1b0ddc6c0` |

Every record verifies:

- expected and actual logical-output hashes are identical;
- Torch equality is true;
- complete input storage is unchanged;
- dynamic output padding is unchanged;
- the packet remains live through external stream synchronization;
- the packet is released only after synchronization;
- explicit unload succeeds and terminal state is `Unloaded`;
- the executable's bound context address is zero after unload.

All launches use the same caller-selected raw current stream
`109459232792160`. It is nonzero, differs from CUDA's `0`, `1` and `2` special
streams, and was checked by the fixed runner against both the selected public
Torch stream and the default stream before each launch.

All records bind context address/ID `109459200361584 / 1`, exactly matching the
runtime observation. Aggregate counts agree with the records:
`module_lifetimes=6`, `explicit_unloads=6`, and
`repetitions_per_case=2`.

## Reference and Artifact replay

All seven replay-file byte counts and SHA-256 values recompute exactly. The
finalizer's isolated exact-DSO replay reconstructs the CompileRequest,
TargetInfo, three KernelBuildSpecs and three Artifacts, then matches runtime
evidence for target identity, callable/kernel ABI, Artifact identity, source,
cache/loader identity and Cubin bytes.

The three accepted Artifact/Cubin pairs are:

- static:
  - Artifact `411f87920e7a9d9f97f66c865a5695b6b5016ec7983009c47df5c6a3c07b88e9`
  - Cubin `6dc121d2574537753229ed537efc5d2558eee26bfac0ad9d21826b5f33632b82`
- dynamic:
  - Artifact `6914638d762ce5aaa963e4845d5c5fc473cf2c102b719de3260c7b27619711f5`
  - Cubin `eabdc1377c66f2879a8cf77e43b3f705e4d725b71f6c8b30244521e97d72ed60`
- scalar:
  - Artifact `28bf2001d40cfd49c641f5280f4c52cbad2c656377c70acf40ad8bb78e273a3f`
  - Cubin `fff77b041e032eaae3804105578f49b22fd26cd5d9cb0d483f3170c2bc1a4735`

Independent CPU-only deserialization through the exact selected DSO confirms
the same identities and that every Artifact has `fallback_used=false`. It also
reproduces canonical supported-compute-dtype order `[FP32,BF16]` with CUDA
hidden, device count zero and CUDA uninitialized.

## CUDA Tile logical block and occupancy evidence limit

Independent Artifact deserialization confirms all three launch ABIs use
`block_dimensions=[1,1,1]`. Static and scalar use static grid `[4,1,1]`;
dynamic uses tile-based runtime grid policy and produced runtime grid
`[6,1,1]`. This agrees with CUDA Tile's documented rule that host block
dimensions are unused and must remain `(1,1,1)` and with TensorIR's reference
runtime.

Each of the six successful prewarms called
`cuOccupancyMaxActiveBlocksPerMultiprocessor` with logical block size one before
publishing `Ready`. Consequently, the finalized run proves that Driver 610.74
returned a legal nonzero residency result and subsequently launched each exact
Cubin with logical host block `[1,1,1]` on this RTX 5090.

The evidence does not retain the occupancy count and does not prove:

- a physical occupancy value, utilization or efficiency;
- that logical-thread register arithmetic derives CUDA Tile's physical worker
  footprint;
- occupancy for other Cubins, schedules, devices or Drivers;
- performance.

Cubin `REQNTID=[128,1,1]` remains compiler-selected physical metadata. Its
exact relation to four selected worker warps remains an inference and is not a
host `CUlaunchConfig.blockDim*` value.

## Safety boundary

The retained safety sidecars recompute exactly:

- preflight:
  `4c788dbd6ee0b6d68277fc8fd66663f88a70d915b66ee1e7042cc263de0964f3`
- gate:
  `be92af0d23965ccc7c7917529149047f108973a084f4bd8d23dbcc1b1cd43883`
- start barrier:
  `d510ef38bdfb757f3ca0c6be2ed6c5300d7313cb3dfe9448c141766a41cd1cfe`
- process:
  `153f9c69b113c36f4e59c7fd272aea3ff281c70776e6a843ad12578621dddbff`
- provisional:
  `954a266ed5d592698649ad1947fe1879e75dce2f3d9ebad9e16c74034076221f`

Admission, pre-release, periodic and post-exit records identify the RTX 5090
as SM120 with Driver 610.74 and satisfy the memory/resource floors. Every
external NVIDIA compute, protected NVIDIA compute, protected-runtime-mapping
and unreadable-map set is empty. The process exited zero, owned PGID `910707`
has no survivor, and no external signal evidence exists. The final report
therefore records `zero_nvidia_interference=true` within this audited policy.

## Provider and fallback boundary

The execution provider is exactly `pypto.tensorir`. The strict Artifacts and
the result aggregate all record `fallback_used=false`; the forbidden provider
import set is empty. The exact child did not import Triton, SGLang or FlashInfer,
and the final report contains no latency, throughput, bandwidth, FLOPS,
CUDA-event or benchmark claim.

This smoke does not constitute a closed-world framework trace. Torch supplies
allocation, copies, stream ownership and CPU reference comparison. The result
does not establish that arbitrary Torch/Inductor/SGLang graphs are covered by
PyPTO, nor that any custom serving operator family is implemented.

## Severity findings

### P0

None.

### P1

None within the minimal runtime milestone.

### P2

None. The report is internally consistent, all exact-file/replay/process joins
recompute, and its explicit scope/non-claim fields preserve the acceptance
boundary.

## Final decision

**GO for the minimal real-SM120 runtime milestone:** exact PyPTO `206447c`
successfully compiled, loaded, prewarmed, launched, synchronized and unloaded
the fixed static, dynamic-stride and scalar TensorIR/CUDA Tile fixtures twice
each on the caller's non-default current stream, with finalized numerical and
replay evidence.

This GO does not advance or imply:

- frontend HIR-to-TensorIR or direct fused-loop lowering;
- general elementwise/matmul correctness or any custom operator-family
  milestone, including paged attention or GDN;
- TorchInductor or SGLang registration/integration;
- Qwen3.5 model correctness or strict model-forward coverage;
- CUDA Graph correctness;
- profiling, tuning or performance.

Those remain separate later acceptance gates.
