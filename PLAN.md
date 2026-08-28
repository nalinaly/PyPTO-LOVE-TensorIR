# PLAN

**Plan:** `PYPTO-NVIDIA-QWEN35-V1`, revision `48`

## Current phase: full-stack execution to Qwen3.5 (D-0018 lightweight gates)

## Authoritative final model gate (2026-08-28)

The final objective is stable inference for **both Qwen3.5-0.8B and
Qwen3.5-9B**, in that order, with model-forward compute coverage exactly 100%
PyPTO. The accepted provider set consists only of handwritten
`pypto-kernels` operators and TorchInductor-fused regions lowered through the
PyPTO CUDA backend. Final strict runs permit no Triton, FlashInfer, CuTeDSL,
sgl-kernel, eager ATen CUDA, cuBLAS/cuBLASLt or unknown compute fallback;
machine-readable coverage must report a non-vacuous denominator,
`coverage = 100%` and `fallback_compute_kernels = 0`.

Both model gates use the exact prompt below; changing, shortening or replacing
it is not acceptance evidence:

```text
为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？
```

For each model, acceptance requires all of the following on the same pinned
weights and tokenizer:

1. compare candidate and unmodified reference-path prefill/decode logits with
   thresholds frozen before the candidate measurement, and publish the full
   max-absolute/max-relative and token-margin evidence rather than only a
   readable answer;
2. use deterministic greedy decoding and require exact generated token IDs and
   decoded text against the reference path;
3. pass at least three fresh model/server starts, each followed by at least ten
   consecutive warm requests using the exact prompt, without crash, hang,
   fallback, coverage drift, state/cache leakage or output drift; and
4. preserve separate machine-readable correctness, stability, coverage and
   performance reports. A compile, Cubin, operator smoke, layer test or one
   successful generation cannot substitute for this model gate.

The immediate path to that gate is to finish the current single-graph,
single-launch fused QK RMSNorm plus partial RoPE compiler transaction, then
close causal paged attention, stateful convolution and single-graph GDN,
complete the zero-diff SGLang route, bring up and stabilize 0.8B, and only then
repeat the full gate for 9B. No model-name, hidden-size or fixed benchmark-shape
special case may be used to satisfy either model.

The historical 22 GiB CPU-v2 admission value is not a compiler memory
requirement and is no longer a prerequisite for this execution path. It was a
conservative host-coexistence reserve derived from an approximate user
authorization, without owned-build peak-memory evidence. Bounded CPU builds
remain capped at `--parallel 2`, must record owned-process RSS and host
`MemAvailable`, retain the 16 GiB running-child pause boundary, and must never
signal a protected external process. The exact-hashed v2 controls remain
unchanged solely so historical evidence stays reproducible.

CP-0039 accepts compile-free HIR-to-TensorIR emission, CP-0040 accepts the
standalone canonical schedule, and CP-0041 accepts compiler-owned frontend
specialization/ABI identity. CP-0042 advances PyPTO to `642ff5b`: the public
`compile_structured_strict` facade prepares HIR, hard-calls the concrete private
producer once, seals source/request/options identity, finalizes the real
callable ABI, constructs one Artifact from that same move-only result and
returns only the joined immutable pair. Backend-ON passes native 7/7 and Python
182/2; backend-OFF passes native 5/5, functional Python 7/7 and full Python
175/9. CP-0043 accepts a separately versioned, two-layer manifest-bound
HIR-authored FP32/BF16 SM120 smoke controller and CPU-only replay finalizer.
Its source, exact-DSO and complete synthetic finalization gates pass. CP-0044
accepts the finalized real-SM120 run: two one-producer frontend compilations and
four fresh non-default-stream lifetimes are numerically correct with no
fallback. The current transaction generalizes the add-only emitter and frontend
identity boundary to a bounded fused-pointwise chain while preserving every
accepted vector-add byte and control. CP-0045 now accepts that compiler/Cubin
transaction at PyPTO `b83fcd3`; V2 GPU numerical correctness is still open.
CP-0046 closes the reviewed nine-case GPU control/finalizer and CPU anchor
boundary. CP-0047/EV-0060 now closes its separately versioned policy-2
real-SM120 execution and CPU-only no-replace finalization. The fixed eighteen
lifetimes pass with candidate-versus-Torch zero ULP and independent-CPU error
at most one ULP. This is the frozen nine-case correctness claim, not a general
operator or performance claim.

The fused-pointwise compiler/Cubin gate is complete. Fresh backend-OFF and
external backend-ON builds, native 3/3 and exact-product Python 1/1 pass; EV-0058
binds both DSOs, four JUnits, all sidecars and diagnostic lineage. Independent
review is GO with P0/P1/P2 = 0. This is CP-0045, not V2 GPU correctness. The
next separate compiler family is
`RowReductionV3` boundary: dense static rank-1-through-32
reduce-last/keep-dim, one flattened outer-row tile/grid (frozen by direct
rank-1/rank-2/rank-3 producer fixtures), and explicit BF16-to-FP32
reduction-to-BF16 conversion so BF16 sum does not silently accumulate in BF16.
Its two source-only commits end at PyPTO `17b2b3c`: the follow-up closes CUDA
Tile element-count and i32 contraction-loop bounds, adds actual no-loop/looped
producer fixtures and complete source/projection goldens. Independent static
re-review is GO with P0/P1/P2 = 0. CP-0048 adds the real-build test fix and
advances the primary checkout to `62eb882`; fresh OFF/ON products, native/
Python gates and four exact Cubin records pass independent review. Reduction
GPU numerical correctness and performance remain pending.
The following structured-matmul source map is also frozen: bounded static BF16
rank-2/equal-batch-rank-3 HIR, TensorIR BF16-by-BF16-to-FP32 matmul, explicit
FP32-to-BF16 output conversion, explicit transpose views, normalization-driven
schedule arity (especially `M=1` decode), and output descriptor 2 as the static
grid source. Source-only implementation now ends at PyPTO `d755117` on top of reviewed
RowReductionV3. Two independent reviews are GO with P0/P1/P2 zero; build,
TensorIR/CUDA Tile production, Cubin, runtime and performance gates remain
pending.

Per D-0018 (2026-08-27) the user relaxed the process gates until the
models run: no per-transaction source reviews, no two-reviewer GO, no
manifest-only ceremonies; a layer passes when its tests and golden
comparisons are green and its build links, with one consolidated review
after 0.8B/9B run. Safety boundaries are unchanged. The execution order
is: finish StructuredMatmulV4 host/Cubin evidence, then generic
pointwise/reduction/indexing/matmul coverage, the TorchInductor PyPTO
CUDA backend plugin, pypto-kernels attention/GDN, the SGLang plugin,
0.8B bring-up, profiling, then 9B, then the D-0017 comparison report.

The user reconfirmed the final objective on 2026-08-26: the end state is
Qwen3.5-9B text generation executing with 100% PyPTO model-forward compute
kernels — handwritten `pypto-kernels` operators plus TorchInductor auto-fused
regions lowered through the PyPTO CUDA backend — benchmarked against the
unmodified SGLang default optimized kernel stack on the same RTX 5090 Laptop
GPU, same model, same workload schedule and batching. Acceptance requires both
the end-to-end comparison and a per-kernel/per-operator breakdown table
(kernel/provider, call counts, GPU time, mean, candidate-versus-baseline
delta per operator class: attention prefill/decode, GDN prefill/decode, GEMM,
pointwise/reduction/indexing fusion), produced from identical profiling
methodology on both lanes. D-0017 freezes this contract.

Checkpoint `CP-0038` accepts the finalized minimal real-SM120
`NvidiaExecutable` correctness v1 report from run `080254`, SHA
`727362d7...272a9`. Static, dynamic-metadata and by-value scalar Artifacts each
complete twice on the caller's non-default stream with reference equality,
external synchronization and explicit unload. Matching v4 finalization joins
all controls, sidecars, serialized compiler inputs, Artifacts, Cubins and live
TargetInfo with no fallback. This closes only the low-level compiler/runtime
launch gate. CP-0042 closes the one-producer frontend Artifact facade, CP-0043
closes its separate smoke controls, and CP-0044 closes HIR-authored vector-add
real-SM120 correctness. CP-0045 closes fused-pointwise compiler/Cubin evidence;
CP-0046 closes its execution controls, CP-0047 closes its real-SM120 numerical
result, and CP-0048 closes RowReductionV3 host compiler/Cubin production.
RowReductionV3 real-SM120 correctness is current, followed by structured
matmul.

Checkpoint `CP-0037` records, but does not accept, v3 run
`pypto-20260825T073624Z-900485-7df250`. Its real GPU child completed six
static/dynamic/scalar lifetimes and published provisional SHA `64c0906b...d34cfe`,
but the no-site finalizer failed because its handwritten dtype order disagreed
with canonical `NvidiaTargetInfo` order. Root A4 `5564008` and manifest-only B4
`7639d82` share exact `[FP32,BF16]` ordering, reject malformed/order-drift
evidence, and pass a clean 225-test/113-subtest suite. No PyPTO rebuild was
needed. CP-0038 later closes the gate with a fresh matching v4 run/finalizer;
cross-version promotion of the v3 provisional remains forbidden.

Checkpoint `CP-0036` records a fail-closed real-runtime diagnostic and accepts
the resulting generic ABI repair, not GPU correctness. Run
`pypto-20260825T052038Z-800777-8e8e83` passed every isolation
gate, observed the real RTX 5090/Runtime/context, compiled all three Cubins and
loaded the first module/function, then failed in static repetition zero during
`prewarm` before packet preparation or launch. PyPTO `206447c` restores the
four-byte dynamic size/stride ABI, validates the live Driver width multiset and
bounded ranges independent of enumeration order, retains signature-ordered
launch pointers, bounds error text, and preserves CUDA Tile's required logical
host block `[1,1,1]`. Fresh ON/OFF builds pass CTest 9/9 and 7/7, exact-DSO
Python 142/2 and 135/9, and the complete product audit. Root `c71f32b` plus
manifest-only `3de4cf7` bind the final DSO through immutable control manifest
v3; post-manifest run `pypto-20260825T071601Z-892819-67acee` passes 224 tests
plus 106 subtests. Its scheduled v3 transaction became the unfinalized CP-0037
diagnostic; CP-0038 later closes the v4 route and frontend execution is current.

Checkpoint `CP-0035` accepts the fixed correctness-only SM120 smoke control
path, not a GPU result. Root implementation `394b75a` and manifest-only commit
`2b53f0a` bind the exact controller, preflight, stop tool, runner, finalizer and
contract blobs. Controller/finalizer/preflights use `-E -B -S`; the child uses
`-I -B -S`, an empty `PYTHONPATH` and no plugins. Parent and child prove the
protected lane has no NVIDIA mapping/compute PID before Torch import, and the
owned watchdog identifies compute only through PID start-tick, descendant and
PGID. Static/dynamic/scalar Cubins compile deterministically CPU-only; replay
through the exact DSO reconstructs full TargetInfo, BuildSpecs, ABI and Cubin.
The post-manifest root suite passes 224 tests plus 106 subtests. At CP-0035 no
CUDA context, module or kernel had been used. Its scheduled live transaction is
now preserved as the CP-0036 failed diagnostic. The later CP-0037 v4 route is
closed by CP-0038; frontend execution is current.

Checkpoint `CP-0034` accepts only the parent-process NVIDIA runtime observation
value at PyPTO `6361f11`. It reuses the private Driver boundary to produce every
live TargetInfo field, numeric Driver/Runtime API versions, authenticated
Runtime-provider path and diagnostic regular-context identity without retaining
handles or loading Cubin. `dlsym(RTLD_DEFAULT)` plus `dladdr` and canonical
expected-path equality never opens libcudart. Distinct-sentinel, provider,
fork-latch and backend-OFF tests pass; fresh ON/OFF products and exact-DSO gates
remain single-DSO and isolated. CP-0034 itself accepted no production
observation; CP-0036 later records the failed diagnostic.

Checkpoint `CP-0033` accepts only the CPU/fake-driver NvidiaExecutable v1
contract at PyPTO `2842a1c`. It adds a separate internal runtime object target,
lazy typed Driver resolution, process/device/context binding, forced function
load, parameter/resource legality, prepared allocation-free launch packets,
graph/module lifetime leases and strict ON/OFF product isolation. Fresh ON
CTest is 9/9 with exact-DSO Python 142 passed/1 skipped; fresh OFF is 7/7 and
134 passed/9 skipped. No real CUDA Driver call or launch occurred. The
exclusive `gpu-benchmark` gate returned 75 for the active protected lane while
reporting no NVIDIA compute PID, so the next transaction remains an exact
non-default-stream RTX 5090 smoke without waiver.

Checkpoint `CP-0032` accepts the persistent ArtifactCache v1 at PyPTO
`c087170`. Cache identity is derived only from the accepted precompile
projection; owner-private descriptor-relative reads fully revalidate canonical
Artifact bytes, and same-shard no-replace publication is durable and
concurrency-safe. Fresh backend-ON/OFF builds, exact-product Python suites, DSO
audits, seven-source provenance negatives and independent API/security/final
reviews pass. No CUDA state, compile-on-miss, eviction, repair, framework or
model behavior is present. The next narrow transaction is a process/device/
CUcontext-bound `NvidiaExecutable` that consumes only a validated Artifact and
accepts the non-null current stream only at launch. CP-0033 now accepts its
CPU/fake-driver contract; real CUDA execution remains the active gate.

Checkpoint `CP-0031` accepts the strict canonical-source producer bridge at
PyPTO `f3bcaac` and TensorIR `1dcb38c`. Exact source bytes plus
CompileRequest/KernelBuildSpec now produce an immutable SM120 Cubin Artifact v1
through the private TensorIR/CUDA Tile/pinned-tileiras route, with no fallback,
ambient override or vendor type in the public API. Fresh ON/OFF products,
source-isolated Python suites, DSO audits, provenance negatives and the root
control regression pass. Provenance covers the clean PyPTO parent plus all six
compiled direct submodules, including msgpack-c, libbacktrace and runtime. The
then-next narrow transaction, the compiler-owned persistent ArtifactCache, is
now accepted by CP-0032. CUDA handles/current-stream launch, frontend-HIR
lowering, operators, frameworks and models remain later gates.

Checkpoint `CP-0030` accepts immutable canonical NVIDIA Artifact v1 at PyPTO
`4a82f2e`. It binds exact clean PyPTO/vendor/pipeline identities, strict SM120
TensorIR options, schedule-to-grid/uniform ABI, one canonical Cubin,
entry/flattened CUDA parameter ABI, cache/loader projections and bounded
canonical MessagePack. TensorIR `b25081a` performs runtime-free CUDA 13.3 ELF,
entry, KPARAM/PARAM_CBANK, constant-bank and PT_LOAD mapping validation. Its
then-next strict producer transaction is now accepted by CP-0031. Cache, CUDA
runtime, framework routes and model execution remain later.

Checkpoint `CP-0029` accepts only the private bounded canonical MessagePack
foundation at PyPTO `6ce1776`. It preserves CompileRequest/KernelBuildSpec wire
identity while adding streaming aggregate allocation limits and BIN support
needed by Artifact. CP-0030 and CP-0031 have now consumed that foundation;
cache, CUDA runtime and frameworks remain later steps.

Checkpoint `CP-0028` accepts only the runtime-free in-memory TensorIR producer
result. TensorIR `2677d1a` emits fully validated TileIR/Cubin bytes plus
complete reconstruction metadata before its legacy runtime object; PyPTO
`4789ae0` preserves one symbol-isolated DSO and a narrow LLVM ABI bridge.

Checkpoint `CP-0027` accepts the private compiler composition only. PyPTO
`5f75568` contains TensorIR `233ab6e`, CUDA Tile `af241704`, and LLVM
`57109bef` in one RPATH-free `pypto_core` DSO; the exact-source compiler suite
passes 123/123. The next transaction is a pointer-free bytes-plus-metadata
Artifact emitted before any runtime-kernel construction. CUDA module/context,
current stream, launch, framework registration and model claims remain open.

Checkpoint `CP-0022` accepts immutable SM120 TargetInfo at PyPTO `042878d` and
freezes the exact Triton dependency/wheel/probe/replacement machinery plus the
shared/exclusive environment transaction law. A live CPU-only control suite
passes beside an active protected lane without exposing CUDA or signalling it.
R0 stays open until the source-anchored Triton wheel replaces the inherited
editable, the CPython 3.12 baseline is locked, and unmodified SGLang baselines
run.

Checkpoint `CP-0023` accepts pointer-free CompileRequest v1 at PyPTO `09e014c`.
Canonical bounded MessagePack, exact toolchain/target policy and the separate
byte/loader-input/device identity projections pass native 2/2 and Python 62/62
CPU-only gates. KernelBuildSpec and every producer/artifact/runtime layer remain
open.

Checkpoint `CP-0024` accepts the exact Triton dependency closure. All ten
archive SHA/byte pairs, expanded trees and manifest `29c073...` are independently
reviewed and source-locked; the networkless tool probe and durable reviewed
cache publication pass. The offline wheel/build/runtime gates remain open.

Checkpoint `CP-0025` accepts pointer-free per-region KernelBuildSpec v1 at
PyPTO `9b3cf71`. Its bounded canonical identity binds source/ABI, semantic
route, exact pipeline revision, all resolved schedule categories,
specialization/mutation and the parent CompileRequest byte identity. Native
CTest is 4/4 and an exact-current-DSO replay is 122/122. Exact producer,
Artifact, cache, CUDA module/current-stream runtime and framework integration
remain open. In parallel, the corrected RPATH-free Triton wheel rebuild is
running under the CPU-only coexistence watchdog.

Checkpoint `CP-0026` freezes the exact PyTorch-pinned Triton reference wheel.
Its complete audit and pip-free fresh probe pass; it remains uninstalled and
cannot satisfy PyPTO strict coverage. The active implementation lane now moves
to the private TensorIR/CUDA Tile/exact-LLVM static build and artifact seam.

Checkpoint `CP-0005` has source/model/environment provenance, the first
standalone semantic operator layer, Torch constructor-dispatch law, and
candidate/baseline process isolation frozen. R0 remains open until exact Triton
wheels replace the inherited editable source, the CPython 3.12 baseline
environment is locked, and unmodified SGLang baselines are captured. In
parallel, the first P1 build-only object boundary is accepted at PyPTO
`042878d...`; the next compiler transaction is the pointer-free CompileRequest
contract, while exact Triton replacement proceeds as R0 compatibility work.

Checkpoint `CP-0006` additionally freezes the strict runtime-coverage evidence
contract: fixed collector revision, closed trace and artifact-registry digest
reconciliation, exact non-vacuous denominators, failure latching, and exclusive
report ownership. It is deliberately not runtime coverage evidence. The next
source-only P1 transaction is immutable SM120 target identity, but it must land
only after the staged single-DSO object-boundary commit is independently gated.

Checkpoint `CP-0007` adds catalog-bound operator provenance identity and the
pinned eager-first CUPTI collector map without moving either boundary into the
wrong project. The collector is not implemented and cannot assert a closed
world yet.

Checkpoint `CP-0008` freezes paged-attention ABI v1 and its metadata reference
contract. This is the semantic boundary for later reference and CUDA Tile
implementations, not an attention correctness or performance milestone.

Checkpoint `CP-0009` adds the deterministic CPU numerical reference and state
continuity gate. Independent PyTorch comparison and all CUDA work remain open.

Checkpoint `CP-0010` freezes unified GDN core ABI v1 and the paired-state lease
contract. Numerical GDN and all state-preparation/CUDA work remain open.

Checkpoint `CP-0011` adds the GDN paired-state numerical reference and exact
partition-continuity gate. Independent Torch comparison and CUDA remain open.

Checkpoint `CP-0012` records a source-reviewed but deliberately unbuilt SM120
TargetInfo candidate in a separate worktree. It does not advance P1 acceptance;
object-DSO validation and ordered integration remain mandatory.

Checkpoint `CP-0013` adds the structured matmul numerical oracle. TensorIR/CUDA
correctness comparison and performance remain open.

Checkpoint `CP-0014` closes the source/package identity boundary between the
standalone operator producer and both zero-diff framework adapters. It does not
make the compiler or any operator executable. The packaging-time heavy
preflight was green, but the action-boundary recheck is now red for a new
protected zcode TP=2 vLLM/gem5 lane. Continue only light work until a fresh
green result permits the single-DSO gate before TargetInfo integration.

Checkpoint `CP-0015` independently cross-checks all three current scalar
operator references against CPU Torch while keeping Torch out of the standalone
runtime product and parent import process. CUDA correctness/performance remains
open. The protected heavy lane still blocks the ordered single-DSO gate.

Checkpoint `CP-0016` freezes the source-only operator benchmark evidence
contract and its fail-closed comparison boundary. It publishes no result and
does not advance performance acceptance. Atomic publication and tuning reuse
remain ordered after real CUBIN and complete TargetInfo identity.

Decision `D-0008` freezes generic StateBundle ownership and the post-TargetInfo,
post-current-stream landing order. It is a design constraint, not an
implementation milestone; no state transfer API or executor exists yet.

Checkpoint `CP-0017` freezes the selected SGLang state lifecycle route and
adapter obligations while keeping registration explicitly unready. It does not
move StateBundle implementation ahead of compiler/runtime prerequisites.

Checkpoint `CP-0018` freezes the pinned TorchInductor backend surface before
implementation. The 31-source audit plus full 346-file manifest proves no
TileKernel/cuTile abstraction exists in this pin and records every reviewed
registry, cache, current-stream, explicit fallback and scheduling bypass that a
zero-diff plugin must own or reject. It does not register a backend or generate
code; all readiness flags remain false until the ordered compiler/runtime gates
close.

Checkpoint `CP-0019` repairs the executable acceptance specifications for the
pending single-DSO and TargetInfo transactions and freezes D-0009 from pinned
TensorIR source. CompileRequest is a data-only program/target policy;
byte-producing region identity belongs to KernelBuildSpec; both TensorIR and
direct CUDA Tile routes produce one exact-producer-bound Artifact; loaded
executables are process/device/CUcontext bound. This checkpoint executes none
of those heavy or runtime gates.

Checkpoint `CP-0020` records the third consecutive goal-turn failure of the
same protected-workload gate: seven `zcode-vllm-tp2-v4` processes remain and
MemAvailable is 28.5 GiB below the 32 GiB floor. The goal is blocked on external
state, not complete or narrowed. Resume at the frozen single-DSO runbook only
after a fresh heavy preflight returns zero.

Checkpoint `CP-0021` records user resume and accepts the single-DSO compiler
boundary. Native ownership/CTest, editable and wheel JUnit, fresh one-DSO
packaging, installed dependency/import/console-script and symlink gates all
pass; two audit-script false positives are preserved with successful recovery
run IDs.

Checkpoint `CP-0022` integrates the exact 34-path TargetInfo candidate as
`042878d`. Fresh native CTest is 2/2; the new one-DSO wheel and clean install
pass 31 targeted cases, 10,209 full-suite cases with 57 unchanged skips, and
the independent symlink case. Source/artifact lineage is rebound by a separate
read-only recovery audit. This accepts target identity/build/package/API only,
not CUDA compile or launch. EV-0035 additionally accepts the exact Triton gate
tooling and live protected CPU-only control path; it explicitly records that no
dependency materialization, wheel, GPU smoke, or replacement has run.

Checkpoint `CP-0023` lands only the immutable program/target CompileRequest
data contract. Independent review forced explicit MessagePack allocation limits
before acceptance. The next compiler commit is per-region KernelBuildSpec; it
must not be collapsed into TensorIR composition or ArtifactCache work.

Checkpoint `CP-0024` records only reviewed compatibility inputs. It does not
promote Triton into PyPTO coverage, prove a wheel, or relax the separate
reference-only GPU-smoke and environment-replacement gates.

Checkpoint `CP-0025` lands only the immutable per-region KernelBuildSpec data
contract. A stale installed editable DSO was detected and explicitly excluded
from acceptance; EV-0038 binds the exact current source DSO replay. The next
compiler transaction is exact producer-bound bytes-plus-metadata Artifact, not
runtime handles or framework codegen.

Checkpoint `CP-0026` closes only the reproducible Triton reference-wheel
boundary. Environment replacement and reference GPU smoke are deferred until
the unmodified SGLang baseline requires them. No further Triton feature work is
on the candidate backend path.

Checkpoint `CP-0027` closes only the unified private compiler build boundary.
TensorIR's public compiler usage requirements now support a real parent build,
and dynamic CUDA declarations no longer leak an absolute toolkit RUNPATH. This
does not yet prove a PyPTO lowering, a kernel artifact, CUDA execution, a wheel,
or any Torch/SGLang route.

Checkpoint `CP-0028` accepts only a process-independent in-memory producer
value, not a persistent PyPTO Artifact. Full TileIR/Cubin structure, SM/ABI,
argument/grid metadata, hostile environment, exact assembler bytes and legacy
reconstruction are gated. Canonical serialization, request/build-spec digest
binding, subprocess transfer and cache publication remain explicitly open.

Checkpoint `CP-0029` accepts only the bounded private MessagePack foundation.
Streaming structural limits close decoded-object amplification before the
allocating parse; ON/OFF builds and canonical request/spec replay pass. No
Artifact schema, producer bridge, cache, CUDA runtime or framework route is
claimed.

Checkpoint `CP-0030` accepts only persistent Artifact v1 and its strict
runtime-free Cubin/ABI validation. The producer result remains a manually
constructed boundary in tests; no compiler entrypoint yet consumes canonical
source plus CompileRequest/KernelBuildSpec to create it. ArtifactCache,
subprocess compilation, CUDA module/current-stream launch and framework/model
work remain explicitly open.

Checkpoint `CP-0031` accepts only the strict in-process source-to-Cubin
producer bridge and its bounded exact-assembler process boundary. It does not
accept cache publication, CUDA loading/launch, PyPTO frontend-HIR lowering,
generic codegen, operators, TorchInductor, SGLang or Qwen execution.

Checkpoint `CP-0032` accepts only trusted-local persistent Artifact storage.
It does not accept CUDA device/context/module/function state, resource or
workspace legality, current-stream launch, CUDA Graph, frontend-HIR lowering,
operators, TorchInductor, SGLang, Qwen correctness/coverage or performance.

Checkpoint `CP-0033` accepts only modeled executable lifecycle, packing,
legality, concurrency/fork rules and ON/OFF product isolation. It does not
accept real libcuda resolution, Cubin module load, current-stream execution,
CUDA numerical correctness, CUDA Graph, frontend-HIR lowering, operators,
TorchInductor, SGLang or Qwen execution.

1. Create the control repository, persistence documents, safety preflight, and
   isolated directory layout.
2. Materialize the authorized PyPTO baseline and clean official upstream
   checkouts at the exact versions in `VERSIONS.lock`.
3. Initialize independent `pypto-kernels` and `pypto-framework-plugins` Git
   projects.
4. Clone the `triton-dev` environment into a project-local prefix without
   modifying the original environment.
5. Copy Qwen3.5 weights from the read-only AMD simulator tree into `models/`
   after the default idle gate, or only with the explicit CPU-only coexistence
   flag while the 24 GiB and protected-NVIDIA-compute boundary remains green;
   recheck at each file EOF/publication boundary and verify every hash.
6. Generate the checkout-grounded `docs/implementation_map.txt` and freeze the
   unmodified SGLang baseline before compiler changes.

## Milestone ladder

- R0: workspace/provenance/baseline.
- P1: PyPTO compiler/backend split with unchanged Ascend tests.
- P2: TensorIR/CUDA Tile SM120 runtime closure and PyTorch current-stream ABI.
- P3: generic fused-loop codegen, structured matmul, runtime/cache/tuning.
- P4: zero-diff TorchInductor compatibility plugin and strict MLP gate.
- P5: paged full-attention correctness and performance.
- P6: GDN decode/prefill correctness and performance.
- P7: zero-diff SGLang plugin and Qwen3.5-0.8B strict coverage.
- P8: 0.8B stabilization and full profiling.
- P9: Qwen3.5-9B correctness, strict coverage, and SM120 tuning.
- P10: final E2E benchmarks, coverage proof, per-kernel PyPTO-versus-SGLang-
  default breakdown, and the final performance report.

Every milestone is correctness-first, then performance, then evidence and a
checkpoint commit. A green smoke test is never promoted to a later acceptance
claim.
