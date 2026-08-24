# PLAN

**Plan:** `PYPTO-NVIDIA-QWEN35-V1`, revision `13`

## Current phase: R0 workspace and provenance bootstrap

Checkpoint `CP-0022` accepts immutable SM120 TargetInfo at PyPTO `042878d` and
freezes the exact Triton dependency/wheel/probe/replacement machinery plus the
shared/exclusive environment transaction law. A live CPU-only control suite
passes beside an active protected lane without exposing CUDA or signalling it.
R0 stays open until the source-anchored Triton wheel replaces the inherited
editable, the CPython 3.12 baseline is locked, and unmodified SGLang baselines
run.

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
- P10: final E2E benchmarks, coverage proof, and performance report.

Every milestone is correctness-first, then performance, then evidence and a
checkpoint commit. A green smoke test is never promoted to a later acceptance
claim.
