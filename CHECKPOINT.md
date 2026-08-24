# CHECKPOINT

**Checkpoint:** `CP-0027`

**Status:** R0 remains open. P1 now accepts the single-DSO boundary, immutable
SM120 TargetInfo, CompileRequest v1, KernelBuildSpec v1, and exact private
TensorIR/CUDA Tile/LLVM compiler composition. The exact Triton wheel remains
uninstalled and baseline-only. No compiled Artifact, runtime launch, CUDA
operator, framework route, or model milestone is accepted.

## Current truth

- The control repository was initialized in
  `/home/zhaosiying/pypto-love-tensor-ir`.
- Authorized PyPTO, standalone kernel/plugin projects, and clean official
  PyTorch/SGLang checkouts are materialized at the exact locked identities.
- A fully independent project-local Conda environment is cloned and the
  installed CUDA PyTorch tree is frozen by a content digest.
- Both Qwen3.5 snapshots are independently copied under `models/`, made
  read-only, and verified byte-for-byte against the tracked manifest. The AMD
  source tree was never modified.
- The standalone kernel project now has a reviewed, versioned, immutable ABI
  for tensor arguments, operator/problem/schedule identity, tuning records, and
  requested-versus-produced artifact provenance.
- It also has reviewed typed semantic families for generic matmul, paged
  attention, and GDN. These deliberately stop before a concrete tensor ABI or
  kernel implementation until the exact SGLang inventory is frozen.
- Its reviewed persistent tuning database now provides complete key/cohort
  auditing, deterministic winner selection, multiprocess-safe atomic
  publication and corruption rejection. It contains no device measurement yet.
- Structured matmul has a reviewed explicit ABI v1 for contiguous/aligned BF16
  rank-2 and batched rank-3 tensors, FP32 accumulation, transpose semantics and
  output non-aliasing. No lowering, schedule, launch or performance exists yet.
- Structured matmul now also has a reviewed standard-library BF16/FP32
  numerical reference for every transpose combination and explicit rank-3
  batching. It is not yet compared with TensorIR/CUDA.
- The standalone operator project now publishes the only canonical
  framework-adapter ABI manifest and digest from its real adapter types,
  semantic configs, full specs, constants, signatures and ordered catalog.
- Framework plugins independently recompute that ABI, validate the live
  bindings, pin the Python source-tree digest, and bind either exact wheel
  RECORD ownership or strict PEP-660 editable metadata. The copied partial ABI
  schema was removed.
- Current source-only packaging rejects untracked native extensions and
  sourceless bytecode. Torch and SGLang adapter entry paths perform identity and
  executable-readiness checks before framework mutation; pre-strict failures
  cannot be swallowed as ordinary fallback exceptions.
- The operator project has 117 tests plus 71 subtests; the plugin has 128
  tests. Both wheel and real editable identity probes pass. No executable
  kernel, framework registration, model correctness, coverage or performance
  is claimed by this boundary.
- The matmul, paged-attention and GDN scalar references now have independent
  CPU Torch cross-checks in device-hidden child processes. They add no
  production dependency and preserve the framework-free parent import proof.
- Attention now includes shared-prefix PREFILL→DECODE continuity with private
  write tails; GDN uses vectorized Torch primitives and paired-state continuity
  rather than copying the production scalar loop structure.
- The standalone project now has a source-only benchmark JSON v1 contract. It
  joins typed config/descriptors/workload, CUBIN provenance, symmetric
  candidate/baseline measurements, reset/mutation correctness, separate
  profile traces and fixed device conditions before deriving any comparison.
- No result JSON or publisher exists, and tuning promotion is hard-disabled.
  The contract cannot be presented as performance evidence. The standalone
  suite is now 117 tests plus 71 subtests.
- Paired GDN state zero/copy/checkpoint ownership is now frozen as generic
  PyPTO runtime infrastructure, not an operator or framework copy kernel. The
  detailed StateBundle ABI/lease/stream design is documentation only and is
  ordered after single-DSO, TargetInfo and current-stream executable gates.
- The framework plugin now has a pinned active-route SGLang state lifecycle
  inventory: 11 sources and 35 sites across UnifiedRadix/MambaComponent,
  unified slot translation, deferred clear/COW, checkpoint/donate and reuse.
  Its immutable scope keeps registration readiness false. That transaction had
  122 passing tests; the current plugin suite has 128 after the later Inductor
  inventory transaction.
- Operator artifact provenance now exposes a canonical complete-field digest
  restricted to the explicit matmul/paged-attention/GDN ABI catalog. It is only
  a future join input: physical cache/loading stays compiler-owned and the
  opaque problem digest is not yet a typed launch proof.
- Paged attention now has one reviewed ABI v1 for prefill/extend and decode,
  including explicit KV append/cache mutation, fixed-capacity metadata,
  pitched split-QKV views, workspace-aware non-aliasing, host-copy metadata
  validation, read-only shared radix pages, and zero inactive output rows. No
  CUDA implementation exists yet.
- Paged attention now also has a reviewed standard-library numerical reference
  implementing BF16 storage, FP32-rounded causal GQA, append-before-attention,
  prefill-to-decode cache continuity and inactive-output zeroing. It is not yet
  cross-checked against a CUDA/PyTorch baseline.
- GDN now has a reviewed unified core ABI v1 for decode and prefill/extend,
  freezing packed layout, no-bias causal conv, FP32 gate/delta semantics,
  paired BF16-conv/FP32-recurrent state, pitched envelope views and canonical
  metadata/output tails. State preparation/copy and CUDA remain open.
- GDN now also has a reviewed standard-library paired-state numerical reference.
  One-shot prefill, checkpoint-segmented prefill and token decode match exactly
  in output, BF16 conv state and FP32 recurrent state. It is not yet compared
  against pinned Torch/SGLang.
- The framework plugin now binds actual imports to the locked CUDA Torch tree
  and clean SGLang checkout, rejects mixed linear-attention providers, and
  fails before launch while the real PyPTO Inductor dispatcher is unavailable.
- Its constructor-dispatch foundation now has a reviewed ContextVar mode,
  original-backend preservation outside that mode, pinned wrapper proxy
  semantics, and strict no-fallback failures inside it. It deliberately does
  not register with Torch until real scheduling/wrapper constructors exist.
- The zero-diff TorchInductor route now has a reviewed 31-source SHA/AST
  inventory plus a manifest for all 346 Python files under `_inductor`. It
  freezes registry/cache, scheduler/wrapper/current-stream,
  template/lowering/fallback, and Triton/CuTeDSL/autotune reference lanes.
- The audit proves there is no pinned TileKernel/cuTile abstraction to reuse,
  records the existing CuTeDSL path, the closed native CUDA selector and
  mandatory `has_triton()` check, and exposes the eight actual
  `get_current_backend()` call sites.
- It also freezes unsafe surfaces that the plugin must own or reject: CSEProxy
  omits `pypto` dtype/shape propagation, extern codegen bypasses the backend,
  foreach accepts only three upstream scheduling types, multi-template can
  select extern, and MM/BMM producers contain explicit extern choices.
- Registration remains false. PyPTO scheduling, wrapper/subgraph wrapper,
  CSE propagation, strict choice filtering, atomic registry installation and
  subprocess compilation are explicitly unimplemented. The full plugin suite
  passes 128 tests; wheel and isolated producer-ABI/source audit pass.
- The framework plugin also freezes 41 source-hashed Qwen3.5 text-path compute
  obligations across generic, matmul, full-attention and GDN providers. This is
  a static inventory only; no site is marked implemented or covered.
- The framework plugin now also has a reviewed strict runtime-coverage evidence
  contract. It reconciles a fixed-revision closed activity trace against a
  digest-bound artifact registry, rejects vacuous/zero-time/misclassified or
  fallback evidence, latches strict failure, and owns report publication across
  processes. Its 68 tests and clean wheel build pass. No runtime collector or
  Qwen trace has been connected, so current runtime coverage remains unclaimed.
- Pinned-source reconnaissance selects the official SGLang HookRegistry plus
  PyTorch's experimental CUPTI monitor for the future collector. Its first
  eager-only trace must remain `closed_world=false`; graph replay and a proven
  complete drain/artifact-launch binding are separate gates.
- The cloned environment is not baseline-ready yet: its Triton distribution is
  still editable from `/home/zhaosiying/codebase/triton`. The runtime audit
  rejects this path. FlashInfer is corrected to the official 0.6.17 wheel and
  the unrelated external study package was removed.
- The local target is an NVIDIA GeForce RTX 5090 Laptop GPU, SM120, 24,463 MiB.
- CUDA Toolkit 13.3 is installed under `/usr/local/cuda-13.3`; the installed
  PyTorch is 2.13.0+cu130.
- The single-DSO compiler object boundary is committed in PyPTO
  `e463bce7849b2239d0457dcae78ccf41c65ffa55`. All 228 production native
  compile rows belong uniquely to `pypto_compiler_objects`; 12 binding rows
  belong to `pypto_core`; the one extra native row is the intentional C++ test
  compile. Native CTest passes 1/1.
- The post-fix editable suite passes 10,176 with 59 skips and exact JUnit
  `(10235, 0 failures, 0 errors, 59 skipped)`. The fresh wheel SHA-256 is
  `bd6d24c9857a409df9d48c604bd329d10808cde354803ee10765680d252f1da1`.
  It has 174 unique safe members and exactly one DSO.
- A real clean-venv install owns both imports and `pypto-ir-trace`; the
  installed DSO has only the five standard runtime `DT_NEEDED` entries. The
  wheel suite passes 10,178 with 57 skips and exact JUnit
  `(10235, 0 failures, 0 errors, 57 skipped)`; the independent symlink probe
  passes. Three independent final reviews report no remaining P0/P1.
- The exact PyTorch-pinned Triton source is now a clean official checkout under
  `upstream/triton`, but the environment still imports the inherited external
  editable distribution until the accepted hermetic workspace wheel is
  transactionally installed for baseline work.
- The exact-wheel transaction now has bounded unreviewed materialization,
  source-frozen archive/manifest review, exact producer RECORD identity,
  offline minimal-bwrap build, native/RECORD wheel audit, pip-free fresh probe,
  reference-only SM120 smoke and reversible replacement tooling. Unit tests are
  source/tooling evidence plus a reviewed dependency closure: all 10 archive
  SHA/byte pairs and manifest `29c073...` are source-locked. The exact wheel is
  now built/audited/probed but deliberately not installed.
- The control implementation is layered as `0c4cc34` (CPU-only isolation and
  materialization), `befe44c` (wheel/native audit and fresh probe), `640c35a`
  (reference-only smoke/finalization), and `c987811` (journaled replacement).
  Follow-up commits `ea39ac5`, `f678fc0`, `5c75ea4`, `fe903fa`, and `cca595c`
  bind LLVM seed bytes, strict Content-Length/strong-ETag Range resume, durable
  no-replace cache publication, stop-race closure and all ten reviewed locks.
  The isolated root suite now passes 198 tests plus 98 subtests. Every
  Python file, every runbook Bash fence and `git diff --check` also pass.
- All project-environment consumers now hold a shared lock; plan shares it and
  apply/recover/rollback require an exact inherited exclusive lock and direct
  child. The replacement uses a durable initializing journal, atomic no-replace
  backup publication, exact prefix-user audits, stdlib RECORD installation and
  idempotent recovery/rollback. No real replacement has run.
- Materialization/promotion runs `pypto-20260824T093330Z-135732-e907f1` and
  `pypto-20260824T100412Z-150285-bbfebd` pass. Independent reviews audited all
  6,503 archive members, 5,853 expanded files and 31 symlinks; seven NVIDIA
  archive hashes match official redistribution manifests. The reviewed cache
  is `caches/triton-build-deps/29c073...`; networkless tool-version probes and
  final require-reviewed verification pass. EV-0037 binds this closure.
- The source-reviewed TargetInfo candidate `9939b88` is integrated as PyPTO
  `042878dd6825bb65ed03f22db7b067fb96277623`. Fresh native CTest passes 2/2;
  the wheel has 177 safe members and one DSO; 31 targeted tests, 10,209 full
  tests with 57 unchanged skips, and the independent symlink case pass. EV-0034
  binds the three successful run IDs, Git tree, wheel/DSO, log and JUnit hashes.
- PyPTO `09e014ceac2d2cac2f667d182bd7de8f0d0bd259` now owns immutable
  CompileRequest v1. It value-copies NvidiaTargetInfo, exact toolchain and
  deterministic Cubin/verification policy; bounded canonical MessagePack and
  distinct byte/loader-input/device projections pass native 2/2 and Python
  62/62 CPU-only gates. EV-0036 binds the independent replay. No region,
  schedule, produced artifact or runtime state is present.
- PyPTO `9b3cf71b6ff2a535aabdd053684a60b47450fac0` now owns immutable
  KernelBuildSpec v1. It binds canonical region source/ABI, one generic
  semantic route, full pipeline revision, eight explicit resolved schedule
  categories, specialization/mutation identities, the parent CompileRequest
  byte identity and optional all-or-none operator catalog provenance. Native
  CTest passes 4/4; an explicit current-source DSO bootstrap runs three native
  contracts and the Python compiler suite 122/122. EV-0038 binds the source,
  DSO, run and recovered stale-editable evidence. No producer, Artifact, cache,
  runtime or CUDA behavior is present.
- The exact Triton reference wheel SHA is `1d58d830...6227a`. Complete
  source/dependency/producer/RECORD/native audit and pip-free fresh probe pass
  in runs `pypto-20260824T134836Z-217013-12ae6e` and
  `pypto-20260824T134906Z-217357-e68fd4`. All 18 ELF files are RPATH-free;
  seven stripped NVIDIA vendor files are bound by exact path/SHA and three
  plugins resolve only to wheel-internal libtriton. EV-0039 binds the recovery
  lineage and evidence. The wheel is deliberately not installed and is not a
  candidate compute dependency.
- PyPTO `5f755687674bd23a513e913cd6f2f20e8b6397ef` now embeds private
  TensorIR `233ab6ed...`, CUDA Tile `af241704...`, and LLVM `57109bef...` in
  the sole `pypto_core` product DSO. TensorIR and CUDA Tile Python/CAPI/CLI
  products are disabled; only build-time `cuda-tile-tblgen` remains.
- The final private compiler DSO SHA is `12b75a4e...c3def5`, has no
  RPATH/RUNPATH and only the five standard runtime `DT_NEEDED` entries. Its
  exact-source compiler suite passes 123/123 and its device-hidden build-info
  probe proves `sm_120a`, compiler-factory availability and every locked
  source/tool identity. EV-0040 binds the commits, DSO and four final runs.
- This compiler composition still returns TensorIR's process-oriented runtime
  kernel rather than a PyPTO persistent Artifact. Kernel bytes/metadata,
  current-stream execution, cache, operators, framework registration and Qwen
  work therefore remain explicitly unaccepted.
- The single-DSO runbook has been repaired and independently approved: it now
  performs a real venv install, audits all wheel DSOs and installed dependency
  closure, verifies every native/binding compile row and enforces the exact
  two-file commit boundary. Its recovery-audit lineage and final passing run
  IDs are recorded in EV-0033.
- A separate read-only recovery audit binds current `042878d` source/tree and
  clean upstreams to the retained wheel/log/JUnit hashes. Its first attempt
  failed on a copied tree-SHA typo and changed no product; the corrected run is
  recorded in EV-0034.
- TensorIR source audit proves its `IRuntimeKernel` is not a persistent
  artifact: complete argument/grid metadata lacks versioned serialization,
  runtime initialization is not synchronized, and support/workspace checks are
  TODO. Ambient overrides and wrong SM120 resource fallbacks also prevent a
  deterministic cache/runtime shortcut.
- Decision D-0009 now orders a data-only CompileRequest, per-region
  KernelBuildSpec, exact LLVM/tileiras producer identity, common
  bytes-plus-metadata Artifact, persistent cache and a process/device/CUcontext-
  bound prewarmed executable. Streams remain per-capture/launch values; workers
  never query CUDA or reuse module/context handles.
- Candidate and baseline framework runs now have separate, reviewed execution
  profiles. Baseline cannot import PyPTO/plugin sources or load general SGLang
  plugins; candidate framework launch cannot start until exact environment,
  selected NVIDIA runtime, entry points and process ownership all verify.
- Frozen baseline launch scripts exist for 0.8B and 9B, but neither is runtime
  evidence and the CPython 3.12 baseline environment is not built yet.
- Prior reconnaissance did not complete a TensorIR SM120 runtime launch. Static
  target support is not runtime acceptance.
- The user explicitly authorized non-benchmark CPU-only coexistence with a
  protected lane. The new policy is opt-in, retains NVIDIA/environment checks,
  uses a 24 GiB launch floor and 16 GiB owned-run pause floor, and never signals
  external PIDs. TargetInfo itself used green windows and no waiver. A separate
  live control run `pypto-20260824T075816Z-91897-64e1ea` passed the full root
  suite beside seven protected heavy processes with waiver=true, no protected
  NVIDIA compute PID, no pause/abort and no external signal. EV-0035 binds the
  log/preflight/process hashes; this is not GPU or performance evidence.

## Resume action

The goal is active, not complete. Freeze Triton as accepted reference-only
infrastructure. Implement PyPTO's private TensorIR/CUDA Tile/exact-LLVM static
composition and deterministic compiled-artifact seam next. Defer the separate
exclusive Triton replacement/reference-smoke gate until the unmodified SGLang
baseline needs it; do not place CUDA modules, contexts, streams or launch
handles in persistent Artifact metadata.
Use explicit CPU-only coexistence only for non-benchmark build/test work; GPU
smoke/benchmarks require exclusive `gpu-benchmark` preflight. Never inherit
TensorIR's SM100 defaults.
