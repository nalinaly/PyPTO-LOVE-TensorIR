# CHECKPOINT

**Checkpoint:** `CP-0043`

**Status:** R0 remains open. P2 accepts finalized minimal real-SM120
`NvidiaExecutable` correctness v1: exact PyPTO/TensorIR/CUDA Tile Artifacts load,
prewarm, execute on the caller's non-default stream, synchronize and unload on
the RTX 5090 with no fallback. It also accepts compile-free deterministic
static `tensor.add` HIR-to-TensorIR emission at PyPTO `07ab9ea`, and now the
standalone bounded schedule identity at `fa85e5a`, and CP-0041 accepts private
compile-free frontend specialization/ABI projections and final BuildSpec
construction. CP-0042 now accepts the public one-producer joined BuildSpec/
Artifact transaction at PyPTO `642ff5b`. CP-0043 accepts the separately
versioned frontend FP32/BF16 correctness-smoke controls and CPU-only exact-DSO
replay boundary at root `47a0c15`. Frontend GPU correctness, operator-family,
framework, model, CUDA Graph, coverage and performance milestones remain open.

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
  prefill-to-decode cache continuity and inactive-output zeroing. Its CPU Torch
  cross-check is accepted; CUDA and SGLang device comparison remain open.
- GDN now has a reviewed unified core ABI v1 for decode and prefill/extend,
  freezing packed layout, no-bias causal conv, FP32 gate/delta semantics,
  paired BF16-conv/FP32-recurrent state, pitched envelope views and canonical
  metadata/output tails. State preparation/copy and CUDA remain open.
- GDN now also has a reviewed standard-library paired-state numerical reference.
  One-shot prefill, checkpoint-segmented prefill and token decode match exactly
  in output, BF16 conv state and FP32 recurrent state. Its independent CPU Torch
  cross-check is accepted; CUDA and pinned-SGLang device comparison remain open.
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
- TensorIR `2677d1a99ae9e4c6627872f01a26a62bf6e832c8` now emits a value-owned
  compiled result before legacy runtime reconstruction. It contains validated
  bytes, actual kind/target, entry names, complete argument/packing/grid and
  workspace metadata, strict/fallback policy, exact assembler identity and no
  CUDA/process/runtime fields.
- The native gate covers four-CTA static grid, dynamic Flat/runtime-grid ABI,
  exact argument counts and pack bounds, full TileIR and bounded CUDA ELF
  structures, OSABI/ABI-specific SM decoding, two identical SM120 Cubins,
  hostile environment rejection, wrong producer SHA and explicit fallback.
- PyPTO `4789ae0f...` places all TensorIR types behind a no-RTTI/no-exception
  bridge, validates clean exact gitlinks, isolates ON/OFF native outputs and
  exports only `PyInit_pypto_core`. The final one DSO SHA is
  `0b69023a...74a4b`; 5/5 native and 123/123 exact-DSO Python tests pass.
- No PyPTO strict producer bridge consumes CompileRequest/KernelBuildSpec yet,
  and no canonical persistent Artifact exists. EV-0041 binds the narrow claim
  and all final runs.
- PyPTO `6ce17761cb26b6593ce8a6f0f8a82cb0cf251cc9` extracts the duplicate
  CompileRequest/KernelBuildSpec codec into one compiler-private implementation.
  A streaming preflight bounds total decoded objects, aggregate container
  items, depth, non-binary encoding, BIN size/count and malformed parser
  exceptions before MessagePack allocates its object tree. Canonical wire bytes
  and error compatibility remain covered by exact-DSO replay.
- Backend-ON and a fresh isolated backend-OFF build both pass the three native
  codec/request/spec tests. The exact ON DSO Python compiler suite passes
  123/123. The OFF DSO reports `compiled=false`, has no TensorIR/CUDA Tile/MLIR/
  LLVM targets or link inputs, no RPATH/RUNPATH, two permitted exports and only
  five standard runtime dependencies. EV-0042 binds those codec-only claims;
  Artifact v1 was still open at CP-0029. The root control regression passes
  198 tests plus 98 subtests.
- TensorIR `1c701ec6f7e7c547f6af02862603981f64e01091` adds bounded runtime-free
  CUDA 13.3 Cubin entry and flattened parameter-ABI validation; follow-up
  `b25081afbeb53c9a882c68b440d06baa9e0f6b31` binds executable/constant
  sections to valid PT_LOAD file/VA mappings. It validates unique
  `STO_CUDA_ENTRY`, canonical text/symbol links, KPARAM ordinals/widths,
  PARAM_CBANK base/extent/section symbol and constant-bank coverage without a
  CUDA call, device query, module load or filesystem access.
- PyPTO `9894f5babdca17d27de7b89540e28fc5c3b3e199` adds immutable canonical
  Artifact v1; `4a82f2e40ce518d16fcbeec647061649564c42af` pins the final loader-safe
  validator. The Artifact binds clean PyPTO/TensorIR/CUDA Tile/LLVM/tileiras
  provenance, pipeline blob `a5398054...`, strict schedule options, grid and
  uniform ABI, entry/flattened argument ABI, complete bytes SHA-256, cache key
  and loader projections. Unsupported, noncanonical, oversized, fallback or
  mismatched inputs fail closed.
- The final clean backend-ON build embeds exactly PyPTO `4a82f2e...`, TensorIR
  `b25081a...`, CUDA Tile `af241704...` and LLVM `57109bef...`; the DSO SHA-256
  is `2b65764e...cd06`. Native CTest passes 8/8, including real pinned
  `tileiras` static, dynamic 12-argument and scalar Cubins. The exact source DSO
  Python compiler suite passes 128/128.
- A fresh backend-OFF product passes 6/6, reports strict Artifact creation as
  fail-closed and contains no private compiler dynamic dependency/export. Both
  ON/OFF DSOs have no RPATH/RUNPATH and only five standard runtime dependencies.
  Configure- and build-time guards reject dirty/stale PyPTO or vendor source
  identities. EV-0043 binds all final runs and independent GO reviews.
- CP-0030 does not accept the strict producer bridge. No entrypoint yet consumes
  canonical source with CompileRequest/KernelBuildSpec and returns this
  Artifact; cache publication, subprocess compilation, CUDA module/context/
  current-stream execution, operator/framework/model correctness and
  performance remain open.
- PyPTO `f3bcaaccdfb169080628d56e461653b0ba3e0ad5` now exposes the
  bytes-only strict producer entrypoint. It verifies canonical source against
  KernelBuildSpec, maps the accepted schedule/target/toolchain contract through
  a private standard-only DTO, invokes concrete TensorIR compilation and builds
  the already accepted Artifact v1 without exposing a TensorIR public product.
- TensorIR `1dcb38c20e53d07c97d3781cae538e33901bae30` executes a private
  byte-verified copy of pinned `tileiras` with exact toolkit environment,
  bounded diagnostics/time/memory/files and fail-closed descriptor-relative
  scratch monitoring. Closed host stdio and post-fork compiler reuse are
  explicit regression cases.
- Pipeline blob `46610e0415598d010981e4bd07d0660c592401ac` binds the exact
  process/resource policy. A fresh backend-ON product passes native 8/8 and the
  exact product DSO compiler suite 132 passed/1 skipped. Its SHA-256 is
  `3438f76a65f8021987b187e49ffaba25355a2f8e7920cf251b2c422fea50a134`.
- A fresh backend-OFF product passes native 6/6 and the exact product DSO suite
  128 passed/5 skipped. Its SHA-256 is
  `b3524cb94e2c845e401dac6d5fac123472d66e8869cb00b4e5feb0e98643da3e`.
  Both products are RPATH-free, depend only on five standard runtimes, expose
  only the version node/Python init entry and leak no private compiler dynamic
  symbols.
- Exact clean provenance covers PyPTO plus all six compiled direct submodules:
  TensorIR, CUDA Tile, LLVM, msgpack-c, libbacktrace and runtime. Wrong
  revisions and tracked/untracked/ignored synthetic source changes are
  rejected. The root control suite passes 198 tests plus 98 subtests. EV-0044
  binds the exact runs, logs, products and final GO review.
- CP-0031 does not accept ArtifactCache, CUDA module/function/context/stream
  loading or launch, workspace/runtime allocation, frontend-HIR lowering,
  generic codegen, operators, frameworks, Qwen correctness/coverage or
  performance.
- PyPTO `c087170444270cbe00f83e1fbf127ddcf3e33926` now exposes the
  compiler-owned ArtifactCache v1. `lookup(request, build_spec)` returns only a
  fully revalidated immutable Artifact or an exact `ENOENT` miss;
  `publish(artifact, request, build_spec)` stages, syncs, validates and
  atomically publishes without replacement, returning `Published` or
  `AlreadyPresent`.
- The fixed trusted-local namespace is
  `pypto-nvidia-artifacts/v1/sha256/<prefix>/<digest>.artifact` beneath a
  caller-created absolute canonical non-root owner directory with exact mode
  `0700`. Descriptor-relative no-follow traversal rejects unsafe writable
  ancestors, symlinks and wrong file ownership/type/mode/link count.
- Lookup bounds file size before allocation, detects read/metadata races and
  fully deserializes canonical Artifact v1 before checking exact request,
  build-spec, producer and path-key identity. Corruption is an error; ordinary
  lookup never deletes, repairs, quarantines or recompiles.
- Publication uses a same-shard exclusive temporary file, full write/fsync/
  readback, exact mode `0400`, `renameat2(RENAME_NOREPLACE)` and shard fsync.
  An `EEXIST` winner must independently pass full validation and canonical byte
  equality. Created directories are permission-fixed and synced before their
  parent directory entry is acknowledged.
- Cache handles are non-copyable/non-movable and creator-PID-bound; an inherited
  post-fork handle fails closed and each process opens its own handle. V1 has no
  CUDA state, compile-on-miss, interprocess compile lock, eviction, repair,
  remote cache, signature or quota. `ENOSPC` remains an explicit availability
  error and the owner-private boundary is not origin authentication.
- The fresh backend-ON product passes native 8/8 and exact-DSO Python 137
  passed/1 skipped. Its DSO SHA-256 is
  `d6c9729dff380335b9a9f0e3b581dc9f026b888b9cbf5b869a89888fe9df0c7b`.
  The fresh backend-OFF product passes native 6/6 and exact-DSO Python 130
  passed/8 skipped; its SHA-256 is
  `7edab542960eaa12d0354a169130724c558f8135c1fee4dd4302af5ff614c7e1`.
  Both are RPATH-free, depend on only five standard runtimes and leak no private
  compiler/CUDA dynamic symbol.
- Exact current provenance covers clean PyPTO plus TensorIR, CUDA Tile, LLVM,
  msgpack-c, libbacktrace and runtime, rejecting every wrong revision and
  representative tracked/untracked/ignored drift. Ruff, source/stub Pyright,
  1,133 header checks, 168 EN/ZH doc pairs/navigation and 1,324 English-only
  checks pass. The isolated root control suite passes 198 tests plus 98
  subtests with CUDA hidden. Independent API, security and final audits report
  no P0/P1. EV-0045 binds the exact runs, products, logs and acceptance
  boundary.
- CP-0032 accepts persistent Artifact storage only. It does not accept CUDA
  device/context/module/function state, support/resource legality, workspace
  allocation, current-stream launch, CUDA Graph, frontend-HIR lowering,
  operators, TorchInductor, SGLang, Qwen correctness/coverage, profiling or
  performance.
- PyPTO `2842a1c5433cb3cfe9e4fbc7664ebe0ad8a4b129` now owns a distinct
  internal NVIDIA runtime object target and public `pypto.runtime.nvidia`
  lifecycle API. Construction from Artifact plus CompileRequest is CUDA-free;
  KernelBuildSpec is already transitively bound into Artifact.
- The executable binds PID/device/regular CUcontext/context ID, uses a
  once-only failure-latched state machine and creator-PID custom deletion, and
  makes explicit unload terminal. Its private backend-ON boundary lazily opens
  only `libcuda.so.1`, resolves typed CUDA 13.3 entrypoints through
  `cuGetProcAddress_v2`, and uses context-bound `cuModule*` APIs without storing
  a stream or reusing TensorIR runtime.
- Prewarm validates explicit active Runtime/Driver API versions, current
  non-green context, normalized UUID/PCI identity, SM120 resources, forced
  function loading, exact parameter widths and block/register/shared-memory/
  cluster/occupancy legality. Partial modules roll back and failures latch.
- Prepared packets allocate and validate shape/stride/scalar slots plus checked
  runtime grid before launch. A successful fake-driver Launch records zero host
  allocations and rejects special default stream handles 0/1/2, foreign or
  green stream contexts. Packet and CUDA Graph leases block module unload;
  actual graph capture is not claimed.
- Fresh backend-ON builds 1,902 edges and passes native 9/9 plus exact-DSO
  Python 142 passed/1 skipped. DSO SHA-256 is
  `ef60b6a9749036eaff27786c5838b24fdd282b97b345315855c2994e2b3e4727`.
  Fresh backend-OFF builds 312 edges and passes native 7/7 plus Python 134
  passed/9 skipped; its SHA-256 is
  `9ffd22915e5c2c694660f06f3db0455ee9df560e828fd6b8c13c8c3f478cee7a`.
- Both products retain only five standard dependencies, no RPATH/RUNPATH and
  only the version/Python-init definitions. ON has exactly one driver source;
  OFF has no driver row and directly tests its fail-closed stub. Neither has
  exported or undefined CUDA/TensorIR/executable symbols.
- Seven-source provenance and all negative revisions/drift pass. Ruff,
  Pyright, 1,144 headers, 169 EN/ZH docs/nav, 1,336 English-only checks and
  root 198 tests plus 98 subtests pass. Three independent reviews report no
  P0/P1. EV-0046 binds the exact products/runs and diagnostic lineage.
- The exclusive GPU gate returned 75 for the active protected ZCode/gem5/
  SGLang lane, with zero NVIDIA compute PIDs. No waiver or external signal was
  used. CP-0033 therefore does not accept real libcuda resolution, Cubin load,
  current-stream execution, numerical correctness, CUDA Graph, frontend
  lowering, operators, frameworks, Qwen coverage or performance.
- PyPTO `6361f110660a77f9a8dc542265575d8f7260b343` now owns the
  public immutable `NvidiaRuntimeObservation`. Parent code supplies a Driver
  release provenance string and an audited expected CUDA Runtime library path;
  PyPTO returns the complete live `NvidiaTargetInfo`, numeric Driver/Runtime API
  versions, authenticated canonical provider path and diagnostic regular-
  context address/ID without retaining a CUDA handle.
- Runtime version discovery uses only `dlsym(RTLD_DEFAULT)` followed by
  `dladdr`; actual and expected paths must canonicalize to the same regular
  file. The observation seam never opens libcudart. Environment ownership of
  the expected file remains the caller's proof obligation.
- Observation reuses the private Driver's pre-mutex PID latch. Distinct-value
  tests cover all 21 TargetTraits fields, identity/revision/dtype propagation,
  provider mismatch/status/version rejection and parent-observation to
  fork-child rejection. Backend-OFF public observation fails before dynamic
  loading. Fork children must use spawn/exec because the latch cannot detect
  arbitrary CUDA initialization performed by other libraries.
- Fresh ON/OFF builds and exact-DSO gates pass CTest 9/9 plus Python 142/2,
  and CTest 7/7 plus Python 135/9 respectively. Both remain one RPATH-free DSO
  with only five standard dependencies/two definitions and no CUDA/TensorIR/
  executable/observation dynamic-symbol leakage. EV-0047 binds the exact final
  products, run metadata, JUnit, compile rows and static gate.
- CP-0034 accepts only the CPU/value observation and product boundary. No
  production dlsym/dladdr, real context/device query, Cubin load, CUDA launch,
  numerical result, graph, framework, model coverage or performance is claimed.
- Root `394b75adbc7babe1000d93938e6fa84493a4277d` implements the
  correctness-only SM120 NvidiaExecutable smoke. Its exact child is
  `-I -B -S`; controller, finalizer and both GPU preflights require
  `-E -B -S`. Ambient Python paths, plugins and fallbacks are absent.
- Root `2b53f0a6cdeffa89b38ad75515b9ea1d1019748a` adds the canonical
  seven-blob control manifest with SHA-256
  `c609e97a2f3e379e332137916d041d14931c0e415cd2f2e769c82eab1650aa09`.
  It binds implementation commit/tree, live/committed bytes and modes, requires
  a clean descendant root, and rejects post-implementation control drift.
- The smoke has distinct fully specified static FP32, dynamic-stride FP32 and
  FP16-plus-FP32-scalar TensorIR cases. CPU-only compilation locks Cubin
  sizes/SHA, and exact-DSO replay reopens CompileRequest, all BuildSpecs and
  Artifacts, then joins complete TargetInfo, ABI, Cubin and execution identity.
- Parent admission, post-Popen pre-release gate, child pre-CUDA gate, periodic
  watchdog and post-exit audit reject protected NVIDIA mappings, unreadable
  maps and foreign compute PIDs. Only PID/start-tick descendants in the owned
  PGID are signalable. The protected CPU lane may coexist only through the
  explicit correctness-smoke policy; `gpu-benchmark` remains exclusive.
- Post-manifest root run `pypto-20260825T050233Z-793333-c8f8cb` passes
  224 tests plus 106 subtests. Exact-DSO semantic replay/TargetInfo run
  `pypto-20260825T045248Z-788969-a1b6bc` returns zero and leaves no ignored
  PyPTO package shadow. No CUDA context, module, stream launch, device result or
  performance measurement is accepted by CP-0035.
- First real run `pypto-20260825T052038Z-800777-8e8e83` passed its admission,
  release and child isolation barriers with empty external compute-PID sets and
  empty protected compute-PID/runtime-mapping/unreadable-map sets. It observed
  the real PyTorch Runtime, RTX 5090 context/device and
  loaded the first static Cubin/function, then failed in repetition zero during
  `prewarm` with a parameter-ABI mismatch. The exact failed predicate was not
  persisted; the reversed-offset diagnosis is source/Cubin inference. It did
  not reach `prepare_launch`,
  `Launch`, provisional publication or numerical comparison. The owned PGID
  exited with no survivor and no external signal.
- PyPTO `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7` fixes the generic defects.
  Cubin `KPARAM` ordinals define signature order. Driver-enumerated parameter
  widths are compared as a multiset and offsets only as bounded, non-overlapping
  ranges, independent of enumeration order; launch pointers remain in signature
  order. Dynamic sizes and strides are bounded and packed as four-byte `int32`
  values in stable zeroed slots. Diagnostics are bounded and invalid later
  dense dimensions are rejected before unsigned stride reconstruction.
- CUDA Tile's documented host block is logical `[1,1,1]`; physical worker
  warps and Cubin `EIATTR_REQNTID=128,1,1` are compiler-selected physical
  metadata whose exact mapping remains an inference. They are not host block
  dimensions. The attempted conventional-CUDA reinterpretation was fully reverted and TensorIR
  remains clean at `1dcb38c`.
- Fresh ON product SHA `15675c47...018c` passes CTest 9/9 and exact-DSO Python
  142/2; fresh OFF SHA `32c2dea0...4109` passes CTest 7/7 and Python 135/9.
  Both remain RPATH-free with five standard dependencies and two definitions;
  the complete DSO/compile-row audit passes.
- CPU-only run `pypto-20260825T070649Z-890066-2a1ae2` preserves empty device
  visibility and wrapper ownership markers, observes Torch device count zero
  with CUDA uninitialized, and recompiles all three expected Cubin size/SHA and
  ABI pairs through the exact final DSO. Earlier `062710`, `062751` and `070611`
  runs remain diagnostics and are not acceptance evidence.
- Root control commit `c71f32bd415a973a2a7756ecc9b1ae59f30df219` and
  manifest-only `3de4cf702662cbaf948c6429acf269fee16a491e` bind the final product
  through immutable v3 manifest SHA
  `978e873788eb7f3aaeba6473a9b7f8a1bcd827fe201d89cb781927f538c9b6e3`.
  Versions 1 and 2 remain unchanged. Clean post-v3 run
  `pypto-20260825T071601Z-892819-67acee` passes 224 tests plus 106 subtests.
- V3 run `pypto-20260825T073624Z-900485-7df250` exited zero after recording six
  real non-default-stream module lifetimes, equal logical/Torch results,
  unchanged inputs/padding and explicit unloads. Its provisional SHA is
  `64c0906b...d34cfe`. The required CPU finalizer then failed before publication
  because it expected handwritten `[BF16,FP32]` while canonical TargetInfo and
  replay order is `[FP32,BF16]`. The provisional is retained diagnostic evidence,
  not accepted GPU correctness.
- Root A4 `5564008fddeaaf0a9861ee5c38c895558f577600` fixes only that
  finalizer/control defect with a shared exact ordered contract and malformed
  evidence matrix. Manifest-only B4 `7639d820f4d74972b493c01adc69c92087eefdea`
  publishes immutable v4 manifest SHA
  `a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf`.
  PyPTO, the DSO, Artifact and Cubin bytes are unchanged.
- Clean post-v4 run `pypto-20260825T074420Z-903996-c8d50d` passes 225 tests
  plus 113 subtests. A v4 finalizer cannot promote the v3 provisional because
  exact contract/control identity is mandatory; at CP-0037 a fresh v4 child was
  still required, and the next bullet records its CP-0038 completion.
- Clean v4 run `pypto-20260825T080254Z-910620-c669d9` and its matching no-site
  finalizer close that gate. Six real current-stream executions match
  references and unload cleanly; independent replay joins all compiler inputs,
  TargetInfo, Artifacts and Cubins without fallback. Final report SHA is
  `727362d7879d58cbee07b11050b17ad149274e8087b0d1872b8f186a66a272a9`.
  This accepts minimal `NvidiaExecutable` runtime correctness only.
- PyPTO `07ab9ea1feb5f5cc5557c7b7c67e7ad33d15974e` adds an internal,
  vendor-independent HIR-to-TensorIR emitter for the exact static contiguous
  FP32/BF16 `tensor.add` normal form. Compiler-owned names, `to_chars`, private
  metadata construction and exhaustive fail-closed validation make source bytes
  deterministic. Fresh OFF and ON builds both pass the unconditional native
  test 1/1. No TensorIR parser/compiler or device is invoked by this gate.
- PyPTO `fa85e5a2c917af78ef94576073eb91c0891c4384` adds standalone
  `pypto.canonical_schedule.v1` serialization, identity, bounded fail-closed
  decoding and the single `pypto.compiler` binding. Nested KernelBuildSpec
  bytes and its representative digest remain unchanged. The retained CP-0039
  build directories were reconfigured/rebuilt at exact head; backend-ON/OFF
  native suites pass 2/2 and exact-DSO Python suites pass 98/98. This is a
  compiler-preparation prerequisite only; it does not construct an Artifact.
- PyPTO `c4cf755b7ef2998cdfb7499a76b80b280bafdf2d` adds the private
  compile-free structured frontend identity boundary. It derives five
  versioned projections plus the raw source digest, owns request/schedule and
  metadata, validates one producer-shaped callable ABI, and produces the final
  KernelBuildSpec without placeholder hashes. Producer-reachable FP32/BF16
  goldens, strict ABI drift, cache-boundary and alias/dtype limitations are
  frozen. Fresh backend-ON native 6/6 and exact-DSO Python 180/2 pass;
  backend-OFF native 4/4 and Python 173/9 pass. No frontend producer call,
  Artifact, public binding or GPU execution is accepted by this gate.
- PyPTO `642ff5bd79ee96b9e5a279a2bc945ad7a78362b7` adds the public immutable
  structured compile facade. It prepares frontend HIR, hard-calls the concrete
  private producer once, seals exact source/request/options identity, finalizes
  the producer ABI into the BuildSpec, constructs the Artifact from that same
  move-only result, validates the complete identity join and returns only after
  success. No producer callback, placeholder identity, provisional Artifact,
  compile-twice discovery or cache publication is exposed. Backend-ON native
  7/7 and exact-DSO Python 182/2 pass; backend-OFF native 5/5, functional Python
  7/7 and full Python 175/9 pass. The retained ON/OFF build-directory names are
  stale, so EV-0055 binds exact paths, DSO hashes and the ON embedded revision.
  No CUDA launch through frontend HIR is accepted by this checkpoint.
- Root implementation `1d1fce4d63320beb3f29a265dd126891f37fb559` plus
  manifest-only `47a0c15de510fbdea1eb029ff4e5f0cc9cdc5b77` adds the
  separate frontend FP32/BF16 SM120 smoke family. Manifest SHA
  `f16c4fba...d8eed` binds five new controls and the three exact unchanged v4
  isolation primitives, while also pinning the complete v4 parent manifest.
  The first review returned NO-GO and prevented publication of a real
  ArtifactTarget API defect, weakened finalizer checks and insufficient tests.
  The amended implementation passes final GO with P0/P1/P2 zero, a 21-test/
  13-subtest family suite, exact-DSO runner and replay-child CPU gates, and a
  clean 246-test/126-subtest root suite. No GPU launch is accepted here.
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
infrastructure. CP-0038 closes the minimal real-SM120 compiler/runtime launch
gate. Preserve its v4 controls and final report; do not rerun it merely to
change evidence. CP-0039 through CP-0041 close compile-free HIR emission,
standalone schedule identity and final frontend BuildSpec identity. CP-0042
closes the one-producer joined BuildSpec/Artifact facade. CP-0043 closes the
separate fixed frontend smoke control and CPU-finalizer boundary. Next run and
finalize its HIR-authored FP32 and BF16 vector add on real SM120. Fused
pointwise, row reduction and structured matmul follow.
Defer the separate exclusive Triton replacement/reference-smoke gate until the
unmodified SGLang baseline needs it.
Use explicit CPU-only coexistence only for non-GPU build/test work. The single
fixed correctness smoke may use its separately audited zero-NVIDIA protected-
CPU policy; all GPU performance benchmarks remain exclusive `gpu-benchmark`
runs. Never inherit TensorIR's SM100 defaults.
