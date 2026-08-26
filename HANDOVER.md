# HANDOVER

This workspace is implementing the approved plan with three independent
projects: the PyPTO compiler, a FlashInfer-like standalone operator library, and
framework compatibility plugins. PyTorch and SGLang themselves are immutable
upstreams.

## Operational boundaries

- Use only paths beneath this workspace for environments, sources, builds,
  caches, logs, sockets, ports, and run state.
- `/home/zhaosiying/amdgpu-sim` may be read only to copy the two frozen model
  snapshots and verify their provenance. Do not run its scripts or import its
  environments.
- zcode may run SGLang, ROCm runtime services, and many gem5 instances. Never
  kill by name or signal a PID not proven by run metadata to belong to this
  workspace. If memory is tight, stop or pause only this project's work.
- The runtime must be NVIDIA-only. Reject HIP/ROCr/GemSim DSOs and environment
  leakage before every build or execution gate.
- Do not reboot, run `wsl --shutdown`, or change driver/kernel configuration
  without explicit user approval.

## Source decisions

- Authorized PyPTO baseline: public upstream commit recorded in
  `VERSIONS.lock`.
- Operators and framework adapters are separate projects; SGLang-specific
  objects must never enter the kernel library.
- TorchInductor and SGLang integrations use official extension hooks plus
  exact-SHA compatibility guards. An API mismatch fails closed.

## Evidence discipline

Every claim points to an `EV-*` record. Every completed transaction updates
`CHECKPOINT.md`, `PLAN.md`, `TODO.md`, and `WORKSPACE.lock`, then commits all
changed project repositories before the root checkpoint commit.

Run root control tests as `python -m pytest -q tests`. Never run unqualified
pytest from the workspace root because it recursively collects the clean full
upstream checkouts and their optional/manual suites.

## Current resume point

- CP-0048/EV-0061 advances the primary PyPTO checkout to clean `62eb882`, tree
  `04d3bca3`, and accepts RowReductionV3 host compiler/Cubin production. Fresh
  OFF/ON DSO SHAs are `95fc6579...93451` and `e1213cf3...e4220`; native tests
  pass 11/11 and 13/13, exact-product Python passes 1/1 on both, and neither DSO
  has RPATH/RUNPATH. Four rank-1/2/3 FP32/BF16 sum/max fixtures produce exact
  nonfallback SM120 Cubins; report SHA is `d06765be...38abb`, and independent
  CUDA-hidden recompilation matches every field. No reduction GPU launch or
  numerical claim exists yet.
- CP-0047/EV-0060 accepts the finalized policy-2 real-SM120 fused-pointwise
  result. Additive control commits end at `6cf5f958`, manifest-only commit
  `f6a064b` binds six v2 controls with SHA `d3b16079...d467036`, and GPU run
  `pypto-20260826T073309Z-1451510-e48ced` publishes immutable mode-`0444`
  report SHA `d4ffafc0...2eeedf0`. All nine cases/two repetitions pass;
  candidate versus Torch is zero ULP and the independent CPU maximum is one
  ULP. Eighteen packets release and executables unload, no fallback/provider or
  external/protected NVIDIA activity appears, and the no-replace retry fails as
  required. This is fixed-fixture correctness only, not performance or general
  FusedPointwiseV2 correctness.
- CP-0045/EV-0058 accepts the fused-pointwise compiler candidate at PyPTO commit
  `b83fcd3ddc497d585bcc45883eede179aff7d4d2`, tree
  `49eda98f3ed8d72bfd14d5a5900cdc0e71ca699d`, on
  `feature/fused-pointwise-v2`. It changes exactly ten tracked source/test/doc
  files, preserves all six exact clean gitlinks and the legacy add source,
  projection and Cubin anchors, and has independent source-review verdict GO
  with P0/P1/P2 = 0. It does not accept V2 GPU numerical correctness.
- Fresh clean-commit backend-OFF configure run
  `pypto-20260826T001612Z-1262389-4b0314` passed. Initial build run
  `pypto-20260826T001705Z-1262860-cb4c7a` paused only its verified owned PGID
  at the 16 GiB floor and later exited through the expected owned 3600-second
  timeout. Continuation `pypto-20260826T012525Z-1297224-e14260` used the
  user-authorized approximate 21.5 GiB (`22544384` KiB) launch floor and
  completed with no pause/abort. Exact DSO SHA is `eb4225cc...7ab80`.
  Native run `pypto-20260826T013235Z-1301725-baa2a2` passes 3/3, JUnit SHA
  `4d805ed4...05195`; exact-product serial Python run
  `pypto-20260826T013335Z-1302159-5819e7` passes 1/1 with exact source/core
  origins, JUnit SHA `958b4f7a...c4dfc`. The complete OFF build is retained at
  `builds/pypto-fused-pointwise-v2-off-b83fcd3-final`.
- An in-source ON configure passed, but its build correctly failed the strict
  clean-source guard and is retained as a diagnostic. Fresh external configure
  `pypto-20260826T014519Z-1311481-aca1c7`, build
  `pypto-20260826T014637Z-1313235-7ee7ee`, native 3/3
  `pypto-20260826T020956Z-1338874-b4a5db` and exact-product Python 1/1
  `pypto-20260826T021126Z-1339184-d74bfa` pass. ON DSO SHA is
  `0e8f33c2...facbe`; JUnits are `b3a57b2c...81140` and
  `44a155a6...9431d`. The source worktree remained literally clean.
- Read-only reduction reconnaissance selected a separate private
  `RowReductionV3` family rather than weakening `FusedPointwiseV2`. The first
  safe contract is dense static rank-1-through-32 reduce-last/keep-dim
  `row_sum` and `row_max`, one input pointer plus one result pointer, one
  flattened outer-row schedule tile and grid `ceil(rows/T)`. Rank 1 has
  `rows=1`; higher ranks use the checked product of every non-reduced extent.
  Direct TensorIR BF16 reduction accumulates in BF16;
  the PyPTO producer must instead emit BF16-to-FP32 `convert`, FP32 `reduce`,
  then FP32-to-BF16 `convert`. Do not accept the arbitrary-rank grid law before
  direct rank-1, rank-2 and dense row-major rank-3 normalized-layout producer
  fixtures pass. Source candidate `1daa7e57893eeb821752ca0fcf07daec4d46080e`
  plus follow-up `17b2b3c655d97076ac6d968ff2e45969da5161a2`
  are committed cleanly on `feature/row-reduction-v3`. The follow-up derives
  the actual contraction tile, bounds the full CUDA Tile element count and i32
  loop trip count, adds N=128/N=256 producer fixtures and completes exact
  source/projection goldens. Independent static re-review returns GO with
  P0/P1/P2 = 0. Commit `62eb882` later fixes immutable negative-fixture
  construction; CP-0048 accepts clean OFF/ON products and four Cubin records.
  Runtime, numerical correctness and performance remain open.
- Read-only structured-matmul reconnaissance selected a later private
  `StructuredMatmulV4` contract: static dense BF16 rank-2 or equal-batch rank-3
  `tensor.matmul`, BF16 physical result ABI, TensorIR BF16-by-BF16-to-FP32
  matmul and an explicit nearest-even FP32-to-BF16 result conversion. TensorIR
  has no transpose flags, so accepted `a_trans`/`b_trans` forms must emit
  explicit rank-aware transpose views without changing physical argument
  shapes or strides. Unit output modes disappear during normalization: decode
  `M=1` uses a one-dimensional N tile, not a synthetic M-by-N schedule. Matmul
  static grid metadata is output-driven and must select descriptor index 2.
  Dynamic shapes, rank mixing, batch broadcast, direct BF16 TensorIR result,
  `c_matrix_nz` and accumulator forms remain fail-closed until later contracts.
  Source commits `6ee412a751fe684c5977828f2d526e9c28d3e787` and
  `d7551176ded1db74c3f185d443f1397a83029bb0` now implement that V4 boundary on
  `feature/structured-matmul-v4`; both independent source reviews are GO with
  P0/P1/P2 zero. The head adds conceptual `B*M*N*K` overflow protection,
  exact rank-2/rank-3 transpose/projection goldens and representative producer
  fixtures. It remains unbuilt with no Cubin, runtime, numerical or performance
  claim.
- Read `state/checkpoints/CP-0048.md` and evidence `EV-0005` through `EV-0061`.
- Root `5564008` plus manifest-only `7639d82` owns the current v4
  correctness-only SM120 smoke. The manifest SHA is `a079c4d2...98bf` and
  binds seven exact control blobs to the Layer-A commit/tree. Controller and
  finalizer require `-E -B -S`; the child and semantic replay use
  `-I -B -S`. V4 run `pypto-20260825T080254Z-910620-c669d9` and its matching
  finalizer are accepted by CP-0038. Final report SHA is
  `727362d7...272a9`; it joins six real non-default-stream lifetimes, references,
  sidecars, compiler inputs, TargetInfo, Artifacts and Cubins with no fallback.
  V3 run `073624` remains an unfinalized diagnostic and is never reused.
- `projects/pypto` is clean at `b83fcd3...`. Single-DSO, immutable SM120
  TargetInfo, Artifact v1, strict canonical-source production, ArtifactCache
  v1, the CPU/fake-driver NvidiaExecutable v1 and the parent runtime-observation
  value are accepted. The observation queries every live TargetInfo field via
  the private Driver, authenticates an already-loaded Runtime symbol provider
  against a caller-audited canonical path, reuses the pre-mutex PID latch and
  retains no handle. The
  executable has its own internal runtime object target, process/device/regular
  CUcontext lifetime, typed lazy Driver resolver, forced-function/resource ABI
  validation, allocation-free prepared launch packets and graph/module leases.
  TensorIR `1dcb38c...` remains private and owns the bounded assembler boundary.
  The accepted CP-0038 runtime DSO remains the earlier exact PyPTO `206447c`
  product; later exact-head DSOs are contract-test evidence, not a replacement
  accepted runtime smoke product. Those accepted runtime ON/OFF
  products remain RPATH-free with five standard dependencies/two
  definitions. ON SHA `15675c47...018c` passes native 9/9 plus exact-DSO
  Python 142/2; OFF SHA `32c2dea0...4109` passes native 7/7 plus Python
  135/9. EV-0049 binds their exact hashes, revisions, compile rows, JUnit and
  product audit. EV-0051 accepts only the minimal real-SM120 launch/numerical
  runtime gate; frontend and operator-family execution remain open.
- CP-0039/EV-0052 accepts the internal compile-free HIR-to-TensorIR emitter at
  `07ab9ea`: one exact static contiguous FP32/BF16 `tensor.add` form emits
  deterministic canonical source and `Input0/Input1/Result0` metadata. Clean
  ON/OFF native tests pass 1/1. It has no Python binding and has not parsed or
  compiled TensorIR, produced an Artifact/Cubin, or executed frontend HIR.
- CP-0040/EV-0053 accepts standalone `pypto.canonical_schedule.v1` identity at
  `fa85e5a`. Its bounded MessagePack wrapper preserves the existing nested
  KernelBuildSpec schedule bytes. Retained CP-0039 build directories were
  reconfigured/rebuilt at exact head; backend-ON/OFF native suites pass 2/2 and
  exact-DSO Python suites pass 98/98. This is the first prerequisite of the
  compiler-owned preparation API, not that API itself.
- CP-0041/EV-0054 accepts the private compile-free structured frontend identity
  boundary at `c4cf755`: exact source, five canonical frontend projections,
  strict schedule prevalidation, producer-shaped ABI validation and final
  BuildSpec construction with no placeholder callable. Fresh backend-ON native
  6/6 and exact-DSO Python 180/2 pass; backend-OFF passes 4/4 and 173/9. It
  invokes no frontend producer and constructs no Artifact.
- CP-0042/EV-0055 accepts the public one-producer structured facade at
  `642ff5b`: one hard-wired private producer invocation yields a sealed
  normalized result, final BuildSpec and immutable Artifact whose complete
  identities join before return. No producer callback, provisional Artifact,
  compile-twice discovery or cache publication exists. Backend-ON native 7/7
  and exact-DSO Python 182/2 pass; backend-OFF native 5/5, functional Python
  7/7 and full Python 175/9 pass. The ON/OFF build-directory names are stale;
  EV-0055 binds the exact DSO paths/hashes and ON embedded revision. CP-0042's
  next step was the separate HIR-authored real-SM120 vector-add smoke, not
  framework work.
- CP-0043/EV-0056 accepts that separate frontend smoke's control boundary:
  implementation `1d1fce4`, manifest-only `47a0c15`, canonical manifest SHA
  `f16c4fba...d8eed`, exact parent-v4 primitive reuse, two pinned HIR/Cubin
  cases, strict non-default-stream lifecycle evidence, and a CPU-only no-replace
  finalizer. The initial NO-GO findings were fixed before publication. Exact
  DSO runner/replay gates pass with CUDA uninitialized; clean root passes
  246 tests plus 126 subtests. CP-0043 itself accepted no frontend GPU result.
- CP-0044/EV-0057 accepts finalized run
  `pypto-20260825T145519Z-1142938-70ac73`. Two exact HIR programs call the
  one-producer facade once each and four fresh non-default-stream executable
  lifetimes match FP32/BF16 CPU references, preserve inputs, release packets and
  unload terminally with no fallback or protected/external NVIDIA activity.
  Final report SHA is `8dbbfbf3...28e8`; independent reconstruction and
  expected no-replace retry rejection are confirmed. This is only two-fixture
  vector-add correctness.
- CP-0045/EV-0058 accepts bounded FusedPointwiseV2 compiler/Cubin production,
  both exact OFF/ON products, native 3/3 and Python 1/1 gates, the maximum
  producer boundary and exact CP44 V1 byte preservation. V2 GPU launch,
  numerical correctness and performance remain separate.
- CP-0046/EV-0059 accepts fused-pointwise correctness controls only:
  implementation `c98f984`, manifest `438c25f`, manifest SHA
  `ce20dd3a...7a6896`, anchors `584f6755...4c97`, nine cases/eighteen
  lifetimes, and a CPU-only finalizer. Post-manifest focused 32/24 and full
  278/150 tests pass. CP-0047/EV-0060 separately accepts the matching policy-2
  real-SM120 result and immutable report; do not rerun it merely to create new
  evidence.
- TensorIR `1dcb38c...` is a local committed feature revision and is fully
  pinned by the PyPTO gitlink/build guards. It has not been published to the
  configured NVIDIA remote; push or otherwise materialize that commit before
  claiming fresh-clone/publication reproducibility. Do not substitute an
  upstream revision that lacks the accepted validator.
- CompileRequest v1 and per-region KernelBuildSpec v1 are accepted as separate
  data-only contracts. KernelBuildSpec passes native 4/4 and exact current-DSO
  Python compiler 122/122; EV-0038 binds commit/tree/source/DSO/run hashes and
  the recovered stale editable-DSO evidence. Producer integration and Artifact
  v1 plus strict production are accepted by CP-0031, ArtifactCache v1 by
  CP-0032, modeled NvidiaExecutable lifecycle by CP-0033 and parent observation
  value by CP-0034. Real module/function load was reached diagnostically in the
  failed first smoke; CP-0038 now accepts the narrow finalized v4 current-stream
  static/dynamic/scalar runtime result.
- `projects/pypto-kernels` is clean at `6f73857...` with typed semantic
  families, canonical process-safe tuning database, matmul invocation ABI v1,
  catalog-bound canonical operator artifact provenance, paged-attention ABI v1
  and its deterministic CPU numerical reference, plus unified GDN core ABI and
  paired-state reference, a structured-matmul reference, and the canonical
  framework-adapter ABI manifest, plus independent CPU Torch cross-checks for
  matmul, shared-prefix paged attention and paired-state GDN, plus the
  source-only canonical operator benchmark evidence contract; 117 tests plus
  71 subtests pass. Torch stays child/test-only. No CUDA operator, generated
  benchmark result, publisher, or device measurement exists yet.
- `projects/pypto-framework-plugins` is clean at `3f9d712...`; its context and
  constructor dispatcher, 41-site Qwen3.5 static inventory, and strict
  trace/artifact coverage evidence contract now include a full producer ABI,
  distribution/import/source identity guard and non-suppressible Torch/SGLang
  preflight ordering, plus a pinned active-route SGLang StateBundle lifecycle
  inventory. It now also has a reviewed 31-source/346-file TorchInductor
  inventory covering registry/cache, scheduler/wrapper/current stream,
  lowering/GEMM fallback and Triton/CuTeDSL/autotune reference lanes. The
  source audit exposes CSEProxy, extern, foreach and multi-template blockers;
  it does not implement them. Its 128 tests, isolated wheel guard, real PEP-660
  guard, and clean wheel pass. The auditor has no CUDA collector or Qwen trace
  yet. `install()` remains intentionally unready/registration-free.
- `upstream/triton` is the clean exact PyTorch pin. Do not use the inherited
  external editable Triton for acceptance.
- The exact Triton dependency/wheel/probe/smoke/replacement tools are present,
  and all ten dependency archive SHA/byte pairs plus manifest `29c073...` are
  source-pinned, reviewed, probed and cached. The exact wheel is accepted and
  frozen; never replace its cache path with an ambient download. The current
  environment must continue to fail the external-editable audit until the
  deferred reversible baseline transaction.
- Those controls are committed as `0c4cc34`, `befe44c`, `640c35a`, and
  `c987811`, followed by acquisition commits `ea39ac5`, `f678fc0`, and
  `5c75ea4`, stop-race fix `fe903fa`, ten-archive source lock `cca595c`, tracked
  offline runner `712a8c2`, generated-evidence retention `8e414e7`, and the
  RPATH-free producer/auditor contract `dd403fe`, internal plugin closure audit
  `234c76f`, and stripped-vendor-ELF identity gate `759a93d`.
  EV-0035 binds the original controls; the current isolated suite is 198 tests
  plus 98 subtests. EV-0037 binds accepted materialization/promotion runs,
  manifest `29c073...`, all archive/tree/overlay hashes and the reviewed cache
  at `caches/triton-build-deps/29c073...`. The first wheel built but was
  correctly rejected for three ELF RUNPATHs; its full 11 GiB tree and evidence
  remain at `builds/triton-wheel-5d6048aa-rejected-rpath-20260824`. The fixed
  runner retains all generated source outputs, disables CMake RPATH and overlays
  a provenance-bound RPATH-free FileCheck without mutating the reviewed cache.
  Corrected wheel SHA `1d58d830...6227a` passes complete audit and pip-free
  fresh probe; EV-0039 binds all hashes and recovery runs. It remains
  uninstalled/reference-only. No GPU smoke or environment replacement has run,
  and no further Triton feature work belongs on the candidate path.
- Protected CPU-only coexistence is explicit. It does not apply to GPU
  benchmarks, hides CUDA from the child, and can signal only a recorded
  workspace group. Run `pypto-20260824T075816Z-91897-64e1ea` passed live beside
  seven protected heavy processes with no protected NVIDIA compute PID or
  pause/abort. This accepts the control path only; TargetInfo still needed no
  waiver, and no GPU/model/performance claim follows.
- The user subsequently authorized about 22 GiB for bounded CPU-heavy work.
  Do not patch the exact-hashed `preflight.py`/`run_isolated.py`: that breaks the
  accepted GPU adapters. Direct attempt `df738f7` was reverted by `7ecc197`.
  D-0016 requires a new CPU-only policy-v2 adapter; policy v1 remains in force
  until that separate layer is reviewed.
- D-0009 and `docs/compile_request_artifact_design.md` ordered data-only
  CompileRequest, per-region KernelBuildSpec, exact LLVM/tileiras producer
  identity, complete bytes-plus-metadata Artifact, compiler-owned persistent
  cache, and a process/device/CUcontext-bound prewarmed executable before any
  real CUDA smoke. Those prerequisites are now accepted through the CP-0034
  observation gate. Workers never query CUDA or carry streams/handles.
- The v4 fixed-command payload/finalizer gate is closed by CP-0038. Preserve its
  controls and final report; do not rerun it merely to change evidence. CP-0042
  closes the single-call frontend Artifact transaction under `pypto.compiler`.
  CP-0043 accepts the separately versioned frontend controller/finalizer and
  CP-0044 accepts its HIR-authored FP32/BF16 SM120 result. CP-0045 accepts
  bounded fused-pointwise compiler/Cubin production while preserving those add
  controls byte-for-byte. CP-0046 accepts the separate fused numerical controls
  and CP-0047 accepts the finalized fixed-fixture result. CP-0048 accepts
  RowReductionV3 compiler/Cubin production at `62eb882`; build its separate
  real-SM120 correctness family next. StructuredMatmulV4 `d755117` then needs
  replay onto `62eb882` before its build. Operator algorithms, framework registration,
  online tuning and model work remain after these compiler gates.
- `docs/coverage_collector_map.md` is the pinned collector implementation map.
  Do not claim `closed_world=true` from ordinary Kineto/NVTX or from the first
  CUPTI-monitor development trace.
- Paged-attention ABI metadata is framework-neutral. Its host validator does
  not prove device contents; SGLang translation and graph lifecycle stay in the
  plugin, while the CUDA Tile kernel stays in `pypto-kernels`.
- GDN state is a paired generation under an exclusive lease. The v1 core
  assumes state is prepared; zero/copy and segmented Radix checkpointing must
  be PyPTO-owned work on the same stream, never Torch/SGLang fallback kernels.
  Ownership, ABI, stream law and ordered prerequisites are frozen in
  `docs/state_bundle_transfer_design.md`; do not implement it before the
  single-DSO, TargetInfo and current-stream executable gates.
- Use `--framework-launch` for every SGLang process. Baseline scripts bind the
  missing `envs/sglang-baseline-py312` lock/profile; candidate launch is expected
  to fail until the external editable Triton is replaced by its workspace wheel.
