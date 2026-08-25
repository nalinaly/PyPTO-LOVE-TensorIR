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

- Read `state/checkpoints/CP-0036.md` and evidence `EV-0005` through `EV-0049`.
- Root `c71f32b` plus manifest-only `3de4cf7` owns the repaired v3
  correctness-only SM120 smoke. The manifest SHA is `978e8737...b6e3` and
  binds seven exact control blobs to the Layer-A commit/tree. Controller and
  finalizer require `-E -B -S`; the child and semantic replay use
  `-I -B -S`. Clean run `pypto-20260825T071601Z-892819-67acee` passes 224
  tests plus 106 subtests; exact-DSO compiler/runtime suites pass, and CPU-only
  run `pypto-20260825T070649Z-890066-2a1ae2` preserves the wrapper markers and
  recompiles all three deterministic Cubins through the v3-selected DSO with
  device count zero and CUDA uninitialized. A serialized v3 smoke replay does
  not exist yet. The retained
  first real run reached module/function prewarm but failed before packet
  preparation or launch.
- `projects/pypto` is clean at `206447c...`. Single-DSO, immutable SM120
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
  Final ON/OFF products remain RPATH-free with five standard dependencies/two
  definitions. ON SHA `15675c47...018c` passes native 9/9 plus exact-DSO
  Python 142/2; OFF SHA `32c2dea0...4109` passes native 7/7 plus Python
  135/9. EV-0049 binds their exact hashes, revisions, compile rows, JUnit and
  product audit. No PyPTO kernel launch or CUDA numerical result is accepted.
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
  failed first smoke; current-stream launch remains unverified.
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
- D-0009 and `docs/compile_request_artifact_design.md` ordered data-only
  CompileRequest, per-region KernelBuildSpec, exact LLVM/tileiras producer
  identity, complete bytes-plus-metadata Artifact, compiler-owned persistent
  cache, and a process/device/CUcontext-bound prewarmed executable before any
  real CUDA smoke. Those prerequisites are now accepted through the CP-0034
  observation gate. Workers never query CUDA or carry streams/handles.
- The fixed-command payload/finalizer is repaired and v3-manifest-bound by
  CP-0036. Next run exactly the command in
  `docs/pypto_nvidia_executable_sm120_smoke.md` under a fresh green
  `gpu-smoke` admission. Do not edit any of the seven manifest-bound controls.
  Capture the printed provisional SHA, wait for the controller to exit, then
  invoke the no-site finalizer. Retain a failed/aborted run as diagnostic only.
  If prewarm next fails at occupancy, investigate a tile-aware physical
  residency check; never change CUDA Tile's logical host block to 128.
  Keep frontend lowering, operator algorithms, framework registration, online
  tuning and model work after this real-runtime gate.
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
