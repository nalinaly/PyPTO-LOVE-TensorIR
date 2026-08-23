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

- Read `state/checkpoints/CP-0018.md` and evidence `EV-0005` through `EV-0030`.
- `projects/pypto` has two intentional uncommitted CMake files implementing the
  single-DSO compiler object boundary. Editable build and 486 focused tests
  pass. The first full run's three failures have layered fixes and targeted
  proof at `EV-0009`; the full rerun and fresh wheel gate have not run.
  Exact safe commands and expected counts are frozen in
  `docs/single_dso_acceptance_gate.md`.
- `projects/pypto-target-info` is a separate clean worktree at unbuilt candidate
  `9939b88`. Never treat it as main-branch or build acceptance. Apply it only
  after the object-DSO CMake commit; expect a small CMake conflict.
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
- The packaging-time preflight was briefly green, but the latest mandatory
  recheck returned 75 while zcode was running `zcode-vllm-tp2-v4` with TP=2
  vLLM workers and gem5. No PyPTO heavy command
  started. Wait for natural exit, rerun preflight, and never signal or clean up
  that lane.
- After the single-DSO full-suite/fresh-wheel gate closes, the next isolated
  PyPTO commit is immutable NVIDIA target identity: explicit SM120 resource
  traits, no SM100 defaults, and no fake legacy backend instance.
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
