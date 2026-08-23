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

## Current resume point

- Read `state/checkpoints/CP-0010.md` and evidence `EV-0005` through `EV-0020`.
- `projects/pypto` has two intentional uncommitted CMake files implementing the
  single-DSO compiler object boundary. Editable build and 486 focused tests
  pass. The first full run's three failures have layered fixes and targeted
  proof at `EV-0009`; the full rerun and fresh wheel gate have not run.
- `projects/pypto-kernels` is clean at `3c220c5...` with typed semantic
  families, canonical process-safe tuning database, matmul invocation ABI v1,
  catalog-bound canonical operator artifact provenance, paged-attention ABI v1
  and its deterministic CPU numerical reference, plus unified GDN core ABI v1;
  82 tests plus 61 subtests and a clean wheel import pass. No CUDA operator or
  device measurement exists yet.
- `projects/pypto-framework-plugins` is clean at `d46a58c...`; its context and
  constructor dispatcher, 41-site Qwen3.5 static inventory, and strict
  trace/artifact coverage evidence contract have 68 passing tests and a clean
  wheel. The auditor has no CUDA collector or Qwen trace yet. `install()`
  remains intentionally unready/registration-free.
- `upstream/triton` is the clean exact PyTorch pin. Do not use the inherited
  external editable Triton for acceptance.
- Re-run the live heavy preflight before resuming any wheel build. At this
  checkpoint zcode is running TP=2 vLLM and gem5, so this project's heavy work
  must wait without signalling or cleaning up that lane.
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
- Use `--framework-launch` for every SGLang process. Baseline scripts bind the
  missing `envs/sglang-baseline-py312` lock/profile; candidate launch is expected
  to fail until the external editable Triton is replaced by its workspace wheel.
