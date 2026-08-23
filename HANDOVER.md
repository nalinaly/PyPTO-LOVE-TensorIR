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

- Read `state/checkpoints/CP-0004.md` and evidence `EV-0005` through `EV-0008`.
- `projects/pypto` has two intentional uncommitted CMake files implementing the
  single-DSO compiler object boundary. Editable build and 486 focused tests
  pass. The first full run's three failures have layered fixes and targeted
  proof at `EV-0009`; the full rerun and fresh wheel gate have not run.
- `projects/pypto-kernels` is clean at `632bf10...` with 24 semantic-contract
  tests and an independently verified wheel import.
- `projects/pypto-framework-plugins` is clean at `8b8b5de...`; its context and
  constructor dispatcher contract has 33 passing tests and clean-wheel proof,
  but `install()` remains intentionally unready/registration-free.
- `upstream/triton` is the clean exact PyTorch pin. Do not use the inherited
  external editable Triton for acceptance.
- Re-run the live heavy preflight before resuming any wheel build. At this
  checkpoint zcode is running TP=2 SGLang and gem5, so this project's heavy work
  must wait without signalling or cleaning up that lane.
