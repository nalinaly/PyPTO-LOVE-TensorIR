# PyPTO fused-pointwise SM120 admission v2

This is a separately versioned admission layer for the fixed nine-case
fused-pointwise correctness gate. CP46 v1 remains byte-for-byte unchanged and
continues to require its original 24 GiB protected-lane admission.

V2 changes only host-memory admission. With the explicit protected-zero-NVIDIA
authorization, both parent preflights and the direct child require at least
`22 * 1024 * 1024 = 23,068,672 KiB` from Linux `MemAvailable`. The comparison
is exact: 23,068,671 KiB rejects and 23,068,672 KiB admits. Without that
authorization, exclusive admission remains 32 GiB.

The living-run watchdog remains unchanged at a 16 GiB owned-run abort floor.
The free-GPU-memory floor remains 4 GiB. Every parent, child, periodic and
post-exit check still rejects external or protected NVIDIA compute PIDs,
protected NVIDIA runtime mappings, unreadable protected maps, an unexpected
GPU/driver, ownership ambiguity, survivors, or a changed control identity.
Only the verified owned process group may be terminated by the inherited stop
primitive; the control never signals an external process.

## Composition and evidence

The v2 files exact-load and hash-check the accepted v1 observation, isolation,
numerical and CPU-replay helpers. They do not edit those files, rewrite their
source, replace `subprocess.run`, or spoof Python process identity:

- `tools/preflight_gpu_smoke_v2.py` owns the policy-2 preflight report;
- `tools/run_pypto_fused_pointwise_sm120_v2_isolated.py` owns both parent
  preflights, the action gate, start barrier, watchdog and post-exit audit;
- `benchmarks/operators/pypto_fused_pointwise_sm120_v2.py` is the authenticated
  direct child and records `mem_available_kib` plus `host_memory_floor_kib`;
- `tools/finalize_pypto_fused_pointwise_sm120_v2.py` directly validates the
  22/32 GiB admission, 16 GiB abort and 4 GiB GPU floors before running the
  lifecycle, guard, frontend, replay, independent CPU-reference and three-way
  numerical audits;
- `_pypto_fused_pointwise_sm120_contract_v2.py` and
  `_pypto_fused_pointwise_sm120_control_manifest_v2.py` bind the new family to
  the complete accepted v1 manifest, runner and compiler/Cubin anchors.

Focused regression coverage freezes every accepted v1 control input used by
this composition: runner, contract, anchor generator, compile-anchor manifest,
control validator, controller, finalizer, preflight, isolation, stop primitive,
focused test, runbook and published v1 manifest. It also freezes the shared
`CASE_SPECS` object, the exact helper globals, and all injected dependency
objects before v2 publication.

The implementation commit intentionally omits
`state/contracts/pypto_fused_pointwise_sm120_v2.json`. The controller and
finalizer therefore fail closed. After source review and CPU-only tests, a
separate manifest-only commit must publish canonical JSON containing the exact
implementation commit/tree and the ordered six control records with bytes,
mode and SHA-256. A clean post-publication focused and full root run is required
before any GPU command.

After that publication and review, the protected-lane command is:

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/run_pypto_fused_pointwise_sm120_v2_isolated.py \
  --allow-protected-zero-nvidia-gpu-smoke \
  --run-id-file runs/next-pypto-fused-pointwise-sm120-v2.json
```

The CPU-only finalizer remains a separate no-replace transaction:

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/finalize_pypto_fused_pointwise_sm120_v2.py \
  --workspace /home/zhaosiying/pypto-love-tensor-ir \
  --run-id <run-id> \
  --expected-provisional-sha256 <sha256>
```

## Claim boundary

V2 reuses the exact v1 HIR, TensorIR, BuildSpec, Artifact, Cubin, DSO, case,
comparison, canary, stream, packet-release and unload anchors. It does not claim
GPU execution or numerical correctness until a fresh admitted run is finalized
and independently reviewed. It also does not claim general FusedPointwiseV2
correctness, other shapes/chains, performance, CUDA Graph, framework/model
correctness, strict coverage, or any reinterpretation of CP44 or CP46 v1.
