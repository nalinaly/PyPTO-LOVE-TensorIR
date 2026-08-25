# PyPTO frontend vector-add SM120 correctness smoke v1

This separately versioned gate accepts only the two fixed
`HIR -> StructuredCompileResult{KernelBuildSpec, Artifact} -> NvidiaExecutable`
vector-add cases below. It is a correctness smoke, not a benchmark. It does not
accept other operator families, TorchInductor, SGLang, a model, CUDA Graph,
coverage, or performance.

## Frozen inputs and controls

The source checkout is PyPTO
`642ff5bd79ee96b9e5a279a2bc945ad7a78362b7` with tree
`77d8078d8df84dd7cf8544350918e25b8282976d`. The backend-on DSO is fixed by
workspace-relative path, size, SHA-256, and its embedded PyPTO revision. Its
directory name still contains `c4cf755`; that label is stale and is not used as
source identity.

The v1 control manifest binds the runner, contract, manifest validator,
versioned controller, finalizer, and the unchanged accepted v4 blobs for
`preflight.py`, `run_isolated.py`, and `stop_run.py`. The root repository must be
clean, the manifest implementation commit must be an ancestor of `HEAD`, and
none of those control blobs may differ. The unchanged parent may create the
standard-library-only child behind its start barrier before manifest
validation; a missing or different manifest fails closed before barrier
release and therefore before any GPU or framework import.

The parent dependency is the complete accepted v4 manifest blob
`a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf`,
not merely three independently repeated file hashes. The v1 validator also
requires its three inherited primitive records to equal that parent manifest.

The versioned controller permits exactly this direct child, without a shell or
additional child arguments:

```text
/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/bin/python
-I
-B
-S
/home/zhaosiying/pypto-love-tensor-ir/benchmarks/operators/pypto_frontend_vector_add_sm120.py
```

## Cases and transaction boundary

The exact cases are:

1. FP32 `[8,8]`, dense strides `[8,1]`, tile `16`, grid `[4,1,1]`,
   13,784-byte Cubin SHA-256
   `dcc529fc856a508642c8b5a98c6fc4e223e10a49cc9f8a200b8984f92b6483ab`;
2. BF16 `[128]`, dense runtime stride `[1]`, tile `16`, grid `[8,1,1]`,
   13,784-byte Cubin SHA-256
   `83afb2df234ad90167351d608052d44f86e26a8ca73959369992cd139943bc13`.

Those Cubin identities were frozen before publication by two independent,
CUDA-hidden, CPU-only compilations through the exact DSO. Both repetitions
reported CUDA uninitialized and identical BuildSpec, Artifact, callable, Cubin,
source, and HIR identities. This pre-publication freezing is separate from the
later accepted child, which still invokes the producer exactly once per case.

For each case, the runner constructs the minimal two-input `tensor.add` HIR with
both parameter directions explicitly set to `In`, calls `pypto.ir.serialize`,
then calls `pypto.ir.deserialize`, proves structural equality and exact
canonical reserialization, and passes only the restored HIR to
`pypto.compiler.compile_structured_strict`. That facade is called exactly once
per case. The returned final BuildSpec and Artifact must join on source,
frontend projections, callable ABI, cache, loader, and Artifact identities with
no fallback. There is no discovery compile, pre-Artifact cache lookup, retry,
or second producer call.

Each one-time Artifact has two independent `NvidiaExecutable` lifetimes. Every
lifetime allocates its own inputs and output, prepares and launches on the
selected non-default current stream, externally synchronizes that stream,
checks output bytes and read-only inputs, explicitly releases the retained
launch packet, and explicitly unloads the executable to terminal `Unloaded`
state. Four successful lifetimes are required in total.

## Isolation and execution

The controller injects this v1 contract into the unchanged v4 admission,
preflight, watchdog, process-group, and stop primitives. Normal GPU benchmark
exclusivity remains unchanged. Protected CPU-lane coexistence is permitted only
when explicitly requested and every v4 zero-NVIDIA mapping, compute-process,
memory, GPU-memory, disk, PID/start-tick, and process-group check succeeds.

Do not run the GPU smoke until the manifest publication commit has been
reviewed. The run-ID file must not already exist.

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/run_pypto_frontend_sm120_isolated.py \
  --allow-protected-zero-nvidia-gpu-smoke \
  --run-id-file runs/next-pypto-frontend-sm120-v1.json
```

Omit `--allow-protected-zero-nvidia-gpu-smoke` when there is no protected
process. To stop only a verified owned run, use the unchanged stop primitive:

```bash
envs/pypto-nvidia/bin/python -E -B -S tools/stop_run.py \
  --run-id <run-id> --signal TERM
```

## CPU-only finalization

The child publishes the read-only CompileRequest, HIR, BuildSpec, and Artifact
replay blobs before execution, so those files alone remain diagnostic. It
publishes the provisional JSON only after all four lifetimes succeed. After the
controller exits, copy the printed provisional SHA-256 and run:

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/finalize_pypto_frontend_vector_add_sm120.py \
  --workspace /home/zhaosiying/pypto-love-tensor-ir \
  --run-id <run-id> \
  --expected-provisional-sha256 <sha256>
```

The finalizer is CPU-only. Its isolated child imports the exact DSO solely to
call `pypto.ir.deserialize` and the data-only deserializers for the
CompileRequest, both BuildSpecs, and both Artifacts, followed by canonical
reserialization. It never invokes either compiler entry point, constructs an
executable, observes a CUDA runtime, or calls a CUDA Runtime/Driver API. The
public PyPTO package may transitively import Torch modules; the finalizer's only
Torch call is the state-only `torch.cuda.is_initialized()` assertion proving
that replay did not initialize CUDA. Final evidence is published read-only and
without replacement.

A failed, aborted, provisional, or unfinalized run is diagnostic evidence only.
It must not be reported as accepted CUDA correctness or as any broader model,
coverage, or performance result.
