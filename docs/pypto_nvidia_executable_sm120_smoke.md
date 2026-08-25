# PyPTO NvidiaExecutable SM120 correctness smoke

This gate accepts only three minimal real-GPU correctness cases and the
`NvidiaExecutable` module/current-stream lifetime. It is not a benchmark and
cannot establish frontend, operator-family, framework, model, CUDA Graph,
coverage, or performance acceptance.

## Frozen child

The controller permits exactly this direct child, with no shell wrapper or
arguments:

```text
/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/bin/python
-I
-B
-S
/home/zhaosiying/pypto-love-tensor-ir/benchmarks/operators/pypto_nvidia_executable_sm120.py
```

`-S` prevents environment `.pth` processing. The runner has only standard
library imports before a parent-owned, no-replace start barrier. After the
barrier it manually exposes the selected site-packages directory, imports
Torch, and bootstraps the exact backend-ON PyPTO DSO. It never imports Triton,
SGLang, FlashInfer, or the framework plugin project.

The runner, contract, preflight, controller, stop tool and finalizer must equal
the exact blobs in
`state/contracts/pypto_nvidia_executable_sm120_v2.json`. Version 1 remains an
immutable record of the pre-ABI-fix control transaction. The root repository
must be clean and the manifest's implementation commit must be an ancestor of
the current checkpoint. Before that manifest is published, launch fails closed.

## Isolation policy

The normal `gpu-benchmark` mode remains completely exclusive. This smoke uses
the separate `gpu-smoke` mode and never produces timings.

With the explicit protected-CPU-lane authorization, admission requires:

- no NVIDIA compute PID outside the not-yet-started owned child;
- no NVIDIA runtime mapping in any process under the protected roots;
- readable `/proc/<pid>/maps` for every protected process;
- at least 24 GiB host memory and 4 GiB free GPU memory;
- at least 64 GiB free workspace disk;
- the exact SM120, driver, Python, Torch, libcudart, PyPTO and source identities.

The parent repeats the audit after `Popen` while the child is blocked. The child
then repeats it before importing Torch. A periodic watchdog treats only the
recorded PID/start-tick descendant closure in the recorded PGID as owned. Any
other NVIDIA compute PID, any protected NVIDIA mapping, an indeterminate audit,
or a resource-floor failure terminates only the verified workspace PGID. It
never signals an external or protected PID.

## Execution

Do not run this command until the control manifest and its checkpoint have been
committed and reviewed. The `--run-id-file` target must not already exist.

```bash
envs/pypto-nvidia/bin/python -E -B -S tools/run_isolated.py \
  --mode gpu-smoke \
  --exact-pypto-nvidia-smoke \
  --allow-protected-zero-nvidia-gpu-smoke \
  --timeout-seconds 1800 \
  --minimum-free-disk-gib 64 \
  --environment pypto-nvidia \
  --framework-profile pypto \
  --environment-lock-mode shared \
  --run-id-file runs/next-pypto-sm120-smoke.json \
  -- \
  /home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/bin/python \
  -I -B -S \
  /home/zhaosiying/pypto-love-tensor-ir/benchmarks/operators/pypto_nvidia_executable_sm120.py
```

If no protected process exists, omit
`--allow-protected-zero-nvidia-gpu-smoke`; all other tokens remain fixed.

The cases are:

1. static FP32 add, `8x8`, pointer-only ABI, grid `[4,1,1]`;
2. dynamic-stride FP32 add, logical `17x9`, flat 12-argument ABI, runtime grid
   `[6,1,1]`, including output-padding preservation;
3. dense FP16 tensor plus by-value FP32 scalar, grid `[4,1,1]`.

Each Artifact is loaded, launched and explicitly unloaded twice. Every packet
is retained through external synchronization. Inputs, logical outputs, dynamic
padding, context address/ID, raw non-default current stream, parameter count,
grid and terminal unload state are checked.

## Finalization

The GPU child publishes replay files and a provisional JSON only after all six
lifetimes succeed. After `run_isolated.py` exits, copy the provisional SHA-256
printed by the child and run:

```bash
envs/pypto-nvidia/bin/python -E -B -S tools/finalize_pypto_nvidia_executable_sm120.py \
  --workspace /home/zhaosiying/pypto-love-tensor-ir \
  --run-id <run-id> \
  --expected-provisional-sha256 <sha256>
```

The CPU-only finalizer joins process, preflight, gate, barrier, provisional and
control-manifest identities. It reopens the exact PyPTO DSO in a `-S` child,
deserializes the CompileRequest, three KernelBuildSpecs and three Artifacts, and
recomputes ABI, producer, Cubin and execution joins. Final evidence is published
without replacement under `reports/data/`.

Any failed or aborted run remains diagnostic evidence only. It must never be
described as a successful CUDA, model, coverage, or performance result.
