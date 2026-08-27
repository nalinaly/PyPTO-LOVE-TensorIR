# Zero-context handover: PyPTO NVIDIA/Qwen3.5 goal

> **2026-08-28 current override:** the historical 2026-08-26 resume sequence
> below is complete and must not be replayed. Read the top section of
> `CHECKPOINT.md` first. The live unaccepted implementation bases are root plan
> commit `d042530` plus this later handover commit, PyPTO `d1b90b7`, TensorIR
> `a48606b`, kernels `5fbf813`, and plugins `0e09d51`.
> Final-source TensorIR/FileCheck/standalone-Cubin gates pass, but the exact
> final PyPTO DSO rebuild and QK GPU numerical gate are pending a reviewed
> 22-GiB CPU-v2 admission window. Never promote the standalone Cubin to a
> PyPTO/runtime/model claim, and never signal the protected ZCode/gem5 lanes.

Date: 2026-08-26 Asia/Shanghai. The persistent goal is active and far from
complete. This document records the exact pause boundary; it does not narrow
the requested end state.

## Read first

From `/home/zhaosiying/pypto-love-tensor-ir`, read completely:

1. `GOAL.md`
2. `PLAN.md`
3. `CHECKPOINT.md`
4. `HANDOVER.md`
5. `TODO.md`
6. `DECISIONS.md`
7. `VERSIONS.lock`, `ENVIRONMENT.lock`, and `WORKSPACE.lock`
8. `docs/implementation_map.txt`
9. `docs/pypto_row_reduction_sm120_smoke.md`
10. `docs/structured_matmul_v4_replay_map.md`
11. this file

Treat the live worktrees, processes, hashes, and tests as authoritative. Do not
promote historical plans or this handover into runtime evidence without fresh
verification.

## Non-negotiable safety and architecture

- Never modify `/home/zhaosiying/amdgpu-sim`,
  `/home/zhaosiying/zcode-lane`, or their processes. They are read-only external
  scopes. Never signal, stop, kill, clean, or restart them, even under OOM.
- Never reboot, run `wsl --shutdown`, or restart the host without explicit user
  approval.
- Use NVIDIA/CUDA only. Reject ROCm/HIP/simulator environment or DSO leakage.
- Never use a broad cleanup command. Preserve every user/external change.
- PyTorch and SGLang upstream checkouts must remain zero-diff.
- All handwritten high-performance operators belong in the independent
  `projects/pypto-kernels` repository. Framework adapters belong in
  `projects/pypto-framework-plugins`; they contain no kernel algorithms.
- Preserve the single mainline:
  `SGLang -> Dynamo -> Inductor -> PyPTO CUDA backend -> CUDA Tile -> SM120`.
  TensorIR is private PyPTO compiler infrastructure, not a second product.
- Final Qwen3.5-9B model-forward compute must be strict 100% PyPTO: no Triton,
  FlashInfer, CuTeDSL, sgl-kernel, eager ATen CUDA, cuBLAS/cuBLASLt, or unknown
  compute fallback.
- Triton is frozen reference/baseline infrastructure only. Do not resume Triton
  feature work or count it as PyPTO coverage.
- Correctness, coverage, performance, and end-to-end acceptance are separate
  evidence tiers. Never promote a compile, Cubin, smoke, or readable output to
  a stronger claim.

## Exact committed state

The committed implementation base immediately before this handover is
`290dffc178f58d73b5ff60957d3b10981e8aa083`
(`docs: freeze StructuredMatmulV4 replay map`). The handover itself may be a
later documentation-only commit, so inspect current HEAD. Important ancestors:

- `23efafaac88fc62698b439b037cb96d95ecbd927`: hardened RowReductionV3
  correctness implementation.
- `34ee759`: separate RowReductionV3 manifest-only commit.
- `0108615`: persisted RowReduction control checkpoint.
- `290dffc`: read-only StructuredMatmulV4 replay/conflict/build map.

Primary `projects/pypto` is clean at
`62eb88251df5bdad95277a9d619d20da9bf121eb`. Do not replay matmul yet.

RowReduction compile anchors are accepted source-only inputs:

- `pypto-20260826T110849Z-17249-b18e99`
- `pypto-20260826T110905Z-17569-3de174`
- `state/contracts/pypto_row_reduction_compile_anchors_v1.json`
  SHA-256 `14af24e4929fd629475cf70a871c1f8400daa59ed22b6f988c9d4a00968418a0`
- control manifest SHA-256
  `7ae64b2273e4906f05e26a070460b713e3d3e5de74194329663dd76dd68ccc31`

The clean focused RowReduction suite passed 34 tests plus 84 subtests. Three
independent final reviews were P0/P1/P2 zero. The clean post-manifest full root
suite and real GPU controller/finalizer have **not** run.

## Dirty uncommitted CPU-v2 draft

Apart from the committed handover documents, the root is intentionally dirty
with exactly six untracked additive files:

- `tools/_pypto_cpu_coexistence_v2_contract.py`
- `tools/preflight_cpu_coexistence_v2.py`
- `tools/run_pypto_cpu_coexistence_v2_isolated.py`
- `tools/_pypto_cpu_coexistence_v2_control_manifest.py`
- `tests/test_pypto_cpu_coexistence_v2.py`
- `docs/pypto_cpu_coexistence_v2.md`

Do not delete, stash, silently rewrite, or publish them. The final manifest
`state/contracts/pypto_cpu_coexistence_v2.json` is deliberately absent, so the
draft cannot launch a child.

At the pause boundary the files had these sizes and hashes, but subsequent
verification is required because the latest safety edits were not rerun:

| File | Bytes | SHA-256 at pause |
|---|---:|---|
| contract | 8,833 | `20cc69ac...47718a3` |
| preflight | 12,276 | `459902cb...11ec33` |
| controller | 44,689 | `b00d1291...ac25c` |
| control validator | 9,936 | `0564c034...67146` |
| tests | 66,502 | `a8f9c1a8...df5b86` |
| docs | 3,364 | `68ef5083...f68de` |

Those hashes predate the last start-gate/termination/test edits and are only a
drift clue, not current pins. The last green result was 37 tests before the
latest edits. Current AST/Ruff/tests and independent reviews are stale. Start
by treating the draft as unverified/no-go.

Latest intended design:

- existing `preflight.py`, `run_isolated.py`, `stop_run.py`, NVIDIA executable
  contract/control, and v4 manifest remain exact and unchanged;
- 22 GiB admission/resume, strictly below 16 GiB pause;
- fixed CUDA-hidden Python start gate prevents the requested command from
  executing before complete durable metadata and ownership verification;
- emergency metadata handles persistent startup capture failure;
- owned/protected/workspace/external NVIDIA PID classes are complete and
  disjoint;
- all group emptiness and survivor decisions use exact kernel-proven PGID
  auditing plus environment ownership;
- STOP/CONT/TERM use exact verified group primitives only; never SIGKILL;
- manifest validation must precede lease acquisition and `Popen`.

Before any CPU-v2 commit, run AST, Ruff, `git diff --check`, and the focused
tests. Audit the latest start-gate failure tests and exact termination rewrite.
Obtain at least two fresh independent P0/P1/P2 reviews. Commit exactly the six
files with the manifest absent. Only then generate and separately commit the
canonical manifest, followed by clean post-manifest tests. Do not use the v2
controller live merely because its unit tests pass.

## Immediate resume sequence

1. Read the files listed above and inspect `git status`, all worktrees, memory,
   disk, NVIDIA compute PIDs, and protected process/runtime mappings.
2. Finish and independently review the dirty CPU-v2 draft as described above,
   or stop and ask the user if its latest source cannot be made safe. Do not
   publish its manifest from an unreviewed tree.
3. Return the root to a clean committed state.
4. Run the deferred clean post-manifest full CPU suite. With current policy-v1
   and sufficient memory, use:

```bash
/usr/bin/env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void \
  PYTHONDONTWRITEBYTECODE=1 envs/pypto-nvidia/bin/python \
  tools/run_isolated.py --mode heavy \
  --allow-protected-cpu-only-coexistence \
  --timeout-seconds 1800 --minimum-free-disk-gib 64 \
  --environment pypto-nvidia --framework-profile pypto \
  --run-id-file runs/next-row-postmanifest-full.json -- \
  /usr/bin/env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/bin/python \
  -B -m pytest -q -p no:cacheprovider tests
```

5. Require the full suite to pass from a clean root. A preflight refusal is not
   a test pass.
6. Recheck the real GPU admission. Performance remains exclusive. The fixed
   correctness lane may coexist only under its reviewed protected-zero-NVIDIA
   policy: protected compute PIDs, protected NVIDIA mappings, unreadable maps,
   and external compute PIDs must satisfy the controller/finalizer contract.
7. Launch only the reviewed RowReduction controller:

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/run_pypto_row_reduction_sm120_isolated.py \
  --allow-protected-zero-nvidia-gpu-smoke \
  --run-id-file runs/next-pypto-row-reduction-sm120.json
```

8. Read the run ID from the no-replace run-id file. Inspect all sidecars and
   provisional output without editing them. Compute the provisional SHA-256,
   then run:

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/finalize_pypto_row_reduction_sm120.py \
  --workspace /home/zhaosiying/pypto-love-tensor-ir \
  --run-id <RUN_ID> \
  --expected-provisional-sha256 <PROVISIONAL_SHA256>
```

9. Repeat the identical finalizer command and require an unchanged
   no-replace rejection. Independently review the final report before creating
   CP-0049/EV-0062. Do not claim general reduction or performance.
10. Only after RowReduction acceptance, follow
    `docs/structured_matmul_v4_replay_map.md` exactly: replay `6ee412a` then
    `d755117` onto `62eb882` in a new worktree, resolve only the documented
    descriptor formatting conflict, and verify the expected trees before
    sequential OFF/ON builds.

## Continue to the actual end state

Do not stop after RowReduction or StructuredMatmul. Continue the milestone
ladder in `PLAN.md`:

1. complete generic pointwise/reduction/indexing/matmul compiler and profiling
   foundations;
2. implement the zero-diff TorchInductor PyPTO CUDA scheduling/kernel/wrapper/
   template backend with strict no-fallback tests;
3. implement and tune standalone `pypto-kernels` paged attention prefill/decode
   and GDN prefill/decode on real SM120;
4. integrate zero-diff SGLang attention, linear-attention, and piecewise
   Inductor/PyPTO plugins;
5. bring up Qwen3.5-0.8B correctness, continuous batching, Radix/chunked prefill,
   CUDA Graph, and strict 100% model-forward coverage;
6. stabilize/profile 0.8B before moving to Qwen3.5-9B;
7. bring up and tune 9B without model-name/hidden-size hacks;
8. prove `coverage=100%` and `fallback_compute_kernels=0`, then run reproducible
   baseline/candidate operator and E2E benchmarks and write the required report.

The goal is complete only when every Definition-of-Done item in `GOAL.md` and
`PLAN.md` has current evidence. Keep the persistent goal active otherwise.
