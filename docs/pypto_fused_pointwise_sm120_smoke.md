# PyPTO fused-pointwise SM120 correctness gate v1

This is the full nine-case numerical gate for the bounded
`FusedPointwiseV2` compiler family. It accepts only the frozen fixtures below
through
`HIR -> compile_structured_strict -> BuildSpec/Artifact/Cubin -> NvidiaExecutable`.
It is neither a benchmark nor general fused-pointwise correctness.

The implementation commit must not run this GPU gate. The separately reviewed
`state/contracts/pypto_fused_pointwise_sm120_v1.json` publication commit is the
release condition; that file is intentionally absent here, so the controller
fails closed before the start barrier is released.

## Frozen compiler and legacy boundary

The compiler source is PyPTO
`b83fcd3ddc497d585bcc45883eede179aff7d4d2`, tree
`49eda98f3ed8d72bfd14d5a5900cdc0e71ca699d`, TensorIR `1dcb38c`, CUDA
Tile `af241704`, and LLVM `57109bef`. The backend-on DSO is
`builds/pypto-fused-pointwise-v2-on-b83fcd3-final/product/pypto_core.cpython-314-x86_64-linux-gnu.so`,
784,043,568 bytes, SHA-256
`0e8f33c263e06777aec06263bf32ca59ac554868529f3fa085212cf27e2facbe`.

`state/contracts/pypto_fused_pointwise_compile_anchors_v1.json` is the
no-replace CPU/compiler case manifest. It binds two independent CUDA-hidden
runs of `tools/generate_pypto_fused_pointwise_anchors.py`, their process and
preflight sidecars, exact DSO and CompileRequest, and every case-keyed HIR,
canonical TensorIR source, five frontend projections, callable ABI, grid,
pointer ABI, zero workspace, BuildSpec, Artifact, and Cubin identity. Both runs
must report Torch CUDA uninitialized before and after compilation and must
publish identical canonical record bytes.

The final generator runs are `pypto-20260826T042728Z-1382280-ce1fa0` and
`pypto-20260826T042750Z-1382496-07c3e7`. Their case-keyed record arrays are
byte-identical with SHA-256
`01e8f99dfb0a1aa0e5788177b7d41cc6ed62983037aa57ed8628bb0a1594b844`;
the canonical binding manifest is 21,490 bytes with SHA-256
`584f6755bbd248de5bb6ddd3ff610da8082667bc892a6cff6583ea42d4c44c97`.

The new control validator also binds, without changing or reinterpreting them:

- the complete accepted v4 isolation manifest, SHA-256 `a079c4d2...98bf`;
- the complete CP43 frontend-vector-add manifest, SHA-256 `f16c4fba...d8eed`;
- the accepted CP44 report, SHA-256 `8dbbfbf3...28e8`, mode `0444`.

Every new control dependency is read, SHA-256 hashed, compiled, and executed
from its exact `.py` bytes. No new control loader consults Python bytecode
caches. The validator rejects a matching direct or `__pycache__` bytecode entry
for the runner, contract, validator, generator, controller, finalizer, or test
before loading another control dependency; `-B` alone is not sufficient.

CP44 remains the legacy two-case add claim. None of its runner, contract,
controller, finalizer, validator, manifest, report, tests, or Cubins is part of
the new case set.

## Exact case matrix

Every artifact is compiled once and reused by two fresh executable lifetimes.
Shapes are logical dense shapes; the one schedule dimension is the normalized
flattened pointwise tile.

| Case | Dtype and shape | Tile | Grid | Inputs / assignments | Purpose |
|---|---|---:|---:|---:|---|
| `arith_fp32_tail` | FP32 `[3,5]` | 8 | `[2,1,1]` | 4 / 8 | arithmetic forms, neg, FMA discriminator, tail 7 |
| `arith_bf16_rank3_tail` | BF16 `[2,3,5]` | 16 | `[2,1,1]` | 4 / 8 | BF16 stage rounding, rank-3 normalization, tail 14 |
| `exp_fp32_tail1` | FP32 `[17]` | 16 | `[2,1,1]` | 1 / 1 | isolated exp and tail 1 |
| `exp_bf16_exact_tile` | BF16 `[8,8]` | 16 | `[4,1,1]` | 1 / 1 | FP32-emulated BF16 exp, exact tiling |
| `recip_fp32_tail` | FP32 `[3,5]` | 8 | `[2,1,1]` | 1 / 1 | isolated reciprocal and tail 7 |
| `recip_bf16_tail1` | BF16 `[17]` | 16 | `[2,1,1]` | 1 / 1 | BF16 candidate against FP32-then-BF16 reference, tail 1 |
| `rsqrt_fp32_rank3_tail` | FP32 `[2,3,5]` | 16 | `[2,1,1]` | 1 / 1 | isolated rsqrt, rank-3 tail |
| `rsqrt_bf16_exact_tile` | BF16 `[8,8]` | 16 | `[4,1,1]` | 1 / 1 | BF16 candidate against FP32-then-BF16 reference |
| `max16x64_fp32_tail1` | FP32 `[17]` | 16 | `[2,1,1]` | 16 / 64 | maximum input, SSA, pointer ABI, and tail |

The arithmetic program is exactly:

```text
v0 = mul(x, x)
v1 = add(v0, z)
v2 = muls(v1, 2^p)
v3 = adds(v2, 0.3)
v4 = subs(v3, 0.3)
v5 = mul(v4, m)
v6 = sub(v5, q)
v7 = neg(v6)
```

`p=23` for FP32 and `p=7` for BF16. Lane zero uses
`x=1+2^-p`, `z=-(1+2^(1-p))`, `m=1`, and `q=+0`; stage-wise typed
rounding must produce the exact `-0` result. This distinguishes separate
multiply/add from contraction or retained extra precision. Other lanes use the
fixed modular formulas in the contract. BF16 arithmetic is
`QB(Q32(lhs) op Q32(rhs))` after every assignment.

The maximum program is 15 sequential adds of `x0...x15` followed by 49
sequential neg operations. Reassociation is not accepted.

## Numerical policy

Arithmetic, neg, reciprocal finite-normal lanes, and the maximum chain require
exact logical bits. Signed zero is significant. Exp and rsqrt use monotonic
ordered integer ULP distance plus a relative check:

- FP32 exp and rsqrt: at most **4 ULP**, `rtol=2e-6`, `atol=0`;
- BF16 exp and rsqrt: at most **1 ULP**, `rtol=1/128`, `atol=0`.

The BF16 oracle is an explicit FP32 operation followed by one BF16
round-to-nearest-even conversion. `high_precision` is not accepted by V2 and is
not part of this gate. Subnormal inputs are excluded rather than silently
weakening the policy.

Repetition zero is finite. Repetition one freezes the special prefixes:

- exp: `[-Inf,+Inf,NaN,-0,+0]` -> `[+0,+Inf,NaN,1,1]`;
- reciprocal: `[+0,-0,+Inf,-Inf,NaN]` -> `[+Inf,-Inf,+0,-0,NaN]`;
- rsqrt: `[-1,-Inf,NaN,-0,+0,+Inf,4]` ->
  `[NaN,NaN,NaN,-Inf,+Inf,+0,0.5]`.

NaN payload and NaN sign are ignored; all other classifications and signs are
exact.

## Runtime transaction

Each lifetime owns fresh input and output allocations. The ABI tensor is an
interior contiguous view surrounded by 16-element exact prefix and suffix
canaries. Sixteen is at least the maximum tile minus one, so even an incorrect
full tile-16 store for a tail of one remains inside the suffix and changes its
hash. The runner records every individual input hash before and after launch
and all guard hashes, so tail masking is affirmative evidence.

The normative Torch reference runs eagerly, one operation per call, on a
distinct non-default reference stream. Reference computation is explicitly
outside candidate coverage. That stream synchronizes before the PyPTO launch.
The candidate then launches on its selected non-default current stream,
externally synchronizes, records raw actual bytes, releases the retained packet
only after synchronization, and explicitly unloads to terminal `Unloaded`
with zero bound context identity.

Replay contains one CompileRequest and, for every case, exact HIR, canonical
source, BuildSpec, Artifact, raw Cubin, and per-execution raw individual inputs,
reference, and actual output. The child may import no Triton, SGLang, or
FlashInfer provider and may use no fallback.

## Isolation and finalization

After the later control-manifest publication, the only accepted direct child is
the isolated `-I -B -S` runner. The existing v4 preflight, process group,
watchdog, start barrier, stream/provider audit, and stop primitive remain
unchanged.

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/run_pypto_fused_pointwise_sm120_isolated.py \
  --allow-protected-zero-nvidia-gpu-smoke \
  --run-id-file runs/next-pypto-fused-pointwise-sm120-v1.json
```

This command **must not run** before the separate control manifest is reviewed.
Omit the protected-lane option only when no protected process exists.

The CPU-only finalizer accepts the externally copied provisional SHA-256:

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/finalize_pypto_fused_pointwise_sm120.py \
  --workspace /home/zhaosiying/pypto-love-tensor-ir \
  --run-id <run-id> \
  --expected-provisional-sha256 <sha256>
```

Its isolated exact-DSO child performs deserialization only and asserts Torch
CUDA remains uninitialized. Independently, the standard-library finalizer
reconstructs all raw inputs and stage-wise CPU references, rechecks exact/ULP,
special, signed-zero, and hash joins, and publishes canonical mode-`0444`
evidence atomically without replacement.

Acceptance does not claim general FusedPointwiseV2 correctness, other chains,
shapes, ranks, subnormals or high-precision behavior; cross-build Cubin
determinism; reduction, matmul or memory lowering; CUDA Graph behavior;
performance; TorchInductor, SGLang, Qwen correctness or strict coverage; or any
extension of CP44.
