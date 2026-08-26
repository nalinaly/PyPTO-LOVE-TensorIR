# StructuredMatmulV4 replay map

This document is source-only preparation. Actual replay is ordered after the
RowReductionV3 clean full CPU regression and real-SM120 numerical acceptance.
It authorizes no build, GPU launch, numerical, profiling, or performance claim.

## Provenance

Current PyPTO is clean at
`62eb88251df5bdad95277a9d619d20da9bf121eb`. The reviewed matmul branch is the
two-commit chain:

```text
17b2b3c
├─ 62eb882  test: construct immutable row reduction descriptors
└─ 6ee412a751fe684c5977828f2d526e9c28d3e787
   └─ d7551176ded1db74c3f185d443f1397a83029bb0
```

Patch IDs are `852a87de71cf1fcebcff8acb21eacae424cc7f39` and
`978f53fa746ecb48047407e4497551589676f28a`. Neither patch is present on
`62eb882`. Preserve the original candidate branch and primary checkout.

## Exact replay

After RowReductionV3 acceptance, create a new worktree/branch from `62eb882`,
then replay in order with `-x`:

```bash
git worktree add -b feature/structured-matmul-v4-replay \
  worktrees/pypto-structured-matmul-v4-replay \
  62eb88251df5bdad95277a9d619d20da9bf121eb

git cherry-pick -x 6ee412a751fe684c5977828f2d526e9c28d3e787
# resolve the one documented fixture conflict, then continue
test "$(git rev-parse HEAD^{tree})" = \
  ec921f0dddec30d4f40d0aaaf6bf06f2615450a2

git cherry-pick -x d7551176ded1db74c3f185d443f1397a83029bb0
test "$(git rev-parse HEAD^{tree})" = \
  cd1b51f56619c25a9497058fc11f2e1a91459d01
```

The second commit is expected to apply cleanly.

## Sole conflict

Only `tests/ut/cpp/structured_tensor_ir_build_spec_test.cpp` conflicts. Retain
the complete immutable two-descriptor construction for both negative fixtures:

```cpp
wrong_result_shape.descriptors = DescriptorVector(
    TensorDescriptor({2, 3}, {3, 1}),
    TensorDescriptor({2, 2}, {2, 1}));

wrong_result_stride.descriptors = DescriptorVector(
    TensorDescriptor({2, 3}, {3, 1}),
    TensorDescriptor({2, 1}, {2, 1}));
```

This is the candidate/`theirs` formatting and preserves the semantic fix from
`62eb882`. Never restore indexed mutation of an immutable descriptor vector,
and do not choose the whole current/`ours` file because that drops matmul tests.

## Exact ten-path inventory

1. `docs/en/dev/backend/01-nvidia-target-info.md`
2. `docs/zh/dev/backend/01-nvidia-target-info.md`
3. `src/codegen/nvidia/tensor_ir_codegen.cpp`
4. `src/codegen/nvidia/tensor_ir_codegen.h`
5. `src/compiler/structured_tensor_ir_build_spec.cpp`
6. `src/compiler/structured_tensor_ir_build_spec.h`
7. `tests/ut/compiler/test_structured_compile.py`
8. `tests/ut/cpp/structured_compile_test.cpp`
9. `tests/ut/cpp/structured_tensor_ir_build_spec_test.cpp`
10. `tests/ut/cpp/tensor_ir_codegen_test.cpp`

No public header, Python binding, type stub, CMake registration, wire format,
or gitlink changes.

## Semantic invariants

- Private `StructuredMatmulV4`, entry `pypto_structured_matmul_v4`, projection
  schema 4 and structured TensorIR route.
- One Opaque function, two distinct exact `In` Vars, direct `tensor.matmul`,
  one assignment and direct return.
- Dense static BF16 rank-2 or equal-batch rank-3 inputs; no broadcast, dynamic
  shape, MemRef/TensorView, mutation, `matmul_acc`, or operand reordering.
- Physical shapes/strides remain unchanged. `a_trans`/`b_trans` become explicit
  rank-aware TensorIR transpose views.
- BF16 by BF16 uses FP32 accumulation followed by nearest-even FP32-to-BF16
  conversion, producing exact logical `[M,N]` or `[B,M,N]` output.
- `K % 128 == 0`, `N % 16 == 0`, and `B*M*N*K <= INT64_MAX`.
- Unit output dimensions disappear from normalized scheduling; an all-unit
  result normalizes to `[1]`.
- Power-of-two schedule arity/tiles, bounded tile product, i32 contraction loop,
  and target grid remain fail-closed.
- Grid shape source is output descriptor `2`; ABI is two inputs plus Result0,
  three pointers, zero workspace, fixed launch/loader identity, no fallback.
- V1/V2/V3 sources and projection goldens remain byte-identical.

## Post-replay ladder

Before builds, require a clean worktree, `git diff --check`, the exact ten-path
set, and the two expected tree hashes above. Build backend-OFF and backend-ON
sequentially in fresh directories with explicit job limits and
`.claude/skills/testing/load-env.sh` sourced.

For both products run the three focused native tests, full CTest inventory, and
only that product's `tests/ut/compiler/test_structured_compile.py`. OFF must
remain disabled at the compiled-backend boundary. ON must embed the new PyPTO
revision plus TensorIR `1dcb38c...`, CUDA Tile `af241704...`, and LLVM
`57109bef...`.

The ON producer matrix contains five cases: rank-2 `a_trans`; rank-3 decode;
rank-3 `a_trans`; rank-3 `b_trans`; and rank-3 batched `a_trans+b_trans`. Every
Artifact must have entry `pypto_structured_matmul_v4`, descriptor-2 static grid,
three pointers, exact grid/tile, zero workspace, nonempty self-hashed SM120
Cubin, canonical BuildSpec/Artifact round trips, and `fallback_used=false`.
Repeat all five in a second CUDA-hidden process and require byte-identical
source, BuildSpec, Artifact, Cubin, and ABI records. Audit DSO RPATH/RUNPATH,
dependencies, origin, CUDA initialization, runtime mappings, and compute PIDs.

These checks establish source/build/host-compiler/Cubin evidence only. GPU
load/launch, numerical correctness, lifecycle, profiling, and performance use
separate later gates.
