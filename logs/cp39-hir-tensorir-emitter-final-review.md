# CP-0039 HIR-to-TensorIR emitter final review

- Review time (UTC): `2026-08-25T09:18:32Z`
- Reviewer canonical task: `/root/nvidia_frontend_emitter_review`
- Review mode: independent source, test, CMake, and retained-evidence audit
- Decision: **GO** for the exact compile-free emitter commit
- Findings: **P0 = 0, P1 = 0, P2 = 2 non-blocking observations**

## Scope and method

This review binds the committed PyPTO HIR-to-TensorIR emitter to its exact
five-file source boundary and to fresh isolated backend-OFF and backend-ON
configure, target-build, and focused CTest evidence.

Read-only checks used `git status`, `git show`, `git diff-tree`,
`git diff --check`, Git blob identity, `sha256sum`, `stat`, CMake cache inspection,
canonical run/preflight JSON inspection, CTest JUnit inspection, and retained
`LastTest.log` inspection. The reviewer did not edit PyPTO, run a build or test,
invoke TensorIR/CUDA Tile, initialize CUDA, or use a GPU. This review log is the
only file written by the reviewer.

## Committed PyPTO identity

- Parent commit: `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`
- Commit: `07ab9ea1feb5f5cc5557c7b7c67e7ad33d15974e`
- Tree: `c02d8ccd03b40193c5547cb9c4956c29a5ed92af`
- Subject: `codegen: emit canonical TensorIR from static add HIR`
- Commit time: `2026-08-25 17:10:33 +0800`
- Exact five-file patch bytes: `62782`
- Exact five-file patch SHA-256:
  `8d29ae623f524526719ae3a0408c5abce073fa653bd8abf2078d62f41fadb29b`

The committed PyPTO checkout is clean at this identity. `git diff --check`
passes, and `git diff-tree` reports exactly three additions and two
modifications:

1. modified `CMakeLists.txt`;
2. added `src/codegen/nvidia/tensor_ir_codegen.cpp`;
3. added `src/codegen/nvidia/tensor_ir_codegen.h`;
4. modified `tests/ut/cpp/CMakeLists.txt`;
5. added `tests/ut/cpp/tensor_ir_codegen_test.cpp`.

## Five-file byte and blob identity

| Path | Bytes | SHA-256 | Git blob |
| --- | ---: | --- | --- |
| `CMakeLists.txt` | 17,829 | `a7994e3d007ab10c7348c6db9263e71e3cfbec4d011efe2ecf7ddc162d35d6ef` | `9e00bf83b2335098e2c632dbea27d9d0388866ca` |
| `src/codegen/nvidia/tensor_ir_codegen.cpp` | 19,628 | `3aced888175b74c18db641cc776f560a1710b624dd148f4eaf1ea9339186eed1` | `68ec7bc9772bce0158350a0ba7e0a1b734cb5071` |
| `src/codegen/nvidia/tensor_ir_codegen.h` | 5,159 | `4c08400186071b07ebbea7b01e30b9837529adc6ced00bb4761eb361f8374309` | `b2f2ef9293c8ea35efcbc47b09207d5ea4b283b8` |
| `tests/ut/cpp/CMakeLists.txt` | 8,263 | `4404ab7f567b979b8f2969db9875efbe029ea529715c66454bd7fb831306647b` | `2c467868373ee15c8c71c6cd7086b792aef57933` |
| `tests/ut/cpp/tensor_ir_codegen_test.cpp` | 34,470 | `e0357652a8e79247b9e4f316864f9f514bbd9ed71bb66da437a8564535c299cb` | `e73c18c34c284299065704aca464a18fcc78d4cb` |

The worktree bytes of all five paths equal these committed bytes.

## Accepted implementation contract

The new surface remains internal under `src/codegen/nvidia`; it adds no public
C++ header, Python binding, type stub, package export, TensorIR type, CUDA Tile
type, framework dependency, or second installed compiler product.

`EmitTensorIr(const ir::ProgramPtr&)` accepts only this bounded HIR form:

- one non-null Program function binding;
- matching `GlobalVar` and bound `Function` names, as required by Program IR;
- one non-runtime-bound `Opaque` function with no level, role, or attributes;
- exactly two distinct exact-`Var` parameters, both direction `In`;
- exactly one declared result;
- plain `TensorType` inputs/result with no `MemRef` or `TensorView`;
- fully static positive rank 1 through 32;
- `INDEX`-typed constant extents whose dense element count fits signed 64-bit;
- only BF16 or FP32, with identical input/call/assignment/result shapes and
  dtypes; and
- exactly `SeqStmts([AssignStmt(result, tensor.add(param0, param1)),
  ReturnStmt(result)])`, allowing reversed distinct operands but rejecting a
  repeated operand, nested expression, side effect, control flow, extra
  statement, attribute, or kwarg.

Canonical source generation is detached from HIR cosmetic names and
process-local identities. It uses fixed ASCII names
`@pypto_vector_add`, `%arg0`, `%arg1`, and `%result0`; never uses `UniqueId`, a
pointer value, or unordered-container iteration in emitted bytes; preserves the
validated operand order; uses locale-independent `to_chars`; emits exactly one
terminal newline; and enforces the 16 MiB source bound.

The TensorIR spelling and layout policy are exact:

- PyPTO FP32 metadata / TensorIR source: `fp32` / `f32`;
- PyPTO BF16 metadata / TensorIR source: `bfloat16` / `bf16`;
- rank 1 has logical dense stride `[1]`, omits the explicit stride attribute,
  and prints a singleton result without parentheses; and
- static rank 2 or greater emits row-major dense
  `nv_tensor_ir.stride = "(...)"` on both inputs and the result signature and
  therefore parenthesizes the attributed singleton result.

The returned metadata is pointer-free and compiler-neutral. Its constructors
are private to the emitter; roles and ordinals are exactly `Input0`, `Input1`,
and `Result0`; dtype code/name, shape, logical dense strides, contiguity, and
explicit-stride state are checked for consistency; and internal construction
failures use `InternalError`, while unsupported input HIR uses bounded
source-located `ValueError` diagnostics.

## Test matrix review

The native test freezes exact FP32 rank-2, BF16 rank-1, and requested FP32
rank-1 `[128]` source bytes and metadata. It also covers:

- deterministic bytes across different function/parameter/result hints;
- reversed distinct operand identity;
- hostile global locale;
- null/empty/multi-function and malformed global bindings;
- wrong function kind, direction, counts, attributes, runtime binding, body
  shape, statement kind, return form, null target/value, and Var subclasses;
- mismatched input, declared, call, and assignment tensor types;
- unsupported dtype, dynamic shape, non-`INDEX` extent, `TensorView`, and
  `MemRef`;
- nested/non-parameter/repeated operands, wrong operator, attrs, kwargs, and
  arity; and
- empty/oversized rank, non-positive extents, signed-64-bit dense-size
  overflow, cosmetic-name isolation, and bounded diagnostics.

The test also proves that the metadata/module constructors are unavailable to
unrelated callers, preserving the single validated construction path.

## Formal isolated run evidence

All six retained process records are schema-2, status `exited`, return code 0.
Each referenced preflight hash equals the recomputed preflight SHA-256. All six
preflights report `ok=true`, zero failures, no protected heavy process, no
protected NVIDIA compute/runtime-mapping PID, no unreadable protected map, no
waiver, and a successful NVIDIA-compute audit. No external process was
signalled.

| Phase | Backend | Run ID | Process bytes / SHA-256 | Preflight bytes / SHA-256 |
| --- | --- | --- | --- | --- |
| Configure | OFF | `pypto-20260825T091116Z-943534-89b3d0` | 2,472 / `556e07c58ec0cffd37419c83acba62234078b40cca0b6679cc86a48ff08ab481` | 6,667 / `cd699c3b15f0eb1044197d2438349c5482b593c9b07ec47d15da54f82fa9bbd2` |
| Configure | ON | `pypto-20260825T091116Z-943536-d963c3` | 2,470 / `8d9e2fc914ff0f19df6eb8a908058be9b0d82f3ef779985a805c0bf3ccbfdb47` | 6,667 / `fc85ecca938a95df84a54e08eaff08182ae3e9896674608e135c8c82c953794b` |
| Target build | OFF | `pypto-20260825T091141Z-944924-ac0540` | 1,833 / `b2bc4eae2848b17ec697538557b766c4ba14f33d3d00d11bcf6bfda1ca597353` | 5,363 / `9393ebe9d50d73ced47bb152514078fe71dea676885d7f16fa452bb3247f6ef8` |
| Target build | ON | `pypto-20260825T091141Z-944928-dfa80f` | 1,832 / `46f3e4f2eb40488aa3c4ae1aef4abea0c11e4435b9953cf6a407aef203666c12` | 5,363 / `dad8b4e07e427c2e4f1297bd6008265a957865a00cdf95d842c519f4deb49bcd` |
| Focused CTest | OFF | `pypto-20260825T091343Z-958355-5ef5a9` | 1,967 / `8ddda9485445bea73542be9c8efe4bdc13722e0d678c88410edbe26e42991518` | 138,400 / `37645df5a9cd9eb725728b3873ee9d06e062e0fb54cfd3d9614ef2451d1614bc` |
| Focused CTest | ON | `pypto-20260825T091556Z-961868-8ee40b` | 1,965 / `b31e07f2a86828f7429ba8bdcd47ad5f8fe20ae62469c2519b47108b2db1cb77` | 4,666 / `67553f2169d9dce05cae93e14b8f17ffde66ce59698ff81ab0540746ed3bfa27` |

The configure records select Debug/Ninja builds with `BUILD_TESTING=ON` and
the exact project-local Python environment. Their caches independently record:

- OFF: `PYPTO_ENABLE_NVIDIA_BACKEND=OFF`, cache SHA-256
  `710a58dd33c2c33582f2b1e89a3625eca880ea596387b6882c752528e38e6689`;
- ON: `PYPTO_ENABLE_NVIDIA_BACKEND=ON`, cache SHA-256
  `e281032e97381427deadf59ff16480cfcb2fb0db904ddff7a7ca705702557639`.

Both target builds compile and link `pypto_tensor_ir_codegen_test` from the
committed checkout:

- OFF executable: 241,346,336 bytes, mode `0755`, SHA-256
  `0c647a14342f6932560e31986797aa5f1618eab63c3cceb17883ce8220ac04dd`;
- ON executable: 560,616,816 bytes, mode `0755`, SHA-256
  `e5441706f7a1dbf56411eb56f3a672ecd1ed9c0d41a86a21adbe8e3b1165aebb`.

Focused CTest results are:

| Backend | Tests | Failures | Disabled | Skipped | Time | JUnit SHA-256 | LastTest.log SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| OFF | 1 | 0 | 0 | 0 | 0.187414 s | `31a02211c020e6d2c91339fdd2e6a6934b14a61ecf31f32782716ac7b0a1dff3` | `702185d665ec933dd6ff2258eeb1ecf6f0a8fdb1207aae1662690dd32dbf10aa` |
| ON | 1 | 0 | 0 | 0 | 0.215575 s | `721e76033edad6f6af1744fecbb3445304921025b7bd514131cf1d8fb6bb6583` | `392916b6b14b785b37107ffd98c61d6862080c7dce3862c432a0ae5c0c3683a4` |

Each JUnit file is 371 bytes and each `LastTest.log` is 849 bytes. The ON target
links the private bridge/dependency objects only when the backend is enabled;
the test translation unit retains normal PyPTO exception support. The OFF
target proves that emitter functionality does not require the private NVIDIA
backend.

## Findings

### P0

None.

### P1

None.

### P2 — non-blocking

1. `MakeDenseStrides` and `MakeDenseStridesForMetadata` duplicate the same
   checked dense-stride arithmetic solely to preserve user-error versus
   internal-error classification. A later cleanup may share a result-returning
   arithmetic helper while keeping the two call-site classifications.
2. The explicit `ObjectKind::Var` checks after successful `As<Var>` on call
   operands and return values are redundant because `As<Var>` already performs
   exact-kind matching. The direct checks on `Function::params_` and
   `AssignStmt::var_` remain necessary because those fields are statically
   typed as `VarPtr` and may hold subclasses.

Neither observation changes accepted behavior, determinism, safety, ABI, or
test coverage and neither blocks the commit.

## Strict scope and non-claims

CP-0039 accepts only a compile-free internal translation from the exact static
tensor-add HIR subset above to deterministic TensorIR text plus detached tensor
input/result metadata, together with backend-OFF and backend-ON build/link and
focused native-test evidence.

It does **not** establish:

- TensorIR parsing or verification of the emitted bytes;
- TensorIR analysis, tile selection, CUDA Tile lowering, TileIR, `tileiras`, or
  Cubin production;
- `CompileRequest`, `KernelBuildSpec`, callable/argument/result/mutation or
  specialization digests, Artifact construction/cache identity, or fallback
  behavior;
- `NvidiaExecutable`, CUDA module/function load, context/current-stream launch,
  synchronization, unload, numerical output, or real-SM120 frontend
  correctness;
- a public C++ or Python compilation API, Python binding/type stub, full
  NVIDIA pass pipeline, multi-function routing, dynamic shape, broadcast,
  non-contiguous view, fused pointwise, reduction, matmul, attention, or GDN;
- CUDA Graph capture/replay, TorchInductor, SGLang, Qwen, strict coverage,
  profiling, or performance.

In particular, the backend-ON test proves that the compile-free emitter target
builds, links, and executes in the private-backend product. It does not invoke
the TensorIR producer or a GPU and cannot be promoted to compilation or runtime
evidence.

## Decision

**GO**, with **P0 = 0, P1 = 0, P2 = 2 non-blocking observations**, for exact
PyPTO commit `07ab9ea1feb5f5cc5557c7b7c67e7ad33d15974e`, tree
`c02d8ccd03b40193c5547cb9c4956c29a5ed92af`, the five-file scope and hashes
recorded above, and only the strict compile-free scope stated here.

The next acceptance gate must derive versioned precompile identities and the
exact expected callable ABI, feed these emitted bytes through the private
TensorIR/CUDA Tile producer, and separately prove Artifact and real-SM120
numerical execution. None of those later gates is implied by this review.
