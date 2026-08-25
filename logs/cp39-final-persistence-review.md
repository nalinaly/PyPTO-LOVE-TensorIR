# CP-0039 final persistence review

- Review time (UTC): `2026-08-25T09:28:37Z`
- Reviewer canonical task: `/root/cp36_persistence_review`
- Review mode: independent read-only persistence/source/evidence audit
- Decision: **GO** for the exact frozen pre-lineage CP-0039 candidate
- Persistence findings: **P0 = 0, P1 = 0, P2 = 0**
- Incorporated source-review observations: **P2 = 2, non-blocking**

## Scope and method

This review joined the exact 8-tracked-plus-3-new persistence candidate to the
committed PyPTO emitter, its exact five-file diff and bytes, fresh isolated
backend-OFF/backend-ON configure/build/CTest evidence, build metadata, the
committed independent review, current source/product attribution, locks,
handover, and the compiler-preparation resume boundary.

Read-only checks used `git status`, `git diff`, `git show`, `git rev-parse`,
`git submodule status`, Git blob inspection, `sha256sum`, `stat`, CMake cache,
Ninja log, compile-command, binary, CTest XML/LastTest.log and retained
preflight/process inspection. No source, control, persistence, build, run,
product, project or upstream file was modified by this reviewer. This ignored
review log is the only file written by the reviewer and is outside the formal
8+3 persistence boundary.

## Committed emitter and review identity

### PyPTO emitter

- Parent: `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`
- Commit: `07ab9ea1feb5f5cc5557c7b7c67e7ad33d15974e`
- Tree: `c02d8ccd03b40193c5547cb9c4956c29a5ed92af`
- Subject: `codegen: emit canonical TensorIR from static add HIR`
- Exact patch bytes: `62782`
- Exact patch SHA-256:
  `8d29ae623f524526719ae3a0408c5abce073fa653bd8abf2078d62f41fadb29b`

The committed diff is exactly five paths: three additions and two
modifications. `git diff --check` passes and current worktree bytes equal the
committed blobs.

| Path | Bytes | SHA-256 | Git blob |
| --- | ---: | --- | --- |
| `CMakeLists.txt` | 17,829 | `a7994e3d007ab10c7348c6db9263e71e3cfbec4d011efe2ecf7ddc162d35d6ef` | `9e00bf83b2335098e2c632dbea27d9d0388866ca` |
| `src/codegen/nvidia/tensor_ir_codegen.cpp` | 19,628 | `3aced888175b74c18db641cc776f560a1710b624dd148f4eaf1ea9339186eed1` | `68ec7bc9772bce0158350a0ba7e0a1b734cb5071` |
| `src/codegen/nvidia/tensor_ir_codegen.h` | 5,159 | `4c08400186071b07ebbea7b01e30b9837529adc6ced00bb4761eb361f8374309` | `b2f2ef9293c8ea35efcbc47b09207d5ea4b283b8` |
| `tests/ut/cpp/CMakeLists.txt` | 8,263 | `4404ab7f567b979b8f2969db9875efbe029ea529715c66454bd7fb831306647b` | `2c467868373ee15c8c71c6cd7086b792aef57933` |
| `tests/ut/cpp/tensor_ir_codegen_test.cpp` | 34,470 | `e0357652a8e79247b9e4f316864f9f514bbd9ed71bb66da437a8564535c299cb` | `e73c18c34c284299065704aca464a18fcc78d4cb` |

### Independent review

- Root review commit: `95e1b9602825a7f01f8f00d03d1521e52dd3caa9`
- Root review tree: `ff2c84b4634eb645c30d421a35394e9e7561633e`
- Report: `logs/cp39-hir-tensorir-emitter-final-review.md`
- Report bytes: `12626`
- Report SHA-256:
  `112f1c76cf205026f7e798d944e102c9820c7a39b6fdc40b50df767ee06293b6`
- Decision: GO, P0 = 0, P1 = 0, P2 = 2 non-blocking observations

The review commit adds exactly that report. Its two P2 observations are:

1. the user-error and internal-error paths duplicate checked dense-stride
   arithmetic and may later share a result-returning helper; and
2. exact-kind checks after successful `As<Var>` on call/return values are
   redundant, while direct checks on statically typed parameter/assignment
   fields remain necessary.

Neither observation changes accepted behavior, determinism, safety, ABI or test
coverage and neither blocks CP-0039.

## Accepted compile-free emitter contract

The implementation is internal C++ under `src/codegen/nvidia`. It adds no
public installed C++ header, Python binding/type stub, TensorIR or CUDA Tile type
at the seam, framework dependency, or second compiler product.

`TensorIrModule EmitTensorIr(const ir::ProgramPtr&)` accepts only one exact
normalized static tensor-add program:

- one non-null Program function binding with matching GlobalVar/Function name;
- one non-runtime-bound Opaque function with no level, role or attributes;
- two distinct exact-Var `In` parameters and one return;
- plain static contiguous rank 1 through 32 TensorTypes, no MemRef/TensorView;
- positive INDEX extents with signed-64-bit-safe dense strides;
- identical FP32 or BF16 input/call/assignment/result types and shapes; and
- exactly `SeqStmts[AssignStmt(tensor.add), ReturnStmt(result)]`, allowing
  reversed distinct operands while rejecting repeats, nesting, side effects,
  control flow, extra statements, attributes and kwargs.

Canonical source uses only `@pypto_vector_add`, `%arg0`, `%arg1`, `%result0`,
preserves semantic operand order, uses locale-independent `std::to_chars`, has
one terminal newline, and excludes cosmetic HIR names, UniqueId, pointers,
spans, comments, unordered iteration and ambient environment.

Rank one omits explicit TensorIR stride attributes and singleton-result
parentheses while metadata records dense stride `[1]` and
`has_explicit_strides=false`. Rank two or greater emits explicit dense strides.
Metadata schema is v1 in exact `Input0, Input1, Result0` order. Private
construction prevents detached source/metadata fabrication.

The accepted gate is deliberately compile-free. It invokes no TensorIR parser,
producer, CUDA Tile, assembler, CUDA runtime or GPU.

## Fresh backend-OFF evidence

- Configure run: `pypto-20260825T091116Z-943534-89b3d0`, rc `0`
  - preflight SHA: `cd699c3b15f0eb1044197d2438349c5482b593c9b07ec47d15da54f82fa9bbd2`
  - process SHA: `556e07c58ec0cffd37419c83acba62234078b40cca0b6679cc86a48ff08ab481`
- Target-build run: `pypto-20260825T091141Z-944924-ac0540`, rc `0`
  - preflight SHA: `9393ebe9d50d73ced47bb152514078fe71dea676885d7f16fa452bb3247f6ef8`
  - process SHA: `b2bc4eae2848b17ec697538557b766c4ba14f33d3d00d11bcf6bfda1ca597353`
- Focused CTest run: `pypto-20260825T091343Z-958355-5ef5a9`, rc `0`, `1/1`
  - preflight SHA: `37645df5a9cd9eb725728b3873ee9d06e062e0fb54cfd3d9614ef2451d1614bc`
  - process SHA: `8ddda9485445bea73542be9c8efe4bdc13722e0d678c88410edbe26e42991518`
  - JUnit SHA: `31a02211c020e6d2c91339fdd2e6a6934b14a61ecf31f32782716ac7b0a1dff3`

Build metadata:

- `PYPTO_ENABLE_NVIDIA_BACKEND=OFF`, `BUILD_TESTING=ON`, Debug/Ninja;
- CMakeCache SHA: `710a58dd33c2c33582f2b1e89a3625eca880ea596387b6882c752528e38e6689`;
- Ninja log: 263 records / 263 unique outputs, SHA
  `673a0c3efb3bd9d275630c5cc820b3628188aeb94f7fb47f23f3017806f59d23`;
- compile commands SHA:
  `50230c7e5581a90e66824a9826257f355095091dafc2f42634dee0738d95e537`;
- test binary: 241,346,336 bytes, mode `0755`, SHA
  `0c647a14342f6932560e31986797aa5f1618eab63c3cceb17883ce8220ac04dd`;
- JUnit: 371 bytes, one test, zero failures/errors/skips;
- LastTest.log: 849 bytes, SHA
  `702185d665ec933dd6ff2258eeb1ecf6f0a8fdb1207aae1662690dd32dbf10aa`.

## Fresh backend-ON evidence

- Configure run: `pypto-20260825T091116Z-943536-d963c3`, rc `0`
  - preflight SHA: `fc85ecca938a95df84a54e08eaff08182ae3e9896674608e135c8c82c953794b`
  - process SHA: `8d9e2fc914ff0f19df6eb8a908058be9b0d82f3ef779985a805c0bf3ccbfdb47`
- Target-build run: `pypto-20260825T091141Z-944928-dfa80f`, rc `0`
  - preflight SHA: `dad8b4e07e427c2e4f1297bd6008265a957865a00cdf95d842c519f4deb49bcd`
  - process SHA: `46f3e4f2eb40488aa3c4ae1aef4abea0c11e4435b9953cf6a407aef203666c12`
- Focused CTest run: `pypto-20260825T091556Z-961868-8ee40b`, rc `0`, `1/1`
  - preflight SHA: `67553f2169d9dce05cae93e14b8f17ffde66ce59698ff81ab0540746ed3bfa27`
  - process SHA: `b31e07f2a86828f7429ba8bdcd47ad5f8fe20ae62469c2519b47108b2db1cb77`
  - JUnit SHA: `721e76033edad6f6af1744fecbb3445304921025b7bd514131cf1d8fb6bb6583`

Build metadata:

- `PYPTO_ENABLE_NVIDIA_BACKEND=ON`, `BUILD_TESTING=ON`, Debug/Ninja;
- compile rows embed exact PyPTO revision `07ab9ea1feb5f5cc5557c7b7c67e7ad33d15974e`;
- CMakeCache SHA: `e281032e97381427deadf59ff16480cfcb2fb0db904ddff7a7ca705702557639`;
- Ninja log: 2,720 records / 2,720 unique outputs, SHA
  `3787657356d2ba9896cbf9188424a8e1e2028cf6951ebfc242007b15c4a605f9`;
- compile commands SHA:
  `30a6df1244ed3e6f1a44a73b36f52097e83c0537e4ecd9c939820b1013d78f3e`;
- test binary: 560,616,816 bytes, mode `0755`, SHA
  `e5441706f7a1dbf56411eb56f3a672ecd1ed9c0d41a86a21adbe8e3b1165aebb`;
- JUnit: 371 bytes, one test, zero failures/errors/skips;
- LastTest.log: 849 bytes, SHA
  `392916b6b14b785b37107ffd98c61d6862080c7dce3862c432a0ae5c0c3683a4`.

Both build graphs compile the emitter and focused test through the normal
compiler object boundary. OFF proves the emitter itself does not require the
private NVIDIA backend; ON proves it integrates with that build composition.
Neither target build is a newly accepted full DSO product.

All six process records are schema-2, exited with rc 0, and join their exact
preflight hashes. Their preflights are green with no protected heavy process,
NVIDIA compute/runtime-mapping PID, unreadable protected map or waiver. No
external process was signalled.

## Accepted runtime product attribution remains historical

The current PyPTO source checkout is `07ab9ea`, but CP-0039 does not replace the
accepted CP-0038 runtime product. The accepted backend-ON/OFF DSO hashes remain:

- ON: `15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c`
- OFF: `32c2dea03e13ea49937df239ed1e7bc2b8a931594c41dcfc0cc6de8375464109`

Both were built from exact PyPTO source
`206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`, not from `07ab9ea`.
VERSIONS.lock records `pypto.nvidia_dso.source_commit=206447c...`;
WORKSPACE.lock records both the DSO source and accepted real-launch source as
`206447c...`; VERSIONS.txt and HANDOVER state the same boundary. `07ab9ea` has
only targeted emitter build/test products and no newly accepted DSO.

## Resolved ignored-source-shadow finding

The persistence audit found three ignored CPython cache files under the
production PyPTO package. They were not part of the committed emitter or native
build evidence, but they violated the established zero-shadow workspace
boundary. All three were moved intact, not deleted, to
`runs/quarantine/cp39-pypto-import-shadows-20260825T0922Z`:

- `__init__.cpython-314.pyc`:
  `566c71e27ce7091387e1c297f2a916960a464c04eebc3cb6fcbd132e2f558fd1`;
- `compile_profiling.cpython-314.pyc`:
  `b21ad561b3edda246ea38e989f7897487792c1a47b08cd2cee1d4dcef6eb4b6e`;
- `compiler/__init__.cpython-314.pyc`:
  `84ee196fb69e751d472bdffb812279f7eded4b287904cbda7b26ee13a8e4dff2`.

Current ignored-file enumeration under `projects/pypto/python/pypto` is empty;
the PyPTO tracked/untracked worktree is clean. The recovery objects do not enter
accepted source, product or run identity.

## Exact persistence boundary

The reviewed candidate contains exactly eight tracked modifications and three
non-ignored untracked files:

1. `GOAL.md`
2. `PLAN.md`
3. `TODO.md`
4. `CHECKPOINT.md`
5. `HANDOVER.md`
6. `VERSIONS.lock`
7. `VERSIONS.txt`
8. `WORKSPACE.lock`
9. `state/checkpoints/CP-0039.md`
10. `state/evidence/EV-0052.json`
11. `state/bitlessons/BL-0057.md`

`git diff --check` passes. No source, test, build, run, control, report, product,
project or upstream path is part of this persistence candidate. PyPTO, all six
PyPTO gitlinks, pypto-kernels, pypto-framework-plugins, PyTorch, SGLang and
Triton are clean at the locked identities.

GOAL, PLAN revision 31, TODO, CHECKPOINT, HANDOVER, both version files and
WORKSPACE.lock consistently identify CP-0039, EV-0052, current source
`07ab9ea`, compile-free-only acceptance, historical runtime-DSO attribution to
`206447c`, and the compiler-preparation resume.

The frozen pre-lineage SHA-256 values are:

- `GOAL.md`: `c8d6681dd33db59c6a1f470baab46a223401d66fbc9003d268593f4608a55233`
- `PLAN.md`: `0dbf9a20db68edc3a1fe595043cc3e8d69ffbe46c6b434880c2855117e4dee7d`
- `TODO.md`: `399b373394071a98a2a4496b6483b3dab05bb743cc957095a99d83dd518483fd`
- `CHECKPOINT.md`: `b7f0d564c97bacb8bc591f07163800ca25782f33e4cb1764ce52cfecb624987d`
- `HANDOVER.md`: `d3530dfe16954d766ad4fe18e88239e51e522847a9d42ecf6922d30b2ad080b0`
- `VERSIONS.lock`: `f7f8868e1a87a1ada5861f5bbaa9b5602e4d29cd0f96d6098435c47f2617edde`
- `VERSIONS.txt`: `0ea0c9096ae8ba5f1e7cac62e2414184e0a218d932ca39c04fe2fcb58623ec4f`
- `WORKSPACE.lock`: `ba20ca5e3167d034317ba657f7895b0e17aa06dce9cfea600943bcbd3e7b7d6d`
- `state/checkpoints/CP-0039.md`:
  `e2a3aacb315a2524f938da8f6eb55d4f8e7f43c7707dda8c034a42066d6f9d6a`
- `state/evidence/EV-0052.json` before review lineage:
  `22edf819e27ecbab10ef76234b2a14d240430060576e2125e360344b0acddbbf`
- `state/bitlessons/BL-0057.md`:
  `97aa0156e9332965ae9590ed003b38344d815acc04a5da2e83a535a50f96633f`

## Accepted scope and non-claims

CP-0039 accepts only:

- compile-free deterministic validation of the exact static contiguous FP32/
  BF16 `tensor.add` HIR form;
- deterministic canonical TensorIR text emission;
- compiler-neutral static input/result metadata; and
- backend-OFF and backend-ON configure/target-build/focused-test integration.

It does not accept a public Python compilation API, TensorIR parsing/analysis/
tile selection, callable-ABI or specialization preparation, KernelBuildSpec,
Artifact/Cubin production, NvidiaExecutable launch, frontend numerical GPU
correctness, broader pointwise/reduction/matmul/operator families, CUDA Graph,
TorchInductor, SGLang, Qwen, strict coverage, profiling or performance.

The next gate is the compiler-owned preparation/compile API: consume the
internal TensorIrModule and one explicit schedule; derive final versioned
argument/result/mutation/static/symbolic and callable identities; return the
final KernelBuildSpec plus strict Artifact without placeholders; bind it under
the single `pypto.compiler` surface; then replace handwritten TensorIR with
HIR-authored vector add and prove real-SM120 correctness. The legacy Ascend pass
pipeline and a second public TensorIR product remain forbidden.

## Permitted lineage-only closure

The sole permitted post-review mutation is adding top-level EV-0052
`review_lineage`. It must bind the CP-0039 SHA, EV-0052 pre-lineage SHA,
BL-0057 SHA, reviewer task, and this report path/bytes/hash, together with the
committed emitter-review identity. Any byte change outside that one value, or
any change to the other ten persistence files, invalidates this decision and
requires re-review. After insertion, JSON parsing, exact 8+3 status and
`git diff --check` must be checked again.

## Decision

**GO** for the exact frozen pre-lineage 8+3 CP-0039 persistence candidate.
Persistence findings are **P0 = 0, P1 = 0, P2 = 0**. The incorporated source
review retains its two explicitly non-blocking P2 observations described above.

Artifact integration, public binding and frontend real-SM120 correctness remain
pending.
