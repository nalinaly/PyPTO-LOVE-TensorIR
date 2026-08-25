# PyPTO NvidiaExecutable `206447c` final evidence review

## Decision

**GO to commit CP-0036.** P0: 0. P1: 0. P2: 0.

This decision accepts only the generic CUDA parameter-ABI/enumeration repair,
its fresh CPU/fake-driver/static-product validation, CUDA Tile logical-host
launch semantics, and the immutable root control manifest v3. It is not a
successful GPU result and does not accept completed real prewarm, a PyPTO CUDA
kernel launch, numerical correctness, CUDA Graph, frontend lowering, operators,
framework integration, model correctness/coverage, profiling, or performance.

The root worktree is intentionally dirty with the CP-0036 persistence draft at
review time. The checkpoint commit must include the reviewed persistence files
and this report and leave the root clean before the v3 GPU-smoke controller can
pass its fail-closed manifest gate.

## Exact review subject

- PyPTO commit:
  `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`
- PyPTO tree:
  `e0357daaefa74dbf676550015e60701996c400fb`
- PyPTO parent:
  `3431a944691f8295d3a55a4e222dadcfe34253f5`
- PyPTO worktree: clean on `feature/nvidia-sm120`
- TensorIR: `1dcb38c20e53d07c97d3781cae538e33901bae30`, clean
- CUDA Tile: `af2417041cc939b87ef56d92cfdcf61737c5457e`
- LLVM/MLIR: `57109befac92811d2253109242ca6fa69c961fb2`
- Root Layer A3 commit:
  `c71f32bd415a973a2a7756ecc9b1ae59f30df219`
- Root Layer A3 tree:
  `89dc321ff8f6913b826f70f09c07d5c045a67790`
- Root manifest-only Layer B3 commit:
  `3de4cf702662cbaf948c6429acf269fee16a491e`
- Root Layer B3 tree:
  `c72970aa075e5769aad945599947b1764440f2c6`
- B3 is the direct child of A3; A3 is an ancestor of current root HEAD.
- Reviewed CP draft SHA-256:
  `e36c475f8d3a97a0721e0b169833ea13895ae7868c3c7699331655877c513ce9`
- Reviewed EV draft SHA-256:
  `54d6a82ed69272a6621ab65750a11784ef4ac0b92b8d0bd61a832e92e4168384`

The cumulative repair from PyPTO `6361f11` to `206447c` changes exactly the
two executable documents, `src/compiler/nvidia_artifact_compiler.cpp`,
`src/runtime/nvidia/executable.cpp`, and
`tests/ut/cpp/nvidia_executable_test.cpp`. `git diff --check` passes.

## Implementation assessment

The repair separates the three relevant orderings correctly:

- validated Cubin `KPARAM` ordinals remain the authority for signature and
  Artifact argument order;
- live Driver parameter information is checked as the complete expected width
  multiset plus bounded, non-overlapping ranges, without assuming Driver
  enumeration or offset order;
- the host launch pointer array remains in signature order and is never
  reordered by Driver offsets.

Dynamic sizes and strides use the producer's existing four-byte `int32` ABI,
with size range `1..INT32_MAX`, stride range `0..INT32_MAX`, and exact native
four-byte values copied into zero-initialized stable eight-byte host slots. The
tests include the retained Cubin-shaped reverse, non-palindromic 12-parameter
sequence; wrong width multisets; overlap, overflow and out-of-frame ranges;
boundary values; invalid dense dimensions; failure latching; and
signature-ordered launch values.

CUDA Tile's reviewed host contract remains grid-driven with logical block
`[1,1,1]` and zero dynamic shared memory. Cubin `EIATTR_REQNTID=[128,1,1]`
is compiler physical metadata; its correlation with four selected worker warps
is explicitly retained as an inference and is not copied into host block
dimensions.

## Retained real-runtime diagnostic

Run `pypto-20260825T052038Z-800777-8e8e83` is correctly recorded only as a
fail-closed diagnostic:

- root `4f2d50089f02dd46836f6c794bba502456ab01e9`;
- PyPTO `6361f110660a77f9a8dc542265575d8f7260b343`;
- return code `1`;
- process SHA-256
  `9085c010791b3d5e8639b12a0e7110105c14bb6c4f331c0fab11255f08c86525`;
- preflight SHA-256
  `5c88ffd0d77e7a42fe674ec6423ac3376f4809ea782a6074225157e8320dc208`;
- gate SHA-256
  `e78e20217a10df8b332568688fd01026267dd8f305a622b24037790d78cea911`;
- start-barrier SHA-256
  `e0849f9f2f511d54014da29c68bdc91fc1a8aee44ce899373551ac8a1437537f`.

The retained CompileRequest, three BuildSpecs, three Artifacts, extracted
static Cubin, and post-run cuobjdump hashes all match EV-0049. The run passed
the recorded admission/release/child isolation boundary, produced all three
serialized Artifacts, and reached the first static module/function prewarm. It
did not reach `prepare_launch`, launch, provisional publication, numerical
comparison, or finalization. Its terminal error text was observed but stderr
was not persisted; CP-0036 and EV-0049 disclose that limitation and treat the
exact failed predicate and reverse-enumeration diagnosis as source/Cubin
inference rather than accepted GPU behavior.

## Focused repair validation

- Focused build `pypto-20260825T064851Z-866863-27412e`: return code 0;
  preflight `ca9cd09a...0695c`; process `f98745a5...a610`.
- Focused fake-driver test `pypto-20260825T064934Z-868286-7d9f71`:
  return code 0; preflight `83503430...c9a6`; process `bd64d99d...79d2`.
- Stale-assertion runs `064708` (rc 8) and `064727` (rc 211) are explicitly
  excluded from acceptance.

## Fresh product evidence

Both build directories have filesystem birth times after the final PyPTO
commit and were configured separately.

### Backend ON

- Configure `pypto-20260825T065002Z-868546-526995`: rc 0; preflight
  `04f70c95...c750`; process `b7069ff4...6147`.
- Build `pypto-20260825T065050Z-869893-badb83`: rc 0; preflight
  `e9c2c524...b0a5`; process `620f1528...fc64`.
- `.ninja_log`: 2,797 records and 2,797 unique output paths; SHA-256
  `10658966ed898c4159493109d2471fa379b8ccc5c5385f6f56d85bd25dfc46f9`.
- `compile_commands.json` SHA-256
  `350bf133cec3173bae03c5db48a4c0689d4b2309b49615a13bf1c160bb7b6b7d`;
  compile definitions bind PyPTO `206447c`, TensorIR `1dcb38c`, CUDA Tile
  `af241704`, and LLVM `57109bef`.
- CTest `pypto-20260825T065932Z-883420-3d116c`: 9/9, zero failures or skips;
  JUnit SHA-256
  `347bb1eddb9ae2aa0a6937c19bc85c54074fcf7a3399ffd8ee65cd17b7ffed68`.
- Exact-DSO Python `pypto-20260825T070443Z-889280-424570`: 144 total,
  142 passed, 2 skipped; JUnit SHA-256
  `62dd5dbc2c30c4d58028735a35829212bd5bfbebe30931c3efb96a9d83704065`.
- DSO: 780,535,416 bytes; SHA-256
  `15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c`.

### Backend OFF

- Configure `pypto-20260825T065955Z-883614-5098e2`: rc 0; preflight
  `319caca9...0f04`; process `c1a3853a...8a1d`.
- Build `pypto-20260825T070007Z-883835-7cb6d3`: rc 0; preflight
  `a4220b68...c347`; process `f8acd90c...75de`.
- `.ninja_log`: 327 records and 327 unique output paths; SHA-256
  `c818484ff257e43481c6fdf25770ecd2f1631ede263114dee5733005d484b7c7`.
- `compile_commands.json` SHA-256
  `fdd60e06f9ca8afd3c8a230e3f5122b2659affc76c28ba9d09831bf7c170fab4`.
- CTest `pypto-20260825T070412Z-889089-b986bd`: 7/7, zero failures or skips;
  JUnit SHA-256
  `27460bd6e97507c17a6b8381a430d86e1a918af98f4a9b8c8ad0744979d8aa60`.
- Exact-DSO Python `pypto-20260825T070508Z-889497-469338`: 144 total,
  135 passed, 9 skipped; JUnit SHA-256
  `2fcee961c432ef2112c84549e70dbf468471ee20c0c70d3df768e77376a1c789`.
- DSO: 434,646,736 bytes; SHA-256
  `32c2dea03e13ea49937df239ed1e7bc2b8a931594c41dcfc0cc6de8375464109`.

### Product and deterministic-compilation gates

- Product audit `pypto-20260825T070539Z-889610-55d705`: rc 0; preflight
  `c33925e8...bdb8`; process `813309c1...a598`.
- Independent static reinspection confirms each DSO has no RPATH/RUNPATH,
  exactly the five standard runtime dependencies, exactly `PYPTO_CORE_1` and
  `PyInit_pypto_core@@PYPTO_CORE_1` as dynamic definitions, and no matching
  CUDA/TensorIR/NvidiaRuntime/NvidiaExecutable/NvidiaObservation dynamic
  symbol. Compile rows have the exact mutually exclusive ON driver/test and
  OFF stub/test composition recorded in EV-0049.
- Deterministic CPU compilation
  `pypto-20260825T070649Z-890066-2a1ae2`: rc 0; preflight
  `e454fb2d...3230`; process `7b24f6e2...1452`. Its inline predicates bind the
  exact final DSO revision, preserve wrapper ownership markers and empty
  `CUDA_VISIBLE_DEVICES`, reproduce all three expected Cubin size/SHA and ABI
  pairs, observe Torch device count zero and CUDA uninitialized, and exclude
  Triton, SGLang, and FlashInfer. Runs `062710`, `062751`, and `070611` are
  correctly retained as non-acceptance diagnostics.

All accepted configure/build/CTest/Python/audit/CPU-compile process and
preflight hashes in EV-0049 were independently recomputed and match their
files. Every accepted run is `gpu_smoke.requested=false`; the native executable
test uses a fake Driver, the Python gates are compiler/public-construction
tests, the product audit is static, and the deterministic compiler gate proves
CUDA remained uninitialized.

## Root control v3 and persistence consistency

- Manifest v3 path:
  `state/contracts/pypto_nvidia_executable_sm120_v3.json`
- Manifest bytes: 1,569
- Manifest SHA-256:
  `978e873788eb7f3aaeba6473a9b7f8a1bcd827fe201d89cb781927f538c9b6e3`
- Schema/kind: `3` / `pypto-nvidia-executable-sm120-controls-v3`
- Its implementation commit/tree equal A3 exactly.
- All seven live control files independently match the manifest's exact byte
  sizes, executable modes, and SHA-256 values.
- Runner SHA-256 remains
  `f22befff45d87097ae42b5725cf33a5e296ed74ff177cd84c2b772be5939abdd`.
- Manifest v1 SHA-256 remains `c609e97a...aa09`; manifest v2 remains
  `63687352...07cd5`; both are byte-unchanged through B3.
- B3 changes only the v3 manifest. A3 changes only the smoke document,
  product contract, and manifest selector.
- Clean post-v3 root run `pypto-20260825T071601Z-892819-67acee`: rc 0;
  process `983cdda7...a223`; preflight `ef376b88...13dd`; JUnit 330 total,
  zero failure/error/skip, SHA-256
  `cbc276bd9a9869f8afb6345e3c3f1ebfa8c3501070a2401ecf5a101bdab7b213`.
  The live-manifest test requires and observed a clean root. Six protected
  heavy processes were present, protected NVIDIA compute/runtime-mapping and
  unreadable-map sets were empty, no coexistence pause occurred, and no
  external signal evidence exists.

`CHECKPOINT.md`, `GOAL.md`, `PLAN.md`, `TODO.md`, `HANDOVER.md`,
`VERSIONS.lock`, `VERSIONS.txt`, and `WORKSPACE.lock` consistently select
CP-0036, EV-0049, PyPTO `206447c`, A3/B3, manifest v3, and the final ON/OFF DSO
hashes. Their accepted scope and non-claims agree with CP-0036/EV-0049.
Canonical JSON validation and root `git diff --check` pass.

## Findings and evidence limits

### P0

None.

### P1

None.

### P2

None.

### Non-blocking evidence limits

1. The retained real-runtime run is a failed diagnostic. Its terminal error
   was not persisted and its exact failed predicate remains inference.
2. No accepted evidence reaches successful real prewarm, current-stream
   launch, explicit real module lifetime completion, provisional publication,
   or numerical comparison.
3. Fresh `206447c` validation is CPU/fake-driver/static-product evidence. It
   does not convert the retained `6361f11` prewarm failure into a successful
   GPU result.
4. Process JSON retains exact commands, return codes, and preflight joins but
   not all terminal stdout. JUnit and product files independently anchor the
   test/product claims; inline assertions plus rc 0 anchor the deterministic
   CPU compile predicates.
5. CUDA Tile physical occupancy/residency behavior remains unverified. A future
   occupancy failure must not be repaired by changing the documented logical
   host block `[1,1,1]`.
6. The root is dirty only because the reviewed CP-0036 persistence transaction
   is not yet committed. GPU launch remains fail-closed until that transaction
   is committed and the root is clean.

## Final gate

**GO to commit CP-0036 with the reviewed subject and this report.** Any change
to the reviewed CP/EV bytes or the recorded identities after the subject hashes
above requires the persistence lineage to be updated or re-reviewed.

After the checkpoint commit, the next permitted action is only the exact v3
correctness smoke under a fresh green `gpu-smoke` admission, followed by the
CPU finalizer if and only if the child publishes a provisional result and exits
cleanly. This report itself is not GPU-smoke success evidence.
