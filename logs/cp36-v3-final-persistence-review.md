# CP-0036 v3 final persistence review

- Review time (UTC): `2026-08-25T07:29:00Z`
- Reviewer canonical task: `/root/cp36_persistence_review`
- Review mode: read-only repository, product, run and persistence audit
- Decision: **GO** for the frozen pre-lineage CP-0036 persistence candidate
- Findings: **P0 = 0, P1 = 0, P2 = 0**
- Real-GPU correctness decision: **not accepted and still pending**

## Scope and method

This review joined the current root and PyPTO Git objects, all three immutable
SM120 control manifests, every manifest-v3 live and committed control blob,
fresh final backend-ON/OFF products, CTest and exact-DSO Python results, the
product-isolation audit, the device-hidden three-Cubin CPU compilation, the
clean post-v3 root suite, the retained failed real-SM120 run, and the complete
12-file persistence candidate.

Read-only checks used `git status`, `git diff`, `git show`, `git rev-parse`,
`git submodule status`, the manifest validator under `-I -B -S`, `sha256sum`,
`stat`, JSON/XML inspection, `readelf`, `nm`, and retained process/preflight
sidecars. No source, control, product, run, upstream, protected process or
tracked persistence file was modified by this reviewer. This ignored review
log is the only file written by the reviewer and is not part of the formal
12-file checkpoint boundary.

## Exact committed implementation and control identities

### Final PyPTO repair

- Commit: `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`
- Tree: `e0357daaefa74dbf676550015e60701996c400fb`
- Parent: `3431a944691f8295d3a55a4e222dadcfe34253f5`
- TensorIR: `1dcb38c20e53d07c97d3781cae538e33901bae30`
- CUDA Tile: `af2417041cc939b87ef56d92cfdcf61737c5457e`
- LLVM: `57109befac92811d2253109242ca6fa69c961fb2`

The final generic contract is enumeration independent: runtime-free Cubin
validation binds every `KPARAM` ordinal to its signature width; the live Driver
widths are compared as a multiset; live offsets are checked only as bounded,
non-overlapping ranges; and the launch pointer array remains signature ordered.
Dynamic sizes/strides are four-byte `int32` values in zeroed eight-byte host
slots and are range checked before packet publication.

CUDA Tile's pinned launch contract remains distinct from conventional CUDA
thread-block semantics. Host block dimensions are unused and must remain
`[1,1,1]`; `EIATTR_REQNTID=[128,1,1]` is compiler-selected physical metadata
consistent with the selected worker warps, with the exact mapping explicitly
treated as an inference. It is not copied into `CUlaunchConfig.blockDim*`.

### Root control v3

- Layer A3 commit: `c71f32bd415a973a2a7756ecc9b1ae59f30df219`
- Layer A3 tree: `89dc321ff8f6913b826f70f09c07d5c045a67790`
- Layer B3 manifest commit: `3de4cf702662cbaf948c6429acf269fee16a491e`
- Layer B3 tree: `c72970aa075e5769aad945599947b1764440f2c6`
- Manifest: `state/contracts/pypto_nvidia_executable_sm120_v3.json`
- Manifest bytes: `1569`
- Manifest SHA-256:
  `978e873788eb7f3aaeba6473a9b7f8a1bcd827fe201d89cb781927f538c9b6e3`
- Runner SHA-256:
  `f22befff45d87097ae42b5725cf33a5e296ed74ff177cd84c2b772be5939abdd`

The clean-root validator joined the manifest to Layer A3 and all seven exact
live/committed file sizes, modes and hashes. Manifests v1 and v2 remain
byte-identical at, respectively:

- `c609e97a2f3e379e332137916d041d14931c0e415cd2f2e769c82eab1650aa09`
- `63687352ee22b96019ca0a0850aae9a80feac65cfd9e182a9505b706d9e07cd5`

PyPTO `3431a94` and root v2 `8176731`/`1cb5638` are immutable predecessor
evidence only. V2 still joined a live width to the same Driver enumeration
index and is not authorized for another GPU smoke.

## Final products and CPU evidence

### Backend ON

- Configure: `pypto-20260825T065002Z-868546-526995`, rc `0`
- Build: `pypto-20260825T065050Z-869893-badb83`, rc `0`
- DSO bytes: `780535416`
- DSO SHA-256:
  `15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c`
- CTest: `pypto-20260825T065932Z-883420-3d116c`, `9/9`
- CTest JUnit SHA-256:
  `347bb1eddb9ae2aa0a6937c19bc85c54074fcf7a3399ffd8ee65cd17b7ffed68`
- Exact-DSO Python: `pypto-20260825T070443Z-889280-424570`,
  `142 passed, 2 skipped`
- Python JUnit SHA-256:
  `62dd5dbc2c30c4d58028735a35829212bd5bfbebe30931c3efb96a9d83704065`

### Backend OFF

- Configure: `pypto-20260825T065955Z-883614-5098e2`, rc `0`
- Build: `pypto-20260825T070007Z-883835-7cb6d3`, rc `0`
- DSO bytes: `434646736`
- DSO SHA-256:
  `32c2dea03e13ea49937df239ed1e7bc2b8a931594c41dcfc0cc6de8375464109`
- CTest: `pypto-20260825T070412Z-889089-b986bd`, `7/7`
- CTest JUnit SHA-256:
  `27460bd6e97507c17a6b8381a430d86e1a918af98f4a9b8c8ad0744979d8aa60`
- Exact-DSO Python: `pypto-20260825T070508Z-889497-469338`,
  `135 passed, 9 skipped`
- Python JUnit SHA-256:
  `2fcee961c432ef2112c84549e70dbf468471ee20c0c70d3df768e77376a1c789`

Product audit `pypto-20260825T070539Z-889610-55d705` returns zero and binds
both DSO hashes, no RPATH/RUNPATH, exactly five standard dependencies and two
permitted definitions, zero CUDA/TensorIR/NvidiaExecutable dynamic-symbol
leakage, and the mutually exclusive ON/OFF production/test compile rows.

CPU compilation `pypto-20260825T070649Z-890066-2a1ae2` returns zero through
the exact final DSO. Its command durably asserts the wrapper-owned run markers,
empty `CUDA_VISIBLE_DEVICES`, Torch device count zero, CUDA uninitialized,
absence of Triton/SGLang/FlashInfer, and all three expected Cubin size/SHA and
kernel-ABI pairs. Runs `062710`, `062751`, and `070611` remain explicitly
unaccepted diagnostics for, respectively, an over-strict Torch assertion,
inner-environment loss of wrapper state, and an obsolete ownership-marker
assertion.

Clean post-manifest root run `pypto-20260825T071601Z-892819-67acee` returns
zero with 224 test cases plus 106 subtests and no skip/failure/error. Its JUnit
SHA-256 is
`cbc276bd9a9869f8afb6345e3c3f1ebfa8c3501070a2401ecf5a101bdab7b213`;
preflight SHA-256 is
`ef376b8825d6c026bf67f6bbffe4ac08086265986a25c9f2f72c871c7b4213dd`;
process SHA-256 is
`983cdda7538c4a3d0b98bb45d91e5dca48621d4109525d52ac49af96aa4ea223`.
It ran beside six protected heavy processes under the explicit CPU-only policy,
with empty NVIDIA compute sets and no coexistence pause or external-signal
evidence.

## Retained failed real-SM120 boundary

Run `pypto-20260825T052038Z-800777-8e8e83` remains diagnostic-only. Its
preflight, parent release and child barrier passed; it returned `1` after the
inherited terminal showed static repetition zero in `NvidiaExecutable`
prewarm following real Runtime/context/module/function observation. The exact
child stderr was not persisted, so the source/Cubin diagnosis is explicitly an
inference and not a replayable error predicate.

It did not reach `prepare_launch`, `NvidiaExecutable::Launch`, provisional
publication, numerical comparison or finalization. Consequently it proves no
kernel launch, current-stream correctness, numerical GPU result or unload
sequence. Its retained key SHA-256 values are:

- preflight: `5c88ffd0d77e7a42fe674ec6423ac3376f4809ea782a6074225157e8320dc208`
- gate: `e78e20217a10df8b332568688fd01026267dd8f305a622b24037790d78cea911`
- barrier: `e0849f9f2f511d54014da29c68bdc91fc1a8aee44ce899373551ac8a1437537f`
- process: `9085c010791b3d5e8639b12a0e7110105c14bb6c4f331c0fab11255f08c86525`
- static diagnostic Cubin:
  `6dc121d2574537753229ed537efc5d2558eee26bfac0ad9d21826b5f33632b82`
- post-run cuobjdump text:
  `40270a5a26c51fcd5b1309ffe1a9781a5d863b7b0cf6607219444c67cfd6b90a`

Pre-release, last and post-exit sidecars contain empty external compute-PID and
protected compute-PID/runtime-mapping/unreadable-map sets. The recorded PGID
has no survivor, and there is no external-signal evidence.

## Exact persistence boundary

The reviewed candidate contains exactly eight tracked modifications and four
non-ignored untracked files:

1. `GOAL.md`
2. `PLAN.md`
3. `TODO.md`
4. `CHECKPOINT.md`
5. `HANDOVER.md`
6. `VERSIONS.lock`
7. `VERSIONS.txt`
8. `WORKSPACE.lock`
9. `state/checkpoints/CP-0036.md`
10. `state/evidence/EV-0049.json`
11. `state/bitlessons/BL-0053.md`
12. `state/bitlessons/BL-0054.md`

`git diff --check` passes. No source, test, tool, control manifest, product,
project checkout or upstream path is part of this persistence candidate.
`DECISIONS.md` remains unchanged because neither the externally specified CUDA
Tile launch contract nor the generic ABI repair introduces a new discretionary
project architecture decision. Root-level `RISKS.md` and `LESSONS.md` do not
exist; the two canonical lessons are recorded as BL-0053 and BL-0054.

The frozen pre-lineage SHA-256 values are:

- `GOAL.md`: `bd498f002b3b1ac2e5cd25eec1a3a6e0bc2317f57a7dadd82099a820ea51e839`
- `PLAN.md`: `11cd4166dd217b320fbb89e8d2d5faf1ba1cc49e31c464b8689da00fde7df177`
- `TODO.md`: `b407d04a6c91d62ba9f7a9c6cd73d12b473625cc89b4508f32899f4486c53596`
- `CHECKPOINT.md`: `0c25c99863a2096c83792f1b0c3ba3c25f54f4cad8780a2c6dc2fb480afc259c`
- `HANDOVER.md`: `1ece2931f8439d0756dfed6e2da2819aeca890ec0ed26501ff90736d8699e44d`
- `VERSIONS.lock`: `082928cada5ab6d116ac1bcbb5d866c714617821bbae171a4e8a140c5aa55217`
- `VERSIONS.txt`: `5fb91653333b40b101943f3af816a29d1e602da2f86f0db9420725445ece3c3b`
- `WORKSPACE.lock`: `93087ed69a71a0edc9a74e4ba5fa1fc3a7e75f4e3e92242cd7be5954a9819651`
- `state/checkpoints/CP-0036.md`:
  `e36c475f8d3a97a0721e0b169833ea13895ae7868c3c7699331655877c513ce9`
- `state/evidence/EV-0049.json` before review lineage:
  `54d6a82ed69272a6621ab65750a11784ef4ac0b92b8d0bd61a832e92e4168384`
- `state/bitlessons/BL-0053.md`:
  `b5eab0d65c749448458aab9a3eec8f9364426a960bd495d31a130da11de21fdd`
- `state/bitlessons/BL-0054.md`:
  `a56b7597747d5eeb59410eb0ab06214a147030c4f66b4b78a99bebe11755bc66`

The root working tree contains only that exact candidate. PyPTO, all six PyPTO
gitlinks, pypto-kernels, pypto-framework-plugins, PyTorch, SGLang and Triton are
clean at the locked identities. The root HEAD is the manifest-only B3 commit
`3de4cf702662cbaf948c6429acf269fee16a491e`; the current dirt intentionally
keeps GPU launch fail-closed until the persistence checkpoint is committed.

## Acceptance and non-claims

CP-0036 records and preserves the failed-run boundary. It accepts only:

- the generic parameter width/range/packing and enumeration-independent ABI
  repair;
- fresh supporting CPU validation and final ON/OFF product identities;
- the corrected CUDA Tile logical-host `[1,1,1]` semantics; and
- immutable root control manifest v3.

It does not accept a successful PyPTO CUDA kernel launch, CUDA numerical
correctness, non-default-current-stream execution, explicit unload behavior,
CUDA Graph, frontend-HIR lowering, operator kernels, TorchInductor/SGLang
registration, Qwen correctness, strict coverage, profiling or performance.

The only current resume action is the exact v3 `gpu-smoke` command after this
persistence candidate is committed and a fresh admission is green. A successful
provisional result must then be finalized in a separate no-site CPU process and
independently reviewed. `REQNTID` must never be copied into the logical host
block; a later occupancy/resource failure requires new tile-aware residency
evidence rather than a speculative launch-dimension change.

## Permitted lineage-only closure

The sole permitted post-review mutation is adding EV-0049 `review_lineage`.
It must bind the exact pre-lineage EV SHA above, CP-0036 SHA, both bitlesson
SHAs, this reviewer task and this report path/hash. Bytes outside that one
top-level value, or any change to the other 11 persistence files, invalidates
this decision and requires re-review. After lineage insertion, JSON parsing,
the exact 8+4 boundary and `git diff --check` must be rechecked before commit.

## Decision

**GO**, with **P0 = 0, P1 = 0, P2 = 0**, for the exact frozen pre-lineage
12-file CP-0036 persistence candidate and the single lineage-only EV-0049
closure described above.

Real GPU correctness remains pending. The root must be clean after the
persistence commit before any v3 GPU-smoke admission can succeed.
