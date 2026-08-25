# CP-0037 final persistence review

- Review time (UTC): `2026-08-25T07:58:10Z`
- Reviewer canonical task: `/root/cp36_persistence_review`
- Review mode: independent read-only persistence, control, run and hash audit
- Decision: **GO** for the exact frozen pre-lineage CP-0037 candidate
- Findings: **P0 = 0, P1 = 0, P2 = 0**
- Finalized GPU correctness: **not accepted; fresh v4 execution and v4 finalization remain pending**

## Review scope

This review joined the exact 8-tracked-plus-3-new persistence candidate to:

- retained v3 real-SM120 child run and provisional evidence;
- the failed no-site v3 finalizer boundary;
- producer-canonical compute-dtype ordering;
- root finalizer/control Layer A4 and immutable manifest Layer B4;
- targeted v4 finalizer tests and the clean post-manifest root suite;
- all current locks, handover and resume instructions; and
- current root, PyPTO, gitlink, standalone-project and upstream identities.

Read-only checks used `git status`, `git diff`, `git show`, `git rev-parse`,
`git submodule status`, manifest/blob comparison, `sha256sum`, `stat`, JSON/XML
inspection and retained process/preflight/gate/barrier/provisional artifacts.
No source, control, persistence, run, product, project or upstream file was
modified by the reviewer. This review log is the only file written for the
review and is outside the formal 8+3 persistence boundary.

## Final committed control identities

### Unchanged PyPTO product

- PyPTO commit: `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`
- PyPTO tree: `e0357daaefa74dbf676550015e60701996c400fb`
- Backend-ON DSO SHA-256:
  `15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c`
- Backend-OFF DSO SHA-256:
  `32c2dea03e13ea49937df239ed1e7bc2b8a931594c41dcfc0cc6de8375464109`
- TensorIR: `1dcb38c20e53d07c97d3781cae538e33901bae30`
- CUDA Tile: `af2417041cc939b87ef56d92cfdcf61737c5457e`
- LLVM: `57109befac92811d2253109242ca6fa69c961fb2`

CP-0037 changes no PyPTO, TensorIR, CUDA Tile, LLVM, DSO, Artifact or Cubin
bytes. The accepted product boundary remains CP-0036; CP-0037 repairs only the
root finalization/control contract.

### Root control v4

- Layer A4 commit: `5564008fddeaaf0a9861ee5c38c895558f577600`
- Layer A4 tree: `b1676a118604c22eebaec787987806f3cf1aeebb`
- Layer B4 manifest commit: `7639d820f4d74972b493c01adc69c92087eefdea`
- Layer B4 tree: `52e37ab60276ebec2e06b46a4b55c39af4c22d62`
- Manifest path: `state/contracts/pypto_nvidia_executable_sm120_v4.json`
- Manifest bytes: `1569`
- Manifest SHA-256:
  `a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf`
- Runner SHA-256:
  `f22befff45d87097ae42b5725cf33a5e296ed74ff177cd84c2b772be5939abdd`

All seven live control files match the v4 manifest's exact bytes, modes and
hashes and the committed Layer A4 blobs. No control path changes after Layer
A4. The implementation and manifest commits are ancestors of the current root.
Control manifests v1 through v3 remain immutable at:

- v1: `c609e97a2f3e379e332137916d041d14931c0e415cd2f2e769c82eab1650aa09`
- v2: `63687352ee22b96019ca0a0850aae9a80feac65cfd9e182a9505b706d9e07cd5`
- v3: `978e873788eb7f3aaeba6473a9b7f8a1bcd827fe201d89cb781927f538c9b6e3`

## V3 GPU child: recorded only, not accepted

Run `pypto-20260825T073624Z-900485-7df250` used clean root
`800a7660e8cc7669cc1a2b592236e262dcb46797`, PyPTO `206447c`, and manifest
v3. It passed admission/release/child gates and exited zero. Its retained
28,414-byte provisional document has SHA-256
`64c0906b7fe57bdddf4c26f7b205a51918a91b93284604f363633ef439d34cfe`
and records child acceptance only as
`gpu-execution-complete-awaiting-run-finalization`.

The provisional durably records:

- static, dynamic and scalar order with two repetitions each;
- six module lifetimes and six explicit unloads;
- non-default current-stream launches and external synchronization;
- all actual logical-output hashes equal to their references;
- all Torch equality checks passing;
- inputs and dynamic padding unchanged;
- all terminal states `Unloaded` with contexts cleared;
- no fallback and no forbidden provider import; and
- the exact three Artifact/Cubin identities retained in EV-0050.

The run-side SHA-256 values are:

- preflight: `76c36a385b81e1feb1bfbb791b339618037e16f25cb9097b95b9423900906652`
- gate: `6e910273d9c3769af720bf31a25e93e884e85eb64198fe0d226b5f706de17ba6`
- barrier: `4d02898ae5435cf249341cd5fa6b08504a2dfeda8f14a81da83a6ee504a0fd45`
- process: `a44850f07fa053011d7f1ff8cb1a3a434f3ee4f547ffbd7ec7cd54024ccafeec`

Pre-release, last and post-exit external/protected NVIDIA compute sets,
protected runtime-mapping sets and unreadable-map sets are empty. The recorded
PGID `900588` has no survivor and there is no external-signal evidence.

This child evidence is not finalized correctness. The required no-site v3
finalizer returned `1`, published no final report, and produced the
session-observed error `FinalizeError: live PyPTO runtime observation differs`.
The error was not persisted as a finalizer sidecar, which EV-0050 labels
explicitly as inherited-terminal observation.

The exact mismatch is nevertheless independently grounded: the producer and
retained provisional encode `supported_compute_dtypes` as `[FP32,BF16]`, while
the v3 finalizer and its synthetic fixture expected `[BF16,FP32]`.
`NvidiaTargetInfo` sorts by stable `DataType` code; FP32 is `0x34` and BF16 is
`0x40`. Therefore FP32 then BF16 is the canonical producer order.

V4 cannot retroactively promote the v3 provisional. Its finalizer, contract and
manifest/control bytes differ, and the finalizer requires exact child/control
identity. V3 remains immutable diagnostic evidence; a fresh v4 child and its
matching v4 finalizer are mandatory.

## Generic v4 finalizer repair and CPU evidence

Layer A4 introduces one shared ordered tuple `("FP32", "BF16")` and exact
equality validation. It deliberately performs no validator-side sorting or set
normalization. The tests cover reversed, missing, duplicate, extra/out-of-
contract, numeric-item and wrong-container representations, and semantic replay
independently rejects dtype-order drift.

Finalizer-focused run `pypto-20260825T074328Z-903562-ee6d75` returns zero with
20 test nodes plus 15 subtests. Its hashes are:

- JUnit: `260395079840eaa32fb40d7aad1aed32bf2f6f6209b0b1f9d7bbbfff7917f83a`
- preflight: `e0832c852ea32f9fbe728d2f490c2f0c300450ac803e153e30a89ccefb02b639`
- process: `8c95707cd81b02a19ef50e9df02b578d43e9f212038495879ac13baea927dfb1`

Clean post-v4 root run `pypto-20260825T074420Z-903996-c8d50d` returns zero
with 225 test nodes plus 113 subtests and no failure/error/skip. Its hashes are:

- JUnit: `9f1d81ed44b9b9b959656d2eff8049b4c529ae8b89fe61c05a8f082783fb1acd`
- preflight: `f14f7d5ddac570839729f76ed12d52fb7570425a3a3a42ddbec4a866ea181c5f`
- process: `4bad7906854f1ec42d819dbbc77d7bc5344b12f3d23071ef763b285f6c1e28cc`

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
9. `state/checkpoints/CP-0037.md`
10. `state/evidence/EV-0050.json`
11. `state/bitlessons/BL-0055.md`

`git diff --check` passes. No source, test, tool, control manifest, run, product,
project or upstream path is part of this persistence candidate. PyPTO, all six
PyPTO gitlinks, pypto-kernels, pypto-framework-plugins, PyTorch, SGLang and
Triton are clean at the locked identities.

GOAL, PLAN revision 29, TODO, CHECKPOINT, HANDOVER, both version files and
WORKSPACE.lock consistently identify CP-0037, EV-0050, root A4/B4, manifest v4,
unchanged PyPTO/DSO identities, diagnostic-only v3 child evidence, and the
fresh-v4-only resume. The stale occupancy/prewarm instruction was removed:
HANDOVER now records that v3 already cleared prewarm, occupancy/resource
validation, launch and unload, and forbids reopening that ABI for the unchanged
v4 child.

The frozen pre-lineage SHA-256 values are:

- `GOAL.md`: `dd524b80f3f2d9ff6a092ccf2b2713e11a3d135610d3002e1dc74c612d08ab4f`
- `PLAN.md`: `fee8c06dc8880008fe9679022426a9d06784ac3dc332110eb3a7585f2997a908`
- `TODO.md`: `0dda222b8129d83abcf0ffff932f40732c1dafdcd94f19a041b120c1ccce566e`
- `CHECKPOINT.md`: `85a1b1187dac261386d83a0008db4f45787ba2c768a4f825b2ee08daf3903f4c`
- `HANDOVER.md`: `f9b3314ab63ec93f35742f138b3ec99307ff51dd45af3e033ac3a96296f0db5e`
- `VERSIONS.lock`: `c3d740b1291727fabb4b0c09af18b8894534e8811a664aaaeb51caaa486a9284`
- `VERSIONS.txt`: `5cf5862eb94772db1f16a9fc841df39e66bd871571b87c8e43f02433de1b470a`
- `WORKSPACE.lock`: `d4561f784f32aeddd5f9f9ab9214937dfdd11df3cabb96a0202992fe9892704e`
- `state/checkpoints/CP-0037.md`:
  `c08c278bb4e0866c7133f51f515a348edbdd7dc20406387aa90d2dbeed0ff087`
- `state/evidence/EV-0050.json` before review lineage:
  `104c70eabb7c6c3adb7ec9ad31f26754cf9f2cddc722c795e92f603825f55cd7`
- `state/bitlessons/BL-0055.md`:
  `6e2d351b550962999c37c3994b1f864fd2e825bed7328e70ba61ebc55519cb9c`

## Accepted scope and non-claims

CP-0037 records but does not accept the v3 real-GPU child/provisional or the v3
finalizer failure. It accepts only:

- the generic canonical compute-dtype evidence order;
- exact v4 finalizer validation and its negative matrix;
- immutable root manifest v4; and
- supporting CPU/control tests.

It does not accept finalized real-SM120 correctness, a successful accepted
PyPTO CUDA correctness milestone, CUDA Graph, frontend-HIR lowering, operator
kernels, TorchInductor/SGLang registration, Qwen correctness, strict coverage,
profiling or performance.

The only resume action is to commit CP-0037, obtain a fresh green `gpu-smoke`
admission, run the exact v4 command from
`docs/pypto_nvidia_executable_sm120_smoke.md`, and finalize only the new v4
provisional with the matching v4 no-site finalizer. Cross-version promotion is
forbidden. Protected amdgpu-sim/zcode processes must never be signalled.

## Permitted lineage-only closure

The sole permitted post-review mutation is adding top-level EV-0050
`review_lineage`. It must bind the CP-0037 SHA, EV-0050 pre-lineage SHA,
BL-0055 SHA, reviewer task, and this report path/bytes/hash. Any byte change
outside that one value, or any change to the other ten persistence files,
invalidates this decision and requires re-review. After insertion, JSON parsing,
the exact 8+3 boundary and `git diff --check` must be checked again.

## Decision

**GO**, with **P0 = 0, P1 = 0, P2 = 0**, for the exact frozen pre-lineage
8+3 CP-0037 persistence candidate and the single lineage-only EV-0050 closure
described above.

Finalized GPU correctness remains pending. The root must be clean after the
CP-0037 commit before any v4 GPU-smoke admission can succeed.
