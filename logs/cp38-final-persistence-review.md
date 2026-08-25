# CP-0038 final persistence review

- Review time (UTC): `2026-08-25T08:16:00Z`
- Reviewer canonical task: `/root/cp36_persistence_review`
- Review mode: independent read-only persistence and finalized-evidence audit
- Decision: **GO** for the exact frozen pre-lineage CP-0038 candidate
- Findings: **P0 = 0, P1 = 0, P2 = 0**

## Scope and method

This review joined the exact 8-tracked-plus-3-new persistence candidate to the
committed finalized-runtime evidence, the v4 child and no-site finalizer, all
run/control/replay/test identities, the current locks and handover, and the new
frontend-HIR resume boundary.

Read-only checks used `git status`, `git diff`, `git show`, `git rev-parse`,
`git submodule status`, manifest/blob comparison, `sha256sum`, `stat`, canonical
JSON/XML inspection, process/preflight/gate/barrier/provisional sidecars, the
published final report, and the two committed independent reviews. No source,
control, persistence, run, report, product, project or upstream file was
modified by the reviewer. This ignored review log is the only file written by
the reviewer and is outside the formal 8+3 persistence boundary.

## Committed finalized-evidence identity

- Evidence commit: `acb79bf844ef36a28edb289d6aaf0e8977423f4b`
- Evidence tree: `95b38dcbc1dfa33538d74757eb21a40c2427ac5f`
- Parent CP-0037 checkpoint: `37c16b3902192a8da59a1d912d2ff3e06bec02fb`
- Parent tree: `cc90c7d8ec7a58ccd0952c4af8756be89a69507e`

The evidence commit contains exactly:

1. `reports/data/pypto-nvidia-executable-sm120-pypto-20260825T080254Z-910620-c669d9.json`
2. `logs/cp38-finalized-runtime-evidence-review.md`
3. `logs/cp38-finalized-runtime-semantics-review.md`

The independent review identities are:

- finalized-runtime evidence review: 9,915 bytes, SHA-256
  `cc648d42c134324d819c0a346899f09bf39f06d148ed60c8707554f18e33c28c`;
- finalized-runtime semantics review: 9,169 bytes, SHA-256
  `83ccd8e4de5a2cd3ad610e1f6427e672afcb8359379ac5e64e8caccd18464429`.

Both reviews report GO with P0/P1/P2 all zero for the minimal finalized
`NvidiaExecutable` runtime milestone and explicitly reject broader frontend,
operator-family, framework, model, CUDA Graph, profiling and performance
claims.

## Final report identity and publication semantics

- Run ID: `pypto-20260825T080254Z-910620-c669d9`
- Report path:
  `reports/data/pypto-nvidia-executable-sm120-pypto-20260825T080254Z-910620-c669d9.json`
- Bytes: `29107`
- SHA-256:
  `727362d7879d58cbee07b11050b17ad149274e8087b0d1872b8f186a66a272a9`
- Schema: `1`
- Smoke: `pypto-nvidia-executable-sm120`
- Status: `accepted-real-sm120-correctness-smoke`
- Published live mode: `0444`
- Git blob mode: `0644`

The distinct mode fields are correct: the no-replace publisher made the live
report read-only, while Git stores only the executable bit and therefore records
the committed regular blob as `100644`. The committed blob bytes equal the live
report bytes. The file is regular, canonical sorted duplicate-key-free JSON
with exactly one terminal newline.

## Exact v4 run and safety joins

The child used clean root `37c16b3902192a8da59a1d912d2ff3e06bec02fb`,
PyPTO `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`, and immutable v4 controls.
It exited zero and published the matching provisional.

Retained SHA-256 values:

- preflight: `4c788dbd6ee0b6d68277fc8fd66663f88a70d915b66ee1e7042cc263de0964f3`
- release gate: `be92af0d23965ccc7c7917529149047f108973a084f4bd8d23dbcc1b1cd43883`
- start barrier: `d510ef38bdfb757f3ca0c6be2ed6c5300d7313cb3dfe9448c141766a41cd1cfe`
- process: `153f9c69b113c36f4e59c7fd272aea3ff281c70776e6a843ad12578621dddbff`
- provisional: `954a266ed5d592698649ad1947fe1879e75dce2f3d9ebad9e16c74034076221f`

The process joins those exact preflight/gate/barrier values and the final report
joins the same run, paths and bytes. Pre-release, periodic and post-exit audits
contain empty external/protected NVIDIA compute, protected runtime-mapping and
unreadable-map sets. No protected heavy process was present. Owned PGID
`910707` has no survivor and there is no external-signal evidence.

## Exact v4 control and product identity

- A4 implementation commit: `5564008fddeaaf0a9861ee5c38c895558f577600`
- A4 tree: `b1676a118604c22eebaec787987806f3cf1aeebb`
- B4 manifest commit: `7639d820f4d74972b493c01adc69c92087eefdea`
- B4 tree: `52e37ab60276ebec2e06b46a4b55c39af4c22d62`
- V4 manifest bytes: `1569`
- V4 manifest SHA-256:
  `a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf`
- Finalizer SHA-256:
  `aad7faf215e2aef0dc626553c1f917e443df0f7ffce4d22425c8276ed23e2f55`
- Runner SHA-256:
  `f22befff45d87097ae42b5725cf33a5e296ed74ff177cd84c2b772be5939abdd`

All seven live controls equal the v4 manifest's exact paths, byte sizes, modes
and hashes and their committed A4 blobs. No later control path changed.
Manifests v1 through v3 remain immutable.

Unchanged product/toolchain identities:

- PyPTO commit/tree:
  `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7` /
  `e0357daaefa74dbf676550015e60701996c400fb`
- TensorIR: `1dcb38c20e53d07c97d3781cae538e33901bae30`
- CUDA Tile: `af2417041cc939b87ef56d92cfdcf61737c5457e`
- LLVM: `57109befac92811d2253109242ca6fa69c961fb2`
- Backend-ON DSO bytes/SHA: `780535416` /
  `15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c`
- CUDA Runtime bytes/SHA: `704288` /
  `96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0`
- Python bytes/SHA: `35989864` /
  `aa85b78409de29d21c7db9a6ea0479fd73a4e245a733ea325f5ecf21772d030f`
- Environment lock SHA:
  `29800d50f635e7188e55a6d6f43bfb4b8ac9ab16c4a21687db2960f18941932a`

## Replay and finalization joins

All seven v4 replay files are regular read-only (`0444`) files and match the
final report's path, size and SHA-256:

- CompileRequest: 1,583 bytes,
  `13c319b832c51188678b51a32b155253a6f896bfd1395044832611df0843adda`;
- static BuildSpec: 1,416 bytes,
  `726ec78502813e816acb01ba64effcf3abbcb53b1e8a7cc59d43fc1928fb003b`;
- static Artifact: 16,690 bytes,
  `411f87920e7a9d9f97f66c865a5695b6b5016ec7983009c47df5c6a3c07b88e9`;
- dynamic BuildSpec: 1,459 bytes,
  `a97ad54f3e31ee1067aa27cf1495792b6742fecd13f2bd8abc0e56476d23b244`;
- dynamic Artifact: 20,483 bytes,
  `6914638d762ce5aaa963e4845d5c5fc473cf2c102b719de3260c7b27619711f5`;
- scalar BuildSpec: 1,416 bytes,
  `15c09132c572c298f08ee2228e91c9fa7cba59e39e2e40f7e2e3c17ff5370ea6`;
- scalar Artifact: 16,808 bytes,
  `28bf2001d40cfd49c641f5280f4c52cbad2c656377c70acf40ad8bb78e273a3f`.

The no-site finalizer's independent exact-DSO replay deserializes the
CompileRequest, all three BuildSpecs and all three Artifacts. The final report
joins replay and runtime on CompileRequest projections, every TargetInfo field,
source/BuildSpec/Artifact/cache/loader/kernel-ABI/Cubin/entry identities, and
fallback state. Replay command SHA is
`ca892e64d69c6bddb1cbe0c02b305b1e17d43c0149c4c29b19ebdaa66e1f88fc`;
stdout SHA is
`b2698fa85b8acebcd71150ea239b2908b988c9f675f010c3082071493d66d816`.

The provisional `runtime` object equals the final report `result` exactly; the
scope, control identity and replay-file records also match exactly.

## Finalized runtime result

The report contains three Artifacts and six executions: two each for static
FP32 add, dynamic-stride FP32 add, and FP16 tensor plus by-value FP32 scalar.
For every execution:

- the caller-selected raw current stream is non-null and non-default;
- expected and actual logical-output SHA values are equal;
- `torch_equal=true`;
- inputs and dynamic/output padding remain unchanged;
- external synchronization completes while the packet remains alive;
- packet release occurs only after synchronization;
- Artifact identity matches the exact case Artifact;
- bound context address/ID match the runtime observation before unload;
- unload is explicit; and
- terminal state is `Unloaded` with bound context zero afterward.

Aggregate counts are exact: six module lifetimes, six explicit unloads, two
repetitions per case, no fallback, and no forbidden provider import.

Pre-smoke checkpoint run `pypto-20260825T080217Z-909769-8f0e3b` passes 225
test nodes plus 113 subtests with no failure/error/skip. Its hashes are:

- JUnit: `9f1ccac58960371a82c7f9ca5c5ed754b71159fbdfb4e92bb3ea65e79fd7dfc7`
- preflight: `3daf7d96aa1f944ca23487d0463e140b7bf5d2c3065bfe249e4011480408acac`
- process: `28ad6716bc0f8fc8e493dd4640506c9f4c0c56b9a8e08fb3900bb5161dbea115`

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
9. `state/checkpoints/CP-0038.md`
10. `state/evidence/EV-0051.json`
11. `state/bitlessons/BL-0056.md`

`git diff --check` passes. No source, test, tool, control manifest, run, report,
product, project or upstream path is part of this persistence candidate. PyPTO,
all six PyPTO gitlinks, pypto-kernels, pypto-framework-plugins, PyTorch, SGLang
and Triton are clean at the locked identities. No ignored production PyPTO
package shadow exists.

GOAL, PLAN revision 30, TODO, CHECKPOINT, HANDOVER, both version files and
WORKSPACE.lock consistently identify CP-0038, EV-0051, the accepted finalized
v4 minimal runtime report and unchanged controls/products. Every current resume
surface retires the GPU rerun. The active path is frontend HIR -> private
TensorIR -> CUDA Tile -> Artifact -> `NvidiaExecutable` correctness for vector
add, fused pointwise, row reduction and simple structured matmul. The stale
historical PLAN sentence saying only the v4 route was current was corrected.

The frozen pre-lineage SHA-256 values are:

- `GOAL.md`: `08247dfef458cfefa31b622ef480cd605d7816281b6101bb4180497935eb039e`
- `PLAN.md`: `aa981c38bd36f8ba7a48338395dd304aa2cc8baaa7935b53f4f12ad6378e90dd`
- `TODO.md`: `a333f26be51219ed7ea6f18834caf6bf8570086c7f91cf97fed56c13c7f313f0`
- `CHECKPOINT.md`: `1d52f38d5fc3c3d7f3e4f70b06e00f535f16dd3a754597b6cc0ab535bd6adf07`
- `HANDOVER.md`: `39ba0390029cb52d590279c467bf1bfbf024a627ff640996a775eedc8185d4b2`
- `VERSIONS.lock`: `fea3c0b3dc650617f0549cf23ddd9375b083c4496646c9c0188675865dc58e30`
- `VERSIONS.txt`: `4a117a1b2a8d7e799c367e05639ff140327c404629beb083bf42e73b989b76dd`
- `WORKSPACE.lock`: `5f71229156815caddc96939674b8427b667f6cac8ada025d31dd2484f6d58d49`
- `state/checkpoints/CP-0038.md`:
  `c146c8c025cb1303d0f9b58a591d2251cc3df1b20f3884fbe4fb5b0d8e0bdb7f`
- `state/evidence/EV-0051.json` before review lineage:
  `548c7553bea65dc4dd7dc4faba13170b8b10cc9dee39ba4f90156bbe742f133d`
- `state/bitlessons/BL-0056.md`:
  `179382076a3c5cc40a98b2f7173b06f7638dd65bdd2a519a55bec233c7a0efa8`

## Accepted scope and non-claims

CP-0038 accepts only the finalized minimal real-SM120 `NvidiaExecutable`
correctness v1 milestone:

- exact PyPTO/TensorIR/CUDA Tile Artifact load, prewarm, current-stream launch,
  synchronization and unload for the three fixed smoke cases;
- numerical/byte correctness and mutation/padding preservation for those cases;
- exact control, run-sidecar, product, compiler-input, Artifact/Cubin and
  TargetInfo replay joins; and
- no fallback for the three smoke Artifacts.

The report's `operator_correctness=true` is interpreted only for those exact
fixtures. CP-0038 does not accept general elementwise or structured matmul,
custom operator-family correctness/performance, frontend HIR lowering, CUDA
Graph, TorchInductor/SGLang integration, Qwen correctness, strict coverage,
profiling or performance.

The next gate is the standalone frontend path recorded above. Each frontend
case requires its own CPU/reference oracle, exact Artifact provenance and real
SM120 correctness evidence. Performance, standalone operator kernels, Inductor
and SGLang remain later gates. The finalized report and v4 controls must be
preserved and the completed smoke must not be rerun merely to change evidence.

## Permitted lineage-only closure

The sole permitted post-review mutation is adding top-level EV-0051
`review_lineage`. It must bind the CP-0038 SHA, EV-0051 pre-lineage SHA,
BL-0056 SHA, reviewer task, and this report path/bytes/hash, together with the
two already committed independent review identities. Any byte change outside
that one value, or any change to the other ten persistence files, invalidates
this decision and requires re-review. After insertion, JSON parsing, exact 8+3
status and `git diff --check` must be checked again.

## Decision

**GO**, with **P0 = 0, P1 = 0, P2 = 0**, for the exact frozen pre-lineage
8+3 CP-0038 persistence candidate and the single lineage-only EV-0051 closure
described above.

The finalized minimal runtime gate is closed. Frontend-HIR and all broader
operator, framework, model, coverage and performance gates remain pending.
