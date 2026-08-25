# CP-0037 v4 finalizer evidence review

## Decision

**GO to commit CP-0037.** P0: 0. P1: 0. P2: 0.

This review accepts only the generic canonical compute-dtype evidence contract,
the v4 finalizer/control implementation, its negative validation matrix, and
supporting CPU/control tests. It records the v3 GPU child and provisional as an
unfinalized diagnostic. It does not accept finalized real-SM120 correctness, a
successful PyPTO CUDA correctness milestone, CUDA Graph, frontend HIR lowering,
operator kernels, TorchInductor or SGLang registration, Qwen correctness,
strict coverage, profiling, or performance.

## Bound persistence subject

- CP-0037 SHA-256:
  `c08c278bb4e0866c7133f51f515a348edbdd7dc20406387aa90d2dbeed0ff087`
- EV-0050 pre-review-lineage SHA-256:
  `104c70eabb7c6c3adb7ec9ad31f26754cf9f2cddc722c795e92f603825f55cd7`
- BL-0055 SHA-256:
  `6e2d351b550962999c37c3994b1f864fd2e825bed7328e70ba61ebc55519cb9c`

The eight tracked persistence changes are `CHECKPOINT.md`, `GOAL.md`,
`HANDOVER.md`, `PLAN.md`, `TODO.md`, `VERSIONS.lock`, `VERSIONS.txt`, and
`WORKSPACE.lock`. They consistently select CP-0037, EV-0050, PyPTO `206447c`,
A4/B4, manifest v4, and a fresh v4 smoke/finalizer as the next action. Root
`git diff --check` and EV canonical JSON parsing pass.

## Source and control identities

- PyPTO remains unchanged and clean at
  `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`, tree
  `e0357daaefa74dbf676550015e60701996c400fb`.
- Backend-ON DSO remains
  `15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c`.
- Backend-OFF DSO remains
  `32c2dea03e13ea49937df239ed1e7bc2b8a931594c41dcfc0cc6de8375464109`.
- A4 implementation commit:
  `5564008fddeaaf0a9861ee5c38c895558f577600`
- A4 tree:
  `b1676a118604c22eebaec787987806f3cf1aeebb`
- A4 parent:
  `800a7660e8cc7669cc1a2b592236e262dcb46797`
- B4 manifest-only commit:
  `7639d820f4d74972b493c01adc69c92087eefdea`
- B4 tree:
  `52e37ab60276ebec2e06b46a4b55c39af4c22d62`
- B4 is the direct child of A4 and adds only
  `state/contracts/pypto_nvidia_executable_sm120_v4.json`.
- Manifest v4 size: 1,569 bytes.
- Manifest v4 SHA-256:
  `a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf`
- Manifest schema/kind:
  `4` / `pypto-nvidia-executable-sm120-controls-v4`.

All seven live control files independently match the v4 manifest's exact byte
sizes, modes, and SHA-256 values. Control manifests v1, v2, and v3 remain
byte-unchanged with SHA-256 values `c609e97a...aa09`, `63687352...07cd5`, and
`978e8737...c9b6e3`, respectively.

A4 changes exactly the smoke document, finalizer fixture/tests, shared contract,
manifest selector, and finalizer. It defines the one ordered tuple
`("FP32", "BF16")`, requires exact ordered equality, and deliberately does
not sort or set-normalize evidence. The order is producer-canonical:
`NvidiaTargetInfo` sorts stable DataType codes FP32 `0x34` then BF16 `0x40`.
The real runner and independent serialized TargetInfo replay preserve this
order.

The negative matrix covers reversed, missing, duplicate, extra/unknown,
numeric-item, and wrong-container representations. Semantic replay separately
rejects supported-dtype order drift, preserving two independent joins.

## V3 real-GPU child diagnostic

Run `pypto-20260825T073624Z-900485-7df250` is bound as follows:

- root commit `800a7660e8cc7669cc1a2b592236e262dcb46797`;
- root tree `7a8764da31506be65eeb41077b49e39f5d6efba9`;
- PyPTO `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`;
- v3 manifest SHA-256
  `978e873788eb7f3aaeba6473a9b7f8a1bcd827fe201d89cb781927f538c9b6e3`;
- child process status `exited`, return code 0;
- process SHA-256
  `a44850f07fa053011d7f1ff8cb1a3a434f3ee4f547ffbd7ec7cd54024ccafeec`;
- preflight SHA-256
  `76c36a385b81e1feb1bfbb791b339618037e16f25cb9097b95b9423900906652`;
- gate SHA-256
  `6e910273d9c3769af720bf31a25e93e884e85eb64198fe0d226b5f706de17ba6`;
- start-barrier SHA-256
  `4d02898ae5435cf249341cd5fa6b08504a2dfeda8f14a81da83a6ee504a0fd45`;
- provisional path
  `runs/pypto-20260825T073624Z-900485-7df250/pypto-nvidia-executable-sm120/provisional.json`;
- provisional size 28,414 bytes;
- provisional SHA-256
  `64c0906b7fe57bdddf4c26f7b205a51918a91b93284604f363633ef439d34cfe`.

The retained provisional records two complete child lifetimes for each static,
dynamic, and scalar case: six non-default-current-stream launches, external
synchronization, equal expected/actual logical hashes, Torch equality,
unchanged inputs and padding, explicit unload, terminal `Unloaded`, no fallback,
and no forbidden provider import. The three Artifact/Cubin identities and sizes
in EV-0050 match the retained provisional.

Pre-release, last, and post-exit external/protected NVIDIA compute sets are
empty. Protected-runtime-mapping and unreadable-map sets are empty. Owned PGID
`900588` has no retained survivor evidence and no external signal evidence is
recorded.

These are child/provisional records, not accepted GPU correctness. EV-0050
preserves the raw child field
`gpu-execution-complete-awaiting-run-finalization` as
`recorded_child_acceptance`, while its current classification is explicitly
`gpu-child-complete-unfinalized-diagnostic-cross-version-promotion-forbidden`.
That distinction avoids rewriting retained evidence and avoids implying that
the v3 document remains eligible for promotion.

## V3 finalizer failure and cross-version prohibition

The required no-site v3 finalizer failed closed with the session-observed error
`FinalizeError: live PyPTO runtime observation differs`. Its stderr was not
persisted as a finalizer sidecar, and no final report was published. Source and
provisional inspection isolate the mismatch to `supported_compute_dtypes`: the
real producer wrote canonical `["FP32", "BF16"]`; the v3 finalizer and fixture
expected `["BF16", "FP32"]`.

The defect belongs to the root finalizer/control fixture, not PyPTO, TensorIR,
the DSO, Artifact, or Cubin. No PyPTO rebuild or device-code change was needed.

V4 cannot retroactively finalize the v3 provisional. The v4 finalizer,
contract, selector, manifest, and exact control identity differ. Finalization
requires the provisional inputs, child gate, pre-release gate, current live
manifest, and contract/control bytes to agree exactly. A compatibility
normalization or cross-version promotion would weaken those joins and is
correctly prohibited. The v3 run therefore remains diagnostic; a fresh v4
child and v4 finalizer are mandatory.

## V4 CPU/control validation

Finalizer-focused run `pypto-20260825T074328Z-903562-ee6d75`:

- return code 0;
- 20 tests plus 15 subtests;
- JUnit SHA-256
  `260395079840eaa32fb40d7aad1aed32bf2f6f6209b0b1f9d7bbbfff7917f83a`;
- preflight SHA-256
  `e0832c852ea32f9fbe728d2f490c2f0c300450ac803e153e30a89ccefb02b639`;
- process SHA-256
  `8c95707cd81b02a19ef50e9df02b578d43e9f212038495879ac13baea927dfb1`.

Clean post-v4 root run `pypto-20260825T074420Z-903996-c8d50d`:

- return code 0;
- 225 tests plus 113 subtests;
- JUnit total 338, zero failures/errors/skips;
- JUnit SHA-256
  `9f1d81ed44b9b9b959656d2eff8049b4c529ae8b89fe61c05a8f082783fb1acd`;
- preflight SHA-256
  `f14f7d5ddac570839729f76ed12d52fb7570425a3a3a42ddbec4a866ea181c5f`;
- process SHA-256
  `4bad7906854f1ec42d819dbbc77d7bc5344b12f3d23071ef763b285f6c1e28cc`.

Both are CPU/control runs with `gpu_smoke.requested=false`. They validate the
finalizer contract and control machinery; they do not finalize or replace the
v3 provisional and do not establish a v4 GPU result.

## Findings

### P0

None.

### P1

None.

### P2

None.

The earlier wording ambiguity was resolved without falsifying the retained
provisional: EV-0050 now records the raw child value separately from the current
unfinalized, cross-version-forbidden classification.

## Final gate

**GO to commit CP-0037 with the bound persistence subject and this report.**

After a clean CP-0037 commit, the only next acceptance route is the exact v4
command under a fresh green `gpu-smoke` admission, followed by the no-site v4
finalizer using only the new v4 provisional and its printed SHA-256. The v3
provisional must remain diagnostic and must not be promoted, rewritten, or
presented as finalized GPU correctness.
