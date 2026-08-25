# CP-0037 v3 runtime-semantics review

- Review time (UTC): `2026-08-25T07:58:04Z`
- Reviewer canonical task: `/root/cuda_tile_semantics_review`
- CP-0037 subject SHA-256:
  `c08c278bb4e0866c7133f51f515a348edbdd7dc20406387aa90d2dbeed0ff087`
- EV-0050 pre-lineage SHA-256:
  `104c70eabb7c6c3adb7ec9ad31f26754cf9f2cddc722c795e92f603825f55cd7`
- BL-0055 SHA-256:
  `6e2d351b550962999c37c3994b1f864fd2e825bed7328e70ba61ebc55519cb9c`
- Root control commit/tree:
  `7639d820f4d74972b493c01adc69c92087eefdea` /
  `52e37ab60276ebec2e06b46a4b55c39af4c22d62`
- PyPTO commit/tree:
  `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7` /
  `e0357daaefa74dbf676550015e60701996c400fb`
- CP-0037 result: **GO**
- Finalized GPU-correctness result: **NO-GO / not accepted**
- Severity: **P0 = 0, P1 = 0, P2 = 0**

## Scope and method

This was a read-only audit of v3 run
`pypto-20260825T073624Z-900485-7df250`, its canonical provisional and replay
files, process/preflight/gate/barrier records, final PyPTO/CUDA Tile launch
semantics, and the CP-0037/EV-0050/BL-0055 persistence subjects. The review did
not run a GPU command, rebuild a product, modify source, finalize or promote the
v3 result, or signal a process.

CP-0037 is reviewed only as acceptance of the generic v4 canonical-dtype
finalizer/control repair and supporting CPU evidence. The v3 child execution
and failed finalization are retained diagnostic inputs, not accepted GPU
correctness.

## V3 provisional identity

The child exited zero and published the canonical 28,414-byte provisional:

- path:
  `runs/pypto-20260825T073624Z-900485-7df250/pypto-nvidia-executable-sm120/provisional.json`
- SHA-256:
  `64c0906b7fe57bdddf4c26f7b205a51918a91b93284604f363633ef439d34cfe`
- provisional state:
  `gpu-execution-complete-awaiting-run-finalization`

The provisional has no duplicate keys, is exact canonical JSON, and remains
read-only. Every listed CompileRequest, KernelBuildSpec and Artifact replay file
exists with the exact byte count and SHA-256 recorded by the provisional.

## Six execution and lifetime records

The provisional contains exactly six records in the required case order, two
repetitions per case:

| Case | Repetitions | Grid | Kernel arguments | Logical result SHA |
| --- | ---: | --- | ---: | --- |
| static FP32 add | `0,1` | `[4,1,1]` | 3 | `866734bdeb1c286f7a08783e2350e7b53961991e1fc99205b75d8717ff3b207c` |
| dynamic-stride FP32 add | `0,1` | `[6,1,1]` | 12 | `98f2270fa9656f025dfabfda29c8ca20035b98dcf93562b5b46d24a91dde24e6` |
| FP16 tensor plus FP32 scalar | `0,1` | `[4,1,1]` | 3 | `f11f9b04a58e9f148d364b14f8fa0f4e3630a1845dcb18fc4d55b7e1b0ddc6c0` |

For all six records:

- expected and actual logical-output SHA-256 values are identical;
- `torch_equal`, `input_unchanged` and `padding_unchanged` are true;
- the raw current stream is the same caller-selected non-default handle,
  `99462171846480`, and is not CUDA's `0`, `1` or `2` special stream;
- the bound context address/ID is
  `99462136466128 / 1`, equal to the runtime observation;
- external stream synchronization completes before packet release;
- `explicit_unload` is true, terminal state is `Unloaded`, and the bound context
  address is cleared to zero after unload.

The aggregate fields independently agree: `module_lifetimes=6`,
`explicit_unloads=6`, `repetitions_per_case=2`, non-default current stream and
external synchronization are true.

These are exact retained child records. Because the required finalizer did not
publish a final report, they are not promoted to accepted correctness evidence.

## CUDA Tile logical block and occupancy implications

CPU-only deserialization of the retained exact Artifacts through the selected
PyPTO DSO confirms all three launch ABIs use
`block_dimensions=[1,1,1]` and `fallback_used=false`. The fixed v3 runner and
PyPTO production Driver copy those logical dimensions into each
`CUlaunchConfig`; CUDA Tile documents that block dimensions are unused and must
remain `(1,1,1)`.

All six executions passed `NvidiaExecutable::Prewarm`, and prewarm invokes
`cuOccupancyMaxActiveBlocksPerMultiprocessor` with logical block size one before
publishing `Ready`. The run therefore establishes, at diagnostic tier, that
Driver 610.74 returned a legal nonzero occupancy result for each of these exact
static, dynamic and scalar Cubins and that subsequent logical-`[1,1,1]`
launches completed numerically.

It does not record the returned occupancy value and does not prove:

- a physical occupancy count or efficiency;
- that PyPTO's logical-thread register arithmetic derives the compiler's
  physical worker footprint;
- behavior for another Cubin, worker schedule, device or Driver;
- any performance result.

Cubin `REQNTID=[128,1,1]` remains compiler-selected physical metadata whose
exact mapping to four selected worker warps is an inference. The successful v3
diagnostic reinforces, rather than weakens, the requirement to keep host block
dimensions `[1,1,1]`.

## Safety, provider and fallback scope

The retained safety identities recompute exactly:

- preflight:
  `76c36a385b81e1feb1bfbb791b339618037e16f25cb9097b95b9423900906652`
- pre-release gate:
  `6e910273d9c3769af720bf31a25e93e884e85eb64198fe0d226b5f706de17ba6`
- start barrier:
  `4d02898ae5435cf249341cd5fa6b08504a2dfeda8f14a81da83a6ee504a0fd45`
- process:
  `a44850f07fa053011d7f1ff8cb1a3a434f3ee4f547ffbd7ec7cd54024ccafeec`

Pre-release, periodic and post-exit audits all contain empty external NVIDIA
compute, protected NVIDIA compute, protected-runtime-mapping and unreadable-map
sets. Owned PGID `900588` has no survivor. The records contain no external
signal evidence.

The provisional scope is exactly `provider=pypto.tensorir` and
`runtime_object=NvidiaExecutable`. All three strict Artifacts record
`fallback_used=false`, the aggregate fallback flag is false, and the forbidden
provider-import list is empty. The exact child does not import Triton, SGLang or
FlashInfer.

This does not establish a closed-world framework/model coverage result. Torch
is used for allocation, copies, stream ownership and the CPU reference; no
TorchInductor/SGLang model graph or CUPTI closed-world trace is part of this
smoke. CP-0037 and EV-0050 correctly leave strict coverage, framework routing,
Qwen correctness, profiling and performance unclaimed.

## Finalizer failure and canonical dtype repair

The required no-site v3 finalizer failed closed before final-report
publication with the session-observed, non-sidecar error:

`FinalizeError: live PyPTO runtime observation differs`

The retained producer observation contains canonical
`supported_compute_dtypes=["FP32","BF16"]`. Source and CPU-only replay confirm
why:

- `NvidiaTargetInfo` sorts supported dtypes by stable `DataType::Code()`;
- FP32 is `0x34` and BF16 is `0x40`;
- deserializing the retained CompileRequest through exact PyPTO `206447c`
  reproduces FP32 followed by BF16 with CUDA hidden and uninitialized;
- the v3 finalizer compared against the handwritten reversed list
  `["BF16","FP32"]`.

The mismatch is therefore in the root finalizer/fixture, not PyPTO, TensorIR,
the DSO, Artifact, parameter ABI, Cubin or GPU execution.

Root Layer A4 `5564008fddeaaf0a9861ee5c38c895558f577600` introduces one shared
ordered `("FP32","BF16")` contract and exact comparison. Its negative matrix
rejects reversed, missing, duplicate, extra/unknown, numeric and wrong-container
forms, while semantic replay separately rejects target dtype-order drift.
Manifest-only Layer B4 `7639d820f4d74972b493c01adc69c92087eefdea`
binds those exact controls through v4 manifest SHA-256
`a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf`.
The targeted suite passes 20 tests plus 15 subtests; the clean post-v4 root
suite passes 225 tests plus 113 subtests.

The v4 finalizer cannot promote the v3 provisional: the contract, finalizer and
control-manifest bytes differ, while finalization requires an exact same-version
control join. No final report for run `073624` exists. A fresh v4 child and its
matching v4 finalizer are mandatory.

## Severity findings

### P0

None.

### P1

None within CP-0037's bounded acceptance scope.

### P2

None. During review, the stale HANDOVER wording was corrected to say that the
current-stream launch is recorded by the unfinalized v3 child but is not
accepted correctness evidence. The final persistence subjects now preserve the
same boundary as CP-0037 and EV-0050.

## Final decision

**GO for CP-0037** to record the unfinalized v3 diagnostic and accept only the
generic canonical dtype-order finalizer repair, immutable v4 controls and
supporting CPU tests. The checkpoint does not overclaim the retained child
records or the session-only finalizer failure.

**NO-GO for finalized GPU correctness.** A successful child provisional is not
the terminal evidence tier. Until a fresh v4 GPU smoke exits cleanly and its
same-version no-site finalizer publishes and independently reviews a final
report, no PyPTO CUDA correctness milestone, frontend/operator/framework/model
result, strict coverage result, CUDA Graph result, profiling result or
performance result is accepted.
