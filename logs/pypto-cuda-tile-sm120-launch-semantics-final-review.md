# PyPTO CUDA Tile SM120 launch-semantics final review

- Review time (UTC): `2026-08-25T07:29:17Z`
- Reviewer canonical task: `/root/cuda_tile_semantics_review`
- PyPTO commit: `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`
- PyPTO tree: `e0357daaefa74dbf676550015e60701996c400fb`
- Root control head: `3de4cf702662cbaf948c6429acf269fee16a491e`
- Root control tree: `c72970aa075e5769aad945599947b1764440f2c6`
- CP-0036 commit-scope result: **GO**
- Real-GPU correctness result: **NO-GO / still unverified**
- Severity: **P0 = 0, P1 = 0, P2 = 2 documentation-precision items**

## Scope and method

This was a source- and retained-evidence-only re-audit of final PyPTO
`206447c`, CUDA Tile `af24170` (`v13.3.3`), private TensorIR `1dcb38c`, local
CUDA 13.3 headers, root control manifest v3, and the current CP-0036,
EV-0049, BL-0053 and BL-0054 drafts. The review did not build, run a GPU
command, create a CUDA context, modify product code, or signal any process.
The only new file is this review.

Reviewed subject digests:

- `state/checkpoints/CP-0036.md`:
  `e36c475f8d3a97a0721e0b169833ea13895ae7868c3c7699331655877c513ce9`
- `state/evidence/EV-0049.json`:
  `54d6a82ed69272a6621ab65750a11784ef4ac0b92b8d0bd61a832e92e4168384`
- `state/bitlessons/BL-0053.md`:
  `b5eab0d65c749448458aab9a3eec8f9364426a960bd495d31a130da11de21fdd`
- `state/bitlessons/BL-0054.md`:
  `a56b7597747d5eeb59410eb0ab06214a147030c4f66b4b78a99bebe11755bc66`

## Host block dimensions: `[1,1,1]` is the required ABI

The final decision to keep `ArtifactLaunchAbi.block_dimensions=[1,1,1]` is
correct and is stronger than an inference from ordinary CUDA conventions:

1. Pinned CUDA Tile's host example says grid dimensions define the Tile Grid,
   while block dimensions are unused and **must** be `(1,1,1)`
   (`projects/pypto/3rdparty/nvidia/cuda-tile/README.md:306-315`).
2. TensorIR's own CUDA Tile runtime constructs `CUlaunchConfig` with the
   computed Tile Grid, `blockDimX/Y/Z=1`, zero dynamic shared memory and spread
   cluster scheduling
   (`projects/pypto/3rdparty/nvidia/tensor-ir/lib/Runtime/CudaTileRuntimeKernel.cpp:143-171`).
3. Final PyPTO records the same logical ABI in the Artifact
   (`projects/pypto/src/compiler/nvidia_artifact_compiler.cpp:276-295`) and
   copies it unchanged into `CUlaunchConfig`
   (`projects/pypto/src/runtime/nvidia/driver_api.cpp:568-588`).

Therefore `EIATTR_REQNTID=[128,1,1]` must not be copied into host
`blockDim*`. Doing so would contradict both CUDA Tile's documented host ABI and
TensorIR's reference runtime.

## `REQNTID` inference boundary

The retained static Cubin independently shows `EIATTR_REQNTID=0x80,1,1`
(`runs/pypto-20260825T052038Z-800777-8e8e83/diagnostic-static-cuobjdump-elf.txt:379-389`).
The selected schedule asks for four worker warps, and CUDA Tile exposes
`num_worker_warps_per_cta` as a compiler optimization hint. Thus 128 physical
threads are consistent with four 32-lane worker warps. The exact
metadata-to-worker mapping is nevertheless an inference: neither the CUDA Tile
host-launch contract nor local CUDA Driver headers define `REQNTID` as a host
launch dimension for Tile kernels.

CP-0036 lines 29-34 and EV-0049's `reqntid_semantics` field preserve this
boundary correctly. They use the observed metadata only as compiler-selected
physical evidence and do not convert it into an Artifact launch parameter.

## Driver enumeration versus signature order

Final `206447c` correctly separates three independent authorities:

- The Cubin validator parses every KPARAM record's ordinal and requires its
  width to equal `expectedSizes[ordinal]`, with complete unique ordinals and
  bounded non-overlapping constant-bank ranges
  (`projects/pypto/3rdparty/nvidia/tensor-ir/lib/Compiler/CudaTile/CudaTileArtifact.cpp:115-227`).
  PyPTO runs this validator both for a newly produced Artifact and when an
  Artifact is deserialized through `ArtifactBuilder::CreateStrict`
  (`projects/pypto/src/compiler/artifact.cpp:700-768,1482-1519`). Cubin KPARAM
  ordinals therefore remain the signature-width authority.
- The live Driver view is not used to invent signature order. Final
  `ValidateLoadedParameterAbi` checks parameter count, the complete width
  multiset, overflow-safe bounded ranges and non-overlap, independent of the
  order in which Driver records are returned
  (`projects/pypto/src/runtime/nvidia/executable.cpp:128-192`).
- The host `kernelParams` array remains in Artifact/Cubin signature order.
  Driver offsets are never used to reorder it. This matches TensorIR's
  signature-ordered argument packer and the CUDA launch API's array-of-N-
  parameter-values contract.

The new non-palindromic twelve-argument fake-Driver case is material: it
reverses mixed 8/4-byte `(offset,width)` records and then verifies that launch
still receives pointer/size/stride values in signature order
(`projects/pypto/tests/ut/cpp/nvidia_executable_test.cpp:1379-1424`). The
negative width-multiset cases, absent/extra parameter cases, overlap,
out-of-frame and overflow cases remain fail-closed.

The failed first real run did not persist its exact internal predicate; only
the bounded top-level parameter-ABI error survives. The source/Cubin analysis
therefore justifies making the loader enumeration-order-independent, but does
not justify claiming that retained evidence has independently proven one exact
production Driver enumeration sequence. CP-0036 and CHECKPOINT.md state this
inference boundary correctly.

## Dynamic parameter packing

TensorIR lowers tensor sizes and explicit dynamic strides to `i32`, and its
artifact validator expects four-byte widths. PyPTO now uses the same width,
accepts sizes in `1..INT32_MAX` and strides in `0..INT32_MAX`, zero-initializes
each stable eight-byte host slot, and copies the native four-byte `int32` into
that storage (`projects/pypto/src/runtime/nvidia/executable.cpp:90-125,602-634`).
The eight-byte slot is storage, not a declaration that the device parameter is
eight bytes. The change is generic across Artifact tensor descriptors and does
not depend on any model, layer, entry-point name or benchmark shape.

## Occupancy and physical-resource caveat

The next unverified prewarm boundary is physical residency, not host block
selection. PyPTO currently passes logical `block_threads=1` to
`cuOccupancyMaxActiveBlocksPerMultiprocessor`
(`projects/pypto/src/runtime/nvidia/executable.cpp:868-882` and
`projects/pypto/src/runtime/nvidia/driver_api.cpp:484-507`). The local CUDA
header describes that argument as the block size intended for launch, so `1`
is consistent with CUDA Tile's host contract. However, CUDA Tile's README and
TensorIR reference runtime do not establish how this generic occupancy API
accounts for TileIR's compiler-selected physical workers.

Likewise, PyPTO's local aggregate register check multiplies per-thread register
count by logical block size one. It must not be represented as an independent
derivation of the physical 128-worker register footprint. Until a live v3
prewarm succeeds, the Driver occupancy result is only an open tile-aware
compatibility gate.

This does not block CP-0036 because CP-0036 accepts no completed prewarm or GPU
correctness. If the next smoke fails at occupancy/resource validation, the
correct response is to establish or revise a Tile-aware residency check. It is
not to change host block dimensions to 128.

## Model-hack audit

No hidden model workaround was found in final PyPTO production changes from
`6361f11` through `206447c`:

- no Qwen, attention, GDN, model name or layer branch;
- no entry-function-name special case;
- no production `8x8` or `17x9` shape branch;
- no fallback to ATen, Triton, FlashInfer, cuBLAS or another compute provider;
- no change to TensorIR/CUDA Tile Cubin generation, schedule identity, Artifact
  schema or expected Cubin bytes.

The fixed add/scalar names and shapes occur only in unit/smoke fixtures. The
production implementation derives widths, counts, ranges and packed values
from the generic Artifact argument layout.

## Persistence and evidence review

CP-0036 and EV-0049 now maintain the required distinction:

- the failed first real run is recorded, not accepted as GPU behavior;
- the accepted result is the generic CPU-reviewed parameter repair, fresh
  ON/OFF products, CPU-only deterministic Cubin reconstruction, product audit
  and immutable root control v3;
- successful prewarm, launch, numerical correctness, CUDA Graph, frontend,
  operator, framework, model, coverage, profiling and performance remain
  explicitly unclaimed.

Key identities were rechecked without executing them:

- manifest v3: 1,569 bytes,
  `978e873788eb7f3aaeba6473a9b7f8a1bcd827fe201d89cb781927f538c9b6e3`;
- backend-ON DSO: 780,535,416 bytes,
  `15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c`;
- backend-OFF DSO: 434,646,736 bytes,
  `32c2dea03e13ea49937df239ed1e7bc2b8a931594c41dcfc0cc6de8375464109`;
- post-v3 root JUnit:
  `cbc276bd9a9869f8afb6345e3c3f1ebfa8c3501070a2401ecf5a101bdab7b213`,
  recording 224 top-level tests plus 106 subtests with no failure;
- post-v3 preflight/process:
  `ef376b8825d6c026bf67f6bbffe4ac08086265986a25c9f2f72c871c7b4213dd` /
  `983cdda7538c4a3d0b98bb45d91e5dca48621d4109525d52ac49af96aa4ea223`.

The root is intentionally dirty only with the pending CP-0036 persistence
drafts and this review. The exact v3 smoke remains fail-closed until the
checkpoint is committed and the root is clean.

## Severity findings

### P0

None.

### P1

None within CP-0036's explicitly CPU/control-only acceptance scope. Occupancy
and real launch remain mandatory later gates, not accepted results.

### P2

1. BL-0054's title says enumeration order **is not** signature order, while its
   body and retained evidence support the narrower rule that enumeration order
   **must not be assumed to be** signature order. The body is correct and the
   stronger title does not affect code or CP scope, but future wording should
   use “need not be signature order.”
2. BL-0053 line 7 says the reverted change would have copied “four physical
   worker warps,” before lines 13-15 correctly mark the exact REQNTID-to-worker
   mapping as an inference. Prefer “the observed 128-thread physical metadata,
   consistent with four selected worker warps.” CP-0036 and EV-0049 already use
   the precise wording.

These are non-blocking bitlesson precision items. They do not alter the
accepted implementation, evidence boundaries or resume action.

## Final decision

**GO for committing CP-0036** as a record of the failed first runtime boundary
and acceptance of final PyPTO `206447c`, fresh CPU/product evidence, and root
control manifest v3. There is no P0 or P1 within that bounded scope.

**NO-GO for GPU correctness.** No retained v3 run has completed prewarm,
prepared a launch packet, launched on the caller's non-default stream, compared
device output, or passed the CPU finalizer. The next authorized action remains
the exact v3 `gpu-smoke` under a fresh green admission. Any occupancy failure
must preserve logical host block `[1,1,1]` and be handled as a separate
Tile-aware physical-resource issue.
