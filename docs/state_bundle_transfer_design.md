# Generic StateBundle zero/copy design

## Decision and status

GDN defines a paired numerical state, but exact zero/copy/checkpoint is a
generic heterogeneous-state lifecycle primitive owned by the PyPTO compiler and
NVIDIA runtime. It is not another GDN operator, is not part of
`GDN_V1_SPEC`, and does not enter the `pypto-kernels` tuning/artifact catalog.

This document is a source design only. No StateBundle API, executor, CUDA
operation, framework adapter or graph support exists yet.

Implementation is ordered after:

1. the staged single-DSO CMake gate and commit;
2. the reviewed `NvidiaTargetInfo` integration and validation;
3. a context-owned compile/executable path with the caller's current raw
   CUstream and completion contract.

Only then should the source contract and executor land. This keeps stream,
target and executable identity from being guessed prematurely.

## Ownership

| Layer | Owns | Must not own |
| --- | --- | --- |
| `pypto-kernels` | GDN state component shape/dtype/layout, BF16 conv + FP32 recurrent paired semantics, GDN reference/kernel | allocator, Radix node, request, stream lease, generic memcpy/memset executor |
| `pypto-framework-plugins` | one-time SGLang metadata snapshot, virtual-to-physical slot translation, new/COW/checkpoint/donate decisions, segment boundary, completion feedback | Torch/SGLang compute/copy fallback, CUDA executor, numerical GDN algorithm |
| `pypto` compiler/runtime | generic StateBundle ABI, generation/lease state machine, address validation, current-stream enqueue, completion and failure invalidation | GDN/Radix/Qwen/SGLang names or policies |

A future lossy/int8 checkpoint codec is a distinct algorithm. If brought into
scope later, it requires its own versioned operator contract in
`pypto-kernels`; it cannot masquerade as exact StateBundle copy.

## Pinned SGLang mapping

For the in-scope non-speculative, full-precision Qwen text path, one SGLang
Mamba/GDN slot contains all layers of:

- one or more conv-state tensors; and
- one temporal/recurrent-state tensor.

The allocator reserves slot zero as a dummy/padded target. New slots are
cleared before extend consumes them. A Radix hit performs copy-on-write into a
new active slot; unfinished requests may copy or donate a complete checkpoint.
The active pool's clear/copy operations cover both conv and temporal state.

Pinned source anchors:

- `upstream/sglang/python/sglang/srt/mem_cache/memory_pool.py`
  (`MambaPool.clear`/`copy_from` and layer cache views);
- `upstream/sglang/python/sglang/srt/mem_cache/allocator/mamba.py`
  (request-level slots and reserved slot zero);
- `upstream/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py`
  (donate, COW and checkpoint lifecycle);
- `upstream/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
  (slot/index metadata and tracked state boundaries).

ReplaySSM cursors/rings, speculative/MTP rollback, int8 checkpoints, HiCache,
disaggregation and quantized codecs remain out of scope. The plugin must reject
these modes until separately designed rather than silently copying an
incomplete state.

## Static ABI v1

The eventual public static object is operator-neutral:

```text
StateBundleSpecV1
  schema_version
  ordered components: StateComponentSpecV1[]
  slot_capacity
  reserved_slots
  target_traits_digest

StateComponentSpecV1
  component_id                 # generic stable ID, no model/layer name
  dtype
  payload_shape                # dimensions after slot axis
  payload_strides
  slot_stride_elements
  alignment_bytes
  zero_bit_pattern
```

Rules:

- component order is semantic and nonempty;
- at least one component has a nonzero payload;
- a zero-element component is legal (GDN conv state when width is one);
- the slot stride covers the whole payload and respects alignment;
- components and slot address ranges cannot overlap;
- v1 numeric dtypes must define an all-zero bit pattern;
- target resource/legality identity comes from `TargetInfo`, never hard-coded
  SM120 or SKU constants in this ABI.

The GDN planner supplies two components: BF16 conv and FP32 recurrent. The
generic ABI does not know that they belong to GDN.

## Launch requests

Zero and copy are separate immutable requests:

```text
StateBundleZeroV1
  destination pool/token
  destination physical slots[]
  active mapping count
  next generations[]

StateBundleCopyV1
  source pool/tokens
  source physical slots[]
  destination pool/tokens
  destination physical slots[]
  active mapping count
  next generations[]
```

Runtime-only fields include component pool pointers, mapping-buffer pointers,
the caller's raw current CUstream and completion handle. Pointers, stream,
slots, generation and request identity never enter an artifact key.

V1 mapping rules:

- active destinations are in range and unique;
- copy sources are in range and may repeat for fan-out;
- source and destination slots are disjoint; self-copy, chains and cycles are
  rejected, so no staging workspace is required;
- source layouts/target identity equal destination layouts/target identity;
- source is read-only and destination is full-payload write-only;
- logical payload bytes are touched, not slot padding or inactive slots;
- unsigned byte-address range construction must not overflow;
- metadata is snapshotted once by the host validator; this does not prove
  device metadata contents.

## Generation and lease law

`StateBundleToken` identifies pool, physical slot, allocation generation,
layout digest and target digest. A destination generation must be exactly the
previous generation plus one; stale/ABA tokens fail.

The lifecycle is publish-after-completion, not hardware-atomic rollback:

1. acquire source read pins (copy only) and destination exclusive leases;
2. mark every destination bundle `PREPARING`;
3. enqueue every component on the same caller CUstream;
4. retain all pins/leases until the stream completion edge;
5. publish the new whole-bundle generation only after every component succeeds;
6. if enqueue or completion fails, invalidate the whole destination generation.

No consumer may observe or reuse a partially copied pair. A checkpoint is a
copy of one complete generation at an exact token boundary. Restore is another
copy or a framework-owned slot-map change; Radix node/eviction policy stays in
the plugin/SGLang layer.

## NVIDIA runtime strategy

V1 should prefer `cudaMemsetAsync` and `cudaMemcpyAsync`/2D variants on the
existing current stream. These are allowed runtime activities, not compute
fallbacks, and create no operator CUBIN, operator artifact provenance or
operator tuning record.

If profiling later proves that a batched generic transfer kernel is necessary,
its executable/cache/provenance belongs to `pypto`, keyed by the StateBundle
ABI, ordered component descriptors, mapping capacity, target traits and exact
compiler/runtime/toolchain/build identities. Any tuning is generic runtime
tuning, not GDN tuning.

CUDA Graph support is initially fail-closed. It can be enabled only after
executable, pool/mapping addresses, generations, lease lifetime and completion
storage are stable across capture/replay, with no allocation/compile/load.

## Required gates

Source contract/reference:

- exact schema/type/shape/stride/alignment/capacity validation;
- paired zero and fan-out/reordered copy;
- source immutability and inactive/padding preservation;
- BF16/FP32 canonical storage and width-one empty conv component;
- out-of-range/duplicate destinations, src/dst overlap, chains/cycles,
  component mismatch, address overflow and once-only metadata snapshot;
- stale generation/ABA, source pins, destination exclusivity, completion-only
  publish and whole-bundle failure invalidation.

Continuity:

- run a GDN prefix, copy the complete bundle, then continue the original and
  copied generations for multiple decode steps;
- require identical output, conv state and recurrent state;
- demonstrate that copying only one component breaks continuity and is rejected.

Runtime:

- current CUstream only; no default stream or host synchronization;
- zero/copy → GDN → optional checkpoint copy stream ordering;
- no overlap with an in-flight writer lease;
- no Torch/SGLang/ATen/Triton/FlashInfer compute/copy kernel;
- strict coverage classifies runtime memcpy/memset separately from compute;
- capture remains rejected until its explicit gate passes.

## Current blocker

The legacy PyPTO runtime exposes Ascend/Torch-oriented allocation and H2D/D2H
operations, not generic D2D/memset, raw CUstream, completion events or paired
generation transactions. Do not adapt that interface with backend conditionals.
Land the NVIDIA executable/current-stream contract first, then implement this
design through the generic runtime boundary.
