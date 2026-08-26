# CP47 fused-pointwise SM120 v2 evidence review

## Decision

Two independent read-only reviewers return GO with P0/P1/P2 equal to zero for
run `pypto-20260826T073309Z-1451510-e48ced`.

The immutable final report is 233,749 canonical bytes, mode `0444`, SHA-256
`d4ffafc053b2924a195f1f046d653233c0076b331f0bc90b075eaa2752eeedf0`.
The provisional SHA-256 is
`e764bde5e4384ea09b994eec27f654ab0493c2eaf317d9bab43ec56591745329`.

## Reconstructed evidence

- Both parent preflights and the child independently admitted the explicit
  policy-2 protected lane at the exact 22 GiB host floor. The 16 GiB owned-run
  abort floor, 4 GiB GPU-free floor, 64 GiB disk floor and 1,800 second timeout
  remained unchanged.
- Every pre-release, periodic, child and post-exit observation contains zero
  external/protected NVIDIA compute PIDs, zero protected runtime mappings and
  zero unreadable protected maps. The owned process group exited with no
  survivor and no external process was signalled.
- All 142 replay files have the exact expected name, size, SHA-256 and mode
  `0444`. The nine HIR, TensorIR source, BuildSpec, Artifact and Cubin families
  join the frozen compiler anchors and PyPTO `b83fcd3` identity.
- The eighteen fresh executable lifetimes use distinct non-default reference
  and candidate streams, externally synchronize, retain and release every
  launch packet, unload explicitly and end with no bound CUDA context.
- All sixty input before/after hash joins and seventy-eight guard/canary records
  match. Arithmetic, reciprocal and maximum cases are bit-exact. FP32 and BF16
  FMA-discriminator lanes are exact negative zero (`0x80000000` and `0x8000`).
- Candidate versus Torch is zero ULP for all eighteen executions. The only
  nonzero independent-CPU differences are at most one ULP for FP32 exp/rsqrt,
  below the frozen four-ULP ceiling. Special classes and signs match; no
  subnormal or `high_precision` behavior is claimed.
- The CPU-only finalizer reports Torch CUDA uninitialized. A second exact
  finalization exits 1 because the immutable report already exists; the report
  SHA-256 and mode remain unchanged and no partial replacement exists.

## Immutable joins

- Process: `6eb362053c3da2ce9f4704f3a49698981d4ca6f8bb6d380a4467e7de020fa9fe`
- Initial preflight: `1da3438352fb7a66cf758f02e1ce0d0b787c9f6dc64267ccbee93c8520df91ad`
- Action preflight: `361fb740e045aa93483dc6e5d4c98c8de22d3233d3c6bdc2e6f1e4f2d301f57d`
- Gate: `f70b6c81615e40f7c9a93e3bfe359e40c5f7445ad24c0bdde4a80c8615734182`
- Barrier: `80f079604e3572f4a342f8093db075d548809d7abb3206990aac3a84973fb557`
- V2 control manifest: `d3b16079c811dd2fbe610ba264d81117e8c4a44886b74caaddb684df2d467036`
- Frozen compile anchors: `584f6755bbd248de5bb6ddd3ff610da8082667bc892a6cff6583ea42d4c44c97`

## Accepted boundary

This accepts correctness only for the frozen nine FP32/BF16 fused-pointwise
cases on the exact SM120/compiler/control identity. It does not accept general
FusedPointwiseV2 correctness, other shapes or chains, subnormal or
`high_precision` behavior, cross-build Cubin determinism, performance, CUDA
Graph, TorchInductor, SGLang, Qwen correctness or strict coverage.
