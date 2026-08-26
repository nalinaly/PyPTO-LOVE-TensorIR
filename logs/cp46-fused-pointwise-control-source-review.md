# CP46 fused-pointwise control source review

## Verdict

GO. P0/P1/P2 = 0/0/0.

Implementation `c98f984ddc4df7cd3354f5fbddadb12df072ed48`, tree
`38b991779d84020f3bbe48944c98df604cf92c2c`, adds exactly nine files and
changes no CP43/44/v4 control or isolation primitive.

The frozen gate contains nine cases and eighteen fresh executable lifetimes:
FP32/BF16 arithmetic, exp, reciprocal and rsqrt plus the FP32
16-input/64-assignment boundary. Reviews verified stage rounding, the signed
zero FMA discriminator, special-value classification/sign, monotonic ULP
checks, direct candidate-to-Torch and candidate-to-CPU checks, 16-element
canaries, runtime tails, distinct reference/candidate streams, packet release,
unload and context clearing.

Exact-source loaders bypass bytecode caches and the validator rejects matching
cache entries before dependency loading. Full synthetic finalization covers
mode-0444 no-replace publication, CPU replay reconstruction and nine classes
of process/control/replay mutation.

Final CUDA-hidden anchor runs `pypto-20260826T042728Z-1382280-ce1fa0` and
`pypto-20260826T042750Z-1382496-07c3e7` retain identical nine-case records,
records SHA `01e8f99d...b844`, and complete per-case HIR/source/BuildSpec/
Artifact/Cubin replay closure with Torch CUDA uninitialized.

Focused run `pypto-20260826T043003Z-1383581-2ec0af` passes 32 tests plus 24
subtests. Source acceptance does not authorize GPU execution or broaden the
fixed-fixture claim.
