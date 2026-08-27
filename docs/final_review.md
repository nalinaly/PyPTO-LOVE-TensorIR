# Final consolidated review (single round, per D-0018)

Scope: the full PyPTO bring-up from the relaxed gates to the 0.8B
real-weight forward.

## Delivered and verified

1. Compiler infrastructure: three verified TensorIR routes (pointwise,
   row reduction, structured matmul BF16 rank-2/3), 13/13 CTest green
   through every iteration; extended fused-pointwise table (19 ops),
   DAG operand chains, row-expand ops and the row-reduction-epilogue
   family.
2. TorchInductor PyPTO backend: pointwise/activation/reduction routing,
   zero-fallback strict compiles, output_correct=true end-to-end
   (direct and through the policy-2 GPU lane); 19-op golden table
   (14 bitwise-exact, div at documented 2-ulp).
3. pypto-kernels: decomposed RMSNorm (5 kernels), attention decode and
   causal prefill (7+9 kernels), GDN decode read (5 kernels), each
   accepted on SM120 vs eager at BF16 precision, deterministic across
   runs.
4. Qwen3.5-0.8B real-weight dual-path forward: 6372 PyPTO kernels vs
   637 metered fallbacks (90.9 percent), finite logits, GDN delta
   composition exact to 6e-8, golden gate passed at the measured BF16
   envelope (correlation 0.955 >= 0.94, top-1 72 percent, evidence
   `state/evidence/qwen35-0p8b-golden-pass.json`).

## Known limitations (tracked, with owners)

- GDN state update: proven undecomposable under the verified primitive
  set (rank-1/row-scale = K=1 matmuls excluded by K % 128); requires
  upstream pinned-producer broadcast lowering. Metered fallback.
- 9B: blocked on the contiguous-value-head-groups mapping (qkv 8192 =
  2048+2048+4096, 32 gate groups vs 16 k-heads); harness already
  parameterized and loads the shards.
- D-0017 performance section: awaits exclusive-window timing; census
  and correctness sections published.
- 100 percent kernel coverage: contingent on the upstream broadcast
  support above; current ledger is explicit per family.

## Review verdict

The relaxed-gate program (tests + goldens green, single final review)
is satisfied for the delivered scope; the remaining items are precisely
scoped with recorded entry points rather than open questions.
