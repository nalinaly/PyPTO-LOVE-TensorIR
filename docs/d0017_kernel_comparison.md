# D-0017: PyPTO vs SGLang-default per-kernel comparison

Status: census + correctness sections complete; the performance section
requires exclusive-GPU timing runs (policy gate) still to be executed.

## Method

Per-model-layer kernel census over the Qwen3.5-0.8B real-weight forward
(`benchmarks/operators/pypto_qwen35_0p8b_forward_sm120.py`), PyPTO-routed
path vs pure-torch eager reference (the offline substitute for the
SGLang-default baseline; transformers is not installable in this
environment). Each PyPTO kernel family is separately accepted against
eager at BF16 precision (CP-0050 through CP-0054).

## Kernel inventory (PyPTO path, 0.8B, prompt 32)

| family | kernel | count |
|---|---|---|
| projections + LM head | StructuredMatmulV4 | 24 layers x (qkv/a/b/z or q/k/v/o + gate/up/down) + head |
| norms | FusedPointwiseV2 5-kernel RMSNorm decomposition | 2/layer + final |
| full attention (6 layers) | 9-kernel decode/prefill decomposition | 9/layer |
| GDN read (18 layers) | 5-kernel read decomposition x2 (q/k) | 10/token/layer |
| MLP activation + residuals | FusedPointwiseV2 | 3/layer |
| TOTAL | | 6372 pypto vs 637 fallback (90.9%) |

Fallback ledger: GDN state update (CP-0055 impossibility, upstream
producer broadcast support required), RoPE, L2 norms, causal conv,
embedding gather, per-token GDN vector algebra.

## Correctness

Both paths produce finite logits; top-1 agreement 69-72 percent, mean
abs diff ~2.26 on +-18 logits. BF16-aligning the reference recovered
only ~3 points, so the residual needs per-layer bisection (attention
kernel score rounding is the prime suspect) before the golden gate can
be declared passed.

## Performance (pending)

Exclusive-window timing of each kernel family vs the eager op it
replaces, then the same for 9B once its GDN v-group mapping lands.
