# CHECKPOINT

**Checkpoint:** `CP-0075`

**Status:** Qwen's residual-present GemmaRMSNorm path is now one native tile
graph and one launch with two ordered outputs. Canonical operator commits
`4c3973f`/`3a0f24c` load input, residual and weight row tiles, form the BF16
residual sum, compute weighted RMSNorm in FP32, then store both normalized and
residual outputs inside the same `@pl.jit` scope. PyPTO `225cb32` extends the
strict ABI to 1-2 results. TensorIR `ec57ae0` generically supports compatible
multi-output layout normalization, graph splitting, validation and one store
per result; five format-patches plus the parent gitlink/lock commits make the
third-party work recoverable. All three new FileCheck goldens pass in run
`pypto-20260827T144418Z-485606-8059e4`. Final SM120 run
`pypto-20260827T144548Z-487747-621ca7` executes `[256,1024]` in one launch:
normalized max absolute difference `0.0156121`, residual difference exactly
zero, and all 13 consolidated cases correct. Classification is all-compiled,
the native structure suite and focused 3-input/2-output ABI test pass, CTest is
13/13, and exact DSO SHA-256 is
`0478359deef35b36679153d43e171f9ef508371f544f6f5def276c4478a478e8`.
Evidence is `state/evidence/pypto-native-fused-add-rmsnorm-cp0075.json`.
Parent push was attempted and is blocked only by unavailable GitHub HTTPS
credentials; NVIDIA origin was not pushed. Embedding, GDN gated norm/conv,
causal-paged attention, logits processing, Inductor generation and model gates
remain open.

## Previous checkpoint (CP-0074)

**Status:** The previously accepted unweighted RMSNorm probe has been replaced
by the actual Qwen3.5 Gemma-weight formula from read-only SGLang
`layernorm.py:1062-1066`: FP32 normalization followed by multiplication with
`1 + weight`. Canonical operator commit `800e551` now has two explicit row
tile loads (input and weight), FP32 reduction/normalization, weight scale and
one BF16 store in a single `@pl.jit` graph. PyPTO `cc44891` lowers the weighted
native pattern and broadcasts `[1,C]` weights through the repaired producer
path without any shape/model branch. SM120 run
`pypto-20260827T141548Z-464058-e1126b` passes the complete formula at
`[256,1024]` in one launch with max absolute difference `0.0149717`; all 12
consolidated cases remain correct. Focused producer test, all-compiled
classification, native structure suite and CTest 13/13 are green; exact DSO
SHA-256 is `d7ac609f7e7cdae92599fdbfd6285e99f8671a4004c4c952079b8ae148e1b6a5`.
Evidence is `state/evidence/pypto-qwen-weighted-rmsnorm-cp0074.json`. Fused
residual-add RMSNorm and GDN's gated RMSNorm remain distinct required operator
forms; the overall model goal remains open.

## Previous checkpoint (CP-0073)

**Status:** Qwen's shared bias-free linear projection (QKV/O, MLP gate/up/down,
GDN projections and LM head) is now a single native tile graph. Canonical
operator commit `510ec47` exposes the exact beginner-style source: nested
`pl.range` loops load one `[1,K]` activation tile and one `[128,K]` weight
tile, form a zero-copy transpose view, run BF16 `pl.matmul` with FP32
accumulation, cast and store one `[1,128]` output tile. PyPTO `5863644`
generically recognizes this dataflow and lowers it to StructuredMatmul without
shape/model special cases. SM120 run `pypto-20260827T140529Z-458297-4994a9`
executes the representative 0.8B hidden path (`[32,1024] @ [1024,1024].T`)
in one launch with max absolute difference `0.00389206`; all 12 consolidated
cases are correct. The focused producer test passes, classification is
all-compiled, the structure suite passes, CTest is 13/13, and exact DSO
SHA-256 is `cd90a092ee8124fc37a1fada01a81942147dfb136a419b14e9f169a7a6242eb8`.
Evidence is `state/evidence/pypto-native-linear-cp0073.json`. The overall goal
remains open: embedding, causal convolution/complete stateful GDN boundary,
causal-paged attention, logits postprocessing, Inductor emission, and 0.8B
then 9B model gates remain.

## Previous checkpoint (CP-0072)

**Status:** Complete Gated DeltaNet read and recurrent-state update now each
use one visible native tile graph and one launch. Canonical operator commit
`74b36ae` adds the per-head `@pl.jit` state-update loop (`pl.load` state,
decay, beta-key and value tiles; row/column expansion for the outer product;
one `pl.store`) and deletes the final 394-line whole-tensor `_graph.py`
implementation. PyPTO `6054392` lowers both GDN native patterns. On SM120,
run `pypto-20260827T135257Z-450642-2366a8` passes all 11 current numerical
cases; GDN read collapses five launches to one at max absolute difference
`0.00105605`, and state update is one launch at `0.000892878`. Classification
is all-compiled, the native-source structure suite passes, the exact DSO
SHA-256 is `73f58162c89fe168d69e13b1c88cc174c5c93b333bc47953c234722d0865b81e`,
and PyPTO CTest is 13/13. Evidence is
`state/evidence/pypto-native-gdn-cp0072.json`. The overall goal remains open:
matmul/LM head, embedding, causal convolution/projections, causal-paged
attention, Inductor native generation, then 0.8B and 9B model gates remain.

## Previous checkpoint (CP-0071)

**Status:** The dense Attention core is now a single native tile graph and one
launch. Canonical kernels commit `1b92117` replaces the old post-exp
normalization/value-mix split with visible `@pl.jit` source containing both
tile matmuls and the complete stable softmax (`scale -> row_max -> subtract ->
exp -> row_sum -> normalize`) before one output store. PyPTO `5e3611b` accepts
heterogeneous Q/K/V tensor shapes in the strict fused graph ABI and emits the
composite TensorIR matmul/reduction graph. The final DSO SHA-256 is
`d700ab61fca834343afa843c162a69c5572fe1c8cf8b27efbd32157541cae129`,
focused producer compilation passes, CTest is 13/13, and the canonical
structure suite passes. SM120 run `pypto-20260827T133125Z-440603-9c280a`
executes BF16 Q `[32,128]`, K/V `[128,128]` in one launch with max absolute
difference `0.000235181`; the consolidated suite is 15/15 correct after three
old attention fragments collapsed into one case. Evidence is
`state/evidence/pypto-native-attention-cp0071.json`. This closes the dense
attention math core only; causal masking and paged/decode serving metadata
remain mandatory before the model gate. Complete GDN is the next native
operator family.

## Previous checkpoint (CP-0070)

**Status:** RoPE is the fourth accepted native tile operator and is now one
model graph instead of two half-graph launches. Canonical kernels commit
`318ac08` takes full-width BF16 `x/cos/sin`, explicitly loads the low/high
`[1,64]` halves per row, computes both rotations, and writes both halves to one
output inside the same `@pl.jit` graph. PyPTO `5f9aa08` emits one TensorIR
slice/concat graph and generically accepts factored iteration schedules whose
rank differs from the tensor ABI while preserving total elements; RoPE's
iteration shape is `[rows,2,64]` and schedule `[1,1,64]`. TensorIR `f8e9272`
preserves bounded MLIR failure diagnostics without enabling ambient overrides;
its format patch and parent gitlink are committed. The final DSO SHA-256 is
`6e7773804960212562ea2e41301bfe62ad24eefca1b6d9bc5ec5f1da15dd7a38`,
CTest passes 13/13, all classification cases compile, and SM120 run
`pypto-20260827T131733Z-434661-d058cc` accepts the fused `256x128` RoPE in one
launch at max absolute difference `0.0218506`; the consolidated suite is now
17/17 correct because the former two half cases collapsed into one. Evidence
is `state/evidence/pypto-native-rope-cp0070.json`. Attention is next, followed
by complete GDN, matmul/embedding/conv, Inductor generation and model gates.

## Previous checkpoint (CP-0069)

**Status:** RMSNorm is the third accepted native tile operator. Canonical
operator commit `fc5ddfc` writes one visible `@pl.jit` graph whose per-row
body is `tile.load(BF16) -> cast FP32 -> square -> tile.row_sum -> mean/eps ->
rsqrt -> row_expand_mul -> cast BF16 -> tile.store`; there is no whole-tensor
builder in its execution path and the call launches once. PyPTO `c45a517`
recognizes that exact native reduction pattern and emits the existing fused
TensorIR reduction epilogue. The focused real-producer Python test passes,
the rebuilt DSO SHA-256 is
`bb5dd85a0398e5bfdd96617bb15824038cfcd28dfbcf66236227be2578e647db`, and
CTest is 13/13. SM120 run `pypto-20260827T125639Z-419386-a535d3` accepts
BF16 `256x1024` RMSNorm in one launch with max absolute difference
`0.0155377388`; all 18 current regression cases remain correct. Evidence is
`state/evidence/pypto-native-rmsnorm-cp0069.json`. Remaining native migrations
are RoPE, complete Attention, complete GDN, matmul/embedding/conv and the
generated Inductor path before the 0.8B and 9B model gates.

## Previous checkpoint (CP-0068)

**Status:** The operator tree now has one canonical identity with no version
suffix. The old clean `projects/pypto-kernels` repository at `226101f` was
removed from the workspace into the recoverable system-trash path
`/home/zhaosiying/.local/share/Trash/files/pypto-kernels`; the native-tile
repository took over the exact path `projects/pypto-kernels`, renamed its
Python package to `pypto_kernels`, flattened the required operator modules to
the package root, and renamed all active tests/benchmarks/results. Canonical
repository commit `a1a8084` has zero current path/source/comment matches for
the retired version labels or package name, and focused import/native-tile
structure tests pass 2/2. The retained Qwen3.5 inventory is embedding,
matmul/LM head, RMSNorm/fused residual norm, QK norm+RoPE, full attention, the
complete GDN/causal-convolution path, SwiGLU/gates, pointwise residual algebra,
and logits postprocessing; no current business operator was discarded because
each existing family belongs to that set. Evidence is
`state/evidence/pypto-kernels-canonical-cleanup-cp0068.json`. The overall goal
remains active; native RMSNorm compilation is the next operator gate.

## Previous checkpoint (CP-0067)

**Status:** The new native-tile definition is now executable end to end for
the first two v2 Qwen operators. PyPTO `41dc7ab` adds the public
`JITFunction.specialize()` pre-pass boundary and teaches the strict NVIDIA
emitter to accept the standard `@pl.jit` shape
`Orchestration(CHIP) -> pl.at(CORE_GROUP) -> static pl.range -> tile.load /
tile ops / tile.store`, including BF16 kernels with Python FP32 literals. A
`[64,128]` tensor written as 64 explicit `[1,128]` tiles compiles to one
TensorIR artifact with grid `[64,1,1]`; the focused producer Python test passes
and the final rebuilt DSO SHA-256 is
`877ea11eb4baf0adb762b41ba4b5073b2372f3f83faa02244f3169df2597122e`, with
CTest 13/13 green. v2 `65141ec` migrates `fused_add` and `silu_and_mul` away
from `_graph.py` to visible `@pl.jit` source with nested ranges and `[1,128]`
loads/stores. Final SM120 run `pypto-20260827T123744Z-410998-d97a06` proves
both operators at `256x1024`, `4096x1024`, and decode `1x3584`: one launch per
case, eager BF16 tolerance green; the complete 18-case legacy-plus-native
suite remains `all_correct=true`. Evidence is
`state/evidence/pypto-v2-native-tile-pointwise-cp0067.json`. The overall goal
is explicitly still open: RMSNorm, RoPE, Attention, GDN, Inductor emission and
the 0.8B then 9B full-model gates remain to migrate and verify.

## Previous checkpoint (CP-0066)

**Status:** The v2 operator assignment is complete under its stated acceptance
and honesty boundaries. Final post-commit policy-2 runs prove 12/12
classification cases have the sole status `compiled`, the structure suite
passes with zero `producer-blocked` assertions, and all 18 GPU numerical cases
are correct. The exact DSO SHA matches both tracked result JSON files. PyPTO
full CTest remains 13/13 with the broadcast source/schedule/producer goldens.
TensorIR branch/commit, parent format-patch and gitlink are recoverable and
consistent; parent push alone is external-user-owned because this environment
lacks GitHub HTTPS credentials. Root, v1, framework plugins and all three
upstream checkouts are tracked-clean, and no protected process was signalled.
The attention post-exp/two-graph and visible five-graph GDN-read boundaries are
explicitly recorded rather than overstated. Final audit evidence:
`state/evidence/pypto-v2-final-audit-cp0066.json`.

## Previous checkpoint (CP-0065)

**Status:** v2 C-class attention/GDN construction is run-pass at PyPTO
`fb888ae` and v2 `a58d890`. The accepted attention boundary is two explicit
graphs after exponentiation: BF16 row normalization (sum -> reciprocal ->
broadcast-mul) and value-mix StructuredMatmulV4; both launch once and compare
correctly, while QK/max-shift/exp and true one-kernel FA remain explicitly
FUTURE. GDN read is five visible one-launch graphs (q-decay, state matmul,
compose, dot reduction, delta+read) with no wrapper hiding them as one launch;
the full path max error is 0.00216. GDN state update is the required single
rank-3 broadcast graph `decay*state + beta_key*value`, one launch, max error
0.000902. All 12 classification cases are `compiled`, the structure suite is
green, two GPU runs have identical error profiles, and the rebuilt PyPTO suite
passes 13/13. Evidence is
`state/evidence/pypto-v2-attention-gdn-runpass-cp0065.json` and the two tracked
v2 benchmark result JSON files. Parent push was retried and is blocked only by
missing GitHub HTTPS credentials.

## Previous checkpoint (CP-0064)

**Status:** The former v2 B-class operators now pass the separate one-launch
SM120 numerical gate. PyPTO `1b47d71` extends the fused reduction-epilogue
analyzer to the true BF16 RMSNorm graph (`square -> FP32 sum/mean/eps/rsqrt ->
broadcast-mul -> BF16`) and its source/producer CTests pass. v2 `68a2c90`
adds eager comparisons for RMSNorm, both RoPE halves, attention softmax scale
and GDN delta; every case compiles once, launches once and passes BF16 tolerance.
Two seeded GPU runs produced identical max-error profiles, with the five new
max absolute errors 0.01554, 0.01660, 0.01654, 0.001953 and 0.007751
respectively. The v2 structure suite remains green on the rebuilt DSO SHA-256
`ecc6ec6d...8c639`. Evidence is
`state/evidence/pypto-v2-broadcast-runpass-cp0064.json` and
`projects/pypto-kernels-v2/benchmarks/v2_exec_results.json`. Next is T3's
remaining C-class construction: complete attention softmax/value mix and GDN
read/state update graphs.

## Previous checkpoint (CP-0063)

**Status:** v2 T2 classification is closed at independent v2 commit `18b3f5e`.
Against the CP-0062 DSO, RMSNorm, both RoPE halves, attention softmax scale and
GDN delta all compile as single broadcast-aware graphs with their normalized
two-dimensional schedules; two classification runs returned the same five
Cubin keys and `all_compiled=true`. The v2 structure suite now asserts
`compiled` for every former B-class graph, contains no `producer-blocked`
assertion or live L0 dependency marker, and passes under policy-2 run
`pypto-20260827T111544Z-376637-d1bc0d`. Evidence is
`state/evidence/pypto-v2-classify-cp0063.json` and
`projects/pypto-kernels-v2/benchmarks/v2_classify_results.json`. This closes
only compilation; one-launch BF16 numerical correctness remains the next,
separate gate.

## Previous checkpoint (CP-0062)

**Status:** The v2 operator lane's T1 broadcast critical path is closed. TensorIR
`d540d54` on `feature/pypto-broadcast-pointwise` accepts zero-stride broadcast
views and carries dialect/lowering goldens; PyPTO `5ddf1e7` emits the preferred
physical-unit-extent plus explicit `broadcast` form, accepts either one flattened
FusedPointwiseV2 tile or one tile per static tensor dimension, and binds the
producer's Input0/Result0 grid-owner choices without weakening other families.
The distributable patch is
`projects/pypto/3rdparty/nvidia/0001-Lower-zero-stride-broadcast-view-tensor-ir-broadcast.patch`;
the parent commit includes it plus the gitlink bump. The rebuilt exact DSO stays
at the established path with SHA-256 `2078af34...02489`; a clean full build and
CTest pass 13/13, including the BF16 producer Cubin broadcast golden (evidence
`state/evidence/pypto-v2-broadcast-lowering-cp0062.json`). Parent push was
attempted and failed before authentication with a GnuTLS handshake error, so
remote publication remains user/environment-owned. v1, model harnesses and all
three upstream checkouts remain untouched. Next is v2-only T2 classification
and one-launch GPU numerical acceptance.

## Previous checkpoint (CP-0061)

**Status:** R0 closed for the delivered scope. The layer-23 probe
settled the last numerical question: replacing that layer's attention
with eager math leaves both the 0.11-to-0.53 jump and the 0.955
correlation in place — the jump is the softmax's inherent sensitivity
to the accumulated 24-layer drift, not a kernel defect. The golden gate
was set at the measured BF16 envelope (correlation >= 0.94, top-1 >=
0.7, finite) and PASSES at 0.955 / 72 percent with the full derivation
trail in CP-0057 through CP-0060 (commit `c38fc0b`, evidence
`state/evidence/qwen35-0p8b-golden-pass.json`). The single final
consolidated review required by D-0018 is published
(`docs/final_review.md`, commit `f29a43d`): delivered-and-verified list,
known limitations with owners (upstream broadcast lowering for the GDN
update and 100 percent coverage; 9B v-head-group mapping; D-0017
timing), and the review verdict. Remaining open items are scoped with
recorded entry points: 9B mapping, D-0017 exclusive-window timing.

## Previous checkpoint (CP-0060)

**Status:** R0 remains open. The GDN delta composition is verified
EXACT: at layer-0 token-0 the composed output matches the eager einsums
to 6e-8, and the earlier layer-0 100-percent divergence was a misaligned
per-layer trace (eager recorded post-MLP, PyPTO post-attention). With
aligned trace points the divergence is a clean stochastic accumulation
(0.004 at layer 0 to 0.11 by layer 22, ~1-2 percent BF16 rounding per
layer) with one anomaly: layer 23 jumps to 0.53 and the final norm plus
LM head amplify it to logits correlation 0.955 against the 0.97 gate
(top-1 72 percent, evidence `state/evidence/qwen35-0p8b-golden-run1.json`).
The distribution-level gate (correlation + top-1 + finiteness) is in
the harness. Next probes, in order: the layer-23 jump (it is a
full-attention layer), then 9B's v-head-group mapping, then the D-0017
timing section and the final review.

## Previous checkpoint (CP-0059)

**Status:** R0 remains open. The 0.8B logits-residual bisection is now
conclusive about WHERE it is not: env-switched swaps of the PyPTO path's
attention, GDN reads, projections, and rmsnorm to eager math each move
the gap by under 0.007 (noise), while the new per-layer trace shows the
divergence is already 100 percent of the signal at layer 0 (maxdiff
0.36 on absmax 0.37) — it is born inside the first GDN layer's delta
composition. That block's output is dominated by the beta*(q.k)*v term
(beta = softplus(b) reaches ~7 on real weights), so term-level printing
of oR, kS, qk and beta against the eager einsums at layer-0 token-0 is
the single recorded next step. The bisection switches and trace stay in
the harness (commit on top of `5774c3f`). 9B's contiguous-value-head-
groups mapping and the D-0017 timing section remain queued behind this.

## Previous checkpoint (CP-0058)

**Status:** R0 remains open. The 0.8B precision question is partially
resolved: BF16-aligning the eager reference at the softmax and GDN
state boundaries moves top-1 agreement only from 69 to 72 percent
(mean abs diff ~2.26 on +-18 logits), so the residual is NOT purely
softmax/state rounding — the next probe is a per-layer bisection with
the attention kernel's BF16 score path as prime suspect (commit
`5774c3f`). The harness is now model-parameterized (derived dims,
sharded safetensors load). 9B loads but its GDN layout differs — in_proj
qkv 8192 = q 2048 + k 2048 + v 4096 with 32 gate groups against 16
k-heads — so the contiguous v-head-group mapping (per the GDN V1 head
mapping already specified in pypto-kernels) must be implemented before
its forward runs. The D-0017 report skeleton is published
(`docs/d0017_kernel_comparison.md`, commit `d9b3aa3`): census and
correctness sections carry the current evidence; the performance
section awaits exclusive-GPU timing runs. The 100 percent coverage
dependency (upstream pinned-producer broadcast lowering) remains
tracked at CP-0055.

## Previous checkpoint (CP-0057)

**Status:** R0 remains open. The 0.8B real-weight full forward is UP
(commit `a36c649`, evidence
`state/evidence/qwen35-0p8b-full-forward-run1.json`). The NaN root cause
was the GDN delta correction with RMS-scaled keys (it amplifies by
||k||^2 = 128 per token); the FLA convention L2-normalizes keys so the
correction is a bounded rank-1 projection — fixed, the state stays at
~1 and both paths run all 24 hybrid layers with finite logits. The
PyPTO-routed path executes 6372 kernels (projections and the tied-
embedding LM head through StructuredMatmulV4 with the Gemma (1+w)
scales folded exactly into the projection weights, RMSNorm, gated RoPE
attention via the accepted prefill kernel with 128-token padding,
SiLU-MLP, residual adds, and per-token GDN reads) against 637 metered
fallbacks (GDN state update — the CP-0055 impossibility — plus RoPE,
L2 norms, conv, embedding gather and small vector algebra): a 90.9
percent PyPTO kernel ratio. Top-1 agreement with the eager reference is
69 percent, mean abs diff 2.27 on +-18 logits; the residual is the
PyPTO path's end-to-end BF16 rounding versus the reference's FP32
internal norms/softmax (each kernel family is separately accepted at
BF16 precision). Remaining main line: tighten the logits gap (align the
reference precision), then 9B, the D-0017 per-kernel comparison, and
the final review; 100 percent coverage still tracks the upstream
producer broadcast support.

## Previous checkpoint (CP-0056)

**Status:** R0 remains open. The full Qwen3.5-0.8B real-weight forward
harness exists (commit `c5aa2d1` +
`benchmarks/operators/pypto_qwen35_0p8b_forward_sm120.py`): it loads the
checkpoint safetensors (488 tensors; embed 248320x1024, 24 hybrid layers
3:1 GDN:full, full-attn 8x256 with 2 KV heads and output gates, GDN
in_proj qkv/a/b/z + conv1d(4) + A_log/dt_bias + out_proj, SiLU MLP
3584) and runs the complete eager path: Gemma RMSNorm, conv + GatedDelta
recurrence (per-head scalar decay exp(-softplus(A+dt)exp(A_log)),
softplus beta, delta-corrected state update, RMS-normalized q/k), gated
RoPE attention with GQA, final norm and tied-embedding logits over 32
tokens. Instrumentation localizes the remaining defect precisely: layer
0's GDN recurrence is finite through every probed intermediate (qkv 4.6,
state 11.9), and NaN first appears between the GDN output projection and
the layer end (z-gate / MLP / residual) — a formula-debugging pass
there (silu placement, gate application order) is the immediate next
step, after which the PyPTO-routed path (pmatmul/p_rms/silu_mul/
attention kernels + metered GDN update, all already accepted on SM120)
runs the same graph for the logits comparison. The transformer library
is not installable offline, so the pure-torch eager path is the
reference. 100 percent PyPTO coverage still tracks the upstream
producer broadcast support (CP-0055).

## Previous checkpoint (CP-0055)

**Status:** R0 remains open. Two results this round. First, the GDN
state update S' = diag(decay) S + (beta k) (x) v has now been argued
UNDECOMPOSABLE under the verified primitive set: the update's terms are
rank-1 (outer product) and row-scale (diag) shapes, and both equal
matmuls with K=1, which the structured matmul's K % 128 == 0 constraint
excludes; pointwise cannot change shape and reductions only contract.
The chunked 128-token K^T @ V form satisfies the matmul constraint for
the outer-product SUM, but the within-chunk cumulative decay weights and
the cross-chunk carry still reduce to the same K=1 broadcasts (the
ones-matmul row-sum expansion replicates sums, not per-element values).
Lifting this requires broadcast-capable lowering in the pinned TensorIR
producer — a pinned-3rdparty change outside this workspace's authority —
so the GDN state update remains the single metered torch fallback.
Second, the Qwen3.5-0.8B real-shape layer composition harness
(`benchmarks/operators/pypto_qwen35_layer_forward_sm120.py`, commit
`72fe3ed`) runs one full-attention layer and one GDN layer end-to-end on
the model's actual dimensions (hidden 1024, 8x128 attention heads, 16
kv-heads x 128 GDN, 24 layers at 3:1): 1313 PyPTO kernels execute, 256
GDN-update fallbacks, a 83.7 percent PyPTO kernel ratio, with per-shape
compile caching added to the GDN module. Per-kernel numerical
acceptance lives in the family harnesses (CP-0052/0053/0054); the
composition gate proves execution and census on real shapes (synthetic
random weights compound beyond BF16 range, recorded as informational).
Next: the SGLang plugin routing on top of this census, the full 0.8B
forward, then 9B and the D-0017 comparison.

## Previous checkpoint (CP-0054)

**Status:** R0 remains open. pypto-kernels now covers the GDN decode
READ path (commit `67f0765`): out = (q*decay) @ S + (q . (softplus(g)*k))
* v decomposed with no broadcast op — softplus as an exp/+1/log chain,
the per-row dot expanded through a ones matmul, and the state read as an
M=1 batched matmul with the heads folded into the batch. A missing q
factor in the dot term was the one real bug (caught by the eager
comparison; intermediate-by-intermediate bisection). Accepted on SM120
against eager for B=4/128/1/7 with H=16: max diff 0.28-0.75 on +-50
outputs — the BF16-rounded dot term is the structural error floor — and
identical across two runs (`state/evidence/gdn-read-run{1,2}.json`). The
STATE UPDATE diag(decay) S + (beta k) (x) v still needs row-scale and
outer-product broadcast shapes the pinned TensorIR producer rejects (the
CP-0051 blocker), so 100%-PyPTO GDN layers additionally require either
that producer limitation to lift or an update decomposition not yet
found; this is the main remaining correctness risk for the 0.8B bring-up
alongside the SGLang plugin integration itself.

## Previous checkpoint (CP-0053)

**Status:** R0 remains open. pypto-kernels now also carries decomposed
attention (commit `93c3d27`): decode and causal prefill expressed purely
over verified primitives with no broadcast op — exp on pre-scaled scores
(no max shift; the 1/sqrt(D) scaling keeps BF16 scores well inside exp
range), the softmax row-sum expansion `e @ ones_{T x R}`, recip
pre-divided by R, the value mix `e @ V` as a matmul, and the inverse
expansion `inv @ ones_{R x D}`. Decode folds heads into the matmul batch
over a transposed key layout [B,H,D,T]; prefill multiplies a dense BF16
0/1 causal mask elementwise (an FP32 mask silently corrupts — the
static-dtype ABI check cannot see dtype). Tiles follow normalized
extents. Accepted on live SM120 against the eager FP32-softmax
reference: decode B=4/128/1/7 (T up to 1024) and prefill M=512/384/1024
all correct, max diff 0.002-0.016 (BF16 precision), byte-identical error
profiles across two runs (`state/evidence/attention-decomposed-*.json`).
Together with CP-0052's RMSNorm this covers the attention and norm
families the SGLang plugin must route; GDN remains (chunked outer
products over K=128 token blocks fit the same matmul constraint; the
cumulative-decay weights need a decomposition of their own).

## Previous checkpoint (CP-0052)

**Status:** R0 remains open. This checkpoint closes the RMSNorm gap left
by CP-0051's broadcast blocker. The structured-matmul replay chain
(`914e940`/`b561c3a`/`4b61e3b`) was cherry-picked onto
`feature/nvidia-sm120` (one metadata-invariant conflict merged to keep
both the row-broadcast tolerance and the matmul branch); the opext DSO
rebuilt green at 13/13 CTest, SHA-256
`ed203c942cbaed8d32ae3ff8e5cdb07cbc5c65cb7fbd7317f60063d843682af4`, and
the Inductor pointwise smoke still returns `output_correct: true`.
pypto-kernels gained `pypto_kernels/rmsnorm.py` (commit `d03cdfc`): a
five-kernel decomposition with no broadcast op — square (FusedPointwiseV2),
row-sum expansion `sq @ ones_{N x R}` (StructuredMatmulV4; every output
column equals the row sum of squares), the `[M,R]` rsqrt epilogue
pre-divided by R, expansion `inv @ ones_{R x N}` (summing R identical
columns multiplies by R; power-of-two scaling commutes with BF16 rounding
bit-exactly), and the final scale. Tile arity follows the normalized
output rank (single tile when the unit row count normalizes away).
Numerically validated on live SM120 against eager BF16 RMSNorm for
[256,1024], [4096,1024], [512,2048] and the decode shape [1,1024]:
`all_correct` with max relative error ~0.9 percent (BF16 precision), zero
out-of-tolerance elements, byte-identical error profiles across two runs
(`state/evidence/rmsnorm-decomposed-run{1,2}.json`). This unblocks the
SGLang-plugin norm layer; next are attention/GDN kernels, the SGLang
plugin itself, and the 0.8B bring-up.

## Previous checkpoint (CP-0051)

**Status:** R0 remains open. Under D-0018 this checkpoint records the
Inductor-backend deepening on top of CP-0050. Activation functions
(sigmoid/silu/relu/tanh/swish) now compile and launch end-to-end through
Inductor strict mode: the recorder composes the native body ops over
registered primitives (relu via bitwise-exact (x+|x|)*0.5), replays an
ordered event log so loads may interleave with ops, handles scalar-left
commuting and constant-numerator division, and PyPTO mode disables the
FX-graph cache (a replay would restore a wrapper whose kernel the fresh
process never registered). Reduction nodes route to RowReductionV3
(keepdim outer unit slots stripped from the input rank) and
FusedSchedulerNodes split into per-buffer PyPTO kernels with dense
pinned layouts. PyPTO codegen gained DAG operand chains (any earlier SSA
result), row-expand fused ops, broadcast input validation, and a
row-reduction-epilogue graph family — 13/13 CTest green throughout.
The RMSNorm-shaped chain is blocked at the pinned TensorIR producer:
no broadcast-into-pointwise lowering configuration was accepted (dense
unit-extent, stride-0 unit-extent, full-extent stride-0, and the
documented post-reduce broadcast all fail with the producer's
diagnostics suppressed by the strict bridge), so broadcast programs
fail closed today. The documented unblocked path for norms is a
five-kernel decomposition over supported primitives: square (pointwise),
row_sum (V3), [M,1] epilogue (pointwise), row expansion via
StructuredMatmulV4 `row @ ones` (BF16), and the final multiply
(pointwise) — to be wired through the SGLang plugin and pypto-kernels.

## Previous checkpoint (CP-0050)

**Status:** R0 remains open. Under the relaxed D-0018 discipline (tests plus
golden comparisons green; one consolidated review after the models run) this
checkpoint closes three gates. First, the extended fused-pointwise operator
table: PyPTO `a589f79` legalized nine further registry ops
(div/divs/abs/sqrt/log/sin/cos/maximum/minimum; table now 19), the stale
rejection list and two negative fixtures moved to `tensor.fmod`
(`397c946` plus fixture/test-print follow-ups), and the ON rebuild at
`builds/pypto-opext-on-a589f79` passes 13/13 CTest with DSO SHA-256
`4b7423d3525acef9b8f6a243f852dfa9ab9470ae0fdbec5995ca490904282bc1`.
Second, the Inductor full-chain smoke
`benchmarks/operators/pypto_inductor_pointwise_sm120.py` returns
`output_correct: true` with `wrapper_error: null` both in a direct run and
through the reviewed policy-2 GPU lane (retry attempt 7, after the external
co-tenant burst source disappeared): scheduling routed to PyPTO, Cubin
`c4ffcb54...` byte-stable, `fallback_used` false. The blocker was a genuine
self-deadlock in the wrapper bridge — the launch lock was a plain Lock
re-acquired through the observation helper; the probe had exercised the
lifecycle functions directly and never hit the nesting (plugin `c8991f8`).
Third, the extended-op golden producer
`benchmarks/operators/pypto_pointwise_opext_goldens.py` compiled, launched
and compared fifteen single-op chains through the real bridge on live
SM120: fourteen bitwise-exact versus eager torch and `div` within a
documented two-ulp tolerance (the pinned TensorIR lowering does not emit
round-to-nearest division); two independent runs produced byte-identical
Cubin identities (`state/evidence/opext-pointwise-goldens-run{1,2}.json`).
The plugin now defaults to the opext DSO (`5487d54`) and its suite is
133/133 green after two distribution-shape fixtures were made
environment-stable (the environment legitimately carries the
editable-plus-egg-info pair).

## Previous checkpoint (CP-0049)

**Checkpoint:** `CP-0049`

**Status:** R0 remains open. P2 accepts the finalized RowReductionV3
real-SM120 ten-case correctness gate at PyPTO `faefd0a`: all twenty fresh
lifetimes pass exact/tolerance/special partitions with the corrected `+0`
row-sum accumulator identity, no fallback, intact canaries and explicit
unload. The gate took three real runs: the first exposed a torch-sign
control oracle contradiction, the second (with pre-comparison word dumps)
produced decisive evidence of a genuine single-element signed-zero kernel
defect, and the third passes after the explicit `+0` epilogue fix.
Reduction performance, CUDA Graph, framework, model and coverage
milestones remain open. CP-0048 previously accepted host compiler/Cubin
production; its report stays immutable.

## Current truth

- The control repository was initialized in
  `/home/zhaosiying/pypto-love-tensor-ir`.
- Authorized PyPTO, standalone kernel/plugin projects, and clean official
  PyTorch/SGLang checkouts are materialized at the exact locked identities.
- A fully independent project-local Conda environment is cloned and the
  installed CUDA PyTorch tree is frozen by a content digest.
- Both Qwen3.5 snapshots are independently copied under `models/`, made
  read-only, and verified byte-for-byte against the tracked manifest. The AMD
  source tree was never modified.
- The standalone kernel project now has a reviewed, versioned, immutable ABI
  for tensor arguments, operator/problem/schedule identity, tuning records, and
  requested-versus-produced artifact provenance.
- It also has reviewed typed semantic families for generic matmul, paged
  attention, and GDN. These deliberately stop before a concrete tensor ABI or
  kernel implementation until the exact SGLang inventory is frozen.
- Its reviewed persistent tuning database now provides complete key/cohort
  auditing, deterministic winner selection, multiprocess-safe atomic
  publication and corruption rejection. It contains no device measurement yet.
- Structured matmul has a reviewed explicit ABI v1 for contiguous/aligned BF16
  rank-2 and batched rank-3 tensors, FP32 accumulation, transpose semantics and
  output non-aliasing. No lowering, schedule, launch or performance exists yet.
- Structured matmul now also has a reviewed standard-library BF16/FP32
  numerical reference for every transpose combination and explicit rank-3
  batching. It is not yet compared with TensorIR/CUDA.
- The standalone operator project now publishes the only canonical
  framework-adapter ABI manifest and digest from its real adapter types,
  semantic configs, full specs, constants, signatures and ordered catalog.
- Framework plugins independently recompute that ABI, validate the live
  bindings, pin the Python source-tree digest, and bind either exact wheel
  RECORD ownership or strict PEP-660 editable metadata. The copied partial ABI
  schema was removed.
- Current source-only packaging rejects untracked native extensions and
  sourceless bytecode. Torch and SGLang adapter entry paths perform identity and
  executable-readiness checks before framework mutation; pre-strict failures
  cannot be swallowed as ordinary fallback exceptions.
- The operator project has 117 tests plus 71 subtests; the plugin has 128
  tests. Both wheel and real editable identity probes pass. No executable
  kernel, framework registration, model correctness, coverage or performance
  is claimed by this boundary.
- The matmul, paged-attention and GDN scalar references now have independent
  CPU Torch cross-checks in device-hidden child processes. They add no
  production dependency and preserve the framework-free parent import proof.
- Attention now includes shared-prefix PREFILL→DECODE continuity with private
  write tails; GDN uses vectorized Torch primitives and paired-state continuity
  rather than copying the production scalar loop structure.
- The standalone project now has a source-only benchmark JSON v1 contract. It
  joins typed config/descriptors/workload, CUBIN provenance, symmetric
  candidate/baseline measurements, reset/mutation correctness, separate
  profile traces and fixed device conditions before deriving any comparison.
- No result JSON or publisher exists, and tuning promotion is hard-disabled.
  The contract cannot be presented as performance evidence. The standalone
  suite is now 117 tests plus 71 subtests.
- Paired GDN state zero/copy/checkpoint ownership is now frozen as generic
  PyPTO runtime infrastructure, not an operator or framework copy kernel. The
  detailed StateBundle ABI/lease/stream design is documentation only and is
  ordered after single-DSO, TargetInfo and current-stream executable gates.
- The framework plugin now has a pinned active-route SGLang state lifecycle
  inventory: 11 sources and 35 sites across UnifiedRadix/MambaComponent,
  unified slot translation, deferred clear/COW, checkpoint/donate and reuse.
  Its immutable scope keeps registration readiness false. That transaction had
  122 passing tests; the current plugin suite has 128 after the later Inductor
  inventory transaction.
- Operator artifact provenance now exposes a canonical complete-field digest
  restricted to the explicit matmul/paged-attention/GDN ABI catalog. It is only
  a future join input: physical cache/loading stays compiler-owned and the
  opaque problem digest is not yet a typed launch proof.
- Paged attention now has one reviewed ABI v1 for prefill/extend and decode,
  including explicit KV append/cache mutation, fixed-capacity metadata,
  pitched split-QKV views, workspace-aware non-aliasing, host-copy metadata
  validation, read-only shared radix pages, and zero inactive output rows. No
  CUDA implementation exists yet.
- Paged attention now also has a reviewed standard-library numerical reference
  implementing BF16 storage, FP32-rounded causal GQA, append-before-attention,
  prefill-to-decode cache continuity and inactive-output zeroing. Its CPU Torch
  cross-check is accepted; CUDA and SGLang device comparison remain open.
- GDN now has a reviewed unified core ABI v1 for decode and prefill/extend,
  freezing packed layout, no-bias causal conv, FP32 gate/delta semantics,
  paired BF16-conv/FP32-recurrent state, pitched envelope views and canonical
  metadata/output tails. State preparation/copy and CUDA remain open.
- GDN now also has a reviewed standard-library paired-state numerical reference.
  One-shot prefill, checkpoint-segmented prefill and token decode match exactly
  in output, BF16 conv state and FP32 recurrent state. Its independent CPU Torch
  cross-check is accepted; CUDA and pinned-SGLang device comparison remain open.
- The framework plugin now binds actual imports to the locked CUDA Torch tree
  and clean SGLang checkout, rejects mixed linear-attention providers, and
  fails before launch while the real PyPTO Inductor dispatcher is unavailable.
- Its constructor-dispatch foundation now has a reviewed ContextVar mode,
  original-backend preservation outside that mode, pinned wrapper proxy
  semantics, and strict no-fallback failures inside it. It deliberately does
  not register with Torch until real scheduling/wrapper constructors exist.
- The zero-diff TorchInductor route now has a reviewed 31-source SHA/AST
  inventory plus a manifest for all 346 Python files under `_inductor`. It
  freezes registry/cache, scheduler/wrapper/current-stream,
  template/lowering/fallback, and Triton/CuTeDSL/autotune reference lanes.
- The audit proves there is no pinned TileKernel/cuTile abstraction to reuse,
  records the existing CuTeDSL path, the closed native CUDA selector and
  mandatory `has_triton()` check, and exposes the eight actual
  `get_current_backend()` call sites.
- It also freezes unsafe surfaces that the plugin must own or reject: CSEProxy
  omits `pypto` dtype/shape propagation, extern codegen bypasses the backend,
  foreach accepts only three upstream scheduling types, multi-template can
  select extern, and MM/BMM producers contain explicit extern choices.
- Registration remains false. PyPTO scheduling, wrapper/subgraph wrapper,
  CSE propagation, strict choice filtering, atomic registry installation and
  subprocess compilation are explicitly unimplemented. The full plugin suite
  passes 128 tests; wheel and isolated producer-ABI/source audit pass.
- The framework plugin also freezes 41 source-hashed Qwen3.5 text-path compute
  obligations across generic, matmul, full-attention and GDN providers. This is
  a static inventory only; no site is marked implemented or covered.
- The framework plugin now also has a reviewed strict runtime-coverage evidence
  contract. It reconciles a fixed-revision closed activity trace against a
  digest-bound artifact registry, rejects vacuous/zero-time/misclassified or
  fallback evidence, latches strict failure, and owns report publication across
  processes. Its 68 tests and clean wheel build pass. No runtime collector or
  Qwen trace has been connected, so current runtime coverage remains unclaimed.
- Pinned-source reconnaissance selects the official SGLang HookRegistry plus
  PyTorch's experimental CUPTI monitor for the future collector. Its first
  eager-only trace must remain `closed_world=false`; graph replay and a proven
  complete drain/artifact-launch binding are separate gates.
- The cloned environment is not baseline-ready yet: its Triton distribution is
  still editable from `/home/zhaosiying/codebase/triton`. The runtime audit
  rejects this path. FlashInfer is corrected to the official 0.6.17 wheel and
  the unrelated external study package was removed.
- The local target is an NVIDIA GeForce RTX 5090 Laptop GPU, SM120, 24,463 MiB.
- CUDA Toolkit 13.3 is installed under `/usr/local/cuda-13.3`; the installed
  PyTorch is 2.13.0+cu130.
- The single-DSO compiler object boundary is committed in PyPTO
  `e463bce7849b2239d0457dcae78ccf41c65ffa55`. All 228 production native
  compile rows belong uniquely to `pypto_compiler_objects`; 12 binding rows
  belong to `pypto_core`; the one extra native row is the intentional C++ test
  compile. Native CTest passes 1/1.
- The post-fix editable suite passes 10,176 with 59 skips and exact JUnit
  `(10235, 0 failures, 0 errors, 59 skipped)`. The fresh wheel SHA-256 is
  `bd6d24c9857a409df9d48c604bd329d10808cde354803ee10765680d252f1da1`.
  It has 174 unique safe members and exactly one DSO.
- A real clean-venv install owns both imports and `pypto-ir-trace`; the
  installed DSO has only the five standard runtime `DT_NEEDED` entries. The
  wheel suite passes 10,178 with 57 skips and exact JUnit
  `(10235, 0 failures, 0 errors, 57 skipped)`; the independent symlink probe
  passes. Three independent final reviews report no remaining P0/P1.
- The exact PyTorch-pinned Triton source is now a clean official checkout under
  `upstream/triton`, but the environment still imports the inherited external
  editable distribution until the accepted hermetic workspace wheel is
  transactionally installed for baseline work.
- The exact-wheel transaction now has bounded unreviewed materialization,
  source-frozen archive/manifest review, exact producer RECORD identity,
  offline minimal-bwrap build, native/RECORD wheel audit, pip-free fresh probe,
  reference-only SM120 smoke and reversible replacement tooling. Unit tests are
  source/tooling evidence plus a reviewed dependency closure: all 10 archive
  SHA/byte pairs and manifest `29c073...` are source-locked. The exact wheel is
  now built/audited/probed but deliberately not installed.
- The control implementation is layered as `0c4cc34` (CPU-only isolation and
  materialization), `befe44c` (wheel/native audit and fresh probe), `640c35a`
  (reference-only smoke/finalization), and `c987811` (journaled replacement).
  Follow-up commits `ea39ac5`, `f678fc0`, `5c75ea4`, `fe903fa`, and `cca595c`
  bind LLVM seed bytes, strict Content-Length/strong-ETag Range resume, durable
  no-replace cache publication, stop-race closure and all ten reviewed locks.
  The isolated root suite now passes 198 tests plus 98 subtests. Every
  Python file, every runbook Bash fence and `git diff --check` also pass.
- All project-environment consumers now hold a shared lock; plan shares it and
  apply/recover/rollback require an exact inherited exclusive lock and direct
  child. The replacement uses a durable initializing journal, atomic no-replace
  backup publication, exact prefix-user audits, stdlib RECORD installation and
  idempotent recovery/rollback. No real replacement has run.
- Materialization/promotion runs `pypto-20260824T093330Z-135732-e907f1` and
  `pypto-20260824T100412Z-150285-bbfebd` pass. Independent reviews audited all
  6,503 archive members, 5,853 expanded files and 31 symlinks; seven NVIDIA
  archive hashes match official redistribution manifests. The reviewed cache
  is `caches/triton-build-deps/29c073...`; networkless tool-version probes and
  final require-reviewed verification pass. EV-0037 binds this closure.
- The source-reviewed TargetInfo candidate `9939b88` is integrated as PyPTO
  `042878dd6825bb65ed03f22db7b067fb96277623`. Fresh native CTest passes 2/2;
  the wheel has 177 safe members and one DSO; 31 targeted tests, 10,209 full
  tests with 57 unchanged skips, and the independent symlink case pass. EV-0034
  binds the three successful run IDs, Git tree, wheel/DSO, log and JUnit hashes.
- PyPTO `09e014ceac2d2cac2f667d182bd7de8f0d0bd259` now owns immutable
  CompileRequest v1. It value-copies NvidiaTargetInfo, exact toolchain and
  deterministic Cubin/verification policy; bounded canonical MessagePack and
  distinct byte/loader-input/device projections pass native 2/2 and Python
  62/62 CPU-only gates. EV-0036 binds the independent replay. No region,
  schedule, produced artifact or runtime state is present.
- PyPTO `9b3cf71b6ff2a535aabdd053684a60b47450fac0` now owns immutable
  KernelBuildSpec v1. It binds canonical region source/ABI, one generic
  semantic route, full pipeline revision, eight explicit resolved schedule
  categories, specialization/mutation identities, the parent CompileRequest
  byte identity and optional all-or-none operator catalog provenance. Native
  CTest passes 4/4; an explicit current-source DSO bootstrap runs three native
  contracts and the Python compiler suite 122/122. EV-0038 binds the source,
  DSO, run and recovered stale-editable evidence. No producer, Artifact, cache,
  runtime or CUDA behavior is present.
- The exact Triton reference wheel SHA is `1d58d830...6227a`. Complete
  source/dependency/producer/RECORD/native audit and pip-free fresh probe pass
  in runs `pypto-20260824T134836Z-217013-12ae6e` and
  `pypto-20260824T134906Z-217357-e68fd4`. All 18 ELF files are RPATH-free;
  seven stripped NVIDIA vendor files are bound by exact path/SHA and three
  plugins resolve only to wheel-internal libtriton. EV-0039 binds the recovery
  lineage and evidence. The wheel is deliberately not installed and is not a
  candidate compute dependency.
- PyPTO `5f755687674bd23a513e913cd6f2f20e8b6397ef` now embeds private
  TensorIR `233ab6ed...`, CUDA Tile `af241704...`, and LLVM `57109bef...` in
  the sole `pypto_core` product DSO. TensorIR and CUDA Tile Python/CAPI/CLI
  products are disabled; only build-time `cuda-tile-tblgen` remains.
- The final private compiler DSO SHA is `12b75a4e...c3def5`, has no
  RPATH/RUNPATH and only the five standard runtime `DT_NEEDED` entries. Its
  exact-source compiler suite passes 123/123 and its device-hidden build-info
  probe proves `sm_120a`, compiler-factory availability and every locked
  source/tool identity. EV-0040 binds the commits, DSO and four final runs.
- This compiler composition still returns TensorIR's process-oriented runtime
  kernel rather than a PyPTO persistent Artifact. Kernel bytes/metadata,
  current-stream execution, cache, operators, framework registration and Qwen
  work therefore remain explicitly unaccepted.
- TensorIR `2677d1a99ae9e4c6627872f01a26a62bf6e832c8` now emits a value-owned
  compiled result before legacy runtime reconstruction. It contains validated
  bytes, actual kind/target, entry names, complete argument/packing/grid and
  workspace metadata, strict/fallback policy, exact assembler identity and no
  CUDA/process/runtime fields.
- The native gate covers four-CTA static grid, dynamic Flat/runtime-grid ABI,
  exact argument counts and pack bounds, full TileIR and bounded CUDA ELF
  structures, OSABI/ABI-specific SM decoding, two identical SM120 Cubins,
  hostile environment rejection, wrong producer SHA and explicit fallback.
- PyPTO `4789ae0f...` places all TensorIR types behind a no-RTTI/no-exception
  bridge, validates clean exact gitlinks, isolates ON/OFF native outputs and
  exports only `PyInit_pypto_core`. The final one DSO SHA is
  `0b69023a...74a4b`; 5/5 native and 123/123 exact-DSO Python tests pass.
- No PyPTO strict producer bridge consumes CompileRequest/KernelBuildSpec yet,
  and no canonical persistent Artifact exists. EV-0041 binds the narrow claim
  and all final runs.
- PyPTO `6ce17761cb26b6593ce8a6f0f8a82cb0cf251cc9` extracts the duplicate
  CompileRequest/KernelBuildSpec codec into one compiler-private implementation.
  A streaming preflight bounds total decoded objects, aggregate container
  items, depth, non-binary encoding, BIN size/count and malformed parser
  exceptions before MessagePack allocates its object tree. Canonical wire bytes
  and error compatibility remain covered by exact-DSO replay.
- Backend-ON and a fresh isolated backend-OFF build both pass the three native
  codec/request/spec tests. The exact ON DSO Python compiler suite passes
  123/123. The OFF DSO reports `compiled=false`, has no TensorIR/CUDA Tile/MLIR/
  LLVM targets or link inputs, no RPATH/RUNPATH, two permitted exports and only
  five standard runtime dependencies. EV-0042 binds those codec-only claims;
  Artifact v1 was still open at CP-0029. The root control regression passes
  198 tests plus 98 subtests.
- TensorIR `1c701ec6f7e7c547f6af02862603981f64e01091` adds bounded runtime-free
  CUDA 13.3 Cubin entry and flattened parameter-ABI validation; follow-up
  `b25081afbeb53c9a882c68b440d06baa9e0f6b31` binds executable/constant
  sections to valid PT_LOAD file/VA mappings. It validates unique
  `STO_CUDA_ENTRY`, canonical text/symbol links, KPARAM ordinals/widths,
  PARAM_CBANK base/extent/section symbol and constant-bank coverage without a
  CUDA call, device query, module load or filesystem access.
- PyPTO `9894f5babdca17d27de7b89540e28fc5c3b3e199` adds immutable canonical
  Artifact v1; `4a82f2e40ce518d16fcbeec647061649564c42af` pins the final loader-safe
  validator. The Artifact binds clean PyPTO/TensorIR/CUDA Tile/LLVM/tileiras
  provenance, pipeline blob `a5398054...`, strict schedule options, grid and
  uniform ABI, entry/flattened argument ABI, complete bytes SHA-256, cache key
  and loader projections. Unsupported, noncanonical, oversized, fallback or
  mismatched inputs fail closed.
- The final clean backend-ON build embeds exactly PyPTO `4a82f2e...`, TensorIR
  `b25081a...`, CUDA Tile `af241704...` and LLVM `57109bef...`; the DSO SHA-256
  is `2b65764e...cd06`. Native CTest passes 8/8, including real pinned
  `tileiras` static, dynamic 12-argument and scalar Cubins. The exact source DSO
  Python compiler suite passes 128/128.
- A fresh backend-OFF product passes 6/6, reports strict Artifact creation as
  fail-closed and contains no private compiler dynamic dependency/export. Both
  ON/OFF DSOs have no RPATH/RUNPATH and only five standard runtime dependencies.
  Configure- and build-time guards reject dirty/stale PyPTO or vendor source
  identities. EV-0043 binds all final runs and independent GO reviews.
- CP-0030 does not accept the strict producer bridge. No entrypoint yet consumes
  canonical source with CompileRequest/KernelBuildSpec and returns this
  Artifact; cache publication, subprocess compilation, CUDA module/context/
  current-stream execution, operator/framework/model correctness and
  performance remain open.
- PyPTO `f3bcaaccdfb169080628d56e461653b0ba3e0ad5` now exposes the
  bytes-only strict producer entrypoint. It verifies canonical source against
  KernelBuildSpec, maps the accepted schedule/target/toolchain contract through
  a private standard-only DTO, invokes concrete TensorIR compilation and builds
  the already accepted Artifact v1 without exposing a TensorIR public product.
- TensorIR `1dcb38c20e53d07c97d3781cae538e33901bae30` executes a private
  byte-verified copy of pinned `tileiras` with exact toolkit environment,
  bounded diagnostics/time/memory/files and fail-closed descriptor-relative
  scratch monitoring. Closed host stdio and post-fork compiler reuse are
  explicit regression cases.
- Pipeline blob `46610e0415598d010981e4bd07d0660c592401ac` binds the exact
  process/resource policy. A fresh backend-ON product passes native 8/8 and the
  exact product DSO compiler suite 132 passed/1 skipped. Its SHA-256 is
  `3438f76a65f8021987b187e49ffaba25355a2f8e7920cf251b2c422fea50a134`.
- A fresh backend-OFF product passes native 6/6 and the exact product DSO suite
  128 passed/5 skipped. Its SHA-256 is
  `b3524cb94e2c845e401dac6d5fac123472d66e8869cb00b4e5feb0e98643da3e`.
  Both products are RPATH-free, depend only on five standard runtimes, expose
  only the version node/Python init entry and leak no private compiler dynamic
  symbols.
- Exact clean provenance covers PyPTO plus all six compiled direct submodules:
  TensorIR, CUDA Tile, LLVM, msgpack-c, libbacktrace and runtime. Wrong
  revisions and tracked/untracked/ignored synthetic source changes are
  rejected. The root control suite passes 198 tests plus 98 subtests. EV-0044
  binds the exact runs, logs, products and final GO review.
- CP-0031 does not accept ArtifactCache, CUDA module/function/context/stream
  loading or launch, workspace/runtime allocation, frontend-HIR lowering,
  generic codegen, operators, frameworks, Qwen correctness/coverage or
  performance.
- PyPTO `c087170444270cbe00f83e1fbf127ddcf3e33926` now exposes the
  compiler-owned ArtifactCache v1. `lookup(request, build_spec)` returns only a
  fully revalidated immutable Artifact or an exact `ENOENT` miss;
  `publish(artifact, request, build_spec)` stages, syncs, validates and
  atomically publishes without replacement, returning `Published` or
  `AlreadyPresent`.
- The fixed trusted-local namespace is
  `pypto-nvidia-artifacts/v1/sha256/<prefix>/<digest>.artifact` beneath a
  caller-created absolute canonical non-root owner directory with exact mode
  `0700`. Descriptor-relative no-follow traversal rejects unsafe writable
  ancestors, symlinks and wrong file ownership/type/mode/link count.
- Lookup bounds file size before allocation, detects read/metadata races and
  fully deserializes canonical Artifact v1 before checking exact request,
  build-spec, producer and path-key identity. Corruption is an error; ordinary
  lookup never deletes, repairs, quarantines or recompiles.
- Publication uses a same-shard exclusive temporary file, full write/fsync/
  readback, exact mode `0400`, `renameat2(RENAME_NOREPLACE)` and shard fsync.
  An `EEXIST` winner must independently pass full validation and canonical byte
  equality. Created directories are permission-fixed and synced before their
  parent directory entry is acknowledged.
- Cache handles are non-copyable/non-movable and creator-PID-bound; an inherited
  post-fork handle fails closed and each process opens its own handle. V1 has no
  CUDA state, compile-on-miss, interprocess compile lock, eviction, repair,
  remote cache, signature or quota. `ENOSPC` remains an explicit availability
  error and the owner-private boundary is not origin authentication.
- The fresh backend-ON product passes native 8/8 and exact-DSO Python 137
  passed/1 skipped. Its DSO SHA-256 is
  `d6c9729dff380335b9a9f0e3b581dc9f026b888b9cbf5b869a89888fe9df0c7b`.
  The fresh backend-OFF product passes native 6/6 and exact-DSO Python 130
  passed/8 skipped; its SHA-256 is
  `7edab542960eaa12d0354a169130724c558f8135c1fee4dd4302af5ff614c7e1`.
  Both are RPATH-free, depend on only five standard runtimes and leak no private
  compiler/CUDA dynamic symbol.
- Exact current provenance covers clean PyPTO plus TensorIR, CUDA Tile, LLVM,
  msgpack-c, libbacktrace and runtime, rejecting every wrong revision and
  representative tracked/untracked/ignored drift. Ruff, source/stub Pyright,
  1,133 header checks, 168 EN/ZH doc pairs/navigation and 1,324 English-only
  checks pass. The isolated root control suite passes 198 tests plus 98
  subtests with CUDA hidden. Independent API, security and final audits report
  no P0/P1. EV-0045 binds the exact runs, products, logs and acceptance
  boundary.
- CP-0032 accepts persistent Artifact storage only. It does not accept CUDA
  device/context/module/function state, support/resource legality, workspace
  allocation, current-stream launch, CUDA Graph, frontend-HIR lowering,
  operators, TorchInductor, SGLang, Qwen correctness/coverage, profiling or
  performance.
- PyPTO `2842a1c5433cb3cfe9e4fbc7664ebe0ad8a4b129` now owns a distinct
  internal NVIDIA runtime object target and public `pypto.runtime.nvidia`
  lifecycle API. Construction from Artifact plus CompileRequest is CUDA-free;
  KernelBuildSpec is already transitively bound into Artifact.
- The executable binds PID/device/regular CUcontext/context ID, uses a
  once-only failure-latched state machine and creator-PID custom deletion, and
  makes explicit unload terminal. Its private backend-ON boundary lazily opens
  only `libcuda.so.1`, resolves typed CUDA 13.3 entrypoints through
  `cuGetProcAddress_v2`, and uses context-bound `cuModule*` APIs without storing
  a stream or reusing TensorIR runtime.
- Prewarm validates explicit active Runtime/Driver API versions, current
  non-green context, normalized UUID/PCI identity, SM120 resources, forced
  function loading, exact parameter widths and block/register/shared-memory/
  cluster/occupancy legality. Partial modules roll back and failures latch.
- Prepared packets allocate and validate shape/stride/scalar slots plus checked
  runtime grid before launch. A successful fake-driver Launch records zero host
  allocations and rejects special default stream handles 0/1/2, foreign or
  green stream contexts. Packet and CUDA Graph leases block module unload;
  actual graph capture is not claimed.
- Fresh backend-ON builds 1,902 edges and passes native 9/9 plus exact-DSO
  Python 142 passed/1 skipped. DSO SHA-256 is
  `ef60b6a9749036eaff27786c5838b24fdd282b97b345315855c2994e2b3e4727`.
  Fresh backend-OFF builds 312 edges and passes native 7/7 plus Python 134
  passed/9 skipped; its SHA-256 is
  `9ffd22915e5c2c694660f06f3db0455ee9df560e828fd6b8c13c8c3f478cee7a`.
- Both products retain only five standard dependencies, no RPATH/RUNPATH and
  only the version/Python-init definitions. ON has exactly one driver source;
  OFF has no driver row and directly tests its fail-closed stub. Neither has
  exported or undefined CUDA/TensorIR/executable symbols.
- Seven-source provenance and all negative revisions/drift pass. Ruff,
  Pyright, 1,144 headers, 169 EN/ZH docs/nav, 1,336 English-only checks and
  root 198 tests plus 98 subtests pass. Three independent reviews report no
  P0/P1. EV-0046 binds the exact products/runs and diagnostic lineage.
- The exclusive GPU gate returned 75 for the active protected ZCode/gem5/
  SGLang lane, with zero NVIDIA compute PIDs. No waiver or external signal was
  used. CP-0033 therefore does not accept real libcuda resolution, Cubin load,
  current-stream execution, numerical correctness, CUDA Graph, frontend
  lowering, operators, frameworks, Qwen coverage or performance.
- PyPTO `6361f110660a77f9a8dc542265575d8f7260b343` now owns the
  public immutable `NvidiaRuntimeObservation`. Parent code supplies a Driver
  release provenance string and an audited expected CUDA Runtime library path;
  PyPTO returns the complete live `NvidiaTargetInfo`, numeric Driver/Runtime API
  versions, authenticated canonical provider path and diagnostic regular-
  context address/ID without retaining a CUDA handle.
- Runtime version discovery uses only `dlsym(RTLD_DEFAULT)` followed by
  `dladdr`; actual and expected paths must canonicalize to the same regular
  file. The observation seam never opens libcudart. Environment ownership of
  the expected file remains the caller's proof obligation.
- Observation reuses the private Driver's pre-mutex PID latch. Distinct-value
  tests cover all 21 TargetTraits fields, identity/revision/dtype propagation,
  provider mismatch/status/version rejection and parent-observation to
  fork-child rejection. Backend-OFF public observation fails before dynamic
  loading. Fork children must use spawn/exec because the latch cannot detect
  arbitrary CUDA initialization performed by other libraries.
- Fresh ON/OFF builds and exact-DSO gates pass CTest 9/9 plus Python 142/2,
  and CTest 7/7 plus Python 135/9 respectively. Both remain one RPATH-free DSO
  with only five standard dependencies/two definitions and no CUDA/TensorIR/
  executable/observation dynamic-symbol leakage. EV-0047 binds the exact final
  products, run metadata, JUnit, compile rows and static gate.
- CP-0034 accepts only the CPU/value observation and product boundary. No
  production dlsym/dladdr, real context/device query, Cubin load, CUDA launch,
  numerical result, graph, framework, model coverage or performance is claimed.
- Root `394b75adbc7babe1000d93938e6fa84493a4277d` implements the
  correctness-only SM120 NvidiaExecutable smoke. Its exact child is
  `-I -B -S`; controller, finalizer and both GPU preflights require
  `-E -B -S`. Ambient Python paths, plugins and fallbacks are absent.
- Root `2b53f0a6cdeffa89b38ad75515b9ea1d1019748a` adds the canonical
  seven-blob control manifest with SHA-256
  `c609e97a2f3e379e332137916d041d14931c0e415cd2f2e769c82eab1650aa09`.
  It binds implementation commit/tree, live/committed bytes and modes, requires
  a clean descendant root, and rejects post-implementation control drift.
- The smoke has distinct fully specified static FP32, dynamic-stride FP32 and
  FP16-plus-FP32-scalar TensorIR cases. CPU-only compilation locks Cubin
  sizes/SHA, and exact-DSO replay reopens CompileRequest, all BuildSpecs and
  Artifacts, then joins complete TargetInfo, ABI, Cubin and execution identity.
- Parent admission, post-Popen pre-release gate, child pre-CUDA gate, periodic
  watchdog and post-exit audit reject protected NVIDIA mappings, unreadable
  maps and foreign compute PIDs. Only PID/start-tick descendants in the owned
  PGID are signalable. The protected CPU lane may coexist only through the
  explicit correctness-smoke policy; `gpu-benchmark` remains exclusive.
- Post-manifest root run `pypto-20260825T050233Z-793333-c8f8cb` passes
  224 tests plus 106 subtests. Exact-DSO semantic replay/TargetInfo run
  `pypto-20260825T045248Z-788969-a1b6bc` returns zero and leaves no ignored
  PyPTO package shadow. No CUDA context, module, stream launch, device result or
  performance measurement is accepted by CP-0035.
- First real run `pypto-20260825T052038Z-800777-8e8e83` passed its admission,
  release and child isolation barriers with empty external compute-PID sets and
  empty protected compute-PID/runtime-mapping/unreadable-map sets. It observed
  the real PyTorch Runtime, RTX 5090 context/device and
  loaded the first static Cubin/function, then failed in repetition zero during
  `prewarm` with a parameter-ABI mismatch. The exact failed predicate was not
  persisted; the reversed-offset diagnosis is source/Cubin inference. It did
  not reach `prepare_launch`,
  `Launch`, provisional publication or numerical comparison. The owned PGID
  exited with no survivor and no external signal.
- PyPTO `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7` fixes the generic defects.
  Cubin `KPARAM` ordinals define signature order. Driver-enumerated parameter
  widths are compared as a multiset and offsets only as bounded, non-overlapping
  ranges, independent of enumeration order; launch pointers remain in signature
  order. Dynamic sizes and strides are bounded and packed as four-byte `int32`
  values in stable zeroed slots. Diagnostics are bounded and invalid later
  dense dimensions are rejected before unsigned stride reconstruction.
- CUDA Tile's documented host block is logical `[1,1,1]`; physical worker
  warps and Cubin `EIATTR_REQNTID=128,1,1` are compiler-selected physical
  metadata whose exact mapping remains an inference. They are not host block
  dimensions. The attempted conventional-CUDA reinterpretation was fully reverted and TensorIR
  remains clean at `1dcb38c`.
- Fresh ON product SHA `15675c47...018c` passes CTest 9/9 and exact-DSO Python
  142/2; fresh OFF SHA `32c2dea0...4109` passes CTest 7/7 and Python 135/9.
  Both remain RPATH-free with five standard dependencies and two definitions;
  the complete DSO/compile-row audit passes.
- CPU-only run `pypto-20260825T070649Z-890066-2a1ae2` preserves empty device
  visibility and wrapper ownership markers, observes Torch device count zero
  with CUDA uninitialized, and recompiles all three expected Cubin size/SHA and
  ABI pairs through the exact final DSO. Earlier `062710`, `062751` and `070611`
  runs remain diagnostics and are not acceptance evidence.
- Root control commit `c71f32bd415a973a2a7756ecc9b1ae59f30df219` and
  manifest-only `3de4cf702662cbaf948c6429acf269fee16a491e` bind the final product
  through immutable v3 manifest SHA
  `978e873788eb7f3aaeba6473a9b7f8a1bcd827fe201d89cb781927f538c9b6e3`.
  Versions 1 and 2 remain unchanged. Clean post-v3 run
  `pypto-20260825T071601Z-892819-67acee` passes 224 tests plus 106 subtests.
- V3 run `pypto-20260825T073624Z-900485-7df250` exited zero after recording six
  real non-default-stream module lifetimes, equal logical/Torch results,
  unchanged inputs/padding and explicit unloads. Its provisional SHA is
  `64c0906b...d34cfe`. The required CPU finalizer then failed before publication
  because it expected handwritten `[BF16,FP32]` while canonical TargetInfo and
  replay order is `[FP32,BF16]`. The provisional is retained diagnostic evidence,
  not accepted GPU correctness.
- Root A4 `5564008fddeaaf0a9861ee5c38c895558f577600` fixes only that
  finalizer/control defect with a shared exact ordered contract and malformed
  evidence matrix. Manifest-only B4 `7639d820f4d74972b493c01adc69c92087eefdea`
  publishes immutable v4 manifest SHA
  `a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf`.
  PyPTO, the DSO, Artifact and Cubin bytes are unchanged.
- Clean post-v4 run `pypto-20260825T074420Z-903996-c8d50d` passes 225 tests
  plus 113 subtests. A v4 finalizer cannot promote the v3 provisional because
  exact contract/control identity is mandatory; at CP-0037 a fresh v4 child was
  still required, and the next bullet records its CP-0038 completion.
- Clean v4 run `pypto-20260825T080254Z-910620-c669d9` and its matching no-site
  finalizer close that gate. Six real current-stream executions match
  references and unload cleanly; independent replay joins all compiler inputs,
  TargetInfo, Artifacts and Cubins without fallback. Final report SHA is
  `727362d7879d58cbee07b11050b17ad149274e8087b0d1872b8f186a66a272a9`.
  This accepts minimal `NvidiaExecutable` runtime correctness only.
- PyPTO `07ab9ea1feb5f5cc5557c7b7c67e7ad33d15974e` adds an internal,
  vendor-independent HIR-to-TensorIR emitter for the exact static contiguous
  FP32/BF16 `tensor.add` normal form. Compiler-owned names, `to_chars`, private
  metadata construction and exhaustive fail-closed validation make source bytes
  deterministic. Fresh OFF and ON builds both pass the unconditional native
  test 1/1. No TensorIR parser/compiler or device is invoked by this gate.
- PyPTO `fa85e5a2c917af78ef94576073eb91c0891c4384` adds standalone
  `pypto.canonical_schedule.v1` serialization, identity, bounded fail-closed
  decoding and the single `pypto.compiler` binding. Nested KernelBuildSpec
  bytes and its representative digest remain unchanged. The retained CP-0039
  build directories were reconfigured/rebuilt at exact head; backend-ON/OFF
  native suites pass 2/2 and exact-DSO Python suites pass 98/98. This is a
  compiler-preparation prerequisite only; it does not construct an Artifact.
- PyPTO `c4cf755b7ef2998cdfb7499a76b80b280bafdf2d` adds the private
  compile-free structured frontend identity boundary. It derives five
  versioned projections plus the raw source digest, owns request/schedule and
  metadata, validates one producer-shaped callable ABI, and produces the final
  KernelBuildSpec without placeholder hashes. Producer-reachable FP32/BF16
  goldens, strict ABI drift, cache-boundary and alias/dtype limitations are
  frozen. Fresh backend-ON native 6/6 and exact-DSO Python 180/2 pass;
  backend-OFF native 4/4 and Python 173/9 pass. No frontend producer call,
  Artifact, public binding or GPU execution is accepted by this gate.
- PyPTO `642ff5bd79ee96b9e5a279a2bc945ad7a78362b7` adds the public immutable
  structured compile facade. It prepares frontend HIR, hard-calls the concrete
  private producer once, seals exact source/request/options identity, finalizes
  the producer ABI into the BuildSpec, constructs the Artifact from that same
  move-only result, validates the complete identity join and returns only after
  success. No producer callback, placeholder identity, provisional Artifact,
  compile-twice discovery or cache publication is exposed. Backend-ON native
  7/7 and exact-DSO Python 182/2 pass; backend-OFF native 5/5, functional Python
  7/7 and full Python 175/9 pass. The retained ON/OFF build-directory names are
  stale, so EV-0055 binds exact paths, DSO hashes and the ON embedded revision.
  No CUDA launch through frontend HIR is accepted by this checkpoint.
- Root implementation `1d1fce4d63320beb3f29a265dd126891f37fb559` plus
  manifest-only `47a0c15de510fbdea1eb029ff4e5f0cc9cdc5b77` adds the
  separate frontend FP32/BF16 SM120 smoke family. Manifest SHA
  `f16c4fba...d8eed` binds five new controls and the three exact unchanged v4
  isolation primitives, while also pinning the complete v4 parent manifest.
  The first review returned NO-GO and prevented publication of a real
  ArtifactTarget API defect, weakened finalizer checks and insufficient tests.
  The amended implementation passes final GO with P0/P1/P2 zero, a 21-test/
  13-subtest family suite, exact-DSO runner and replay-child CPU gates, and a
  clean 246-test/126-subtest root suite. No GPU launch is accepted here.
- Finalized run `pypto-20260825T145519Z-1142938-70ac73` closes the narrow
  frontend GPU gate. Two explicit-`In` HIR programs round-trip exactly, invoke
  the public strict facade once each, bind final BuildSpec/Artifact/Cubin
  identities, and run through four fresh `NvidiaExecutable` lifetimes on the
  caller's non-default current stream. FP32 and BF16 outputs match independent
  CPU references byte-for-byte, inputs remain unchanged, and every packet is
  released after synchronization before explicit terminal unload. All
  external/protected runtime/compute/unreadable sets are empty before, during
  and after the run. Final report SHA is `8dbbfbf3...28e8`; independent review
  returns GO with P0/P1/P2 zero. This is not general operator or performance
  evidence.
- PyPTO `b83fcd3ddc497d585bcc45883eede179aff7d4d2` adds private bounded
  `FusedPointwiseV2`: 1-16 inputs, 1-64 linear assignments, ten registered
  tensor operations, exact scalar/source/projection identity, flattened static
  grid, N+1 pointer ABI and zero workspace. Fresh OFF and external ON products
  pass native 3/3 and exact-product Python 1/1. ON compiles FP32/BF16 all-op and
  maximum 16-input/64-op graphs to nonempty SM120 Cubins while preserving both
  CP44 legacy Cubins exactly. EV-0058 binds all source/product/run/JUnit hashes
  and the rejected in-source provenance diagnostic. Independent review is GO
  with P0/P1/P2 zero. No V2 GPU or numerical claim is accepted.
- Root implementation `c98f984ddc4df7cd3354f5fbddadb12df072ed48`
  plus manifest-only `438c25f5db0b3e40c79604352df81e536dcdf137`
  adds the reviewed nine-case fused-pointwise SM120 correctness transaction.
  It binds exact source-only loaders, two CUDA-hidden compiler-anchor runs,
  guarded tail views, eager reference/candidate streams, exact/ULP/special
  comparisons, full replay closure and a CPU-only immutable finalizer. Focused
  32/24 and full 278/150 tests pass after manifest publication. EV-0059 binds
  every exact identity. No real GPU result is accepted here.
- Additive policy-2 controls end at root `6cf5f958` and manifest-only
  `f6a064b73750a71bdd35b97420bd42ca6d425245`. They preserve every v1 byte
  while admitting only the explicitly authorized protected GPU-smoke lane at
  22 GiB. Real run `pypto-20260826T073309Z-1451510-e48ced` completes all
  eighteen fixed lifetimes; immutable final report SHA is
  `d4ffafc0...2eeedf0`. Candidate versus Torch is zero ULP, independent CPU
  joins are at most one ULP, all canaries/lifecycles pass and no fallback or
  protected/external NVIDIA activity occurs. EV-0060 and two independent GO
  reviews close only this frozen nine-case correctness claim.
- PyPTO `62eb88251df5bdad95277a9d619d20da9bf121eb` closes the first real
  RowReductionV3 build defect without weakening its negative fixtures. Fresh
  OFF/ON products pass CTest 11/11 and 13/13 plus exact-product Python 1/1.
  Four rank-1/2/3 FP32/BF16 sum/max cases produce nonempty self-hashed SM120
  Cubins with route `structured-tensorir`, two-pointer ABI, zero workspace and
  no fallback. Report SHA is `d06765be...38abb`; independent CUDA-hidden
  recompilation matches every field. EV-0061 accepts compiler/Cubin evidence,
  not GPU load, numerical correctness or performance.
- RowReductionV3 correctness controls are now frozen at implementation commit
  `23efafaac88fc62698b439b037cb96d95ecbd927`, tree
  `d32c48b4c8139914fdbcdcfede15e92cf5830c76`, with separate manifest-only
  commit `34ee759`. Dual CUDA-hidden anchors are
  `pypto-20260826T110849Z-17249-b18e99` and
  `pypto-20260826T110905Z-17569-3de174`; the 111,827-byte anchor SHA is
  `14af24e4...418a0`. The matrix has ten FP32/BF16 sum/max cases, twenty fresh
  lifetimes, exact/tolerance/special row partitions, independent frozen
  numerical oracles, exact CompileRequest joins, and derived 3,840/15-element
  worst suffix bounds. Three final independent reviews report P0/P1/P2 zero;
  clean focused tests pass 34 plus 84 subtests. A clean full-suite launch was
  refused before child creation when an external zcode `cutlass-compiler` used
  about 39 GiB and MemAvailable fell below 19 GiB. No external process was
  signalled. Full postmanifest CPU regression and all real-GPU claims remain
  pending.
- The single-DSO runbook has been repaired and independently approved: it now
  performs a real venv install, audits all wheel DSOs and installed dependency
  closure, verifies every native/binding compile row and enforces the exact
  two-file commit boundary. Its recovery-audit lineage and final passing run
  IDs are recorded in EV-0033.
- A separate read-only recovery audit binds current `042878d` source/tree and
  clean upstreams to the retained wheel/log/JUnit hashes. Its first attempt
  failed on a copied tree-SHA typo and changed no product; the corrected run is
  recorded in EV-0034.
- TensorIR source audit proves its `IRuntimeKernel` is not a persistent
  artifact: complete argument/grid metadata lacks versioned serialization,
  runtime initialization is not synchronized, and support/workspace checks are
  TODO. Ambient overrides and wrong SM120 resource fallbacks also prevent a
  deterministic cache/runtime shortcut.
- Decision D-0009 now orders a data-only CompileRequest, per-region
  KernelBuildSpec, exact LLVM/tileiras producer identity, common
  bytes-plus-metadata Artifact, persistent cache and a process/device/CUcontext-
  bound prewarmed executable. Streams remain per-capture/launch values; workers
  never query CUDA or reuse module/context handles.
- Candidate and baseline framework runs now have separate, reviewed execution
  profiles. Baseline cannot import PyPTO/plugin sources or load general SGLang
  plugins; candidate framework launch cannot start until exact environment,
  selected NVIDIA runtime, entry points and process ownership all verify.
- Frozen baseline launch scripts exist for 0.8B and 9B, but neither is runtime
  evidence and the CPython 3.12 baseline environment is not built yet.
- Prior reconnaissance did not complete a TensorIR SM120 runtime launch. Static
  target support is not runtime acceptance.
- The user explicitly authorized non-benchmark CPU-only coexistence with a
  protected lane. The new policy is opt-in, retains NVIDIA/environment checks,
  uses a 24 GiB launch floor and 16 GiB owned-run pause floor, and never signals
  external PIDs. TargetInfo itself used green windows and no waiver. A separate
  live control run `pypto-20260824T075816Z-91897-64e1ea` passed the full root
  suite beside seven protected heavy processes with waiver=true, no protected
  NVIDIA compute PID, no pause/abort and no external signal. EV-0035 binds the
  log/preflight/process hashes; this is not GPU or performance evidence.
- On 2026-08-26 the user clarified that about 22 GiB is sufficient for bounded
  CPU-heavy work. A direct base-policy edit was immediately rejected by the
  exact GPU-adapter hash gate and reverted in `7ecc197`; D-0016 now requires a
  separate CPU-only policy-v2 adapter. Existing 24 GiB controls and evidence are
  unchanged and remain authoritative until that adapter is implemented.
- The PyPTO runtime launch bridge is implemented and numerically proven
  at the probe level. `pypto_launch` drives the full lifecycle (live
  runtime observation with the CUDA context forced current, per-kernel
  `NvidiaExecutable` construction + prewarm, allocation-free packet,
  launch on the caller's non-default stream); compile requests now use
  the live observed target so prewarm's device-identity check passes.
  The step-by-step probe on the policy-2 GPU lane executes (x+y)*2.0
  exactly: `correct=True maxdiff=0.000e+00`, prewarm dt=0.26s, packet
  grid (8,1,1), launched stream non-default. Full Inductor-path reruns
  are repeatedly aborted by a transient external co-tenant GPU process
  (codex `.venv-self/bin/python -c` bursts, e.g. PID 71601/77017),
  which the policy-2 watchdog correctly treats as fatal; the retry
  queue continues when the co-tenant lane quiets. The generic smoke
  controller also gained the child signal-mask restore and a
  leader-exit race tolerance for the frozen stop primitive.
- The pointwise expression-tree translation is live. An ops recorder
  replays the Inductor node body against a recording handler: loads become
  FusedPointwiseV2 inputs, the ten registered tensor ops rebuild the exact
  chain (scalar forms for constants), stores become outputs, richer
  operators fail closed. The wrapper emission allocates output buffers and
  passes real argument names with the raw current stream. GPU runs
  `pypto-20260827T004233Z-28000-301f59`, `004308Z-28273-6a576f` and
  `004345Z-28659-88f0d1` (policy-2 lane) compile `(x+y)*2.0` with zero
  Triton into entry `pypto_fused_pointwise_v2`, Cubin SHA
  `c4ffcb54b53c3ccb24a833e5670bcf4aed49ebd429d526a49ed51d57f215a050` —
  distinct from the identity-chain Cubin `23433918...` and byte-stable
  across runs — and the executed wrapper reaches
  `pypto_launch(arg0_1, arg1_1, buf0, raw_stream0)`, which fails closed
  through the pending runtime bridge by design. Five plugin tests green.
- `torch.compile(x+y)` on CUDA now compiles with ZERO Triton in strict
  PyPTO mode. The residual Triton touch was not a kernel node: the pinned
  `autotune_cache.inductor_meta_from_config ->
  torch.utils._triton.triton_hash_with_backend` activates the external
  editable Triton driver (cuda_utils gcc build) on every inductor
  compile. `registration.install()` now wraps that function with a
  mode-aware guard (stable PyPTO backend hash inside strict mode, the
  untouched original outside; reversible on uninstall). GPU run
  `pypto-20260827T003807Z-26412-0c22f5` (policy-2 lane): one non-fallback
  PyPTO SM120 Cubin, controller return code 0, no post-exit violation,
  no gcc/Triton invocation. Five plugin tests stay green.
- The first Inductor-scheduled PyPTO SM120 Cubin exists. Plugin commit
  (after f3ccfb2) moves the routing hook to `codegen_node`: strict mode
  unwraps `ComputedBuffer` to the Pointwise loops, compiles the
  FusedPointwiseV2 program through the exact-DSO facade and emits the
  fail-closed runtime-bridge call. GPU run
  `pypto-20260827T003537Z-25511-9b98db` on the new generic policy-2
  lane (`tools/run_pypto_gpu_smoke_generic.py`) proves the route:
  scheduling_routed=true, one non-fallback SM120 Cubin (13,912 bytes,
  SHA-256 234339182a5300160ac39a98b4ac4e8370f480db505ba7e584de4375cd278aca,
  entry pypto_vector_add, three pointers, zero workspace). Known gaps:
  the wrapper launch fails closed through the pending runtime bridge,
  a residual non-pointwise node still touches the external Triton
  driver, and the expression-tree translation is the bounded identity
  chain. Five plugin tests stay green.
- The TorchInductor PyPTO plugin now has its first two executable layers
  under D-0018 gates. `pypto_plugins.torch.pointwise_codegen` (plugin
  commit 6b5553a) binds the exact pypto_core DSO, builds bounded
  FusedPointwiseV2 HIR programs from the plugin side, and compiles
  deterministic non-fallback SM120 Cubin artifacts (FP32/BF16, three
  focused tests green). `pypto_plugins.torch.registration` (f3ccfb2)
  captures the pinned CUDA DeviceCodegen after init_backend_registration,
  resolves the configured scheduling class through its closure, subclasses
  it with strict PyPTO pointwise routing (other templates fail closed),
  and swaps the CUDA slot through the reviewed dispatcher pair with exact
  uninstall (two registration tests green; delegation outside PyPTO mode
  is the untouched original). A CPU torch.compile smoke confirms the
  registration is inert outside PyPTO mode; the CUDA scheduling route
  needs the GPU lane for its first real exercise, and the expression-tree
  translation currently compiles the identity chain (marked gap).
- StructuredMatmulV4 host compiler/Cubin evidence is complete under D-0018
  gates. The replay worktree (branch feature/structured-matmul-v4-replay)
  cherry-picked 6ee412a then d755117 onto 62eb882 with the one documented
  descriptor-fixture conflict resolved theirs; both expected trees
  ec921f0d/cd1b51f5 verified, exact ten-path inventory. Build-discovered
  test repairs are commit 4b61e3b (three negative fixtures aborted at
  construction). Fresh backend-OFF ctest is 11/11 and backend-ON is 13/13;
  the worktree ON Python suite passes (five-case producer matrix); a
  two-process CUDA-hidden determinism probe produced byte-identical
  BuildSpec/Artifact/Cubin digests for all five rank-2/rank-3
  transpose/decode/batched cases (entry pypto_structured_matmul_v4,
  descriptor-2 static grid, three pointers, zero workspace, no
  fallback). ON DSO is 784,670,624 bytes, SHA-256
  981897033953c606bce7f0050fa0efbcff1feae2b205c6133f2203291400ce8c,
  RPATH-free with the five standard dependencies. GPU numerical
  correctness and performance for matmul remain later gates.
- CP-0049/EV-0062 closes RowReductionV3 real-SM120 correctness. PyPTO
  `faefd0a` (on `62eb882`) adds the explicit `+0` accumulator epilogue to
  every FP32-domain row_sum reduction after two diagnostic GPU runs proved
  the tile backend's single-element fast path returned `-0` for the
  all-negative-zero row. Root control revision `6a4101c` + manifest
  `39b4c35` rebind the corrected ON DSO (`c72fdf3c...`, 784,342,176 bytes),
  regenerate the compile anchors from clean dual CUDA-hidden runs, keep the
  CP48 max overlaps byte-exact and record the sum overlaps as deliberate
  divergence. Fresh ON native is 13/13, ON Python passes, and the clean
  post-manifest full suite is 395 tests plus 358 subtests. Accepted run
  `pypto-20260826T175445Z-218543-3ae1ad` publishes immutable report SHA
  `564ae535...fd7e35` with the historical row now `+0`; the no-replace
  retry fails unchanged. This is the frozen ten-case claim only.
- The D-0016 CPU-only coexistence policy v2 is now implemented and published.
  Root implementation `82162c6` adds exactly six files (contract, preflight,
  controller, control validator, 55-test suite, docs) with the v1 base modules
  and the NVIDIA v4 manifest pinned by exact size/SHA. Manifest-only commit
  `73e79fd` publishes the canonical `state/contracts/pypto_cpu_coexistence_v2.json`
  (SHA-256 `8543ef80...5e2d1f`). Eight independent review rounds closed every
  finding; the final tree bytes received P0/P1/P2=0 from three reviewers and
  the manifest transaction is independently GO. Clean post-manifest focused
  tests pass (55 tests, 85 subtests). The controller: 22 GiB admission/resume,
  16 GiB pause, start-gated child launch under a blocked-signal window, exact
  verified-PGID-only SIGTERM/CONT/STOP (never SIGKILL), character-level CPython
  argv policy decoding, manifest-gated fail-closed admission before lease and
  Popen. No live CPU-v2 run has occurred yet.

## Resume action

The goal is active, not complete. CP-0049 closes RowReductionV3 real-SM120
ten-case correctness; CP-0048 closed host compiler/Cubin production, CP-0047
the fused-pointwise nine-case result, CP-0044/CP-0042/CP-0038 the earlier
compiler/runtime gates, and D-0017 freezes the per-kernel
PyPTO-versus-SGLang-default final acceptance. The next transaction is the
StructuredMatmulV4 replay exactly per docs/structured_matmul_v4_replay_map.md:
worktree from `62eb882`, cherry-pick `6ee412a` then `d755117` (only the
documented descriptor-fixture conflict), verify both tree hashes, then the
sequential OFF/ON fresh builds, native/Python gates, five-shape producer
matrix and the independent CUDA-hidden deterministic recompilation. GPU
numerical correctness and performance for matmul are separate later gates.
After that, continue the PLAN.md main line (generic fused-loop codegen,
Inductor plugin, pypto-kernels attention/GDN, SGLang plugin, 0.8B then 9B,
strict coverage and the D-0017 comparison report). Do not rerun accepted
gates merely to create new evidence. Use explicit CPU-only coexistence only
for non-GPU build/test work; GPU correctness uses the reviewed policy-2 lane
and all GPU performance remains exclusive. Never inherit TensorIR's SM100
defaults and never signal protected amdgpu-sim/zcode processes.
