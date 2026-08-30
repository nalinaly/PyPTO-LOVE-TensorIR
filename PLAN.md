# PLAN

**Plan:** `PYPTO-NVIDIA-QWEN35-V1`, revision `63`

## Current phase: resource-gated Qwen inference qualification and publication closure

## User-brief audit and release-plan closure (2026-08-30, revision 57)

Revision 57 is a direct requirement-to-evidence audit of the original writing
brief and both follow-up requests. It adds explicit gates where the previous
plan was directionally correct but still allowed an ambiguous implementation,
an incomplete article-demo denominator, a placeholder visual, or a document
claim to get ahead of the corresponding experiment. No pending result is
promoted by this revision.

### A. Non-negotiable implementation correction: typed NVIDIA path

The PyPTO-to-TensorIR bridge is first a code task, then a prose task. Before
any final blog/README claim, audit every bridge and backend emission path that
can construct TensorIR, CUDA Tile or MLIR programs. Replace canonical program
construction based on `source +=`, format strings, ad-hoc textual snippets or
stringly-typed operation names with the pinned NVIDIA TensorIR typed
`OpBuilder`/ODS/module APIs. Explicitly construct and verify operand/result
types, ranks, element strides, layouts, tile/iteration spaces, mutation/result
anchors and ABI metadata. A textual printer may remain for diagnostics,
serialization and FileCheck fixtures, but it is not the normative construction
path and must never be described as the backend implementation.

This correction has its own acceptance gate:

1. record the exact source symbols changed in PyPTO and TensorIR, the official
   NVIDIA API symbols they use, and the resulting source/tree/package hashes;
2. add a source-level lint/test that rejects string-built canonical bridge
   programs while allowing explicitly marked debug printers and fixtures;
3. compile positive typed examples and a negative verifier example for invalid
   shape/layout/stride/mutation combinations; and
4. replay the native, package, FileCheck and Qwen-shaped lowering tests after
   the refactor. A green old test with an unreviewed textual emitter is not
   acceptance evidence.

The article must state the boundary precisely: PyPTO owns the DSL/HIR and
specialization; TensorIR is a typed tensor-level, tile-aware compiler layer;
CUDA Tile is a downstream NVIDIA GPU IR/code-generation target. TensorIR is not
CUDA Tile, and the bridge is not justified merely because both systems use
tiles. A PyPTO tile shape lowers only when its static iteration, dtype, layout,
stride and mutation contracts are representable and verifier-valid; otherwise
the route fails closed. The final bridge proof must include one Inductor
SwiGLU trace, one stateful/layout-heavy handwritten trace, and one rejected
geometry, each bound to source/IR/artifact hashes.

### B. Article-demo completeness and educational motivation

The linked WeChat article is an input corpus, not a citation-only example. The
exact URL must appear in the body and source register. Freeze an article-time
snapshot, license/provenance record and byte hash of every copied source file
under `demo/`. Build a manifest from the article's complete code/demo inventory:

1. identify every executable entrypoint and every transitive/tutorial file;
2. copy executable source byte-for-byte, with no line edits, normalization or
   hidden generated replacement;
3. run every executable entrypoint through the NVIDIA compatibility launcher,
   recording command, platform, seed, shape/dtype, compile/runtime stage,
   return code, stdout/stderr hashes, golden implementation, tolerance and raw
   numerical result; and
4. classify prose snippets and genuinely non-entrypoint support files without
   silently removing runnable demos from the denominator. Any runnable demo
   that still fails because of an Ascend-only dependency is a release blocker,
   not a pass labelled “unsupported”.

The compatibility launcher/adapters may live outside `demo/`, but may not
rewrite the copied source. Hash the corpus before and after each matrix run.
The final README and blog each show the same representative unchanged demo,
its exact command, successful numerical/precision result and evidence hash.
The demo chapter must explicitly explain the motivation: PyPTO-LOVE-TensorIR
lets readers without an Ascend device learn the PyPTO DSL programming model on
an NVIDIA platform. This motivation does not relax source fidelity or the
all-entrypoint execution requirement.

### C. Model, kernel and coverage claims

The exact user prompt, pinned non-thinking chat-template input, independent
Transformers oracle, stock SGLang reference, three fresh candidate starts and
ten warm requests per start remain mandatory. The terminal transcript must
show the generated tokens and enough decoded text to prove a real inference
run, plus the PyPTO/Inductor kernel-launch trace where available. It must not
present the model's answer to the subjective prompt as factual endorsement.

The 100% claim is a trace-derived set proof for model-forward compute, not a
handwritten operator list: every observed compute activity maps exactly once to
either a handwritten `pypto-kernels` artifact or an Inductor-generated PyPTO
DSL artifact, with nonzero CUPTI denominators and no unknown/eager/Triton/
FlashInfer/cuBLAS/other fallback compute. Framework staging, sampling,
memcpy/memset and CPU offload are visible in separate scopes. The docs must
say explicitly whether the two sources cover all model operators and show the
union/intersection counts for both 0.8B and 9B.

The operator chapter contains a generated inventory of the independent
handwritten library, identifies the Qwen call sites, and walks through one
complex source-selected kernel. The framework chapter separately lists every
PyPTO-to-TensorIR and Inductor feature/change. The generated-kernel excerpt is
extracted from the exact release artifact and includes FX body, generated
PyPTO DSL and current-stream launch wrapper. It must answer with measured
evidence whether the complete MLP is fused: do not imply that matmul boundaries
disappear if only the SwiGLU pointwise region is fused.

### D. Causal ablation, cold start and attribution matrix

Freeze the following disjoint comparisons so “PyPTO is faster” has a defined
denominator:

1. a causal Qwen3.5-9B chat-template/64-output ablation holding handwritten
   operators, weights, prompt, sampling, SGLang settings and shapes fixed while
   switching target pointwise regions between eager ATen and PyPTO Inductor;
2. an official control of stock PyTorch eager versus official NVIDIA Inductor
   on the same region and full-model workload; and
3. the product comparison of accepted 100% PyPTO against matched and optimized
   stock SGLang, including the requested PyPTO-as-percent-of-default number.

For each lane, retain raw per-start samples and report warm TTFT, TPOT/ITL,
end-to-end latency, output/decode/total-token throughput, memory and launch
counts. Report the within-start median, median across fresh starts, p90/p99
and deterministic start-level 95% bootstrap interval; never pool requests as
independent samples. Count launches separately for the generated subgraph,
`ModelRunner.forward` and whole request, and distinguish event count, unique
artifact count and source-graph count.

Cold measurements use empty Dynamo/Inductor/PyPTO caches and split engine/weight
startup, first compile-trigger request and warm execution. Record PyPTO's
Dynamo capture, Inductor lowering, PyPTO specialization, TensorIR/CUDA Tile
compile, artifact load and first-launch phases where observable. Compare the
same-shape PyPTO graph-compile wall time with official NVIDIA Inductor; phases
that the official backend cannot expose are labelled PyPTO-only diagnostics,
not an apples-to-apples advantage. The opening and conclusion may announce
fusion speedup and launch reduction only from this fixed full-model ablation.
The real fusion boundary, number of fused operators and residual gap must be
shown in both an exact table and source-derived kernel excerpt.

### E. Visual and terminal evidence (five roles, not fabricated)

The original four screenshot requirements and the follow-up representative
demo requirement resolve to five terminal roles: (1) build success, (2)
operator correctness, (3) performance/breakdown, (4) exact-prompt 9B
inference/token stream plus launch log, and (5) one unchanged article demo with
its precision result. When GUI capture is available, each is a genuine Ubuntu
shell hosted by the requested PowerShell/Windows Terminal purple profile. The
capture script must expose the command, run ID and result without secrets or
irrelevant workstation paths. If this environment cannot capture that profile,
keep an explicit user-capture handoff with the exact command and sidecar; never
generate or edit a fake terminal image.

Every ablation and breakdown visual (Inductor causal result, end-to-end
comparison, launch counts, compile phases, semantic operator times and residual
attribution) is generated with GPT-Image-2 from immutable machine-readable JSON.
Before the first real generation, read and follow the `imagegen` skill. Each
figure records model, prompt, source JSON/hash, run IDs, image hash and alt
text; the adjacent table remains the numerical authority. Missing API
authorization leaves a clearly marked gate and blocks final publication rather
than permitting a substitute model or hand-drawn “result”. Visually inspect
labels, units, denominator, ordering and narrow rendering before embedding the
PNG in the local Markdown blog and single-file HTML.

### F. Reproduction, document parity and legal boundary

The Chinese README is the default entry and links to an English README, which
links back. They are semantically equivalent and generated from one immutable
release summary: source layout, prerequisites/hardware/model license,
reconstruction, exact `--jobs 24` build/install, operator correctness,
unchanged article-demo matrix, performance-only scripts, exact-prompt model
run, expected evidence fields, troubleshooting and recoverable cleanup. The
performance script must not accept or import correctness references, golden
outputs, tolerances or token/text comparison logic. Both READMEs introduce the
regression scripts and the same typical demo screenshot/result.

The local blog uses the requested hierarchy (`一、...`, then `1. ...`), remains
accurate and readable, and contains the two-feature opening, glossary/pipeline,
source-linked backgrounds, independent demo chapter, implementation and
operator detail, correctness/coverage, causal ablation, baseline/cold-start
table, breakdown figures, limitations and a conclusion repeating the measured
headline with the same denominator. The HTML is generated from that Markdown
as one offline file with inlined assets and passes desktop/narrow viewport
checks. Blog Markdown/HTML and diagnostic probes stay local; only the audited
README/reproduction assets enter the commit.

The opening NOTE must quote/paraphrase the CANN legal boundary accurately,
identify the project as non-commercial personal research, link the cited
interview and its approximately 2:34:00 timestamp, link the exact WeChat
article, and invite takedown contact. Non-commercial intent or an interview
remark is not asserted as a license exception. Public README push remains
conditional on the written-authorization/legal gate; once cleared, commit and
push the README-only publication change with a final staged-diff audit.

### G. Final requirement audit and execution order

Create a final requirement matrix mapping every sentence of the original brief
and follow-ups to a file/section, command, evidence JSON, screenshot/figure
sidecar or an explicit unresolved blocker. The audit fails on missing links,
stale source IDs/hashes, old 19-token model claims, `xx`/`TBD`/`PENDING_*`,
operator numbers promoted to model headlines, unmatched denominators, missing
image provenance, screenshot/result mismatch, bilingual drift, dirty upstream
trees, or blog/HTML staged for commit. Preserve rejected/failed runs instead of
deleting them, and include an evidence-boundary summary in the final handoff.

The ordered work after this revision is:

1. finish and test the typed NVIDIA OpBuilder/ODS refactor and update source,
   package and identity locks;
2. complete the immutable article-demo corpus/matrix and representative capture;
3. resolve the 0.8B semantic/candidate gate, then the 9B correctness/coverage
   gate, with the independent oracle and exact chat workload;
4. run the causal/official/product performance matrices, cold-start timing,
   CUPTI reconciliation and operator breakdown;
5. read the imagegen skill, generate and audit all GPT-Image-2 figures, capture
   the five terminal roles, and freeze one release evidence bundle;
6. render the local Chinese blog/HTML and parity-checked bilingual README; and
7. run the requirement matrix and source/legal/staging audits, then commit/push
   only the authorized README deliverable.

Current candidate-vs-reference near-tie behavior, foreign GPU occupancy,
article-demo failures and missing GPT-Image-2 authorization remain blockers;
this revision records them rather than hiding them behind historical evidence.

## Critical semantic and evidence amendment (2026-08-30, revision 56)

Revision 56 adds a missing model-input and independent-oracle boundary exposed by
the first runnable 9B reference. It supersedes every earlier sentence that calls
the 19 raw tokenizer IDs a user-facing Qwen3.5 inference workload. It does not
invalidate the operator, typed-lowering or provider-coverage mechanisms, but it
does invalidate their use as final model/blog evidence until the semantic lane
below is rerun.

### A. Separate the diagnostic and user-facing workloads

The exact Chinese prompt remains unchanged. Two workloads now have distinct
names and claims:

1. **Raw-token kernel diagnostic:** the historical 19 IDs produced by
   `tokenizer.encode(prompt, add_special_tokens=False)`. This is allowed only for
   shape-specific compiler/operator debugging and for comparison with historical
   traces. It is not a chat request, cannot supply a screenshot or prose answer,
   and cannot fill any whole-model correctness, performance, launch-reduction or
   100% PyPTO headline slot.
2. **Chat-template release workload:** the pinned model tokenizer applies its
   checked-in chat template to one `user` message with
   `add_generation_prompt=true` and the explicitly recorded thinking-mode
   setting. The release uses `enable_thinking=false` so the fixed 64-token window
   shows the direct answer rather than a partial reasoning trace. At the current
   pinned revisions this is expected to produce 31 IDs,
   but the IDs, rendered template text, tokenizer/config/template SHA-256 and
   token count are generated and verified from both model snapshots rather than
   copied by hand. This exact formatted sequence is the input for the stock
   reference, all three PyPTO starts, ten warm requests per start, matched and
   optimized performance lanes, CUPTI coverage, screenshots and final documents.

Every report names `workload_kind` and includes both the human prompt and exact
model input IDs. The renderer and audit reject cross-scope arithmetic or a raw
diagnostic presented as chat inference. Changing the input length requires
regenerating all dependent operator shapes, Inductor ablations, launch counts,
profiles, figures and document values; the previously accepted 0.8B 19-token
gate remains historical diagnostic evidence and must be rerun for release.

### B. Add an independent semantic oracle

Candidate-versus-stock equality alone is insufficient because both paths can
share a model integration or prompt-formatting defect. Before freezing either
model reference:

1. run the same pinned checkpoint and exact chat-template IDs through an
   independent Hugging Face Transformers Qwen3.5 implementation in a separately
   identified environment;
2. compare first-token/top-k logits and a short greedy prefix against SGLang,
   recording tolerances, versions, weight hashes and state/cache policy;
3. require finite, nondegenerate output and document an explicit semantic smoke
   rule (no replacement-character corruption or pathological repeated suffix in
   the short prefix), without pretending this heuristic measures answer quality;
4. only after the independent oracle is reconciled may stock SGLang become the
   high-throughput 64-step reference for PyPTO.

If Transformers cannot run within the single-GPU memory envelope, use a
source-grounded CPU/offload or layer-by-layer first-token oracle. Do not weaken
the test to tokenization-only success. If the independent implementation and
SGLang disagree, diagnose the first differing layer/state before any 9B
candidate or performance run. If both produce the same surprising output, state
that result accurately and do not market it as a meaningful answer.

### C. Record shared SGLang compatibility fairly

The pinned SGLang 9B path needs two backend-neutral CPU-offload repairs: Gemma
RMSNorm's derived buffer must follow an offloaded parameter, and
`torch.func.functional_call` must receive `tie_weights=false` for Qwen's explicit
parameter aliases. Install the same fail-closed compatibility module in PyPTO,
matched stock and optimized stock lanes; persist its component/disposition
record in every run. It is correctness infrastructure, not a PyPTO optimization,
and contributes zero credited speedup. Vendored SGLang stays zero-diff.

### D. Revised immediate execution order

1. Complete the independent 9B Transformers/SGLang chat-template diagnostic
   when the bounded controller admits host/GPU resources; preserve unrelated
   workloads and never bypass the 12 GiB host or 4 GiB GPU floors.
2. Freeze one generated chat workload manifest shared by 0.8B and 9B, update
   schemas/tests/renderers, and mark all 19-token model results historical.
3. Rerun 0.8B stock reference and three-start/ten-request PyPTO correctness,
   state/cache and closed-world coverage on the chat workload.
4. Run the corresponding 9B reference and three candidate starts only after the
   0.8B release gate passes.
5. Run causal eager/PyPTO-Inductor and eager/NVIDIA-Inductor ablations, then
   matched/optimized full-model performance and profiles with the same chat
   input; regenerate every dependent GPT-Image-2 prompt and screenshot sidecar.
6. Update the blog, HTML and bilingual README from the final immutable evidence,
   then audit against revisions 55 and 56. Blog/HTML remain local; README push
   remains blocked by the legal authorization gate.

### E. Freeze scientific performance reporting

All headline measurements run serially with no foreign NVIDIA compute process.
Record GPU name/UUID, power mode, clocks, temperature, throttle reasons, driver,
CUDA, Torch, SGLang, model and compiler identities, host-memory floor and cache
disposition. Interleave lane order according to the checked-in schedule instead
of running every candidate last. Reject, rather than silently discard, a start
for a predefined reason such as thermal throttling, resource-floor violation,
dropped CUPTI records, source/environment drift or incomplete token count.

Keep request samples grouped by fresh process start. Report the within-start
median, the median across starts, p90/p99 tails and a deterministic 95% bootstrap
interval over start-level estimates; never pool requests to inflate sample size.
Publish raw per-start values and formulas. Model metrics include cold engine
startup, first compile-trigger request, TTFT, TPOT, ITL, end-to-end latency,
output/decode/total token throughput, peak GPU/host memory and launch count. The
blog defines each metric and its direction once, while the README gives the exact
command and JSON field. A performance result with missing resource telemetry,
unmatched workload/configuration or an unreconciled profile residual cannot fill
the opening announcement.

### F. Keep model output claims honest

The requested prompt is intentionally subjective and may mention a work or
premise the model handles inaccurately. A terminal transcript proves only that
the frozen model/runtime generated a deterministic token sequence; it is not an
endorsement or factual validation of the answer. The blog shows enough output to
demonstrate coherent generation, reports exact token/text hashes, and separates
semantic smoke from numerical candidate/reference parity. It does not score
answer quality from one prompt or call fluency a compiler correctness metric.

### G. Add a reader-facing map and source register

Before implementation detail, define `DSL`, `HIR`, `tile`, `TensorIR`, `CUDA
Tile`, `TorchDynamo`, `Inductor`, `kernel launch`, `prefill` and `decode` in a
compact glossary. Include one evidence-backed ownership table and one pipeline
diagram showing:

`Qwen/SGLang -> Dynamo/Inductor or handwritten PyPTO -> PyPTO HIR -> typed
TensorIR -> CUDA Tile IR -> Cubin -> current CUDA stream`.

The diagram distinguishes compiler stages from runtimes and labels the two major
features; it must not imply that TensorIR is CUDA Tile or that every PyPTO tile
shape is automatically legal. Keep a final source register containing official
project URLs, pinned commits, model/demo revisions, the exact WeChat and
Bilibili links, license pages and retrieval dates. Every factual project/history
claim maps to an inline link or source-register entry; source code and live
evidence remain primary for implementation and performance claims.

## Final writing blueprint and newly closed omissions (2026-08-30, revision 55)

Revision 55 re-audits the original writing brief together with both follow-up
requests. It does not promote any pending experiment into a result. It closes
the remaining planning gaps below and supersedes weaker wording in revision 54.

### A. Freeze the narrative before filling measurements

The Chinese blog has one title and NOTE, followed by exactly this first-level
order. The independent demo chapter stays immediately after the two background
introductions, as requested:

1. `一、结论先行`: announce the two engineering features and only current,
   whole-model headline numbers; state scope and denominator beside every
   percentage.
2. `二、PyPTO 与 TensorIR`: introduce both projects, their official URLs and
   pinned revisions, then prove why their boundary is technically valid.
3. `三、文章 Demo 运行与精度核验`: explain the educational motivation, link the
   WeChat article verbatim, execute its unchanged demos, report golden checks,
   and show the representative Ubuntu terminal screenshot.
4. `四、工程结构与复现入口`: give the concise blog build path and point to the
   stricter bilingual README procedure.
5. `五、框架实现`: separately cover PyPTO-to-TensorIR and the TorchInductor
   PyPTO backend, including every release feature and fail-closed boundary.
6. `六、算子实现`: inventory the independent handwritten library, explain one
   representative complex kernel from exact source, and show the real generated
   Inductor kernel and the actual fusion boundary.
7. `七、系统正确性与 100% PyPTO coverage`: separate operator, state/cache,
   model-token/logit, stability and closed-world provider evidence.
8. `八、性能、baseline 与差距归因`: report the causal Inductor ablation,
   matched/optimized SGLang comparison, cold cost, launch count, semantic
   operator breakdown and residual.
9. `九、限制与后续工作`: state unsupported shapes/features and negative
   performance results without marketing language.
10. `十、总结`: repeat the same measured whole-model speedup/ratio and launch
    reduction as the opening, then summarize both major features.

Second-level headings use `1. ...`, `2. ...`. The prose is a concise technical
report: define a term before relying on it, use one concrete trace for each
abstract compiler claim, and keep implementation detail only when it explains a
design decision, correctness boundary or measured result. The final document
must contain no `xx`, `TBD`, stale `PENDING_*`, operator number presented as a
model number, or success claim without a current-release evidence hash. A local
work-in-progress may retain explicit pending markers; it is not a finished
article or release until all required fields are resolved.

### B. Turn the PyPTO-to-TensorIR rationale into a proof

The bridge discussion cannot rely only on architectural prose. Generate a
hash-bound lowering trace for at least one Inductor-generated SwiGLU graph and
one stateful/layout-heavy handwritten graph. Each trace follows one value from:

`PyPTO DSL tile/shape/stride/mutation -> specialized HIR -> TensorIR typed ODS
op and iteration space -> selected tile/layout -> CUDA Tile IR -> Cubin ABI`.

The trace records source symbols, IR hashes, schedule/tile sizes, element
strides, layouts, mutation/result anchoring, compiler revisions and artifact
identity. It also contains a negative geometry that the typed verifier rejects.
This is the evidence for the precise conclusion: TensorIR is a tensor-level,
tile-aware compiler frontend whose NVIDIA lowering target is CUDA Tile IR; it is
not CUDA Tile itself, and it is not accepted merely because both systems use the
word "tile". PyPTO's tile shape lowers only when its static iteration, layout,
stride, dtype and mutation contracts are representable; unsupported geometry
fails closed. The blog cites the exact pinned TensorIR pipeline/source that
establishes this relationship and never describes string concatenation as the
normative backend.

### C. Make the Inductor experiment causal

The earlier three-mode wording is not sufficient to attribute a whole-model
change to the PyPTO Inductor backend. Freeze two related but distinct matrices:

1. **Candidate causal ablation:** hold the handwritten PyPTO operators, model,
   prompt, sampling, SGLang settings and input shapes constant; compare the
   target pointwise regions executed eagerly with the same regions compiled by
   the PyPTO Inductor backend. This lane isolates automatic graph capture and
   fusion. The eager lane may contain ATen compute and is therefore an ablation,
   not a 100% PyPTO product claim.
2. **Official compiler control:** on the same region and full-model workload,
   compare stock PyTorch eager with official NVIDIA Inductor. Do not compare an
   official optimized region with a candidate lane that changed other kernels.
3. **Product comparison:** compare the accepted 100% PyPTO candidate separately
   with matched and optimized stock SGLang. Report `PyPTO / baseline` throughput
   as the requested percent-of-default metric as well as the signed speedup,
   with formulas and higher/lower-is-better direction.

For each matrix, report operator-level and full-model scopes separately. Count
CUDA events inside explicit subgraph, `ModelRunner.forward`, and whole-request
windows; distinguish launches, unique artifacts and source graphs. Cold timing
uses empty Dynamo/Inductor/PyPTO caches and separates engine/weight startup from
the first compile-trigger request. Instrument the comparable graph-compile wall
time and, for PyPTO, break it down into Dynamo capture, Inductor lowering,
PyPTO frontend/specialization, TensorIR/tileiras compilation, artifact load and
first launch. Internal phases that official Inductor cannot expose are shown as
PyPTO-only diagnostics, not a false apples-to-apples comparison.

The opening and conclusion may announce graph-fusion speedup and launch
reduction only from the full fixed Qwen3.5-9B chat-template-input/64-output
ablation. The
existing SwiGLU probe remains useful operator-level evidence but cannot fill
that headline slot.

### D. Prove the model operator closure

"Handwritten plus Inductor-generated kernels cover all Qwen operators" requires
a set proof, not an inventory assembled by hand. Persist a release manifest
derived from the actual 0.8B and 9B traces that maps every model-forward compute
activity and source node to exactly one of:

* a handwritten `pypto-kernels` graph and exact source/artifact identity; or
* an Inductor-generated PyPTO DSL graph and exact FX/source/artifact identity.

The union must equal the observed model-forward compute set. CUPTI call-count
and GPU-time denominators must be nonzero and fully reconciled. Framework input
staging, allocator/memcpy/memset, tokenizer and sampling are separately named,
counted and scoped; a compute kernel cannot disappear from the denominator just
because it was labelled "framework". Exclusions require explicit external
correlation provenance, and unknown, dropped, eager, Triton, FlashInfer,
cuBLAS/cuBLASLt, sgl-kernel or other fallback compute makes the 100% claim fail.

The handwritten inventory and the implementation feature matrix are generated
from current source and trace identities. They distinguish PyPTO changes,
TensorIR changes, package/plugin changes, supported Qwen call sites and
unsupported cases. This matrix is the completeness audit for the implementation
chapters.

### E. Tighten unchanged article-demo acceptance

Every executable demo presented by the linked article must run on the NVIDIA
compatibility path with its copied source byte-for-byte unchanged and pass its
declared golden/precision check. `--help`, import success, source hashing or a
CANN-only error is not execution evidence. A compatibility launcher, backend
adapter or environment setup may live outside `demo/`, but may not rewrite the
copied files. Hash the corpus before and after every matrix run.

Drafts and transitive library modules that are not runnable demos remain in the
provenance inventory and may be marked non-entrypoint. A runnable demo cannot be
excluded merely because it currently depends on CANN: it is a blocker until an
external compatibility path executes it unchanged, or the final article must
state honestly that the user's all-demo requirement is unfinished. The matrix
records every attempted command, compile/runtime stage, return code, stdout and
stderr hash, seed, shape/dtype, golden implementation, tolerance and raw error.
The representative README screenshot shows the command, successful execution
and numerical result for the same manifest entry and evidence hash.

### F. Make README reproduction independently testable

`README.md` is the default Chinese entry and links to `README_EN.md` at the top;
the English file links back. Both must be semantically equivalent, not merely
share a few numbers. Their formal runbook covers prerequisites and supported
hardware, disk/model requirements and model license, source reconstruction,
environment bootstrap, exact `--jobs 24` build, install, operator correctness,
article demos, performance-only benchmarks, exact-prompt 0.8B diagnostic and
9B release inference, evidence interpretation, expected runtime, limitations,
troubleshooting and recoverable cleanup.

A fresh-clone/fresh-prefix transcript executes every public command in order
without workstation-only absolute paths, implicit editable installs, ambient
cache state or undocumented environment variables. It verifies that correctness
and performance entrypoints are separate checked-in regression products. A
performance process must not accept a golden/reference/tolerance argument,
import correctness orchestration, or compare tensors/tokens/text; it only emits
timing, launch, resource and provenance data. Bilingual section IDs, commands,
links, supported/unsupported statements and rendered measurements are checked
for parity automatically.

### G. Keep source excerpts, screenshots and figures evidentiary

Source excerpts for the representative handwritten operator and generated
Inductor kernel are extracted from the exact release source/artifact by a tool
that records path, revision, line range and SHA-256. The prose may shorten them
with explicit ellipses but cannot manually reconstruct or silently normalize
the implementation. The generated kernel view includes the FX body, generated
PyPTO DSL and current-stream launch wrapper, and states truthfully whether a
whole MLP, only SwiGLU pointwise, or another boundary was fused.

The five terminal roles are build, operator correctness, performance/breakdown,
9B exact-prompt inference, and one typical unchanged article demo. A genuine
Ubuntu shell hosted by the requested PowerShell/Windows Terminal purple profile
is required when GUI capture is available; otherwise the exact user-capture
command and sidecar remain an explicit handoff, not a mock success. Screenshots
must expose the command, result and run ID without secrets or irrelevant host
paths, and remain secondary to hashed JSON/log evidence.

Every ablation and breakdown visual, including Inductor, end-to-end, semantic
kernel-time, launch-count and compile-time breakdowns, is generated with
GPT-Image-2 from immutable JSON. Maintain a figure inventory so none are
silently omitted. Record model, prompt, source JSON/hash, run IDs and image hash;
visually audit labels, units, ordering, sign and denominator against the adjacent
exact Markdown table. Add useful Chinese/English alt text. A generated image
never supplies or changes a number, and a terminal screenshot is never generated
by an image model.

The single-file HTML inlines styles and all image bytes, preserves Chinese
heading hierarchy/code/tables, contains no external runtime assets, and is
render-smoked at desktop and narrow viewports. Broken links, clipped tables,
missing images, stale evidence references or unreadable generated labels fail
the document gate.

### H. Verify attribution and final publication state

Before freezing the NOTE, verify the exact CANN license wording, the public
project URLs, the interview speaker/title and the approximately 2:34:00 passage.
Mark the interview statement as a sourced paraphrase unless a checked transcript
supports quotation; neither non-commercial intent nor the interview is described
as a legal exception. Preserve the exact Bilibili and WeChat links, third-party
demo commit/hashes/licenses and takedown contact. Written authorization remains
required before public push.

The final audit has one row for every original and follow-up request, with
document section, source/evidence, status and allowed wording. It additionally
checks factual citations, command replay, language consistency, claim scope,
percentage formulas, source and package locks, provider closure, screenshot and
GPT-Image-2 manifests, offline HTML rendering and Git boundaries. The blog
Markdown and HTML remain local and absent from commits; README and necessary
reproduction/assets may be committed only after the complete release audit, and
push remains subject to the legal authorization gate.

### I. Ordered execution from the current checkout

1. Repair the direct typed paged-attention adapter against the accepted legacy
   PyPTO semantic reference; then formalize and re-lock the framework package.
2. Rebuild/install from the atomic source lock and rerun native/CTest/operator,
   CUDA Graph and CUPTI probe regressions.
3. Complete Qwen3.5-0.8B correctness, three-start stability and strict coverage;
   only then repeat the same gates for 9B.
4. Run the causal Inductor matrices, matched/optimized end-to-end performance,
   cold-compile instrumentation and independent CUPTI/NVTX breakdowns serially
   on an idle GPU.
5. Execute every article demo unchanged with golden checks and complete the five
   terminal screenshot sidecars.
6. Generate all GPT-Image-2 figures from the frozen release summary, render the
   blog/HTML and bilingual README, and run the requirement/consistency/fresh-
   reproduction audits.
7. Keep blog/HTML local; create small auditable README/reproduction commits and
   push only after the legal gate is explicitly cleared.

## Omission audit and execution contract (2026-08-29, revision 54)

The second user follow-up does not replace the original brief; it adds release
gates.  The following checks are now explicit because they were easy to imply
in prose while still being absent from a reproducible release:

1. **Article-demo execution is a separate gate.**  A byte-for-byte copy,
   source-hash audit, or `--mode help` success is not an execution or numerical
   correctness result.  The matrix must attempt every non-draft entrypoint from
   a clean Ubuntu environment, record the exact command, return code, stdout and
   stderr hashes, compile/golden/runtime stages, and max-error comparison when a
   runtime exists.  Draft and Ascend-CANN-only files are not silently counted as
   passes: they remain hash-audited with an explicit exclusion reason.  The
   present `57/57` help audit therefore remains only an inventory result; the
   article demo execution gate is still pending on the missing Ascend runtime and
   the Qwen3-14B API mismatch.
2. **A typical-demo screenshot is required in both READMEs.**  In addition to
   the four release roles, add a manifest role for a representative unchanged
   article example (the planned example is `examples/beginner/hello_world.py`).
   The role must show the command and the visible compile/golden/runtime outcome
   in the Ubuntu shell hosted by the requested PowerShell-purple terminal.  Until
   a real pass exists, use a clearly labelled `PENDING_SCREENSHOT` slot with the
   exact command; never manufacture a success terminal.  Link the same image and
   sidecar from `README.md` and `README_EN.md`, and describe the blocker beside it
   when the result is pending.
3. **Screenshot provenance is machine-checkable.**  Every screenshot sidecar
   must contain role, repository-relative PNG path, PNG SHA-256, command, run ID,
   evidence path/SHA-256, UTC capture start/end, viewport/window dimensions,
   Ubuntu/Windows-Terminal/PowerShell-purple metadata, and a pass/pending status.
   The audit must reject a stale image, a missing viewport, an evidence hash
   mismatch, or a pending image captioned as success.  A screenshot is never a
   substitute for its JSON/log evidence.
4. **One evidence bundle owns all document numbers.**  The opening announcement,
   conclusion, Chinese blog, Chinese README, and English README must be rendered
   from the same release-summary inputs.  The audit must compare every displayed
   acceleration percentage, launch denominator/reduction, cold-compile time,
   throughput/latency value, model token result, and coverage value by scope.  A
   provisional operator result can appear only with an `operator-level` label;
   it must never be copied into a whole-model claim.
5. **The model claim remains ordered and fail-closed.**  Run Qwen3.5-0.8B first
   and Qwen3.5-9B only after the smaller model passes.  Each model requires the
   frozen generated non-thinking chat-template input (currently expected to be
   31 IDs) plus
   64 generated IDs, three fresh starts, ten warm requests per
   start, exact greedy token IDs/text, prefill/decode logit and state/cache
   tolerances frozen before measurement, 100% non-vacuous PyPTO compute
   coverage, zero fallback/unknown compute, and separate correctness, stability,
   coverage, and performance reports.  A readable answer, one next-token run,
   Cubin, unit test, or compile result cannot advance this gate.
6. **The performance experiment has two explicitly different scopes.**  The
   checked-in performance-only scripts must report eager versus official NVIDIA
   Inductor versus PyPTO Inductor on the real 9B SwiGLU geometry (warm latency,
   throughput, first-call graph-compile wall time, launch count and fusion
   reduction), and separately report the full chat-template-input+64 end-to-end
   model A/B against
   matched and optimized SGLang.  CUPTI/NVTX must align logical operator classes
   and reconcile their deltas to the end-to-end GPU-time gap; no stage percentage
   is filled from a launch count alone.
7. **GPT-Image-2 is an evidence consumer, not an evidence source.**  Generate
   every ablation/breakdown visual only from the immutable JSON produced by the
   corresponding run, record model/prompt/evidence hashes and scope, embed the
   PNG in the blog, both READMEs and HTML, and keep exact numbers in adjacent
   tables.  If the authorized GPT-Image-2 API key is unavailable, retain
   `PENDING_GPT_IMAGE2` and do not substitute another model or hand-draw a result.
8. **Release identity must be atomic.**  The PyPTO nested checkout, bundle,
   patch series, source lock, installed wheel build-info, tests, screenshots and
   reports must all name the same commit/tree.  Any experimental nested commit
   that has not passed native build, CTest, numerical gates and source-release
   replay is explicitly non-release and must not be cited by the blog.  This
   check also covers package subtree revisions and the pinned TensorIR/CUDA
   Tile/LLVM/SGLang/model identities.
9. **Commit/push and local artifacts are separated.**  The blog Markdown and
   single-file HTML stay local and ignored.  README changes may be committed
   after the final audit; public push remains blocked until the written
   license/rights authorization described in `LEGAL_NOTICE.md` is verified.  A
   local commit is not represented as a public release, and a requested push is
   never simulated by a log line.
10. **Implementation claims need source traceability.**  Before prose is frozen,
    build a feature matrix whose rows are every bridge, Inductor, handwritten
    operator, state/lifecycle and profiling feature; each row names the exact
    source symbol/path, PyPTO/TensorIR/SGLang revision, test or golden report,
    and the corresponding blog/README paragraph.  The matrix must distinguish
    PyPTO changes from TensorIR changes and list unsupported/fail-closed cases,
    so “all features” is auditable rather than an unbounded assertion.
11. **GPU occupancy creates a queue, not a weaker standard.**  While another
    workload owns the RTX 5090, only CPU/static checks, source audits and
    performance-script dry validation may run.  The queued formal commands keep
    frozen model/tokenizer revisions, exact prompt, GPU identity, environment
    lock, fresh-start counts and output paths; when the protected-process gate
    is clear (or the user explicitly authorizes coexistence), execute the queue
    in order and regenerate every dependent number, screenshot, GPT-Image-2
    prompt and HTML/README marker.  No historical or operator-only number may
    fill a whole-model slot merely because the GPU is busy.

The required final audit therefore has two outputs: a pass/fail checklist for
the original and follow-up requirements, and a table of unresolved gates with
the exact command, evidence path, owner/action and the wording allowed in the
documents.  The final documents must be regenerated after any new model,
profile, screenshot, image, or source-lock result; no historical screenshot or
metric may be carried across a revision without a hash-bound re-audit.

## Requirements audit and newly added demo/image work (2026-08-29)

The original writing brief and the follow-up requirements are one acceptance
contract. The following items are explicit deliverables and must not be
silently replaced by prose placeholders:

### A. Legal, attribution, and scope note

The blog opens with a clearly marked NOTE: this research checkout adapts the
CANN/PyPTO software to an NVIDIA device despite the CANN open-source license's
platform restriction; it is a non-commercial personal research project. The
note attributes the public goal of a common PyPTO frontend to Liao Heng's
interview with Xiaojun at **2:34:00**, links the exact Bilibili video
`https://www.bilibili.com/video/BV1nB3u6tERu/?vd_source=f2f41aa7b5e3cc8e0a23942779ccea11`,
and states that a rights holder may contact the author for removal. This is a
description and takedown contact, not a claim of legal permission; the README
must carry the same boundary.

### B. Source article and demo provenance

The source article link must appear verbatim in the Chinese blog and README:
`https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ`. The article is locked to
the fetched title and article-time source repository commit
`hw-native-sys/pypto-lib@6c292d30ccc787ee4e1fe61541fd3faec0dafa65`
(the last `main` commit before the article's `2026-08-28 17:30 +08:00`
publication timestamp).
Article-facing demo files are copied byte-for-byte under `demo/` (with a
manifest containing path, SHA-256, upstream URL, commit, and license notice),
not edited to make the compatibility run look native. The complete article-time
import now contains all 11 teaching examples, Golden/contract modules, the
complete Qwen3-14B tree, and the complete DeepSeek V4-Flash MTP tree (151 source
files total). The manifest also records 66 CLI entry points: 57 runnable and 9
explicitly excluded draft or Ascend-CANN-only entries. Representative entries
are:

* `examples/beginner/hello_world.py`
* `examples/intermediate/rms_norm.py`
* `examples/intermediate/softmax.py`
* `examples/intermediate/gemm.py`
* `golden/__init__.py`, `golden/runner.py`, `golden/spec.py`, and
  `golden/validation.py`
* `models/qwen3_14b/decode_fwd.py`, `models/qwen3_14b/contract.py`,
  `models/qwen3_14b/greedy_sample.py`, `models/qwen3_14b/paged_attention_pypto.py`,
  `models/qwen3_14b/paged_attention_cce.py`,
  `models/qwen3_14b/rope_qkv_regen.py`, `models/qwen3_14b/rms_lm_head.py`,
  `models/qwen3_14b/topk_select.py`, `models/qwen3_14b/turboquant_kv.py`, and
  `models/deepseek_v4_flash_mtp/{decode_layer,decode_fwd,prefill_fwd}.py`

The manifest also records transitive Python/C++ modules imported by those files.
`tools/run_article_demo_matrix.py --mode help` verifies all 57 runnable CLI
entrypoints; draft and Ascend-CANN-only files remain hash-audited and explicitly
skipped. The compatibility launcher may set platform/device variables, but the
copied demo source itself must remain unchanged. If an upstream demo cannot
execute on this NVIDIA checkout, the report must show the exact blocker and mark
it pending rather than claiming success.

### C. Demo execution and screenshot evidence

After the PyPTO/TensorIR background chapter, the blog gets an independent
chapter **“Demo 运行与精度核验”**. Each copied demo is run from a clean Ubuntu
environment using a recorded command, exit code, source hash, and numerical
comparison. The README gives the same commands and expected evidence paths.
Four screenshot gates are required and are separate from text logs:

1. native build/configuration completes;
2. operator correctness regression passes (golden/max-error summary);
3. performance/ablation and kernel-launch breakdown completes; and
4. the fixed Qwen3.5-9B end-to-end prompt generates tokens with strict
   correctness and stability checks.

Screenshots must be captured from the Ubuntu shell hosted in the requested
PowerShell purple terminal when that GUI is available. A screenshot sidecar
records viewport, timestamp, command, run ID, and SHA-256 of the underlying
log/report. If WSL has no capturable GUI, keep an explicitly labelled
`PENDING_SCREENSHOT` slot with the exact command and do not present a rendered
terminal mockup as execution evidence.

Current screenshot evidence is provisional: the build, structural operator,
and SwiGLU performance captures are real Ubuntu/PowerShell-purple-terminal
images; the model image is intentionally a blocker/pending capture. The formal
typed paged-attention numerical report is
`state/evidence/paged-attention-typed-tensorir-20260829.json`.

### D. Two core features and architecture narrative

The opening announcement names exactly two major features: (1) PyPTO DSL/HIR
bridged through typed NVIDIA ODS/`OpBuilder` TensorIR and then CUDA Tile, and
(2) the TorchInductor PyPTO backend. After separate PyPTO and TensorIR
introductions, an independent argument explains the boundary: TensorIR is a
typed tensor-level IR with tile-aware layout/iteration propagation, and CUDA
Tile is its GPU code-generation target; it is not itself the CUDA Tile runtime
and PyPTO is not described as emitting canonical source strings. The text maps
static tile shape, dtype, element strides, layout, mutation, and failure-closed
contracts across the bridge and cites the pinned NVIDIA source.

### E. Correctness, performance, and ablation evidence

Correctness scripts and performance-only scripts are separate checked-in
regression artifacts. Performance scripts never substitute for a golden check;
correctness scripts publish max-absolute/max-relative error, token IDs/text,
state/cache drift, and provider coverage. The human prompt is exactly:

`为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？`

The model claim requires Qwen3.5-0.8B then 9B, generated IDs from the pinned
chat template (the historical 19 raw IDs are diagnostic only), three fresh
starts, ten warm requests per start, deterministic greedy IDs/text, 100% PyPTO compute coverage,
and zero fallback compute kernels. The 9B comparison uses the same weights,
GPU, batching/workload, and methodology for eager, official NVIDIA/Inductor,
and PyPTO-Inductor lanes. It reports warm latency/throughput, first-call cold
compile time, launch counts, fusion launch reduction, and per-class
kernel/operator GPU-time breakdown. The opening and conclusion repeat the same
scope and denominator. The existing 9B SwiGLU numbers remain explicitly
**operator-level provisional evidence**; they cannot be promoted to whole-model
speedup until the chat-template-input/64-output end-to-end gate passes.

### F. GPT-Image-2 visual evidence

Every ablation and breakdown figure is generated from the immutable evidence
JSON using the `imagegen` workflow and labelled with run ID, scope, denominator,
and “operator-level”/“whole-model” status. Exact values remain in adjacent
machine-readable tables; generated images are visual summaries, never the sole
source of a number. The final PNGs live under the report asset directory and
are embedded in both Markdown and rendered HTML. The built-in image generator
is preferred; if unavailable, the GPT-Image-2 CLI fallback requires an explicit
API-key-authorized run and its provenance must be recorded. No fabricated
terminal screenshot or unreadable generated text is acceptable.

The current imagegen skill dry-run confirms the `gpt-image-2` CLI contract, but
this environment has no authorized `OPENAI_API_KEY`. The three requested
ablation/breakdown PNGs therefore remain `PENDING_GPT_IMAGE2`; no substitute
model or hand-drawn result may be used.

### G. README/blog and final audit

`README.md` (Chinese default) and `README_EN.md` (English) link to each other,
describe source layout, pinned build prerequisites, exact compile/correctness/
performance/demo/model commands, regression scripts, evidence paths, and the
same measured numbers and legal boundary as the blog. The blog uses the
requested Chinese hierarchy (`一、...`, `1. ...`) and includes source excerpts,
the handwritten operator inventory, the Inductor fusion boundary, demo
screenshots, model output, performance table, breakdown figures, and a closing
summary. Before commit/push, run a requirement-by-requirement audit that checks
links, hashes, commands, screenshot sidecars, image embeds, numerical
denominators, source-lock cleanliness, and that no pending gate is worded as a
completed result.

## Latest execution audit (2026-08-29)

The current accepted compiler identity is PyPTO
`f0ab91b38ae237c82eaad57be454f783bb5fccee` and TensorIR
`3623b37eddbec0f74f014867fca3dd296ec91868`. The TensorIR scatter-result
relayout fixes the recurrent GDN state-output orientation through typed
gather/scatter/layout operations rather than string emission. Native build
`288/288`, CTest `13/13`, source replay (251 PyPTO patches and 85 TensorIR
patches), bundle materialization and installed build-info all passed for this
identity.

The exact locked-wheel operator regression is GPU run
`pypto-gpu-bounded-20260829T151852Z-2013486-89bb59`. All eight suites passed:
25 compile-classification cases, 32 handwritten numerical cases, 14 stateful
real-model-shape cases, 14 paged-attention cases, four QK cases, eight linear/
LM-head cases, the stateful CUDA-Graph replay, and four Inductor SwiGLU cases.
The durable summary is
`state/evidence/operator-regression-locked-20260829.json`. This accepts the
operator layer only; it is not an end-to-end model or performance result.

The new GDN result repairs the earlier model mismatch. Stock reference run
`pypto-gpu-bounded-20260829T153419Z-2022351-e2bed3` produced the frozen 64-token
sequence. With the accepted legacy attention adapter, candidate run
`pypto-gpu-bounded-20260829T153523Z-2023101-4f78de` produced that exact sequence
for all ten Engine requests. The overall report still failed because CUPTI was
started after the Engine initialized CUDA, so this is exact Engine-output
evidence but not the final correctness-plus-coverage gate.

The coverage harness now starts Engine work in an isolated spawned child and
uses a hash-locked `cupti-python` overlay without contaminating the release
Python prefix. A formal no-wrapper probe observed nonzero activities with zero
dropped records. Explicit SGLang input-buffer staging provenance accounts for
framework bookkeeping without exempting unannotated compute. These harness and
plugin changes still require final package identity, clean install and a full
model rerun.

Replacing the legacy attention adapter with the direct typed paged-prefill and
decode entrypoints removed the external mask-fill activity but is not accepted:
run `pypto-gpu-bounded-20260829T165621Z-2052859-588558` diverges at generated
token step 2 (`expected=271`, `observed=198`). The paired diagnostic records
large differences in both prefill and decode (the first prefill comparison has
`max_abs=3.06298828125`). The immediate blocker is therefore semantic parity of
the direct typed paged-attention adapter, not GDN and not CUPTI. Do not cite this
adapter as release-ready, and do not start the 9B gate until it is repaired and
the complete 0.8B gate passes.

`state/evidence/qwen35-model-gate-status-20260829.json`, the blog/README source
identities and the previous documentation audit still describe the older
`5e39819` state. They are deliberately stale inputs now, not current evidence,
and must be regenerated after framework-package lock/install and the next formal
model run. The article-demo full execution, typical real screenshot, four other
final screenshots, GPT-Image-2 figures, full-model Inductor ablation and all 9B
claims remain pending release gates.

## Blog and Inductor requirements (2026-08-29)

The blog is a deliverable with two headline engineering features, not a
post-hoc progress note:

1. After introducing PyPTO and TensorIR separately, explain the backend
   boundary precisely. PyPTO owns the user DSL/HIR, JIT specialization,
   framework/runtime and artifact identity. TensorIR owns a typed tensor graph,
   layout propagation, iteration-space/tile selection and lowering to CUDA Tile
   IR. The text must answer explicitly that TensorIR is tensor-level but
   tile-aware, and that CUDA Tile is its GPU code-generation target. It must
   show why PyPTO's static tile shape, dtype, element stride, layout and
   mutation contract are sufficient inputs for TensorIR to continue lowering,
   and where unsupported geometry must fail closed. Use the pinned NVIDIA ODS
   and `OpBuilder` path as the normative implementation; do not describe
   hand-built canonical source strings as the backend.
2. Quantify the TorchInductor PyPTO backend with an ablation. The experiment
   must include the real Qwen3.5-9B SwiGLU geometry and, when claiming a model
   result, the full fixed chat-template-input/64-output model workload. Compare eager,
   official NVIDIA Inductor CUDA, and PyPTO; report warm latency/throughput,
   first-call/cold compile cost, CUDA kernel launch count, and the reduction
   caused by fusion. Report acceleration as a percentage with its denominator
   stated. Compare PyPTO graph-compile cost against the official Inductor
   graph-compile cost on the same shape and environment. Operator-level data
   must be labeled as operator-level and cannot be promoted to whole-model
   speedup; missing end-to-end data remains an explicit pending gate.

The durable execution order is: (1) keep the typed TensorIR/GDN transaction
green, (2) freeze and diagnose the chat-template workload, (3) rerun the 0.8B
gate, (4) only after it passes run the 9B gate, (5) run the separate Inductor
ablation and CUPTI breakdown, and (6)
render the Chinese blog/HTML and bilingual README from one evidence bundle.
The article-demo matrix/hash audit is independent and already complete. The
opening and conclusion of the blog must repeat measured launch reduction and
speedup with the same scope and denominator. A negative or slower PyPTO
microbenchmark, or a failed model gate, must be reported honestly rather than
hidden behind a launch-count claim.

## Authoritative final model gate (2026-08-28)

The final objective is stable inference for **both Qwen3.5-0.8B and
Qwen3.5-9B**, in that order, with model-forward compute coverage exactly 100%
PyPTO. The accepted provider set consists only of handwritten
`pypto-kernels` operators and TorchInductor-fused regions lowered through the
PyPTO CUDA backend. Final strict runs permit no Triton, FlashInfer, CuTeDSL,
sgl-kernel, eager ATen CUDA, cuBLAS/cuBLASLt or unknown compute fallback;
machine-readable coverage must report a non-vacuous denominator,
`coverage = 100%` and `fallback_compute_kernels = 0`.

Both model gates use the exact prompt below; changing, shortening or replacing
it is not acceptance evidence:

```text
为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？
```

For each model, acceptance requires all of the following on the same pinned
weights and tokenizer:

1. compare candidate and unmodified reference-path prefill/decode logits with
   thresholds frozen before the candidate measurement, and publish the full
   max-absolute/max-relative and token-margin evidence rather than only a
   readable answer;
2. use deterministic greedy decoding and require exact generated token IDs and
   decoded text against the reference path;
3. pass at least three fresh model/server starts, each followed by at least ten
   consecutive warm requests using the exact prompt, without crash, hang,
   fallback, coverage drift, state/cache leakage or output drift; and
4. preserve separate machine-readable correctness, stability, coverage and
   performance reports. A compile, Cubin, operator smoke, layer test or one
   successful generation cannot substitute for this model gate.

The accepted low-level path now includes fused QK RMSNorm plus partial RoPE,
causal paged attention, packed GDN projection, stateful causal convolution and
recurrent GDN. The immediate work is the zero-diff SGLang route: connect the
accepted StateBundle lifecycle to CUDA Graph capture/replay and radix-cache
copy/clear, close the remaining Dynamo/Inductor and handwritten-operator
inventory with strict provider tracing, then bring up and stabilize 0.8B.
Only after the complete 0.8B model gate passes does the same gate repeat for
9B. No model-name, hidden-size or fixed benchmark-shape special case may be
used to satisfy either model.

The historical 22 GiB CPU-v2 admission value is not a compiler memory
requirement and is no longer a prerequisite for this execution path. It was a
conservative host-coexistence reserve derived from an approximate user
authorization, without owned-build peak-memory evidence. Per the user's later
override, bounded CPU builds use `--parallel 24`, must record owned-process RSS
and host `MemAvailable`, retain the 16 GiB running-child safety boundary, and
must never signal a protected external process. The exact-hashed v2 controls
remain unchanged solely so historical evidence stays reproducible.

CP-0086 accepts the stateful operator layer at PyPTO `15fd226`, TensorIR
`b478e09`, clean `pypto-kernels@9caff6f` and framework plugin `c606574`.
Projection is exact; causal-convolution decode/prefill state is exact with
worst output error `8.84444e-4`; recurrent GDN state error is at most
`2.98024e-8` and worst output error is `0.0121547`. Three fresh controlled
processes produce byte-identical numerical content after removing only the
run ID. Each Conv T5 process also executes ten repetitions with zero output or
state drift. Long Conv/GDN prefills deliberately use ordered one-token PyPTO
primitives (`T` launches), not the rejected nondeterministic multi-token
mutation graph. This is correctness/stability evidence, not performance.

The active bounded controllers contain no 22 GiB gate and reject any build or
CTest command whose parallelism is not exactly 24. The final product-producing
build started with `22553348 KiB` available (below 22 GiB), completed with
`--parallel 24`, and measured `387080 KiB` peak summed owned-PGID RSS. Final
CTest is 13/13 with `-j24`. The 16 GiB initial/running safety line is retained;
it is not a compiler-footprint claim. Evidence is
`state/evidence/pypto_stateful_sm120_cp0086.json`.

CP-0039 accepts compile-free HIR-to-TensorIR emission, CP-0040 accepts the
standalone canonical schedule, and CP-0041 accepts compiler-owned frontend
specialization/ABI identity. CP-0042 advances PyPTO to `642ff5b`: the public
`compile_structured_strict` facade prepares HIR, hard-calls the concrete private
producer once, seals source/request/options identity, finalizes the real
callable ABI, constructs one Artifact from that same move-only result and
returns only the joined immutable pair. Backend-ON passes native 7/7 and Python
182/2; backend-OFF passes native 5/5, functional Python 7/7 and full Python
175/9. CP-0043 accepts a separately versioned, two-layer manifest-bound
HIR-authored FP32/BF16 SM120 smoke controller and CPU-only replay finalizer.
Its source, exact-DSO and complete synthetic finalization gates pass. CP-0044
accepts the finalized real-SM120 run: two one-producer frontend compilations and
four fresh non-default-stream lifetimes are numerically correct with no
fallback. The current transaction generalizes the add-only emitter and frontend
identity boundary to a bounded fused-pointwise chain while preserving every
accepted vector-add byte and control. CP-0045 now accepts that compiler/Cubin
transaction at PyPTO `b83fcd3`; V2 GPU numerical correctness is still open.
CP-0046 closes the reviewed nine-case GPU control/finalizer and CPU anchor
boundary. CP-0047/EV-0060 now closes its separately versioned policy-2
real-SM120 execution and CPU-only no-replace finalization. The fixed eighteen
lifetimes pass with candidate-versus-Torch zero ULP and independent-CPU error
at most one ULP. This is the frozen nine-case correctness claim, not a general
operator or performance claim.

The fused-pointwise compiler/Cubin gate is complete. Fresh backend-OFF and
external backend-ON builds, native 3/3 and exact-product Python 1/1 pass; EV-0058
binds both DSOs, four JUnits, all sidecars and diagnostic lineage. Independent
review is GO with P0/P1/P2 = 0. This is CP-0045, not V2 GPU correctness. The
next separate compiler family is
`RowReductionV3` boundary: dense static rank-1-through-32
reduce-last/keep-dim, one flattened outer-row tile/grid (frozen by direct
rank-1/rank-2/rank-3 producer fixtures), and explicit BF16-to-FP32
reduction-to-BF16 conversion so BF16 sum does not silently accumulate in BF16.
Its two source-only commits end at PyPTO `17b2b3c`: the follow-up closes CUDA
Tile element-count and i32 contraction-loop bounds, adds actual no-loop/looped
producer fixtures and complete source/projection goldens. Independent static
re-review is GO with P0/P1/P2 = 0. CP-0048 adds the real-build test fix and
advances the primary checkout to `62eb882`; fresh OFF/ON products, native/
Python gates and four exact Cubin records pass independent review. Reduction
GPU numerical correctness and performance remain pending.
The following structured-matmul source map is also frozen: bounded static BF16
rank-2/equal-batch-rank-3 HIR, TensorIR BF16-by-BF16-to-FP32 matmul, explicit
FP32-to-BF16 output conversion, explicit transpose views, normalization-driven
schedule arity (especially `M=1` decode), and output descriptor 2 as the static
grid source. Source-only implementation now ends at PyPTO `d755117` on top of reviewed
RowReductionV3. Two independent reviews are GO with P0/P1/P2 zero; build,
TensorIR/CUDA Tile production, Cubin, runtime and performance gates remain
pending.

Per D-0018 (2026-08-27) the user relaxed the process gates until the
models run: no per-transaction source reviews, no two-reviewer GO, no
manifest-only ceremonies; a layer passes when its tests and golden
comparisons are green and its build links, with one consolidated review
after 0.8B/9B run. Safety boundaries are unchanged. The execution order
is: finish StructuredMatmulV4 host/Cubin evidence, then generic
pointwise/reduction/indexing/matmul coverage, the TorchInductor PyPTO
CUDA backend plugin, pypto-kernels attention/GDN, the SGLang plugin,
0.8B bring-up, profiling, then 9B, then the D-0017 comparison report.

The user reconfirmed the final objective on 2026-08-26: the end state is
Qwen3.5-9B text generation executing with 100% PyPTO model-forward compute
kernels — handwritten `pypto-kernels` operators plus TorchInductor auto-fused
regions lowered through the PyPTO CUDA backend — benchmarked against the
unmodified SGLang default optimized kernel stack on the same RTX 5090 Laptop
GPU, same model, same workload schedule and batching. Acceptance requires both
the end-to-end comparison and a per-kernel/per-operator breakdown table
(kernel/provider, call counts, GPU time, mean, candidate-versus-baseline
delta per operator class: attention prefill/decode, GDN prefill/decode, GEMM,
pointwise/reduction/indexing fusion), produced from identical profiling
methodology on both lanes. D-0017 freezes this contract.

Checkpoint `CP-0038` accepts the finalized minimal real-SM120
`NvidiaExecutable` correctness v1 report from run `080254`, SHA
`727362d7...272a9`. Static, dynamic-metadata and by-value scalar Artifacts each
complete twice on the caller's non-default stream with reference equality,
external synchronization and explicit unload. Matching v4 finalization joins
all controls, sidecars, serialized compiler inputs, Artifacts, Cubins and live
TargetInfo with no fallback. This closes only the low-level compiler/runtime
launch gate. CP-0042 closes the one-producer frontend Artifact facade, CP-0043
closes its separate smoke controls, and CP-0044 closes HIR-authored vector-add
real-SM120 correctness. CP-0045 closes fused-pointwise compiler/Cubin evidence;
CP-0046 closes its execution controls, CP-0047 closes its real-SM120 numerical
result, and CP-0048 closes RowReductionV3 host compiler/Cubin production.
RowReductionV3 real-SM120 correctness is current, followed by structured
matmul.

Checkpoint `CP-0037` records, but does not accept, v3 run
`pypto-20260825T073624Z-900485-7df250`. Its real GPU child completed six
static/dynamic/scalar lifetimes and published provisional SHA `64c0906b...d34cfe`,
but the no-site finalizer failed because its handwritten dtype order disagreed
with canonical `NvidiaTargetInfo` order. Root A4 `5564008` and manifest-only B4
`7639d82` share exact `[FP32,BF16]` ordering, reject malformed/order-drift
evidence, and pass a clean 225-test/113-subtest suite. No PyPTO rebuild was
needed. CP-0038 later closes the gate with a fresh matching v4 run/finalizer;
cross-version promotion of the v3 provisional remains forbidden.

Checkpoint `CP-0036` records a fail-closed real-runtime diagnostic and accepts
the resulting generic ABI repair, not GPU correctness. Run
`pypto-20260825T052038Z-800777-8e8e83` passed every isolation
gate, observed the real RTX 5090/Runtime/context, compiled all three Cubins and
loaded the first module/function, then failed in static repetition zero during
`prewarm` before packet preparation or launch. PyPTO `206447c` restores the
four-byte dynamic size/stride ABI, validates the live Driver width multiset and
bounded ranges independent of enumeration order, retains signature-ordered
launch pointers, bounds error text, and preserves CUDA Tile's required logical
host block `[1,1,1]`. Fresh ON/OFF builds pass CTest 9/9 and 7/7, exact-DSO
Python 142/2 and 135/9, and the complete product audit. Root `c71f32b` plus
manifest-only `3de4cf7` bind the final DSO through immutable control manifest
v3; post-manifest run `pypto-20260825T071601Z-892819-67acee` passes 224 tests
plus 106 subtests. Its scheduled v3 transaction became the unfinalized CP-0037
diagnostic; CP-0038 later closes the v4 route and frontend execution is current.

Checkpoint `CP-0035` accepts the fixed correctness-only SM120 smoke control
path, not a GPU result. Root implementation `394b75a` and manifest-only commit
`2b53f0a` bind the exact controller, preflight, stop tool, runner, finalizer and
contract blobs. Controller/finalizer/preflights use `-E -B -S`; the child uses
`-I -B -S`, an empty `PYTHONPATH` and no plugins. Parent and child prove the
protected lane has no NVIDIA mapping/compute PID before Torch import, and the
owned watchdog identifies compute only through PID start-tick, descendant and
PGID. Static/dynamic/scalar Cubins compile deterministically CPU-only; replay
through the exact DSO reconstructs full TargetInfo, BuildSpecs, ABI and Cubin.
The post-manifest root suite passes 224 tests plus 106 subtests. At CP-0035 no
CUDA context, module or kernel had been used. Its scheduled live transaction is
now preserved as the CP-0036 failed diagnostic. The later CP-0037 v4 route is
closed by CP-0038; frontend execution is current.

Checkpoint `CP-0034` accepts only the parent-process NVIDIA runtime observation
value at PyPTO `6361f11`. It reuses the private Driver boundary to produce every
live TargetInfo field, numeric Driver/Runtime API versions, authenticated
Runtime-provider path and diagnostic regular-context identity without retaining
handles or loading Cubin. `dlsym(RTLD_DEFAULT)` plus `dladdr` and canonical
expected-path equality never opens libcudart. Distinct-sentinel, provider,
fork-latch and backend-OFF tests pass; fresh ON/OFF products and exact-DSO gates
remain single-DSO and isolated. CP-0034 itself accepted no production
observation; CP-0036 later records the failed diagnostic.

Checkpoint `CP-0033` accepts only the CPU/fake-driver NvidiaExecutable v1
contract at PyPTO `2842a1c`. It adds a separate internal runtime object target,
lazy typed Driver resolution, process/device/context binding, forced function
load, parameter/resource legality, prepared allocation-free launch packets,
graph/module lifetime leases and strict ON/OFF product isolation. Fresh ON
CTest is 9/9 with exact-DSO Python 142 passed/1 skipped; fresh OFF is 7/7 and
134 passed/9 skipped. No real CUDA Driver call or launch occurred. The
exclusive `gpu-benchmark` gate returned 75 for the active protected lane while
reporting no NVIDIA compute PID, so the next transaction remains an exact
non-default-stream RTX 5090 smoke without waiver.

Checkpoint `CP-0032` accepts the persistent ArtifactCache v1 at PyPTO
`c087170`. Cache identity is derived only from the accepted precompile
projection; owner-private descriptor-relative reads fully revalidate canonical
Artifact bytes, and same-shard no-replace publication is durable and
concurrency-safe. Fresh backend-ON/OFF builds, exact-product Python suites, DSO
audits, seven-source provenance negatives and independent API/security/final
reviews pass. No CUDA state, compile-on-miss, eviction, repair, framework or
model behavior is present. The next narrow transaction is a process/device/
CUcontext-bound `NvidiaExecutable` that consumes only a validated Artifact and
accepts the non-null current stream only at launch. CP-0033 now accepts its
CPU/fake-driver contract; real CUDA execution remains the active gate.

Checkpoint `CP-0031` accepts the strict canonical-source producer bridge at
PyPTO `f3bcaac` and TensorIR `1dcb38c`. Exact source bytes plus
CompileRequest/KernelBuildSpec now produce an immutable SM120 Cubin Artifact v1
through the private TensorIR/CUDA Tile/pinned-tileiras route, with no fallback,
ambient override or vendor type in the public API. Fresh ON/OFF products,
source-isolated Python suites, DSO audits, provenance negatives and the root
control regression pass. Provenance covers the clean PyPTO parent plus all six
compiled direct submodules, including msgpack-c, libbacktrace and runtime. The
then-next narrow transaction, the compiler-owned persistent ArtifactCache, is
now accepted by CP-0032. CUDA handles/current-stream launch, frontend-HIR
lowering, operators, frameworks and models remain later gates.

Checkpoint `CP-0030` accepts immutable canonical NVIDIA Artifact v1 at PyPTO
`4a82f2e`. It binds exact clean PyPTO/vendor/pipeline identities, strict SM120
TensorIR options, schedule-to-grid/uniform ABI, one canonical Cubin,
entry/flattened CUDA parameter ABI, cache/loader projections and bounded
canonical MessagePack. TensorIR `b25081a` performs runtime-free CUDA 13.3 ELF,
entry, KPARAM/PARAM_CBANK, constant-bank and PT_LOAD mapping validation. Its
then-next strict producer transaction is now accepted by CP-0031. Cache, CUDA
runtime, framework routes and model execution remain later.

Checkpoint `CP-0029` accepts only the private bounded canonical MessagePack
foundation at PyPTO `6ce1776`. It preserves CompileRequest/KernelBuildSpec wire
identity while adding streaming aggregate allocation limits and BIN support
needed by Artifact. CP-0030 and CP-0031 have now consumed that foundation;
cache, CUDA runtime and frameworks remain later steps.

Checkpoint `CP-0028` accepts only the runtime-free in-memory TensorIR producer
result. TensorIR `2677d1a` emits fully validated TileIR/Cubin bytes plus
complete reconstruction metadata before its legacy runtime object; PyPTO
`4789ae0` preserves one symbol-isolated DSO and a narrow LLVM ABI bridge.

Checkpoint `CP-0027` accepts the private compiler composition only. PyPTO
`5f75568` contains TensorIR `233ab6e`, CUDA Tile `af241704`, and LLVM
`57109bef` in one RPATH-free `pypto_core` DSO; the exact-source compiler suite
passes 123/123. The next transaction is a pointer-free bytes-plus-metadata
Artifact emitted before any runtime-kernel construction. CUDA module/context,
current stream, launch, framework registration and model claims remain open.

Checkpoint `CP-0022` accepts immutable SM120 TargetInfo at PyPTO `042878d` and
freezes the exact Triton dependency/wheel/probe/replacement machinery plus the
shared/exclusive environment transaction law. A live CPU-only control suite
passes beside an active protected lane without exposing CUDA or signalling it.
R0 stays open until the source-anchored Triton wheel replaces the inherited
editable, the CPython 3.12 baseline is locked, and unmodified SGLang baselines
run.

Checkpoint `CP-0023` accepts pointer-free CompileRequest v1 at PyPTO `09e014c`.
Canonical bounded MessagePack, exact toolchain/target policy and the separate
byte/loader-input/device identity projections pass native 2/2 and Python 62/62
CPU-only gates. KernelBuildSpec and every producer/artifact/runtime layer remain
open.

Checkpoint `CP-0024` accepts the exact Triton dependency closure. All ten
archive SHA/byte pairs, expanded trees and manifest `29c073...` are independently
reviewed and source-locked; the networkless tool probe and durable reviewed
cache publication pass. The offline wheel/build/runtime gates remain open.

Checkpoint `CP-0025` accepts pointer-free per-region KernelBuildSpec v1 at
PyPTO `9b3cf71`. Its bounded canonical identity binds source/ABI, semantic
route, exact pipeline revision, all resolved schedule categories,
specialization/mutation and the parent CompileRequest byte identity. Native
CTest is 4/4 and an exact-current-DSO replay is 122/122. Exact producer,
Artifact, cache, CUDA module/current-stream runtime and framework integration
remain open. In parallel, the corrected RPATH-free Triton wheel rebuild is
running under the CPU-only coexistence watchdog.

Checkpoint `CP-0026` freezes the exact PyTorch-pinned Triton reference wheel.
Its complete audit and pip-free fresh probe pass; it remains uninstalled and
cannot satisfy PyPTO strict coverage. The active implementation lane now moves
to the private TensorIR/CUDA Tile/exact-LLVM static build and artifact seam.

Checkpoint `CP-0005` has source/model/environment provenance, the first
standalone semantic operator layer, Torch constructor-dispatch law, and
candidate/baseline process isolation frozen. R0 remains open until exact Triton
wheels replace the inherited editable source, the CPython 3.12 baseline
environment is locked, and unmodified SGLang baselines are captured. In
parallel, the first P1 build-only object boundary is accepted at PyPTO
`042878d...`; the next compiler transaction is the pointer-free CompileRequest
contract, while exact Triton replacement proceeds as R0 compatibility work.

Checkpoint `CP-0006` additionally freezes the strict runtime-coverage evidence
contract: fixed collector revision, closed trace and artifact-registry digest
reconciliation, exact non-vacuous denominators, failure latching, and exclusive
report ownership. It is deliberately not runtime coverage evidence. The next
source-only P1 transaction is immutable SM120 target identity, but it must land
only after the staged single-DSO object-boundary commit is independently gated.

Checkpoint `CP-0007` adds catalog-bound operator provenance identity and the
pinned eager-first CUPTI collector map without moving either boundary into the
wrong project. The collector is not implemented and cannot assert a closed
world yet.

Checkpoint `CP-0008` freezes paged-attention ABI v1 and its metadata reference
contract. This is the semantic boundary for later reference and CUDA Tile
implementations, not an attention correctness or performance milestone.

Checkpoint `CP-0009` adds the deterministic CPU numerical reference and state
continuity gate. Independent PyTorch comparison and all CUDA work remain open.

Checkpoint `CP-0010` freezes unified GDN core ABI v1 and the paired-state lease
contract. Numerical GDN and all state-preparation/CUDA work remain open.

Checkpoint `CP-0011` adds the GDN paired-state numerical reference and exact
partition-continuity gate. Independent Torch comparison and CUDA remain open.

Checkpoint `CP-0012` records a source-reviewed but deliberately unbuilt SM120
TargetInfo candidate in a separate worktree. It does not advance P1 acceptance;
object-DSO validation and ordered integration remain mandatory.

Checkpoint `CP-0013` adds the structured matmul numerical oracle. TensorIR/CUDA
correctness comparison and performance remain open.

Checkpoint `CP-0014` closes the source/package identity boundary between the
standalone operator producer and both zero-diff framework adapters. It does not
make the compiler or any operator executable. The packaging-time heavy
preflight was green, but the action-boundary recheck is now red for a new
protected zcode TP=2 vLLM/gem5 lane. Continue only light work until a fresh
green result permits the single-DSO gate before TargetInfo integration.

Checkpoint `CP-0015` independently cross-checks all three current scalar
operator references against CPU Torch while keeping Torch out of the standalone
runtime product and parent import process. CUDA correctness/performance remains
open. The protected heavy lane still blocks the ordered single-DSO gate.

Checkpoint `CP-0016` freezes the source-only operator benchmark evidence
contract and its fail-closed comparison boundary. It publishes no result and
does not advance performance acceptance. Atomic publication and tuning reuse
remain ordered after real CUBIN and complete TargetInfo identity.

Decision `D-0008` freezes generic StateBundle ownership and the post-TargetInfo,
post-current-stream landing order. It is a design constraint, not an
implementation milestone; no state transfer API or executor exists yet.

Checkpoint `CP-0017` freezes the selected SGLang state lifecycle route and
adapter obligations while keeping registration explicitly unready. It does not
move StateBundle implementation ahead of compiler/runtime prerequisites.

Checkpoint `CP-0018` freezes the pinned TorchInductor backend surface before
implementation. The 31-source audit plus full 346-file manifest proves no
TileKernel/cuTile abstraction exists in this pin and records every reviewed
registry, cache, current-stream, explicit fallback and scheduling bypass that a
zero-diff plugin must own or reject. It does not register a backend or generate
code; all readiness flags remain false until the ordered compiler/runtime gates
close.

Checkpoint `CP-0019` repairs the executable acceptance specifications for the
pending single-DSO and TargetInfo transactions and freezes D-0009 from pinned
TensorIR source. CompileRequest is a data-only program/target policy;
byte-producing region identity belongs to KernelBuildSpec; both TensorIR and
direct CUDA Tile routes produce one exact-producer-bound Artifact; loaded
executables are process/device/CUcontext bound. This checkpoint executes none
of those heavy or runtime gates.

Checkpoint `CP-0020` records the third consecutive goal-turn failure of the
same protected-workload gate: seven `zcode-vllm-tp2-v4` processes remain and
MemAvailable is 28.5 GiB below the 32 GiB floor. The goal is blocked on external
state, not complete or narrowed. Resume at the frozen single-DSO runbook only
after a fresh heavy preflight returns zero.

Checkpoint `CP-0021` records user resume and accepts the single-DSO compiler
boundary. Native ownership/CTest, editable and wheel JUnit, fresh one-DSO
packaging, installed dependency/import/console-script and symlink gates all
pass; two audit-script false positives are preserved with successful recovery
run IDs.

Checkpoint `CP-0022` integrates the exact 34-path TargetInfo candidate as
`042878d`. Fresh native CTest is 2/2; the new one-DSO wheel and clean install
pass 31 targeted cases, 10,209 full-suite cases with 57 unchanged skips, and
the independent symlink case. Source/artifact lineage is rebound by a separate
read-only recovery audit. This accepts target identity/build/package/API only,
not CUDA compile or launch. EV-0035 additionally accepts the exact Triton gate
tooling and live protected CPU-only control path; it explicitly records that no
dependency materialization, wheel, GPU smoke, or replacement has run.

Checkpoint `CP-0023` lands only the immutable program/target CompileRequest
data contract. Independent review forced explicit MessagePack allocation limits
before acceptance. The next compiler commit is per-region KernelBuildSpec; it
must not be collapsed into TensorIR composition or ArtifactCache work.

Checkpoint `CP-0024` records only reviewed compatibility inputs. It does not
promote Triton into PyPTO coverage, prove a wheel, or relax the separate
reference-only GPU-smoke and environment-replacement gates.

Checkpoint `CP-0025` lands only the immutable per-region KernelBuildSpec data
contract. A stale installed editable DSO was detected and explicitly excluded
from acceptance; EV-0038 binds the exact current source DSO replay. The next
compiler transaction is exact producer-bound bytes-plus-metadata Artifact, not
runtime handles or framework codegen.

Checkpoint `CP-0026` closes only the reproducible Triton reference-wheel
boundary. Environment replacement and reference GPU smoke are deferred until
the unmodified SGLang baseline requires them. No further Triton feature work is
on the candidate backend path.

Checkpoint `CP-0027` closes only the unified private compiler build boundary.
TensorIR's public compiler usage requirements now support a real parent build,
and dynamic CUDA declarations no longer leak an absolute toolkit RUNPATH. This
does not yet prove a PyPTO lowering, a kernel artifact, CUDA execution, a wheel,
or any Torch/SGLang route.

Checkpoint `CP-0028` accepts only a process-independent in-memory producer
value, not a persistent PyPTO Artifact. Full TileIR/Cubin structure, SM/ABI,
argument/grid metadata, hostile environment, exact assembler bytes and legacy
reconstruction are gated. Canonical serialization, request/build-spec digest
binding, subprocess transfer and cache publication remain explicitly open.

Checkpoint `CP-0029` accepts only the bounded private MessagePack foundation.
Streaming structural limits close decoded-object amplification before the
allocating parse; ON/OFF builds and canonical request/spec replay pass. No
Artifact schema, producer bridge, cache, CUDA runtime or framework route is
claimed.

Checkpoint `CP-0030` accepts only persistent Artifact v1 and its strict
runtime-free Cubin/ABI validation. The producer result remains a manually
constructed boundary in tests; no compiler entrypoint yet consumes canonical
source plus CompileRequest/KernelBuildSpec to create it. ArtifactCache,
subprocess compilation, CUDA module/current-stream launch and framework/model
work remain explicitly open.

Checkpoint `CP-0031` accepts only the strict in-process source-to-Cubin
producer bridge and its bounded exact-assembler process boundary. It does not
accept cache publication, CUDA loading/launch, PyPTO frontend-HIR lowering,
generic codegen, operators, TorchInductor, SGLang or Qwen execution.

Checkpoint `CP-0032` accepts only trusted-local persistent Artifact storage.
It does not accept CUDA device/context/module/function state, resource or
workspace legality, current-stream launch, CUDA Graph, frontend-HIR lowering,
operators, TorchInductor, SGLang, Qwen correctness/coverage or performance.

Checkpoint `CP-0033` accepts only modeled executable lifecycle, packing,
legality, concurrency/fork rules and ON/OFF product isolation. It does not
accept real libcuda resolution, Cubin module load, current-stream execution,
CUDA numerical correctness, CUDA Graph, frontend-HIR lowering, operators,
TorchInductor, SGLang or Qwen execution.

1. Create the control repository, persistence documents, safety preflight, and
   isolated directory layout.
2. Materialize the authorized PyPTO baseline and clean official upstream
   checkouts at the exact versions in `VERSIONS.lock`.
3. Initialize independent `pypto-kernels` and `pypto-framework-plugins` Git
   projects.
4. Clone the `triton-dev` environment into a project-local prefix without
   modifying the original environment.
5. Copy Qwen3.5 weights from the read-only AMD simulator tree into `models/`
   after the default idle gate, or only with the explicit CPU-only coexistence
   flag while the 24 GiB and protected-NVIDIA-compute boundary remains green;
   recheck at each file EOF/publication boundary and verify every hash.
6. Generate the checkout-grounded `docs/implementation_map.txt` and freeze the
   unmodified SGLang baseline before compiler changes.

## Milestone ladder

- R0: workspace/provenance/baseline.
- P1: PyPTO compiler/backend split with unchanged Ascend tests.
- P2: TensorIR/CUDA Tile SM120 runtime closure and PyTorch current-stream ABI.
- P3: generic fused-loop codegen, structured matmul, runtime/cache/tuning.
- P4: zero-diff TorchInductor compatibility plugin and strict MLP gate.
- P5: paged full-attention correctness and performance.
- P6: GDN decode/prefill correctness and performance.
- P7: zero-diff SGLang plugin and Qwen3.5-0.8B strict coverage.
- P8: 0.8B stabilization and full profiling.
- P9: Qwen3.5-9B correctness, strict coverage, and SM120 tuning.
- P10: final E2E benchmarks, coverage proof, per-kernel PyPTO-versus-SGLang-
  default breakdown, and the final performance report.

Every milestone is correctness-first, then performance, then evidence and a
checkpoint commit. A green smoke test is never promoted to a later acceptance
claim.
# Execution checkpoint: 2026-08-30 (revision 58)

The revision-57 writing plan is now executed through the evidence and document
stages below. These are the current facts to use when resuming; older
19-token, 250-patch, or single-start numbers are historical only.

- Typed NVIDIA construction: complete. Canonical bridge paths use the pinned
  TensorIR ODS/OpBuilder builder; source lint, positive/negative verifier,
  native CTest, and source replay/clone probe pass. PyPTO is c27629e (300
  commits), TensorIR is db41d073 (89 commits).
- Package/source identity: complete. The framework-plugin subtree split is
  9ee85c3 with tree f1971eb; vendor/source-lock and the fresh-clone verifier
  agree. The typed DSO and current wheels are bound in the current evidence
  sidecars.
- Operator correctness: complete. The current GPU run
  pypto-gpu-bounded-20260830T150140Z-2411664-e68395 passes all eight suites;
  the compact binding is state/evidence/operator-regression-current.json.
- Model correctness/coverage: complete for both Qwen3.5-0.8B and 9B. Each has
  three current-wheel candidate starts, ten stable Engine requests per start,
  one teacher-forced strict trace, and zero unknown/fallback compute. The
  compact bindings are qwen35-0.8b-model-gate-current.json and
  qwen35-9b-model-gate-current.json.
- Performance: the four-start pair is retained as diagnostic evidence but was
  invalidated by revision 62: 3/4 PyPTO starts crossed the high-frequency NVML
  4 GiB GPU free floor. Its 17.31% ratio is not a formal headline. Matched and
  optimized stock comparisons both require resource-compliant reruns.
- Full-model CUPTI/NVTX phase profile: attempted candidate collection was
  stopped at model-load KV-cache qualification (4 GiB controller floor); no
  phase percentage or residual attribution is promoted.
- Full-model eager control: one timing-only matched-provider run is complete.
  It shows 15.3418 versus 15.4100 output tok/s, but the compile-request lane
  did not invoke CompilerInterface, so no causal whole-model compile speedup is
  claimed; the supported causal result remains the SwiGLU operator ablation.
- Inductor ablation/breakdown: complete at operator scope. The current
  performance-only six-report summary is qwen35-9b-inductor-ablation-current.json;
  it records 6-to-1 launches (83.33%), official NV speedups of +30.54%/+19.92%,
  PyPTO changes of -79.60%/-80.91%, and compile overhead of +54.44%/+65.72%.
  The aligned eight-start operator matrix remains the attribution evidence.
- Documentation: Chinese README, English README, local blog Markdown and
  single-file HTML have been refreshed. The blog has the independent TensorIR
  suitability chapter, the unchanged article-demo chapter, the exact
  Inductor kernel excerpt, current tables, and the requested numbered headings.
  The five-role screenshot manifest is provisional.
- Final requirement audit matrix: docs/final_requirement_matrix.md maps each
  original and follow-up requirement to evidence or an explicit open gate.
- GPT-Image-2: pending. OPENAI_API_KEY is absent; do not substitute another
  model. Keep the prompts and PENDING_GPT_IMAGE2 markers until authorized
  generation and image/hash inspection are available.
- Article demos/GUI: the byte-for-byte 151-file/66-entrypoint corpus and
  57/57 help audit are complete, but Ascend simpler_setup/KernelType.MIX
  runtime blockers remain. Windows Terminal purple captures for the current
  model and typical demo are pending; no fake screenshots are allowed.

Resume order:

1. If a clean GPU/Windows capture window becomes available, capture the two
   pending roles and optionally qualify optimized stock; bind each result to
   the same run JSON and update the manifest.
2. If OPENAI_API_KEY is authorized locally, generate the three GPT-Image-2
   figures from immutable current evidence and record model/prompt/input/image
   hashes.
3. Re-run the final audit and render_blog.py. Do not stage the local blog/HTML
   or diagnostic probes.
4. Completed: staged only the authorized README/reproduction deliverables,
   excluded the local blog/HTML/probes, committed b4eae99, and pushed
   origin/release/qwen35-sm120-v1.

---
# Requirement-audit checkpoint: 2026-08-30 (revision 59)

This checkpoint is a second-pass audit of the original blog/README brief and
both follow-up requests. It records clarifications that were implicit in
revision 58 so a later execution cannot accidentally satisfy the prose while
missing the requested evidence. It does not promote any pending experiment.

## A. No omitted writing requirements

The final document workflow must preserve the following order and scope:

1. **Opening announcement and legal NOTE.** Start with the measured headline,
   its denominator, launch/fusion scope and the current limitation; then give
   the CANN boundary, non-commercial research context, the linked interview
   timestamp and takedown route. A stated research motive or interview remark
   is never presented as a licence exception.
2. **Background before implementation.** Introduce PyPTO and TensorIR
   separately, including their public URLs, roles and core features. Immediately
   after those introductions, add an independent section explaining why this
   bridge is useful. The explanation must answer all three questions explicitly:
   whether a PyPTO-selected tile shape can lower, whether TensorIR is
   tile-aware, and whether CUDA Tile is a target/backend rather than a synonym
   for TensorIR. The answer is conditional on typed shape/stride/layout/dtype,
   iteration and mutation contracts, with a fail-closed example.
3. **Independent article-demo chapter.** This chapter appears after the
   background and before the framework/operator implementation chapters. It
   states the NVIDIA-learning motivation, links the exact WeChat article, names
   the byte-for-byte corpus under `demo/`, explains the all-entrypoint matrix,
   and shows one unchanged command plus a numerical/golden precision result.
   Until the Ascend-only runtime blocker is removed, it must show the blocker
   and `PENDING`, never a fabricated success screenshot.
4. **Framework chapters.** Keep PyPTO-to-TensorIR and TorchInductor-to-PyPTO as
   two separately titled feature inventories. The first must name the PyPTO and
   TensorIR changes and the typed NVIDIA ODS/`OpBuilder` boundary; the second
   must name capture, graph matching, generated DSL, artifact/cache, stream and
   SGLang integration behavior. Canonical construction may not use string
   concatenation, format-built operation names or ad-hoc textual IR. Printers
   are diagnostic/fixture-only.
5. **Operator chapters.** Inventory the independent handwritten library, map
   it to Qwen call sites, and walk through one source-selected complex kernel.
   Separately show an exact Inductor-generated fused kernel and state which
   operators/layers fuse and which matmul boundaries remain. The union claim is
   trace-based and must state its `ModelRunner.forward` denominator.
6. **System test and performance chapters.** Show the exact Chinese prompt,
   generated token stream, launch/provider trace and semantic/correctness
   evidence. Report eager, official NVIDIA Inductor, PyPTO Inductor, matched
   stock and (only if qualified) optimized stock with units, start-level
   aggregation, cold/first-compile boundaries and launch counts. Put the same
   measured fusion/launch result in the opening and conclusion. Operator-only
   speedups are never labelled whole-model acceleration.
7. **Reproduction and publication.** Chinese README is the default and links
   to English (and back); both expose build, correctness, performance-only
   regression, demo matrix and exact-prompt model commands. The blog remains
   local, the offline single-file HTML is regenerated from it, and only the
   audited README/reproduction deliverables are staged. A final requirement
   matrix, link/hash check, bilingual parity check and staged-diff check are
   mandatory before any publication update.

## B. Evidence and visual invariants

- The five terminal roles are independent gates: build, operator correctness,
  performance/breakdown, exact-prompt Qwen3.5-9B inference, and one typical
  unchanged article demo. The preferred capture is Ubuntu hosted by the
  requested purple PowerShell/Windows Terminal profile. A user-capture handoff
  with an exact command and sidecar is the only valid fallback when that GUI
  is unavailable; a black, edited or mismatched frame is discarded.
- Every ablation/breakdown figure must be generated with GPT-Image-2 from the
  immutable evidence JSON for that result. The figure sidecar records model,
  prompt, source hash, run IDs and image hash; the table remains authoritative.
  Missing local authorization leaves `PENDING_GPT_IMAGE2` and forbids a
  substitute model or hand-drawn result.
- “100% PyPTO” means every CUPTI compute activity in the declared
  `ModelRunner.forward` window maps exactly once to a handwritten or
  Inductor-generated PyPTO artifact, with zero unknown/fallback activities. It
  does not silently include tokenizer, sampling, memcpy/memset, CPU staging or
  work outside that window.

## C. Latest resource-gated probe (not an accepted result)

The temporary optimized-config probe used `cpu_offload_gb=4` and
`mem_fraction_static=0.78`, with Inductor compile threads limited to one. Run
`pypto-gpu-bounded-20260830T160256Z-2428456-b945e5` reached model loading and
CUDA-graph capture, then the bounded controller stopped it with
`abort_reason=nvidia-coexistence-audit` after an `nvidia-smi` query timed out;
no performance report was produced. Post-cleanup found no owned or foreign
compute process. This run is retained as a resource/telemetry diagnostic only:
it supplies no optimized throughput, graph-replay, overlap or compile-time
number and must not change the formal lane or document headline.

When a clean exclusive window is available, qualify optimized stock in this
order: (1) zero foreign compute and host/GPU floor preflight, (2) one dry run
with the formal lane configuration, (3) four fresh starts with the same
chat-template/64-token workload and resource fields, (4) source/hash and
runtime-observed graph/overlap checks, and (5) regenerate the pair summary,
figures, README/blog/HTML and audits. If any step fails, preserve the raw run
and keep the lane `OPEN`.

## D. Resume and completion contract

The next safe actions are: capture the two pending terminal roles when the
Windows GUI is controllable; obtain explicit local GPT-Image-2 authorization
and generate the three requested figures; resolve the Ascend runtime or obtain
an authorized compatible device for the unchanged demo matrix; optionally
qualify optimized stock and the full-model CUPTI/NVTX profile under exclusive
resources; then rerun the final audit/render. Do not rerun already accepted
operator/model gates merely to replace a screenshot, and do not edit the
copied article sources.

The goal is complete only when every requirement row is either PASS with a
fresh, hash-bound artifact or is explicitly accepted by the user as an open
limitation. Until then, `PLAN.md`, the final requirement matrix, raw failed
runs and pending sidecars are the durable handoff; no wording change may turn
an open gate into a performance or correctness claim.

---
# Resource-gated inference checkpoint: 2026-08-30 (revision 60)

The formal single-start optimized lane was retried after a clean admission
window became available. `sglang-optimized` with the frozen `matched` memory
configuration (`cpu_offload_gb=2`, `mem_fraction_static=0.69`) passed baseline
identity checks, loaded all four Qwen3.5-9B shards, and entered the configured
CUDA-Graph capture. During capture the controller observed only its own CUDA
PID and a 5,197 MiB GPU-free margin, but three `nvidia-smi --query-gpu` calls
timed out at the 10-second boundary. The bounded policy therefore terminated
run `pypto-gpu-bounded-20260830T161746Z-2433114-b55eb2` with
`nvidia-telemetry-unavailable`; no timing report was written and no optimized
number is accepted. Cleanup was complete and the post-audit found no compute
PID. The compact, sanitized record is
`state/evidence/optimized-lane-diagnostic-current.json`.

This is a tooling/resource qualification result, not evidence that CUDA Graph
or overlap is slow or fast. The safety controller remains fail-closed; do not
increase its timeout or bypass its audit merely to obtain a percentage. On the
next genuinely stable window, rerun one formal start first, then the required
four-start matrix only if the report contains complete warm/cold metrics,
runtime-observed graph state, source identity and resource telemetry. A valid
matrix must replace the invalidated four-start matched diagnostic pair and
trigger regeneration of all dependent summaries and figures.

---
# Measurement-controller checkpoint: 2026-08-31 (revision 61)

The repeated optimized-lane aborts exposed a measurement-infrastructure issue:
the WSL driver can block an `nvidia-smi --query-gpu` call while a CUDA Graph is
being captured even though the bounded child is the only NVIDIA compute PID.
The frozen `tools/preflight.py` contract was deliberately left byte-identical
(`0b9884f8...053e3d1`). Instead, controller commit `296a89e` adds a narrow
read-only ctypes NVML fallback in `tools/nvidia_nvml.py`, used only after the
frozen CLI query raises an OS/command/runtime error. If both APIs fail, the
original error is propagated and the controller remains fail-closed. Every
controller audit now records whether each query came from `nvidia-smi` or
`nvml-ctypes`.

The fallback has unit coverage and a live `-E -B -S` identity/PID probe on the
RTX 5090. It is a tooling fix, not performance evidence. Re-run the formal
optimized lane only after protected workloads leave; accept the lane only if
the resulting report contains complete warm/cold metrics, resource telemetry,
source identity and runtime graph fields. The optimized aborts alone do not
change matched numbers; revision 62 separately invalidates that pair from its
own high-frequency resource evidence.

---
# Performance-evidence correction: 2026-08-31 (revision 62)

The high-frequency NVML summaries in the previously labelled accepted
PyPTO/matched pair exposed a validation omission in
`tools/summarize_qwen_performance_pair.py`. The controller's one-second
`nvidia-smi` audit accepted all four children, but the independent 100 ms NVML
sampler recorded PyPTO minimum free memory of 4,185,329,664; 4,185,264,128;
4,185,067,520; and 4,295,708,672 bytes. The first three are below the fixed
4,294,967,296-byte floor. Therefore the pair is now
`invalidated-resource-and-control`; its 2.6671/15.4100 tok/s and 17.31% ratio remain
reproducible diagnostic values only, not a release headline.

The old pair also failed the control comparison: PyPTO used
`cpu_offload_gb=0`, while matched used `2`. A new performance-only memory
envelope sets both timing lanes to `cpu_offload_gb=2` and
`mem_fraction_static=0.78`; correctness/model-gate memory settings remain
unchanged. The pair summarizer now invokes `matched_lane_comparability` and
rejects any remaining control mismatch before computing a ratio.

The summarizer now fails closed on every start for NVML availability, integer
resource fields, nonempty sampling, 4 GiB GPU free, 12 GiB host free, thermal
throttling and across-start configuration drift. The corresponding regression
tests include an exact-floor pass and a one-byte-below rejection. Correctness,
stability and 100% model-forward coverage gates are independent and remain
accepted.

The timing-only eager sidecar is regenerated against the invalidated pair. Its
single eager run and all four matched starts independently pass the resource
floors, so that non-causal control remains usable; it records that the source
pair is not accepted and consumes only the matched subset. The audit binds its
source SHA and forbids promoting the observed difference to a compile speedup.

Resume performance work only after protected workloads leave. First rerun four
PyPTO and four matched starts under the same frozen workload with the repaired
controller and summarizer; all eight reports must pass their 100 ms resource
summaries. Then qualify optimized stock and collect the three-lane CUPTI/NVTX
matrix. Regenerate README/blog/HTML and GPT-Image-2 inputs only from that new
accepted evidence. Until then, every 17.31% occurrence must say `diagnostic`,
and the requirement matrix keeps both full-model comparisons `OPEN`.

---
# Matched-performance contract correction: 2026-08-31 (revision 63)

Revision 62 found a second independent reason the old pair was not matched:
PyPTO used `cpu_offload_gb=0`, while stock matched used `2`. The sidecar status
is therefore `invalidated-resource-and-control`. Correctness and model coverage
retain their accepted lane-specific memory settings; only the performance and
profile products receive a new common 9B envelope:
`cpu_offload_gb=2, mem_fraction_static=0.78` for both PyPTO and matched.
Optimized stock retains its separately disclosed `2 / 0.69` capture envelope.

The timing worker now validates its own 100 ms NVML summary before leaving
`status=complete`: nonempty integer resource fields, 4 GiB GPU free, 12 GiB
host free, no NVML error and no thermal throttling are mandatory. The pair
summarizer repeats this validation, checks within-lane configuration stability,
and invokes `matched_lane_comparability` before computing a ratio. The eager
control consumes only the independently resource-valid matched subset and
records that its source pair is not accepted.

The CUPTI/NVTX worker now uses the same performance-only server configuration,
collects the same high-frequency resource stream, and carries resource minima
into reconciliation. All nine reports must share one GPU identity and pass the
resource validator before any phase gap is reported. The formal commands in
both READMEs and the blog use `--optimized-memory-mode matched`; historical
`19+64` performance wording is rejected by the document audit.

The next exclusive resource window runs, in order:

1. one PyPTO and one matched qualification start using the new common envelope;
2. only if both reports are complete, the independent interleaved 8-start
   `--pair-matrix` (four PyPTO and four matched starts);
3. only if the pair and `matched_lane_comparability` pass, the 12-start full
   matrix including optimized stock, followed by the nine-start CUPTI/NVTX
   matrix; and
4. regenerate the pair/profile summaries, README/blog/HTML, image prompts and
   final audit from those exact reports.

Protected zcode/gem5/SGLang workloads currently keep heavy preflight red. Do
not signal them and do not launch any formal GPU lane until that preflight
returns zero naturally.
