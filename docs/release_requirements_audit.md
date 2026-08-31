# Qwen3.5 SM120 release requirements audit

This checklist is the publication gate for the Chinese blog, single-file HTML,
and bilingual README. A checked item must link to current-release source or
machine-readable evidence; historical checkpoints do not satisfy an unchecked
release item.

Current machine-readable audit: state/evidence/blog-requirements-audit-current.json
(status=pass, with explicit open gates for optimized stock, full-model
CUPTI/NVTX profile, Windows GUI captures, article-demo device runtime, and
GPT-Image-2 authorization). The item-by-item mapping is
docs/final_requirement_matrix.md; this checklist remains the detailed review
template and is not permission to promote an open item.

## Source and licensing

- [ ] The release source reconstructs the exact PyPTO, TensorIR, CUDA Tile,
      kernel, plugin, PyTorch, and SGLang revisions from a fresh clone.
- [ ] Bundle identity, patch replay, final trees, submodules, and clean-source
      state pass independently.
- [ ] Public commands contain no workstation-only absolute source, model, DSO,
      driver, or CUDA runtime paths.
- [ ] Third-party licenses and modifications are attributed accurately.
- [ ] The exact CANN license wording, PyPTO/TensorIR public URLs, interview
      speaker/title, approximately 2:34:00 passage, and the WeChat article URL
      are source-checked; paraphrase is not presented as a verbatim quote.
- [ ] Every article demo is pinned by upstream commit/path/hash/license and its
      source is byte-for-byte identical before and after compatibility runs.
- [ ] Written authorization for public NVIDIA use/distribution has been
      reviewed before any push; non-commercial intent and interview remarks
      are not represented as a license exception.

## Compiler and framework implementation

- [ ] The implementation inventory covers every release PyPTO/TensorIR change
      and maps each feature to source, tests, Qwen call sites, and providers.
- [ ] The blog distinguishes typed ODS/OpBuilder modules from canonical MLIR
      text emitters and does not claim that text emission is fully removed.
- [ ] Canonical PyPTO-to-TensorIR/CUDA Tile construction contains no
      string-concatenated program path (`source +=`, format-built operations or
      stringly-typed op names); typed OpBuilder/ODS construction is covered by
      a source lint and positive/negative verifier tests.
- [ ] Hash-bound positive and negative traces demonstrate
      `PyPTO DSL -> HIR -> TensorIR ODS/iteration/tile/layout -> CUDA Tile IR ->
      Cubin ABI`, including the exact condition under which a PyPTO tile shape
      lowers or fails closed.
- [ ] The Inductor backend uses the pinned public backend entry point and keeps
      upstream PyTorch and SGLang source trees unchanged.
- [ ] A real Qwen packed SwiGLU subgraph executes through
      `torch.compile(backend="pypto")` with row-pitched input strides.
- [ ] The model trace contains nonzero `pypto.generic` artifacts with stable
      `torch-inductor:*` source nodes; configuration flags alone do not count.
- [ ] The README and blog state that SwiGLU pointwise is fused but the complete
      MLP, including its two matmul boundaries, is not one kernel.
- [ ] Exact-release source extraction records path/revision/line/hash for the
      representative handwritten kernel, FX body, generated PyPTO DSL, and
      current-stream launch wrapper used in the article.
- [ ] A trace-derived operator-closure manifest proves that handwritten and
      Inductor-generated PyPTO artifact sets jointly equal every observed Qwen
      model-forward compute activity; a manually curated inventory is not the
      proof.

## Correctness and coverage

- [ ] The exact-final-revision operator regression covers all 18 handwritten
      graphs and the Inductor SwiGLU at real 0.8B/9B shapes.
- [ ] Correctness reports include dtype, shape, stride, seed, tolerance,
      revision, DSO identity, state mutation, and raw error metrics.
- [ ] The external article-demo policy classifies every entrypoint against the
      byte-locked source: computational entries have a named NVIDIA adapter;
      entries using distributed, CCE, NPU/ACL, or simpler hardware APIs carry
      source-line evidence and are explicitly skipped.
- [ ] Every supported teaching computation compiles/runs (where marked strict)
      or performs an independent CUDA numerical reference check. Reference-only
      results are never described as strict PyPTO compiler evidence.
- [ ] Draft/transitive non-entrypoint files and unmapped model computations are
      distinguished from supported demos and remain visible in the denominator.
- [ ] Clean stock SGLang produces a frozen reference before candidate
      measurement.
- [ ] Qwen3.5-0.8B and 9B complete multi-token prefill/decode correctness.
- [ ] Qwen3.5-9B passes three fresh starts by ten exact-prompt requests with
      exact greedy token sequence and frozen per-step logit policy.
- [ ] Every accepted prefill/decode model-forward window reports non-vacuous
      100% PyPTO compute coverage, zero fallback, zero unknown activity, zero
      dropped CUPTI records, and zero policy violations.
- [ ] Tokenizer, sampling, host work, memcpy/memset, and any CPU offload are
      visible and clearly outside the model-forward compute denominator.
- [ ] Framework bookkeeping exclusions have explicit external-correlation
      provenance and separate counts; merely labelling a compute kernel as
      framework cannot remove it from the denominator.

## Performance and attribution

- [ ] The performance process accepts no correctness policy or reference
      logits and executes no value, token, or text comparison.
- [ ] Candidate, matched stock, and optimized stock use the same model, prompt,
      pinned 31-input chat-template/64-output workload, BF16, TP1, and
      concurrency one.
- [ ] A candidate causal ablation holds all handwritten PyPTO/model settings
      fixed and changes only eager pointwise regions versus PyPTO-Inductor
      fusion; this eager lane is not represented as 100% PyPTO.
- [ ] A separate stock eager versus official NVIDIA Inductor control uses the
      same region and full-model workload, and product comparisons against
      matched/optimized SGLang are not conflated with compiler ablations.
- [ ] Resolved backends and actual torch.compile/CUDA Graph state are recorded,
      rather than inferred from requested flags.
- [ ] Each lane has four valid fresh starts and forty raw requests in the
      frozen interleaved order; invalid/co-tenant starts are retained as
      rejected evidence rather than selectively sampled.
- [ ] TTFT, TPOT, ITL, E2E latency, output throughput, cold costs, memory,
      clocks, power, temperature, and throttle reasons are reported with raw
      samples and confidence intervals.
- [ ] Cold results distinguish engine/weight startup, first compile-trigger
      request and warm execution; PyPTO records Dynamo, Inductor, frontend,
      TensorIR/tileiras, artifact-load and first-launch phases where observable.
- [ ] Launch counts are separately scoped to generated subgraph,
      `ModelRunner.forward`, and whole request, and distinguish event count,
      unique artifact count and source graph count.
- [ ] PyPTO relative performance is computed from median output tokens/second
      separately against matched and optimized stock; percent-of-baseline and
      signed speedup formulas/directions are printed beside the values.
- [ ] Independent CUPTI/NVTX profiles compare semantic operator phases and
      reconcile phase deltas plus host/scheduler/memcpy/graph residuals to the
      total performance gap.
- [ ] Linear and LM-head shapes receive explicit A/B measurement because the
      historical diagnostic profile identified matmul as the dominant cost.

## Documentation and visual evidence

- [ ] `README.md` is Chinese by default and links to `README_EN.md`; English
      links back to Chinese.
- [ ] Bilingual section IDs, commands, links, supported/unsupported statements
      and rendered measurements pass semantic parity checks.
- [ ] A fresh reader can follow README commands through bootstrap, build,
      operator correctness, unchanged article demos, pure performance, and
      exact-prompt 0.8B diagnostic plus 9B inference.
- [ ] A fresh-clone/fresh-prefix transcript executes the README in order without
      hidden editable installs, ambient caches, undocumented variables or
      workstation-only paths.
- [ ] Correctness and performance are separate checked-in regression products;
      the performance process accepts no golden/reference/tolerance input and
      imports or executes no numerical/token/text comparison.
- [ ] README and blog metrics are rendered from one immutable release summary
      and pass an automated consistency check.
- [ ] The blog uses Chinese numbered first-level and Arabic numbered
      second-level section headings and includes all requested architecture,
      framework, operator, testing, breakdown, limitation, and summary topics.
- [ ] Build, operator correctness, performance, 9B generation, and one typical
      unchanged article demo each have a genuine Ubuntu/Windows Terminal purple
      profile screenshot or an explicitly documented user-capture handoff and
      exact command; no screenshot is fabricated or image-generated.
- [ ] Every ablation/breakdown figure is generated with GPT-Image-2 from a
      hashed immutable JSON source, has model/prompt/run/image provenance and
      useful alt text, and is visually checked against an adjacent exact table.
- [ ] Source excerpts and every screenshot/figure sidecar point to the same
      release identity as the prose; stale images fail closed.
- [ ] The HTML is a single offline file with inline styling, figures, and image
      data, renders without external assets, and passes desktop plus narrow
      viewport checks without clipped tables, code or Chinese text.
- [ ] The finished documents contain no `xx`, `TBD`, stale `PENDING_*`, missing
      required result or operator-level number promoted to a model headline.
- [ ] The Markdown blog and HTML remain local and absent from every commit.

## Final repository boundary

- [ ] Build and CPU test parallelism is exactly 24; timing-sensitive GPU runs
      are serial by experimental design.
- [ ] Existing user modifications in `projects/pypto-kernels` are byte-for-byte
      unchanged.
- [ ] Upstream PyTorch, SGLang, and Triton remain clean.
- [ ] No external process was signalled or terminated.
- [ ] Local commits are small and auditable.
- [ ] The staged commit contains README/reproduction assets only after a final
      `git diff --cached` audit; local blog Markdown/HTML and diagnostic probes
      are absent.
- [ ] No public push occurs before the license authorization gate is cleared.
