# Qwen3.5 SM120 release requirements audit

This checklist is the publication gate for the Chinese blog, single-file HTML,
and bilingual README. A checked item must link to current-release source or
machine-readable evidence; historical checkpoints do not satisfy an unchecked
release item.

## Source and licensing

- [ ] The release source reconstructs the exact PyPTO, TensorIR, CUDA Tile,
      kernel, plugin, PyTorch, and SGLang revisions from a fresh clone.
- [ ] Bundle identity, patch replay, final trees, submodules, and clean-source
      state pass independently.
- [ ] Public commands contain no workstation-only absolute source, model, DSO,
      driver, or CUDA runtime paths.
- [ ] Third-party licenses and modifications are attributed accurately.
- [ ] Written authorization for public NVIDIA use/distribution has been
      reviewed before any push; non-commercial intent and interview remarks
      are not represented as a license exception.

## Compiler and framework implementation

- [ ] The implementation inventory covers every release PyPTO/TensorIR change
      and maps each feature to source, tests, Qwen call sites, and providers.
- [ ] The blog distinguishes typed ODS/OpBuilder modules from canonical MLIR
      text emitters and does not claim that text emission is fully removed.
- [ ] The Inductor backend uses the pinned public backend entry point and keeps
      upstream PyTorch and SGLang source trees unchanged.
- [ ] A real Qwen packed SwiGLU subgraph executes through
      `torch.compile(backend="pypto")` with row-pitched input strides.
- [ ] The model trace contains nonzero `pypto.generic` artifacts with stable
      `torch-inductor:*` source nodes; configuration flags alone do not count.
- [ ] The README and blog state that SwiGLU pointwise is fused but the complete
      MLP, including its two matmul boundaries, is not one kernel.

## Correctness and coverage

- [ ] The exact-final-revision operator regression covers all 18 handwritten
      graphs and the Inductor SwiGLU at real 0.8B/9B shapes.
- [ ] Correctness reports include dtype, shape, stride, seed, tolerance,
      revision, DSO identity, state mutation, and raw error metrics.
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

## Performance and attribution

- [ ] The performance process accepts no correctness policy or reference
      logits and executes no value, token, or text comparison.
- [ ] Candidate, matched stock, and optimized stock use the same model, prompt,
      19-input/64-output workload, BF16, TP1, and concurrency one.
- [ ] Resolved backends and actual torch.compile/CUDA Graph state are recorded,
      rather than inferred from requested flags.
- [ ] Each lane has four valid fresh starts and forty raw requests in the
      frozen interleaved order; invalid/co-tenant starts are retained as
      rejected evidence rather than selectively sampled.
- [ ] TTFT, TPOT, ITL, E2E latency, output throughput, cold costs, memory,
      clocks, power, temperature, and throttle reasons are reported with raw
      samples and confidence intervals.
- [ ] PyPTO relative performance is computed from median output tokens/second
      separately against matched and optimized stock.
- [ ] Independent CUPTI/NVTX profiles compare semantic operator phases and
      reconcile phase deltas plus host/scheduler/memcpy/graph residuals to the
      total performance gap.
- [ ] Linear and LM-head shapes receive explicit A/B measurement because the
      historical diagnostic profile identified matmul as the dominant cost.

## Documentation and visual evidence

- [ ] `README.md` is Chinese by default and links to `README_EN.md`; English
      links back to Chinese.
- [ ] A fresh reader can follow README commands through bootstrap, build,
      operator correctness, pure performance, and exact-prompt 9B inference.
- [ ] README and blog metrics are rendered from one immutable release summary
      and pass an automated consistency check.
- [ ] The blog uses Chinese numbered first-level and Arabic numbered
      second-level section headings and includes all requested architecture,
      framework, operator, testing, breakdown, limitation, and summary topics.
- [ ] Build, operator correctness, performance, and 9B generation each have a
      genuine Ubuntu/Windows Terminal screenshot or an explicitly documented
      user-capture placeholder and exact command; no screenshot is fabricated.
- [ ] The HTML is a single offline file with inline styling, figures, and image
      data, and renders without external assets.
- [ ] The Markdown blog and HTML remain local and absent from every commit.

## Final repository boundary

- [ ] Build and CPU test parallelism is exactly 24; timing-sensitive GPU runs
      are serial by experimental design.
- [ ] Existing user modifications in `projects/pypto-kernels` are byte-for-byte
      unchanged.
- [ ] Upstream PyTorch, SGLang, and Triton remain clean.
- [ ] No external process was signalled or terminated.
- [ ] Local commits are small and auditable.
- [ ] No public push occurs before the license authorization gate is cleared.
