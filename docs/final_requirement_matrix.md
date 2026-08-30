# Final Requirement Matrix

Recorded 2026-08-30 for release/qwen35-sm120-v1. The Markdown blog and its
HTML are local deliverables; the matrix itself is a publication audit.

| Requirement | Location/evidence | Status |
|---|---|---|
| Chinese technical report with numbered headings | reports/local-blog/pypto-tensorir-rtx5090-qwen35-9b.md, headings 一 through 十 | PASS |
| CANN boundary, non-commercial research, interview at about 02:34, takedown contact | blog NOTE, README NOTE, LEGAL_NOTICE.md | PASS, not a license exception |
| PyPTO and TensorIR background and public URLs | blog 二.1/二.2; README links | PASS |
| Why TensorIR is a suitable PyPTO backend | blog 二.3/二.4; README independent section | PASS |
| Tile-shape lowerability conditions and fail-closed cases | TypedTensorIrModuleSpec/OpBuilder source and blog table | PASS |
| PyPTO-to-TensorIR feature inventory | blog 五.1; typed builder and source replay | PASS |
| TorchInductor backend feature inventory | blog 五.2/五.3; plugin tests and current source evidence | PASS |
| Handwritten operator library inventory and one complex walkthrough | blog 六; pypto-kernels source | PASS |
| Exact generated fused kernel and current-stream wrapper | state/evidence/qwen35-9b-inductor-source-current.json | PASS |
| Complete MLP fusion claim checked | blog states gate/up and down matmuls remain separate | PASS |
| 100% model operator coverage proof | qwen35-{0.8b,9b}-model-gate-current.json; 22,108/33,448 calls | PASS, ModelRunner.forward compute scope only |
| Qwen prompt and stable 64-token inference | current model gate reports; prompt is recorded verbatim | PASS |
| Operator correctness regression | operator-regression-current.json; 8 suites, all_correct=true | PASS |
| Performance-only regression entry points | run_performance_regression.py, run_operator_performance.py, run_inductor_ablation.py | PASS |
| Eager versus official NV Inductor ablation | qwen35-9b-inductor-ablation-current.json | PASS at SwiGLU operator scope |
| Eager versus PyPTO Inductor ablation | qwen35-9b-inductor-ablation-current.json | PASS at SwiGLU operator scope; PyPTO is slower |
| Cold-start and compile-time comparison | same ablation JSON plus full-model pair summary | PASS with stated timing boundaries |
| Full-model eager control | qwen35-9b-eager-compile-ablation-current.json | PASS as non-causal control; CompilerInterface was not invoked |
| End-to-end PyPTO versus matched stock | qwen35-9b-performance-pair-current.json, 4 starts/lane | PASS; PyPTO 17.31% of matched throughput |
| End-to-end PyPTO versus optimized stock | optimized lane run attempts | OPEN: 4 GiB GPU-free controller floor |
| Operator performance breakdown | release-operator-ab aggregation.json, 7 aligned cases | PASS |
| Full-model CUPTI/NVTX phase breakdown | profile attempt 20260830T152705Z | OPEN: KV-cache memory qualification |
| Linked article URL and unchanged source import | demo/pypto-lib/SOURCE_MANIFEST.json | PASS: 151 files, 66 entrypoints, byte hashes |
| Run every linked article demo unchanged with precision | article demo matrix reports | OPEN: Ascend simpler_setup/KernelType.MIX runtime blockers; 57/57 help passes |
| Typical demo README and result screenshot | demo/README.md, demo/README_EN.md, screenshot manifest | OPEN: GUI capture pending; black frame discarded |
| Four requested terminal roles plus typical demo | current five-role screenshot manifest | OPEN for model/demo Windows GUI captures; no fabricated image |
| GPT-Image-2 visuals for ablations/breakdowns | gpt-image2-ablation-prompts-20260829.json | OPEN: OPENAI_API_KEY absent; no model substitution |
| Bilingual README parity and language switch | README.md, README_EN.md | PASS by audit; both use current numbers |
| Single-file offline HTML | reports/local-blog/*.html, 5 embedded data images | PASS; browser viewport automation unavailable in this environment |
| Plan persistence | PLAN.md revision-58 checkpoint and memory ad-hoc note | PASS |
| README/reproduction commit boundary | commit b4eae99; origin/release/qwen35-sm120-v1 | PASS; blog/HTML/probes excluded |

Open items are intentionally not converted into PASS by wording, placeholders,
or historical checkpoints.
