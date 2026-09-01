# Final Requirement Matrix

Recorded 2026-08-31 for release/qwen35-sm120-v1. The Markdown blog and its
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
| 100% model operator coverage proof | qwen35-{0.8b,9b}-model-gate-current.json; 33,448/33,448 calls | PASS, ModelRunner.forward compute scope only |
| Qwen prompt and stable 64-token inference | current model gate reports; prompt is recorded verbatim | PASS |
| Operator correctness regression | operator-regression-current.json; 8 suites, all_correct=true | PASS |
| Performance-only regression entry points | run_performance_regression.py, run_operator_performance.py, run_inductor_ablation.py | PASS |
| Eager versus official NV Inductor ablation | qwen35-9b-inductor-ablation-current.json | PASS at SwiGLU operator scope |
| Eager versus PyPTO Inductor ablation | qwen35-9b-inductor-ablation-current.json | PASS at SwiGLU operator scope; PyPTO is slower |
| Cold-start and compile-time comparison | Inductor ablation JSON plus accepted full-model pair | PASS for resource-qualified pair; full-model compile-trigger wall time includes one complete 31+64 request and is not compiler-only |
| Full-model eager control | qwen35-9b-eager-compile-ablation-current.json | PASS as non-causal control; CompilerInterface was not invoked |
| End-to-end PyPTO versus matched stock | qwen35-9b-release-results-current.json | PASS: 4+4 fresh starts in the 12-start matrix; PyPTO is 15.6208% of matched (CI 15.5862%-15.7022%) |
| End-to-end PyPTO versus optimized stock | qwen35-9b-release-results-current.json | PASS: 4+4 fresh starts; PyPTO is 18.7143% of optimized (CI 18.6881%-18.7533%); optimized uses the user-authorized completion-only GPU-memory policy |
| Operator performance breakdown | state/evidence/qwen35-9b-operator-performance-breakdown-current.json, 7 aligned cases | PASS: checked-in byte-identical aggregation, 4+4 fresh starts, source/package identity and bootstrap CIs |
| Full-model CUPTI/NVTX phase breakdown | qwen35-9b-release-results-current.json | PASS as hybrid three-lane evidence: PyPTO/optimized strict compiled, matched descriptive noncompiled; 3 starts/lane, 5 requests/start, 64 windows/request, optimized 315/315 graph launches/start |
| Descriptive stock CUDA phase breakdown | qwen35-9b-descriptive-stock-profile-breakdown-current.json; 3+3 fresh starts, raw CUPTI trace hashes | PASS with explicit non-causal boundary: matched requested compile but CompilerInterface was not invoked; phase zeros/unattributed activity are not execution claims |
| Linked article URL and unchanged source import | demo/pypto-lib/SOURCE_MANIFEST.json | PASS: 151 files, 66 entrypoints, byte hashes |
| Article-demo compatibility policy and source-line classification | state/evidence/article-demo-compatibility-policy-current.json; manifest SHA | PASS: 66/66 entries classified; 17 hardware API skips, 8 drafts, 40 CUDA-reference computations, 1 strict PyPTO computation, zero unmapped |
| Run computational linked demos with precision on NVIDIA | state/evidence/article-demo-matrix-nvidia-current.json; run_article_demo_nvidia.py | PASS for all 41 computational entries; strict PyPTO artifact 1/1, independent CUDA references 40/40; hardware APIs are explicitly skipped |
| Run every linked article demo unchanged with precision | article demo matrix reports | PASS within user-authorized scope: imported source remains byte-identical, every computational entry passes through an external adapter, and all 17 hardware-facing entries retain source-evidenced skips |
| Typical demo README and result screenshot | demo/README.md, demo/README_EN.md, screenshot manifest | PASS: strict hello result/golden evidence and real Ubuntu/PowerShell-purple `PrintWindow` capture are hash-bound |
| Four requested terminal roles plus typical demo | current five-role screenshot manifest | PASS: build/operator/model are hash-validated accepted-run evidence replays; performance is current operator evidence; article-demo GUI role is a completed live run |
| GPT-Image-2 visuals for ablations/breakdowns | gpt-image2-ablation-prompts-20260829.json | OPEN: OPENAI_API_KEY absent; no model substitution |
| Bilingual README parity and language switch | README.md, README_EN.md | PASS by audit; both use current numbers |
| Single-file offline HTML | reports/local-blog/*.html, 5 embedded data images | PASS; browser viewport automation unavailable in this environment |
| Plan persistence | PLAN.md revision-75 checkpoint and memory ad-hoc note | PASS |
| README/reproduction commit boundary | current release branch and staged-diff audit | PASS; blog/HTML/probes excluded |

Open items are intentionally not converted into PASS by wording, placeholders,
or historical checkpoints.
