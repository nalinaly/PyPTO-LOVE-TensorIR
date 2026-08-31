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
| Cold-start and compile-time comparison | Inductor ablation JSON plus invalidated full-model pair | PASS at operator scope; full-model cold timings are diagnostic until the resource-compliant rerun |
| Full-model eager control | qwen35-9b-eager-compile-ablation-current.json | PASS as non-causal control; CompilerInterface was not invoked |
| End-to-end PyPTO versus matched stock | qwen35-9b-performance-pair-current.json; matched-performance-qualification-current.json | OPEN: old diagnostic pair violated resource/control gates; first corrected qualification was interrupted by a newly started protected host-heavy workload |
| End-to-end PyPTO versus optimized stock | optimized lane run attempts; optimized-lane-diagnostic-current.json; formal run pypto-gpu-bounded-20260830T161746Z-2433114-b55eb2 | OPEN: historical attempts failed resource/telemetry qualification; NVML fallback is fixed, but a protected-heavy-free window is unavailable; no percentage promoted |
| Operator performance breakdown | release-operator-ab aggregation.json, 7 aligned cases | PASS |
| Full-model CUPTI/NVTX phase breakdown | profile attempt 20260830T152705Z; corrected performance-only profile envelope | OPEN: prior zero-offload attempt failed KV-cache qualification; corrected profile awaits an accepted pair and protected-heavy-free window |
| Linked article URL and unchanged source import | demo/pypto-lib/SOURCE_MANIFEST.json | PASS: 151 files, 66 entrypoints, byte hashes |
| Article-demo compatibility policy and source-line classification | state/evidence/article-demo-compatibility-policy-current.json; manifest SHA | PASS: 66/66 entries classified; 17 hardware API skips, 8 drafts, 31 unmapped model computations, 10 teaching computations covered |
| Run computational linked demos with precision on NVIDIA | state/evidence/article-demo-matrix-nvidia-current.json; run_article_demo_nvidia.py | PASS for 10/10 teaching computations; strict PyPTO artifact 1/1, independent CUDA references 9/9; hardware APIs are explicitly skipped |
| Run every linked article demo unchanged with precision | article demo matrix reports | PARTIAL by user-authorized hardware boundary: unchanged source/help audit is complete; hardware-facing entries are skipped and 31 model computations await bounded adapters |
| Typical demo README and result screenshot | demo/README.md, demo/README_EN.md, screenshot manifest | PARTIAL: strict hello result and golden evidence pass; GUI capture still requires a real Ubuntu/PowerShell window |
| Four requested terminal roles plus typical demo | current five-role screenshot manifest | PARTIAL: build/operator/model are hash-validated accepted-run evidence replays; performance is current operator evidence; article-demo GUI role remains open |
| GPT-Image-2 visuals for ablations/breakdowns | gpt-image2-ablation-prompts-20260829.json | OPEN: OPENAI_API_KEY absent; no model substitution |
| Bilingual README parity and language switch | README.md, README_EN.md | PASS by audit; both use current numbers |
| Single-file offline HTML | reports/local-blog/*.html, 5 embedded data images | PASS; browser viewport automation unavailable in this environment |
| Plan persistence | PLAN.md revision-67 checkpoint and memory ad-hoc note | PASS |
| README/reproduction commit boundary | commits c336cfa/cfeb10c; origin/release/qwen35-sm120-v1 | PASS; blog/HTML/probes excluded |

Open items are intentionally not converted into PASS by wording, placeholders,
or historical checkpoints.
