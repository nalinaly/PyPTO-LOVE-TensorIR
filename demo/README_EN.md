# Demo Directory

[简体中文](README.md)

`pypto-lib/` is a byte-for-byte import of the article-time snapshot from
[让 Python 写 NPU 算子所写即所得！华为昇腾开源 PyPTO-Lib，实现 Qwen3-14B 与 DeepSeek V4-Flash 全部算子！](https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ).
Do not edit imported files below `demo/pypto-lib/`. `SOURCE_MANIFEST.json`
locks 151 files and 66 entry points; a separate policy records whether each
entry uses a hardware API.

## NVIDIA Computational Path

Generate and verify the external policy, then run the complete matrix:

~~~bash
python3 tools/classify_article_demos.py
envs/pypto-release/bin/python tools/run_article_demo_matrix.py \
  --backend nvidia --mode run --device 0 \
  --output state/evidence/article-demo-matrix-nvidia-current.json
~~~

The current RTX 5090 result passes all 41 computational entries: 40 independent
CUDA numerical references and one strict PyPTO -> TensorIR -> CUDA Tile artifact
for `hello_world.py`. The matrix retains 17 hardware-API skips and eight
provenance-only drafts; no computational entry remains unmapped. CUDA reference
entries are not counted as strict NVIDIA compiler passes.
`compatibility_status=complete` and each
`hardware_api_evidence` field are the release decision boundary.

Typical strict entry:

~~~bash
envs/pypto-release/bin/python tools/run_article_demo_nvidia.py \
  --demo examples/beginner/hello_world.py --device 0 \
  --run-id article-demo-nvidia-hello-screenshot \
  --output state/evidence/article-demos-nvidia/011-hello_world-screenshot.json
~~~

The terminal prints `strict-pypto-nvidia`, `golden_pass=True`, the artifact name,
and `fallback_used=False`; the current `y` result has `max_abs_diff=0.0`. The report binds imported-source and policy hashes,
artifact/cubin hashes, and the 128-element tile; the upstream source remains
unchanged.

The other 40 computational entries use independent CUDA Torch references for their
mathematical results. Their reports explicitly set
`strict_compiler_evidence=false`; they do not replace strict PyPTO compiler
evidence. This split lets readers study computational semantics on NVIDIA
without presenting Ascend orchestration or CPU/golden-only output as backend
support.

## Original Ascend Commands and Audit

To reproduce the article's original CLI/help or run unchanged source on an
authorized Ascend runtime, use the original launcher:

~~~bash
envs/pypto-release/bin/python tools/run_article_demo_matrix.py \
  --backend ascend --mode help --output runs/article-demo-matrix-help.json
envs/pypto-release/bin/python tools/run_article_demo.py \
  --demo examples/beginner/hello_world.py --platform a2a3sim \
  --output runs/article-demo-hello-world.json
~~~

`--backend ascend` executes the article command unchanged. This checkout lacks
the Ascend `simpler_setup` extension, so device failures are recorded as
blockers. Distributed hardware, CCE kernels, NPU/ACL, and simpler-runtime
entries are explicitly skipped in the NVIDIA matrix; no precision result is
fabricated.

## Typical Screenshot

The strict `hello_world` run was captured in a real Windows Terminal
Ubuntu/PowerShell-purple window. The same command, run ID, report hash, and
visible golden result are bound by
[`article-demo-screenshot-manifest-current.json`](../state/evidence/article-demo-screenshot-manifest-current.json).
The window is 1549×925 with 5184/5335 `PrintWindow` visible samples; the command
exited 0 and `y.max_abs_diff=0.0`. A prior black-frame attempt was not accepted.

![hello_world.py strict NVIDIA result](../docs/assets/screenshots/article-demo-typical.png)
