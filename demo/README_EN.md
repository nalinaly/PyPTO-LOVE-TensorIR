# Demo Directory

pypto-lib/ is a byte-for-byte import of the article-time source snapshot
from [让 Python 写 NPU 算子所写即所得！华为昇腾开源 PyPTO-Lib，实现 Qwen3-14B 与 DeepSeek V4-Flash 全部算子！](https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ).
Do not edit imported files below demo/pypto-lib/.

Typical entry point:

~~~bash
envs/pypto-release/bin/python tools/run_article_demo.py \
  --demo examples/beginner/hello_world.py --platform a2a3sim \
  --output runs/article-demo-hello-world.json
~~~

The launcher verifies SOURCE_MANIFEST.json, then records compile, input,
golden, and runtime stages with stdout/stderr hashes. The local CLI/help audit
passes, but device execution still lacks the Ascend simpler_setup extension;
that blocker is not a precision pass.

Ubuntu/PowerShell purple-terminal capture slot:

PENDING_SCREENSHOT: docs/assets/screenshots/article-demo-typical.png

Any replacement must bind the same command, run ID, report SHA, and visible
golden result.
