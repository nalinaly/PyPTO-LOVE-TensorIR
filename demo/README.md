# Demo 目录

pypto-lib/ 是微信文章
[让 Python 写 NPU 算子所写即所得！华为昇腾开源 PyPTO-Lib，实现 Qwen3-14B 与 DeepSeek V4-Flash 全部算子！](https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ)
在文章时点的 byte-for-byte 源码快照。不要编辑 demo/pypto-lib/ 内的导入文件。

典型入口：

~~~bash
envs/pypto-release/bin/python tools/run_article_demo.py \
  --demo examples/beginner/hello_world.py --platform a2a3sim \
  --output runs/article-demo-hello-world.json
~~~

launcher 会先验证 SOURCE_MANIFEST.json，再记录 compile、input、golden、
runtime 四个阶段和 stdout/stderr SHA。当前本机完成 CLI/help audit，但设备
runtime 仍缺少 Ascend simpler_setup 扩展；报告中的 blocker 不是精度通过。

典型 Ubuntu/PowerShell 紫色终端截图槽位：

PENDING_SCREENSHOT: docs/assets/screenshots/article-demo-typical.png

替换截图时必须绑定同一次命令、run ID、报告 SHA 和可见的 golden 结果。
