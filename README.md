# PyPTO LOVE TensorIR：在 RTX 5090 上运行 Qwen3.5

[简体中文](README.md) | [English](README_EN.md)

> [!IMPORTANT]
> 本仓库是个人、非营利的编译器研究项目；将 PyPTO 运行在 NVIDIA 卡上在表面上
> 与 CANN Open Software License Agreement Version 2.0 对非华为 AI 处理器用途
> 和分发的限制存在冲突。非营利目的、公开访谈中的跨平台愿景或删除承诺都不构成许可证豁免。
> 本项目不代表华为、NVIDIA、PyTorch 或 SGLang；权利方如认为内容不当，请通过
> 仓库 Issues 联系作者删除。完整边界见 LEGAL_NOTICE.md。
>
> 访谈出处（约 2 小时 34 分钟）：
> https://www.bilibili.com/video/BV1nB3u6tERu/?vd_source=f2f41aa7b5e3cc8e0a23942779ccea11

本项目的两个核心 feature：

1. PyPTO DSL/HIR 通过 typed TensorIR ODS/OpBuilder 降低到 NVIDIA TensorIR，
   再经 CUDA Tile 与 tileiras 生成 SM120 Cubin；
2. 标准 torch.compile(backend="pypto") TorchInductor 后端，把可融合的
   pointwise/reduction 子图自动生成为 PyPTO DSL。

主执行链：

~~~text
SGLang -> pypto-kernels + TorchDynamo/Inductor -> PyPTO HIR
       -> typed TensorIR ModuleOp -> CUDA Tile -> tileiras -> sm_120a Cubin
       -> PyPTO Artifact/NvidiaExecutable -> caller-owned CUDA stream
~~~

![PyPTO NVIDIA SM120 架构](docs/assets/pypto-nvidia-architecture.svg)

## 当前实测结论

平台：RTX 5090 Laptop（SM120，24 GiB），Qwen3.5-9B text-only，chat-template
non-thinking 输入 31 token，greedy 输出 64 token，BF16、TP1、并发 1。

| 项目 | 实测结果 |
|---|---|
| 9B correctness/coverage | 当前 wheel 的 3 个 fresh start 全部完成；每个 10/10 请求和 strict teacher-forced trace 通过 |
| model-forward coverage | 已通过 trace：33,448 compute calls = 31,400 手写 PyPTO + 2,048 Inductor PyPTO，unknown/fallback 0，100% |
| 0.8B current wheel | stock reference + 3 个 candidate fresh start 全部完成；每个 trace 22,108/22,108 covered calls = 20,572 手写 + 1,536 Inductor |
| PyPTO output throughput | **正式资源合规 pair：2.4124 tok/s**（四次 fresh-start 中位数） |
| matched SGLang | **正式资源合规 pair：15.3708 tok/s**（同 workload/control） |
| PyPTO / matched | **15.695%**，95% bootstrap CI **[15.634%, 15.753%]**；相对 matched 为 **-84.305%** |
| Pair acceptance | `accepted=true`；8 个 start 全部高于 4 GiB GPU-free/12 GiB host-free floor，控制字段零 mismatch |
| Cold/compile-trigger | PyPTO `16164.45/29252.20 ms`；matched `15316.70/23815.46 ms`；前者比后者高 **22.83%**（含一次完整 31+64 请求，不是 compiler-only） |
| 历史失效 pair | 旧 `2.6671/15.4100 tok/s`、17.31% 已移至 `qwen35-9b-performance-pair-invalidated-20260830.json`，只作诊断对照 |
| optimized SGLang | 未报告；本轮 CUDA-graph capture 到 4000 MiB free 时被 4096 MiB controller floor 停止（run `pypto-gpu-bounded-20260831T034327Z-2531381-964f87`） |

optimized 的未晋级探测也保留在同一 diagnostic sidecar：`0.68` 因 GDN
state cache 无法容纳一个请求而失败；`0.685` 在 graph capture 期间降到
2454 MiB free；3/4 GiB CPU offload 分别在 4088/3784 MiB 被门禁停止。它们都
没有性能报告，正式 2 GiB offload、0.69 配置不变。官方 memory-saver 依赖未
进入锁定环境，post-capture KV sizing 又与当前 disabled prefill graph 合同不兼容，
因此未用临时装包或改 serving mode 绕过门禁。

accepted trace 的 artifact union 与 model-forward compute intersection：

| model | total calls | handwritten | Inductor | unknown/fallback | coverage |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 22,108 | 20,572 | 1,536 | 0 | 100% |
| Qwen3.5-9B | 33,448 | 31,400 | 2,048 | 0 | 100% |

整模正式 pair 的四次 fresh-start 汇总（每次先取十个请求的 p50）：

| lane | E2E | TTFT | TPOT | output tok/s | cold engine | 首次编译触发请求 |
|---|---:|---:|---:|---:|---:|---:|
| PyPTO | 26529.17 ms | 3159.24 ms | 370.86 ms | 2.4124 | 16164.45 ms | 29252.20 ms |
| matched | 4163.75 ms | 70.10 ms | 64.97 ms | 15.3708 | 15316.70 ms | 23815.46 ms |

首次编译触发请求包含编译和一个完整 31+64 请求，不是 compiler-only 时间。
正式 pair 的 PyPTO 最低 GPU free 为 `5,123,887,104 B`，最低主机可用内存为
`47,233,052 KiB`；8 个 start 均通过 4 GiB/12 GiB 门禁且未观测 thermal throttle。
两条 lane 统一为 `cpu_offload_gb=2, mem_fraction_static=0.78`，控制字段零 mismatch。
旧的失效 pair 仍保留为
`state/evidence/qwen35-9b-performance-pair-invalidated-20260830.json`，不能与当前
正式结果混用。资格与 pair 绑定见
`state/evidence/matched-performance-qualification-current.json`。
该旧报告的状态仍明确为 `invalidated-resource-and-control`；当前正式报告状态为
`complete` 且 `acceptance.accepted=true`。
完整 JSON 为
state/evidence/qwen35-9b-performance-pair-current.json；模型 gate
sidecar 为 state/evidence/qwen35-0.8b-model-gate-current.json 和
state/evidence/qwen35-9b-model-gate-current.json；生成
kernel 的 DSL、wrapper 和 artifact hash 见
state/evidence/qwen35-9b-inductor-source-current.json。
整模 eager control 的非因果对照见
state/evidence/qwen35-9b-eager-compile-ablation-current.json。

整模 eager control 也已单独运行一次：关闭 torch.compile、保持 matched provider，
output 15.3418 tok/s、E2E 4171.67 ms；正式 pair 中 matched compile-request
的中位数为 15.3708 tok/s、4163.75 ms。但 matched 配置禁用 CUDA Graph 后
CompilerInterface/Inductor 实际未调用，因此这不是有效的整模 compile 因果加速率；
machine-readable 记录为 state/evidence/qwen35-9b-eager-compile-ablation-current.json。

## 为什么 TensorIR 适合作为 PyPTO backend

[PyPTO](https://github.com/hw-native-sys/pypto) 负责用户 DSL、HIR、静态
specialization、artifact/cache/runtime；[TensorIR](https://github.com/NVIDIA/tensor-ir)
是 tensor-level、tile-aware 的 MLIR compiler frontend，负责 layout propagation、
tiling、graph splitting 和 TensorIR-to-CUDA-Tile lowering。CUDA Tile 是下游
GPU IR/工具链目标，TensorIR 不是 CUDA Tile 本身，也不是 cuTile runtime。

PyPTO tile shape 能继续 lower 的条件是：shape、dtype、element stride、layout、
迭代空间和 mutation/alias ABI 都静态且通过 TensorIR verifier。桥接构造
TypedTensorIrModuleSpec，再用 mlir::OpBuilder 和 ODS *Op::create 构造
ModuleOp；文本 print 只用于规范序列化和诊断。动态/重叠 stride、未支持的
contraction、错误 result anchor 或不合法对齐会 fail closed。

| PyPTO | TensorIR | 条件 |
|---|---|---|
| pl.range / tile | iteration space / tile sizes | 静态有界 |
| pl.load/store/view | typed slice/reshape/transpose/gather/scatter | layout/访问可验证 |
| pl.InOut | read-write operand + mutation metadata | alias 与结果一致 |
| matmul/reduce/broadcast | ODS ops | rank、dtype、contraction 受支持 |

## 实现

### PyPTO -> TensorIR

NVIDIA TargetInfo/SM120 探测；不可变 CompileRequest、CanonicalSchedule、
KernelBuildSpec；shape/dtype/stride/mutation/toolchain cache identity；typed
GraphOp/stride attrs/results/verifier；pointwise、row reduction、BF16/FP32
matmul、norm、gather/scatter、RoPE、dense/paged attention、KV write、
causal Conv1D、GDN projection/recurrent；Cubin/ABI/grid/workspace 校验；
NvidiaExecutable 的 process/device/CUcontext/current-stream 生命周期。

核心文件：.sources/pypto/src/codegen/nvidia/tensor_ir_codegen.cpp、
.sources/pypto/src/codegen/nvidia/typed_tensor_ir_module_spec.h、
.sources/pypto/src/compiler/nvidia_typed_tensor_ir_builder.cpp。
canonical NVIDIA construction 禁止 source-string 拼接，lint 与 negative
geometry verifier 已加入回归。

### TorchInductor -> PyPTO

插件注册 torch_dynamo_backends=pypto 并调用完整 compile_fx；在 context-local
范围替换 CUDA scheduler/wrapper，离开 context 恢复官方 backend；strict 模式
关闭 fallback；录制 node body 的 load/op/store，保留真实 row-pitched stride；
生成 @pl.jit graph 和 pypto_launch(current_stream) wrapper；artifact/prewarm
cache、稳定 source hash 和 CUPTI external correlation 均有记录。

Qwen MLP 没有整体融合：

~~~text
gate/up handwritten linear -> one Inductor-generated PyPTO SwiGLU
                              -> down handwritten linear
~~~

自动融合的只有 packed gate/up 后的 FP32 cast、sigmoid、两次乘法和 BF16 cast。

## 手写算子库

packages/pypto-kernels 是独立包，15 个 Python 模块、18 个 @pl.jit graph：
dense/masked/paged attention、paged gather/KV write/prefill、causal Conv1D、
recurrent GDN、GDN projection、Q/K RMSNorm + partial RoPE + gate、embedding/
integer gather、BF16 linear、BF16-rounded FP32 LM head、三类 RMSNorm、RoPE、
sigmoid-mul、SiLU-mul。

代表性复杂实现是
[gdn_recurrent_kernel](packages/pypto-kernels/src/pypto_kernels/gdn.py)，
组合 pl.InOut、静态循环、row reduction/broadcast、FP32 matmul 和
outer-product state update。长 prefill 按 token 有序推进，不伪称 mega-kernel。

## 消融与 breakdown

固定 Qwen3.5-9B SwiGLU，20 warmup、100 CUDA-event calls：

| phase/mode | warm ms | first call ms | events | 相对 eager |
|---|---:|---:|---:|---:|
| prefill 19x24576 eager | 0.042966 | 35.25 | 6 | 0 |
| prefill official NV Inductor | 0.032915 | 1098.33 | 1 | +30.54% |
| prefill PyPTO Inductor | 0.210618 | 1696.24 | 1 | -79.60% |
| decode 1x24576 eager | 0.040543 | 32.81 | 6 | 0 |
| decode official NV Inductor | 0.033808 | 1045.90 | 1 | +19.92% |
| decode PyPTO Inductor | 0.212401 | 1733.25 | 1 | -80.91% |

compiled mode 都把 6 events 降到 1，launch reduction 83.33%。PyPTO 首调用比
官方 NV backend 长 54.44%（prefill）/65.72%（decode）。

对齐的 8-start operator A/B（每 lane 4 start）：

| case | PyPTO latency / stock |
|---|---:|
| SwiGLU decode / prefill | 23.52x / 23.47x |
| gate-up linear decode / prefill | 10.89x / 180.02x |
| down linear decode / prefill | 37.11x / 189.69x |
| FP32 LM head | 6.10x |

这些是逻辑功能对齐的 microbenchmark；它们解释了当前整模慢的主要来源，
不能被外推成 PyPTO 加速。

### 全模型 CUPTI 描述性 breakdown

严格三 lane（PyPTO、matched、optimized）的编译执行 profile 仍是独立门禁；
为了先给出可核查的阶段证据，我们另外采集了 3 个 PyPTO 和 3 个
`sglang-matched` fresh start，每个 start 5 个请求、每个请求 64 个
`ModelRunner.forward` 窗口。matched 的配置保留了 `enable_torch_compile=true`，
但关闭 CUDA Graph 后本 pinned SGLang 没有调用 CompilerInterface，因此下面
只表示 stock CUDA kernel 的描述性 CUPTI 活动，不是 Inductor 编译或加速证明。

| forward compute（每请求 GPU activity duration） | PyPTO p50 | stock matched p50 | 差值 |
|---|---:|---:|---:|
| 全部 compute | 22.318812 ms | 1.285792 ms | +21.033020 ms |
| `unattributed_compute` | 20.184082 ms | 0.435080 ms | +19.749002 ms |
| `attention_core_gate` | 1.154360 ms | 0.003228 ms | +1.151132 ms |
| `lm_head` | 0.931866 ms | 0 ms（未形成同名关联） | +0.931866 ms |

`unattributed_compute` 和表中 stock 的 0 并不表示对应算子没有执行；它们
表示当前 external-correlation/module-hook 规则没有把活动映射到同一逻辑名。
阶段差值之和为 `21.031480 ms`，与总差值的 `1.539578 ms` 中位数残差来自
分别取阶段/总量中位数。完整 phase、CI、每个 raw CUPTI trace 的 SHA 和资源
边界见 [`qwen35-9b-descriptive-stock-profile-breakdown-current.json`](state/evidence/qwen35-9b-descriptive-stock-profile-breakdown-current.json)。
该 sidecar 可由以下显式模式重建（不会改变严格三 lane matrix）：

```bash
envs/pypto-release/bin/python tools/profile_qwen35.py collect \
  --lane sglang-matched --model-path models/Qwen3.5-9B \
  --optimized-memory-mode matched --allow-noncompiled-matched
envs/pypto-release/bin/python tools/profile_qwen35.py reconcile \
  --allow-noncompiled-matched \
  --profile pypto=runs/<pypto-start>/qwen35-9b-profile-pypto.json \
  --profile sglang-matched=runs/<matched-start>/qwen35-9b-profile-sglang-matched.json \
  --output runs/<breakdown>/descriptive-reconciliation.json
```

用户要求所有消融/breakdown 图用 GPT-Image-2。当前环境没有 OPENAI_API_KEY，
所以保持 PENDING_GPT_IMAGE2，不用其他模型替代；提示词和 provenance 见
state/evidence/gpt-image2-ablation-prompts-20260829.json。

## 目录结构

~~~text
packages/pypto-kernels/             独立手写 PyPTO 算子库
packages/pypto-framework-plugins/   SGLang 与 TorchInductor 插件
.sources/pypto/                     锁定的 PyPTO/TensorIR 源码工作树
vendor/                             bundle、patch series、source-lock
benchmarks/release/                 workload、correctness、performance、profile 合同
tools/                              构建、受控运行、汇总与审计入口
demo/pypto-lib/                     文章 demo 的 byte-for-byte 导入
state/evidence/                     小型、可提交的结果 sidecar
runs/                               本机 raw report（不提交）
~~~

## 构建与测试

环境：Ubuntu 26.04/WSL2、CPython 3.14.6、PyTorch 2.13.0+cu130、CUDA
13.3.73、SGLang 0.5.18、CMake 3.31.10、Ninja 1.13。CPU 使用 24 jobs，
GPU 测试串行。

~~~bash
python3 tools/verify_source_release.py --replay-patches
python3 tools/bootstrap_release.py --jobs 24
python3 tools/bootstrap_release_environment.py
envs/pypto-release/bin/python tools/build_release.py --stage all --jobs 24
envs/pypto-release/bin/python tools/download_release_models.py --model all
envs/pypto-release/bin/python tools/run_operator_regression.py --stage all
~~~

源码身份：PyPTO c27629e993a52b47d41fb898c749279dce44221b（300 commits）；
TensorIR db41d0733eb73971ee03a74faca81d1af6e6aef7（89 commits）；CUDA Tile
af2417041cc939b87ef56d92cfdcf61737c5457e；LLVM
57109befac92811d2253109242ca6fa69c961fb2；SGLang
71de97b264b04dcd514cf904003028aefe9775c8。

模型 correctness：

~~~bash
envs/pypto-release/bin/python tools/run_transformers_semantic_oracle.py \
  --model-path models/Qwen3.5-9B \
  --output runs/semantic-oracle-qwen35-9b-chat-nonthinking.json
envs/pypto-release/bin/python tools/run_model_correctness.py all \
  --model-path models/Qwen3.5-9B \
  --semantic-oracle runs/semantic-oracle-qwen35-9b-chat-nonthinking.json
~~~

性能-only 脚本不读取 logits/token/text，不做 allclose/cosine/top-k/tolerance：

~~~bash
envs/pypto-release/bin/python tools/run_performance_regression.py \
  --pair-matrix --model-path models/Qwen3.5-9B \
  --optimized-memory-mode matched
# pair 通过后，再运行包含 optimized stock 的完整三 lane 矩阵
envs/pypto-release/bin/python tools/run_performance_regression.py \
  --matrix --model-path models/Qwen3.5-9B \
  --optimized-memory-mode matched
envs/pypto-release/bin/python tools/run_qwen35_eager_control.py \
  --model-path models/Qwen3.5-9B
envs/pypto-release/bin/python tools/profile_qwen35.py matrix \
  --model-path models/Qwen3.5-9B \
  --optimized-memory-mode matched \
  --performance-matrix runs/release-performance-matrix-<id>/summary.json
envs/pypto-release/bin/python tools/run_operator_performance.py \
  --matrix --model-path models/Qwen3.5-9B
envs/pypto-release/bin/python tools/run_inductor_ablation.py \
  --mode pypto --phase prefill --output runs/ablation-prefill-pypto.json
envs/pypto-release/bin/python tools/summarize_inductor_ablation.py \
  --prefill-eager runs/ablation-current-prefill-eager.json \
  --prefill-inductor_nv runs/ablation-current-prefill-inductor-nv.json \
  --prefill-pypto runs/ablation-current-prefill-pypto.json \
  --decode-eager runs/ablation-current-decode-eager.json \
  --decode-inductor_nv runs/ablation-current-decode-inductor-nv.json \
  --decode-pypto runs/ablation-current-decode-pypto.json \
  --output state/evidence/qwen35-9b-inductor-ablation-current.json
~~~

## 文章 Demo 与截图

动机是让没有昇腾设备的用户在 NVIDIA 上学习 PyPTO DSL。原文链接：
[让 Python 写 NPU 算子所写即所得！华为昇腾开源 PyPTO-Lib，实现 Qwen3-14B 与 DeepSeek V4-Flash 全部算子！](https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ)

文章时点的 pypto-lib commit
6c292d30ccc787ee4e1fe61541fd3faec0dafa65 已 byte-for-byte 导入
`demo/pypto-lib/`。`SOURCE_MANIFEST.json` 锁定 151 文件、66 entrypoint；外部
策略 [`article-demo-compatibility-policy-current.json`](state/evidence/article-demo-compatibility-policy-current.json)
为每个入口记录硬件 API 证据和 NVIDIA 兼容模式。

先生成策略，再运行 NVIDIA 计算类矩阵：

~~~bash
python3 tools/classify_article_demos.py
envs/pypto-release/bin/python tools/run_article_demo_matrix.py \
  --backend nvidia --mode run --device 0 \
  --output state/evidence/article-demo-matrix-nvidia-current.json
~~~

当前 RTX 5090 真机结果：全部 41 个计算入口通过（40 个独立 CUDA 数值参考，
`hello_world.py` 另有严格 PyPTO -> TensorIR -> CUDA Tile artifact）；17 个通信、
CCE、NPU/ACL 或 Ascend runtime 硬件 API 入口按支持边界跳过，8 个 draft 仅保留
provenance。`computational_unmapped_count=0`，矩阵的 `compatibility_status=complete`、
`hardware_api_evidence`、manifest 前后 SHA 是发布门禁。

40 个 CUDA reference 覆盖 9 个教学例子、2 个 Qwen3 sampling，以及 29 个
DeepSeek V4 compressor/indexer/sparse-attention/HC/MoE/decode/prefill 计算入口；
每份 child report 都记录逐输出容差和误差计数。

典型严格计算入口：

~~~bash
envs/pypto-release/bin/python tools/run_article_demo_nvidia.py \
  --demo examples/beginner/hello_world.py --device 0 \
  --run-id article-demo-nvidia-hello-screenshot \
  --output state/evidence/article-demos-nvidia/011-hello_world-screenshot.json
~~~

终端应显示 `strict-pypto-nvidia`、`golden_pass=True`、artifact 名称和
`fallback_used=False`；当前真机 `y` 的 `max_abs_diff=0.0`。报告绑定导入源/策略 SHA、artifact/cubin SHA 和
128-element tile，原始文件不被改写。其余 40 个计算入口的报告显式写入
`strict_compiler_evidence=false`，它们是独立 CUDA Torch 数值参考，用于在
NVIDIA 上学习计算语义，不能替代严格 PyPTO 编译证据。

如需复现文章原始 CLI/help，或在具备授权 Ascend runtime 的机器上运行未改写
源码，仍使用：

~~~bash
envs/pypto-release/bin/python tools/run_article_demo_matrix.py \
  --backend ascend --mode help --output runs/article-demo-matrix-help.json
envs/pypto-release/bin/python tools/run_article_demo.py \
  --demo examples/beginner/hello_world.py --platform a2a3sim \
  --output runs/article-demo-hello-world.json
~~~

`--backend ascend` 的设备阶段在本机受 `simpler_setup`/`KernelType.MIX` 阻断，
报告中的 blocker 不是精度通过。硬件通信、CCE、NPU/ACL 和 simpler runtime
入口在 NVIDIA 矩阵中明确跳过，不伪造结果。典型 `hello_world.py` 已在真实
Windows Terminal 的 Ubuntu/PowerShell-purple 窗口中运行，`exit_code=0`，并显示
`strict-pypto-nvidia`、`golden_pass=True`、`fallback_used=False` 和
`max_abs_diff=0.0`。截图及窗口 metadata 由
[`article-demo-screenshot-manifest-current.json`](state/evidence/article-demo-screenshot-manifest-current.json)
绑定，PNG 为 1549×925；失败的黑屏尝试未被采用。

Windows Terminal GUI capture 已通过非黑像素 smoke；performance 角色已用当前
immutable ablation JSON 重新捕获并绑定窗口尺寸、命令、时间和 PNG SHA。
model 角色严格校验并回放三次已接受的 9B run，截图明确标注不是 live rerun；
build 覆盖 wheel build、install/pip-check 和 CTest 13/13 三阶段；operator
也以同样方式校验并回放已接受的原始证据；article demo 的计算矩阵已完成，
GUI 角色也已完成。五个已完成角色均绑定 command source、数值/运行 evidence、capture
metadata 和 PNG SHA。PowerShell 模板：
tools/windows/capture_terminal.ps1。

![wheel build、install 与 CTest 证据回放](docs/assets/screenshots/build-ctest.png)

![8/8 算子回归证据回放](docs/assets/screenshots/operator-correctness.png)

![Qwen3.5-9B 已接受推理证据回放](docs/assets/screenshots/model-inference.png)

![9B SwiGLU 算子级消融](docs/assets/screenshots/performance-ablation.png)

![典型 hello_world.py 严格 NVIDIA demo](docs/assets/screenshots/article-demo-typical.png)

模型固定 prompt：

为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？

## 限制与许可证

只验证 SM120/RTX 5090 Laptop；静态 shape/stride specialization；完整 MLP
不跨 matmul 融合；GDN 长 prefill 为有序 tokenwise launch；PyPTO matmul/
LM-head/launch overhead 尚未优化；optimized lane、严格三 lane 全模型
CUPTI/NVTX profile 和 GPT-Image-2 图像仍是门禁。资源合规的
PyPTO/matched 四-start pair 已接受，
其非编译 matched 的描述性阶段 profile 见对应 sidecar。
TensorIR 上游为 early release。

PyPTO 使用 CANN Open Software License Agreement 2.0；TensorIR 使用 Apache
2.0 with LLVM Exceptions；framework plugin 使用 Apache 2.0。授权边界见
packages/pypto-kernels/LICENSE_STATUS.md。
