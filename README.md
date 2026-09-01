# PyPTO ♥ TensorIR：在 NVIDIA RTX 5090 上用 100% PyPTO kernel 推理 Qwen3.5

[简体中文](README.md) | [English](README_EN.md)

> [!IMPORTANT]
> 本仓库是**个人、非营利**的编译器研究项目。将 PyPTO 运行在 NVIDIA GPU 上，表面上与
> CANN Open Software License Agreement Version 2.0 中限定华为 AI 处理器用途与分发的条款
> 存在冲突；个人研究与非营利目的不构成许可证豁免。本项目的动机与华为首席科学家廖恒博士
> 在张小珺访谈（约 2 小时 34 分钟处）表达的"希望 PyPTO 成为面向所有 AI 芯片的公共前端 DSL"
> 的愿景一致；该公开表态不等于对许可证的修改或对本项目的授权。权利方如认为内容不当，
> 请通过仓库 Issues 联系作者删除。完整分析见 [LEGAL_NOTICE.md](LEGAL_NOTICE.md)。
> 访谈：<https://www.bilibili.com/video/BV1nB3u6tERu/?vd_source=f2f41aa7b5e3cc8e0a23942779ccea11>

## 这是什么

一个把 [PyPTO](https://github.com/hw-native-sys/pypto)（华为 CANN 生态的 kernel DSL 前端，
定位类似昇腾生态的 Triton）带到 NVIDIA GPU 上的编译器实现：**在 RTX 5090（SM120）上，
Qwen3.5-9B 推理 forward 过程的全部 GPU kernel 均由 PyPTO 前端表达并经本仓库的编译管线生成**，
经 CUPTI 逐 kernel 审计为 100% 覆盖（零 fallback），输出与 SGLang 原生实现逐 token 一致。

两个核心 feature：

1. **PyPTO → TensorIR 桥接**：PyPTO HIR（`@pl.jit` tile graph 静态特化）经 typed
   TensorIR（MLIR ODS/OpBuilder）→ CUDA Tile → `tileiras` 生成 `sm_120a` cubin，
   产物为可校验、可缓存、在调用者 CUDA stream 上 launch 的 `NvidiaExecutable`。
   PyPTO fork 300 commits，TensorIR fork 89 commits（补齐 row gather/scatter 布局、
   多输出融合、运行时无关产物契约等）。
2. **TorchInductor 的 PyPTO 后端**：标准 `torch.compile(model, backend="pypto")` 即可使用。
   复用完整官方 `compile_fx`，将可融合的 pointwise/尾轴归约子图自动生成为 PyPTO DSL kernel；
   不支持的场景 fail-closed，绝不静默回退 Triton。在 Qwen3.5-9B 真实形状的 SwiGLU
   算子级消融中，融合能力与官方 Triton 后端等价：**6 个 eager kernel 融合为 1 个
   （launch 次数 −83.33%）**；冷编译一次 **1.70 s**（官方后端 1.10 s，长 54–66%）；
   生成 kernel 热 path 当前比 eager 慢 79.6%（pointwise 调度未调优；官方后端为 +30.5%）。

```text
SGLang（未打补丁）
  └─ pypto-kernels（手写算子库）+ TorchDynamo/TorchInductor（Pypto 后端）
       └─ PyPTO HIR → typed TensorIR ModuleOp → CUDA Tile → tileiras → sm_120a Cubin
            └─ PyPTO Artifact / NvidiaExecutable → caller-owned CUDA stream
```

![架构](docs/assets/pypto-nvidia-architecture.svg)

**为什么承接方是 TensorIR**：TensorIR 本身就以 tile 为一等公民——布局传播、tile 选择、
多输出融合全部作用于 tile graph；它不是"cuTile 的后端"，而是 **cuTile 的前端/生产者**
（负责 tile 布局与结构化 lowering，产出 CUDA Tile IR 交 `tileiras` 汇编）。因此职责精确
互补：PyPTO 管 DSL/静态特化/产物契约/launch，TensorIR 管 tile 布局与 lowering，cuTile
管最终代码生成。PyPTO 的 tile 约定经 `CanonicalSchedule` 显式传入并被继续 lower——
手写库全部 21 个 graph 与 Inductor 生成的 kernel 均经此路径完成编译（101 个正确性用例、
整模型 33,448 次 compute launch 零 fallback 为证）。

在此之上，`packages/pypto-kernels` 是一套独立于框架的手写 PyPTO 算子库（13 模块 /
21 个 `@pl.jit` graph），覆盖 Qwen3.5 混合注意力结构（24 层 GDN + 8 层全注意力）所需的
attention / GDN / RoPE / RMSNorm / 因果卷积 / 线性投影 / embedding / LM head 全部算子；
MLP 激活由 Inductor 后端自动融合。SGLang 以官方插件机制集成，**本体零补丁**。

## 实测结论（RTX 5090 Laptop 24 GiB，2026-09 冻结证据）

| 项目 | 结果 |
|---|---|
| Qwen3.5-9B 正确性 | PASS：3 次冷启动 × 10 请求 × 64 token，与 SGLang 原生实现**逐 token 一致**（输出序列 SHA-256 唯一） |
| model-forward PyPTO coverage | **100%**：33,448 次 compute 调用 = 31,400 手写 + 2,048 Inductor 生成，fallback 0 |
| 算子回归 | PASS：8 套件 / 101 用例（2026-09-01 真机复跑） |
| 端到端吞吐 | PyPTO **2.3393 tok/s**；SGLang matched **14.9754**；SGLang optimized **12.5000** |
| 相对性能 | **PyPTO = matched 的 15.62%**（95% CI [15.59%, 15.70%]）；**= optimized 的 18.71%**（CI [18.69%, 18.75%]） |
| 瓶颈归因 | CUPTI/NVTX 对账闭合：差距主要来自未优化的结构化 matmul（6–195×），非桥接机制本身 |

| 指标（p50） | PyPTO | matched | optimized |
|---|---:|---:|---:|
| E2E | 27358.76 ms | 4273.67 ms | 5119.99 ms |
| TTFT | 3343.10 ms | 73.21 ms | 151.39 ms |
| TPOT | 381.23 ms | 66.66 ms | 78.86 ms |
| 输出吞吐 | 2.3393 tok/s | 14.9754 tok/s | 12.5000 tok/s |
| 峰值显存 | 17.49 GiB | 18.09 GiB | 22.08 GiB |

三条 lane：**PyPTO**（全部算子走 PyPTO，关闭 CUDA Graph/overlap）；**matched**（SGLang
默认算子，配置与 PyPTO 对齐：`mem_fraction_static=0.78, cpu_offload_gb=2`，同样关闭
CUDA Graph/overlap）；**optimized**（SGLang 官方默认最优，开 CUDA Graph/overlap）。
负载：chat 模板 31 输入 token + 贪心 64 输出，BF16，TP1，并发 1；每 lane 4 次独立冷启动，
headline 取各次冷启动 p50 的中位数，CI 为 10,000 次 percentile bootstrap。机器可读结果：
`state/evidence/qwen35-9b-release-results-current.json`。

**四张真机运行截图**（PowerShell 中 `wsl -d Ubuntu` 真实执行；侧车 JSON 含命令与 PNG SHA-256，
复现方法见[截图复现](#截图复现)）：

| 环节 | 命令 | 截图 |
|---|---|---|
| 构建（wheels→native→CTest→install 四阶段，`status: complete`） | `tools/build_release.py --stage all` | ![build](docs/assets/screenshots/build-release.png) |
| 算子正确性（8 套件 101 用例，`all_correct: true`） | `tools/run_operator_regression.py --stage all` | ![operator](docs/assets/screenshots/operator-correctness.png) |
| 算子级性能 A/B（PyPTO vs SGLang stock，4+4 冷启动） | `tools/run_operator_performance.py --matrix` | ![perf](docs/assets/screenshots/operator-performance.png) |
| 端到端推理（固定 prompt 64 token 贪心解码 + 逐 token 门禁） | `tools/run_model_correctness.py all` | ![model](docs/assets/screenshots/model-inference.png) |

## 环境要求

| 组件 | 版本（锁定） |
|---|---|
| GPU | NVIDIA RTX 5090（SM120 / compute capability 12.0，24 GiB） |
| OS | Ubuntu 26.04（WSL2 可用；本项目即在该环境实测） |
| CUDA Toolkit | 13.3（`tileiras` 汇编器随 toolkit 提供，构建前按路径 + SHA-256 + 版本校验） |
| Python | CPython 3.14.6 |
| PyTorch | 2.13.0+cu130 |
| SGLang | 0.5.18（`71de97b`，未修改） |
| 构建 | CMake 3.31、Ninja、C++ 工具链（`--jobs 24` 为契约要求） |

Python 依赖由哈希完整的 lock 文件驱动：`environment/conda-linux-64.lock`、
`environment/python-requirements.lock`；运行期锁：`environment/release-runtime.json`。
源码身份由 `vendor/source-lock.json` + git bundle + 391 个 patch 锁定。

## 快速开始

以下命令在仓库根目录执行；第 1–4 步一次性，之后可重复运行任意测试。

```bash
# 0.（可选）校验三个上游源码树与 bundle+patch 完全一致（重放全部 391 个 patch）
python3 tools/verify_source_release.py --replay-patches

# 1. 物化上游源码树：.sources/{pypto,tensor-ir,sglang}（从 vendor/git bundle 检出锁定 revision）
python3 tools/bootstrap_release.py --jobs 24

# 2. 创建正式 release 环境 envs/pypto-release（lock 驱动；亦生成基线环境）
python3 tools/bootstrap_release_environment.py

# 3. 四阶段构建：wheels → native（CMake/Ninja，内嵌 TensorIR 私有子项目）→ CTest（13/13）→ install
envs/pypto-release/bin/python tools/build_release.py --stage all --jobs 24

# 4. 下载模型（Qwen3.5-0.8B / 9B → models/，MANIFEST 校验 SHA-256）
envs/pypto-release/bin/python tools/download_release_models.py --model all
```

构建产物为三个 wheel（`pypto`、`pypto-kernels`、`pypto-framework-plugins`）安装进
release 环境；每次构建的原始日志与 wheel 清单落盘 `runs/<run-id>/`。

## 算子正确性回归（regression test，已入库）

```bash
envs/pypto-release/bin/python tools/run_operator_regression.py --stage all
```

结构门（`pytest -n24` 静态契约校验）+ 8 个 GPU 套件 / 101 用例：编译分类、数值正确性、
真实模型形状的状态算子（GDN/conv 多 token）、paged attention、QK-norm+RoPE、线性投影/LM
head、CUDA Graph 生命周期、Inductor SwiGLU。通过标准：全部套件 `passed: true` 且
`all_correct: true`（2026-09-01 真机复跑通过，报告落盘 `runs/<run-id>/operator-numerical-regression.json`）。

## 性能回归（只测性能，含 baseline 对比；已入库）

```bash
# ① 算子级 A/B：7 个功能对齐算子，PyPTO vs SGLang stock，4+4 次冷启动
envs/pypto-release/bin/python tools/run_operator_performance.py --matrix --model-path models/Qwen3.5-9B

# ② 整模三 lane 矩阵：pypto / sglang-matched / sglang-optimized，3×4 次冷启动
envs/pypto-release/bin/python tools/run_performance_regression.py --matrix \
    --model-path models/Qwen3.5-9B --optimized-memory-mode matched

# ③ SwiGLU 融合消融：eager vs 官方 Inductor(CUDA/Triton) vs PyPTO 后端
envs/pypto-release/bin/python tools/run_inductor_ablation.py --mode pypto --phase prefill \
    --output runs/ablation-prefill-pypto.json
envs/pypto-release/bin/python tools/summarize_inductor_ablation.py \
    --prefill-eager runs/ablation-prefill-eager.json \
    --prefill-inductor-nv runs/ablation-prefill-nv.json \
    --prefill-pypto runs/ablation-prefill-pypto.json \
    --decode-eager runs/ablation-decode-eager.json \
    --decode-inductor-nv runs/ablation-decode-nv.json \
    --decode-pypto runs/ablation-decode-pypto.json \
    --output state/evidence/qwen35-9b-inductor-ablation-current.json
```

计时边界为 CUDA event；每用例 20 次预热 + 30 批 × 100 次调用。历史正式结果冻结于
`state/evidence/qwen35-9b-operator-performance-breakdown-current.json`、
`qwen35-9b-inductor-ablation-current.json`。复跑后可用 `python3 tools/print_operator_ab_table.py`
打印最新一次 A/B 聚合的逐算子对比表。

## 端到端推理：100% PyPTO 的 Qwen3.5-9B（固定 prompt 复现）

```bash
# ① 一次性生成 HuggingFace transformers 语义 oracle（首步 logits top-k 校验基准）
envs/pypto-release/bin/python tools/run_transformers_semantic_oracle.py \
    --model-path models/Qwen3.5-9B --device cuda \
    --output runs/semantic-oracle-qwen35-9b-chat-nonthinking.json

# ② 端到端正确性：stock reference + 3 次冷启动 candidate，逐 token ID 门禁 + coverage 审计
envs/pypto-release/bin/python tools/run_model_correctness.py all \
    --model-path models/Qwen3.5-9B \
    --semantic-oracle runs/semantic-oracle-qwen35-9b-chat-nonthinking.json
```

固定 prompt（chat 模板 non-thinking，31 输入 token，贪心 64 输出）：

```text
为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？
```

预期输出（64 token 上限内，与 SGLang 原生实现逐 token 一致）：

```text
关于“鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作”这一说法，**目前并不存在客观依据，
这很可能是一个网络误传、营销号夸大其词，或者是将其他作品的评价张冠李戴了**。

事实上，《月鳞绮纪》（原名
```

通过标准：30 请求全部逐 token 一致（输出序列 SHA-256 唯一）、teacher-forced logits 冻结
策略通过、CUPTI coverage 33,448/33,448 且 fallback=0。复跑后可用
`python3 tools/print_model_gate_live.py` 打印最新一次运行的通过状态、coverage 与输出文本。

## 目录结构

```text
packages/pypto-kernels/            # 手写 PyPTO 算子库（13 模块 / 21 graph）
packages/pypto-framework-plugins/  # torch.compile(backend="pypto") 后端 + SGLang 插件
.sources/pypto                     # PyPTO fork（+300 commits：NVIDIA target、编译契约、运行时）
.sources/tensor-ir                 # TensorIR fork（+89 commits：gather/scatter 布局、artifact 契约）
.sources/sglang                    # SGLang v0.5.18（未修改）
vendor/                            # 源码身份锁：git bundle、391 patch、source-lock.json
tools/                             # 构建 / 正确性 / 性能 / 推理全部入口脚本
benchmarks/release/                # lane、workload、算子清单（冻结测试契约）
state/evidence/                    # 冻结实测证据 JSON
models/ tests/ demo/ docs/         # 模型 / harness 测试 / 文章 Demo 语料 / 图与截图
envs/ builds/ runs/ caches/        # 环境 / 构建产物 / 运行原始输出（gitignore）
```

## 实现概要

**PyPTO 侧**（`.sources/pypto`）：NVIDIA target 身份与 SM120 traits；不可变 `CompileRequest v1`
（`ToolchainIdentity` 哈希 pypto/tensor-ir/cuda-tile/LLVM/CUDA toolkit/tileiras 的版本与
SHA）→ `CanonicalSchedule`/`KernelBuildSpec` → 严格确定性编译门面
`compile_structured_strict[_cached]`（规范 MessagePack 序列化，同输入必同产物）；
`codegen/nvidia/tensor_ir_codegen.cpp` 把 HIR 严格子集分析为确定性 TensorIrModule，
`typed_tensor_ir_module_spec.h` + `compiler/nvidia_typed_tensor_ir_builder.cpp` 用 MLIR
ODS/OpBuilder 构建 typed `ModuleOp`（无字符串拼 IR，lint 强制）；`Artifact v1`（cubin +
完整 kernel ABI）与进程缓存；driver-only 运行时 `NvidiaExecutable`（仅 `dlopen libcuda`，
`PrepareLaunch` 生成不可变 launch packet，在调用者 stream 上发射，支持 CUDA Graph 租约）；
`tileiras` 以路径+SHA-256+版本校验后在受限子进程中运行；JIT 特化保留逐元素 stride
（row-pitched / 零 stride 广播）。

**TensorIR 侧**（`.sources/tensor-ir`）：row gather/scatter 布局 lowering（paged KV、
embedding、GDN 状态池所需）；惰性输入加载与一次性布局转换；兼容多输出融合；残差图切分
修复；`CudaTilePreparedArtifact/CompiledArtifact` 运行时无关产物契约与免 CUDA 的 cubin
校验；`tileiras` 子进程加固（关闭 stdio、显式超时、scratch 配额）；单位维 matmul、
归约切片、零 stride 广播等 lowering 正确性。

**Inductor 后端**（`packages/pypto-framework-plugins`）：entry point 注册 `pypto` 后端，
内部调用完整官方 `compile_fx`（`cuda_backend="pypto"`、`implicit_fallbacks=False`、
Dynamo 严格失败）；可逆地替换 Inductor CUDA 调度器与 wrapper（上下文外自动还原 Triton），
并将 Triton 哈希钉为常量使其不进入缓存身份；仅接受 Pointwise 与尾轴 Reduction 节点
（其余 `StrictCoverageError`）；pointwise 体在 ops 记录器中重放并**生成字面量 `@pl.jit`
DSL 源码**，stride 精确特化后走 `compile_structured_strict_cached`；wrapper 单行
`pypto_launch(...)` 负责 ABI 校验与 caller-stream launch；CUPTI coverage 审计工具随包提供。

**手写算子库**（`packages/pypto-kernels`）：一个算子 = 一个 `@pl.jit` graph；launch 前做
shape/stride ABI 校验；与 Inductor 后端共享同一编译通路。清单：attention（稠密/掩码/
paged decode/paged prefill/KV 写入/gather，7 graph）、GDN recurrent（L2 归一化 + softplus
衰减门 + 秩一状态更新 + 输出投影，`pl.InOut` 状态池）、GDN 打包投影、宽度 4 因果卷积、
QK RMSNorm+部分 RoPE+gate 切分、BF16 线性投影、FP32 LM head、embedding/token-id gather、
RMSNorm 三变体、NeoX RoPE、sigmoid 门控、SwiGLU。

**SGLang 集成**（零补丁）：官方插件机制注册 AROUND 钩子，替换约 20 个内部调用点为
`pypto-kernels` 算子；注册 `--attention-backend pypto` 与 GDN backend；注册失败即
`SystemExit`（防插件加载器吞异常）。

## 性能详情与归因

**算子级 A/B**（PyPTO p50 / stock p50，ms/调用）：

| 算子（形状） | PyPTO | stock | 倍数 |
|---|---:|---:|---:|
| SwiGLU decode 1×24576 | 0.2035 | 0.0094 | 21.7× |
| SwiGLU prefill 31×24576 | 0.2030 | 0.0091 | 22.2× |
| gate/up 线性 decode 1×4096×24576 | 2.5967 | 0.2389 | 10.9× |
| gate/up 线性 prefill 31×4096×24576 | 45.0186 | 0.2512 | 179.2× |
| down 线性 decode 1×12288×4096 | 4.5270 | 0.1228 | 36.9× |
| down 线性 prefill 31×12288×4096 | 25.2897 | 0.1299 | 194.7× |
| FP32 LM head 1×4096×248320 | 14.5464 | 2.4238 | 6.0× |

**SwiGLU 融合消融**（热 call ms / 冷编译 ms / kernel 数 / 相对 eager）：

| 形状 · 模式 | 热 call | 冷编译 | kernels | vs eager |
|---|---:|---:|---:|---:|
| prefill 19×24576 · eager | 0.042966 | 35.3 | 6 | — |
| prefill · 官方 Inductor（Triton） | 0.032915 | 1098.3 | 1 | +30.5% |
| prefill · PyPTO 后端 | 0.210618 | 1696.2 | 1 | −79.6% |
| decode 1×24576 · eager | 0.040543 | 32.8 | 6 | — |
| decode · 官方 Inductor（Triton） | 0.033808 | 1045.9 | 1 | +19.9% |
| decode · PyPTO 后端 | 0.212401 | 1733.3 | 1 | −80.9% |

**CUPTI 逻辑阶段归因**（p50 ms/请求）：PyPTO forward compute 合计 22318.81，其中
未归因桶（手写线性层）20184.08、attention core+gate 1154.36、LM head 931.87；matched
合计 1285.79、optimized 合计 2131.56。与 optimized 的 E2E 差距 22238.77 ms 中，采样
compute 差距 20187.25 ms + 非采样残差 2051.52 ms，独立阶段对账残差 −9.11 ms（闭合）。

**结论**：端到端 15.62%/18.71% 的差距主要来自结构化 matmul 未做 tensor core 流水与
tile autotune（GEMM 类 6–195×，prefill 形态最差），其次是 GDN 逐 token 发射与 decode
高 launch 密度；融合机制本身与官方 Triton 后端等价（6→1 kernel，launch −83.33%）。
详细数字与解释见 `state/evidence/` 对应 JSON。

## 截图复现

四张截图由 `tools/windows/capture_powershell.ps1` 在 Windows Terminal（Ubuntu 紫色
profile）中真实运行捕获：窗口内嵌套 PowerShell 提示符执行 `wsl -d Ubuntu`，进入真实
Ubuntu 提示符后运行命令，完成后 `PrintWindow` 截图。在 PowerShell 中：

```powershell
$Repo = "\\wsl.localhost\Ubuntu\home\<user>\pypto-love-tensor-ir"
& "$Repo\tools\windows\capture_powershell.ps1" -Title "build-release" `
  -LinuxCommand "envs/pypto-release/bin/python tools/build_release.py --stage all --jobs 24 2>&1 | tail -1 | python3 -m json.tool" `
  -OutputPath "$Repo\docs\assets\screenshots\build-release.png" `
  -MetadataPath "$Repo\state\evidence\build-release-capture-current.json" `
  -Workspace "/home/<user>/pypto-love-tensor-ir"
```

每次捕获生成 sidecar JSON（命令、退出码、窗口尺寸、PNG SHA-256、非空白像素采样）；
截图绑定证据清单由 `python3 tools/generate_article_screenshot_manifest.py` 重新生成。

## 限制与许可

- 结构化 matmul 未做深度优化（无 tensor core 流水 / autotune），为当前性能差距主因；
- Inductor 后端仅融合 pointwise/尾轴归约，GEMM+epilogue 与跨节点融合为 fail-closed 待办；
- GDN 长 prefill 为逐 token 有序 launch（状态边界数值契约），非批式 recurrent 图；
- 所有 kernel 按 shape/stride 静态特化，新形状需编译（产物落盘可复用）；
- 许可边界：PyPTO/CANN 许可证不允许在非华为处理器上运行与分发（见开头 IMPORTANT 与
  [LEGAL_NOTICE.md](LEGAL_NOTICE.md)）；TensorIR 为 Apache-2.0 with LLVM Exceptions；
  `pypto-framework-plugins` 为 Apache-2.0；`pypto-kernels` 暂未声明独立许可证。
