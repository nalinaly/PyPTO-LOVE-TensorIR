# 给 codex 的启动 prompt（v2：TensorIR 修改已授权，100% 硬性要求）

直接复制以下内容给 codex：

```
你在工作区 /home/zhaosiying/pypto-love-tensor-ir 接手一个进行中的项目。
上一个 agent（zcode）已把全部上下文写入交接文档，你的任务是读完它并执行
其中 §6 的任务清单。用户已明确：**必须实现 100% PyPTO kernel 覆盖，
且已授权直接修改 TensorIR 第三方源码**。

【第一步：按顺序通读，读完前禁止改任何代码】
1. docs/HANDOVER_codex.md          ← 唯一入口（v2），全部上下文
2. CHECKPOINT.md 的 CP-0055 → CP-0061（最近七个检查点）
3. HANDOVER §2 列的三个项目仓的当前 HEAD 代码：
   - projects/pypto-kernels/src/pypto_kernels/{rmsnorm,attention,gdn_kernel}.py
   - benchmarks/operators/pypto_qwen35_0p8b_forward_sm120.py（主 harness）

【安全约束，最高优先级，违反即事故】
- /home/zhaosiying/amdgpu-sim、/home/zhaosiying/zcode-lane 及其进程只读，
  永不修改/kill/signal，即使它们占 GPU。
- upstream/pytorch、upstream/sglang、upstream/triton 保持 zero-diff，
  只能读（读 sglang 的 qwen3_5.py 核对公式是任务 L3 的一部分）。
- **TensorIR 修改授权**：projects/pypto/3rdparty/nvidia/tensor-ir 是 git
  submodule（origin 只读指向 NVIDIA，本地分支 feature/pypto-private-build）。
  修改流程（HANDOVER §1 有完整版）：submodule 内开分支 commit →
  git format-patch 把补丁存进父仓 → 父仓 gitlink bump 并 commit → push
  父仓到 github.com/hw-native-sys/pypto.git。submodule 无法 push 到
  NVIDIA，不要尝试。改后先 commit 再 cmake（strict clean-source guard
  会拒绝脏树）。
- 禁止模型特判（model_name / hidden_size==4096 之类分支）。
- 构建只走 tools/run_isolated.py（500s 分块、绝对路径、--run-id-file
  指向的文件重跑前必须 rm -f）。

【你的任务 = HANDOVER §6，按序执行，100% 是硬性要求】
L0  （critical path）在 TensorIR producer 实现 broadcast-into-pointwise
    lowering（W2 源码线索：LayoutPropagationImpl.cpp 的管线 vs 已注册
    未生效的 BroadcastOpConversion；stride-0 静态维在 parse 阶段被拒）。
    交付：submodule commit + 父仓补丁 + gitlink bump + push + CTest
    broadcast golden + DSO 重建 + 全量回归。
L0b GDN 状态更新 kernel 化，清掉 576 个回退。
L1  修 9B 层数硬编码（24→32，从权重推导 layer_types）。
L2  9B 相关性主导门重跑（L1 之后），证据收 state/evidence/。
L3  对照 upstream/sglang 源码逐条修正 eager 参考公式（最大精度杠杆），
    重标定 golden 阈值（目标 corr≥0.99 / top-1≥90%），注明源码行号。
L4  按 §5 消账表清掉其余 fallback（z-silu→conv→L2-norm→RoPE→embedding
    one-hot matmul→向量代数），直到 census fallback == 0。
L5  D-0017 独占窗口逐家族计时，按 §7 如实归因结构性开销。

【工作纪律】
- 每完成一个任务：更新 CHECKPOINT.md（从 CP-0062 起编号）+ git commit
  + 证据 json 落 state/evidence/。
- 被阻塞时不许静默跳过：写进 CHECKPOINT 的“已知阻塞”段并继续下一项。
- 不要重做 §3 列出的已验证工作；不要“顺手优化”已验收的 kernel。
- 完成定义 = HANDOVER §8 的验收清单能全绿（验收由另一个 reviewer 执行，
  你需要留下可核对的证据链）。

现在开始：先复述你理解的任务清单和约束（不超过 20 行）给我确认，
然后从 L0 开始。
```
