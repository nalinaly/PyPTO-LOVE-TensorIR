# 给 codex 的启动 prompt（v2 算子路线：run-pass + fix；zcode 只做集成）

直接复制以下内容给 codex：

```
你在工作区 /home/zhaosiying/pypto-love-tensor-ir 接手算子工作。你没有我们
对话的上下文，所以先读材料再动手。分工已定：**你负责让所有 v2 算子
"run pass"（可编译、可 launch、数值验证通过）并修复阻塞；模型前向的
集成由另一个 reviewer（zcode）在你完成后进行——你不要去改模型 harness。**

【背景：新的算子形态要求（这是用户的新决定，优先级最高）】
旧实现 projects/pypto-kernels（下称 v1）把 attention 拆成 9 个 kernel、
用 128 倍冗余 ones-matmul 做广播展开——用户认定这不是目标形态。调研
结论（docs/ascend_style_evidence.md，官方来源）：昇腾生态的算子是
"多个小算子融合为一个大算子"（CANN 官方定义），attention/rmsnorm/rope
都是单一融合算子（npu_fusion_attention / npu_rms_norm /
aclnnApplyRotaryPosEmb）。因此新要求：

  **一个模型算子 = 一个 PyPTO TensorIR graph**（编译一次/形状，调用即
  一次 launch）；tile 写进 CanonicalSchedule（调度即 tiling）；不做
  Python 侧多 launch 编排；不用 ones-matmul 展开。

落点是新目录 projects/pypto-kernels-v2/（独立 git 仓，最新提交
70243ab），与 v1 完全隔离。

【第一步：按顺序通读，读完前禁止改代码】
1. docs/ascend_style_evidence.md        ← 为什么是昇腾风格（证据与结论）
2. projects/pypto-kernels-v2/README.md   ← v2 状态表与设计
3. projects/pypto-kernels-v2/src/pypto_kernels_v2/_boot.py   ← classify()
   机制：RuntimeError=producer 阶段失败（图合法），ValueError=HIR 发射
   被拒（图的 bug）
4. projects/pypto-kernels-v2/tests/test_v2_operators.py 与
   benchmarks/v2_exec_sm120.py           ← 验收模式
5. docs/HANDOVER_codex.md 的 §4 W2 与 §1 ← 广播阻塞的四种已试编码、
   源码级线索、TensorIR 修改的 submodule 机制（用户已授权修改）
6. CHECKPOINT.md 的 CP-0055/CP-0051     ← 不可分解性论证与调试史

【v2 当前状态（三档）】
A. 已在 SM120(5090) 上编译+launch+数值验证通过：
   silu_and_mul、fused_add、gdn_compose（见 benchmarks/v2_exec_results.json）
B. 单 graph 已写好、HIR 合法、但被 pinned producer 的广播 lowering 拒绝
   （classify() 返回 producer-blocked）：rmsnorm（1 graph）、rope 偶/奇半
   （各 1 graph）、attention softmax 缩放（1 graph）、gdn delta 广播
   （1 graph）
C. 设计已定待 B 解锁：attention（softmax 段 3 graph + 值混合 matmul）、
   gdn 完整读路径、GDN 状态更新（HANDOVER L0b）

【你的任务】
T1（fix，关键路径）：在 projects/pypto/3rdparty/nvidia/tensor-ir 实现
    broadcast-into-pointwise 的 lowering。线索：BroadcastOpConversion 在
    lib/Conversion/TensorToCudaTile/AffineMapImpl.cpp:3584 已注册但被
    layout-propagation 管线绕过（该管线期望广播折叠进 load，
    emitLoadMaybeBroadcast）；stride-0 静态维度在 parse 阶段被拒；
    显式 broadcast op 是 IterationSpaceTransition，可能在 skeleton
    构建处失败。调试时可在本地测试脚本临时放宽诊断拿到真实错误
    （不要改严格桥接的生产行为）。
    交付：submodule 分支 commit + git format-patch 补丁存进父仓 +
    gitlink bump + 父仓 push（无认证则记录）+ pypto CTest 新增广播
    golden + DSO 重建。注意：DSO 重建后若产物路径/sha 变化，同步更新
    v2 的 _boot.DSO_PATH。
T2（run pass）：让 B 档全部算子变成 A 档——每翻转一个，把
    tests/test_v2_operators.py 里对应断言从 producer-blocked 改为
    compiled，并在 benchmarks/ 里加 GPU 数值验收（对照 eager，BF16
    精度容差，模式照抄 v2_exec_sm120.py；注意 launch 参数顺序 =
    builder 输入顺序，v2 曾在这里踩坑）。
T3（补齐 C 档）：B 档解锁后，按 README 状态表把 attention softmax 段、
    gdn 完整读路径、GDN 状态更新（L0b）以 v2 单 graph 形态实现并
    run-pass。真单核 FlashAttention 需要专用 graph kind，记 FUTURE
    不强求。
T4：每完成一项，更新 CHECKPOINT.md（接续现有编号）+ commit + 证据落盘。
    v2 是独立仓，在它里面 commit；pypto/父仓的改动各自 commit。

【验收定义（由 zcode 执行，你留证据链）】
- tests/test_v2_operators.py 全绿且不再有 producer-blocked 断言
- benchmarks 全部 all_correct=true，结果 json 落盘
- classify() 三态里只剩 compiled
- pypto CTest 13/13 + 新增广播 golden 全绿
- 你不动：projects/pypto-kernels（v1）、模型 harness
  （benchmarks/operators/pypto_qwen35_0p8b_forward_sm120.py）、
  upstream/*（zero-diff，只读）

【安全约束，最高优先级，违反即事故】
- /home/zhaosiying/amdgpu-sim、/home/zhaosiying/zcode-lane 及其进程
  只读，永不修改/kill/signal，即使它们占 GPU。
- 不重启机器；禁止模型特判（model_name/hidden_size==4096 之类）。
- 构建走 tools/run_isolated.py（500s 分块、绝对路径、--run-id-file
  重跑前 rm -f）；改 pypto C++ 或 3rdparty 后必须先 commit 再
  configure/build（strict clean-source guard）。
- TensorIR submodule 的 NVIDIA origin 只读：修改只在本地分支 +
  format-patch 进父仓，不 push 到 NVIDIA。

现在开始：先复述你理解的任务、分工边界和约束（不超过 15 行）给我
确认，然后从 T1 开始。
```
