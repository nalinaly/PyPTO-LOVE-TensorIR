# pypto-kernels v2 — 昇腾风格的高层次算子库

依据 `docs/ascend_style_evidence.md` 的结论重写：**一个模型算子 = 一个
PyPTO TensorIR graph**（编译一次/形状，调用即一次 launch），tile 进
`CanonicalSchedule`（调度即 tiling）——对应昇腾"融合算子 + 算子内
block/tiling"的形态。不做 Python 侧多 launch 编排，不用 ones-matmul
展开。**本目录独立于 v1**（codex 正在改 v1 与 pypto 编译器），只共享
不可变的 DSO 产物。

## 状态表

| 算子 | graph 数 | 状态 | 说明 |
|---|---|---|---|
| `silu_and_mul`（SwiGLU） | 1 | **可执行** ✅ | 纯 pointwise 链；GPU 验收 vs eager 全绿（benchmarks/v2_exec_sm120.py） |
| `fused_add`（残差加） | 1 | **可执行** ✅ | 同上；fused_add_rmsnorm 的前半 |
| `rmsnorm` | 1 | blocked-on-L0 | 单 graph：sum→scale→+eps→rsqrt→broadcast→mul（`npu_rms_norm` 的直接对应物）。**HIR 已被发射层接受**，仅 producer 广播 lowering 拒绝——tests 用 classify() 证明失败在 producer 而非 HIR |
| `rope` | 1 | blocked-on-L0 | cos/sin 表 [M,1] 行广播输入 + 逐点旋转（`aclnnApplyRotaryPosEmb` 对应物）；位置静态，表属数据准备 |
| `gdn` read | 2 | blocked-on-L0 | 读路径单 pointwise graph（广播操作数）+ M=1 状态读 matmul；状态更新同依赖（HANDOVER L0b） |
| `attention` | 2 | blocked-on-L0+ | softmax 段（归约+广播尾羽）+ 值混合 matmul；真单核 FA 需专用 graph kind（FUTURE） |

blocked-on-L0 = 等 codex 的 L0（pinned producer 广播 lowering）落地，
这些算子**无需改接口**即变为可执行；softmax 段还需 pypto 尾羽分析器
泛化（sum→1/sum→broadcast-mul），属编译器侧后续项。

## 运行

```
envs/pypto-nvidia/bin/python -B tests/test_v2_operators.py      # 结构：单 graph + 状态分类
envs/pypto-nvidia/bin/python -B benchmarks/v2_exec_sm120.py     # 可执行算子 GPU 验收
```

## 与 v1 的关系

v1（分解式）保持不动，直到 v2 全算子可执行后整体替换；届时 v2 就是
HANDOVER L4b（融合阶段）的落点：算子边界内融合、归约/matmul 是天然
graph 边界，与昇腾融合算子的边界划分一致。
