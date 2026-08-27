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
| `rmsnorm` | 1 | **可执行** ✅ | BF16 单 graph：square→sum→scale→+eps→rsqrt→broadcast→mul；GPU eager 对拍全绿 |
| `rope` | 2（偶/奇各一） | **可执行** ✅ | 每半一个 pointwise graph（row_expand_mul×2 + sub/add）；偶/奇 GPU eager 对拍全绿 |
| `gdn` read | 5 | **可执行** ✅ | q-decay、state matmul、compose、dot reduce、delta+read 五个显式 graph 全部 run-pass；不隐藏为单 launch |
| `gdn` state update | 1 | **可执行** ✅ | rank-3 broadcast DAG：decay·state + beta_key⊗value，一 graph/一 launch |
| `attention` post-exp path | 2 | **可执行** ✅ | exponent normalize（sum→recip→broadcast-mul）+ value-mix matmul 均 run-pass；QK/max-shift/exp 与真单核 FA 仍属专用 graph kind（FUTURE） |

CP-0062 已关闭 broadcast producer 阻塞；分类证据在
`benchmarks/v2_classify_results.json`。`compiled` 与 GPU 数值 `all_correct`
仍是两个独立门，后者不得由分类结果代替。

## 运行

```
envs/pypto-nvidia/bin/python -B tests/test_v2_operators.py      # 结构：单 graph + 状态分类
envs/pypto-nvidia/bin/python -B benchmarks/v2_exec_sm120.py     # 可执行算子 GPU 验收
```

## 与 v1 的关系

v1（分解式）保持不动，直到 v2 全算子可执行后整体替换；届时 v2 就是
HANDOVER L4b（融合阶段）的落点：算子边界内融合、归约/matmul 是天然
graph 边界，与昇腾融合算子的边界划分一致。
