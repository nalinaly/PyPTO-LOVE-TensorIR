# pypto-kernels — Qwen3.5 原生 tile DSL 融合算子库

唯一合格形态是 `examples/beginner/03_scalar_ops.py` 的原生写法：每个模型
算子用 `@pl.jit`，在 `pl.at(CORE_GROUP)` 内显式写 `pl.range`、
`pl.load(tile_shape)`、tile 运算和 `pl.store`。一个算子编译成一个
TensorIR graph，调用时一次 launch；不做 Python 多 launch 编排，也不做
ones-matmul 广播展开。仅仅用整张量 `tensor.*` graph 再附加 schedule
元数据不算完成。

## 状态表

| 算子 | graph 数 | 状态 | 说明 |
|---|---|---|---|
| `silu_and_mul`（SwiGLU） | 1 | **原生 tile** ✅ | `[1,128]` load/compute/store；静态双层 `pl.range`；一次 launch |
| `fused_add`（残差加） | 1 | **原生 tile** ✅ | `[1,128]` load/add/store；静态双层 `pl.range`；一次 launch |
| `fused_add_rmsnorm` | 1 | **原生 tile** ✅ | 同一 graph/launch 返回 residual sum 与 Qwen weighted norm |
| `linear`（投影/LM head） | 1 | **原生 tile** ✅ | `[1,K] @ [128,K].T` 输出 tile；一次 launch |
| `rmsnorm` | 1 | **原生 tile** ✅ | Qwen `normalized * (1+weight)` 完整公式；一次 launch |
| `rope` | 1 | **原生 tile** ✅ | 低/高半旋转在一个图内，两个 store 合成一个结果；一次 launch |
| `gdn_read` | 1 | **原生 tile** ✅ | state matmul、softplus、dot、delta 与相加在一个图；一次 launch |
| `gdn_state_update` | 1 | **原生 tile** ✅ | state/decay/beta/value 分块 load，tile 内 outer-product 更新，一次 launch |
| `gated_rmsnorm` | 1 | **原生 tile** ✅ | GDN `RMSNorm(x,weight) * SiLU(gate)` 单图/单 launch |
| `attention` | 1 | **原生 tile 核心** ✅ | QK→稳定 softmax→PV 全在一个图；一次 launch；paged/causal serving 扩展仍待接入 |

CP-0062 已关闭 broadcast producer 阻塞；分类证据在
`benchmarks/classify_results.json`。`compiled` 与 GPU 数值 `all_correct`
仍是两个独立门，后者不得由分类结果代替。

## 运行

```
envs/pypto-nvidia/bin/python -B tests/test_operators.py      # 结构：单 graph + 状态分类
envs/pypto-nvidia/bin/python -B benchmarks/exec_sm120.py     # 可执行算子 GPU 验收
```

本仓是唯一算子实现；模型接入只允许引用这些原生 tile 算子。
