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
| `rmsnorm` | 1 | **原生 tile** ✅ | 行 tile 内 FP32 square/reduce/rsqrt/broadcast；一次 launch |
| `rope` | 2 | **待迁移并融合** | 旧偶/奇 graph 可执行，但必须改成一个原生 tile 算子 |
| `gdn` read/update | 5+1 | **待迁移并融合** | 旧多 graph 仅作数值基线，目标是模型算子单 graph/launch |
| `attention` | 2（不完整） | **待迁移并补齐** | 旧 post-exp 分解仅作基线，目标是原生 tile attention graph |

CP-0062 已关闭 broadcast producer 阻塞；分类证据在
`benchmarks/classify_results.json`。`compiled` 与 GPU 数值 `all_correct`
仍是两个独立门，后者不得由分类结果代替。

## 运行

```
envs/pypto-nvidia/bin/python -B tests/test_operators.py      # 结构：单 graph + 状态分类
envs/pypto-nvidia/bin/python -B benchmarks/exec_sm120.py     # 可执行算子 GPU 验收
```

本仓是唯一算子实现；模型接入只允许引用这些原生 tile 算子。
