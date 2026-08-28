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
| `fused_add_rmsnorm` | 1 | **原生 tile** ✅ | 同一 graph/launch 返回 residual sum 与 Qwen weighted norm |
| `sigmoid_mul` | 1 | **原生 tile** ✅ | full-attention 输出门 `value*sigmoid(gate)`；一次 launch |
| `embedding` / integer gather | 2 | **原生 tile** ✅ | BF16 token row gather 与 INT32/64 slot-table gather；无 one-hot；各一次 launch |
| `linear`（投影/LM head） | 2 | **原生 tile** ✅ | BF16 输出与 FP32 LM-head 输出；每次调用一个 structured matmul launch |
| `qk_rmsnorm_rope` | 1 | **原生 tile** ✅ | Q/K norm + partial RoPE + gate split 单图；最终 DSO 上单 launch，Q/K 对参考零误差且 gate 精确一致 |
| `rmsnorm` | 1 | **原生 tile** ✅ | Qwen `normalized * (1+weight)` 完整公式；一次 launch |
| `rope` | 1 | **原生 tile** ✅ | 低/高半旋转在一个图内，两个 store 合成一个结果；一次 launch |
| `gdn_recurrent` | 1 | **原生 tile** ✅ | T1 primitive 完成 Q/K L2 norm、稳定 gate/beta、FP32 state decay/outer-product、InOut 回写与 BF16 输出；batch decode 一次 launch，prefill 按 token 顺序发射可 CUDA-Graph 捕获的 PyPTO launches；T13 state 误差不超过 `2.24e-8` |
| `gdn_projection` | 1 | **原生 tile** ✅ | 一次 launch 写入 output-major packed buffer，并返回四个连续零拷贝 view；13-row gate 对输入切片逐位一致 |
| `gated_rmsnorm` | 1 | **原生 tile** ✅ | GDN `RMSNorm(x,weight) * SiLU(gate)` 单图/单 launch |
| `attention` | 4 | **原生 tile** ✅ | dense + paged decode/cache-write/causal prefill；0.8B/9B、batch2 row-pitched cache 与 valid-length mask 数值门通过，最坏误差 `0.03333` |
| `causal_conv1d` | 1 | **原生 tile** ✅ | `[B,T]` width-4 FP32 累加 + SiLU，读取并 InOut 回写 BF16 row-pitched plane-major `[3,D]` history；decode/prefill state 逐位一致 |

发布回归不会读取或覆写仓内历史 JSON。`compiled` 与 GPU 数值
`all_correct` 是两个独立门，后者不得由分类结果代替；每次结果必须显式
写到控制仓 `runs/<run-id>/`。

## 运行

```bash
python -m pytest -q -n24 tests
python -B benchmarks/classify_sm120.py --output "$RUN_DIR/classify.json"
python -B benchmarks/exec_sm120.py --output "$RUN_DIR/operators.json"
python -B benchmarks/stateful_sm120.py --output "$RUN_DIR/stateful.json"
python -B benchmarks/paged_attention_sm120.py \
  --output "$RUN_DIR/paged-attention.json"
```

`pypto` 与 `pypto-framework-plugins` 应作为已安装分发包提供。仅诊断时可用
`PYPTO_KERNEL_DSO_PATH`、`PYPTO_KERNEL_PACKAGE_PATH`、
`PYPTO_KERNEL_CUDA_DRIVER_LABEL` 和 `PYPTO_KERNEL_CUDART` 覆盖自动发现；
发布命令不得写死某台工作站的路径。

本仓是唯一手写算子实现；模型接入只允许引用这些原生 tile 算子。项目的
公开发布仍受 `LICENSE_STATUS.md` 所述授权门约束。
