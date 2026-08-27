# 昇腾算子风格 vs 当前 pypto-kernels 实现：证据与结论

日期：2026-08-27。问题：PyPTO kernel 是否应该像
`projects/pypto-kernels/src/pypto_kernels/attention.py` 那样（多 kernel
分解 + ones-matmul 展开）实现？昇腾平台也是这样吗？

## 证据（昇腾生态，官方/一手来源）

| 模型算子 | 昇腾的形态 | 来源 |
|---|---|---|
| Attention | **单一融合算子** `torch_npu.npu_fusion_attention`（底层 FlashAttentionScore），官方推荐"融合算子替换"原生实现；另有 Attention+LayerNorm 单核融合实践 | 昇腾社区 TorchNPU 60RC1/600/60RC3/720 性能调优指南 |
| RMSNorm | **单一融合算子** `torch_npu.npu_rms_norm`；`fused_add_rmsnorm`（残差 add + RMSNorm 融合为一核，省一次全局显存写读）是 910B 上的标准练习 | 昇腾社区 TorchNPU 融合算子替换文档；浙大 HPC101 Lab3.5 |
| RoPE | 预置融合算子 `aclnnApplyRotaryPosEmb`、`RotaryMul` 融合替换 | CANN AOL 加速库接口文档；TorchNPU 调优指南 |
| 激活 | RMSNorm+SwiGLU 融合单核（"算子融合减少 DDR 访问，片上计算最大化利用 UB 带宽"） | CSDN 智能体开发者社区 |
| 官方定义 | **"融合算子是指将多个独立的'小算子'融合起来成为一个'大算子'"** | CANN 商用版 8.0.RC1 开发文档·融合算子编程 |

## 结论（诚实）

**当前 pypto-kernels（v1）不是昇腾风格。** v1 的 9-kernel attention、
5-kernel RMSNorm、128 倍冗余 ones-matmul 展开是"正确性优先"的
workaround——根源是固定 TensorIR producer 当时拒绝广播 lowering
（CHECKPOINT CP-0051 定性，HANDOVER W2）。它达到了数值验收，但结构上
与昇腾"一个模型算子 = 一个融合 kernel + 显式 tiling"的形态相反。
用户判断正确。

## 昇腾风格 → PyPTO 的翻译

| 昇腾概念 | PyPTO 对应物 |
|---|---|
| 一个融合算子（一次 launch） | **一个 TensorIR graph**（一次 `compile_structured_strict`） |
| 算子内的 block/tiling 代码 | `CanonicalSchedule` 的 `dim_000...` tile 参数（调度即 tiling） |
| 算子开发者写整个算子 | 一个 HIR builder 函数 = 一个算子，**不做 Python 侧多 launch 编排** |
| 融合发生在算子边界内 | pointwise DAG 链在 graph 内融合；归约/matmul 是天然 graph 边界 |

诚实边界：昇腾 Ascend C 可以在单核内写 KV-block 循环（真单核 FA）；
PyPTO 当前 graph 算子集没有跨族（归约+matmul）单 graph 的表达，
所以 attention 的下限是 2 个 graph（softmax 段 + 值混合 matmul），
真单核 FA 需要专用 graph kind（后续项，属 pypto 编译器改动）。

## 行动

在 `projects/pypto-kernels-v2/` 按上述翻译重实现（不影响 codex 正在
修改的 v1 / pypto 编译器 / TensorIR）。每个算子 = 一个 graph builder；
今天即可执行的有 silu_and_mul（昇腾 SwiGLU 融合的直接对应物）与
fused_add；依赖广播 lowering（codex L0）的算子以单 graph 形态写好并
在 HIR 层验证，状态显式标注。状态表见该目录 README.md。

来源：
- [FlashAttentionScore-融合算子替换-NPU亲和适配优化（昇腾社区）](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/ptmoddevg/trainingmigrguide/performance_tuning_0027.html)
- [torch_npu.npu_fusion_attention API](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000762.html)
- [RmsNorm & RmsNormGrad 融合算子替换](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0024.html)
- [融合算子编程（CANN 8.0.RC1 开发文档）](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0071.html)
- [aclnnApplyRotaryPosEmb（CANN AOL）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/aolapi/context/aclnnApplyRotaryPosEmb.md)
- [HPC101 Lab3.5：fused_add_rmsnorm 算子开发](https://hpc101.zjusct.io/lab/Lab3.5-AscendC-Op/)
