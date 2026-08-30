# PyPTO LOVE TensorIR: Qwen3.5 on an RTX 5090

[简体中文](README.md) | [English](README_EN.md)

> [!IMPORTANT]
> This is a personal, non-commercial compiler research project. Running PyPTO
> on an NVIDIA card appears to conflict with the CANN Open Software License
> Agreement Version 2.0,
> which places explicit restrictions on use and distribution for non-Huawei AI
> processors. Non-commercial intent, a public interview statement, or a
> takedown promise is not a license exception. The work does not represent
> Huawei, NVIDIA, PyTorch, or SGLang. Rights holders may contact the author
> through the repository Issues to request removal. See LEGAL_NOTICE.md.
>
> Interview attribution (approximately 2 h 34 min):
> https://www.bilibili.com/video/BV1nB3u6tERu/?vd_source=f2f41aa7b5e3cc8e0a23942779ccea11

This project has two core features:

1. PyPTO DSL/HIR is lowered through typed TensorIR ODS/OpBuilder to NVIDIA
   TensorIR, then through CUDA Tile and tileiras to an SM120 Cubin.
2. A standard torch.compile(backend="pypto") TorchInductor backend turns
   eligible pointwise/reduction subgraphs into PyPTO DSL.

The single execution path is:

~~~text
SGLang -> pypto-kernels + TorchDynamo/Inductor -> PyPTO HIR
       -> typed TensorIR ModuleOp -> CUDA Tile -> tileiras -> sm_120a Cubin
       -> PyPTO Artifact/NvidiaExecutable -> caller-owned CUDA stream
~~~

![PyPTO NVIDIA SM120 architecture](docs/assets/pypto-nvidia-architecture.svg)

## Measured Status

Platform: NVIDIA GeForce RTX 5090 Laptop GPU (SM120, 24 GiB). Workload:
Qwen3.5-9B text-only, non-thinking chat template, 31 input tokens, 64 greedy
output tokens, BF16, TP1, concurrency 1.

| Item | Measured status |
|---|---|
| 9B correctness/coverage | Three fresh starts on the current wheel completed 10/10 Engine requests and one strict teacher-forced trace each |
| Model-forward coverage | Accepted traces: 33,448 compute calls = 31,400 handwritten PyPTO + 2,048 Inductor PyPTO, with zero unknown/fallback calls (100%) |
| 0.8B current wheel | Stock reference plus three candidate fresh starts completed; each trace covers 22,108/22,108 calls = 20,572 handwritten + 1,536 Inductor |
| PyPTO output throughput | Diagnostic median of four fresh starts: 2.6671 tok/s; the resource recheck found 3/4 starts below the 4 GiB GPU-free floor, so this is not a release headline |
| Matched SGLang | 15.4100 tok/s in the same diagnostic pair; its starts completed, but candidate resource violations and different offload controls invalidate the pair |
| PyPTO / matched | Diagnostic ratio 17.31%, CI [17.22%, 17.45%]; a formal ratio requires resource- and control-compliant reruns |
| Optimized SGLang | Not reported; its CUDA-graph configuration fell below the 4 GiB free-memory floor on this GPU |

Artifact-union versus model-forward compute intersection for accepted traces:

| model | total calls | handwritten | Inductor | unknown/fallback | coverage |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 22,108 | 20,572 | 1,536 | 0 | 100% |
| Qwen3.5-9B | 33,448 | 31,400 | 2,048 | 0 | 100% |

Diagnostic four-start end-to-end medians (not a release headline):

| lane | E2E | TTFT | TPOT | output tok/s | cold engine | first compile-trigger request |
|---|---:|---:|---:|---:|---:|---:|
| PyPTO | 23996.36 ms | 3154.40 ms | 330.86 ms | 2.6671 | 13578.98 ms | 25982.23 ms |
| matched | 4153.16 ms | 69.81 ms | 64.61 ms | 15.4100 | 15208.36 ms | 24013.73 ms |

The first compile-trigger request includes compilation and one complete 31+64
request; it is not compiler-only time. High-frequency NVML sampling observed a
PyPTO minimum of 4,185,067,520 free bytes, 109,899,776 bytes below the
4,294,967,296-byte floor; the first three starts crossed that floor. The values
remain reproducible diagnostics, but the pair status is
`invalidated-resource-and-control` and the formal matched ratio must be rerun. The
old pair also used `cpu_offload_gb=0` for PyPTO and `2` for matched. The new
performance-only configuration gives both lanes `cpu_offload_gb=2` and
`mem_fraction_static=0.78`; correctness configurations remain unchanged. The
matched lane keeps
enable_torch_compile=true but disables CUDA graphs, so this pinned SGLang
version does not invoke the global CompilerInterface; the report records
backend_invocation_observed=false. The PyPTO SwiGLU hook observes two generated
Inductor wrappers containing pypto_launch.

Machine-readable sources (including the two model-gate sidecars):

- state/evidence/qwen35-9b-performance-pair-current.json
- state/evidence/qwen35-0.8b-model-gate-current.json
- state/evidence/qwen35-9b-model-gate-current.json
- runs/release-operator-ab-20260830T123727Z-2370421-d0d384/aggregation.json
- state/evidence/qwen35-9b-inductor-ablation-current.json
- state/evidence/qwen35-9b-inductor-source-current.json

One full-model eager control was also run with torch.compile disabled while
keeping the matched providers: output 15.3418 tok/s and E2E 4171.67 ms. The
diagnostic matched compile-request median in the invalidated pair is 15.4100
tok/s and 4153.16 ms, but disabling
CUDA graphs prevents the pinned SGLang CompilerInterface/Inductor invocation.
This is therefore not a causal whole-model compile speedup; the timing-only
record is state/evidence/qwen35-9b-eager-compile-ablation-current.json.

These are measured implementation results, not performance targets. Fewer
launches do not automatically mean lower latency; current dominant costs are
handwritten matmul/LM-head and PyPTO launch/schedule overhead.

## Why TensorIR Fits as the PyPTO Backend

[PyPTO](https://github.com/hw-native-sys/pypto) owns the user DSL, HIR, static
specialization, artifact/cache, and runtime. [TensorIR](https://github.com/NVIDIA/tensor-ir)
is a tensor-level, tile-aware MLIR compiler frontend for layout propagation,
tiling, graph splitting, and TensorIR-to-CUDA-Tile lowering. CUDA Tile is the
downstream GPU IR/toolchain target; TensorIR is neither CUDA Tile nor a cuTile
runtime.

A PyPTO tile shape can continue lowering only when shape, dtype, element
stride, layout, iteration space, and mutation/alias ABI are static and pass
the TensorIR verifier. The bridge creates a TypedTensorIrModuleSpec, then uses
mlir::OpBuilder and ODS *Op::create to construct ModuleOp. Text printing is
only canonical serialization and diagnostics. Dynamic or overlapping strides,
unsupported contractions, invalid result anchors, and illegal alignment fail
closed.

| PyPTO fact | TensorIR representation | Required condition |
|---|---|---|
| pl.range / tile | iteration space / tile sizes | statically bounded |
| pl.load/store/view | typed slice/reshape/transpose/gather/scatter | access and layout verifiable |
| pl.InOut | read-write operand and mutation metadata | alias and result agree |
| matmul/reduce/broadcast | ODS operations | supported rank, dtype, contraction |

## Implementation

### PyPTO -> TensorIR

The bridge provides NVIDIA TargetInfo and SM120/runtime discovery; immutable
CompileRequest, CanonicalSchedule, and KernelBuildSpec; cache identity over
shape/dtype/stride/mutation/toolchain; typed GraphOp/stride attributes/results
and verification; pointwise, row reduction, BF16/FP32 matmul, norms,
gather/scatter, RoPE, dense/paged attention, KV writes, causal Conv1D, GDN
projection/recurrent lowering; Cubin/ABI/grid/workspace checks; and
NvidiaExecutable process/device/CUcontext/current-stream lifetime checks.

Key files are .sources/pypto/src/codegen/nvidia/tensor_ir_codegen.cpp,
.sources/pypto/src/codegen/nvidia/typed_tensor_ir_module_spec.h, and
.sources/pypto/src/compiler/nvidia_typed_tensor_ir_builder.cpp. Canonical
NVIDIA construction rejects source-string concatenation; lint and negative
geometry verifier tests are part of the regression contract.

### TorchInductor -> PyPTO

The plugin registers torch_dynamo_backends=pypto and calls full compile_fx.
Within a context-local scope it replaces CUDA scheduling and wrapper dispatch,
then restores the official backend. Strict mode disables fallback. The adapter
records node-body load/op/store sequences, preserves real row-pitched strides,
creates @pl.jit graphs, launches through pypto_launch on the current stream,
and records callable/artifact/prewarm caches, stable source hashes, and CUPTI
external correlation.

The complete Qwen MLP is not fused:

~~~text
handwritten gate/up linear -> one Inductor-generated PyPTO SwiGLU
                              -> handwritten down linear
~~~

Only the packed gate/up FP32 casts, sigmoid, two multiplies, and BF16 cast are
automatically fused.

## Handwritten Operator Library

packages/pypto-kernels is independent of the SGLang plugin. Its 15 Python
modules contain 18 @pl.jit graphs for dense/masked/paged attention, paged
gather/KV write/prefill, causal Conv1D, recurrent GDN, GDN projection, Q/K
RMSNorm plus partial RoPE and gate, embedding/integer gather, BF16 linear,
BF16-rounded FP32 LM head, three RMSNorm variants, RoPE, sigmoid-mul, and
SiLU-mul.

The most complex representative is
[gdn_recurrent_kernel](packages/pypto-kernels/src/pypto_kernels/gdn.py). It
combines pl.InOut, static loops, row reduction/broadcast, FP32 matmul, and
outer-product state updates. Long prefill advances state in token order; it is
not presented as one mega-kernel.

## Ablation and Breakdown

Fixed Qwen3.5-9B SwiGLU shape, 20 warmups and 100 CUDA-event calls:

| phase/mode | warm ms | first call ms | events | vs eager |
|---|---:|---:|---:|---:|
| prefill 19x24576 eager | 0.042966 | 35.25 | 6 | 0 |
| prefill official NV Inductor | 0.032915 | 1098.33 | 1 | +30.54% |
| prefill PyPTO Inductor | 0.210618 | 1696.24 | 1 | -79.60% |
| decode 1x24576 eager | 0.040543 | 32.81 | 6 | 0 |
| decode official NV Inductor | 0.033808 | 1045.90 | 1 | +19.92% |
| decode PyPTO Inductor | 0.212401 | 1733.25 | 1 | -80.91% |

Both compiled modes reduce six events to one (83.33% launch reduction).
PyPTO's first call is 54.44% longer than official NV Inductor for prefill and
65.72% longer for decode.

Aligned eight-start operator A/B (four starts per lane):

| case | PyPTO latency / stock |
|---|---:|
| SwiGLU decode / prefill | 23.52x / 23.47x |
| gate-up linear decode / prefill | 10.89x / 180.02x |
| down linear decode / prefill | 37.11x / 189.69x |
| FP32 LM head | 6.10x |

These are logically aligned microbenchmarks. They explain why the current
whole-model result is slower despite fewer pointwise launches; they are not a
claim that PyPTO is faster.

The requested ablation/breakdown images must be generated with GPT-Image-2.
OPENAI_API_KEY is absent in this environment, so assets remain
PENDING_GPT_IMAGE2 and no other model is substituted. Prompts and provenance
are recorded in state/evidence/gpt-image2-ablation-prompts-20260829.json.

## Repository Layout

~~~text
packages/pypto-kernels/             standalone handwritten PyPTO operators
packages/pypto-framework-plugins/   SGLang and TorchInductor plugins
.sources/pypto/                     locked PyPTO/TensorIR source worktree
vendor/                             bundles, patch series, source lock
benchmarks/release/                 workload/correctness/performance/profile contracts
tools/                              build, controlled-run, summary, and audit entry points
demo/pypto-lib/                     byte-for-byte article-demo import
state/evidence/                     small check-in-ready evidence sidecars
runs/                               local raw reports (not committed)
~~~

## Build and Test

Validated environment: Ubuntu 26.04 on WSL2, CPython 3.14.6, PyTorch
2.13.0+cu130, CUDA 13.3.73, SGLang 0.5.18, CMake 3.31.10, and Ninja 1.13.
Use 24 CPU build/test jobs; GPU tests are serial.

~~~bash
python3 tools/verify_source_release.py --replay-patches
python3 tools/bootstrap_release.py --jobs 24
python3 tools/bootstrap_release_environment.py
envs/pypto-release/bin/python tools/build_release.py --stage all --jobs 24
envs/pypto-release/bin/python tools/download_release_models.py --model all
envs/pypto-release/bin/python tools/run_operator_regression.py --stage all
~~~

Pinned identities: PyPTO c27629e993a52b47d41fb898c749279dce44221b (300
commits); TensorIR db41d0733eb73971ee03a74faca81d1af6e6aef7 (89 commits);
CUDA Tile af2417041cc939b87ef56d92cfdcf61737c5457e; LLVM
57109befac92811d2253109242ca6fa69c961fb2; SGLang
71de97b264b04dcd514cf904003028aefe9775c8.

Correctness:

~~~bash
envs/pypto-release/bin/python tools/run_transformers_semantic_oracle.py \
  --model-path models/Qwen3.5-9B \
  --output runs/semantic-oracle-qwen35-9b-chat-nonthinking.json
envs/pypto-release/bin/python tools/run_model_correctness.py all \
  --model-path models/Qwen3.5-9B \
  --semantic-oracle runs/semantic-oracle-qwen35-9b-chat-nonthinking.json
~~~

Performance-only entry points do not read logits/token/text and do not run
allclose, cosine, top-k, or tolerance checks:

~~~bash
envs/pypto-release/bin/python tools/run_performance_regression.py \
  --pair-matrix --model-path models/Qwen3.5-9B \
  --optimized-memory-mode matched
# After the pair passes, run the full three-lane matrix with optimized stock.
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

## Article Demos and Screenshots

The motivation is to let readers without Ascend hardware learn the PyPTO DSL
on an NVIDIA platform. The source article is:
[让 Python 写 NPU 算子所写即所得！华为昇腾开源 PyPTO-Lib，实现 Qwen3-14B 与 DeepSeek V4-Flash 全部算子！](https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ)

The article-time pypto-lib commit
6c292d30ccc787ee4e1fe61541fd3faec0dafa65 is imported byte-for-byte under
demo/pypto-lib/. SOURCE_MANIFEST.json locks 151 files and 66 entry points.

~~~bash
envs/pypto-release/bin/python tools/run_article_demo_matrix.py \
  --mode help --output runs/article-demo-matrix-help.json
envs/pypto-release/bin/python tools/run_article_demo.py \
  --demo examples/beginner/hello_world.py --platform a2a3sim \
  --output runs/article-demo-hello-world.json
~~~

The CLI/help audit passes for 57 runnable entries. Device execution remains
blocked by the Ascend simpler_setup runtime and the pl.KernelType.MIX API
difference; reports classify these as blockers rather than NVIDIA success.
The unchanged typical-demo and successful-9B PowerShell captures remain:

- PENDING_SCREENSHOT: docs/assets/screenshots/article-demo-typical.png
- PENDING_SCREENSHOT: docs/assets/screenshots/model-inference.png

Existing build/operator/performance purple-terminal captures are bound to their
2026-08-29 runs and do not replace current JSON evidence. This environment
cannot control Windows GUI, so it does not fabricate terminal images.

Fixed model prompt:

为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？

## Limits and License

Only SM120/RTX 5090 Laptop is validated. Shapes and strides are statically
specialized; the complete MLP is not fused across matmul; long GDN prefill is
token-ordered; PyPTO matmul/LM-head/launch overhead remains to be optimized.
The optimized lane, a resource-compliant four-start matched rerun, full-model
CUPTI/NVTX profile, full article-demo runtime, GPT-Image-2 assets, and new
PowerShell captures remain publication gates.
TensorIR is marked early release upstream.

PyPTO uses CANN Open Software License Agreement 2.0. TensorIR uses Apache 2.0
with LLVM Exceptions. The framework plugin uses Apache 2.0. See
packages/pypto-kernels/LICENSE_STATUS.md for the kernels authorization boundary.
