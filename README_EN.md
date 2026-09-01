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

<!-- RELEASE_RESULTS:SUMMARY_BEGIN -->

| Item | Qwen3.5-9B release-v1 |
|---|---:|
| 64-token greedy correctness | PASS (3 fresh starts, 30 requests) |
| model-forward PyPTO compute coverage | 100% |
| Operator regression | PASS (8 suites, 101 cases) |
| PyPTO / matched SGLang | 15.62% |
| PyPTO / optimized SGLang | 18.71% |
| Performance bottleneck attribution | CUPTI/NVTX reconciliation complete |

![Ubuntu/PowerShell purple terminal: replay of the accepted wheel, native, CTest 13/13, and install gates; the four reports share one wheel artifact set.](docs/assets/screenshots/build-ctest.png)

![Ubuntu/PowerShell purple terminal: replay of 8/8 operator suites, 101 cases, and the structure gate for the current DSO; explicitly not a live GPU view.](docs/assets/screenshots/operator-correctness.png)

![Ubuntu/PowerShell purple terminal: replay of the 64-token output and 100% PyPTO coverage for the same prompt; current-identity evidence is one stock reference plus three fresh candidate starts.](docs/assets/screenshots/model-inference.png)

![Ubuntu/PowerShell purple terminal: operator-level SwiGLU fusion ablation; the adjacent whole-model table comes from the current 12-start three-lane matrix, and the screenshot itself is not a whole-model result.](docs/assets/screenshots/performance-ablation.png)

<!-- RELEASE_RESULTS:SUMMARY_END -->

Platform: NVIDIA GeForce RTX 5090 Laptop GPU (SM120, 24 GiB). Workload:
Qwen3.5-9B text-only, non-thinking chat template, 31 input tokens, 64 greedy
output tokens, BF16, TP1, concurrency 1.

| Item | Measured status |
|---|---|
| 9B correctness/coverage | Three fresh starts on the current wheel completed 10/10 Engine requests and one strict teacher-forced trace each |
| Model-forward coverage | Accepted traces: 33,448 compute calls = 31,400 handwritten PyPTO + 2,048 Inductor PyPTO, with zero unknown/fallback calls (100%) |
| 0.8B current wheel | Stock reference plus three candidate fresh starts completed; each trace covers 22,108/22,108 calls = 20,572 handwritten + 1,536 Inductor |
| PyPTO output throughput | **2.3393 tok/s** (median of four fresh starts) |
| Matched SGLang | **14.9754 tok/s** (same workload, CUDA Graph/overlap disabled) |
| Optimized SGLang | **12.5000 tok/s** (official Inductor + CUDA Graph + overlap) |
| PyPTO / matched | **15.6208%**, 95% bootstrap CI **[15.5862%, 15.7022%]**; signed change **-84.3792%** |
| PyPTO / optimized | **18.7143%**, 95% bootstrap CI **[18.6881%, 18.7533%]**; signed change **-81.2857%** |
| Matrix acceptance | 12/12 starts completed; PyPTO/matched retain the 4 GiB GPU-free floor; optimized has no fixed GPU-free floor and is accepted only on complete execution, no OOM/crash, and intact controls |
| Cold/compile-trigger | PyPTO `15725.20/29025.66 ms`; matched `14895.79/24243.59 ms`; optimized `122555.13/11921.59 ms`; optimized cold includes Inductor/CUDA Graph capture and the first request is still not compiler-only |

The previous 4096 MiB-floor failure and memory probes remain in
`state/evidence/optimized-lane-diagnostic-current.json` as historical diagnostics.
The accepted optimized configuration remains `cpu_offload_gb=2,
mem_fraction_static=0.69`; only its fixed runtime GPU-free floor was removed by
explicit user authorization. Foreign-process isolation, the 12 GiB host floor,
telemetry, timeout, thermal, OOM/exit-code, and natural-cleanup gates remain.

Artifact-union versus model-forward compute intersection for accepted traces:

| model | total calls | handwritten | Inductor | unknown/fallback | coverage |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 22,108 | 20,572 | 1,536 | 0 | 100% |
| Qwen3.5-9B | 33,448 | 31,400 | 2,048 | 0 | 100% |

Formal four-start three-lane medians (each start first reduced to a ten-request p50):

| lane | E2E | TTFT | TPOT | output tok/s | cold engine | first compile-trigger request |
|---|---:|---:|---:|---:|---:|---:|
| PyPTO | 27358.76 ms | 3343.10 ms | 381.23 ms | 2.3393 | 15725.20 ms | 29025.66 ms |
| matched | 4273.67 ms | 73.21 ms | 66.66 ms | 14.9754 | 14895.79 ms | 24243.59 ms |
| optimized | 5119.99 ms | 151.39 ms | 78.86 ms | 12.5000 | 122555.13 ms | 11921.59 ms |

The first compile-trigger request includes compilation and one complete 31+64
request; it is not compiler-only time. Minimum GPU-free memory was
`6,872,449,024`, `6,226,878,464`, and `1,940,078,592 B` for PyPTO, matched,
and optimized respectively. All 12 starts passed the 12 GiB host gate without
thermal throttling. PyPTO/matched used `cpu_offload_gb=2,
mem_fraction_static=0.78`; optimized used `2/0.69` and records
`gpu_free_floor_mode=disabled-completion-only`. The previous invalidated pair is retained at
`state/evidence/qwen35-9b-performance-pair-invalidated-20260830.json` and must
not be mixed with the current headline. Historical qualification is recorded
in `state/evidence/matched-performance-qualification-current.json`. The
controller stops its own run if a protected workload appears and never signals
an external process. The
matched lane keeps
enable_torch_compile=true but disables CUDA graphs, so this pinned SGLang
version does not invoke the global CompilerInterface; the report records
backend_invocation_observed=false. The PyPTO SwiGLU hook observes two generated
Inductor wrappers containing pypto_launch.

Machine-readable sources (including the two model-gate sidecars):

- state/evidence/qwen35-9b-release-results-current.json
- state/evidence/qwen35-0.8b-model-gate-current.json
- state/evidence/qwen35-9b-model-gate-current.json
- state/evidence/qwen35-9b-operator-performance-breakdown-current.json
- state/evidence/qwen35-9b-inductor-ablation-current.json
- state/evidence/qwen35-9b-inductor-source-current.json

A historical full-model eager control (15.3418 tok/s) and its historical matched
compile-request control both failed to invoke CompilerInterface, so they do not
establish a causal whole-model compile speedup. That boundary is recorded in
state/evidence/qwen35-9b-eager-compile-ablation-current.json. The supported
eager/NV-Inductor/PyPTO causal comparison remains the fixed SwiGLU ablation below.

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
| SwiGLU decode / prefill | 21.70x / 22.22x |
| gate-up linear decode / prefill | 10.87x / 179.19x |
| down linear decode / prefill | 36.87x / 194.69x |
| FP32 LM head | 6.00x |

These are logically aligned microbenchmarks. They explain why the current
whole-model result is slower despite fewer pointwise launches; they are not a
claim that PyPTO is faster.

### Whole-model CUPTI/NVTX hybrid breakdown

Each lane has three fresh starts, five requests per start, and 64 nonempty
`ModelRunner.forward` windows per request. PyPTO and optimized prove compiled
execution; every optimized start also observes 315/315 `cudaGraphLaunch`
callbacks. Matched does not invoke CompilerInterface because its frozen
configuration disables CUDA Graphs, so it remains a descriptive stock control,
not an Inductor speedup result.

| forward compute (GPU activity duration per request) | PyPTO | matched descriptive | optimized compiled |
|---|---:|---:|---:|
| all compute | 22318.812 ms | 1285.792 ms | 2131.558 ms |
| `unattributed_compute` | 20184.082 ms | 435.080 ms | 1830.714 ms |
| `attention_core_gate` | 1154.360 ms | 3.228 ms | 3.889 ms |
| `lm_head` | 931.866 ms | 0 (no same-name correlation) | 0 (no same-name correlation) |

Versus optimized, PyPTO has a `22238.774 ms` E2E gap, a `20187.254 ms`
profiled GPU-compute gap, and a `2051.520 ms` non-profile residual. Independent
phase medians produce a `-9.109 ms` reconciliation residual. Unattributed or
zero values describe attribution coverage, not operator non-execution. Full
phase CIs, graph-replay counts, raw-trace hashes, resources, and identity are in
[`qwen35-9b-release-results-current.json`](state/evidence/qwen35-9b-release-results-current.json).
Rebuild the complete hybrid matrix with:

```bash
envs/pypto-release/bin/python tools/profile_qwen35.py matrix \
  --model-path models/Qwen3.5-9B --optimized-memory-mode matched \
  --allow-noncompiled-matched \
  --performance-matrix runs/release-performance-matrix-<id>/summary.json
```

The five requested ablation/breakdown images must use GPT-Image-2. This process
still has no `OPENAI_API_KEY`, so all assets remain `PENDING_GPT_IMAGE2`; no
other model is substituted. Prompts, source hashes, and output paths are in
state/evidence/gpt-image2-ablation-prompts-20260829.json. After generation and
per-image visual review, `tools/finalize_gpt_image2_assets.py` validates PNG,
prompt/source/image hashes, and manual-review fields before writing the final
provenance manifest.

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
`demo/pypto-lib/`. `SOURCE_MANIFEST.json` locks 151 files and 66 entry points;
the external
[`article-demo-compatibility-policy-current.json`](state/evidence/article-demo-compatibility-policy-current.json)
records hardware evidence and the NVIDIA compatibility mode for each entry.

Generate the policy, then run the NVIDIA computational matrix:

~~~bash
python3 tools/classify_article_demos.py
envs/pypto-release/bin/python tools/run_article_demo_matrix.py \
  --backend nvidia --mode run --device 0 \
  --output state/evidence/article-demo-matrix-nvidia-current.json
~~~

The current RTX 5090 result passes all 41 computational entries: 40 independent
CUDA numerical references plus one strict PyPTO -> TensorIR -> CUDA Tile artifact
for `hello_world.py`. Seventeen communication, CCE, NPU/ACL, or Ascend-runtime
hardware-API entries are skipped under the supported boundary, and eight drafts
remain provenance-only. `computational_unmapped_count=0`;
`compatibility_status=complete`, `hardware_api_evidence`, and the before/after manifest
hashes are the release gate.

The 40 CUDA references cover nine teaching examples, two Qwen3 sampling
entries, and 29 DeepSeek V4 compressor/indexer/sparse-attention/HC/MoE/decode/
prefill computations. Every child report records per-output tolerances and
error counts.

Typical strict computational entry:

~~~bash
envs/pypto-release/bin/python tools/run_article_demo_nvidia.py \
  --demo examples/beginner/hello_world.py --device 0 \
  --output runs/article-demo-hello-nvidia.json
~~~

The terminal prints `strict-pypto-nvidia`, `golden_pass=True`, the artifact name,
and `fallback_used=False`. The report binds imported-source and policy hashes,
artifact/cubin hashes, and the 128-element tile; the upstream file is never
rewritten. The other 40 computational reports explicitly set
`strict_compiler_evidence=false`: they are independent CUDA Torch numerical
references for studying computational semantics, not strict PyPTO compiler
evidence.

To reproduce the article's original CLI/help, or run unchanged source on an
authorized Ascend runtime, use:

~~~bash
envs/pypto-release/bin/python tools/run_article_demo_matrix.py \
  --backend ascend --mode help --output runs/article-demo-matrix-help.json
envs/pypto-release/bin/python tools/run_article_demo.py \
  --demo examples/beginner/hello_world.py --platform a2a3sim \
  --output runs/article-demo-hello-world.json
~~~

The `--backend ascend` device stage is blocked here by
`simpler_setup`/`KernelType.MIX`; that blocker is not a precision pass.
Distributed hardware, CCE, NPU/ACL, and simpler-runtime entries are explicitly
skipped in the NVIDIA matrix. The unchanged `hello_world.py` has now run in a
real Windows Terminal Ubuntu/PowerShell-purple window with `exit_code=0`; the
screen shows `strict-pypto-nvidia`, `golden_pass=True`, `fallback_used=False`,
and `max_abs_diff=0.0`. The screenshot and window metadata are bound by
[`article-demo-screenshot-manifest-current.json`](state/evidence/article-demo-screenshot-manifest-current.json);
the PNG is 1549×925. A failed black-frame attempt was not accepted.

Windows Terminal GUI capture now passes a nonblank-pixel smoke. The performance
role was recaptured from the current immutable ablation JSON and is bound to
window dimensions, command, timestamps, and PNG SHA. The model role strictly
validates and replays the three accepted 9B runs, and explicitly says it is not
a live rerun. Build covers wheel build, install/pip-check, and CTest 13/13;
operator similarly validates and replays its accepted raw evidence. All five
completed roles bind command source, numerical/run evidence, capture metadata,
and PNG SHA. The computational matrix and GUI demo role are complete.

![Wheel build, install, and CTest evidence replay](docs/assets/screenshots/build-ctest.png)

![Accepted 8/8 operator regression evidence replay](docs/assets/screenshots/operator-correctness.png)

![Accepted Qwen3.5-9B inference evidence replay](docs/assets/screenshots/model-inference.png)

![Qwen3.5-9B operator-level SwiGLU ablation](docs/assets/screenshots/performance-ablation.png)

![Typical hello_world.py strict NVIDIA demo](docs/assets/screenshots/article-demo-typical.png)

Fixed model prompt:

为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？

## Limits and License

Only SM120/RTX 5090 Laptop is validated. Shapes and strides are statically
specialized; the complete MLP is not fused across matmul; long GDN prefill is
token-ordered; PyPTO matmul/LM-head/launch overhead remains to be optimized.
The three-lane performance matrix and hybrid CUPTI/NVTX profile are accepted.
Matched is a noncompiled descriptive control; PyPTO and optimized are strict
compiled lanes. Only GPT-Image-2 assets and the final document audit remain
publication gates.
TensorIR is marked early release upstream.

PyPTO uses CANN Open Software License Agreement 2.0. TensorIR uses Apache 2.0
with LLVM Exceptions. The framework plugin uses Apache 2.0. See
packages/pypto-kernels/LICENSE_STATUS.md for the kernels authorization boundary.
The historical pair retains status `invalidated-resource-and-control`; current
formal results are in `qwen35-9b-release-results-current.json`.
