# PyPTO ♥ TensorIR: Qwen3.5 Inference with 100% PyPTO Kernels on an NVIDIA RTX 5090

[简体中文](README.md) | [English](README_EN.md)

> [!IMPORTANT]
> This repository is a **personal, non-profit** compiler research project. Running
> PyPTO on NVIDIA GPUs appears, on its face, to conflict with the CANN Open
> Software License Agreement Version 2.0, which restricts use and distribution to
> systems with Huawei AI processors; personal research and non-profit intent are
> not license exemptions. The motivation aligns with the vision expressed by
> Dr. Liao Heng, Huawei Chief Scientist, at approximately 2h34m of an interview
> with Zhang Xiaojun: that PyPTO should become a common frontend DSL for **all**
> AI chips. That public statement is not a license change and not authorization
> for this project. Rights holders who believe any content is inappropriate may
> contact the author via repository Issues for removal. Full analysis in
> [LEGAL_NOTICE.md](LEGAL_NOTICE.md).
> Interview: <https://www.bilibili.com/video/BV1nB3u6tERu/?vd_source=f2f41aa7b5e3cc8e0a23942779ccea11>

## What this is

A compiler implementation that brings [PyPTO](https://github.com/hw-native-sys/pypto)
(the kernel DSL frontend of the Huawei CANN ecosystem — its positioning is
analogous to Triton in the Ascend stack) to NVIDIA GPUs: **on an RTX 5090
(SM120), every GPU kernel in the Qwen3.5-9B inference forward pass is expressed
in the PyPTO frontend and produced by this repository's compilation pipeline**.
A per-kernel CUPTI audit measures 100% coverage (zero fallback), and outputs
match the native SGLang implementation token for token. It is also a learning
platform: the official PyPTO tutorial article's complete demo suite runs
**unmodified** on this pipeline (see [Article demos](#article-demos-run-the-official-tutorial-unmodified)).

Two core features:

1. **PyPTO → TensorIR bridge**: statically specialized PyPTO HIR (`@pl.jit`
   tile graphs) is lowered through typed TensorIR (MLIR ODS/OpBuilder) →
   CUDA Tile → `tileiras` into an `sm_120a` cubin, packaged as a validated,
   cacheable `NvidiaExecutable` launched on the caller's CUDA stream.
   The PyPTO fork carries 300 commits; the TensorIR fork carries 89
   (row gather/scatter layouts, multi-output fusion, a runtime-free artifact
   contract, and more).
2. **A PyPTO backend for TorchInductor**: standard
   `torch.compile(model, backend="pypto")`. It reuses the full official
   `compile_fx` and auto-generates fusible pointwise / trailing-axis-reduction
   subgraphs as PyPTO DSL kernels; unsupported cases fail closed and never
   silently fall back to Triton. In the real-shape Qwen3.5-9B SwiGLU operator
   ablation its fusion capability matches the official Triton backend:
   **6 eager kernels fuse into 1 (−83.33% launches)**; one cold compile costs
   **1.70 s** (official backend 1.10 s, 54–66% longer); the generated kernel's
   warm path is currently 79.6% slower than eager (the gap is in fused-pointwise
   code generation, isolated from scheduling by the tile-sweep experiment;
   the official backend is +30.5%).

```text
SGLang (unpatched)
  └─ pypto-kernels (handwritten operators) + TorchDynamo/TorchInductor (PyPTO backend)
       └─ PyPTO HIR → typed TensorIR ModuleOp → CUDA Tile → tileiras → sm_120a Cubin
            └─ PyPTO Artifact / NvidiaExecutable → caller-owned CUDA stream
```

![Architecture](docs/assets/pypto-nvidia-architecture.svg)

**Why TensorIR is the right host**: TensorIR is inherently tile-oriented —
layout propagation, tile selection, and multi-output fusion all operate on
tile graphs; it is not "a backend of cuTile" but rather **cuTile's
frontend/producer** (it owns tile layout and structured lowering, emitting
CUDA Tile IR for the `tileiras` assembler). The responsibilities therefore
compose precisely: PyPTO owns the DSL/static specialization/artifact
contract/launch, TensorIR owns tile layout and lowering, and cuTile owns final
code generation. PyPTO's tile conventions are passed in explicitly through
`CanonicalSchedule` and continue to be lowered — all 21 handwritten graphs plus
the Inductor-generated kernels compile through this path (101 correctness
cases and 29,728 whole-model compute launches with zero fallback attest to it).

On top of this, `packages/pypto-kernels` is a framework-independent handwritten
PyPTO operator library (13 modules / 21 `@pl.jit` graphs) covering every
operator of Qwen3.5's hybrid-attention topology (24 GDN layers + 8 full
attention layers): attention, GDN, RoPE, RMSNorm, causal conv, linear
projections, embedding, LM head; the MLP activation is auto-fused by the
Inductor backend. SGLang integrates through its official plugin mechanism —
**zero patches to SGLang itself**.

## Measured results (RTX 5090 Laptop 24 GiB; evidence frozen 2026-09)

| Item | Result |
|---|---|
| Qwen3.5-9B correctness | PASS: 3 fresh starts × 10 requests × 64 tokens, **token-identical** to native SGLang (unique output-sequence SHA-256) |
| model-forward PyPTO coverage | **100%**: 29,728 compute calls = 27,680 handwritten + 2,048 Inductor-generated, 0 fallback |
| Operator regression | PASS: 8 suites / 101 cases (re-run on hardware 2026-09-01) |
| End-to-end throughput | PyPTO **11.1635 tok/s**; SGLang matched **15.5813**; SGLang optimized **13.1459** |
| Relative performance | **PyPTO = 71.65% of matched** (95% CI [71.56%, 72.11%]); **84.92% of optimized** (CI [84.83%, 88.18%]) |
| vs previous release | 2.3393 tok/s (15.62%/18.71%) on the same matrix → **4.77× end-to-end**, entirely from scheduling-layer optimization with token-exact correctness preserved |
| GPU compute total | PyPTO forward compute **2,221.72 ms/request < optimized's 2,327.79**; the remaining E2E gap is entirely host-side launch residual (970.63 ms) |

| Metric (p50) | PyPTO | matched | optimized |
|---|---:|---:|---:|
| E2E | 5732.99 ms | 4107.48 ms | 4868.44 ms |
| TTFT | 351.02 ms | 70.26 ms | 145.48 ms |
| TPOT | 85.32 ms | 64.08 ms | 74.95 ms |
| Output throughput | 11.1635 tok/s | 15.5813 tok/s | 13.1459 tok/s |
| Peak GPU memory | 18.92 GiB | 18.70 GiB | 22.68 GiB |

![Three-lane end-to-end](docs/assets/charts/three-lane-end-to-end.png)

Lanes: **PyPTO** (all operators via PyPTO; zero CPU offload,
`mem_fraction_static=0.78`; CUDA graphs/overlap off — the correctness child
alone uses 0.80 with the completion-only GPU policy); **matched** (SGLang
default operators, `0.78 / 2 GiB offload`, CUDA graphs/overlap off);
**optimized** (official defaults with CUDA graphs/overlap on, `0.69 /
2 GiB offload`). Workload: chat-template 31 input tokens + greedy 64 output
tokens, BF16, TP1, concurrency 1; 4 fresh process starts per lane; the
headline is the median of per-start p50s with a 10,000-resample percentile
bootstrap CI. Machine-readable results:
`state/evidence/qwen35-9b-release-results-current.json`.

**Live-run screenshots** (real execution of `wsl -d Ubuntu` inside
PowerShell at native 3872×2312, showing each command's raw log; see
[Screenshot reproduction](#screenshot-reproduction)):

| Stage | Command | Screenshot |
|---|---|---|
| Build (raw ninja incremental-rebuild log) | `ninja -C builds/qwen35-sm120-v1/native` | ![ninja](docs/assets/screenshots/build-ninja-raw.png) |
| Build (raw CTest log, 13/13) | `ctest --test-dir builds/qwen35-sm120-v1/native` | ![ctest](docs/assets/screenshots/build-ctest-raw.png) |
| Operator correctness (8 suites, 101 cases) | `tools/run_operator_regression.py --stage all` | ![operator](docs/assets/screenshots/operator-correctness.png) |
| Operator performance A/B (4+4 fresh starts) | `tools/run_operator_performance.py --matrix` | ![perf](docs/assets/screenshots/operator-performance.png) |
| End-to-end inference (raw SGLang log, full 480-token generation) | `demo/raw_sglang_inference.py` | ![sglang](docs/assets/screenshots/sglang-raw-inference.png) |
| Correctness gate (fixed prompt, 64-token greedy + per-token gate) | `tools/run_model_correctness.py all` | ![model](docs/assets/screenshots/model-inference.png) |

## Performance optimization: schedule tiles and the launch path

Everything that moved the needle from 18.71% to **84.92%** (4.77×
end-to-end, in two rounds) happened **at the scheduling layer**, with the
token-level gate green at every step:

- **Locating the tile cliff**: the CUPTI per-kernel audit showed structured
  matmul at 94.6% of compute; the root cause was a 128-column schedule tile
  that gave the down-projection decode GEMV only **32 CTAs on ~170 SMs**
  (1.5% of bandwidth). `tools/sweep_linear_tiles.py` (checked in) compiles
  every candidate tile for every production shape and times it with CUDA
  events while bit-comparing outputs; **tile=32** is the universal optimum
  (a hard cliff exists at ≥64): gate/up decode 2.60→0.38 ms, down 4.53→0.22 ms,
  LM head 14.55→3.68 ms, prefill 45→8.5 ms.
- **Numerical invariance**: the tile only changes how output columns split
  across CTAs; the in-element K accumulation order is untouched — every tile's
  output was bit-identical in the sweep and the model-level token gate stayed
  green throughout. Prefill adds a `[2,32]` row block (two rows share one
  weight stream, another 2×) that is likewise bit-identical.
- **Experiments the gate rejected (recorded honestly)**: a `[16,32]` row block
  selects the tensor-core MMA path and is 14× faster in the microbenchmark,
  but its different FP accumulation order breaks token-level agreement —
  faster but wrong, reverted. CPU-offloading the candidate lane is
  incompatible with the PyPTO weight hooks (deterministically degraded
  output), disabled.
- **Attention tile + launch-packet caching**: attention value tiles 64→32;
  `launch_graph` reuses immutable launch packets (steady decode reuses the
  same buffers, so hits skip per-operand validation and C++ argument packing;
  kernels still read the caller's current buffers, and any new combination
  falls back to full validation).
- **An honest negative result on the Inductor side**: the same tile sweep on
  the generated fused-pointwise kernel came out **slower** (0.238 vs 0.211 ms
  at tile 32); 128 stays optimal. That kernel's gap to eager lives in the
  host launch path and tileiras code generation (SASS-verified scalar
  `LDG.E.U16` loads), listed as future work.
- **Shape generality (bucket retry ladder + sacrificial-subprocess probe)**:
  tileiras accepts scattered (bucket, table) pairs with no discernible
  pattern (bucket 336 rejected, 352 accepted, same geometry). Every new
  geometry compiles once in a dedicated-session subprocess (success publishes
  to the persistent cache, which the engine then hits; timeout kills the
  whole group), and a rejected bucket automatically retries at 64/128
  alignment — slots past the valid length are already masked to −inf inside
  the kernel, so numerics and ABI are unchanged. In both engine geometries
  tested, every 64-multiple bucket compiles.

Operator-level before/after (PyPTO / cuBLAS):

| Operator (shape) | Before | After |
|---|---:|---:|
| gate/up linear decode 1×4096×24576 | 10.9× | **1.19×** |
| down linear decode 1×12288×4096 | 36.9× | **1.34×** |
| FP32 LM head 1×4096×248320 | 6.0× | **1.51×** |
| gate/up linear prefill 31×4096×24576 | 179.2× | **14.4×** |
| down linear prefill 31×12288×4096 | 194.7× | **13.7×** |

![Operator A/B](docs/assets/charts/operator-ab-breakdown.png)

Round two (launch structure): the per-head decode launches merge into one
all-heads launch (16 launches → 1 at batch 1; the grid widens 16×; per-head
arithmetic and reduction order are unchanged). tileiras rejects the merged
graph — and on scattered bucket widths corrupts the compiling process heap —
so every new geometry is probed in a **sacrificial subprocess** (success
publishes the artifact to the persistent cache, which the parent then hits;
any failure falls back to the per-head path). Disassembly plus CUPTI also
proved the SwiGLU kernel itself runs in ~1 µs (the measured 0.21 ms was
nine-tenths host launch cost), so the Inductor wrapper's `pypto_launch`
gained the same packet caching as the handwritten path.

| Metric | Before | Round 1 | Round 2 (final) |
|---|---:|---:|---:|
| Output throughput | 2.3393 tok/s | 9.6068 | **11.1635 (4.77×)** |
| = of optimized | 18.71% | 72.95% | **84.92%** |
| TPOT | 381.23 ms | 100.19 | **85.32 (4.5×)** |
| Attention / request | 1154 ms | 1203 ms | **268 ms** |
| Forward compute / request | 22318.81 ms | 3066.64 | **2221.72 (10.0×)** |

CUPTI phase attribution (GPU ms per request): attention 1154→268 (4.3× from
the merged launch), unattributed (handwritten linears) 20184→1710, LM head
932→198; **PyPTO forward compute (2221.72) is now below the optimized lane
(2327.79)** — of the 864.55 ms E2E gap versus optimized, the entire remainder
is the 970.63 ms host-side non-profiled residual, with the independent phase
reconciliation residual at −5.24 ms (closed).

## Environment requirements

| Component | Version (pinned) |
|---|---|
| GPU | NVIDIA RTX 5090 (SM120 / compute capability 12.0, 24 GiB) |
| OS | Ubuntu 26.04 (WSL2 works; all measurements were taken on WSL2) |
| CUDA toolkit | 13.3 (ships the `tileiras` assembler; verified by path + SHA-256 + version before build) |
| Python | CPython 3.14.6 |
| PyTorch | 2.13.0+cu130 |
| SGLang | 0.5.18 (`71de97b`, unmodified) |
| Build | CMake 3.31, Ninja, C++ toolchain (`--jobs 24` is contractual) |

Python dependencies are lock-driven with full hashes:
`environment/conda-linux-64.lock`, `environment/python-requirements.lock`;
runtime lock: `environment/release-runtime.json`. Source identity is pinned by
`vendor/source-lock.json` + git bundles + 391 patches.

## Quick start

Run from the repository root; steps 1–4 are one-time.

```bash
# 0. (optional) Verify the three upstream trees match bundles+patches byte-for-byte
python3 tools/verify_source_release.py --replay-patches

# 1. Materialize upstream trees: .sources/{pypto,tensor-ir,sglang} from vendor bundles
python3 tools/bootstrap_release.py --jobs 24

# 2. Create the formal release environment envs/pypto-release (lock-driven)
python3 tools/bootstrap_release_environment.py

# 3. Four-stage build: wheels → native (CMake/Ninja with TensorIR as a private subproject)
#    → CTest (13/13) → install
envs/pypto-release/bin/python tools/build_release.py --stage all --jobs 24

# 4. Download models (Qwen3.5-0.8B / 9B → models/, SHA-256 checked against the manifest)
envs/pypto-release/bin/python tools/download_release_models.py --model all
```

The build produces three wheels (`pypto`, `pypto-kernels`, `pypto-framework-plugins`)
installed into the release environment; raw logs and wheel manifests land in
`runs/<run-id>/`.

## Operator correctness regression (checked-in regression test)

```bash
envs/pypto-release/bin/python tools/run_operator_regression.py --stage all
```

Structure gate (`pytest -n24` static contract checks) + 8 GPU suites / 101
cases: compile classification, numerical correctness, stateful operators at
real model shapes (multi-token GDN/conv), paged attention, QK-norm+RoPE,
linear/LM head, CUDA-graph lifecycle, and Inductor SwiGLU. Pass criteria: all
suites `passed: true` and `all_correct: true` (re-run and passed on hardware
2026-09-01; report at `runs/<run-id>/operator-numerical-regression.json`).

## Performance regression (performance-only, includes baseline comparison; checked in)

```bash
# ① Operator-level A/B: 7 aligned operators, PyPTO vs SGLang stock, 4+4 fresh starts
envs/pypto-release/bin/python tools/run_operator_performance.py --matrix --model-path models/Qwen3.5-9B

# ② Whole-model three-lane matrix: pypto / sglang-matched / sglang-optimized, 3×4 fresh starts
envs/pypto-release/bin/python tools/run_performance_regression.py --matrix \
    --model-path models/Qwen3.5-9B --optimized-memory-mode matched

# ③ Tile sweep (optimization tool: per-tile real compilation + timing + bit comparison)
envs/pypto-release/bin/python tools/sweep_linear_tiles.py --output runs/linear-tile-sweep.json

# ④ SwiGLU fusion ablation: eager vs official Inductor (CUDA/Triton) vs PyPTO backend
envs/pypto-release/bin/python tools/run_inductor_ablation.py --mode pypto --phase prefill \
    --output runs/ablation-prefill-pypto.json
envs/pypto-release/bin/python tools/summarize_inductor_ablation.py \
    --prefill-eager runs/ablation-prefill-eager.json \
    --prefill-inductor-nv runs/ablation-prefill-nv.json \
    --prefill-pypto runs/ablation-prefill-pypto.json \
    --decode-eager runs/ablation-decode-eager.json \
    --decode-inductor-nv runs/ablation-decode-nv.json \
    --decode-pypto runs/ablation-decode-pypto.json \
    --output state/evidence/qwen35-9b-inductor-ablation-current.json
```

Timing uses CUDA events; each case runs 20 warmups + 30 batches × 100 calls.
After a re-run, `python3 tools/print_operator_ab_table.py` prints the newest
A/B table. The formal frozen results live in
`state/evidence/qwen35-9b-operator-performance-breakdown-current.json` and
`qwen35-9b-inductor-ablation-current.json`.

## End-to-end inference: 100% PyPTO Qwen3.5-9B (fixed-prompt reproduction)

```bash
# ① One-time: generate the HuggingFace transformers semantic oracle (first-step logits top-k)
envs/pypto-release/bin/python tools/run_transformers_semantic_oracle.py \
    --model-path models/Qwen3.5-9B --device cuda \
    --output runs/semantic-oracle-qwen35-9b-chat-nonthinking.json

# ② End-to-end correctness: stock reference + 3 fresh candidate starts,
#    per-token ID gate + coverage audit
envs/pypto-release/bin/python tools/run_model_correctness.py all \
    --model-path models/Qwen3.5-9B \
    --semantic-oracle runs/semantic-oracle-qwen35-9b-chat-nonthinking.json
```

Fixed prompt (chat template, non-thinking; 31 input tokens, greedy 64 output):

```text
为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？
```

Expected output (within the 64-token cap, token-identical to native SGLang):

```text
关于“鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作”这一说法，**目前并不存在客观依据，
这很可能是一个网络误传、营销号夸大其词，或者是将其他作品的评价张冠李戴了**。

事实上，《月鳞绮纪》（原名
```

Pass criteria: all 30 requests token-identical (unique output-sequence SHA-256),
teacher-forced frozen-logits policy passes, and CUPTI coverage 29,728/29,728
with 0 fallback. After a re-run, `python3 tools/print_model_gate_live.py`
prints the latest run's pass status, coverage audit, and generated output.

## Article demos: run the official tutorial unmodified

The complete demo suite from the official PyPTO tutorial
[article](https://mp.weixin.qq.com/s/7tLlTbomH9OqyUbZDbBEhQ) (151 files /
66 entrypoints) is checked in **byte-for-byte** under `demo/pypto-lib/`
(upstream revision `6c292d30`) and runs unmodified on this pipeline: all 41
compute entrypoints pass (1 strict PyPTO→TensorIR→CUDA-Tile `hello_world`
with `max_abs_diff=0.0`; 40 CUDA numerical references); 17 Ascend-hardware
entrypoints are skipped honestly. Run the whole corpus at once:

```bash
envs/pypto-release/bin/python tools/run_article_demo_matrix.py --backend nvidia --mode run --device 0
```

The shortest single-file demo is `demo/raw_hello_world.py` — a self-contained
PyPTO DSL source run directly via
`envs/pypto-release/bin/python demo/raw_hello_world.py`; it prints the fully
specialized HIR, the compiled artifact state, and the elementwise comparison:

![hello_world raw run](docs/assets/screenshots/hello-world-raw.png)

(Single-demo entry for the tutorial corpus:
`envs/pypto-release/bin/python -B tools/run_article_demo_nvidia.py
--demo examples/beginner/hello_world.py --device 0`.)

## Repository layout

```text
packages/pypto-kernels/            # handwritten PyPTO operator library (13 modules / 21 graphs)
packages/pypto-framework-plugins/  # torch.compile(backend="pypto") backend + SGLang plugin
.sources/pypto                     # PyPTO fork (+300 commits: NVIDIA target, compile contracts, runtime)
.sources/tensor-ir                 # TensorIR fork (+89 commits: gather/scatter layouts, artifact contract)
.sources/sglang                    # SGLang v0.5.18 (unmodified)
vendor/                            # source identity locks: git bundles, 391 patches, source-lock.json
tools/                             # all entry scripts: build / correctness / performance / tile sweep / inference
benchmarks/release/                # lanes, workload, operator manifests (frozen test contracts)
state/evidence/                    # frozen measurement-evidence JSONs
docs/assets/                       # architecture figure, screenshots, charts
models/ tests/ demo/               # models / harness tests / article-demo corpus
envs/ builds/ runs/ caches/        # environments / build artifacts / raw run outputs (gitignored)
```

## Implementation summary

**PyPTO side** (`.sources/pypto`): NVIDIA target identity and SM120 traits; an
immutable `CompileRequest v1` whose `ToolchainIdentity` hashes the
pypto/tensor-ir/cuda-tile/LLVM/CUDA-toolkit/tileiras revisions; a
`CanonicalSchedule`/`KernelBuildSpec` pipeline behind the strict deterministic
facade `compile_structured_strict[_cached]` (canonical MessagePack — identical
inputs produce identical artifacts); `codegen/nvidia/tensor_ir_codegen.cpp`
analyzes the strict HIR subset into a deterministic TensorIrModule, and
`typed_tensor_ir_module_spec.h` + `compiler/nvidia_typed_tensor_ir_builder.cpp`
build a typed `mlir::ModuleOp` via ODS/OpBuilder (no string-built IR — enforced
by lint); the serializable `Artifact v1` (cubin + full kernel ABI) with an
in-process cache; a driver-only runtime `NvidiaExecutable` (dlopens `libcuda`
only, `PrepareLaunch` builds an immutable launch packet fired on the caller's
stream, with a CUDA-graph lease); `tileiras` verified by path/SHA/version then
run in a restricted subprocess; JIT specialization preserves per-element
strides (row-pitched layouts, zero-stride broadcasts).

**TensorIR side** (`.sources/tensor-ir`): row gather/scatter layout lowering
(required by paged KV, embedding, and the GDN state pool); lazy input loads
with one-shot layout conversion; compatible multi-output fusion; residual
graph-splitting fixes; the runtime-free
`CudaTilePreparedArtifact/CompiledArtifact` contract with CUDA-free cubin
validation; `tileiras` subprocess hardening (closed stdio, explicit timeout,
scratch quotas); plus lowering-correctness work (unit-dim matmul, reduction
slicing, zero-stride broadcast, and more).

**Inductor backend** (`packages/pypto-framework-plugins`): an entry-point
`pypto` backend that internally runs the full official `compile_fx`
(`cuda_backend="pypto"`, `implicit_fallbacks=False`, strict Dynamo failures);
reversibly swaps the Inductor CUDA scheduler and wrapper (restores Triton
outside the context) and pins the Triton hash out of cache identity; accepts
only Pointwise and trailing-axis Reduction nodes (everything else raises
`StrictCoverageError`); replays pointwise bodies through an ops recorder and
**emits literal `@pl.jit` DSL source**, specializes with stride-exact meta
tensors, and compiles via `compile_structured_strict_cached`; the generated
wrapper calls a single `pypto_launch(...)` that validates the ABI and launches
on the caller's stream; CUPTI coverage-audit tooling ships with the package.

**Handwritten operator library** (`packages/pypto-kernels`): one operator =
one `@pl.jit` graph; shape/stride ABI validation before every launch (steady
state hits the launch-packet cache); shares the same compilation path as the
Inductor backend. Inventory: attention (dense/masked/paged-decode — a single
all-heads launch at batch 1, guarded by the sacrificial-subprocess probe
against tileiras compile crashes on scattered bucket widths —/
paged-prefill/KV-write/gather — 7 graphs), GDN recurrent (L2 norm + softplus
decay gating + rank-1 state update + output projection over an `pl.InOut`
state pool), packed GDN projection, width-4 causal conv, QK RMSNorm + partial
RoPE + gate split, BF16 linear projections, FP32 LM head, embedding/token-id
gather, three RMSNorm variants, NeoX RoPE, sigmoid gating, and SwiGLU.

**SGLang integration** (zero patches): the official plugin registry wraps
~20 internal call sites via AROUND hooks, dispatching to `pypto-kernels`
operators; registers `--attention-backend pypto` and the GDN backend;
registration failure raises `SystemExit` (SGLang's loader swallows exceptions).

## Performance details and attribution

**Operator-level A/B** (PyPTO p50 / stock p50, ms/call):

| Operator (shape) | PyPTO | stock | Multiple | Before |
|---|---:|---:|---:|---:|
| gate/up linear decode 1×4096×24576 | 0.2859 | 0.2392 | **1.20×** | 10.9× |
| down linear decode 1×12288×4096 | 0.1649 | 0.1232 | **1.34×** | 36.9× |
| FP32 LM head 1×4096×248320 | 2.6728 | 2.4276 | **1.10×** | 6.0× |
| gate/up linear prefill 31×4096×24576 | 3.6025 | 0.2496 | 14.4× | 179.2× |
| down linear prefill 31×12288×4096 | 1.7528 | 0.1293 | 13.6× | 194.7× |
| SwiGLU decode 1×24576 | 0.2029 | 0.0093 | 21.9× | 21.7× |
| SwiGLU prefill 31×24576 | 0.2028 | 0.0093 | 21.8× | 22.2× |

**SwiGLU fusion ablation** (warm call ms / cold compile ms / kernel count /
vs eager):

| Shape · mode | Warm call | Cold compile | Kernels | vs eager |
|---|---:|---:|---:|---:|
| prefill 19×24576 · eager | 0.042966 | 35.3 | 6 | — |
| prefill · official Inductor (Triton) | 0.032915 | 1098.3 | 1 | +30.5% |
| prefill · PyPTO backend | 0.210618 | 1696.2 | 1 | −79.6% |
| decode 1×24576 · eager | 0.040543 | 32.8 | 6 | — |
| decode · official Inductor (Triton) | 0.033808 | 1045.9 | 1 | +19.9% |
| decode · PyPTO backend | 0.212401 | 1733.3 | 1 | −80.9% |

![SwiGLU ablation](docs/assets/charts/inductor-swiglu-ablation.png)

**CUPTI logical-phase attribution** (p50 ms/request): total PyPTO forward
compute **2,221.72** (was 22,318.81, 10.0×), of which the unattributed
bucket (handwritten linears) 1,709.89, attention core+gate 268.37, LM head
198.39; matched 1,337.55; optimized 2,327.79. Of the 864.55 ms E2E gap
versus optimized the compute gap is **−106.07 ms** (PyPTO is lower) — the
entire remainder is the 970.63 ms host-side non-profiled residual, with the
independent phase reconciliation residual at −5.24 ms (closed).

![CUPTI attribution](docs/assets/charts/cupti-phase-attribution.png)

**Conclusion**: decode linear algebra now sits at 1.1–1.3× of cuBLAS and the
merged launch drops attention to 268 ms/request (4.3×); **PyPTO's total
forward GPU compute is already below the optimized lane's** — the remaining
E2E gap is entirely host-side launch residual, pointing next at whole-step
CUDA-graph capture (the graph lease already exists) and numerically safe
prefill row-blocking.

## Screenshot reproduction

The screenshots are live captures produced by
`tools/windows/capture_powershell.ps1`: DPI-aware capture forced to the full
work area (native 3872×2312); a PowerShell prompt runs `wsl -d Ubuntu`, the
Ubuntu shell runs the command, and `PrintWindow` captures the frame on
completion. From PowerShell:

```powershell
$Repo = "\\wsl.localhost\Ubuntu\home\<user>\pypto-love-tensor-ir"
& "$Repo\tools\windows\capture_powershell.ps1" -Title "build-release" `
  -LinuxCommand "envs/pypto-release/bin/python tools/build_release.py --stage all --jobs 24 2>&1 | tail -1 | python3 -m json.tool" `
  -OutputPath "$Repo\docs\assets\screenshots\build-release.png" `
  -MetadataPath "$Repo\state\evidence\build-release-capture-current.json" `
  -Workspace "/home/<user>/pypto-love-tensor-ir"
```

Each capture writes a sidecar JSON (command, exit code, window size, PNG
SHA-256, non-blank pixel samples); the binding manifest is regenerated with
`python3 tools/generate_article_screenshot_manifest.py`.

## Limitations and licensing

- Host-side launch path (largest remaining item, 970.63 ms/request): the
  per-step launch/orchestration overhead of ~430 launches needs whole-step
  CUDA-graph capture (the lease is ready) and launch batching;
- Prefill GEMMs at 13.6–14.4×: more aggressive row blocking selects the
  tensor-core path, changes FP accumulation order, and breaks token-level
  agreement — numerically safe multi-row tiles are the follow-up;
- Fused pointwise: CUPTI proves the kernel itself runs in ~1 µs — the
  measured 0.20 ms is mostly per-call host launch cost (the Inductor
  wrapper's Python overhead); the scalar `LDG.E.U16` loads (SASS-verified)
  would cap wider workloads, a tileiras vectorization feedback item;
- Decode launch density: ~430 launches/step of host overhead remain after
  packet caching and the merged launch (see the item above); it shares the
  same whole-step CUDA-graph TODO;
- A zero-offload 9B candidate sits near the ceiling of a 24 GiB consumer card
  shared with the display; the correctness child uses the completion-only GPU
  policy, and CPU offload is incompatible with the PyPTO weight hooks;
- The Inductor backend fuses pointwise/trailing-axis reductions only;
  GEMM+epilogue and cross-node fusion are fail-closed TODOs;
- Long GDN prefills launch token-by-token in order; every kernel is statically
  specialized per shape/stride (artifacts persist on disk and are reused);
- Licensing: the PyPTO/CANN license does not permit running on or
  redistributing to non-Huawei processors (see the IMPORTANT note above and
  [LEGAL_NOTICE.md](LEGAL_NOTICE.md)); TensorIR is Apache-2.0 with LLVM
  exceptions; `pypto-framework-plugins` is Apache-2.0; `pypto-kernels` does
  not yet declare an independent license.
