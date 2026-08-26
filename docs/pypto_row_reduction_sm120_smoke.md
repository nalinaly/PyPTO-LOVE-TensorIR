# PyPTO RowReductionV3 SM120 correctness gate

This separately controlled gate accepts only ten fixed `RowReductionV3` cases
through `HIR -> compile_structured_strict -> BuildSpec/Artifact/Cubin ->
NvidiaExecutable`. It is not a general reduction or performance claim.

The exact compiler source is PyPTO `62eb88251df5bdad95277a9d619d20da9bf121eb`,
tree `04d3bca3e0b35b796f7745ded27a26dd61e25c67`. The backend-ON DSO is
`builds/pypto-row-reduction-v3-on-62eb882-final/product/pypto_core.cpython-314-x86_64-linux-gnu.so`,
784,224,056 bytes, SHA-256 `e1213cf3...e4220`. The accepted CP48 compiler/Cubin
report remains immutable at SHA-256 `d06765be...38abb`.

## Fixed matrix

| Case | Dtype, shape | Op | Outer tile/grid | Contraction |
|---|---|---|---|---|
| `rank1_fp32_sum_n1` | FP32 `[1]` | sum | `1 / [1,1,1]` | `r=1`, 1 chunk |
| `rank1_fp32_sum_n7` | FP32 `[7]` | sum | `1 / [1,1,1]` | `r=1`, 7 chunks |
| `rank2_bf16_sum_n256_tail` | BF16 `[17,256]` | sum | `16 / [2,1,1]`, tail 1 | `r=128`, 2 chunks |
| `rank2_bf16_max_n17` | BF16 `[2,17]` | max | `2 / [1,1,1]` | `r=1`, 17 chunks |
| `rank2_fp32_max_n128_tail` | FP32 `[5,128]` | max | `4 / [2,1,1]`, tail 1 | `r=128`, 1 chunk |
| `rank2_fp32_max_n96_tail` | FP32 `[5,96]` | max | `4 / [2,1,1]`, tail 1 | `r=32`, 3 chunks |
| `rank2_bf16_sum_n129` | BF16 `[8,129]` | sum | `8 / [1,1,1]` | `r=1`, 129 chunks |
| `rank3_fp32_sum_n17_tail` | FP32 `[2,3,17]` | sum | `4 / [2,1,1]`, tail 2 | `r=1`, 17 chunks |
| `rank3_bf16_max_n17` | BF16 `[2,16,17]` | max | `16 / [2,1,1]` | `r=1`, 17 chunks |
| `rank3_fp32_max_n257_tail` | FP32 `[2,3,257]` | max | `4 / [2,1,1]`, tail 2 | `r=1`, 257 chunks |

Every Artifact is compiled once and reused by two fresh executable lifetimes:
20 launches, packet releases and terminal unloads in total. Four overlapping
cases join the exact CP48 TensorIR source and Cubin hashes.

The compile-anchor manifest binds two independent CUDA-hidden isolated runs,
`pypto-20260826T110849Z-17249-b18e99` and
`pypto-20260826T110905Z-17569-3de174`, including process/preflight sidecars
and 51 replay files each. Their normalized ten-case records are byte-identical.

## Numerical and memory contract

BF16 source is explicitly widened to FP32 before both sum and max, reduced in
FP32, then converted once to BF16. The N=129 and N=256 BF16 sum rows begin with
`1.0` followed by `2^-8`; the accepted outputs are respectively BF16 words
`0x3fc0` (1.5) and `0x4000` (2.0). Sequential BF16 accumulation produces the
negative-control word `0x3f80` (1.0) and is rejected.

Every output row in both repetitions has exactly one authoritative mode:
raw-word exact, finite/subnormal sum tolerance, or special class/sign. Rank-1
repetition-zero sums and every finite max are exact. The N=129/N=256 BF16
discriminator row is exact while their remaining non-dyadic rows are tolerant;
the rank-3 FP32 sum rows are tolerant. FP32 tolerance requires both at most 16
ULP and `rtol=2e-6, atol=0`; BF16 requires both at most 1 ULP and
`rtol=1/128, atol=0`. Infinity and signed zero preserve class/sign, while NaN
payload/sign are ignored but NaN class is mandatory. Subnormals never use the
special-class shortcut. The special matrix covers both dtypes and both
operators: max `+0 > -0`, all `-0 -> -0`, and sum all `-0 -> +0`, sole signed
infinity, mixed infinities and NaN propagation.

The compile anchors also freeze two independently generated input-byte hashes,
CPU-reference word vectors/hashes, output class/sign vectors and comparison
partitions per case. The runner and CPU-only finalizer each reconstruct and
join all 20 records, preventing a correlated runner/finalizer oracle error from
silently accepting itself.

Each input allocation has 4,096 guard elements on both sides. The anchored
lowering uses a 128-element reduction budget, `r=min(128, lowbit(N))`, and a
full materialized row tile. Exact address enumeration gives the worst suffix
span `(32-17)*256 = 3,840` elements for `[17,256]/T16`; index 3,839 is reachable
and index 3,840 is not. Each output has 16 guards per side; the worst exact
suffix span is 15 rows. Prefix/suffix and input/output roles use four distinct
exactly representable words in both FP32 and BF16. Before/after hashes prove no
writes into canaries and logical input immutability; the derived allocation
bound makes any lowered speculative read remain in-bounds. The finalizer
independently reconstructs every sentinel hash.

Torch reference computation uses a distinct non-default stream, widens BF16,
synchronizes before candidate coverage and stays outside that coverage. The
candidate launches on the selected non-default current stream, outside CUDA
Graph capture, synchronizes externally, releases its packet, then unloads with
zero terminal context identity. Triton, SGLang and FlashInfer providers and all
fallback are forbidden.

Finalization deserializes the replayed CompileRequest/BuildSpecs/Artifacts in a
separate CUDA-hidden process. Its byte, loader-compatibility and device-autotune
identity digests must match the provisional runtime and the ten anchored
BuildSpecs. The replay interpreter, PyPTO DSO and libcudart must also match their
contract size/SHA pins; co-mutating a live file and its provisional integrity
record is rejected.

## Admission, publication and claim boundary

The controller reuses the accepted policy-2 admission primitives by exact hash:
22 GiB protected-lane admission, 32 GiB exclusive admission, 16 GiB owned-run
abort and 4 GiB free-GPU floor. Both parent preflights, the independent child
gate, periodic watchdog and post-exit audit reject protected/external compute,
protected runtime mappings and unreadable maps. Only the verified owned process
group is eligible for the inherited stop primitive.

The implementation intentionally omits
`state/contracts/pypto_row_reduction_sm120_v1.json`. The controller and finalizer
therefore fail closed. After source review and CPU-only tests, publish that one
canonical file in a separately reviewed manifest-only commit, then run clean
post-manifest focused/full CPU gates before any GPU command.

After publication, the GPU command is:

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/run_pypto_row_reduction_sm120_isolated.py \
  --allow-protected-zero-nvidia-gpu-smoke \
  --run-id-file runs/next-pypto-row-reduction-sm120.json
```

The standard-library finalizer is a separate CPU-only no-replace transaction:

```bash
envs/pypto-nvidia/bin/python -E -B -S \
  tools/finalize_pypto_row_reduction_sm120.py \
  --workspace /home/zhaosiying/pypto-love-tensor-ir \
  --run-id <run-id> \
  --expected-provisional-sha256 <sha256>
```

This family does not claim general RowReductionV3 behavior, unlisted shapes or
special domains, performance, CUDA Graph, framework/model correctness, strict
coverage, or any reinterpretation of accepted CP47/CP48 files and reports.
