# CP-0038 finalized NvidiaExecutable runtime evidence review

## Decision

**GO to accept the minimal real-SM120 `NvidiaExecutable` runtime milestone.**

P0: 0. P1: 0. P2: 0.

This decision is limited to the finalized static, dynamic-stride, and
tensor-plus-scalar TensorIR correctness smoke and six exact
`NvidiaExecutable` module/current-stream lifetimes. It is not general
operator-family acceptance and does not accept CUDA Graph, frontend HIR
lowering, production matmul/attention/GDN kernels, TorchInductor or SGLang
integration, Qwen correctness/coverage, profiling, or performance.

## Final report identity

- Run ID: `pypto-20260825T080254Z-910620-c669d9`
- Report path:
  `reports/data/pypto-nvidia-executable-sm120-pypto-20260825T080254Z-910620-c669d9.json`
- File type: regular file
- Mode: `0444`
- Bytes: 29,107
- SHA-256:
  `727362d7879d58cbee07b11050b17ad149274e8087b0d1872b8f186a66a272a9`
- Schema/smoke: `1` / `pypto-nvidia-executable-sm120`
- Status: `accepted-real-sm120-correctness-smoke`

The report is canonical sorted JSON with one terminal newline. It was published
under the finalizer's no-replace path and is not a symlink.

## Run and safety joins

The exact child command is the v4 fixed direct child:

```text
/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/bin/python
-I
-B
-S
/home/zhaosiying/pypto-love-tensor-ir/benchmarks/operators/pypto_nvidia_executable_sm120.py
```

- Process status/return code: `exited` / `0`
- Process SHA-256:
  `153f9c69b113c36f4e59c7fd272aea3ff281c70776e6a843ad12578621dddbff`
- Preflight SHA-256:
  `4c788dbd6ee0b6d68277fc8fd66663f88a70d915b66ee1e7042cc263de0964f3`
- Pre-release gate SHA-256:
  `be92af0d23965ccc7c7917529149047f108973a084f4bd8d23dbcc1b1cd43883`
- Start-barrier SHA-256:
  `d510ef38bdfb757f3ca0c6be2ed6c5300d7313cb3dfe9448c141766a41cd1cfe`
- Provisional SHA-256:
  `954a266ed5d592698649ad1947fe1879e75dce2f3d9ebad9e16c74034076221f`

The process metadata joins those exact preflight, gate, and barrier hashes. The
barrier joins the exact gate. The provisional and final report join the same
run ID, paths, documents, and hashes.

Pre-release, last, and post-exit audits all report empty external NVIDIA
compute, protected NVIDIA compute, protected runtime-mapping, and unreadable-map
sets. The parent and child static Torch identity documents agree; CUDA was
uninitialized before release. The final report's
`zero_nvidia_interference=true` is supported by those three audits. No protected
heavy process was present during this run.

## Root v4 control identity

- Root checkpoint commit at execution/finalization:
  `37c16b3902192a8da59a1d912d2ff3e06bec02fb`
- Root tree:
  `cc90c7d8ec7a58ccd0952c4af8756be89a69507e`
- A4 implementation commit:
  `5564008fddeaaf0a9861ee5c38c895558f577600`
- A4 implementation tree:
  `b1676a118604c22eebaec787987806f3cf1aeebb`
- B4 manifest commit:
  `7639d820f4d74972b493c01adc69c92087eefdea`
- B4 manifest tree:
  `52e37ab60276ebec2e06b46a4b55c39af4c22d62`
- Manifest path:
  `state/contracts/pypto_nvidia_executable_sm120_v4.json`
- Manifest bytes: 1,569
- Manifest SHA-256:
  `a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf`

The gate, child pre-CUDA record, provisional, and final report contain the same
v4 control identity. All seven live control files independently match the
report and v4 manifest in path, byte size, mode, and SHA-256:

- runner `f22befff...abdd`;
- contract `fa477d91...13bf`;
- manifest selector `bfa0e5c6...597a`;
- finalizer `aad7faf2...2f55`;
- preflight `0b9884f8...e3d1`;
- controller `978686ac...7465c8`;
- stop tool `879a2e38...17ddfe4`.

The report records `root_clean=true`. The current CP-0038 persistence drafts
were created after final report publication and do not alter the execution-time
clean-root identity.

## Exact PyPTO, environment, and DSO identity

- PyPTO commit:
  `206447cf8c68b9cff1b86e01f0b40bfd689cd7a7`
- PyPTO tree:
  `e0357daaefa74dbf676550015e60701996c400fb`
- PyPTO source was clean.
- TensorIR:
  `1dcb38c20e53d07c97d3781cae538e33901bae30`
- CUDA Tile:
  `af2417041cc939b87ef56d92cfdcf61737c5457e`
- LLVM/MLIR:
  `57109befac92811d2253109242ca6fa69c961fb2`
- PyPTO DSO path:
  `builds/pypto-executable-abi-on-206447c-final/product/pypto_core.cpython-314-x86_64-linux-gnu.so`
- PyPTO DSO bytes: 780,535,416
- PyPTO DSO SHA-256:
  `15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c`
- Python SHA-256:
  `aa85b78409de29d21c7db9a6ea0479fd73a4e245a733ea325f5ecf21772d030f`
- CUDA Runtime SHA-256:
  `96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0`
- Environment lock SHA-256:
  `29800d50f635e7188e55a6d6f43bfb4b8ac9ab16c4a21687db2960f18941932a`

The provisional integrity records for the contract, runner, environment lock,
versions lock, workspace lock, DSO, and CUDA Runtime all match their live byte
sizes and hashes. The report's exact Python, runner, DSO, and Runtime records
also match live files.

## Replay file and semantic joins

All replay files are regular read-only (`0444`) files and match report path,
size, and SHA-256:

- CompileRequest: 1,583 bytes,
  `13c319b832c51188678b51a32b155253a6f896bfd1395044832611df0843adda`;
- static BuildSpec: 1,416 bytes,
  `726ec78502813e816acb01ba64effcf3abbcb53b1e8a7cc59d43fc1928fb003b`;
- static Artifact: 16,690 bytes,
  `411f87920e7a9d9f97f66c865a5695b6b5016ec7983009c47df5c6a3c07b88e9`;
- dynamic BuildSpec: 1,459 bytes,
  `a97ad54f3e31ee1067aa27cf1495792b6742fecd13f2bd8abc0e56476d23b244`;
- dynamic Artifact: 20,483 bytes,
  `6914638d762ce5aaa963e4845d5c5fc473cf2c102b719de3260c7b27619711f5`;
- scalar BuildSpec: 1,416 bytes,
  `15c09132c572c298f08ee2228e91c9fa7cba59e39e2e40f7e2e3c17ff5370ea6`;
- scalar Artifact: 16,808 bytes,
  `28bf2001d40cfd49c641f5280f4c52cbad2c656377c70acf40ad8bb78e273a3f`.

The exact no-site semantic replay was independently rerun read-only. It matches
the final report in full:

- command SHA-256:
  `ca892e64d69c6bddb1cbe0c02b305b1e17d43c0149c4c29b19ebdaa66e1f88fc`;
- stdout SHA-256:
  `b2698fa85b8acebcd71150ea239b2908b988c9f675f010c3082071493d66d816`;
- replay document equality: exact.

Replay CompileRequest digests equal the runtime CompileRequest digests. Replay
TargetInfo equals the runtime observation on every target field. Replay case
records equal the final Artifact records on source, BuildSpec, Artifact, cache,
loader, Cubin, kernel ABI, entry, and fallback identities.

Canonical supported compute dtype order is exactly `["FP32", "BF16"]` in
both runtime observation and independent serialized TargetInfo replay. No
sorting or set-normalization is performed by the finalizer.

## Finalized execution evidence

The report's Artifact, execution, observation, child-gate, scope, and runtime
summary documents exactly equal their provisional counterparts.

| Case | Repetitions | Storage dtype | Grid | Kernel args | Cubin SHA-256 |
| --- | ---: | --- | --- | ---: | --- |
| static | 2 | float32 | `[4,1,1]` | 3 | `6dc121d2...32b82` |
| dynamic | 2 | float32 | `[6,1,1]` | 12 | `eabdc137...72ed60` |
| scalar | 2 | float16 | `[4,1,1]` | 3 | `fff77b04...1a4735` |

For all six executions:

- the raw current stream is non-null and non-default;
- the stream was externally synchronized;
- the packet was retained until synchronization;
- expected and actual logical byte hashes are equal;
- `torch_equal=true`;
- input bytes are unchanged;
- dynamic/output padding is unchanged;
- the Artifact identity matches the case Artifact;
- bound context address/ID match the observed context before unload;
- unload is explicit;
- terminal state is `Unloaded`;
- bound context after unload is zero.

The summary proves six module lifetimes, six explicit unloads, non-default
current-stream use, external synchronization, no fallback, and no forbidden
provider import.

## Finalizer completion

The final report binds the v4 finalizer at:

- path `tools/finalize_pypto_nvidia_executable_sm120.py`;
- SHA-256
  `aad7faf215e2aef0dc626553c1f917e443df0f7ffce4d22425c8276ed23e2f55`.

The report can only be assembled after the finalizer has validated canonical
schemas, external provisional anchor, process/preflight/gate/barrier joins,
control identity, live integrity files, Runtime/TargetInfo identity, scope,
replay file identities, independent exact-DSO semantic replay, all six
executions, and no performance-like fields. Publication occurs last through the
read-only no-replace publisher. The published report and independently repeated
semantic replay show no remaining finalizer gap.

## Findings

### P0

None.

### P1

None.

### P2

None.

## Accepted scope

Accepted:

- real RTX 5090/SM120 execution through the exact PyPTO `NvidiaExecutable`;
- exact v4 control, PyPTO, DSO, CUDA Runtime, TargetInfo, Artifact, BuildSpec,
  Cubin, and replay identities;
- the three minimal static, dynamic-stride, and tensor-plus-scalar cases;
- numerical/byte correctness for those cases against Torch/reference output;
- two complete non-default-current-stream lifetimes per case;
- synchronization, packet lifetime, explicit unload, and terminal unload state;
- no fallback and no forbidden provider import.

This is the minimal `NvidiaExecutable` runtime milestone. The report's
`operator_correctness=true` is interpreted only for these three exact smoke
cases, not as acceptance of a general or production operator library.

## Not claimed

The report explicitly does not claim:

- performance;
- CUDA Graph correctness;
- frontend HIR lowering;
- TorchInductor or SGLang integration;
- Qwen3.5 correctness or strict coverage.

This review additionally preserves the project boundary that no matmul,
paged-attention, GDN, framework route, model-forward, coverage, profiling, or
performance milestone follows merely from this runtime smoke.

## Final gate

**GO to persist acceptance of the minimal finalized real-SM120
`NvidiaExecutable` runtime milestone, with the scope and nonclaims above.**
