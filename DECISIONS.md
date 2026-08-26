# DECISIONS

## D-0001: Three-project implementation boundary

`pypto` owns compiler/runtime infrastructure; `pypto-kernels` owns every
custom high-performance operator; `pypto-framework-plugins` owns only pinned
framework adaptation.

## D-0002: Official framework source remains zero-diff

Torch and SGLang changes are installed through OOT registration and SGLang's
plugin/HookRegistry mechanisms. Compatibility code is exact-SHA gated and is
not described as a stable upstream ABI.

## D-0003: RTX 5090 Laptop is the final target

All capacity and performance claims refer to the local 24 GB/82-SM Laptop GPU,
not the 32 GB desktop RTX 5090.

## D-0004: 9B capacity does not justify hidden scope relaxation

Minimal BF16 9B failure stops the project for user direction. CPU offload,
quantization, or external compute fallback are not automatic alternatives.

## D-0005: Protected AMD/zcode scopes are never managed by this project

This project observes their resource use only. It never edits their trees,
reuses their processes, shares caches/endpoints, or signals their PIDs.

## D-0006: Runtime coverage is a manifest-bound evidence decision

Provider names alone never prove coverage. A strict decision requires a fixed
collector protocol, closed normalized trace, immutable artifact registry,
exact digest/provenance reconciliation, non-vacuous call and GPU-time totals,
latched violations, and single-owner durable reports. Collector completeness,
model correctness, and performance remain separate acceptance evidence.

## D-0007: Operator ABI and payload identity are producer-owned

`pypto-kernels` publishes the only framework-adapter ABI manifest. Plugins pin
and independently recompute it, validate live bindings, and separately prove
wheel or editable source ownership. Package version equality alone is never a
compatibility claim. Native executable payloads stay rejected until their own
digest-bound manifest and readiness gate exist.

## D-0008: Paired-state transfer is generic runtime infrastructure

GDN owns the semantic fact that BF16 conv and FP32 recurrent state form one
generation. Exact zero/copy/checkpoint, leases, generations, stream enqueue and
completion belong to the generic PyPTO compiler/runtime. The framework plugin
translates Radix lifecycle but does not execute copies. State transfer is not a
GDN operator, operator artifact or operator tuning record.

## D-0009: TensorIR runtime objects are not persistent PyPTO artifacts

Pinned `ICompiler::compile()` returns an in-process `IRuntimeKernel` while the
complete argument/grid reconstruction metadata remains inside
`CudaTileFrontendResult` and private strategy objects. PyPTO must first create a
deterministic bytes-plus-metadata seam, then own the artifact/cache and a
process/device/CUcontext-bound prewarmed executable state machine.
CompileRequest remains a pointer-free program/target policy; byte-affecting
per-region route/schedule/specialization lives in a separately versioned
KernelBuildSpec. Exact LLVM/tileiras producer identity enters every artifact
key. Streams are per-capture/launch dynamic values and never part of the
request or artifact.

## D-0010: PyTorch-pinned Triton is isolated reference infrastructure

The project environment must contain a non-editable wheel built from PyTorch's
exact Triton commit and its own pinned LLVM/CUDA toolchain. TensorIR's LLVM is a
different producer and must never be substituted. Dependency archives first
enter a bounded `materialized-unreviewed` state; review freezes every archive
SHA and the canonical manifest SHA in source before any downloaded executable
is run. A source/producer/dependency-bound wheel audit and fresh probe precede
the reversible environment replacement.

Triton remains a compatibility/reference lane only. Its successful SM120 smoke
can never count as PyPTO strict model-forward coverage or as a compute fallback.

## D-0011: Protected CPU-only coexistence is explicit and reversible

The user authorized non-benchmark CPU build/test work beside a protected
zcode/gem5 lane when that lane has no active NVIDIA compute PID. Default heavy
preflight remains fail-closed. The explicit coexistence policy retains every
SM120, CUDA-vs-HIP, DSO and environment check, uses a 24 GiB launch floor, and
keeps a living runner that pauses only its verified workspace-owned process
group at 16 GiB, a disk floor, timeout, failed NVIDIA audit, or protected
NVIDIA compute activation. The child sees no CUDA device; any owned NVIDIA
compute PID is a terminal policy violation. It automatically resumes only
after recoverable host-resource pressure clears.

The flag is invalid for GPU benchmarks. No condition permits this project to
signal, stop, clean or otherwise manage an amdgpu-sim/zcode process.

The same explicit policy may guard read-only model copying. The importer
rechecks memory and protected NVIDIA compute at every file EOF and immediately
before atomic publication; default invocation still rejects protected heavy
activity.

## D-0012: Project-environment replacement is one locked transaction

Every `run_isolated.py` consumer of `envs/pypto-nvidia` holds a shared flock
from preflight through child reaping. Triton plan holds the same shared lock;
apply/recover/rollback require the exclusive lock and an exact direct-child
interpreter/tool/action shape. The inherited descriptor, device/inode and exact
controller PID/start-ticks bind the child to its controller. Closing the parent
descriptor never explicitly unlocks a still-live inherited descriptor.
After a verified TERM, a failed CONT/STOP revalidation is treated as complete
only when the kernel proves the exact recorded PGID is empty; survivors or
ambiguous enumeration return 75 without another signal.

Replacement is stdlib/RECORD-owned and journaled, never pip-driven. The formal
backup root becomes visible only after an initializing journal is durable and
is published with no-replace atomic rename. Every mutation boundary performs a
fresh prefix-user audit; signals trigger rollback, and a later recover/rollback
is idempotent from the durable journal. Framework consumers cannot observe a
partially replaced prefix.

## D-0013: CompileRequest is bounded policy data, not a kernel identity

CompileRequest v1 value-owns immutable NvidiaTargetInfo and exact compiler
toolchain/policy. Its canonical decoder applies payload, container, string and
depth limits before schema conversion; malformed bytes must become a bounded
PyPTO `ValueError`, never ambient allocation pressure. Byte identity excludes
physical device/driver/capacity fields while including codegen-relevant
resources and exact producer inputs. Loader compatibility remains an input
projection until an actual artifact supplies kind/target/runtime requirements.

No region IR, semantic route, schedule, specialization, tensor/workspace
pointer, CUDA stream/handle or produced byte belongs in CompileRequest. Those
byte-affecting per-region fields are mandatory in the separately versioned
KernelBuildSpec before any producer or artifact-cache implementation.

## D-0014: Download provenance is representation-bound and locally licensed

Every network archive records a strong ETag and effective HTTPS URL. A resumed
download sends `Range` plus `If-Range` and accepts bytes only from the identical
strong ETag/effective URL/total/offset contract. Source-locked workspace seeds
must be independent, safely permissioned files and are copied plus rehashed
before extraction. Archive, manifest, review and cache publication are
no-replace and directory-fsynced; extracted trees normalize to deterministic
0755 directories and 0644/0755 files across umasks.

The accepted dependency cache is for workspace-local build and validation.
Presence of upstream license files does not authorize publishing a combined
wheel. NVIDIA CUDA EULA terms and the PyPTO authorization require a separate
wheel-level licensing review before any redistribution or public fused fork.

## D-0015: Strict Artifact identity joins source, schedule and loader ABI

A clean toolchain revision, producer-options digest and callable-ABI digest are
necessary but insufficient when checked independently. Artifact v1 also
cross-checks every duplicated schedule/launch fact and statically attests the
pinned CUDA 13.3 loader chain: unique entry, text/symbol linkage, flattened
KPARAM widths/ranges, PARAM_CBANK section/base/extent, constant bank and
coherent permissioned PT_LOAD file-to-VA mapping.

The structured TensorIR pipeline is a source-controlled canonical JSON value
identified by a verified Git blob, not an unexplained 40-character constant.
PyPTO and all vendored compiler sources must be clean at configure and build
time; no dirty-source override may mint a strict Artifact under locked commit
names. This remains runtime-free validation. It does not replace the future
trusted producer bridge, disassemble arbitrary SASS, load CUDA or prove kernel
execution.

## D-0016: A 22 GiB CPU threshold requires a new policy layer

The user clarified that roughly 22 GiB of available host memory is sufficient
for bounded CPU-heavy work and that the existing 24 GiB value is approximate.
The accepted `preflight.py` and `run_isolated.py` bytes remain unchanged because
GPU-smoke adapters and historical evidence pin them exactly. A future CPU-only
coexistence policy v2 may use 22 GiB for admission and resume, but it must be a
new source/manifest family rather than mutating accepted module globals.

That layer must retain the 16 GiB owned-run pause floor, CUDA-hidden child,
NVIDIA/action-boundary/periodic audits, disk and timeout controls, and verified
workspace-owned PGID signalling. It never applies to GPU-smoke or performance
runs and never authorizes signalling an amdgpu-sim/zcode process.

## D-0018: Lightweight execution gates until the models run (2026-08-27)

The user found the per-transaction ceremony (two independent P0/P1/P2=0
source reviews, separate manifest-only commits, multi-round review
fix-loops) too slow relative to the real deliverable. Until Qwen3.5-0.8B
and 9B run end-to-end with strict 100% PyPTO kernels, the process gates
are relaxed:

- No per-transaction source-code reviews and no two-reviewer GO
  requirement. A single consolidated review happens once, after the
  models run.
- A layer is accepted when its automated evidence passes: the relevant
  focused/full test suites are green, golden comparisons (source
  goldens, projection digests, Cubin/byte hashes, numerical references)
  match, and fresh builds link. "Tests + goldens green" replaces
  "reviewed GO" as the gate.
- Manifest-only publication ceremonies are optional; controls may be
  committed together with their implementation.
- Multi-round review fix-loops (the CPU-v2 style rounds) are replaced
  by: fix → rerun tests → move on.

The following are NOT relaxed: protected external scopes stay
read-only and unsignalled; PyTorch/SGLang/Triton upstream checkouts
stay zero-diff; all kernel algorithms stay in pypto-kernels and plugins
stay algorithm-free; no model-name/hidden-size/layer-id special casing
(specialization only from target/dtype/static dims/layout/phase/tuning
bucket); correctness still precedes performance claims; Triton stays
reference-only. The D-0017 per-kernel comparison report remains the
final acceptance.

## D-0017: Final acceptance is a per-kernel PyPTO-versus-SGLang-default comparison

The user reconfirmed on 2026-08-26 that the end state is Qwen3.5-9B running
with 100% PyPTO model-forward compute kernels — handwritten `pypto-kernels`
operators and TorchInductor auto-fused regions lowered through the PyPTO CUDA
backend — and that final performance acceptance is comparative, not absolute.

The baseline lane is the unmodified SGLang default optimized kernel stack
under the pinned SGLang/PyTorch versions on the same RTX 5090 Laptop GPU.
The candidate lane is the strict PyPTO run with `coverage=100%` and
`fallback_compute_kernels=0`. Both lanes use the same model, workload
schedule, batching, lengths and profiling methodology.

The required deliverables are: (1) end-to-end throughput/latency for both
lanes; (2) a per-kernel/per-operator breakdown table for both lanes covering
at least attention prefill/decode, GDN prefill/decode, GEMM/structured matmul,
and pointwise/reduction/indexing fusion classes, with kernel/provider, call
counts, total and mean GPU time, and candidate-versus-baseline deltas per
class; (3) the strict-coverage proof for the candidate lane; (4) hot-kernel
profiling evidence (Nsight or equivalent) for the largest regressions or wins.
E2E numbers alone do not satisfy this contract, and the breakdown must come
from identical measurement methodology on both lanes rather than vendor
marketing or unrelated configurations.
