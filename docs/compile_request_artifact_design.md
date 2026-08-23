# CompileRequest, TensorIR artifact, and NVIDIA executable design

This document freezes the source-grounded dependency boundary after
`NvidiaTargetInfo`. It is a design constraint, not an implementation or runtime
claim.

## Why the boundary must be split

The pinned TensorIR API cannot currently be treated as a persistent compiler
artifact API:

- `include/tensor_ir/Compiler/Compiler.h` returns only an in-process
  `IRuntimeKernelPtr` from `ICompiler::compile()`.
- `CudaTileFrontendResult` contains the entry name, runtime name,
  `KernelArgLayout`, resolved iteration shape and static/runtime-grid choice,
  but `ICompiler::compile()` hides that result inside a runtime object.
- `CudaTileRuntimeKernel` exposes device bytes, but the argument packer and grid
  computer are private process objects with no versioned serialization.
- `initializeRuntimeState()` has no once/mutex state despite the interface's
  thread-safety statement. Repeated/concurrent loads can overwrite the module
  handle; `checkSupport()` and workspace sizing are still TODOs.

Therefore an `IRuntimeKernel` is neither a cache record nor an executable ABI.
PyPTO must first obtain immutable bytes plus complete reconstruction metadata.

## Mandatory landing order

1. Accept and commit the single-DSO object boundary.
2. Integrate, build, package and accept `NvidiaTargetInfo` candidate
   `9939b885...`.
3. Add only the data contract for `CompileRequest`.
4. Add the pointer-free per-region `KernelBuildSpec` data contract.
5. Materialize exact LLVM, complete the private TensorIR/CUDA Tile static build
   composition, and resolve/hash the exact `tileiras` executable.
6. Add a deterministic TensorIR compiled-artifact metadata seam.
7. Add the PyPTO-owned `Artifact` schema and persistent `ArtifactCache`.
8. Add a prewarmed `NvidiaExecutable` with explicit load/unload leases.
9. Add framework tensor and raw-current-stream launch adapters.
10. Connect Inductor async compile, Python/subgraph wrapper and CUDA Graph
   prewarm/capture.
11. Open concurrent/subprocess compilation and graph replay only after their
    own evidence gates.

No step may be presented as a later step's acceptance.

## CompileRequest v1

The first request commit is data-only, immutable, pointer-free and versioned.
It owns a value-copy of `NvidiaTargetInfo` and defines deterministic canonical
serialization suitable for an explicitly invoked worker process.

Allowed fields are limited to:

- request schema and compiler ABI versions;
- backend and immutable target information;
- requested artifact policy (the first strict gate requires Cubin);
- deterministic compiler policy and exact toolchain identity;
- explicit verification policy that does not encode per-region schedule or
  produced bytes.

Live target discovery belongs to parent-owned `target_query.cpp`. Before
compile or capture, the parent selects a device, validates SM120/resources and
the locked NVIDIA toolchain, constructs `NvidiaTargetInfo`, and value-copies it
into the request. A compile thread/subprocess never re-queries the GPU,
initializes CUDA or depends on an ambient current device.

The system provides three distinct identity projections:

- byte compile identity includes architecture, all codegen-relevant resource
  traits, TensorIR/CUDA Tile/exact LLVM revisions, compiler ABI, normalized
  CUDA toolkit root, and resolved `tileiras` path/version/SHA-256, but excludes
  device ordinal, UUID, PCI identity, driver-only loader compatibility and
  debug dump/timing controls;
- loader compatibility binds actual artifact kind/target plus CUDA
  driver/runtime requirements without pretending they changed compiled bytes;
- device/autotune identity additionally includes physical device identity and
  measured hardware resources.

The request must not contain:

- raw streams, tensor pointers, workspace pointers or CUDA handles;
- actual artifact kind, compiled bytes or runtime module state;
- model names, layer ids, operator schedule/tile constants or benchmark keys;
- a global `TensorIR` versus direct-CUDA-Tile switch. Region routing belongs to
  the later `PipelineRegistry`, because one program may contain generic,
  structured and serving regions simultaneously.

The first request commit does not modify legacy `ir.compile`, `PassContext`,
`CompiledProgram`, runtime, plugins or kernels. It does not link TensorIR or
register a backend.

## Per-region KernelBuildSpec

CompileRequest is program/target policy and cannot identify a kernel by itself.
Every compiled region therefore has a separate immutable, pointer-free and
versioned `KernelBuildSpec` containing:

- canonical source/IR and callable ABI digests;
- semantic route (`generic-direct`, `structured-tensorir`, or
  `serving-direct`) and compiler pipeline revision;
- resolved schedule, tile, layout, persistence, CTA/warp/stage and every other
  byte-affecting option;
- static/symbolic specialization and argument/result/mutation ABI;
- operator/problem/schedule digests when an operator catalog entry applies;
- the relevant CompileRequest byte-identity digest.

It contains no model name or framework object. CompileRequest plus
KernelBuildSpec plus exact producer identity form artifact/cache identity.
Structured TensorIR and direct CUDA Tile producers emit the same PyPTO Artifact
schema; direct generic/serving paths cannot bypass provenance, cache validation
or executable reconstruction.

## Deterministic TensorIR seam

The private PyPTO bridge must call or extend the pinned
`CudaTileFrontend`/serialization path and return a versioned value containing:

- immutable device-code bytes;
- actual artifact kind derived from byte magic, never copied from the request;
- requested and actual compute targets;
- TileIR bytecode version when applicable;
- runtime kernel and entry-function names;
- complete serialized `KernelArgLayout`;
- static grid dimensions or the complete runtime-grid policy/metadata;
- compiler/options/target identity digests and source revisions;
- exact LLVM revision and resolved `tileiras` real path, version, executable
  SHA-256, CUDA toolkit root and toolkit version;
- explicit workspace ABI metadata.

Strict v1 requests Cubin and requires actual ELF/Cubin bytes. TensorIR's
documented Cubin-to-TileIR fallback is a compile error for this gate. A TileIR
driver-JIT path is a separate later capability and cache schema.

Ambient TensorIR environment overrides are forbidden in strict compilation.
The pinned compiler currently reads strategy, candidate-count and debug/load
paths from `TENSOR_IR_*` variables without incorporating them into
`CompileOptions::toUniqueString()`. Debug dump/print/timing controls are
observability only and stay outside byte identity; bytecode-load and every
byte-affecting override are rejected in strict mode. The parent resolves
`tileiras` once and passes its verified identity explicitly; a worker never
chooses an assembler through ambient `PATH`. The bridge must reject remaining
ambient overrides or use an upstreamable option that disables them. It never
temporarily mutates process environment around concurrent compilation.

## SM120 resource corrections before automatic selection

The pinned TensorIR path currently has two incorrect RTX 5090 defaults:

- `CudaTileCompileOptions` does not propagate an actual SM count into
  `TensorToCudaTilePipelineOptions.smCount`, so SM120 can fall back to 148 SM
  instead of the candidate's measured 82 SM;
- `TileCandidateGenerator` assumes 227 KiB usable shared memory per CTA for all
  compute capabilities at least 100, while the target candidate records an
  opt-in maximum of 101376 bytes on this device.

Both values must come from the accepted `TargetTraits` and enter compiler/cache
identity. An initial explicit-schedule smoke may set `maxCandidates=0`; no
automatic tile-selection performance or legality claim is allowed until the
resource path is corrected.

## Artifact and cache ownership

`pypto` owns the physical artifact cache. `pypto-kernels` may request an
operator/schedule and record provenance, but it cannot load modules or create a
second cache.

The artifact cache must validate canonical metadata, bytes digest, actual kind,
CompileRequest/KernelBuildSpec digests, exact compiler/LLVM/tileiras producer,
target compatibility and compiler ABI before publication or reuse. Publication
is atomic/no-replace and readers never observe partial bytes. Cache lookup does
not load a CUDA module; Artifact itself is process/context-free.

## NvidiaExecutable and stream law

`NvidiaExecutable` reconstructs launch metadata from a validated artifact and
owns an explicit state machine: unloaded, prewarming, ready, failed and
unloading. Prewarm binds the current process/fork generation, target device
ordinal and live `CUcontext`, and validates them against TargetInfo/loader
compatibility. Load is once-only with error latching; unload cannot race a
launch or live CUDA Graph lease. CUDA module/kernel/stream handles are never
serialized, inherited across `fork()`, or reused in another process/context.

The stream is a dynamic argument of every launch:

```text
CompileRequest -> Artifact -> prewarmed NvidiaExecutable
framework invocation -> raw current CUstream -> NvidiaExecutable.launch(...)
```

A null stream is rejected. TensorIR's implicit null/default-stream behavior is
not an allowed fallback. Launch validates the current process, device, context
and stream context against the prewarmed executable. Each worker receives an
immutable request explicitly and creates its own child `PassContext`; a
`PassContext` object is never shared concurrently.

Before CUDA Graph capture, the system must finish cache lookup, compilation,
module/function load, ABI/support validation, workspace sizing/allocation,
autotuning and artifact-registry publication. During capture, each wrapper
invocation resolves and validates the raw current capture stream once, uses
preallocated/strictly bounded launch argument/grid storage, and records the
prewarmed launch without file I/O, JIT, module load, device allocation or host
heap growth. Replay launches the instantiated graph; it does not re-enter
per-kernel wrappers or redo current-stream lookup. A different process,
context/device, workspace/executable address or graph-stream semantic requires
recapture. The graph retains executable/workspace leases through destruction.

## Subprocess boundary

`ContextVar` state does not cross a thread or process pool automatically. A
subprocess receives only the canonical pointer-free request and source/IR
payload, and may return only artifact bytes plus metadata. It must not create a
CUDA context, load a module, allocate workspace or carry a stream. The parent
process validates, publishes, loads and launches the artifact.

## Required evidence gates

- C++ and Python canonical request round-trip and malformed-input rejection;
- CompileRequest and per-region KernelBuildSpec canonical round-trip tests;
- byte identity, loader compatibility and device/autotune identity tests;
- explicit worker payload propagation with no ambient backend/target reads;
- exact LLVM/TensorIR/CUDA Tile single-product staged-install audit;
- resolved `tileiras` path/version/SHA/toolkit-root identity tests;
- deterministic compile under environment-override rejection;
- SM count/shared-memory propagation tests for SM120;
- artifact reconstruction tests from bytes plus metadata only;
- requested-versus-actual artifact kind/magic rejection;
- concurrent prewarm/load/unload/error-latch tests;
- process/fork/device/CUcontext/stream mismatch rejection tests;
- raw non-default current-stream launch before framework integration;
- capture-safe prewarm and stable-workspace tests before CUDA Graph claims.
