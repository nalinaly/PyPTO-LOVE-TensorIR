# TODO

## Active R0

- [x] Initialize the root control repository and persistence skeleton.
- [x] Commit the root bootstrap transaction.
- [x] Clone authorized PyPTO at the exact locked SHA.
- [x] Clone clean PyTorch and SGLang official checkouts.
- [x] Initialize `pypto-kernels` and `pypto-framework-plugins` repositories.
- [x] Vendor/pin TensorIR and CUDA Tile inside the PyPTO project.
- [x] Clone `triton-dev` into `envs/pypto-nvidia` without mutating the source.
- [x] Copy and hash Qwen3.5 model snapshots after the protected-workload gate.
- [x] Produce checkout-grounded `docs/implementation_map.txt` with exact
      ownership, extension points, dependency direction and forbidden edits.
- [ ] Run unmodified SGLang 0.8B then minimal 9B baseline.
- [x] Freeze plugin-free, selected-prefix 0.8B/9B baseline launch commands.
- [ ] Freeze the remaining R0 baseline evidence without rolling back the
      already active P2 compiler/runtime phase.
- [ ] Replace the inherited external editable Triton with an in-workspace build
      of PyTorch's exact `5d6048aa...` pin.
- [x] Commit bounded dependency materialization, exact wheel/native audit,
      pip-free fresh probe, finalized reference-only SM120 smoke and reversible
      stdlib environment-replacement gates in four reviewable control commits.
- [x] Enforce shared environment locks for every consumer and an inherited,
      direct-child exclusive lock for replacement/recovery transactions.
- [x] Materialize, independently review and source-lock all ten exact Triton
      dependency archives; promote manifest `29c073...`, pass networkless
      tool-version probes and durably publish the reviewed cache.
- [x] Build the exact pinned Triton reference wheel from the reviewed cache and
      pass complete wheel/native audit plus pip-free fresh probe; keep it
      uninstalled until the baseline replacement gate.
- [x] Materialize and verify the exact official Triton source/tree under
      `upstream/triton` (wheel accepted; environment install gate remains open).
- [x] Replace inherited editable FlashInfer with official `0.6.17`.
- [x] Remove the unrelated external torch-compile-study editable package.
- [x] Freeze the source-hashed Qwen3.5 text compute inventory.
- [x] Define and independently review the strict normalized-trace/artifact
      coverage evidence contract; no runtime trace is claimed.
- [x] Finish native/editable/fresh-wheel/clean-install validation of the PyPTO
      single-DSO object boundary and commit only its two CMake files as
      `e463bce`.
- [x] Repair and independently approve the single-DSO acceptance runbook:
      real venv install, complete DSO/dependency/compile-row audit and exact
      staged-file boundary.
- [x] Produce a three-review-approved SM120 TargetInfo source candidate in a
      separate worktree; explicitly mark it unbuilt/unverified.
- [x] Apply candidate `9939b88` after object-DSO acceptance and pass fresh
      native 2/2 CTest, one-DSO wheel, 31 TargetInfo cases, 10,209-pass/57-skip
      full regression and the independent symlink probe at PyPTO `042878d`.
- [x] Freeze and independently approve the conflict-checked TargetInfo
      acceptance runbook with fresh native, one-DSO wheel and exact test-count
      gates.
- [x] Source-audit TensorIR compiler/runtime persistence and freeze D-0009:
      `IRuntimeKernel` is not a cache artifact; exact bytes-plus-metadata and
      process/device/CUcontext executable ownership are required.
- [x] Implement and independently gate pointer-free CompileRequest v1 at PyPTO
      `09e014c`: bounded canonical MessagePack, exact target/toolchain policy,
      three identity projections, native 2/2 and Python 62/62.
- [x] Implement and independently gate pointer-free per-region KernelBuildSpec
      v1 at PyPTO `9b3cf71`: bounded canonical MessagePack, all explicit
      resolved schedule categories, exact current-DSO replay and native 4/4.
- [x] Compose exact TensorIR/CUDA Tile/LLVM sources privately inside the one
      PyPTO DSO, bind CUDA 13.3/tileiras identity, export correct embedding
      headers, and remove all product RPATH/RUNPATH leakage.
- [x] Bind SM120 82-SM/101376-byte resources and deterministic TensorIR options
      into the compiled Artifact identity before ArtifactCache or
      NvidiaExecutable implementation.
- [x] Implement and independently gate TensorIR's runtime-free in-memory
      compiled-result primitive before runtime-kernel construction, including
      complete entry/ABI/grid/workspace metadata, full TileIR/Cubin validation,
      exact verified assembler bytes, deterministic Cubin and packer bounds.
- [x] Extract and independently gate the compiler-private bounded canonical
      MessagePack codec, including pre-allocation aggregate/object/depth/BIN
      limits, malformed parser exception conversion and fresh ON/OFF builds.
- [x] Implement and independently gate the PyPTO strict producer bridge that
      consumes CompileRequest, KernelBuildSpec and canonical source, validates
      every locked producer, maps the complete resolved schedule, and forbids
      fallback/ambient policy.
- [x] Implement the compiler-owned persistent ArtifactCache with bounded
      trusted-local reads, exact cache-key/provenance validation, atomic
      no-replace publication and no CUDA/runtime/framework state.
- [x] Implement and independently gate the CPU/fake-driver contract for a
      process/device/CUcontext-bound
      `NvidiaExecutable` that loads only a validated Artifact, validates exact
      entry/resource/argument/grid/workspace ABI, prewarms outside graph
      capture, latches failure, owns graph leases, and accepts a non-null raw
      current stream only at launch.
- [x] Implement and independently gate the parent-only live NVIDIA runtime
      observation value at PyPTO `6361f11`: complete TargetInfo propagation,
      authenticated already-loaded Runtime provider, real private-Driver fork
      latch, no handle retention and backend-OFF fail-closed behavior.
- [x] Implement, source-lock and independently gate the exact-product SM120
      correctness-smoke controller, static/dynamic/scalar runner, protected
      zero-NVIDIA policy, replay artifacts and CPU-only finalizer. This is not
      real CUDA evidence.
- [x] Preserve the first fail-closed real-SM120 run, repair unordered CUDA
      parameter ranges, four-byte dynamic size/stride packing and
      enumeration-order-independent width validation at PyPTO `206447c`, and
      rebind the exact product through control manifest v3.
- [x] Preserve the v3 real-GPU child/provisional as unfinalized diagnostic,
      repair producer-canonical `[FP32,BF16]` evidence validation, add the full
      malformed/order-drift matrix, and bind immutable control manifest v4.
- [x] Under a fresh green `gpu-smoke` gate, run and finalize the exact-product
      v4 real SM120 static/dynamic/scalar non-default-current-stream
      correctness and explicit unload smoke; do not advance frontend lowering
      from CPU/fake-driver or prewarm-failure evidence alone.
- [x] Implement and independently gate the compile-free internal HIR-to-TensorIR
      emitter for strict static contiguous FP32/BF16 `tensor.add` at PyPTO
      `07ab9ea`, including deterministic metadata and clean ON/OFF tests.
- [x] Version and independently gate standalone bounded canonical schedule
      identity at PyPTO `fa85e5a` without changing nested KernelBuildSpec bytes.
- [x] Add and independently gate compiler-owned canonical frontend
      specialization/ABI preparation and final BuildSpec construction from a
      producer-shaped callable ABI at PyPTO `c4cf755`.
- [x] Refactor one strict producer transaction to return final BuildSpec plus
      Artifact and bind the immutable result under `pypto.compiler` at PyPTO
      `642ff5b`.
- [x] Implement, independently review and manifest-bind the separate frontend
      FP32/BF16 SM120 correctness-smoke controller/finalizer at root `47a0c15`.
- [x] Execute separately versioned HIR-authored FP32/BF16 vector add through the
      returned Artifact on real SM120 and finalize it in a CPU-only replay at
      run `pypto-20260825T145519Z-1142938-70ac73`.
- [x] Extend the accepted frontend compiler path to bounded fused pointwise at
      PyPTO `b83fcd3`, preserving CP44 V1 bytes and producing SM120 Cubins.
- [x] Implement, independently review and manifest-bind the nine-case
      fused-pointwise SM120 controller/finalizer at root `c98f984`/`438c25f`.
- [x] Run, CPU-finalize and independently review the policy-2 real-SM120
      fused-pointwise numerical gate at run `pypto-20260826T073309Z-1451510-e48ced`.
- [x] Build and gate RowReductionV3 at clean PyPTO `62eb882`, including OFF/ON
      products, native/Python tests and four exact nonfallback SM120 Cubins.
- [ ] Execute and independently gate RowReductionV3 FP32/BF16 numerical
      correctness on real SM120 before claiming runtime coverage.
- [x] Implement and independently source-review StructuredMatmulV4 at
      `d755117`, including transpose, FP32 accumulation and normalized ABI.
- [ ] Build, produce Cubins and gate StructuredMatmulV4 before Inductor
      integration.
- [x] Implement bounded canonical PyPTO Artifact v1 serialization with bytes,
      ABI, request/build-spec/producer digests and malformed-input rejection.
- [x] Isolate the TensorIR/LLVM ABI bridge, hide every non-Python DSO export,
      enforce clean gitlink revisions, and isolate native ON/OFF output paths.
- [x] Canonicalize operator-only artifact provenance without creating a second
      compiler artifact cache in `pypto-kernels`.
- [x] Freeze the pinned Torch/SGLang runtime coverage collector source map.
- [ ] Implement the eager-only CUPTI-monitor development collector after the
      compiler ArtifactCache/launch provenance contract exists; it must emit
      `closed_world=false` initially.
- [x] Define and independently review paged-attention ABI v1 for prefill/extend
      and decode, including KV append and host metadata reference validation.
- [x] Implement and independently review the paged-attention ABI v1 numerical
      reference, including prefill-to-decode state continuity.
- [x] Cross-check paged attention against an independent Torch CPU expression,
      including shared-prefix prefill-to-decode cache continuity.
- [ ] Implement CUDA Tile attention decode and prefill/extend kernels
      separately after the compiler/runtime launch foundation lands.
- [x] Define and independently review unified GDN core ABI v1 with paired-state
      lifecycle, variable-length metadata and exact numerical semantics.
- [x] Implement and independently review the GDN paired-state CPU numerical
      reference; one-shot, segmented prefill and token decode match exactly.
- [x] Cross-check GDN against a structurally independent vectorized Torch CPU
      expression, including paired-state prefill-to-decode continuity.
- [x] Freeze ownership and the generic StateBundle zero/copy/checkpoint design;
      keep it out of the GDN operator catalog.
- [x] Freeze the pinned active SGLang UnifiedRadix/MambaComponent lifecycle
      inventory and fail-closed adapter readiness order.
- [x] Freeze the pinned TorchInductor zero-diff backend surface with exact
      source/AST contracts, the full `_inductor` Python manifest, and explicit
      fail-closed CSE/extern/foreach/multi-template/GEMM obligations.
- [ ] After CompileRequest, KernelBuildSpec, exact artifact/current-stream
      executable and operator gates, implement plugin-owned PyPTO scheduling,
      CSE dtype/shape propagation, Python/subgraph wrapper, strict template
      choice filtering, and an atomic reversible CUDA registry transaction.
- [ ] After single-DSO, TargetInfo and current-stream executable gates,
      implement generic PyPTO StateBundle zero/copy needed for new slots and
      segmented Radix checkpoints.
- [x] Add and independently review the structured matmul BF16/FP32 numerical
      reference for all transpose and explicit-batch cases.
- [x] Cross-check structured matmul against independent Torch CPU FP32
      accumulation across all transpose and explicit-batch variants.
- [x] Publish the producer-owned canonical framework-adapter ABI manifest from
      `pypto-kernels`; remove the plugin's copied partial schema.
- [x] Pin ABI/source/distribution identity for isolated wheel and real PEP-660
      installs, and make Torch/SGLang pre-strict failure non-suppressible.
- [x] Define and independently review the canonical operator benchmark JSON v1
      contract with symmetric baseline/candidate evidence and no live result.
- [ ] After real CUBIN and complete TargetInfo identity land, implement atomic
      no-replace benchmark publication under ignored artifacts and run the
      first CUDA-event measurements.

## Protected coexistence

The user explicitly authorized non-benchmark CPU-only coexistence with a
protected zcode/gem5 lane. Use only the explicit heavy coexistence flag: 24 GiB
launch floor, living-runner 16 GiB pause floor, action-boundary and periodic
NVIDIA PID audit, and verified signals only to this workspace's recorded PGID.
GPU benchmarks never coexist. External signals and cleanup remain forbidden.
The isolated root control suite has now passed live beside seven protected
heavy processes with no protected NVIDIA compute PID, no pause/abort and no
external signal; EV-0035 binds that run. This is control evidence only, not a
GPU/compiler/model/performance result.

The user has additionally authorized an approximate 22 GiB admission/resume
threshold for bounded CPU-heavy work. Implement it only as a separately
versioned CPU-coexistence adapter and manifest; do not edit the exact-hashed
policy-v1 modules or reinterpret EV-0035. Until that adapter is reviewed, the
existing executable control continues to enforce 24 GiB.

## Current resume point

- [ ] Read `docs/NEXT_AGENT_ZERO_CONTEXT_HANDOVER.md`; the user paused the
      current run and requested a zero-context transfer.
- [ ] Revalidate the six untracked CPU-v2 draft files from scratch before any
      commit or live use; latest test/review results are stale after safety
      edits.

- [x] Commit hardened RowReductionV3 SM120 controls (`23efafa`).
- [x] Publish the separate canonical manifest (`34ee759`).
- [x] Regenerate and validate two CUDA-hidden anchor runs; focused tests pass
      34 plus 84 subtests and all three final reviews are GO.
- [ ] After the external zcode `cutlass-compiler` releases host memory, rerun
      the clean postmanifest full CPU suite.
- [ ] Only then run the fixed RowReductionV3 GPU controller, CPU finalizer, and
      unchanged no-replace retry; preserve the provisional/final report.
- [x] Freeze the read-only StructuredMatmulV4 replay/conflict/build map.
- [ ] After RowReduction acceptance, replay `6ee412a` then `d755117` exactly as
      documented in `docs/structured_matmul_v4_replay_map.md`.
