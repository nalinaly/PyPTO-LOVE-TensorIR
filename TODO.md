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
- [ ] Freeze R0 evidence and advance `PLAN.md` to P1.
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
- [ ] Bind SM120 82-SM/101376-byte resources and deterministic TensorIR options
      into the compiled Artifact identity before ArtifactCache or
      NvidiaExecutable implementation.
- [ ] Implement the pointer-free bytes-plus-metadata Artifact returned before
      TensorIR runtime-kernel construction; include entry/ABI/resolved
      shape-grid metadata and exact byte/producer digests.
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
