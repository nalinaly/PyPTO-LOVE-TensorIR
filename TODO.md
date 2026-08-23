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
- [x] Materialize and verify the exact official Triton source/tree under
      `upstream/triton` (source-only; wheel/install gate remains open).
- [x] Replace inherited editable FlashInfer with official `0.6.17`.
- [x] Remove the unrelated external torch-compile-study editable package.
- [x] Freeze the source-hashed Qwen3.5 text compute inventory.
- [x] Define and independently review the strict normalized-trace/artifact
      coverage evidence contract; no runtime trace is claimed.
- [ ] Finish full-suite and fresh-wheel validation of the staged PyPTO
      single-DSO object boundary, then commit only its two CMake files.
- [x] Produce a three-review-approved SM120 TargetInfo source candidate in a
      separate worktree; explicitly mark it unbuilt/unverified.
- [ ] After object-DSO acceptance, apply candidate `9939b88`, resolve CMake,
      and pass native/Python/wheel/unchanged-Ascend validation.
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
- [ ] Cross-check the reference against an independent PyTorch implementation,
      then implement CUDA Tile decode and prefill/extend kernels separately.
- [x] Define and independently review unified GDN core ABI v1 with paired-state
      lifecycle, variable-length metadata and exact numerical semantics.
- [x] Implement and independently review the GDN paired-state CPU numerical
      reference; one-shot, segmented prefill and token decode match exactly.
- [ ] Cross-check GDN reference against the pinned independent Torch/SGLang
      reference once candidate framework launch provenance is hermetic.
- [ ] Implement PyPTO-owned paired state zero/copy needed for new slots and
      segmented Radix checkpoints.
- [x] Add and independently review the structured matmul BF16/FP32 numerical
      reference for all transpose and explicit-batch cases.
- [x] Publish the producer-owned canonical framework-adapter ABI manifest from
      `pypto-kernels`; remove the plugin's copied partial schema.
- [x] Pin ABI/source/distribution identity for isolated wheel and real PEP-660
      installs, and make Torch/SGLang pre-strict failure non-suppressible.

## Safety hold

The latest heavy preflight is green and no protected heavy workload was
observed, so the next native build gate may proceed. This is not a durable
lease: rerun preflight before every heavy action. Protected zcode processes are
still present; observation is allowed and signals or cleanup remain forbidden.
