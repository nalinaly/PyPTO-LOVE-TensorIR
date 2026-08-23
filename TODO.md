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
- [ ] Add immutable `BackendType::Nvidia` and explicit SM120 TargetInfo/traits
      as the next isolated compiler commit; do not link TensorIR yet.
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
- [ ] Implement GDN CPU numerical reference and prove one-shot prefill,
      segmented prefill and token decode state equivalence.
- [ ] Implement PyPTO-owned paired state zero/copy needed for new slots and
      segmented Radix checkpoints.

## Safety hold

Heavy build/model/server work remains paused while protected zcode SGLang,
vLLM, gem5, or high-parallel gem5 builds are active. The currently observed
protected lane is zcode's TP=2 formal vLLM/gem5 run. Observation is allowed;
signals and cleanup are forbidden.
