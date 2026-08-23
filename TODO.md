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
- [ ] After single-DSO, TargetInfo, CompileRequest/current-stream artifact and
      operator-executable gates, implement plugin-owned PyPTO scheduling,
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

## Safety hold

The action-boundary heavy preflight is red for the observed
`zcode-vllm-tp2-v4` TP=2 vLLM/gem5 lane. No native PyPTO action began. Continue
only light work, wait for natural exit, and rerun preflight. Observation is
allowed; signals and cleanup remain forbidden.
