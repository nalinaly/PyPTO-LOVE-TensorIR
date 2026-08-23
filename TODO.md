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
- [ ] Produce local-source `docs/implementation_map.txt`.
- [ ] Run unmodified SGLang 0.8B then minimal 9B baseline.
- [ ] Freeze R0 evidence and advance `PLAN.md` to P1.
- [ ] Replace the inherited external editable Triton with an in-workspace build
      of PyTorch's exact `5d6048aa...` pin.
- [x] Replace inherited editable FlashInfer with official `0.6.17`.
- [x] Remove the unrelated external torch-compile-study editable package.

## Safety hold

Heavy build/model/server work remains paused while protected zcode SGLang,
gem5, or high-parallel gem5 builds are active. Observation is allowed; signals
and cleanup are forbidden.
