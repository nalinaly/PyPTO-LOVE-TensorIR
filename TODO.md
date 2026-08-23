# TODO

## Active R0

- [x] Initialize the root control repository and persistence skeleton.
- [ ] Commit the root bootstrap transaction.
- [ ] Clone authorized PyPTO at the exact locked SHA.
- [ ] Clone clean PyTorch and SGLang official checkouts.
- [ ] Initialize `pypto-kernels` and `pypto-framework-plugins` repositories.
- [ ] Vendor/pin TensorIR and CUDA Tile inside the PyPTO project.
- [ ] Clone `triton-dev` into `envs/pypto-nvidia` without mutating the source.
- [ ] Copy and hash Qwen3.5 model snapshots after the protected-workload gate.
- [ ] Produce local-source `docs/implementation_map.txt`.
- [ ] Run unmodified SGLang 0.8B then minimal 9B baseline.
- [ ] Freeze R0 evidence and advance `PLAN.md` to P1.

## Safety hold

Heavy build/model/server work remains paused while protected zcode SGLang,
gem5, or high-parallel gem5 builds are active. Observation is allowed; signals
and cleanup are forbidden.

