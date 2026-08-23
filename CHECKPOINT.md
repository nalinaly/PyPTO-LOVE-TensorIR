# CHECKPOINT

**Checkpoint:** `CP-0004`

**Status:** R0 started; no compiler or model milestone accepted.

## Current truth

- The control repository was initialized in
  `/home/zhaosiying/pypto-love-tensor-ir`.
- Authorized PyPTO, standalone kernel/plugin projects, and clean official
  PyTorch/SGLang checkouts are materialized at the exact locked identities.
- A fully independent project-local Conda environment is cloned and the
  installed CUDA PyTorch tree is frozen by a content digest.
- Both Qwen3.5 snapshots are independently copied under `models/`, made
  read-only, and verified byte-for-byte against the tracked manifest. The AMD
  source tree was never modified.
- The standalone kernel project now has a reviewed, versioned, immutable ABI
  for tensor arguments, operator/problem/schedule identity, tuning records, and
  requested-versus-produced artifact provenance.
- It also has reviewed typed semantic families for generic matmul, paged
  attention, and GDN. These deliberately stop before a concrete tensor ABI or
  kernel implementation until the exact SGLang inventory is frozen.
- The framework plugin now binds actual imports to the locked CUDA Torch tree
  and clean SGLang checkout, rejects mixed linear-attention providers, and
  fails before launch while the real PyPTO Inductor dispatcher is unavailable.
- Its constructor-dispatch foundation now has a reviewed ContextVar mode,
  original-backend preservation outside that mode, pinned wrapper proxy
  semantics, and strict no-fallback failures inside it. It deliberately does
  not register with Torch until real scheduling/wrapper constructors exist.
- The cloned environment is not baseline-ready yet: its Triton distribution is
  still editable from `/home/zhaosiying/codebase/triton`. The runtime audit
  rejects this path. FlashInfer is corrected to the official 0.6.17 wheel and
  the unrelated external study package was removed.
- The local target is an NVIDIA GeForce RTX 5090 Laptop GPU, SM120, 24,463 MiB.
- CUDA Toolkit 13.3 is installed under `/usr/local/cuda-13.3`; the installed
  PyTorch is 2.13.0+cu130.
- The first native object-target build reached Ninja edge 251/260 and failed
  while compiling the binding consumer because the public `comm_layout.h`
  dependency on `runtime/src/common` was declared private. The minimal PUBLIC
  build-interface fix is staged but remains uncommitted until all build and
  packaging gates pass.
- After correcting that usage requirement, the incremental editable build and
  486 object-boundary-focused tests pass. The first full run reported 10,173
  pass, 58 skip, and three failures. Their product/test/harness fixes are now
  independently committed and pass targeted checks, but a full rerun and fresh
  wheel build remain mandatory before object-target acceptance.
- The exact PyTorch-pinned Triton source is now a clean official checkout under
  `upstream/triton`, but the environment still imports the inherited external
  editable distribution until a hermetic workspace wheel is built and
  installed.
- Prior reconnaissance did not complete a TensorIR SM120 runtime launch. Static
  target support is not runtime acceptance.
- Protected zcode work was active at R0 start. No heavy PyPTO task may begin
  until the live safety preflight is green.

## Resume action

Run `python tools/preflight.py --mode heavy`. If and only if it is green,
rerun the full PyPTO suite and continue the fresh wheel build in
`builds/pypto-wheel.dfB4Xk`, then inspect/install it into a clean target outside
the checkout and run the symlink probe there.
Next build the exact Triton wheel and remove the external editable runtime. If
protected zcode work remains active, continue only source/test work that does
not compile, launch a model, or claim runtime acceptance.
