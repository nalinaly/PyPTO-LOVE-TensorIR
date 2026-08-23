# CHECKPOINT

**Checkpoint:** `CP-0007`

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
- Its reviewed persistent tuning database now provides complete key/cohort
  auditing, deterministic winner selection, multiprocess-safe atomic
  publication and corruption rejection. It contains no device measurement yet.
- Structured matmul has a reviewed explicit ABI v1 for contiguous/aligned BF16
  rank-2 and batched rank-3 tensors, FP32 accumulation, transpose semantics and
  output non-aliasing. No lowering, schedule, launch or performance exists yet.
- Operator artifact provenance now exposes a canonical complete-field digest
  restricted to the explicit matmul/paged-attention/GDN ABI catalog. It is only
  a future join input: physical cache/loading stays compiler-owned and the
  opaque problem digest is not yet a typed launch proof.
- The framework plugin now binds actual imports to the locked CUDA Torch tree
  and clean SGLang checkout, rejects mixed linear-attention providers, and
  fails before launch while the real PyPTO Inductor dispatcher is unavailable.
- Its constructor-dispatch foundation now has a reviewed ContextVar mode,
  original-backend preservation outside that mode, pinned wrapper proxy
  semantics, and strict no-fallback failures inside it. It deliberately does
  not register with Torch until real scheduling/wrapper constructors exist.
- The framework plugin also freezes 41 source-hashed Qwen3.5 text-path compute
  obligations across generic, matmul, full-attention and GDN providers. This is
  a static inventory only; no site is marked implemented or covered.
- The framework plugin now also has a reviewed strict runtime-coverage evidence
  contract. It reconciles a fixed-revision closed activity trace against a
  digest-bound artifact registry, rejects vacuous/zero-time/misclassified or
  fallback evidence, latches strict failure, and owns report publication across
  processes. Its 68 tests and clean wheel build pass. No runtime collector or
  Qwen trace has been connected, so current runtime coverage remains unclaimed.
- Pinned-source reconnaissance selects the official SGLang HookRegistry plus
  PyTorch's experimental CUPTI monitor for the future collector. Its first
  eager-only trace must remain `closed_world=false`; graph replay and a proven
  complete drain/artifact-launch binding are separate gates.
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
- Candidate and baseline framework runs now have separate, reviewed execution
  profiles. Baseline cannot import PyPTO/plugin sources or load general SGLang
  plugins; candidate framework launch cannot start until exact environment,
  selected NVIDIA runtime, entry points and process ownership all verify.
- Frozen baseline launch scripts exist for 0.8B and 9B, but neither is runtime
  evidence and the CPython 3.12 baseline environment is not built yet.
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
not compile, launch a model, or claim runtime acceptance. After the object-DSO
gate is committed, add immutable NVIDIA SM120 target identity in a separate
compiler commit; never inherit TensorIR's SM100 defaults.
