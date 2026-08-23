# CHECKPOINT

**Checkpoint:** `CP-0019`

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
- Structured matmul now also has a reviewed standard-library BF16/FP32
  numerical reference for every transpose combination and explicit rank-3
  batching. It is not yet compared with TensorIR/CUDA.
- The standalone operator project now publishes the only canonical
  framework-adapter ABI manifest and digest from its real adapter types,
  semantic configs, full specs, constants, signatures and ordered catalog.
- Framework plugins independently recompute that ABI, validate the live
  bindings, pin the Python source-tree digest, and bind either exact wheel
  RECORD ownership or strict PEP-660 editable metadata. The copied partial ABI
  schema was removed.
- Current source-only packaging rejects untracked native extensions and
  sourceless bytecode. Torch and SGLang adapter entry paths perform identity and
  executable-readiness checks before framework mutation; pre-strict failures
  cannot be swallowed as ordinary fallback exceptions.
- The operator project has 117 tests plus 71 subtests; the plugin has 128
  tests. Both wheel and real editable identity probes pass. No executable
  kernel, framework registration, model correctness, coverage or performance
  is claimed by this boundary.
- The matmul, paged-attention and GDN scalar references now have independent
  CPU Torch cross-checks in device-hidden child processes. They add no
  production dependency and preserve the framework-free parent import proof.
- Attention now includes shared-prefix PREFILL→DECODE continuity with private
  write tails; GDN uses vectorized Torch primitives and paired-state continuity
  rather than copying the production scalar loop structure.
- The standalone project now has a source-only benchmark JSON v1 contract. It
  joins typed config/descriptors/workload, CUBIN provenance, symmetric
  candidate/baseline measurements, reset/mutation correctness, separate
  profile traces and fixed device conditions before deriving any comparison.
- No result JSON or publisher exists, and tuning promotion is hard-disabled.
  The contract cannot be presented as performance evidence. The standalone
  suite is now 117 tests plus 71 subtests.
- Paired GDN state zero/copy/checkpoint ownership is now frozen as generic
  PyPTO runtime infrastructure, not an operator or framework copy kernel. The
  detailed StateBundle ABI/lease/stream design is documentation only and is
  ordered after single-DSO, TargetInfo and current-stream executable gates.
- The framework plugin now has a pinned active-route SGLang state lifecycle
  inventory: 11 sources and 35 sites across UnifiedRadix/MambaComponent,
  unified slot translation, deferred clear/COW, checkpoint/donate and reuse.
  Its immutable scope keeps registration readiness false. That transaction had
  122 passing tests; the current plugin suite has 128 after the later Inductor
  inventory transaction.
- Operator artifact provenance now exposes a canonical complete-field digest
  restricted to the explicit matmul/paged-attention/GDN ABI catalog. It is only
  a future join input: physical cache/loading stays compiler-owned and the
  opaque problem digest is not yet a typed launch proof.
- Paged attention now has one reviewed ABI v1 for prefill/extend and decode,
  including explicit KV append/cache mutation, fixed-capacity metadata,
  pitched split-QKV views, workspace-aware non-aliasing, host-copy metadata
  validation, read-only shared radix pages, and zero inactive output rows. No
  CUDA implementation exists yet.
- Paged attention now also has a reviewed standard-library numerical reference
  implementing BF16 storage, FP32-rounded causal GQA, append-before-attention,
  prefill-to-decode cache continuity and inactive-output zeroing. It is not yet
  cross-checked against a CUDA/PyTorch baseline.
- GDN now has a reviewed unified core ABI v1 for decode and prefill/extend,
  freezing packed layout, no-bias causal conv, FP32 gate/delta semantics,
  paired BF16-conv/FP32-recurrent state, pitched envelope views and canonical
  metadata/output tails. State preparation/copy and CUDA remain open.
- GDN now also has a reviewed standard-library paired-state numerical reference.
  One-shot prefill, checkpoint-segmented prefill and token decode match exactly
  in output, BF16 conv state and FP32 recurrent state. It is not yet compared
  against pinned Torch/SGLang.
- The framework plugin now binds actual imports to the locked CUDA Torch tree
  and clean SGLang checkout, rejects mixed linear-attention providers, and
  fails before launch while the real PyPTO Inductor dispatcher is unavailable.
- Its constructor-dispatch foundation now has a reviewed ContextVar mode,
  original-backend preservation outside that mode, pinned wrapper proxy
  semantics, and strict no-fallback failures inside it. It deliberately does
  not register with Torch until real scheduling/wrapper constructors exist.
- The zero-diff TorchInductor route now has a reviewed 31-source SHA/AST
  inventory plus a manifest for all 346 Python files under `_inductor`. It
  freezes registry/cache, scheduler/wrapper/current-stream,
  template/lowering/fallback, and Triton/CuTeDSL/autotune reference lanes.
- The audit proves there is no pinned TileKernel/cuTile abstraction to reuse,
  records the existing CuTeDSL path, the closed native CUDA selector and
  mandatory `has_triton()` check, and exposes the eight actual
  `get_current_backend()` call sites.
- It also freezes unsafe surfaces that the plugin must own or reject: CSEProxy
  omits `pypto` dtype/shape propagation, extern codegen bypasses the backend,
  foreach accepts only three upstream scheduling types, multi-template can
  select extern, and MM/BMM producers contain explicit extern choices.
- Registration remains false. PyPTO scheduling, wrapper/subgraph wrapper,
  CSE propagation, strict choice filtering, atomic registry installation and
  subprocess compilation are explicitly unimplemented. The full plugin suite
  passes 128 tests; wheel and isolated producer-ABI/source audit pass.
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
- A separate clean worktree now contains source-reviewed SM120 TargetInfo
  candidate `9939b88`. It has explicit resource/toolchain/dtype identity and
  fail-closed legacy isolation, but has not been built or imported. It must be
  applied only after the single-DSO CMake transaction is accepted.
- The single-DSO runbook has been repaired and independently approved: it now
  performs a real venv install, audits all wheel DSOs and installed dependency
  closure, verifies every native/binding compile row and enforces the exact
  two-file commit boundary. It remains unexecuted while preflight is red.
- A separate approved TargetInfo runbook freezes the conflict-free ordered
  cherry-pick, fresh native 2/2 CTest, fresh one-DSO wheel/installed-DSO audit,
  31 targeted tests and expected 10,208-pass/58-skip full suite. It also
  remains unexecuted.
- TensorIR source audit proves its `IRuntimeKernel` is not a persistent
  artifact: complete argument/grid metadata lacks versioned serialization,
  runtime initialization is not synchronized, and support/workspace checks are
  TODO. Ambient overrides and wrong SM120 resource fallbacks also prevent a
  deterministic cache/runtime shortcut.
- Decision D-0009 now orders a data-only CompileRequest, per-region
  KernelBuildSpec, exact LLVM/tileiras producer identity, common
  bytes-plus-metadata Artifact, persistent cache and a process/device/CUcontext-
  bound prewarmed executable. Streams remain per-capture/launch values; workers
  never query CUDA or reuse module/context handles.
- Candidate and baseline framework runs now have separate, reviewed execution
  profiles. Baseline cannot import PyPTO/plugin sources or load general SGLang
  plugins; candidate framework launch cannot start until exact environment,
  selected NVIDIA runtime, entry points and process ownership all verify.
- Frozen baseline launch scripts exist for 0.8B and 9B, but neither is runtime
  evidence and the CPython 3.12 baseline environment is not built yet.
- Prior reconnaissance did not complete a TensorIR SM120 runtime launch. Static
  target support is not runtime acceptance.
- The packaging-time safety preflight was green, but the latest required
  recheck is red: zcode is running `zcode-vllm-tp2-v4` with TP=2 vLLM/gem5.
  No PyPTO heavy action began. Protected processes remain
  untouchable; continue light work and rerun preflight after natural exit.

## Resume action

Wait for the observed protected `zcode-vllm-tp2-v4` lane to exit naturally,
then run
`python tools/preflight.py --mode heavy`. If and only if it is green, execute
`docs/single_dso_acceptance_gate.md`, commit only the two CMake files and
persist evidence. Then apply and execute
`docs/target_info_acceptance_gate.md` before any CompileRequest code.
Next build the exact Triton wheel and remove the external editable runtime. If
protected zcode work remains active, continue only source/test work that does
not compile, launch a model, or claim runtime acceptance. Never inherit
TensorIR's SM100 defaults.
