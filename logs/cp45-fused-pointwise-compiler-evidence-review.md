# CP45 bounded fused-pointwise compiler evidence review

## Verdict

GO. P0/P1/P2 = 0/0/0.

Candidate `b83fcd3ddc497d585bcc45883eede179aff7d4d2`, tree
`49eda98f3ed8d72bfd14d5a5900cdc0e71ca699d`, has sole parent
`642ff5bd79ee96b9e5a279a2bc945ad7a78362b7`. Exactly ten intended
source/test/doc files differ; the worktree and all six pinned gitlinks are
clean and exact. No build artifact is committed.

## Compiler evidence

- Backend-OFF product DSO SHA-256 is
  `eb4225cc741fb992a8715419c021770a1ea2bac323382d73b3d452c7f3d7ab80`.
  Native 3/3 and exact-product Python 1/1 pass with JUnit hashes
  `4d805ed4...05195` and `958b4f7a...c4dfc`.
- Backend-ON product DSO SHA-256 is
  `0e8f33c263e06777aec06263bf32ca59ac554868529f3fa085212cf27e2facbe`.
  Native 3/3 and exact-product Python 1/1 pass with JUnit hashes
  `b3a57b2c...81140` and `44a155a6...9431d`.
- Both DSOs are RPATH/RUNPATH-free and depend only on libstdc++, libm,
  libgcc_s, libc, and the ELF loader.
- The ON product binds exact PyPTO, TensorIR, CUDA Tile, LLVM, CUDA 13.3,
  tileiras and `sm_120a` identities. Its exact-product Python lane reasserts
  every field and both import origins.
- Native coverage includes deterministic V2 all-op source, FP32/BF16 real
  TensorIR/CUDA Tile Cubin production, the 16-input/64-assignment boundary,
  schedule/grid/descriptor/mutation/workspace validation, and serialization/
  identity joins.
- Legacy V1 classification remains first. CP44 FP32 `[8,8]` and BF16 `[128]`
  source, five projection domains, callable ABI, 13,784-byte Cubins, and Cubin
  hashes remain exact.

## Diagnostic lineage and isolation

The initial OFF build paused only its verified owned PGID at the 16 GiB floor,
then reached its owned timeout; a fresh continuation completed. An in-source ON
build correctly failed the strict clean-source guard and is retained as a
diagnostic. The accepted ON configure/build uses an external workspace-local
binary directory while compiling the exact clean worktree. No `.git/info`
exclusion or provenance relaxation was used.

All accepted process sidecars join their preflight hashes, have no coexistence
abort, report no NVIDIA compute PID, request no GPU smoke, and use no protected
waiver. No external process was signalled.

## Evidence boundary

Accepted: exact source/gitlinks, OFF/ON products, V1 byte preservation, bounded
V2 HIR/source/projections/schedule/ABI, and CPU compiler production of nonempty
SM120 Cubins.

Not accepted: V2 GPU launch, numerical/transcendental correctness, V2 Cubin
byte determinism, CUDA Graph/current-stream runtime behavior, performance,
full PyPTO regression, TorchInductor, SGLang, Qwen, or strict coverage.
