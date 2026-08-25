# CP-0044 frontend SM120 GPU evidence review

Reviewer: `/root/frontend_gpu_evidence_review`

Decision: `GO_FRONTEND_SM120_VECTOR_ADD_V1`

- P0: 0
- P1: 0
- P2: 0

The independent audit verified canonical final report size 32,676 bytes, mode
`0444`, and SHA-256
`8dbbfbf3ed791cc38d552fbd8f37e34f60d1d0262e9626195ab084b530f228e8`.
Reconstructing the complete report from retained sidecars and replay blobs
produced byte-for-byte equality. Re-finalization passed all validation and
failed only at no-replace publication, leaving the existing report inode,
mtime, mode, size and hash unchanged.

All root/control/parent-manifest, PyPTO/six-gitlink, exact-DSO, CompileRequest,
HIR, BuildSpec, Artifact and Cubin identities join. Two
`compile_structured_strict` calls produced two Artifacts. Four fresh
`NvidiaExecutable` lifetimes completed on the caller's non-default stream with
capture-free launch, external synchronization, unchanged inputs, matching
references, packet release, explicit unload and terminal `Unloaded` state.

Independent CPU reconstruction reproduced all four input and output hashes,
including the BF16 FP32-add/round-once reference. Preflight, gate, barrier,
child, periodic and post-exit audits contain no external/protected NVIDIA
compute PID, protected NVIDIA runtime mapping or unreadable protected map.
Fallback is disabled and absent; Triton, SGLang and FlashInfer were not
imported.

Accepted scope is only the fixed FP32 `[8,8]` and BF16 `[128]` vector-add
fixtures through HIR, one-producer structured compilation, Artifact/Cubin and
`NvidiaExecutable` on the real RTX 5090 Laptop SM120. This review does not
accept general vector-add/BF16 behavior, another operator, CUDA Graph,
performance, TorchInductor, SGLang, Qwen or strict model coverage.
