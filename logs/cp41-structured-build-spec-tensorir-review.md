# CP-0041 pinned TensorIR ABI reachability review

Reviewer: `/root/tensorir_abi_map`

Decision: `GO`

- P0: 0
- P1: 0
- P2: 0

The final review verified that FP32 `[8,8]`, tile `16`, grid `{4,1,1}` matches
the already accepted pinned TensorIR producer fixture, while BF16 `[128]`, tile
`16`, grid `{8,1,1}` satisfies the pinned power-of-two, tile-bound and flattened
static-grid rules. It independently recomputed the source and projection
digests and confirmed the metadata-schema gate, producer-shaped `y=z=1`,
private projection construction, explicit dtype/role limitations, implicit
rank-one stride coverage, complete ABI-drift matrix, final KernelBuildSpec
round-trip and unchanged legacy Artifact route.

This review accepts only the compile-free preparation/finalization contract.
It does not accept a real producer call through that contract or any device
execution.
