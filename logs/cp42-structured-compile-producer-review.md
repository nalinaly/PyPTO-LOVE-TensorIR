# CP-0042 pinned-producer identity and facade review

Reviewer: `/root/tensorir_abi_map`

Decision: `GO`

- P0: 0
- P1: 0
- P2: 0

The final review verified the hard-wired one-producer route from structured
preparation through normalized producer output, final BuildSpec and Artifact.
The private move-only carrier seals exact source, request and producer-options
identity before the invocation can be consumed. Artifact creation recomputes
those identities from the final request and BuildSpec and rejects source,
request or options rebinding before provenance construction.

The review also verified that legacy `Artifact::CompileStrict` wire behavior
and early process claiming remain intact, the public facade returns only the
joined immutable result, backend-OFF fails closed, real producer success and
failure paths have exact invocation deltas, and the narrow Artifact compiler
object remains free of frontend unresolved symbols.

This review accepts source-level producer reachability and identity joining
only. It does not accept CUDA module load, kernel launch, numerical correctness,
framework integration or performance.
