# CP-0043 frontend SM120 control first review

Reviewer: `/root/frontend_smoke_v1_review`

Decision: `NO-GO` for superseded commit
`5904db38b3fb6e9447c4ba4a7a2a801b0b10558b`

- P0: 0
- P1: 3
- P2: 2

The first review found one deterministic ArtifactTarget API error, a materially
weaker finalizer than the accepted v4 boundary, and no end-to-end synthetic
finalization test. It also found inaccurate claims about pre-child manifest
validation, replay publication timing and instrumented CUDA API counts.

Before publication, the implementation was amended to use the real
`ArtifactTarget.compute_capability` API, restore exact nested process/preflight/
gate/barrier/audit schemas, recursively reject broader claims, reject extra
replay files, add a complete synthetic `finalize()`/no-replace/mutation suite,
and narrow all documentation and final evidence claims. The superseded commit
identifier is retained only in this review record; it was never merged,
manifested or executed on a GPU.
