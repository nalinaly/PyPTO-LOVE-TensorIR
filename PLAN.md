# PLAN

**Plan:** `PYPTO-NVIDIA-QWEN35-V1`, revision `3`

## Current phase: R0 workspace and provenance bootstrap

Checkpoint `CP-0005` has source/model/environment provenance, the first
standalone semantic operator layer, Torch constructor-dispatch law, and
candidate/baseline process isolation frozen. R0 remains open until exact Triton
wheels replace the inherited editable source, the CPython 3.12 baseline
environment is locked, and unmodified SGLang baselines are captured. In
parallel, the first P1 build-only object boundary is staged but is not accepted
until its full-test and fresh-wheel gates close.

Checkpoint `CP-0006` additionally freezes the strict runtime-coverage evidence
contract: fixed collector revision, closed trace and artifact-registry digest
reconciliation, exact non-vacuous denominators, failure latching, and exclusive
report ownership. It is deliberately not runtime coverage evidence. The next
source-only P1 transaction is immutable SM120 target identity, but it must land
only after the staged single-DSO object-boundary commit is independently gated.

Checkpoint `CP-0007` adds catalog-bound operator provenance identity and the
pinned eager-first CUPTI collector map without moving either boundary into the
wrong project. The collector is not implemented and cannot assert a closed
world yet.

Checkpoint `CP-0008` freezes paged-attention ABI v1 and its metadata reference
contract. This is the semantic boundary for later reference and CUDA Tile
implementations, not an attention correctness or performance milestone.

1. Create the control repository, persistence documents, safety preflight, and
   isolated directory layout.
2. Materialize the authorized PyPTO baseline and clean official upstream
   checkouts at the exact versions in `VERSIONS.lock`.
3. Initialize independent `pypto-kernels` and `pypto-framework-plugins` Git
   projects.
4. Clone the `triton-dev` environment into a project-local prefix without
   modifying the original environment.
5. Copy Qwen3.5 weights from the read-only AMD simulator tree into `models/`
   only when protected zcode/gem5 workloads are idle; verify every hash.
6. Generate the checkout-grounded `docs/implementation_map.txt` and freeze the
   unmodified SGLang baseline before compiler changes.

## Milestone ladder

- R0: workspace/provenance/baseline.
- P1: PyPTO compiler/backend split with unchanged Ascend tests.
- P2: TensorIR/CUDA Tile SM120 runtime closure and PyTorch current-stream ABI.
- P3: generic fused-loop codegen, structured matmul, runtime/cache/tuning.
- P4: zero-diff TorchInductor compatibility plugin and strict MLP gate.
- P5: paged full-attention correctness and performance.
- P6: GDN decode/prefill correctness and performance.
- P7: zero-diff SGLang plugin and Qwen3.5-0.8B strict coverage.
- P8: 0.8B stabilization and full profiling.
- P9: Qwen3.5-9B correctness, strict coverage, and SM120 tuning.
- P10: final E2E benchmarks, coverage proof, and performance report.

Every milestone is correctness-first, then performance, then evidence and a
checkpoint commit. A green smoke test is never promoted to a later acceptance
claim.
