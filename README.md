# PyPTO Framework Plugins

Commit-pinned compatibility adapters that connect the PyPTO compiler and the
standalone `pypto-kernels` operator library to official, unmodified PyTorch and
SGLang installations.

This project owns registration, framework metadata translation, strict
fallback auditing, and version guards. It must not contain CUDA Tile kernel
algorithms, model-specific compiler primitives, or copied framework source.
Its declared runtime dependency is the separately versioned `pypto-kernels`
wheel. Registration validates the operator project's producer-owned canonical
framework-adapter ABI manifest and digest instead of copying a partial schema.
It also pins installed-distribution ownership, import origin, and the canonical
Python source-tree digest before touching a framework backend. ABI and source
identity are not compiled-kernel readiness; registration remains fail-closed
until real PyPTO executables exist.

This source-only stage rejects native extension modules and sourceless bytecode
inside the operator package. Future compiled executables must gain their own
digest-bound payload manifest and readiness proof before this guard is widened.

Supported framework baseline:

- PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`
- SGLang `71de97b264b04dcd514cf904003028aefe9775c8`

The adapters fail closed when a framework identity or expected hook contract
does not match. Support for the actual PyPTO scheduling/wrapper and SGLang
backend classes is intentionally gated by explicit readiness flags as the
compiler and operator projects are brought up.

The current Torch layer also defines the context-local mode and constructor
dispatch contract needed to coexist with Inductor's process-global CUDA slot.
It preserves the original CUDA backend outside PyPTO mode and rejects missing,
C++-wrapper, and FX-wrapper paths inside PyPTO mode. It still performs no
registration because real PyPTO scheduling and wrapper constructors have not
landed; `install()` remains deliberately fail-closed.

The SGLang layer freezes a source-hashed, AST-checked inventory of the Qwen3.5
CUDA text path. Every direct fused helper and backend abstraction is assigned
to a future strict provider (`pypto.generic`, `matmul`, `attention`, or `gdn`),
while host-only shape arithmetic is recorded separately. This inventory is a
coverage obligation, not a kernel implementation or model correctness claim.

A framework-neutral coverage auditor checks adapter-supplied normalized GPU
activities against a closed-world trace manifest and a digest-bound PyPTO
artifact registry, then publishes deterministic, durable JSON. Strict mode
rejects any unregistered or non-PyPTO model-forward compute event immediately;
it also rejects incomplete, empty, zero-time, misclassified, or digest-mismatched
evidence. The normalizer protocol revision is fixed. Each report path is held
under an interprocess lock for the auditor lifetime, cannot overwrite an
existing run, and only the owning instance may advance its exact partial report
to final. Sampling and CUDA runtime memcpy/memset activities remain visible but
are deliberately outside the model-forward compute denominator.

The evidence boundary is explicit: this module does not collect GPU activity,
prove that the adapter delivered a complete event stream, or validate the
adapter's timing method. A `strict_policy_passed` result is therefore an exact
policy decision over the bound normalized trace; it is not by itself model
correctness, profiler completeness, or performance evidence. No runtime trace
has been collected yet.
