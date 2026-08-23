# PyPTO Framework Plugins

Commit-pinned compatibility adapters that connect the PyPTO compiler and the
standalone `pypto-kernels` operator library to official, unmodified PyTorch and
SGLang installations.

This project owns registration, framework metadata translation, strict
fallback auditing, and version guards. It must not contain CUDA Tile kernel
algorithms, model-specific compiler primitives, or copied framework source.

Supported framework baseline:

- PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`
- SGLang `71de97b264b04dcd514cf904003028aefe9775c8`

The adapters fail closed when a framework identity or expected hook contract
does not match. Support for the actual PyPTO scheduling/wrapper and SGLang
backend classes is intentionally gated by explicit readiness flags as the
compiler and operator projects are brought up.

