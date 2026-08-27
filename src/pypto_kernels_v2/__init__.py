"""pypto-kernels v2: Ascend-style high-level operators over PyPTO.

One model operator = one PyPTO TensorIR graph + tile schedule; running an
operator is compile-once-per-shape, launch-per-call. See README.md and
docs/ascend_style_evidence.md in the workspace root for the rationale.
"""
