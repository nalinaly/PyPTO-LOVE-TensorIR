"""Qwen3.5 native tile operators over PyPTO.

One model operator is one explicit ``@pl.jit`` tile graph and one launch.
"""

__all__ = (
    "attention",
    "causal_conv1d",
    "fused_add",
    "fused_add_rmsnorm",
    "gdn",
    "gated_rmsnorm",
    "linear",
    "rmsnorm",
    "rope",
    "silu_and_mul",
)
