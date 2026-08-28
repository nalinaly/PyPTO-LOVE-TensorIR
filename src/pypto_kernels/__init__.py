"""Qwen3.5 native tile operators over PyPTO.

One model operator is one explicit ``@pl.jit`` tile graph and one launch.
"""

__version__ = "0.1.0"

__all__ = (
    "attention",
    "causal_conv1d",
    "embedding",
    "fused_add_rmsnorm",
    "gdn",
    "gdn_projection",
    "gated_rmsnorm",
    "linear",
    "qk_rmsnorm_rope",
    "rmsnorm",
    "rope",
    "sigmoid_mul",
    "silu_and_mul",
)
