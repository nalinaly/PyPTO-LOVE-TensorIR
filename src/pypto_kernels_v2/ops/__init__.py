"""v2 operators: one model operator = one PyPTO graph (Ascend style).

Status vocabulary (see _boot.classify):
  compiled          — the single graph lowers through the strict producer
  executable        — compiled, one-launch and numerically accepted on SM120
"""

from . import (attention_design, fused_add, gdn, rmsnorm, rope,  # noqa: F401
                 silu_and_mul)

__all__ = ("attention_design", "fused_add", "gdn", "rmsnorm", "rope",
           "silu_and_mul")
