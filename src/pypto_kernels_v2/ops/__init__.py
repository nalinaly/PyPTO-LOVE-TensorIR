"""v2 operators: one model operator = one PyPTO graph (Ascend style).

Status vocabulary (see _boot.classify):
  executable        — single graph, compiles and launches on SM120 today
  blocked-on-L0     — single-graph HIR is valid; the pinned producer's
                      broadcast lowering (codex L0) is the only blocker
"""

from . import (attention_design, fused_add, gdn, rmsnorm, rope,  # noqa: F401
                 silu_and_mul)

__all__ = ("attention_design", "fused_add", "gdn", "rmsnorm", "rope",
           "silu_and_mul")
