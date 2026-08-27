"""Attention: 2-graph design (the floor before a dedicated FA graph kind).

Ascend runs FlashAttention as ONE kernel via loops over KV blocks; the
PyPTO graph op set has no cross-family (reduction+matmul) single graph
yet, so the honest Ascend-style floor is:

  graph 1  softmax(QK^T * scale)  — row reduction + broadcast epilogue
  graph 2  probs @ V              — StructuredMatmulV4

Both halves are EXPRESSIBLE today (graph 1 needs the generalized epilogue
analyzer in the pypto compiler — codex territory; broadcast lowering L0).
This module records the design and the dependency; v1's accepted kernels
remain the executable stand-in until then.
"""

from __future__ import annotations

STATUS = "blocked-on-L0 (graph 1 needs generalized epilogue + broadcast)"
GRAPHS = 2
BLOCKED_ON = ("pypto epilogue analyzer generalization (softmax: "
              "sum -> 1/sum -> broadcast-mul) + producer broadcast "
              "lowering (codex L0); graph 2 is plain StructuredMatmulV4")
FUTURE = ("a dedicated flash-attention graph kind (KV-block loops inside "
          "one graph) for true single-kernel parity with Ascend")


# --- single-graph builder for the softmax broadcast stage ---

from .._boot import classify  # noqa: E402
from .._graph import softmax_scale_graph  # noqa: E402


def build_softmax_scale(rows: int, tokens: int):
    """p = e * (1/sum(e)) as ONE pointwise graph with [M,1] broadcast."""

    return softmax_scale_graph(rows, tokens)


def softmax_status(rows: int = 256, tokens: int = 1024) -> dict[str, str]:
    return classify(build_softmax_scale(rows, tokens), [128])
