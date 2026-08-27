"""Attention: 2-graph design (the floor before a dedicated FA graph kind).

Ascend runs FlashAttention as ONE kernel via loops over KV blocks; the
PyPTO graph op set has no cross-family (reduction+matmul) single graph
yet, so the honest Ascend-style floor is:

  graph 1  softmax(QK^T * scale)  — row reduction + broadcast epilogue
  graph 2  probs @ V              — StructuredMatmulV4

The broadcast scale graph is executable after CP-0062. Composing the complete
softmax segment and the value-mix graph is the next v2 task; true one-kernel FA
still needs a dedicated graph kind.
"""

from __future__ import annotations

STATUS = "softmax scale executable; complete two-graph attention pending"
GRAPHS = 2
FUTURE = ("a dedicated flash-attention graph kind (KV-block loops inside "
          "one graph) for true single-kernel parity with Ascend")


# --- single-graph builder for the softmax broadcast stage ---

from .._boot import classify  # noqa: E402
from .._graph import softmax_scale_graph, tiles_for  # noqa: E402


def build_softmax_scale(rows: int, tokens: int):
    """p = e * (1/sum(e)) as ONE pointwise graph with [M,1] broadcast."""

    return softmax_scale_graph(rows, tokens)


def softmax_status(rows: int = 256, tokens: int = 1024) -> dict[str, str]:
    return classify(build_softmax_scale(rows, tokens), tiles_for(rows, tokens))
