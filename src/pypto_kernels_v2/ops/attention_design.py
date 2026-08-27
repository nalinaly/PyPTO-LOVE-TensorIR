"""Attention: 2-graph design (the floor before a dedicated FA graph kind).

Ascend runs FlashAttention as ONE kernel via loops over KV blocks; the
PyPTO graph op set has no cross-family (reduction+matmul) single graph
yet, so the honest Ascend-style floor is:

  graph 1  normalize precomputed positive exponent values
           — row reduction + reciprocal + broadcast epilogue
  graph 2  probs @ V — StructuredMatmulV4

Both graphs compile, launch and pass numerical acceptance. QK, scale, max-shift
and exp are outside this post-exp acceptance boundary; true one-kernel FA still
needs a dedicated graph kind.
"""

from __future__ import annotations

STATUS = "post-exp normalization/value-mix path executable"
GRAPHS = 2
FUTURE = ("a dedicated flash-attention graph kind (KV-block loops inside "
          "one graph) for true single-kernel parity with Ascend")


# --- single-graph builder for the softmax broadcast stage ---

from .._boot import classify  # noqa: E402
from .._graph import (matmul_graph, row_normalize_graph, softmax_scale_graph,
                      tiles_for)  # noqa: E402


def build_softmax_scale(rows: int, tokens: int):
    """p = e * (1/sum(e)) as ONE pointwise graph with [M,1] broadcast."""

    return softmax_scale_graph(rows, tokens)


def softmax_status(rows: int = 256, tokens: int = 1024) -> dict[str, str]:
    return classify(build_softmax_scale(rows, tokens), tiles_for(rows, tokens))


def build_softmax_normalize(rows: int, tokens: int):
    """Normalize positive exponent values in one reduction-epilogue graph."""

    return row_normalize_graph(rows, tokens)


def softmax_normalize_status(rows: int = 256,
                             tokens: int = 128) -> dict[str, str]:
    return classify(build_softmax_normalize(rows, tokens),
                    tiles_for(rows, tokens))


def build_value_mix(rows: int, tokens: int, value_dim: int):
    """probs @ value as one StructuredMatmulV4 graph."""

    return matmul_graph([rows, tokens], [tokens, value_dim])


def value_mix_status(rows: int = 256, tokens: int = 128,
                     value_dim: int = 128) -> dict[str, str]:
    return classify(build_value_mix(rows, tokens, value_dim),
                    tiles_for(rows, value_dim))
