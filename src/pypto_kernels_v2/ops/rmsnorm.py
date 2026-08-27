"""RMSNorm: ONE graph, the direct analog of torch_npu.npu_rms_norm.

sum -> scale(1/N) -> +eps -> rsqrt -> broadcast -> mul, all in a single
row-reduction-epilogue graph — no ones-matmul expansion, no multi-launch
orchestration. Status: EXECUTABLE after the broadcast producer and
multidimensional FusedPointwiseV2 schedule landed at CP-0062.
"""

from __future__ import annotations

from typing import Any

from .._boot import classify
from .._graph import row_reduction_epilogue_graph, tiles_for

STATUS = "executable"
GRAPHS = 1


def build(rows: int, cols: int, eps: float = 1e-6) -> Any:
    """The single-graph program for x * rsqrt(mean(x^2) + eps), FP32."""

    return row_reduction_epilogue_graph(rows, cols, eps, 1.0 / cols)


def status(rows: int = 256, cols: int = 1024) -> dict[str, str]:
    """Compile-classification evidence for the single graph.

    The epilogue graph lowers through the fused-pointwise entry, whose
    schedule takes exactly one flattened tile.
    """

    return classify(build(rows, cols), tiles_for(rows, cols))
