"""RMSNorm: ONE graph, the direct analog of torch_npu.npu_rms_norm.

sum -> scale(1/N) -> +eps -> rsqrt -> broadcast -> mul, all in a single
row-reduction-epilogue graph — no ones-matmul expansion, no multi-launch
orchestration. Status: BLOCKED-ON-L0 — the HIR is accepted by the PyPTO
emission layer; only the pinned producer's broadcast lowering (codex L0)
rejects it today. classify() proves the split: the error is a producer
RuntimeError, not an emission ValueError.
"""

from __future__ import annotations

from typing import Any

from .._boot import classify
from .._graph import row_reduction_epilogue_graph

STATUS = "blocked-on-L0"
GRAPHS = 1
BLOCKED_ON = "broadcast lowering in the pinned TensorIR producer (codex L0)"


def build(rows: int, cols: int, eps: float = 1e-6) -> Any:
    """The single-graph program for x * rsqrt(mean(x^2) + eps), FP32."""

    return row_reduction_epilogue_graph(rows, cols, eps, 1.0 / cols)


def status(rows: int = 256, cols: int = 1024) -> dict[str, str]:
    """Compile-classification evidence for the single graph.

    The epilogue graph lowers through the fused-pointwise entry, whose
    schedule takes exactly one flattened tile.
    """

    return classify(build(rows, cols), [128])
