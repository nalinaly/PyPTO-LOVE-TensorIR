"""RoPE: single-graph halves (the analog of aclnnApplyRotaryPosEmb).

Each output half is ONE pointwise graph with [M,1] row-broadcast cos/sin
inputs (rotate_half layout); interleaving the halves is layout prep.
Status: BLOCKED-ON-L0 — classify() shows the graphs are valid HIR and
only the producer broadcast lowering rejects them.
"""

from __future__ import annotations

from .._boot import classify
from .._graph import rope_half_graph

STATUS = "blocked-on-L0"
GRAPHS = 2  # even half + odd half; each is a single graph
BLOCKED_ON = "broadcast lowering in the pinned TensorIR producer (codex L0)"


def build_even(rows: int, half: int):
    """x1*cos - x2*sin as one graph."""

    from .._graph import pointwise_graph
    from .._boot import bootstrap
    dtype = bootstrap()["pypto"].DataType.BF16
    return pointwise_graph(
        [rows, half], dtype,
        [("tensor.row_expand_mul", ["x1", "cos"]),
         ("tensor.row_expand_mul", ["x2", "sin"]),
         ("tensor.sub", ["$0", "prev"])],
        broadcast_inputs=["cos", "sin"])


def build_odd(rows: int, half: int):
    """x1*sin + x2*cos as one graph."""

    from .._graph import pointwise_graph
    from .._boot import bootstrap
    dtype = bootstrap()["pypto"].DataType.BF16
    return pointwise_graph(
        [rows, half], dtype,
        [("tensor.row_expand_mul", ["x1", "sin"]),
         ("tensor.row_expand_mul", ["x2", "cos"]),
         ("tensor.add", ["$0", "prev"])],
        broadcast_inputs=["cos", "sin"])


def status(rows: int = 256, half: int = 512) -> dict[str, str]:
    return classify(build_even(rows, half), [128])
