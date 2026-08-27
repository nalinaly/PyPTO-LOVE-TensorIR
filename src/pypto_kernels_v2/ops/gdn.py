"""GDN read path: single-graph builders with an explicit status split.

compose (q*(softplus(g)*k)) is pure pointwise and EXECUTABLE today; the
delta broadcast (dot*v over the value dim) and the state-update are the
broadcast-dependent parts — BLOCKED-ON-L0, same dependency as the v1
analysis (HANDOVER W2/L0b).
"""

from __future__ import annotations

from .._boot import classify
from .._graph import gdn_compose_graph, gdn_delta_graph

STATUS = "mixed: compose executable, delta broadcast blocked-on-L0"
GRAPHS = 5  # compose, row_sum, recip, delta-broadcast, state-read matmul
BLOCKED_ON = "broadcast lowering in the pinned TensorIR producer (codex L0)"


def compose_status(heads: int = 16, dk: int = 128) -> dict[str, str]:
    return classify(gdn_compose_graph(heads, dk), [128])


def delta_status(heads: int = 16, dv: int = 128) -> dict[str, str]:
    return classify(gdn_delta_graph(heads, dv), [128])
