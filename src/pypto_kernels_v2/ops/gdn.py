"""GDN read path: single-graph builders with an explicit status split.

compose (q*(softplus(g)*k)) and the delta broadcast (dot*v over the value
dimension) are executable single graphs after CP-0062. The complete read path
and state update remain the next v2 construction task.
"""

from __future__ import annotations

from .._boot import classify
from .._graph import gdn_compose_graph, gdn_delta_graph, tiles_for

STATUS = "compose and delta executable; full read/update pending"
GRAPHS = 5  # compose, row_sum, recip, delta-broadcast, state-read matmul


def compose_status(heads: int = 16, dk: int = 128) -> dict[str, str]:
    return classify(gdn_compose_graph(heads, dk), [128])


def delta_status(heads: int = 16, dv: int = 128) -> dict[str, str]:
    return classify(gdn_delta_graph(heads, dv), tiles_for(heads, dv))
