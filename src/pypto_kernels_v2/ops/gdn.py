"""GDN read path: single-graph builders with an explicit status split.

The complete read path is five explicit graphs: q-decay, state matmul, compose,
dot reduction and delta+read. State update is a separate one-graph rank-3
broadcast DAG. Every graph compiles and has SM120 numerical evidence; no wrapper
hides the five read launches behind a single call.
"""

from __future__ import annotations

from .._boot import classify
from .._graph import (gdn_compose_graph, gdn_delta_combine_graph,
                      gdn_delta_graph, gdn_q_decay_graph, row_sum_graph,
                      gdn_state_read_graph, gdn_state_update_graph, tiles_for)

STATUS = "complete read path and single-graph state update executable"
GRAPHS = 5  # q-decay, state matmul, compose, dot reduction, delta+read
UPDATE_GRAPHS = 1


def compose_status(heads: int = 16, dk: int = 128) -> dict[str, str]:
    return classify(gdn_compose_graph(heads, dk), [128])


def delta_status(heads: int = 16, dv: int = 128) -> dict[str, str]:
    return classify(gdn_delta_graph(heads, dv), tiles_for(heads, dv))


def q_decay_status(heads: int = 16, dk: int = 128) -> dict[str, str]:
    return classify(gdn_q_decay_graph(heads, dk), [128])


def dot_status(heads: int = 16, dk: int = 128) -> dict[str, str]:
    tile = tiles_for(heads)
    return classify(row_sum_graph(heads, dk), tile)


def state_read_status(heads: int = 16, dk: int = 128,
                      dv: int = 128) -> dict[str, str]:
    return classify(gdn_state_read_graph(heads, dk, dv),
                    tiles_for(heads, dv))


def delta_combine_status(heads: int = 16,
                         dv: int = 128) -> dict[str, str]:
    return classify(gdn_delta_combine_graph(heads, dv),
                    tiles_for(heads, dv))


def state_update_status(heads: int = 16, dk: int = 128,
                        dv: int = 128) -> dict[str, str]:
    return classify(gdn_state_update_graph(heads, dk, dv),
                    tiles_for(heads, dk, dv))
