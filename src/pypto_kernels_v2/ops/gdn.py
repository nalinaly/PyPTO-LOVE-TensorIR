"""GDN decode read: ONE graph (read path), update separately.

out = (q*decay) @ S + (q . (softplus(g)*k)) * v as a single pointwise
graph with row-broadcast operands for the dot expansion and the state
read as an M=1 matmul graph. Status: read BLOCKED-ON-L0 (broadcast
operands); the state update additionally needs the same broadcast
support (see HANDOVER W2/L0b).
"""

from __future__ import annotations

STATUS = "blocked-on-L0"
GRAPHS = 2
BLOCKED_ON = ("broadcast lowering in the pinned TensorIR producer "
              "(codex L0); the state update is the same dependency "
              "(HANDOVER L0b)")
