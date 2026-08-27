"""RoPE: ONE graph (the analog of aclnnApplyRotaryPosEmb / RotaryMul).

out_even = x_e*cos[p] - x_o*sin[p]; out_odd = x_e*sin[p] + x_o*cos[p]
with cos/sin tables as [M,1] row-broadcast inputs — a single pointwise
graph with row-expand operands. Positions are static per compile, so the
tables are data prep (not counted as compute). Status: BLOCKED-ON-L0
(row-broadcast operands are exactly what the producer rejects today).
"""

from __future__ import annotations

STATUS = "blocked-on-L0"
GRAPHS = 1
BLOCKED_ON = "broadcast lowering in the pinned TensorIR producer (codex L0)"
