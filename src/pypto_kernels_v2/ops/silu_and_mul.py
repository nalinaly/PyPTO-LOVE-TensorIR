"""SiLU-and-mul: one graph (the Ascend SwiGLU fused-operator analog).

silu(g) * u with silu composed over registered pointwise primitives:
neg -> exp -> +1 -> recip -> *g -> *u, all inside ONE FusedPointwiseV2
graph, one launch. Status: EXECUTABLE (pure pointwise, no broadcast).
"""

from __future__ import annotations

import threading
from typing import Any

from .._boot import bootstrap, launch_graph
from .._graph import pointwise_graph

_lock = threading.RLock()
_cache: dict[tuple, str] = {}

STATUS = "executable"
GRAPHS = 1


def _key(shape: tuple[int, ...], dtype_name: str) -> tuple:
    return ("silu_and_mul", shape, dtype_name)


def compile_for(shape: tuple[int, ...], dtype_name: str = "bfloat16") -> str:
    from .._boot import compile_graph
    cached = _cache.get(_key(shape, dtype_name))
    if cached is not None:
        return cached
    dtype = bootstrap()["pypto"].DataType.BF16 if dtype_name == "bfloat16" \
        else bootstrap()["pypto"].DataType.FP32
    program = pointwise_graph(
        list(shape), dtype,
        [("tensor.neg", ["g"]),
         ("tensor.exp", ["prev"]),
         ("tensor.adds", ["prev", 1.0]),
         ("tensor.recip", ["prev"]),
         ("tensor.mul", ["prev", "g"]),
         ("tensor.mul", ["prev", "u"])])
    graph_key = compile_graph(program, [128])
    with _lock:
        _cache[_key(shape, dtype_name)] = graph_key
    return graph_key


def silu_and_mul(gate: Any, up: Any, stream: Any = None) -> Any:
    """out = silu(gate) * up, one graph launch. Dense [M, N] BF16."""

    import torch

    if stream is None:
        stream = torch.cuda.Stream()
    if gate.dtype is not torch.bfloat16 or up.dtype is not torch.bfloat16:
        raise ValueError("silu_and_mul v2 needs BF16 operands")
    shape = tuple(gate.shape)
    if tuple(up.shape) != shape or not gate.is_contiguous():
        raise ValueError("silu_and_mul v2 needs matching contiguous operands")
    graph_key = compile_for(shape, "bfloat16")
    out = torch.empty(shape, dtype=torch.bfloat16, device=gate.device)
    launch_graph(graph_key, (gate, up, out), stream.cuda_stream)
    return out
