"""Fused residual add: one graph (half of Ascend's fused_add_rmsnorm)."""

from __future__ import annotations

import threading
from typing import Any

from .._boot import bootstrap, launch_graph
from .._graph import pointwise_graph
from .._boot import compile_graph

_lock = threading.RLock()
_cache: dict[tuple, str] = {}

STATUS = "executable"
GRAPHS = 1


def compile_for(shape: tuple[int, ...], dtype_name: str = "bfloat16") -> str:
    cached = _cache.get((shape, dtype_name))
    if cached is not None:
        return cached
    dtype = bootstrap()["pypto"].DataType.BF16 if dtype_name == "bfloat16" \
        else bootstrap()["pypto"].DataType.FP32
    program = pointwise_graph(
        list(shape), dtype, [("tensor.add", ["a", "b"])])
    graph_key = compile_graph(program, [128])
    with _lock:
        _cache[(shape, dtype_name)] = graph_key
    return graph_key


def fused_add(a: Any, b: Any, stream: Any = None) -> Any:
    """out = a + b, one graph launch. Dense matching-shape BF16."""

    import torch

    if stream is None:
        stream = torch.cuda.Stream()
    if a.dtype is not torch.bfloat16 or tuple(a.shape) != tuple(b.shape):
        raise ValueError("fused_add v2 needs matching BF16 operands")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("fused_add v2 needs contiguous operands")
    graph_key = compile_for(tuple(a.shape), "bfloat16")
    out = torch.empty_like(a)
    launch_graph(graph_key, (a, b, out), stream.cuda_stream)
    return out
