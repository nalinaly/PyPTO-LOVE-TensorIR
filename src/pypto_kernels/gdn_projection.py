"""Qwen3.5 GDN projection split through one packed PyPTO result."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import bootstrap, compile_graph, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_lock = threading.RLock()
_cache: dict[tuple[int, int, int, int, int, int, int], str] = {}

STATUS = "native-tile packed executable"
GRAPHS = 1


@pl.jit
def gdn_projection_kernel(
    projected_qkvz: pl.Tensor,
    projected_ba: pl.Tensor,
    mixed_width: pl.INT64,
    packed_out: pl.Out[pl.Tensor],
):
    """Pack four output-major projection segments into one physical result."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(projected_qkvz.shape[0]):
            mixed = pl.load(
                projected_qkvz,
                [row, 0],
                [1, mixed_width],
            )
            z_flat = pl.load(
                projected_qkvz,
                [row, mixed_width],
                [1, projected_qkvz.shape[1] - mixed_width],
            )
            b = pl.load(
                projected_ba,
                [row, 0],
                [1, projected_ba.shape[1] // 2],
            )
            a = pl.load(
                projected_ba,
                [row, projected_ba.shape[1] // 2],
                [1, projected_ba.shape[1] // 2],
            )
            pl.store(mixed, [0, row * mixed_width], packed_out)
            pl.store(
                z_flat,
                [
                    0,
                    projected_qkvz.shape[0] * mixed_width
                    + row * (projected_qkvz.shape[1] - mixed_width),
                ],
                packed_out,
            )
            pl.store(
                b,
                [
                    0,
                    projected_qkvz.shape[0] * projected_qkvz.shape[1]
                    + row * (projected_ba.shape[1] // 2),
                ],
                packed_out,
            )
            pl.store(
                a,
                [
                    0,
                    projected_qkvz.shape[0]
                    * (projected_qkvz.shape[1] + projected_ba.shape[1] // 2)
                    + row * (projected_ba.shape[1] // 2),
                ],
                packed_out,
            )
    return packed_out


def _validate_shape(
    rows: int,
    q_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
) -> None:
    if (
        rows <= 0
        or q_heads <= 0
        or value_heads <= 0
        or value_heads % q_heads
        or key_dim <= 0
        or key_dim % 128
        or value_dim <= 0
        or value_dim % 16
    ):
        raise ValueError("GDN projection received invalid static geometry")


def build(
    rows: int,
    q_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    qkvz_row_stride: int | None = None,
    ba_row_stride: int | None = None,
) -> Any:
    _validate_shape(rows, q_heads, value_heads, key_dim, value_dim)
    import torch

    mixed_width = 2 * q_heads * key_dim + value_heads * value_dim
    z_width = value_heads * value_dim
    qkvz_width = mixed_width + z_width
    ba_width = 2 * value_heads
    qkvz_row_stride = qkvz_width if qkvz_row_stride is None else qkvz_row_stride
    ba_row_stride = ba_width if ba_row_stride is None else ba_row_stride
    if qkvz_row_stride < qkvz_width or ba_row_stride < ba_width:
        raise ValueError("GDN projection row strides must cover each logical row")
    qkvz = torch.empty_strided(
        (rows, qkvz_width),
        (qkvz_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    ba = torch.empty_strided(
        (rows, ba_width),
        (ba_row_stride, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    packed_elements = rows * (mixed_width + z_width + 2 * value_heads)
    packed = torch.empty((1, packed_elements), dtype=torch.bfloat16, device="meta")
    return gdn_projection_kernel.specialize(qkvz, ba, mixed_width, packed)


def compile_for(
    rows: int,
    q_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    qkvz_row_stride: int,
    ba_row_stride: int,
) -> str:
    _validate_shape(rows, q_heads, value_heads, key_dim, value_dim)
    shape_key = (
        rows,
        q_heads,
        value_heads,
        key_dim,
        value_dim,
        qkvz_row_stride,
        ba_row_stride,
    )
    cached = _cache.get(shape_key)
    if cached is not None:
        return cached
    graph_key = compile_graph(
        build(
            rows,
            q_heads,
            value_heads,
            key_dim,
            value_dim,
            qkvz_row_stride,
            ba_row_stride,
        ),
        [1, 16],
        provider="pypto.matmul",
        source_node="pypto_kernels.gdn_projection:projection",
    )
    with _lock:
        _cache[shape_key] = graph_key
    return graph_key


def split_projection(
    projected_qkvz: Any,
    projected_ba: Any,
    *,
    q_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    stream: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """Return contiguous ``(mixed_qkv, z, b, a)`` from one PyPTO launch."""

    import torch

    if (
        projected_qkvz.ndim != 2
        or projected_ba.ndim != 2
        or projected_qkvz.dtype is not torch.bfloat16
        or projected_ba.dtype is not torch.bfloat16
        or projected_qkvz.stride(1) != 1
        or projected_ba.stride(1) != 1
        or projected_qkvz.device != projected_ba.device
    ):
        raise ValueError(
            "GDN projection inputs must be rank-2 BF16 with unit inner strides"
        )
    rows = int(projected_qkvz.shape[0])
    _validate_shape(rows, q_heads, value_heads, key_dim, value_dim)
    mixed_width = 2 * q_heads * key_dim + value_heads * value_dim
    z_width = value_heads * value_dim
    if tuple(projected_qkvz.shape) != (rows, mixed_width + z_width) or tuple(
        projected_ba.shape
    ) != (rows, 2 * value_heads):
        raise ValueError("GDN projection input widths are incompatible")
    if stream is None:
        stream = torch.cuda.current_stream(projected_qkvz.device)
    if (
        projected_qkvz.stride(0) < projected_qkvz.shape[1]
        or projected_ba.stride(0) < projected_ba.shape[1]
    ):
        raise ValueError("GDN projection row strides overlap logical rows")
    graph_key = compile_for(
        rows,
        q_heads,
        value_heads,
        key_dim,
        value_dim,
        int(projected_qkvz.stride(0)),
        int(projected_ba.stride(0)),
    )
    component_sizes = (
        rows * mixed_width,
        rows * z_width,
        rows * value_heads,
        rows * value_heads,
    )
    packed = torch.empty(
        (1, sum(component_sizes)),
        dtype=torch.bfloat16,
        device=projected_qkvz.device,
    )
    launch_graph(
        graph_key,
        (projected_qkvz, projected_ba, packed),
        stream.cuda_stream,
    )
    flat = packed.view(-1)
    offsets = (
        0,
        component_sizes[0],
        sum(component_sizes[:2]),
        sum(component_sizes[:3]),
    )
    mixed = flat.narrow(0, offsets[0], component_sizes[0]).view(rows, mixed_width)
    z = flat.narrow(0, offsets[1], component_sizes[1]).view(
        rows, value_heads, value_dim
    )
    b = flat.narrow(0, offsets[2], component_sizes[2]).view(rows, value_heads)
    a = flat.narrow(0, offsets[3], component_sizes[3]).view(rows, value_heads)
    return mixed, z, b, a
