"""Gated DeltaNet read and update operators for Qwen3.5."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import bootstrap, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_lock = threading.RLock()
_read_cache: dict[tuple[int, int, int], str] = {}
_update_cache: dict[tuple[int, int, int], str] = {}

STATUS = "native-tile read executable"
GRAPHS = 1
UPDATE_GRAPHS = 1


@pl.jit
def gdn_read_kernel(
    query: pl.Tensor,
    decay: pl.Tensor,
    gate: pl.Tensor,
    key: pl.Tensor,
    value: pl.Tensor,
    state: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Read recurrent state and add the gated delta term per head."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for head in pl.range(query.shape[0]):
            query_tile = pl.load(query, [head, 0], [1, query.shape[1]])
            decay_tile = pl.load(decay, [head, 0], [1, query.shape[1]])
            gate_tile = pl.load(gate, [head, 0], [1, query.shape[1]])
            key_tile = pl.load(key, [head, 0], [1, query.shape[1]])
            value_tile = pl.load(value, [head, 0], [1, value.shape[1]])
            state_box = pl.load(
                state,
                [head, 0, 0],
                [1, query.shape[1], value.shape[1]],
                target_memory=pl.MemorySpace.Mat,
            )
            state_tile = pl.reshape(state_box, [query.shape[1], value.shape[1]])

            query_decay = pl.mul(query_tile, decay_tile)
            state_read = pl.matmul(query_decay, state_tile, out_dtype=pl.FP32)

            query_wide = pl.cast(query_tile, target_type=pl.FP32)
            gate_wide = pl.cast(gate_tile, target_type=pl.FP32)
            key_wide = pl.cast(key_tile, target_type=pl.FP32)
            value_wide = pl.cast(value_tile, target_type=pl.FP32)
            gate_exp = pl.exp(gate_wide)
            gate_shifted = pl.add(gate_exp, 1.0)
            gate_softplus = pl.log(gate_shifted)
            gated_key = pl.mul(gate_softplus, key_wide)
            composed = pl.mul(query_wide, gated_key)
            dot_scratch = pl.create_tile(
                [1, query.shape[1]],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            dot = pl.row_sum(composed, dot_scratch)
            delta = pl.row_expand_mul(value_wide, dot)
            combined = pl.add(state_read, delta)
            result = pl.cast(combined, target_type=pl.BF16)
            pl.store(result, [head, 0], out)
    return out


@pl.jit
def gdn_state_update_kernel(
    state: pl.Tensor,
    decay: pl.Tensor,
    beta_key: pl.Tensor,
    value: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Apply decay and the beta-key/value outer product per head."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for head in pl.range(state.shape[0]):
            state_box = pl.load(
                state,
                [head, 0, 0],
                [1, state.shape[1], state.shape[2]],
            )
            decay_box = pl.load(decay, [head, 0, 0], [1, state.shape[1], 1])
            beta_box = pl.load(beta_key, [head, 0, 0], [1, state.shape[1], 1])
            value_box = pl.load(value, [head, 0, 0], [1, 1, state.shape[2]])
            state_tile = pl.reshape(state_box, [state.shape[1], state.shape[2]])
            decay_tile = pl.reshape(decay_box, [state.shape[1], 1])
            beta_tile = pl.reshape(beta_box, [state.shape[1], 1])
            value_tile = pl.reshape(value_box, [1, state.shape[2]])
            decayed = pl.row_expand_mul(state_tile, decay_tile)
            beta_full = pl.row_expand(state_tile, beta_tile)
            outer = pl.col_expand_mul(beta_full, value_tile)
            updated = pl.add(decayed, outer)
            result = pl.reshape(updated, [1, state.shape[1], state.shape[2]])
            pl.store(result, [head, 0, 0], out)
    return out


def _validate_shape(heads: int, key_dim: int, value_dim: int) -> None:
    if heads <= 0 or key_dim <= 0 or value_dim <= 0 or key_dim % 128 or value_dim % 16:
        raise ValueError(
            "GDN read needs positive dimensions, key_dim divisible by 128 and "
            "value_dim divisible by 16"
        )


def build_read(heads: int, key_dim: int, value_dim: int) -> Any:
    _validate_shape(heads, key_dim, value_dim)
    import torch

    key_like = torch.empty((heads, key_dim), dtype=torch.bfloat16, device="meta")
    value = torch.empty((heads, value_dim), dtype=torch.bfloat16, device="meta")
    state = torch.empty(
        (heads, key_dim, value_dim), dtype=torch.bfloat16, device="meta"
    )
    return gdn_read_kernel.specialize(
        key_like, key_like, key_like, key_like, value, state, value
    )


def compile_read(heads: int, key_dim: int, value_dim: int) -> str:
    _validate_shape(heads, key_dim, value_dim)
    shape_key = (heads, key_dim, value_dim)
    cached = _read_cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    key_like = torch.empty((heads, key_dim), dtype=torch.bfloat16, device="meta")
    value = torch.empty((heads, value_dim), dtype=torch.bfloat16, device="meta")
    state = torch.empty(
        (heads, key_dim, value_dim), dtype=torch.bfloat16, device="meta"
    )
    graph_key = compile_jit_kernel(
        gdn_read_kernel,
        (key_like, key_like, key_like, key_like, value, state, value),
        [1, 64],
    )
    with _lock:
        _read_cache[shape_key] = graph_key
    return graph_key


def gdn_read(
    query: Any,
    decay: Any,
    gate: Any,
    key: Any,
    value: Any,
    state: Any,
    stream: Any = None,
) -> Any:
    """Return the complete GDN read result from one graph launch."""

    import torch

    tensors = (query, decay, gate, key, value, state)
    if any(
        tensor.dtype is not torch.bfloat16 or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError("GDN read needs contiguous BF16 tensors")
    if query.ndim != 2 or value.ndim != 2 or state.ndim != 3:
        raise ValueError("GDN read expects rank-2 vectors and rank-3 state")
    heads, key_dim = (int(query.shape[0]), int(query.shape[1]))
    value_dim = int(value.shape[1])
    if any(tuple(tensor.shape) != (heads, key_dim) for tensor in (decay, gate, key)):
        raise ValueError("GDN query/decay/gate/key shapes must match")
    if tuple(value.shape) != (heads, value_dim) or tuple(state.shape) != (
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError("GDN value/state shapes are incompatible")
    if stream is None:
        stream = torch.cuda.current_stream(query.device)
    graph_key = compile_read(heads, key_dim, value_dim)
    out = torch.empty_like(value)
    launch_graph(
        graph_key,
        (query, decay, gate, key, value, state, out),
        stream.cuda_stream,
    )
    return out


def read_status(
    heads: int = 16, key_dim: int = 128, value_dim: int = 128
) -> dict[str, str]:
    try:
        return {
            "status": "compiled",
            "key": compile_read(heads, key_dim, value_dim),
        }
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}


def build_state_update(heads: int, key_dim: int, value_dim: int) -> Any:
    _validate_shape(heads, key_dim, value_dim)
    import torch

    state = torch.empty(
        (heads, key_dim, value_dim), dtype=torch.bfloat16, device="meta"
    )
    decay = torch.empty((heads, key_dim, 1), dtype=torch.bfloat16, device="meta")
    value = torch.empty((heads, 1, value_dim), dtype=torch.bfloat16, device="meta")
    return gdn_state_update_kernel.specialize(state, decay, decay, value, state)


def compile_state_update(heads: int, key_dim: int, value_dim: int) -> str:
    _validate_shape(heads, key_dim, value_dim)
    shape_key = (heads, key_dim, value_dim)
    cached = _update_cache.get(shape_key)
    if cached is not None:
        return cached

    import torch

    state = torch.empty(
        (heads, key_dim, value_dim), dtype=torch.bfloat16, device="meta"
    )
    decay = torch.empty((heads, key_dim, 1), dtype=torch.bfloat16, device="meta")
    value = torch.empty((heads, 1, value_dim), dtype=torch.bfloat16, device="meta")
    graph_key = compile_jit_kernel(
        gdn_state_update_kernel,
        (state, decay, decay, value, state),
        [1, 32, 32],
    )
    with _lock:
        _update_cache[shape_key] = graph_key
    return graph_key


def gdn_state_update(
    state: Any,
    decay: Any,
    beta_key: Any,
    value: Any,
    stream: Any = None,
) -> Any:
    """Return the updated recurrent state from one native tile graph launch."""

    import torch

    tensors = (state, decay, beta_key, value)
    if any(
        tensor.dtype is not torch.bfloat16 or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError("GDN state update needs contiguous BF16 tensors")
    if state.ndim != 3:
        raise ValueError("GDN state must be rank 3")
    heads, key_dim, value_dim = map(int, state.shape)
    if tuple(decay.shape) != (heads, key_dim, 1) or tuple(beta_key.shape) != (
        heads,
        key_dim,
        1,
    ):
        raise ValueError("GDN decay and beta_key shapes are incompatible")
    if tuple(value.shape) != (heads, 1, value_dim):
        raise ValueError("GDN value shape is incompatible")
    if stream is None:
        stream = torch.cuda.current_stream(state.device)
    graph_key = compile_state_update(heads, key_dim, value_dim)
    out = torch.empty_like(state)
    launch_graph(
        graph_key,
        (state, decay, beta_key, value, out),
        stream.cuda_stream,
    )
    return out


def state_update_status(
    heads: int = 16, key_dim: int = 128, value_dim: int = 128
) -> dict[str, str]:
    try:
        return {
            "status": "compiled",
            "key": compile_state_update(heads, key_dim, value_dim),
        }
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
