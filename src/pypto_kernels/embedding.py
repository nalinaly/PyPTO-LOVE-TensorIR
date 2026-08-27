"""Token embedding lookup as one native PyPTO tile graph and one launch."""

from __future__ import annotations

import threading
from typing import Any

from ._boot import bootstrap, compile_jit_kernel, launch_graph

bootstrap()
import pypto.language as pl  # noqa: E402

_TILE_WIDTH = 128
_ROW_TILE = 8
_lock = threading.RLock()
_cache: dict[tuple[int, int, int], str] = {}

STATUS = "native-tile executable"
GRAPHS = 1


@pl.jit
def embedding_kernel(
    weight: pl.Tensor,
    repeated_token_ids: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Gather each token row and store explicit hidden-dimension tiles."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(out.shape[0]):
            for block in pl.range(out.shape[1] // 128):
                token_id = pl.read(repeated_token_ids, [row, block * 128])
                value = pl.load(weight, [token_id, block * 128], [1, 128])
                pl.store(value, [row, block * 128], out)
    return out


def _validate_shape(tokens: int, vocab_size: int, hidden_size: int) -> None:
    if tokens <= 0 or vocab_size <= 0 or hidden_size <= 0:
        raise ValueError("embedding needs positive tokens, vocabulary and hidden size")
    if hidden_size % _TILE_WIDTH:
        raise ValueError(
            f"embedding hidden size must be divisible by {_TILE_WIDTH}; "
            f"got {hidden_size}"
        )


def build(tokens: int, vocab_size: int, hidden_size: int) -> Any:
    _validate_shape(tokens, vocab_size, hidden_size)
    import torch

    weight = torch.empty(
        (vocab_size, hidden_size), dtype=torch.bfloat16, device="meta"
    )
    ids = torch.empty((tokens, hidden_size), dtype=torch.int64, device="meta")
    out = torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device="meta")
    return embedding_kernel.specialize(weight, ids, out)


def compile_for(tokens: int, vocab_size: int, hidden_size: int) -> str:
    _validate_shape(tokens, vocab_size, hidden_size)
    key = (tokens, vocab_size, hidden_size)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    import torch

    weight = torch.empty(
        (vocab_size, hidden_size), dtype=torch.bfloat16, device="meta"
    )
    ids = torch.empty((tokens, hidden_size), dtype=torch.int64, device="meta")
    out = torch.empty((tokens, hidden_size), dtype=torch.bfloat16, device="meta")
    graph_key = compile_jit_kernel(
        embedding_kernel,
        (weight, ids, out),
        [_ROW_TILE, _TILE_WIDTH],
    )
    with _lock:
        _cache[key] = graph_key
    return graph_key


def embedding(token_ids: Any, weight: Any, stream: Any = None) -> Any:
    """Gather BF16 embedding rows from INT64 token ids in one launch."""

    import torch

    if (
        token_ids.ndim != 1
        or token_ids.dtype is not torch.int64
        or not token_ids.is_contiguous()
    ):
        raise ValueError("embedding token_ids must be contiguous rank-1 INT64")
    if (
        weight.ndim != 2
        or weight.dtype is not torch.bfloat16
        or not weight.is_contiguous()
    ):
        raise ValueError("embedding weight must be contiguous rank-2 BF16")
    tokens = int(token_ids.shape[0])
    vocab_size, hidden_size = map(int, weight.shape)
    _validate_shape(tokens, vocab_size, hidden_size)
    if stream is None:
        stream = torch.cuda.current_stream(weight.device)
    graph_key = compile_for(tokens, vocab_size, hidden_size)
    out = torch.empty((tokens, hidden_size), dtype=weight.dtype, device=weight.device)
    repeated_ids = token_ids.as_strided(
        (tokens, hidden_size), (1, 0)
    )
    launch_graph(graph_key, (weight, repeated_ids, out), stream.cuda_stream)
    return out


def status(tokens: int = 32, vocab_size: int = 248320,
           hidden_size: int = 1024) -> dict[str, str]:
    try:
        return {
            "status": "compiled",
            "key": compile_for(tokens, vocab_size, hidden_size),
        }
    except RuntimeError as error:
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:
        return {"status": "hir-rejected", "error": str(error)[:200]}
