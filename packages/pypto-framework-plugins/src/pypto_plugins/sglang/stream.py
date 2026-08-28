"""Current-stream bridge for PyPTO runtimes that reject CUDA default streams."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Iterator


_lock = threading.RLock()
_streams: dict[int, Any] = {}


def _device_index(device: Any) -> int:
    import torch

    rendered = torch.device(device)
    if rendered.type != "cuda":
        raise ValueError("PyPTO SGLang stream bridge requires a CUDA device")
    return torch.cuda.current_device() if rendered.index is None else rendered.index


@contextmanager
def pypto_stream(device: Any) -> Iterator[Any]:
    """Yield a non-default current stream with caller ordering preserved."""

    import torch

    index = _device_index(device)
    caller = torch.cuda.current_stream(index)
    if caller != torch.cuda.default_stream(index):
        yield caller
        return
    with _lock:
        worker = _streams.get(index)
        if worker is None:
            worker = torch.cuda.Stream(device=index)
            _streams[index] = worker
        worker.wait_stream(caller)
        try:
            yield worker
        finally:
            caller.wait_stream(worker)
