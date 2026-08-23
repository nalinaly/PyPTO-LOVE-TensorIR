"""TorchInductor installation contract.

The source-provided CUDA device registry is process global while Inductor
configuration is context-local. The real scheduling/wrapper registration is
added only after the PyPTO compiler/runtime ABI exists; until then this module
fails loudly rather than delegating to Triton under a PyPTO label.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

from .errors import BackendNotReadyError
from .versions import assert_torch_compatible


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


def install() -> None:
    """Install the process-global CUDA dispatcher exactly once."""
    global _INSTALLED
    assert_torch_compatible()
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        raise BackendNotReadyError(
            "PyPTO Inductor scheduling/wrapper is not implemented yet; refusing to register a fallback backend."
        )


@contextlib.contextmanager
def mode(*, strict: bool = True) -> Iterator[None]:
    """Enter a per-compile PyPTO Inductor context."""
    install()
    # Reached only after install() has a real dispatcher implementation.
    from torch._inductor import config

    with config.patch({"cuda_backend": "pypto"}):
        yield

