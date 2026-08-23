"""TorchInductor installation contract.

The source-provided CUDA device registry is process global while Inductor
configuration is context-local. The real scheduling/wrapper registration is
added only after the PyPTO compiler/runtime ABI exists; until then this module
fails loudly rather than delegating to Triton under a PyPTO label.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator

from .errors import BackendNotReadyError
from .versions import assert_torch_compatible


_INSTALL_LOCK = threading.RLock()
_INSTALLED = False


STRICT_INDUCTOR_PATCHES = {
    "cuda_backend": "pypto",
    "implicit_fallbacks": False,
}
STRICT_DYNAMO_PATCHES = {
    "suppress_errors": False,
    "fail_on_recompile_limit_hit": True,
    "skip_code_recursive_on_recompile_limit_hit": False,
}


def prepare_process_strict() -> None:
    """Disable process-level Dynamo/Inductor fallback before compilation."""

    fx_compile_mode = os.environ.get("TORCHINDUCTOR_FX_COMPILE_MODE", "NORMAL")
    if fx_compile_mode.upper() != "NORMAL":
        raise SystemExit(
            "PyPTO v1 requires in-process FX compilation; "
            f"got TORCHINDUCTOR_FX_COMPILE_MODE={fx_compile_mode!r}"
        )
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config

    dynamo_config.suppress_errors = False
    dynamo_config.fail_on_recompile_limit_hit = True
    dynamo_config.skip_code_recursive_on_recompile_limit_hit = False
    inductor_config.implicit_fallbacks = False
    requested_suppression = os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", "").lower()
    if requested_suppression not in ("", "0", "false", "off", "no"):
        raise SystemExit(
            "TORCHDYNAMO_SUPPRESS_ERRORS conflicts with strict PyPTO coverage"
        )


def install() -> None:
    """Install the process-global CUDA dispatcher exactly once."""
    global _INSTALLED
    prepare_process_strict()
    assert_torch_compatible()
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        raise BackendNotReadyError(
            "PyPTO Inductor scheduling/wrapper is not implemented yet; "
            "refusing to register a fallback backend."
        )


@contextlib.contextmanager
def mode(*, strict: bool = True) -> Iterator[None]:
    """Enter a per-compile PyPTO Inductor context."""
    install()
    # Reached only after install() has a real dispatcher implementation.
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config

    inductor_patches = {"cuda_backend": "pypto"}
    dynamo_patches: dict[str, object] = {}
    if strict:
        inductor_patches.update(STRICT_INDUCTOR_PATCHES)
        dynamo_patches.update(STRICT_DYNAMO_PATCHES)
    with dynamo_config.patch(dynamo_patches), inductor_config.patch(inductor_patches):
        yield
