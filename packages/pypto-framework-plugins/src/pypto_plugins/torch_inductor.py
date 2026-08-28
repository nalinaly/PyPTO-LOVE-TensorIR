"""TorchInductor installation contract for the native PyPTO CUDA backend."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator

from .operator_library import assert_operator_library_compatible
from .torch import registration
from .torch.context import activate_mode
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
            "PyPTO requires in-process FX compilation; "
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


def assert_backend_executable_ready() -> None:
    """Verify that scheduling, wrapper and runtime bridge entry points exist."""

    from .torch import runtime_bridge, scheduling

    required = (
        registration.install,
        registration.uninstall,
        scheduling.make_pypto_cuda_scheduling,
        runtime_bridge.pypto_launch,
    )
    if not all(callable(value) for value in required):
        raise RuntimeError("PyPTO Inductor executable components are incomplete")


def install() -> None:
    """Install the process-global CUDA dispatcher exactly once."""
    global _INSTALLED
    assert_operator_library_compatible()
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        assert_backend_executable_ready()
        assert_torch_compatible()
        prepare_process_strict()
        registration.install()
        if not registration.installed():
            raise RuntimeError("PyPTO CUDA dispatcher installation did not publish")
        _INSTALLED = True


def uninstall() -> None:
    """Restore TorchInductor's captured CUDA backend registration."""

    global _INSTALLED
    with _INSTALL_LOCK:
        registration.uninstall()
        _INSTALLED = False


@contextlib.contextmanager
def mode(*, strict: bool = True) -> Iterator[None]:
    """Enter a per-compile PyPTO Inductor context."""
    install()
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config

    inductor_patches = {"cuda_backend": "pypto"}
    dynamo_patches: dict[str, object] = {}
    if strict:
        inductor_patches.update(STRICT_INDUCTOR_PATCHES)
        dynamo_patches.update(STRICT_DYNAMO_PATCHES)
    with activate_mode(strict=strict):
        with (
            dynamo_config.patch(dynamo_patches),
            inductor_config.patch(inductor_patches),
        ):
            yield
