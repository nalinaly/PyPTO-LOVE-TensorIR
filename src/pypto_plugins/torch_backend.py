"""``torch.compile(backend='pypto')`` entry point backed by Inductor."""

from __future__ import annotations

from typing import Any

from .operator_library import assert_operator_library_compatible
from .torch_inductor import STRICT_INDUCTOR_PATCHES, install, mode


def _compile_config_patches(options: Any) -> dict[str, object]:
    """Return mandatory patches after rejecting every unreviewed option."""

    if options in (None, {}):
        return dict(STRICT_INDUCTOR_PATCHES)
    if not isinstance(options, dict):
        raise TypeError(f"PyPTO backend options must be a dict, got {type(options)}")
    raise ValueError(
        "PyPTO backend options are not enabled yet; refusing unreviewed config "
        f"keys {sorted(options)}"
    )


def compile_backend(graph_module: Any, example_inputs: list[Any], **kwargs: Any) -> Any:
    """Compile through full TorchInductor with PyPTO CUDA code generation."""
    try:
        assert_operator_library_compatible()
        unknown = sorted(set(kwargs) - {"options"})
        if unknown:
            raise TypeError(f"unsupported PyPTO backend keyword arguments: {unknown}")
        patches = _compile_config_patches(kwargs.get("options"))
        install()
    except SystemExit:
        raise
    except Exception as error:
        raise SystemExit(f"PyPTO backend preflight failed: {error}") from error
    with mode(strict=True):
        from torch._inductor.compile_fx import compile_fx

        return compile_fx(graph_module, example_inputs, config_patches=patches)
