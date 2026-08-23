"""``torch.compile(backend='pypto')`` entry point backed by Inductor."""

from __future__ import annotations

from typing import Any

from .torch_inductor import mode


def compile_backend(graph_module: Any, example_inputs: list[Any], **kwargs: Any) -> Any:
    """Compile through full TorchInductor with PyPTO CUDA code generation."""
    from torch._inductor.compile_fx import compile_fx

    with mode(strict=True):
        return compile_fx(graph_module, example_inputs, config_patches=kwargs.get("options"))

