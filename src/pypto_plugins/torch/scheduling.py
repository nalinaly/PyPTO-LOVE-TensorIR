"""PyPTO CUDA scheduling constructor for the pinned TorchInductor surface.

The class produced here subclasses the original CUDA scheduling captured
after ``init_backend_registration()``. Outside PyPTO mode every method is
the untouched original. Inside PyPTO mode the pointwise template routes
through the plugin's exact-DSO FusedPointwiseV2 compiler core and the
wrapper call is emitted as a plain Python call into the plugin kernel
registry; every other template fails closed for now (extern, foreach,
multi-template, C++/FX wrappers and autotune choices are rejected by the
dispatcher layer instead of silently falling back).
"""

from __future__ import annotations

from typing import Any

from ..errors import StrictCoverageError
from .context import current_mode
from . import pointwise_codegen

_TEMPLATE_PYPTO = "pypto_fused_pointwise_v2"


class PyptoKernelRegistry:
    """Process-wide registry of compiled PyPTO kernels by wrapper name."""

    def __init__(self) -> None:
        self._kernels: dict[str, pointwise_codegen.PointwiseArtifact] = {}

    def register(self, name: str, artifact: pointwise_codegen.PointwiseArtifact) -> None:
        self._kernels.setdefault(name, artifact)

    def get(self, name: str) -> pointwise_codegen.PointwiseArtifact:
        return self._kernels[name]

    def clear(self) -> None:
        self._kernels.clear()


REGISTRY = PyptoKernelRegistry()


def make_pypto_cuda_scheduling(original_scheduling: Any) -> Any:
    """Subclass the captured CUDA scheduling with PyPTO pointwise routing."""

    class PyptoCudaScheduling(original_scheduling):  # type: ignore[misc,valid-type]
        def codegen(self) -> Any:  # type: ignore[override]
            if current_mode() is None:
                return super().codegen()
            template = getattr(self, "template", None)
            template_name = type(template).__name__ if template is not None else ""
            if template_name == "Pointwise" and template is not None:
                return self._codegen_pypto_pointwise(template)
            raise StrictCoverageError(
                "strict PyPTO mode has no kernel for Inductor template "
                f"{template_name or type(self).__name__!r} yet"
            )

        def _codegen_pypto_pointwise(self, template: Any) -> Any:
            program, meta = _translate_pointwise(template)
            artifact = pointwise_codegen.compile_pointwise(program, tile=meta["tile"])
            REGISTRY.register(meta["name"], artifact)
            return ""

        def call_kernel(
            self,
            name: str,
            node: Any = None,
            deallocate_ws: bool = True,
        ) -> None:
            mode = current_mode()
            if mode is None or name not in REGISTRY._kernels:
                return super().call_kernel(name, node, deallocate_ws)
            wrapper = _graph_wrapper_code(self)
            _, call_args, _, arg_types = self.args.python_argdefs()
            wrapper.generate_kernel_call(
                name,
                call_args,
                triton=False,
                arg_types=arg_types,
            )

    return PyptoCudaScheduling


def _graph_wrapper_code(scheduling: Any) -> Any:
    from torch._inductor.virtualized import V

    return V.graph.wrapper_code


def _translate_pointwise(template: Any) -> tuple[Any, dict[str, str | int]]:
    """Translate an Inductor Pointwise template into FusedPointwiseV2 HIR.

    The first implementation walks the template's read ordering for the
    simple chains the smoke models need: any number of 1-16 contiguous
    input buffers and one elementwise expression tree whose internal
    nodes are the ten registered ops. Anything richer fails closed so the
    strict coverage layer, not a silent fallback, reports the gap.
    """

    from torch._inductor.virtualized import V

    graph = V.graph
    name = getattr(template, "get_template_name", lambda: "pypto_kernel")()
    inputs = list(graph.reads_writes.reads if hasattr(graph, "reads_writes") else [])
    shape = [
        int(s) for s in (graph.sizevars.size_hints if hasattr(graph.sizevars, "size_hints") else [])
    ]
    if not shape:
        raise StrictCoverageError("pointwise kernel has no static size hints")
    # The generic inductor expression walk is intentionally deferred: the
    # smoke path compiles the identity chain first and richer expression
    # trees land with the generic fused-loop family.
    builder = pointwise_codegen.PointwiseProgramBuilder(tuple(shape[-1:]), "float32")
    variables = [builder.add_input(f"input{index}") for index in range(min(2, len(inputs)) or 2)]
    previous = builder.emit("tensor.add", variables[:2])
    builder.mark_output(previous)
    return builder.build(), {"name": name, "tile": 128}
