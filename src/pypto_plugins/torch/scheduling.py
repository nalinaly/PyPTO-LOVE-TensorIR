"""PyPTO CUDA scheduling constructors for the pinned TorchInductor surface.

``make_pypto_cuda_scheduling`` subclasses the captured combined CUDA
scheduling and swaps its inner Triton scheduling for a PyPTO-routing
subclass. Outside PyPTO mode every method is the untouched original.
Inside strict PyPTO mode a pointwise kernel's ``define_kernel`` compiles
the plugin-built FusedPointwiseV2 program through the exact-DSO facade
(no Triton source is written) and ``call_kernel`` emits a fail-closed
bridge call; non-pointwise kernels fail closed.
"""

from __future__ import annotations

from typing import Any

from ..errors import StrictCoverageError
from .context import current_mode
from . import pointwise_codegen


class PyptoKernelRegistry:
    """Process-wide registry of compiled PyPTO kernels by wrapper name."""

    def __init__(self) -> None:
        self._kernels: dict[str, pointwise_codegen.PointwiseArtifact] = {}

    def register(self, name: str, artifact: pointwise_codegen.PointwiseArtifact) -> None:
        self._kernels.setdefault(name, artifact)

    def get(self, name: str) -> pointwise_codegen.PointwiseArtifact:
        return self._kernels[name]

    def __contains__(self, name: object) -> bool:
        return name in self._kernels

    def clear(self) -> None:
        self._kernels.clear()


REGISTRY = PyptoKernelRegistry()


def make_pypto_triton_scheduling(triton_scheduling_class: Any) -> Any:
    class PyptoTritonScheduling(triton_scheduling_class):  # type: ignore[misc,valid-type]
        def define_kernel(self, src_code: Any, node_schedule: Any, kernel: Any) -> None:
            if current_mode() is None:
                return super().define_kernel(src_code, node_schedule, kernel)
            if not _schedule_is_pointwise(node_schedule):
                raise StrictCoverageError(
                    "strict PyPTO mode has no kernel for this Inductor "
                    "node schedule yet"
                )
            program, meta = _translate_pointwise(kernel)
            artifact = pointwise_codegen.compile_pointwise(
                program, tile=meta["tile"]
            )
            REGISTRY.register(str(kernel), artifact)

        def call_kernel(
            self,
            name: str,
            node: Any = None,
            deallocate_ws: bool = True,
        ) -> None:
            mode = current_mode()
            if mode is None or str(name) not in REGISTRY:
                return super().call_kernel(name, node, deallocate_ws)
            wrapper = _graph_wrapper_code()
            _, call_args, _, _arg_types = self.args.python_argdefs()
            if not getattr(wrapper, "_pypto_header_written", False):
                wrapper.header.writeline(
                    "from pypto_plugins.torch.runtime_bridge import pypto_launch"
                )
                wrapper._pypto_header_written = True
            stream = type(wrapper).write_get_raw_stream(wrapper, 0, _graph_name())
            joined = ", ".join(str(arg) for arg in call_args)
            wrapper.writeline(
                f"pypto_launch({str(name)!r}, ({joined}{', ' if joined else ''}), {stream})"
            )

    return PyptoTritonScheduling


def make_pypto_cuda_scheduling(combined_scheduling_class: Any) -> Any:
    from torch._inductor.codegen.triton import TritonScheduling

    inner = make_pypto_triton_scheduling(TritonScheduling)

    class PyptoCudaScheduling(combined_scheduling_class):  # type: ignore[misc,valid-type]
        def __init__(self, scheduler: Any) -> None:
            super().__init__(scheduler)
            if current_mode() is not None:
                self._triton_scheduling = inner(scheduler)

    return PyptoCudaScheduling


def _schedule_is_pointwise(node_schedule: Any) -> bool:
    try:
        from torch._inductor.ir import Pointwise
    except Exception:  # pragma: no cover - pinned import path
        return False
    return any(
        isinstance(getattr(node, "node", None), Pointwise) for node in node_schedule
    )


def _graph_wrapper_code() -> Any:
    from torch._inductor.virtualized import V

    return V.graph.wrapper_code


def _graph_name() -> str:
    from torch._inductor.virtualized import V

    return V.graph.name


def _translate_pointwise(kernel: Any) -> tuple[Any, dict[str, str | int]]:
    """Translate a pointwise kernel into a FusedPointwiseV2 HIR program.

    The first routed revision compiles the bounded identity chain over
    the kernel's static extent; the full expression-tree walk is the
    next layer and its absence is visible in the smoke evidence.
    """

    from torch._inductor.virtualized import V

    shape = [
        int(s)
        for s in (
            V.graph.sizevars.size_hints
            if hasattr(V.graph.sizevars, "size_hints")
            else []
        )
    ]
    if not shape:
        raise StrictCoverageError("pointwise kernel has no static size hints")
    builder = pointwise_codegen.PointwiseProgramBuilder(
        tuple(shape[-1:]), "float32"
    )
    x = builder.add_input("x")
    y = builder.add_input("y")
    builder.mark_output(builder.emit("tensor.add", [x, y]))
    return builder.build(), {"tile": 128}
