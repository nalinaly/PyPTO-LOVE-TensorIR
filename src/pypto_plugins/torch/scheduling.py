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
        def codegen_node(self, node: Any) -> None:
            if current_mode() is None:
                return super().codegen_node(node)
            inner = getattr(node, "node", None)
            if not _is_pointwise_node(inner):
                raise StrictCoverageError(
                    "strict PyPTO mode has no kernel for Inductor node "
                    f"{type(inner).__name__!r} yet"
                )
            self._codegen_pypto_pointwise_node(node)

        def _codegen_pypto_pointwise_node(self, node: Any) -> None:
            program, meta = _translate_pointwise(node)
            name = f"pypto_kernel_{len(REGISTRY._kernels)}"
            artifact = pointwise_codegen.compile_pointwise(
                program, tile=meta["tile"], registry_name=name
            )
            REGISTRY.register(name, artifact)
            wrapper = _graph_wrapper_code()
            if not getattr(wrapper, "_pypto_header_written", False):
                wrapper.header.writeline(
                    "from pypto_plugins.torch.runtime_bridge import pypto_launch"
                )
                wrapper.header.writeline("import torch as _pypto_torch")
                wrapper.header.writeline(
                    "pypto_stream = _pypto_torch.cuda.Stream()"
                )
                wrapper._pypto_header_written = True
            for output in node.get_outputs():
                buffer = getattr(output, "node", output)
                if not hasattr(buffer, "get_defining_op"):
                    buffer = getattr(output, "buffer", None)
                if buffer is not None and hasattr(buffer, "get_defining_op"):
                    wrapper.codegen_allocation(buffer)
            stream = "pypto_stream.cuda_stream"
            call_args = _node_call_args(node)
            joined = ", ".join(call_args)
            wrapper.writeline(
                f"pypto_launch({name!r}, ({joined}{', ' if joined else ''}), {stream})"
            )

        def define_kernel(self, src_code: Any, node_schedule: Any, kernel: Any) -> None:
            if current_mode() is None:
                return super().define_kernel(src_code, node_schedule, kernel)
            if not _schedule_is_pointwise(node_schedule):
                raise StrictCoverageError(
                    "strict PyPTO mode has no kernel for this Inductor "
                    "node schedule yet"
                )
            program, meta = _translate_pointwise(kernel)
            name = f"pypto_kernel_{len(REGISTRY._kernels)}"
            artifact = pointwise_codegen.compile_pointwise(
                program, tile=meta["tile"], registry_name=name
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
                wrapper.header.writeline("import torch as _pypto_torch")
                wrapper.header.writeline(
                    "pypto_stream = _pypto_torch.cuda.Stream()"
                )
                wrapper._pypto_header_written = True
            stream = "pypto_stream.cuda_stream"
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


def _is_pointwise_node(inner: Any) -> bool:
    try:
        from torch._inductor.ir import ComputedBuffer, Pointwise
    except Exception:  # pragma: no cover - pinned import path
        return False
    if isinstance(inner, ComputedBuffer):
        inner = inner.data
    return isinstance(inner, Pointwise)


def _node_call_args(node: Any) -> list[str]:
    args: list[str] = []
    for dep in node.read_writes.reads:
        name = getattr(dep, "name", None)
        if not isinstance(name, str):
            name = getattr(
                getattr(dep, "buffer", None), "get_name", lambda: None
            )()
        if isinstance(name, str) and name not in args:
            args.append(name)
    for output in node.get_outputs():
        args.append(output.get_name())
    return args


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


class _OpsRecorder:
    """Record the ops sequence of one pointwise body for HIR translation.

    Every handler attribute is a callable returning a proxy value; loads and
    registered tensor ops append ordered events (loads may interleave with
    ops, so replay follows the event order rather than assuming all loads
    come first), stores become outputs, and anything else fails closed.
    """

    _BINARY = {
        "add": "tensor.add",
        "sub": "tensor.sub",
        "mul": "tensor.mul",
        "div": "tensor.div",
        "truediv": "tensor.div",
        "fdiv": "tensor.div",
        "maximum": "tensor.maximum",
        "minimum": "tensor.minimum",
        "max": "tensor.maximum",
        "min": "tensor.minimum",
    }
    _UNARY = {
        "neg": "tensor.neg",
        "exp": "tensor.exp",
        "recip": "tensor.recip",
        "rsqrt": "tensor.rsqrt",
        "abs": "tensor.abs",
        "sqrt": "tensor.sqrt",
        "log": "tensor.log",
        "sin": "tensor.sin",
        "cos": "tensor.cos",
    }
    _COMPOSED = ("sigmoid", "relu", "tanh")

    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.events: list[tuple[object, ...]] = []
        self.proxies: dict[int, object] = {}
        self._counter = 0

    class _Proxy:
        __slots__ = ("key",)

        def __init__(self, key: int) -> None:
            self.key = key

    def _next_proxy(self) -> "_OpsRecorder._Proxy":
        self._counter += 1
        proxy = self._Proxy(self._counter)
        self.proxies[proxy.key] = proxy
        return proxy

    def _emit(self, op_name: str, operands: list[object]) -> "_OpsRecorder._Proxy":
        self.events.append(("op", op_name, operands))
        return self._next_proxy()

    @staticmethod
    def _is_constant(value: object) -> bool:
        import sympy

        return isinstance(value, (int, float, sympy.Integer, sympy.Float, sympy.Rational))

    def _constant_float(self, value: object) -> float:
        return float(value)

    def _compose(self, name: str, operand: object) -> "_OpsRecorder._Proxy":
        if name == "sigmoid":
            # 1 / (1 + exp(-x)) over registered primitives.
            value = self._emit("tensor.neg", [operand])
            value = self._emit("tensor.exp", [value])
            value = self._emit("tensor.adds", [value, 1.0])
            return self._emit("tensor.recip", [value])
        if name == "relu":
            # (x + |x|) * 0.5 is bitwise-identical to max(x, 0): positive x
            # doubles then halves exactly, non-positive x cancels to +0.
            magnitude = self._emit("tensor.abs", [operand])
            doubled = self._emit("tensor.add", [magnitude, operand])
            return self._emit("tensor.muls", [doubled, 0.5])
        # tanh(x) = 2 / (1 + exp(-2x)) - 1 over registered primitives
        # (tolerance-level, like division, versus the libdevice intrinsic).
        scaled = self._emit("tensor.muls", [operand, -2.0])
        activated = self._emit("tensor.exp", [scaled])
        denominator = self._emit("tensor.adds", [activated, 1.0])
        reciprocal = self._emit("tensor.recip", [denominator])
        doubled = self._emit("tensor.muls", [reciprocal, 2.0])
        return self._emit("tensor.adds", [doubled, -1.0])

    def __getattr__(self, name: str) -> Any:
        def handler(*args: object, **kwargs: object) -> object:
            if name == "load":
                buffer_name = str(args[0])
                if buffer_name not in self.inputs:
                    self.inputs.append(buffer_name)
                self.events.append(("load", buffer_name))
                return self._next_proxy()
            if name == "store":
                value = args[2] if len(args) > 2 else None
                self.outputs.append(str(args[0]))
                self.events.append(
                    ("store", str(args[0]), getattr(value, "key", None))
                )
                return None
            if name == "constant":
                return args[0]
            if name in self._COMPOSED and len(args) == 1 and not self._is_constant(args[0]):
                return self._compose(name, args[0])
            if name in self._BINARY:
                first, second = args[0], args[1]
                if self._is_constant(first) and self._is_constant(second):
                    raise StrictCoverageError(
                        "binary op with two constant operands is not a "
                        "registered form"
                    )
                if self._is_constant(first):
                    # Scalar-left forms commute or decompose over registered
                    # scalar-right ops; each emitted op pairs with exactly
                    # one proxy so the replay keys stay aligned.
                    if name in ("add", "mul", "maximum", "minimum", "max", "min"):
                        self.events.append(
                            ("op", self._BINARY[name] + "s",
                             [second, self._constant_float(first)])
                        )
                        return self._next_proxy()
                    if name in ("truediv", "div", "fdiv"):
                        numerator = self._constant_float(first)
                        reciprocal = self._emit("tensor.recip", [second])
                        if numerator == 1.0:
                            return reciprocal
                        return self._emit("tensor.muls", [reciprocal, numerator])
                    raise StrictCoverageError(
                        "scalar-left binary op is not a registered form"
                    )
                if self._is_constant(second):
                    self.events.append(
                        ("op", self._BINARY[name] + "s",
                         [first, self._constant_float(second)])
                    )
                else:
                    self.events.append(("op", self._BINARY[name], [first, second]))
                return self._next_proxy()
            if name in self._UNARY and len(args) == 1:
                self.events.append(("op", self._UNARY[name], [args[0]]))
                return self._next_proxy()
            raise StrictCoverageError(
                f"strict PyPTO pointwise translation has no op {name!r} yet"
            )

        return handler


def _translate_pointwise(node: Any) -> tuple[Any, dict[str, str | int]]:
    """Translate a pointwise node's real ops sequence into FusedPointwiseV2 HIR.

    The node body executes once against the recording ops handler; loads and
    registered ops rebuild the exact chain in event order (loads may
    interleave with ops and one buffer may be loaded repeatedly), stores
    become outputs, and any other operator fails closed.
    """

    import sympy
    from torch._inductor.virtualized import V

    inner_node = getattr(node, "node", None)
    data = getattr(inner_node, "data", None)
    size_source = data if data is not None else node
    ranges = [int(s) for s in getattr(size_source, "get_size", lambda: [])()]
    if not ranges:
        raise StrictCoverageError("pointwise kernel has no static size hints")
    body = getattr(node, "_body", None)
    if body is None:
        body = getattr(data, "get_reduction_size", None)
    body = getattr(node, "_body", None) or getattr(data, "_body", None)
    if body is None:
        raise StrictCoverageError("pointwise node has no executable body")
    recorder = _OpsRecorder()
    index_vars = [[sympy.Symbol(f"i{index}") for index in range(len(ranges))]]
    with V.set_ops_handler(recorder):
        body(*index_vars)
    if not recorder.inputs or not recorder.outputs or not recorder.events:
        raise StrictCoverageError(
            "pointwise body did not produce a translatable chain "
            f"(inputs={len(recorder.inputs)}, events={len(recorder.events)}, "
            f"outputs={len(recorder.outputs)})"
        )
    dtype_name = "float32"
    builder = pointwise_codegen.PointwiseProgramBuilder(
        tuple(ranges[-1:]), dtype_name
    )
    values: dict[int, Any] = {}
    variables: dict[str, Any] = {}
    stored_keys: list[int] = []
    last_op_key: int | None = None
    next_key = 1
    for event in recorder.events:
        kind = event[0]
        if kind == "load":
            buffer_name = str(event[1])
            variable = variables.get(buffer_name)
            if variable is None:
                variable = builder.add_input(buffer_name)
                variables[buffer_name] = variable
            values[next_key] = variable
            next_key += 1
        elif kind == "op":
            op_name = event[1]
            operands = event[2]
            arguments = []
            for operand in operands:
                if isinstance(operand, _OpsRecorder._Proxy):
                    arguments.append(values[operand.key])
                else:
                    arguments.append(builder.scalar(operand))
            values[next_key] = builder.emit(op_name, arguments)
            last_op_key = next_key
            next_key += 1
        elif kind == "store":
            if event[2] is not None:
                stored_keys.append(int(event[2]))
    output_keys = stored_keys if stored_keys else [last_op_key]
    for output_key in output_keys:
        if output_key is not None:
            builder.mark_output(values[output_key])
    return builder.build(), {"tile": 128}
