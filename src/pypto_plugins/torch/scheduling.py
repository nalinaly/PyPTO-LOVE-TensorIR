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
            nodes = [node]
            if type(node).__name__ == "FusedSchedulerNode":
                removed = getattr(getattr(self, "scheduler", None), "removed_ops", ())
                nodes = [
                    sub_node
                    for sub_node in node.get_nodes()
                    if sub_node.get_name() not in removed
                ]
            for sub_node in nodes:
                inner = getattr(sub_node, "node", None)
                import os as _os
                if _os.environ.get("PYPTO_DEBUG_NODES"):
                    import sys as _sys
                    data = getattr(inner, "data", None)
                    print(
                        f"NODE node={type(sub_node).__name__} "
                        f"inner={type(inner).__name__} data={type(data).__name__}",
                        file=_sys.stderr,
                        flush=True,
                    )
                if _is_pointwise_node(inner):
                    self._codegen_pypto_pointwise_node(sub_node)
                    continue
                if _is_reduction_node(inner):
                    self._codegen_pypto_reduction_node(sub_node)
                    continue
                raise StrictCoverageError(
                    "strict PyPTO mode has no kernel for Inductor node "
                    f"{type(inner).__name__!r} yet"
                )

        def _codegen_pypto_reduction_node(self, node: Any) -> None:
            program = _translate_reduction(node)
            name = f"pypto_kernel_{len(REGISTRY._kernels)}"
            artifact = pointwise_codegen.compile_pointwise(
                program, tile=128, registry_name=name
            )
            REGISTRY.register(name, artifact)
            _emit_pypto_node_launch(node, name, meta)

        def _codegen_pypto_pointwise_node(self, node: Any) -> None:
            program, meta = _translate_pointwise(node)
            name = f"pypto_kernel_{len(REGISTRY._kernels)}"
            artifact = pointwise_codegen.compile_pointwise(
                program, tile=meta["tile"], registry_name=name
            )
            REGISTRY.register(name, artifact)
            _emit_pypto_node_launch(node, name, meta)

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


def _is_reduction_node(inner: Any) -> bool:
    try:
        from torch._inductor.ir import ComputedBuffer, Reduction
    except Exception:  # pragma: no cover - pinned import path
        return False
    if isinstance(inner, ComputedBuffer):
        inner = inner.data
    return isinstance(inner, Reduction)


def _translate_reduction(node: Any) -> Any:
    """Build the RowReductionV3 HIR for a trailing-axis Inductor reduction."""

    inner = getattr(node, "node", None)
    data = getattr(inner, "data", None)
    if data is None:
        data = inner
    reduction_type = data.get_reduction_type()
    if reduction_type == "sum":
        op_name = "tensor.row_sum"
    elif reduction_type == "max":
        op_name = "tensor.row_max"
    else:
        raise StrictCoverageError(
            f"strict PyPTO reduction has no mode {reduction_type!r} yet"
        )
    outer = [int(extent) for extent in data.get_size()]
    reduced = [int(extent) for extent in data.get_reduction_size()]
    if not outer or not reduced:
        raise StrictCoverageError("reduction needs both outer and reduction extents")
    # A keepdim reduction reports the [M,1] output extent inside the outer
    # loop ranges; those trailing unit slots are not input dimensions.
    while len(outer) > 1 and outer[-1] == 1:
        outer.pop()
    input_shape = [*outer, *reduced]
    modules = pointwise_codegen.bootstrap_pypto()
    ir = modules["ir"]
    pypto = modules["pypto"]
    dtype = pypto.DataType.FP32
    span = ir.Span("pypto_plugins.scheduling", 1, 1)
    input_type = ir.TensorType(input_shape, dtype)
    result_type = ir.TensorType([*input_shape[:-1], 1], dtype)
    input_value = ir.Var("input", input_type, span)
    result = ir.Var("result", result_type, span)
    call = ir.Call(ir.get_op(op_name), [input_value], result_type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span
    )
    function = ir.Function(
        "ignored_row_reduction_name", [input_value], [result_type], body, span
    )
    return ir.Program([function], "pypto_plugins_row_reduction", span)


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


def _force_dense_output_layout(output: Any) -> None:
    """Pin a PyPTO-produced buffer to dense row-major strides.

    The PyPTO kernel ABI carries dense static strides in the artifact, and
    Inductor's layout heuristics can pick exotic strides (a [M,1] row with
    stride (1,M)) for intermediates; every layout consumer in this path is
    index-based, so dense is always safe here.
    """
    buffer = getattr(output, "node", output)
    if not hasattr(buffer, "get_size") or not hasattr(buffer, "layout"):
        return
    try:
        from torch._inductor.ir import FixedLayout

        size = [int(extent) for extent in buffer.get_size()]
        strides: list[int] = []
        running = 1
        for extent in reversed(size):
            strides.append(running)
            running *= extent
        strides.reverse()
        buffer.layout = FixedLayout(
            buffer.get_device(), buffer.get_dtype(), size, strides
        )
    except Exception:  # pragma: no cover - layout pinning is best-effort
        pass


def _emit_pypto_node_launch(node: Any, name: str, meta: dict | None = None) -> None:
    wrapper = _graph_wrapper_code()
    if not getattr(wrapper, "_pypto_header_written", False):
        wrapper.header.writeline(
            "from pypto_plugins.torch.runtime_bridge import pypto_launch"
        )
        wrapper.header.writeline("import torch as _pypto_torch")
        wrapper.header.writeline("pypto_stream = _pypto_torch.cuda.Stream()")
        wrapper._pypto_header_written = True
    for output in node.get_outputs():
        buffer = getattr(output, "node", output)
        _force_dense_output_layout(buffer)
        if not hasattr(buffer, "get_defining_op"):
            buffer = getattr(output, "buffer", None)
        if buffer is not None and hasattr(buffer, "get_defining_op"):
            wrapper.codegen_allocation(buffer)
    stream = "pypto_stream.cuda_stream"
    broadcast_buffers = set((meta or {}).get("broadcast_buffers", ()))
    output_shape = (meta or {}).get("output_shape") or []
    call_args = []
    for arg in _node_call_args(node):
        if arg in broadcast_buffers:
            # The kernel ABI declares the full iteration-space extent with
            # zero trailing strides for broadcast inputs; pass the
            # addressing-equivalent expanded stride-zero view.
            shape_text = repr(tuple(output_shape))
            stride_text = "(1," + "0," * (len(output_shape) - 1) + ")"
            stride_text = "(" + stride_text[1:-1].rstrip(",") + ")" if len(output_shape) > 1 else "(1,)"
            call_args.append(
                f"_pypto_torch.as_strided({arg}, {shape_text}, {stride_text})"
            )
        else:
            call_args.append(arg)
    joined = ", ".join(call_args)
    wrapper.writeline(
        f"pypto_launch({name!r}, ({joined}{', ' if joined else ''}), {stream})"
    )


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
    # Linear compositions that may replay elementwise over a materialized
    # broadcast; relu's (x+|x|)*0.5 is not linear in the row alone and stays
    # fail-closed on broadcast operands.
    _COMPOSED_ROW_PRIMITIVES = {
        "sigmoid": (
            ("tensor.neg", None),
            ("tensor.exp", None),
            ("tensor.adds", 1.0),
            ("tensor.recip", None),
        ),
        "tanh": (
            ("tensor.muls", -2.0),
            ("tensor.exp", None),
            ("tensor.adds", 1.0),
            ("tensor.recip", None),
            ("tensor.muls", 2.0),
            ("tensor.adds", -1.0),
        ),
    }

    def __init__(self, loop_arity: int = 0) -> None:
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.events: list[tuple[object, ...]] = []
        self.proxies: dict[int, object] = {}
        self.broadcast_keys: set[int] = set()
        self._loop_arity = loop_arity
        self._counter = 0
        # Elementwise ops distribute over broadcast, so row-domain scalar and
        # unary applications are deferred until a full-shape tensor combines
        # with the row; they replay against the materialized broadcast.
        self._pending_rows: dict[int, list[tuple[str, object | None]]] = {}

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
                proxy = self._next_proxy()
                import os as _os
                if _os.environ.get("PYPTO_DEBUG_NODES"):
                    import sys as _sys
                    print(
                        f"LOAD {buffer_name!r} index={args[1:]!r} "
                        f"kwargs={kwargs!r}",
                        file=_sys.stderr,
                        flush=True,
                    )
                if self._is_broadcast_index(args[1] if len(args) > 1 else None):
                    self.broadcast_keys.add(proxy.key)
                    self.events.append(("broadcast_load", buffer_name))
                else:
                    self.events.append(("load", buffer_name))
                return proxy
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
                operand = args[0]
                if self._is_broadcast_proxy(operand):
                    for op_name, scalar in self._COMPOSED_ROW_PRIMITIVES[name]:
                        self._pending_rows.setdefault(operand.key, []).append(
                            (op_name, scalar)
                        )
                    return operand
                return self._compose(name, operand)
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
                if self._is_broadcast_proxy(first) and self._is_constant(second):
                    # Row-domain scalar op: defer until the row meets a full
                    # tensor, then replay the scalar against the materialized
                    # broadcast (elementwise ops distribute over broadcast).
                    self._pending_rows.setdefault(first.key, []).append(
                        (self._BINARY[name] + "s", self._constant_float(second))
                    )
                    return first
                if self._is_constant(first) and self._is_broadcast_proxy(second):
                    raise StrictCoverageError(
                        "scalar-left binary op over a broadcast row is not a "
                        "registered form"
                    )
                if self._is_broadcast_proxy(first) or self._is_broadcast_proxy(second):
                    # Row-expand fused ops broadcast a [M,1,...] input over
                    # the trailing extents; the row input must be the second
                    # operand, so commute only when that preserves semantics.
                    commutative = name in ("add", "mul", "maximum", "minimum",
                                           "max", "min")
                    if self._is_broadcast_proxy(first) and not commutative:
                        # Materialize the broadcast, then apply the base op
                        # with the row on the left (e.g. row / tensor).
                        self.events.append(
                            ("op", "tensor.row_expand", [second, first])
                        )
                        expanded = self._next_proxy()
                        self.events.append(
                            ("op", self._BINARY[name], [expanded, second])
                        )
                        return self._next_proxy()
                    tensor_operand, row_operand = (
                        (second, first) if self._is_broadcast_proxy(first)
                        else (first, second)
                    )
                    if self._is_broadcast_proxy(tensor_operand):
                        raise StrictCoverageError(
                            "binary op with two broadcast operands is not a "
                            "registered form"
                        )
                    resolved_row = self._flush_pending_row(row_operand, tensor_operand)
                    if resolved_row is row_operand:
                        self.events.append(
                            ("op", "tensor.row_expand_"
                             + self._BINARY[name].split(".")[-1],
                             [tensor_operand, resolved_row])
                        )
                    else:
                        # The pending composition already materialized the
                        # broadcast; combine with the plain base op.
                        self.events.append(
                            ("op", self._BINARY[name], [tensor_operand, resolved_row])
                        )
                    return self._next_proxy()
                if self._is_constant(second):
                    self.events.append(
                        ("op", self._BINARY[name] + "s",
                         [first, self._constant_float(second)])
                    )
                else:
                    self.events.append(("op", self._BINARY[name], [first, second]))
                return self._next_proxy()
            if name in self._UNARY and len(args) == 1:
                operand = args[0]
                if self._is_broadcast_proxy(operand) and not self._is_constant(operand):
                    self._pending_rows.setdefault(operand.key, []).append(
                        (self._UNARY[name], None)
                    )
                    return operand
                self.events.append(("op", self._UNARY[name], [operand]))
                return self._next_proxy()
            raise StrictCoverageError(
                f"strict PyPTO pointwise translation has no op {name!r} yet"
            )

        return handler

    def _flush_pending_row(
        self, row_operand: object, full_operand: object
    ) -> object:
        """Materialize a pending row composition against a full-shape peer.

        The broadcast input expands via ``tensor.row_expand`` using the full
        operand as the shape definer; any deferred row-domain scalar or unary
        applications then replay elementwise over the materialized value.
        """
        pending = self._pending_rows.pop(getattr(row_operand, "key", None), [])
        if not pending:
            return row_operand
        self.events.append(("op", "tensor.row_expand", [full_operand, row_operand]))
        materialized = self._next_proxy()
        value = materialized
        for op_name, scalar in pending:
            operands = [value] if scalar is None else [value, scalar]
            self.events.append(("op", op_name, operands))
            value = self._next_proxy()
        return value

    def _is_broadcast_proxy(self, value: object) -> bool:
        return (
            isinstance(value, _OpsRecorder._Proxy)
            and value.key in self.broadcast_keys
        )

    def _is_broadcast_index(self, index: object) -> bool:
        """A broadcast row read addresses only the leading loop variable.

        Inductor hands the ops handler a flat stride-composed index: a full
        [M,N] read spans ``N*i0 + i1`` while a [M,1] row read is exactly
        ``i0``. Any load whose flat index never mentions the trailing loop
        variables broadcasts over those extents.
        """
        import sympy

        if isinstance(index, (list, tuple)):
            if len(index) != 1:
                return False
            expr = index[0]
        else:
            expr = index
        arity = self._loop_arity
        if not arity or not isinstance(expr, sympy.Basic):
            return False
        names = {symbol.name for symbol in expr.free_symbols}
        used = [name for name in names if name.startswith("i")]
        if not used:
            return False
        try:
            positions = [int(name[1:]) for name in used]
        except ValueError:
            return False
        return max(positions) < arity - 1


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
    outputs = getattr(node, "get_outputs", lambda: [])()
    output = outputs[0] if outputs else None
    output_buffer = getattr(output, "node", output)
    if hasattr(output_buffer, "get_size"):
        ranges = [int(extent) for extent in output_buffer.get_size()]
    else:
        size_source = data if data is not None else node
        ranges = [int(extent) for extent in size_source.get_size()]
    if not ranges:
        raise StrictCoverageError("pointwise kernel has no static size hints")
    body = getattr(node, "_body", None)
    if body is None:
        body = getattr(data, "get_reduction_size", None)
    body = getattr(node, "_body", None) or getattr(data, "_body", None)
    if body is None:
        raise StrictCoverageError("pointwise node has no executable body")
    var_count = len(getattr(body, "var_ranges", {}) or {})
    if var_count == 0:
        var_count = len(ranges)
    recorder = _OpsRecorder(loop_arity=var_count)
    index_vars = [[sympy.Symbol(f"i{index}") for index in range(var_count)]]
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
        tuple(ranges), dtype_name
    )
    values: dict[int, Any] = {}
    variables: dict[str, Any] = {}
    stored_keys: list[int] = []
    last_op_key: int | None = None
    next_key = 1
    for event in recorder.events:
        kind = event[0]
        import os as _os
        if _os.environ.get("PYPTO_DEBUG_NODES"):
            import sys as _sys
            print(f"EVENT {event!r}", file=_sys.stderr, flush=True)
        if kind in ("load", "broadcast_load"):
            buffer_name = str(event[1])
            variable = variables.get(buffer_name)
            if variable is None:
                if kind == "broadcast_load":
                    variable = builder.add_broadcast_input(buffer_name)
                else:
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
    broadcast_buffers = sorted(
        {
            str(event[1])
            for event in recorder.events
            if event[0] == "broadcast_load"
        }
    )
    return builder.build(), {
        "tile": 128,
        "broadcast_buffers": broadcast_buffers,
        "output_shape": list(ranges),
    }
