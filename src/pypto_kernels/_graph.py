"""Temporary whole-tensor baselines pending native tile migration.

One operator = one PyPTO TensorIR graph. A builder returns exactly one
`ir.Program`; running an operator is at most "compile once per shape,
launch once per call". Chain operands: "prev" references the immediately
preceding result, any other string names an input, floats are scalars.
"""

from __future__ import annotations

from typing import Any

from ._boot import bootstrap

SPAN_FILE = "pypto_kernels"


def pointwise_graph(
    shape: list[int],
    dtype: Any,
    ops: list[tuple[str, list[Any]]],
    broadcast_inputs: list[str] | None = None,
    broadcast_shapes: dict[str, list[int]] | None = None,
) -> Any:
    """Build one FusedPointwiseV2 graph from a DAG op chain.

    Names listed in ``broadcast_inputs`` get the [M, 1, ...] row type so
    row-expand fused ops can consume them (the Ascend-style in-graph
    broadcast; these are exactly the graphs blocked on producer
    broadcast lowering today).
    """
    broadcast_inputs = broadcast_inputs or []
    resolved_broadcast_shapes = dict(broadcast_shapes or {})
    for name in broadcast_inputs:
        resolved_broadcast_shapes.setdefault(name, [shape[0]] + [1] * (len(shape) - 1))
    for name, input_shape in resolved_broadcast_shapes.items():
        if (
            len(input_shape) != len(shape)
            or not any(
                source == 1 and target > 1 for source, target in zip(input_shape, shape)
            )
            or any(
                source != target and source != 1
                for source, target in zip(input_shape, shape)
            )
        ):
            raise ValueError(
                f"broadcast input {name} shape {input_shape} is incompatible "
                f"with {shape}"
            )

    ir = bootstrap()["ir"]
    span = ir.Span(SPAN_FILE, 1, 1)
    tensor_type = ir.TensorType(shape, dtype)
    inputs: dict[str, Any] = {}
    statements = []
    results: list[Any] = []
    previous: Any = None
    for op_name, operands in ops:
        args = []
        for operand in operands:
            if operand == "prev":
                args.append(previous)
            elif isinstance(operand, str) and operand.startswith("$"):
                args.append(results[int(operand[1:])])
            elif isinstance(operand, str):
                if operand not in inputs:
                    if operand in resolved_broadcast_shapes:
                        broadcast_type = ir.TensorType(
                            resolved_broadcast_shapes[operand], dtype
                        )
                        inputs[operand] = ir.Var(operand, broadcast_type, span)
                    else:
                        inputs[operand] = ir.Var(operand, tensor_type, span)
                args.append(inputs[operand])
            else:
                args.append(ir.ConstFloat(float(operand), dtype, span))
        result = ir.Var("ignored", tensor_type, span)
        call = ir.Call(ir.get_op(op_name), args, tensor_type, span)
        statements.append(ir.AssignStmt(result, call, span))
        results.append(result)
        previous = result
    statements.append(ir.ReturnStmt([previous], span))
    function = ir.Function(
        "ignored_fused_pointwise",
        list(inputs.values()),
        [tensor_type],
        ir.SeqStmts(statements, span),
        span,
    )
    return ir.Program([function], SPAN_FILE, span)


def row_reduction_epilogue_graph(
    rows: int, cols: int, eps: float, mean_scale: float
) -> Any:
    """One graph: sum -> scale -> shift -> rsqrt -> broadcast-mul.

    This is the Ascend-style single-kernel RMSNorm shape: the reduction,
    the [M,1] epilogue, the broadcast back over columns and the final
    scale all live in ONE graph (`x * rsqrt(mean(x^2) + eps)`), the
    direct analog of torch_npu.npu_rms_norm. BF16 input/output are widened to
    FP32 inside the graph for square, reduction and epilogue arithmetic.
    """

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    span = ir.Span(SPAN_FILE, 1, 1)
    dtype = pypto.DataType.BF16
    full = ir.TensorType([rows, cols], dtype)
    row = ir.TensorType([rows, 1], dtype)
    x = ir.Var("x", full, span)
    square = ir.Var("square", full, span)
    acc = ir.Var("acc", row, span)
    t1 = ir.Var("t1", row, span)
    t2 = ir.Var("t2", row, span)
    t3 = ir.Var("t3", row, span)
    out = ir.Var("out", full, span)
    statements = [
        ir.AssignStmt(
            square, ir.Call(ir.get_op("tensor.mul"), [x, x], full, span), span
        ),
        ir.AssignStmt(
            acc, ir.Call(ir.get_op("tensor.row_sum"), [square], row, span), span
        ),
        ir.AssignStmt(
            t1,
            ir.Call(
                ir.get_op("tensor.muls"),
                [acc, ir.ConstFloat(mean_scale, dtype, span)],
                row,
                span,
            ),
            span,
        ),
        ir.AssignStmt(
            t2,
            ir.Call(
                ir.get_op("tensor.adds"),
                [t1, ir.ConstFloat(eps, dtype, span)],
                row,
                span,
            ),
            span,
        ),
        ir.AssignStmt(t3, ir.Call(ir.get_op("tensor.rsqrt"), [t2], row, span), span),
        ir.AssignStmt(
            out, ir.Call(ir.get_op("tensor.row_expand_mul"), [x, t3], full, span), span
        ),
        ir.ReturnStmt([out], span),
    ]
    function = ir.Function(
        "ignored_row_reduction_epilogue",
        [x],
        [full],
        ir.SeqStmts(statements, span),
        span,
    )
    return ir.Program([function], SPAN_FILE, span)


def matmul_graph(lhs_shape: list[int], rhs_shape: list[int]) -> Any:
    """One StructuredMatmulV4 graph (dense BF16 rank-2/3)."""

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    span = ir.Span(SPAN_FILE, 1, 1)
    dtype = pypto.DataType.BF16
    result_shape = [*lhs_shape[:-1], rhs_shape[-1]]
    lt = ir.TensorType(lhs_shape, dtype)
    rt = ir.TensorType(rhs_shape, dtype)
    ot = ir.TensorType(result_shape, dtype)
    lhs = ir.Var("lhs", lt, span)
    rhs = ir.Var("rhs", rt, span)
    result = ir.Var("result", ot, span)
    call = ir.Call(ir.get_op("tensor.matmul"), [lhs, rhs], ot, span)
    function = ir.Function(
        "structured_matmul",
        [lhs, rhs],
        [ot],
        ir.SeqStmts(
            [ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span
        ),
        span,
    )
    return ir.Program([function], SPAN_FILE, span)


def rope_half_graph(rows: int, half: int) -> Any:
    """One graph for one output half of RoPE (rotate_half layout).

    out1 = x1*cos - x2*sin with cos/sin as [M,1] row-broadcast inputs —
    the single-graph analog of aclnnApplyRotaryPosEmb's per-half math.
    The odd half (x1*sin + x2*cos) is the same shape; interleaving the
    halves back is layout prep, not compute.
    """

    pypto = bootstrap()["pypto"]
    dtype = pypto.DataType.BF16
    return pointwise_graph(
        [rows, half],
        dtype,
        [
            ("tensor.row_expand_mul", ["x1", "cos"]),
            ("tensor.row_expand_mul", ["x2", "sin"]),
            ("tensor.sub", ["$0", "prev"]),
        ],
        broadcast_inputs=["cos", "sin"],
    )


def softmax_scale_graph(rows: int, tokens: int) -> Any:
    """One graph for the softmax broadcast scale: p = e * (1/sum(e)).

    The row-broadcast multiply against the [M,1] inverse-sum is the
    attention softmax stage's broadcast-dependent single graph; the
    row_sum and its reciprocal are separate (compilable) graphs.
    """

    pypto = bootstrap()["pypto"]
    dtype = pypto.DataType.BF16
    return pointwise_graph(
        [rows, tokens],
        dtype,
        [("tensor.row_expand_mul", ["e", "inv_sum"])],
        broadcast_inputs=["inv_sum"],
    )


def row_normalize_graph(rows: int, columns: int) -> Any:
    """One BF16 graph: x / sum(x) over the trailing dimension."""

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    span = ir.Span(SPAN_FILE, 1, 1)
    dtype = pypto.DataType.BF16
    full = ir.TensorType([rows, columns], dtype)
    row = ir.TensorType([rows, 1], dtype)
    x = ir.Var("x", full, span)
    total = ir.Var("total", row, span)
    inverse = ir.Var("inverse", row, span)
    out = ir.Var("out", full, span)
    function = ir.Function(
        "ignored_row_normalize",
        [x],
        [full],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    total, ir.Call(ir.get_op("tensor.row_sum"), [x], row, span), span
                ),
                ir.AssignStmt(
                    inverse,
                    ir.Call(ir.get_op("tensor.recip"), [total], row, span),
                    span,
                ),
                ir.AssignStmt(
                    out,
                    ir.Call(
                        ir.get_op("tensor.row_expand_mul"), [x, inverse], full, span
                    ),
                    span,
                ),
                ir.ReturnStmt([out], span),
            ],
            span,
        ),
        span,
    )
    return ir.Program([function], SPAN_FILE, span)


def row_sum_graph(rows: int, columns: int) -> Any:
    """One BF16 RowReductionV3 graph returning [rows, 1]."""

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    span = ir.Span(SPAN_FILE, 1, 1)
    dtype = pypto.DataType.BF16
    full = ir.TensorType([rows, columns], dtype)
    row = ir.TensorType([rows, 1], dtype)
    x = ir.Var("x", full, span)
    total = ir.Var("total", row, span)
    function = ir.Function(
        "ignored_row_sum",
        [x],
        [row],
        ir.SeqStmts(
            [
                ir.AssignStmt(
                    total, ir.Call(ir.get_op("tensor.row_sum"), [x], row, span), span
                ),
                ir.ReturnStmt([total], span),
            ],
            span,
        ),
        span,
    )
    return ir.Program([function], SPAN_FILE, span)


def gdn_delta_graph(heads: int, dv: int) -> Any:
    """One graph for the GDN delta term's broadcast: out = dot * v.

    dot [H,1] broadcasts over the value dimension — the broadcast-
    dependent single graph of the GDN read path.
    """

    pypto = bootstrap()["pypto"]
    dtype = pypto.DataType.BF16
    return pointwise_graph(
        [heads, dv],
        dtype,
        [("tensor.row_expand_mul", ["v", "dot"])],
        broadcast_inputs=["dot"],
    )


def gdn_delta_combine_graph(heads: int, dv: int) -> Any:
    """One graph: read + dot*v with dot broadcast over Dv."""

    pypto = bootstrap()["pypto"]
    return pointwise_graph(
        [heads, dv],
        pypto.DataType.BF16,
        [("tensor.row_expand_mul", ["value", "dot"]), ("tensor.add", ["prev", "read"])],
        broadcast_inputs=["dot"],
    )


def gdn_q_decay_graph(heads: int, dk: int) -> Any:
    """One pointwise graph: q_decay = q * decay."""

    pypto = bootstrap()["pypto"]
    return pointwise_graph(
        [heads, dk], pypto.DataType.BF16, [("tensor.mul", ["q", "decay"])]
    )


def gdn_state_read_graph(heads: int, dk: int, dv: int) -> Any:
    """One batched matmul graph: [H,1,Dk] @ [H,Dk,Dv]."""

    return matmul_graph([heads, 1, dk], [heads, dk, dv])


def gdn_state_update_graph(heads: int, dk: int, dv: int) -> Any:
    """One rank-3 graph for decay*state + beta_key outer value."""

    pypto = bootstrap()["pypto"]
    return pointwise_graph(
        [heads, dk, dv],
        pypto.DataType.BF16,
        [
            ("tensor.row_expand_mul", ["state", "decay"]),
            ("tensor.row_expand", ["state", "beta_key"]),
            ("tensor.row_expand", ["state", "value"]),
            ("tensor.mul", ["$1", "$2"]),
            ("tensor.add", ["$0", "$3"]),
        ],
        broadcast_shapes={
            "decay": [heads, dk, 1],
            "beta_key": [heads, dk, 1],
            "value": [heads, 1, dv],
        },
    )


def gdn_compose_graph(heads: int, dk: int) -> Any:
    """One graph for the GDN operand composition: q * (softplus(g) * k).

    Pure pointwise (softplus composed as exp/+1/log) — this graph is
    EXECUTABLE today; only the broadcast consumers are blocked.
    """

    pypto = bootstrap()["pypto"]
    dtype = pypto.DataType.BF16
    return pointwise_graph(
        [heads, dk],
        dtype,
        [
            ("tensor.exp", ["g"]),
            ("tensor.adds", ["prev", 1.0]),
            ("tensor.log", ["prev"]),
            ("tensor.mul", ["prev", "k"]),
            ("tensor.mul", ["prev", "q"]),
        ],
    )


def tiles_for(*extents: int) -> list[int]:
    """One power-of-two tile per normalized (non-unit) extent, capped 32."""

    normalized = [e for e in extents if e != 1]
    return [max(1, min(32, 1 << (e.bit_length() - 1))) for e in normalized]
