"""Single-graph HIR builders shared by the v2 operators.

One operator = one PyPTO TensorIR graph. A builder returns exactly one
`ir.Program`; running an operator is at most "compile once per shape,
launch once per call". Chain operands: "prev" references the immediately
preceding result, any other string names an input, floats are scalars.
"""

from __future__ import annotations

from typing import Any

from ._boot import bootstrap

SPAN_FILE = "pypto_kernels_v2"


def pointwise_graph(shape: list[int], dtype: Any,
                    ops: list[tuple[str, list[Any]]],
                    broadcast_inputs: list[str] | None = None) -> Any:
    """Build one FusedPointwiseV2 graph from a DAG op chain.

    Names listed in ``broadcast_inputs`` get the [M, 1, ...] row type so
    row-expand fused ops can consume them (the Ascend-style in-graph
    broadcast; these are exactly the graphs blocked on producer
    broadcast lowering today).
    """
    broadcast_inputs = broadcast_inputs or []

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
                    if operand in broadcast_inputs:
                        row_type = ir.TensorType(
                            [shape[0]] + [1] * (len(shape) - 1), dtype)
                        inputs[operand] = ir.Var(operand, row_type, span)
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
        "ignored_fused_pointwise", list(inputs.values()), [tensor_type],
        ir.SeqStmts(statements, span), span)
    return ir.Program([function], SPAN_FILE, span)


def row_reduction_epilogue_graph(rows: int, cols: int, eps: float,
                                 mean_scale: float) -> Any:
    """One graph: sum -> scale -> shift -> rsqrt -> broadcast-mul.

    This is the Ascend-style single-kernel RMSNorm shape: the reduction,
    the [M,1] epilogue, the broadcast back over columns and the final
    scale all live in ONE graph (`x * rsqrt(mean(x^2) + eps)`), the
    direct analog of torch_npu.npu_rms_norm. FP32 rank-2 today (the
    epilogue analyzer's dtype bound); the HIR is accepted by EmitTensorIr
    and only the pinned producer's broadcast lowering rejects it.
    """

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    span = ir.Span(SPAN_FILE, 1, 1)
    dtype = pypto.DataType.FP32
    full = ir.TensorType([rows, cols], dtype)
    row = ir.TensorType([rows, 1], dtype)
    x = ir.Var("x", full, span)
    acc = ir.Var("acc", row, span)
    t1 = ir.Var("t1", row, span)
    t2 = ir.Var("t2", row, span)
    t3 = ir.Var("t3", row, span)
    out = ir.Var("out", full, span)
    statements = [
        ir.AssignStmt(acc, ir.Call(ir.get_op("tensor.row_sum"), [x], row, span),
                      span),
        ir.AssignStmt(t1, ir.Call(
            ir.get_op("tensor.muls"),
            [acc, ir.ConstFloat(mean_scale, dtype, span)], row, span), span),
        ir.AssignStmt(t2, ir.Call(
            ir.get_op("tensor.adds"),
            [t1, ir.ConstFloat(eps, dtype, span)], row, span), span),
        ir.AssignStmt(t3, ir.Call(ir.get_op("tensor.rsqrt"), [t2], row, span),
                      span),
        ir.AssignStmt(out, ir.Call(ir.get_op("tensor.row_expand_mul"),
                                   [x, t3], full, span), span),
        ir.ReturnStmt([out], span),
    ]
    function = ir.Function(
        "ignored_row_reduction_epilogue", [x], [full],
        ir.SeqStmts(statements, span), span)
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
        "structured_matmul", [lhs, rhs], [ot],
        ir.SeqStmts([ir.AssignStmt(result, call, span),
                     ir.ReturnStmt([result], span)], span), span)
    return ir.Program([function], SPAN_FILE, span)




def rope_half_graph(rows: int, half: int) -> Any:
    """One graph for one output half of RoPE (rotate_half layout).

    out1 = x1*cos - x2*sin with cos/sin as [M,1] row-broadcast inputs —
    the single-graph analog of aclnnApplyRotaryPosEmb's per-half math.
    The odd half (x1*sin + x2*cos) is the same shape; interleaving the
    halves back is layout prep, not compute.
    """

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    dtype = pypto.DataType.BF16
    return pointwise_graph(
        [rows, half], dtype,
        [("tensor.row_expand_mul", ["x1", "cos"]),
         ("tensor.row_expand_mul", ["x2", "sin"]),
         ("tensor.sub", ["$0", "prev"])],
        broadcast_inputs=["cos", "sin"])


def softmax_scale_graph(rows: int, tokens: int) -> Any:
    """One graph for the softmax broadcast scale: p = e * (1/sum(e)).

    The row-broadcast multiply against the [M,1] inverse-sum is the
    attention softmax stage's broadcast-dependent single graph; the
    row_sum and its reciprocal are separate (compilable) graphs.
    """

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    dtype = pypto.DataType.BF16
    return pointwise_graph(
        [rows, tokens], dtype,
        [("tensor.row_expand_mul", ["e", "inv_sum"])],
        broadcast_inputs=["inv_sum"])


def gdn_delta_graph(heads: int, dv: int) -> Any:
    """One graph for the GDN delta term's broadcast: out = dot * v.

    dot [H,1] broadcasts over the value dimension — the broadcast-
    dependent single graph of the GDN read path.
    """

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    dtype = pypto.DataType.BF16
    return pointwise_graph(
        [heads, dv], dtype,
        [("tensor.row_expand_mul", ["v", "dot"])],
        broadcast_inputs=["dot"])


def gdn_compose_graph(heads: int, dk: int) -> Any:
    """One graph for the GDN operand composition: q * (softplus(g) * k).

    Pure pointwise (softplus composed as exp/+1/log) — this graph is
    EXECUTABLE today; only the broadcast consumers are blocked.
    """

    ir = bootstrap()["ir"]
    pypto = bootstrap()["pypto"]
    dtype = pypto.DataType.BF16
    return pointwise_graph(
        [heads, dk], dtype,
        [("tensor.exp", ["g"]),
         ("tensor.adds", ["prev", 1.0]),
         ("tensor.log", ["prev"]),
         ("tensor.mul", ["prev", "k"]),
         ("tensor.mul", ["prev", "q"])])


def tiles_for(*extents: int) -> list[int]:
    """One power-of-two tile per normalized (non-unit) extent, capped 32."""

    normalized = [e for e in extents if e != 1]
    return [max(1, min(32, 1 << (e.bit_length() - 1))) for e in normalized]
