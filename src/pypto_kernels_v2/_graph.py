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
                    ops: list[tuple[str, list[Any]]]) -> Any:
    """Build one FusedPointwiseV2 graph from a DAG op chain."""

    ir = bootstrap()["ir"]
    span = ir.Span(SPAN_FILE, 1, 1)
    tensor_type = ir.TensorType(shape, dtype)
    inputs: dict[str, Any] = {}
    statements = []
    previous: Any = None
    for op_name, operands in ops:
        args = []
        for operand in operands:
            if operand == "prev":
                args.append(previous)
            elif isinstance(operand, str):
                if operand not in inputs:
                    inputs[operand] = ir.Var(operand, tensor_type, span)
                args.append(inputs[operand])
            else:
                args.append(ir.ConstFloat(float(operand), dtype, span))
        result = ir.Var("ignored", tensor_type, span)
        call = ir.Call(ir.get_op(op_name), args, tensor_type, span)
        statements.append(ir.AssignStmt(result, call, span))
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


def tiles_for(*extents: int) -> list[int]:
    """One power-of-two tile per normalized (non-unit) extent, capped 32."""

    normalized = [e for e in extents if e != 1]
    return [max(1, min(32, 1 << (e.bit_length() - 1))) for e in normalized]
