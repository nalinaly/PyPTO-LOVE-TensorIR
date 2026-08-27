"""Structure tests: one operator = one graph, statuses classified."""

import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels-v2/src")

from pypto_kernels_v2._boot import bootstrap
from pypto_kernels_v2._graph import pointwise_graph, tiles_for
from pypto_kernels_v2.ops import fused_add, gdn, rmsnorm, rope, silu_and_mul


def test_each_operator_is_one_program():
    m = bootstrap()
    ir = m["ir"]
    bf16 = m["pypto"].DataType.BF16
    p = pointwise_graph([64, 128], bf16, [("tensor.add", ["a", "b"])])
    assert len(p.functions) == 1
    fn = list(p.functions.values())[0]
    assert len(fn.body.stmts) == 2  # one assignment + one return
    assert silu_and_mul.GRAPHS == 1 and fused_add.GRAPHS == 1
    assert rmsnorm.GRAPHS == 1 and rope.GRAPHS == 1


def test_rmsnorm_single_graph_is_producer_blocked_not_hir_rejected():
    result = rmsnorm.status(rows=256, cols=1024)
    assert result["status"] == "producer-blocked", result
    assert "producer" in result["error"]


def test_blocked_dependencies_are_declared():
    assert "L0" in rmsnorm.BLOCKED_ON
    assert "L0" in rope.BLOCKED_ON
    assert "L0" in gdn.BLOCKED_ON


if __name__ == "__main__":
    test_each_operator_is_one_program()
    test_rmsnorm_single_graph_is_producer_blocked_not_hir_rejected()
    test_blocked_dependencies_are_declared()
    print("ALL V2 STRUCTURE TESTS PASSED")
