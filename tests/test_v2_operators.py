"""Structure tests: one operator = one graph, statuses classified."""

import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels-v2/src")

from pypto_kernels_v2._boot import bootstrap
from pypto_kernels_v2._graph import pointwise_graph, tiles_for
from pypto_kernels_v2.ops import (attention_design, fused_add, gdn,
                                 rmsnorm, rope, silu_and_mul)


def test_each_operator_is_one_program():
    m = bootstrap()
    ir = m["ir"]
    bf16 = m["pypto"].DataType.BF16
    p = pointwise_graph([64, 128], bf16, [("tensor.add", ["a", "b"])])
    assert len(p.functions) == 1
    fn = list(p.functions.values())[0]
    assert len(fn.body.stmts) == 2  # one assignment + one return
    assert silu_and_mul.GRAPHS == 1 and fused_add.GRAPHS == 1
    assert rmsnorm.GRAPHS == 1 and rope.GRAPHS == 2  # even/odd halves


def _one_program(p):
    assert len(p.functions) == 1
    fn = list(p.functions.values())[0]
    return fn


def test_rope_halves_are_single_graphs_and_compiled():
    even = rope.build_even(256, 512)
    odd = rope.build_odd(256, 512)
    assert len(_one_program(even).body.stmts) == 4  # 3 assignments + return
    assert len(_one_program(odd).body.stmts) == 4
    even_result = rope.status()
    odd_result = rope.odd_status()
    assert even_result["status"] == "compiled", even_result
    assert odd_result["status"] == "compiled", odd_result


def test_gdn_compose_and_delta_compile():
    compose = gdn.compose_status()
    assert compose["status"] == "compiled", compose
    delta = gdn.delta_status()
    assert delta["status"] == "compiled", delta


def test_attention_softmax_scale_is_single_graph_compiled():
    program = attention_design.build_softmax_scale(256, 1024)
    assert len(_one_program(program).body.stmts) == 2  # 1 assignment + return
    result = attention_design.softmax_status()
    assert result["status"] == "compiled", result


def test_rmsnorm_single_graph_is_compiled():
    result = rmsnorm.status(rows=256, cols=1024)
    assert result["status"] == "compiled", result


def test_broadcast_dependencies_are_closed():
    assert rmsnorm.STATUS == "executable"
    assert rope.STATUS == "executable"
    assert not hasattr(rmsnorm, "BLOCKED_ON")
    assert not hasattr(rope, "BLOCKED_ON")
    assert not hasattr(gdn, "BLOCKED_ON")


if __name__ == "__main__":
    test_each_operator_is_one_program()
    test_rope_halves_are_single_graphs_and_compiled()
    test_gdn_compose_and_delta_compile()
    test_attention_softmax_scale_is_single_graph_compiled()
    test_rmsnorm_single_graph_is_compiled()
    test_broadcast_dependencies_are_closed()
    print("ALL V2 STRUCTURE TESTS PASSED")
