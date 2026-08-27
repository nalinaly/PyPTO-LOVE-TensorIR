"""Structure tests: one operator = one graph, statuses classified."""

import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")

import torch

from pypto_kernels._boot import bootstrap
from pypto_kernels._graph import gdn_state_update_graph, pointwise_graph
from pypto_kernels import attention, fused_add, gdn, rmsnorm, rope, silu_and_mul


def test_each_operator_is_one_program():
    m = bootstrap()
    bf16 = m["pypto"].DataType.BF16
    p = pointwise_graph([64, 128], bf16, [("tensor.add", ["a", "b"])])
    assert len(p.functions) == 1
    fn = list(p.functions.values())[0]
    assert len(fn.body.stmts) == 2  # one assignment + one return
    assert silu_and_mul.GRAPHS == 1 and fused_add.GRAPHS == 1
    assert rmsnorm.GRAPHS == 1 and rope.GRAPHS == 2  # even/odd halves


def test_pointwise_operators_use_native_tile_dsl():
    sample = torch.empty((3, 256), dtype=torch.bfloat16, device="meta")
    programs = {
        "fused_add": fused_add.fused_add_kernel.specialize(sample, sample, sample),
        "silu_and_mul": silu_and_mul.silu_and_mul_kernel.specialize(
            sample, sample, sample
        ),
    }
    for name, program in programs.items():
        rendered = str(program)
        assert "pl.at(level=pl.Level.CORE_GROUP)" in rendered, name
        assert "pl.range(3)" in rendered, name
        assert "pl.range(2)" in rendered, name
        assert "pl.tile.load" in rendered, name
        assert "pl.tile.store" in rendered, name
        assert "tensor." not in rendered, name


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
    program = attention.build_softmax_scale(256, 1024)
    assert len(_one_program(program).body.stmts) == 2  # 1 assignment + return
    result = attention.softmax_status()
    assert result["status"] == "compiled", result


def test_attention_normalize_and_value_mix_compile():
    normalize = attention.softmax_normalize_status()
    value_mix = attention.value_mix_status()
    assert normalize["status"] == "compiled", normalize
    assert value_mix["status"] == "compiled", value_mix


def test_gdn_complete_read_components_and_update_compile():
    results = {
        "q_decay": gdn.q_decay_status(),
        "compose": gdn.compose_status(),
        "dot": gdn.dot_status(),
        "state_read": gdn.state_read_status(),
        "delta_combine": gdn.delta_combine_status(),
        "state_update": gdn.state_update_status(),
    }
    assert all(result["status"] == "compiled" for result in results.values()), results
    update = _one_program(gdn_state_update_graph(16, 128, 128))
    assert len(update.body.stmts) == 6  # 5 fused assignments + return


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
    test_pointwise_operators_use_native_tile_dsl()
    test_rope_halves_are_single_graphs_and_compiled()
    test_gdn_compose_and_delta_compile()
    test_attention_softmax_scale_is_single_graph_compiled()
    test_attention_normalize_and_value_mix_compile()
    test_gdn_complete_read_components_and_update_compile()
    test_rmsnorm_single_graph_is_compiled()
    test_broadcast_dependencies_are_closed()
    print("ALL OPERATOR STRUCTURE TESTS PASSED")
