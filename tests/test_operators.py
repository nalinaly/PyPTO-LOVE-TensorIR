"""Structure tests: one operator = one graph, statuses classified."""

import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")

import torch

from pypto_kernels import (
    attention,
    fused_add,
    fused_add_rmsnorm,
    gdn,
    linear,
    rmsnorm,
    rope,
    silu_and_mul,
)


def test_each_operator_is_one_program():
    assert all(
        graphs == 1
        for graphs in (
            silu_and_mul.GRAPHS,
            fused_add.GRAPHS,
            fused_add_rmsnorm.GRAPHS,
            rmsnorm.GRAPHS,
            rope.GRAPHS,
            attention.GRAPHS,
            linear.GRAPHS,
            gdn.GRAPHS,
            gdn.UPDATE_GRAPHS,
        )
    )


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


def test_rmsnorm_uses_native_tile_reduction():
    program = rmsnorm.build(2, 256)
    rendered = str(program)
    assert "pl.range(2)" in rendered
    assert rendered.count("pl.tile.load") == 2
    assert "pl.tile.row_sum" in rendered
    assert "pl.tile.row_expand_mul" in rendered
    assert "pl.tile.add" in rendered
    assert "pl.tile.mul" in rendered
    assert "pl.tile.store" in rendered
    assert "tensor." not in rendered


def test_fused_add_rmsnorm_is_one_native_tile_graph_with_two_outputs():
    program = fused_add_rmsnorm.build(2, 256)
    rendered = str(program)
    function = _one_program(program)
    assert len(function.body.stmts) == 2  # one scope + return tuple
    assert len(function.return_types) == 2
    assert rendered.count("pl.tile.load") == 3
    assert "pl.tile.row_sum" in rendered
    assert "pl.tile.row_expand_mul" in rendered
    assert rendered.count("pl.tile.store") == 2
    assert "tensor." not in rendered
    result = fused_add_rmsnorm.status(rows=2, columns=256)
    assert result["status"] == "compiled", result


def _one_program(p):
    assert len(p.functions) == 1
    fn = list(p.functions.values())[0]
    return fn


def test_rope_is_one_native_tile_graph_and_compiled():
    program = rope.build(2, 64)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert rendered.count("pl.tile.load") == 4
    assert rendered.count("pl.tile.store") == 2
    assert "pl.tile.sub" in rendered and "pl.tile.add" in rendered
    assert "tensor." not in rendered
    result = rope.status()
    assert result["status"] == "compiled", result


def test_attention_is_one_native_tile_graph_and_compiled():
    program = attention.build(2, 128, 128, 128)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert rendered.count("pl.tile.matmul") == 2
    assert "pl.tile.row_max" in rendered
    assert "pl.tile.row_sum" in rendered
    assert "pl.tile.exp" in rendered
    assert "pl.tile.store" in rendered
    assert "tensor." not in rendered
    result = attention.status(rows=2)
    assert result["status"] == "compiled", result


def test_linear_is_one_native_tile_graph_and_compiled():
    program = linear.build(2, 128, 256)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert "pl.range(2)" in rendered
    assert rendered.count("pl.tile.load") == 2
    assert "pl.tile.transpose_view" in rendered
    assert "pl.tile.matmul" in rendered
    assert "pl.tile.store" in rendered
    assert "tensor." not in rendered
    result = linear.status(rows=2, in_features=128, out_features=256)
    assert result["status"] == "compiled", result


def test_gdn_read_and_update_are_one_native_tile_graph_each():
    program = gdn.build_read(2, 128, 128)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert "pl.tile.matmul" in rendered
    assert "pl.tile.row_sum" in rendered
    assert "pl.tile.row_expand_mul" in rendered
    assert "pl.tile.store" in rendered
    assert "tensor." not in rendered
    update_program = gdn.build_state_update(2, 128, 128)
    update_rendered = str(update_program)
    assert len(_one_program(update_program).body.stmts) == 2  # one scope + return
    assert update_rendered.count("pl.tile.load") == 4
    assert "pl.tile.row_expand_mul" in update_rendered
    assert "pl.tile.row_expand" in update_rendered
    assert "pl.tile.col_expand_mul" in update_rendered
    assert "pl.tile.store" in update_rendered
    assert "tensor." not in update_rendered
    results = {"read": gdn.read_status(), "state_update": gdn.state_update_status()}
    assert all(result["status"] == "compiled" for result in results.values()), results


def test_rmsnorm_single_graph_is_compiled():
    result = rmsnorm.status(rows=256, cols=1024)
    assert result["status"] == "compiled", result


def test_broadcast_dependencies_are_closed():
    assert rmsnorm.STATUS.endswith("executable")
    assert rope.STATUS.endswith("executable")
    assert not hasattr(rmsnorm, "BLOCKED_ON")
    assert not hasattr(rope, "BLOCKED_ON")
    assert not hasattr(gdn, "BLOCKED_ON")


if __name__ == "__main__":
    test_each_operator_is_one_program()
    test_pointwise_operators_use_native_tile_dsl()
    test_rmsnorm_uses_native_tile_reduction()
    test_fused_add_rmsnorm_is_one_native_tile_graph_with_two_outputs()
    test_rope_is_one_native_tile_graph_and_compiled()
    test_attention_is_one_native_tile_graph_and_compiled()
    test_linear_is_one_native_tile_graph_and_compiled()
    test_gdn_read_and_update_are_one_native_tile_graph_each()
    test_rmsnorm_single_graph_is_compiled()
    test_broadcast_dependencies_are_closed()
    print("ALL OPERATOR STRUCTURE TESTS PASSED")
