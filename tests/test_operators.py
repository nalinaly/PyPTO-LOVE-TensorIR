"""Structure tests: one operator = one graph, statuses classified."""

import sys

sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-kernels/src")

import pytest
import torch

from pypto_kernels import (
    attention,
    causal_conv1d,
    embedding,
    fused_add_rmsnorm,
    gdn,
    gated_rmsnorm,
    linear,
    qk_rmsnorm_rope,
    rmsnorm,
    rope,
    sigmoid_mul,
    silu_and_mul,
)


def test_each_operator_is_one_program():
    assert attention.GRAPHS == 4  # dense + decode + cache write + prefill
    assert all(
        graphs == 1
        for graphs in (
            silu_and_mul.GRAPHS,
            fused_add_rmsnorm.GRAPHS,
            sigmoid_mul.GRAPHS,
            gated_rmsnorm.GRAPHS,
            rmsnorm.GRAPHS,
            rope.GRAPHS,
            causal_conv1d.GRAPHS,
            embedding.GRAPHS,
            linear.GRAPHS,
                qk_rmsnorm_rope.GRAPHS,
                gdn.GRAPHS,
            )
        )
    assert gdn.UPDATE_GRAPHS == 0


def test_pointwise_operators_use_native_tile_dsl():
    sample = torch.empty((3, 256), dtype=torch.bfloat16, device="meta")
    programs = {
        "sigmoid_mul": sigmoid_mul.sigmoid_mul_kernel.specialize(
            sample, sample, sample
        ),
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


def test_gated_rmsnorm_is_one_complete_native_tile_graph():
    program = gated_rmsnorm.build(2, 128)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert rendered.count("pl.tile.load") == 3
    assert "pl.tile.row_sum" in rendered
    assert "pl.tile.row_expand_mul" in rendered
    assert "pl.tile.exp" in rendered and "pl.tile.recip" in rendered
    assert rendered.count("pl.tile.store") == 1
    assert "tensor." not in rendered


def test_causal_conv1d_stateful_decode_and_prefill_share_one_graph():
    program = causal_conv1d.build(2, 1, 128)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert rendered.count("pl.InOut[") == 1
    assert "pl.range(2)" in rendered
    assert rendered.count("pl.tile.load") == 3
    assert rendered.count("pl.tile.concat") == 2
    assert "pl.tile.exp" in rendered and "pl.tile.recip" in rendered
    assert rendered.count("pl.tile.store") == 2
    prefill = str(causal_conv1d.build(1, 5, 128))
    assert "pl.range(5)" in prefill


def test_embedding_is_one_dynamic_row_gather_graph():
    program = embedding.build(8, 1024, 256)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert "pl.range(8)" in rendered
    assert "pl.range(2)" in rendered
    assert "pl.tensor.read" in rendered
    assert "pl.tile.load" in rendered
    assert "pl.tile.store" in rendered


def test_qk_norm_partial_rope_gate_is_one_native_graph():
    program = qk_rmsnorm_rope.build(2, 8, 2, 256, 64, 1024)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert "pl.range(8)" in rendered and "pl.range(2)" in rendered
    assert rendered.count("pl.tile.row_sum") == 2
    assert "pl.tensor.read" in rendered
    assert "pl.tile.slice" in rendered
    assert "pl.tile.concat" in rendered
    assert rendered.count("pl.tile.store") == 3


def _one_program(p):
    assert len(p.functions) == 1
    fn = list(p.functions.values())[0]
    return fn


def test_rope_is_one_native_tile_graph():
    program = rope.build(2, 64)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert rendered.count("pl.tile.load") == 4
    assert rendered.count("pl.tile.store") == 2
    assert "pl.tile.sub" in rendered and "pl.tile.add" in rendered
    assert "tensor." not in rendered


def test_attention_is_one_native_tile_graph():
    program = attention.build(2, 128, 128, 128)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert rendered.count("pl.tile.matmul") == 2
    assert "pl.tile.row_max" in rendered
    assert "pl.tile.row_sum" in rendered
    assert "pl.tile.exp" in rendered
    assert "pl.tile.store" in rendered
    assert "tensor." not in rendered


def test_paged_attention_decode_gathers_physical_kv_rows_in_one_graph():
    program = attention.build_paged_decode(1, 8, 2, 16, 256, 1024, 65, 4096)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert "pl.range(8)" in rendered and "pl.range(16)" in rendered
    assert rendered.count("pl.tensor.read") == 4
    assert "physical_i64" in rendered and "virtual_to_physical" in rendered
    assert "pl.cast" in rendered
    assert rendered.count("pl.tile.gather_row") == 2
    assert "transpose=True" in rendered
    assert rendered.count("pl.tile.matmul") == 2
    assert "pl.tile.ci" in rendered
    assert "pl.tile.cmps" in rendered
    assert "pl.tile.sels" in rendered
    assert "pl.tile.row_max" in rendered and "pl.tile.row_sum" in rendered
    assert rendered.count("pl.tile.store") == 1

    wide_program = attention.build_paged_decode(
        1, 8, 2, 512, 256, 1024, 65, 4096
    )
    wide_rendered = str(wide_program)
    assert "pl.range(512)" in wide_rendered
    assert "pl.tile.ci" in wide_rendered
    assert "pl.tile.cmps" in wide_rendered
    assert "pl.tile.sels" in wide_rendered

    large_model_program = attention.build_paged_decode(
        1, 16, 4, 16, 256, 1024, 65, 4096
    )
    large_model_rendered = str(large_model_program)
    assert "pl.range(16)" in large_model_rendered
    assert large_model_rendered.count("pl.tile.gather_row") == 2
    assert large_model_rendered.count("pl.tile.matmul") == 2

    batched_program = attention.build_paged_decode(
        2, 8, 2, 16, 256, 1024, 65, 4096
    )
    batched_rendered = str(batched_program)
    assert "pl.range(2)" in batched_rendered
    assert batched_rendered.count("pl.tile.gather_row") == 2


def test_paged_cache_layout_accepts_static_row_pitch() -> None:
    key_cache = torch.empty_strided(
        (1024, 512), (2048, 1), dtype=torch.bfloat16, device="meta"
    )
    value_cache = torch.empty_strided(
        (1024, 512), (2048, 1), dtype=torch.bfloat16, device="meta"
    )
    assert (
        attention._paged_cache_row_stride(
            key_cache, value_cache, operation="test"
        )
        == 2048
    )
    overlap = torch.empty_strided(
        (1024, 512), (256, 1), dtype=torch.bfloat16, device="meta"
    )
    with pytest.raises(ValueError, match="non-overlapping static row pitch"):
        attention._paged_cache_row_stride(overlap, overlap, operation="test")


def test_paged_cache_write_declares_mutation_and_one_graph():
    program = attention.build_paged_cache_write(1024, 1, 512)
    rendered = str(program)
    fn = _one_program(program)
    assert len(fn.body.stmts) == 2
    assert rendered.count("pl.InOut[") == 2
    assert rendered.count("pl.tensor.read") == 2
    assert "virtual_row_i64" in rendered and "virtual_to_physical" in rendered
    assert "pl.cast" in rendered
    assert rendered.count("pl.tile.load") == 2
    assert rendered.count("pl.tile.store") == 3
    assert "pl.tile.add" in rendered

    prefill_program = attention.build_paged_cache_write(1024, 13, 512)
    prefill_rendered = str(prefill_program)
    assert "pl.range(13)" in prefill_rendered
    assert prefill_rendered.count("pl.tile.store") == 3


def test_paged_prefill_is_one_causal_gqa_graph():
    program = attention.build_paged_prefill(13, 8, 2, 16, 256, 1024, 65, 4096)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert "kv_heads: pl.Scalar[pl.INDEX] = 2" in rendered
    assert "queries_per_kv: pl.Scalar[pl.INDEX] = 8 // kv_heads" in rendered
    assert "pl.range(kv_heads)" in rendered
    assert "pl.range(queries_per_kv)" in rendered
    assert "pl.range(13)" in rendered
    assert "pl.range(16)" in rendered
    assert rendered.count("pl.tensor.read") == 4
    assert "physical_i64" in rendered and "virtual_to_physical" in rendered
    assert rendered.count("pl.tile.gather_row") == 2
    assert rendered.count("pl.tile.matmul") == 2
    assert "pl.tile.cmps" in rendered and "pl.tile.sels" in rendered
    assert rendered.count("pl.tile.store") == 1

    large_model = attention.build_paged_prefill(
        13, 16, 4, 16, 256, 1024, 65, 4096
    )
    large_rendered = str(large_model)
    assert "kv_heads: pl.Scalar[pl.INDEX] = 4" in large_rendered
    assert "queries_per_kv: pl.Scalar[pl.INDEX] = 16 // kv_heads" in large_rendered
    assert large_rendered.count("pl.tile.gather_row") == 2


def test_linear_is_one_native_tile_graph():
    program = linear.build(2, 128, 256)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert "pl.range(2)" in rendered
    assert rendered.count("pl.tile.load") == 2
    assert "pl.tile.transpose_view" in rendered
    assert "pl.tile.matmul" in rendered
    assert "pl.tile.store" in rendered
    assert "tensor." not in rendered


def test_gdn_recurrent_is_one_mutation_declared_graph():
    program = gdn.build_recurrent(
        2, 1, 8, 16, 128, 128, 65, 16 * 128 * 128 + 4096
    )
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert rendered.count("pl.InOut[") == 1
    assert "pl.range(2)" in rendered and "pl.range(16)" in rendered
    assert rendered.count("pl.tile.row_sum") == 2  # Q and K L2 norms
    assert rendered.count("pl.tile.matmul") == 2  # state*K and Q*state
    assert "pl.tile.abs" in rendered and "pl.tile.maximums" in rendered
    assert "pl.tile.row_expand_mul" in rendered
    assert "pl.tile.row_expand" in rendered
    assert "pl.tile.col_expand_mul" in rendered
    assert rendered.count("pl.tile.store") == 2  # output plus final FP32 state
    prefill = str(gdn.build_recurrent(1, 3, 8, 16, 128, 128, 65))
    assert "pl.range(3)" in prefill


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
    test_gated_rmsnorm_is_one_complete_native_tile_graph()
    test_causal_conv1d_stateful_decode_and_prefill_share_one_graph()
    test_embedding_is_one_dynamic_row_gather_graph()
    test_qk_norm_partial_rope_gate_is_one_native_graph()
    test_rope_is_one_native_tile_graph()
    test_attention_is_one_native_tile_graph()
    test_paged_attention_decode_gathers_physical_kv_rows_in_one_graph()
    test_paged_cache_write_declares_mutation_and_one_graph()
    test_paged_prefill_is_one_causal_gqa_graph()
    test_linear_is_one_native_tile_graph()
    test_gdn_recurrent_is_one_mutation_declared_graph()
    test_broadcast_dependencies_are_closed()
    print("ALL OPERATOR STRUCTURE TESTS PASSED")
