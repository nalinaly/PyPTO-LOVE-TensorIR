"""Structure tests: one operator = one graph, statuses classified."""

from pathlib import Path
from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from pypto_kernels import (
    _boot,
    attention,
    causal_conv1d,
    embedding,
    fused_add_rmsnorm,
    gdn,
    gdn_projection,
    gated_rmsnorm,
    linear,
    qk_rmsnorm_rope,
    rmsnorm,
    rope,
    sigmoid_mul,
    silu_and_mul,
)


def test_runtime_reuses_executables_and_acquires_new_graph_leases(monkeypatch):
    created = []

    class FakeExecutable:
        def __init__(self, artifact, request):
            self.inputs = (artifact, request)
            self.prewarmed = []
            self.leases = 0
            created.append(self)

        def prewarm(self, version):
            self.prewarmed.append(version)

        def acquire_cuda_graph_lease(self):
            self.leases += 1
            return (self.inputs, self.leases)

    nvidia = ModuleType("pypto.runtime.nvidia")
    nvidia.NvidiaExecutable = FakeExecutable
    nvidia.observe_current_nvidia_runtime = lambda *_args: SimpleNamespace(
        cuda_runtime_api_version=13030
    )
    runtime = ModuleType("pypto.runtime")
    runtime.nvidia = nvidia
    pypto = ModuleType("pypto")
    pypto.__path__ = []
    pypto.runtime = runtime
    monkeypatch.setitem(sys.modules, "pypto", pypto)
    monkeypatch.setitem(sys.modules, "pypto.runtime", runtime)
    monkeypatch.setitem(sys.modules, "pypto.runtime.nvidia", nvidia)

    key = "unit-runtime-cache"
    monkeypatch.setitem(_boot._GRAPHS, key, ("artifact", "request"))
    _boot._EXECUTABLES.pop(key, None)
    first = _boot._ready_executable(key)
    second = _boot._ready_executable(key)
    assert first is second
    assert created == [first]
    assert first.prewarmed == [13030]
    leases = _boot.acquire_cuda_graph_leases()
    assert leases[key] == (("artifact", "request"), 1)
    assert key not in _boot.acquire_cuda_graph_leases({key})
    _boot._EXECUTABLES.pop(key, None)


def test_launch_graph_wraps_executable_in_artifact_annotation(monkeypatch):
    class FakeExecutable:
        def prepare_launch(self, arguments):
            assert arguments == []
            return "packet"

        def launch(self, packet, stream):
            launched.append((packet, stream))

    nvidia = ModuleType("pypto.runtime.nvidia")
    nvidia.NvidiaLaunchArgument = SimpleNamespace(tensor=lambda *_args: None)
    runtime = ModuleType("pypto.runtime")
    runtime.nvidia = nvidia
    pypto = ModuleType("pypto")
    pypto.__path__ = []
    pypto.runtime = runtime
    monkeypatch.setitem(sys.modules, "pypto", pypto)
    monkeypatch.setitem(sys.modules, "pypto.runtime", runtime)
    monkeypatch.setitem(sys.modules, "pypto.runtime.nvidia", nvidia)

    key = "unit-annotated-launch"
    artifact = SimpleNamespace(
        kernel_abi=SimpleNamespace(
            argument_layout=SimpleNamespace(operand_descriptors=[])
        )
    )
    record = object()
    monkeypatch.setitem(_boot._GRAPHS, key, (artifact, "request"))
    monkeypatch.setitem(_boot._GRAPH_RECORDS, key, record)
    monkeypatch.setattr(_boot, "_ready_executable", lambda _key: FakeExecutable())
    launched = []
    annotated = []

    @contextmanager
    def annotate(observed):
        annotated.append(observed)
        yield

    import pypto_plugins.activity_trace as activity_trace

    monkeypatch.setattr(activity_trace, "annotate_artifact_launch", annotate)
    _boot.launch_graph(key, (), 123)
    assert annotated == [record]
    assert launched == [("packet", 123)]


def test_compile_structured_uses_uncached_only_without_configured_cache(
    monkeypatch,
):
    uncached_calls = []
    cached_calls = []
    compiler = SimpleNamespace(
        compile_structured_strict=lambda *args: uncached_calls.append(args)
        or "uncached",
        compile_structured_strict_cached=lambda *args: cached_calls.append(args)
        or "cached",
    )
    monkeypatch.delenv("PYPTO_CACHE_DIR", raising=False)
    monkeypatch.delenv("PYPTO_STRICT_COVERAGE", raising=False)
    monkeypatch.setattr(_boot, "_artifact_cache_handle", None)
    monkeypatch.setattr(_boot, "_artifact_cache_pid", None)
    monkeypatch.setattr(_boot, "_artifact_cache_root", None)

    assert _boot._compile_structured(compiler, "program", "request", "schedule") == "uncached"
    assert uncached_calls == [("program", "request", "schedule")]
    assert cached_calls == []


def test_compile_structured_cold_and_warm_share_one_lazy_cache_handle(monkeypatch):
    created = []
    uncached_calls = []
    cached_calls = []

    class FakeArtifactCache:
        def __init__(self, root):
            self.root = root
            created.append(self)

    compiler = SimpleNamespace(
        ArtifactCache=FakeArtifactCache,
        compile_structured_strict=lambda *args: uncached_calls.append(args),
        compile_structured_strict_cached=lambda *args: cached_calls.append(args)
        or ("cold" if len(cached_calls) == 1 else "warm"),
    )
    monkeypatch.setenv("PYPTO_CACHE_DIR", "/cache/root")
    monkeypatch.setenv("PYPTO_STRICT_COVERAGE", "1")
    monkeypatch.setattr(_boot, "_artifact_cache_handle", None)
    monkeypatch.setattr(_boot, "_artifact_cache_pid", None)
    monkeypatch.setattr(_boot, "_artifact_cache_root", None)

    assert _boot._compile_structured(compiler, "p", "r", "s") == "cold"
    assert _boot._compile_structured(compiler, "p", "r", "s") == "warm"
    assert len(created) == 1
    assert created[0].root == "/cache/root"
    assert [call[-1] for call in cached_calls] == [created[0], created[0]]
    assert uncached_calls == []


def test_artifact_cache_rebuilds_after_pid_change_and_resets_snapshot(monkeypatch):
    created = []
    current_pid = [101]

    class FakeArtifactCache:
        def __init__(self, root):
            self.root = root
            created.append(self)

    compiler = SimpleNamespace(ArtifactCache=FakeArtifactCache)
    monkeypatch.setenv("PYPTO_CACHE_DIR", "/cache/root")
    monkeypatch.setattr(_boot.os, "getpid", lambda: current_pid[0])
    monkeypatch.setattr(_boot, "_artifact_cache_handle", None)
    monkeypatch.setattr(_boot, "_artifact_cache_pid", None)
    monkeypatch.setattr(_boot, "_artifact_cache_root", None)
    monkeypatch.setattr(_boot, "_compile_snapshot_pid", current_pid[0])
    monkeypatch.setattr(_boot, "_COMPILE_SNAPSHOT", [{"parent": "record"}])

    parent_cache = _boot._artifact_cache_for(compiler)
    assert _boot._artifact_cache_for(compiler) is parent_cache
    current_pid[0] = 202
    child_cache = _boot._artifact_cache_for(compiler)
    assert child_cache is not parent_cache
    assert len(created) == 2
    assert _boot.artifact_compile_snapshot() == []


def test_artifact_cache_configuration_fails_closed(monkeypatch):
    compiler = SimpleNamespace(
        ArtifactCache=lambda _root: (_ for _ in ()).throw(
            RuntimeError("unsafe or missing cache root")
        )
    )
    monkeypatch.setattr(_boot, "_artifact_cache_handle", None)
    monkeypatch.setattr(_boot, "_artifact_cache_pid", None)
    monkeypatch.setattr(_boot, "_artifact_cache_root", None)

    monkeypatch.delenv("PYPTO_CACHE_DIR", raising=False)
    monkeypatch.setenv("PYPTO_STRICT_COVERAGE", "1")
    with pytest.raises(RuntimeError, match="strict coverage requires"):
        _boot._artifact_cache_for(compiler)

    monkeypatch.setenv("PYPTO_STRICT_COVERAGE", "0")
    monkeypatch.setenv("PYPTO_CACHE_DIR", "relative/cache")
    with pytest.raises(RuntimeError, match="non-empty absolute"):
        _boot._artifact_cache_for(compiler)

    monkeypatch.setenv("PYPTO_CACHE_DIR", "/missing/cache")
    with pytest.raises(RuntimeError, match="unsafe or missing"):
        _boot._artifact_cache_for(compiler)

    compiler_without_cached_api = SimpleNamespace(
        ArtifactCache=lambda _root: object(),
        compile_structured_strict=lambda *_args: "uncached",
    )
    monkeypatch.setenv("PYPTO_CACHE_DIR", "/cache/root")
    with pytest.raises(RuntimeError, match="requires compile_structured_strict_cached"):
        _boot._compile_structured(
            compiler_without_cached_api, "program", "request", "schedule"
        )


def test_compile_snapshot_uses_full_cache_key_and_returns_detached_copy(
    monkeypatch,
):
    monkeypatch.setattr(_boot, "_compile_snapshot_pid", _boot.os.getpid())
    monkeypatch.setattr(_boot, "_COMPILE_SNAPSHOT", [])
    cache_key = "a" * 64
    artifact = SimpleNamespace(
        cache_key_digest=cache_key,
        identity_digest="c" * 64,
    )
    assert _boot._artifact_cache_key(artifact) == cache_key

    for disposition in sorted(_boot._COMPILE_DISPOSITIONS):
        result = SimpleNamespace(
            build_spec=SimpleNamespace(identity_digest="b" * 64),
            artifact=artifact,
            disposition=SimpleNamespace(name=disposition),
        )
        _boot._record_compile_snapshot(
            result,
            provider="pypto.attention",
            source_node="pypto_kernels.attention:paged_decode",
            cache_key=cache_key,
        )
    snapshot = _boot.artifact_compile_snapshot()
    assert len(snapshot) == 4
    assert {record["disposition"] for record in snapshot} == set(
        _boot._COMPILE_DISPOSITIONS
    )
    assert all(record["cache_key"] == cache_key for record in snapshot)
    assert all(record["build_spec_identity"] == "b" * 64 for record in snapshot)
    assert all(record["artifact_identity"] == "c" * 64 for record in snapshot)
    snapshot[0]["cache_key"] = "mutated"
    assert _boot.artifact_compile_snapshot()[0]["cache_key"] == cache_key

    with pytest.raises(RuntimeError, match="lowercase SHA-256"):
        _boot._artifact_cache_key(
            SimpleNamespace(cache_key_digest="d" * 16)
        )
    with pytest.raises(RuntimeError, match="unknown disposition"):
        _boot._record_compile_snapshot(
            SimpleNamespace(
                build_spec=SimpleNamespace(identity_digest="b" * 64),
                artifact=artifact,
                disposition=SimpleNamespace(name="CacheHitAfterWait"),
            ),
            provider="pypto.attention",
            source_node="pypto_kernels.attention:paged_decode",
            cache_key=cache_key,
        )


def test_each_operator_is_one_program():
    assert attention.GRAPHS == 7  # dense + masked + gather + decode + cache write + prefill + copy
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
            qk_rmsnorm_rope.GRAPHS,
            gdn.GRAPHS,
            gdn_projection.GRAPHS,
        )
    )
    assert embedding.GRAPHS == 2
    assert linear.GRAPHS == 2
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
    assert str(programs["sigmoid_mul"]).count("pl.tile.cast") == 3
    pitched_gate = torch.empty_strided(
        (3, 256), (512, 1), dtype=torch.bfloat16, device="meta"
    )
    pitched_program = sigmoid_mul.sigmoid_mul_kernel.specialize(
        sample, pitched_gate, sample
    )
    assert "pl.TensorView(stride=[512, 1]" in str(pitched_program)
    packed = torch.empty((3, 512), dtype=torch.bfloat16, device="meta")
    packed_program = silu_and_mul.silu_and_mul_kernel.specialize(
        packed[:, :256], packed[:, 256:], sample
    )
    assert str(packed_program).count("pl.TensorView(stride=[512, 1]") == 2
    assert silu_and_mul._tiles(1, 3584, 7168, 7168, 3584) == [128]
    assert silu_and_mul._tiles(19, 3584, 3584, 3584, 3584) == [128]
    assert silu_and_mul._tiles(19, 3584, 7168, 7168, 3584) == [1, 128]
    assert sigmoid_mul._tiles(1, 1024, 1024, 1024) == [128]
    assert sigmoid_mul._tiles(19, 1024, 1024, 1024) == [128]
    assert sigmoid_mul._tiles(19, 1024, 2048, 2048) == [1, 128]


def test_rmsnorm_uses_native_tile_reduction():
    assert rmsnorm._tiles(1) == [128]
    assert rmsnorm._tiles(19) == [1, 128]
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
    assert fused_add_rmsnorm._tiles(1) == [128]
    assert fused_add_rmsnorm._tiles(19) == [1, 128]
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
    assert gated_rmsnorm._tiles(1) == [128]
    assert gated_rmsnorm._tiles(19) == [1, 128]
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
    assert rendered.count("pl.tile.load") == 5
    assert "history1_box" in rendered and "256 + channel_offset" in rendered
    assert "pl.tile.exp" in rendered and "pl.tile.recip" in rendered
    assert rendered.count("pl.tile.store") == 4
    prefill = str(causal_conv1d.build(1, 5, 128))
    assert "pl.range(5)" in prefill
    with pytest.raises(ValueError, match="bounded to one"):
        causal_conv1d.compile_for(1, 2, 128, 65, 128 * 3, "int32")


def test_embedding_is_one_dynamic_row_gather_graph():
    assert embedding._tiles(1) == [128]
    assert embedding._tiles(19) == [8, 128]
    assert embedding._tiles(32) == [8, 128]
    program = embedding.build(8, 1024, 256)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert "pl.range(8)" in rendered
    assert "pl.range(2)" in rendered
    assert "pl.tensor.read" in rendered
    assert "pl.tile.load" in rendered
    assert "pl.tile.store" in rendered
    integer_program = embedding.integer_gather_kernel.specialize(
        torch.empty((2, 1), dtype=torch.int32, device="meta"),
        torch.empty((1, 1), dtype=torch.int64, device="meta"),
        torch.empty((1, 1), dtype=torch.int32, device="meta"),
    )
    integer_rendered = str(integer_program)
    assert "table: pl.Tensor[[2, 1], pl.INT32]" in integer_rendered
    assert "pl.tensor.read" in integer_rendered


def test_qk_norm_partial_rope_gate_is_one_native_graph():
    assert qk_rmsnorm_rope._tiles(1) == [1, 1, 1, 1, 32]
    assert qk_rmsnorm_rope._tiles(19) == [1, 1, 1, 1, 1, 32]
    program = qk_rmsnorm_rope.build(2, 8, 2, 256, 64, 1024)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert "pl.range(8)" in rendered and "pl.range(2)" in rendered
    assert rendered.count("pl.tile.row_sum") == 2
    assert "pl.tensor.read" in rendered
    assert "pl.tile.slice" in rendered
    assert "pl.tile.concat" in rendered
    assert rendered.count("pl.tile.store") == 3
    pitched = str(
        qk_rmsnorm_rope.build(2, 8, 2, 256, 64, 1024, 4608, 1024)
    )
    assert "pl.TensorView(stride=[4608, 1]" in pitched
    assert "pl.TensorView(stride=[1024, 1]" in pitched
    pitched_fp32_cache = str(
        qk_rmsnorm_rope.build(
            2, 8, 2, 256, 64, 1024, 4608, 1024, True, True
        )
    )
    assert "cos_sin_cache: pl.Tensor[[1024, 64], pl.FP32]" in pitched_fp32_cache
    assert "repeated_positions: pl.Tensor[[2, 64], pl.INT32]" in pitched_fp32_cache


@pytest.mark.parametrize(
    ("rows", "q_heads", "kv_heads"),
    ((1, 8, 2), (19, 8, 2), (1, 16, 4), (19, 16, 4)),
)
def test_qk_real_qwen35_decode_and_prefill_geometries(
    rows: int, q_heads: int, kv_heads: int
) -> None:
    rendered = str(
        qk_rmsnorm_rope.build(rows, q_heads, kv_heads, 256, 64, 262144)
    )
    assert f"pl.range({rows})" in rendered or rows == 1
    assert f"pl.range({q_heads})" in rendered
    assert f"pl.range({kv_heads})" in rendered
    assert rendered.count("pl.tile.store") == 3


def _one_program(p):
    assert len(p.functions) == 1
    fn = list(p.functions.values())[0]
    return fn


def test_rope_is_one_native_tile_graph():
    assert rope._tiles(1) == [1, 64]
    assert rope._tiles(19) == [1, 1, 64]
    program = rope.build(2, 64)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert rendered.count("pl.tile.load") == 4
    assert rendered.count("pl.tile.store") == 2
    assert "pl.tile.sub" in rendered and "pl.tile.add" in rendered
    assert "tensor." not in rendered


def test_attention_is_one_native_tile_graph():
    assert attention._dense_tiles(1) == [64]
    assert attention._dense_tiles(19) == [1, 64]
    assert attention._dense_tiles(64) == [1, 64]
    program = attention.build(2, 128, 128, 128)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert rendered.count("pl.tile.matmul") == 2
    assert "pl.tile.row_max" in rendered
    assert "pl.tile.row_sum" in rendered
    assert "pl.tile.exp" in rendered
    assert "pl.tile.store" in rendered
    assert "tensor." not in rendered


def test_paged_attention_decode_builds_one_q_head_graph():
    assert attention._paged_decode_partition_count(8) == 8
    assert attention._paged_decode_partition_count(16) == 16
    assert attention._paged_decode_tiles(1, 1) == [64]
    assert attention._paged_decode_tiles(1, 4) == [1, 64]
    assert attention._paged_decode_tiles(2, 1) == [1, 64]
    with pytest.raises(ValueError, match="positive Q heads"):
        attention._paged_decode_partition_count(0)
    program = attention.build_paged_decode(
        1,
        1,
        1,
        16,
        256,
        1024,
        65,
        4096,
        cache_row_stride=512,
        query_row_stride=2048,
        result_row_stride=2048,
    )
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2
    assert "pl.range(1)" in rendered and "pl.range(16)" in rendered
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

    assert "pl.TensorView(stride=[2048, 256, 1]" in rendered
    assert "pl.TensorView(stride=[512, 1]" in rendered

    wide_program = attention.build_paged_decode(
        1,
        1,
        1,
        512,
        256,
        1024,
        65,
        4096,
        cache_row_stride=512,
        query_row_stride=2048,
        result_row_stride=2048,
    )
    wide_rendered = str(wide_program)
    assert "pl.range(512)" in wide_rendered
    assert "pl.tile.ci" in wide_rendered
    assert "pl.tile.cmps" in wide_rendered
    assert "pl.tile.sels" in wide_rendered

    large_model_program = attention.build_paged_decode(
        1,
        1,
        1,
        16,
        256,
        1024,
        65,
        4096,
        cache_row_stride=1024,
        query_row_stride=4096,
        result_row_stride=4096,
    )
    large_model_rendered = str(large_model_program)
    assert "pl.range(1)" in large_model_rendered
    assert large_model_rendered.count("pl.tile.gather_row") == 2
    assert large_model_rendered.count("pl.tile.matmul") == 2

    batched_program = attention.build_paged_decode(
        2,
        1,
        1,
        16,
        256,
        1024,
        65,
        4096,
        cache_row_stride=2048,
        query_row_stride=2048,
        result_row_stride=2048,
    )
    batched_rendered = str(batched_program)
    assert "pl.range(2)" in batched_rendered
    assert batched_rendered.count("pl.tile.gather_row") == 2


def test_paged_attention_decode_reuses_one_artifact_and_zero_copy_views(
    monkeypatch,
):
    batch_size, q_heads, kv_heads, head_dim = 2, 8, 2, 256
    query_width = q_heads * head_dim
    cache_width = kv_heads * head_dim
    query_row_stride, cache_row_stride = 4096, 2048
    query_backing = torch.empty(
        7 + (batch_size - 1) * query_row_stride + query_width,
        dtype=torch.bfloat16,
    )
    query = query_backing.as_strided(
        (batch_size, query_width),
        (query_row_stride, 1),
        storage_offset=7,
    )
    cache_elements = 11 + (1024 - 1) * cache_row_stride + cache_width
    key_backing = torch.empty(cache_elements, dtype=torch.bfloat16)
    value_backing = torch.empty(cache_elements + 2, dtype=torch.bfloat16)
    key_cache = key_backing.as_strided(
        (1024, cache_width), (cache_row_stride, 1), storage_offset=11
    )
    value_cache = value_backing.as_strided(
        (1024, cache_width), (cache_row_stride, 1), storage_offset=13
    )
    req_to_token = torch.empty((65, 4096), dtype=torch.int32)
    request_index = torch.empty((batch_size,), dtype=torch.int64)
    valid_tokens = torch.empty((batch_size,), dtype=torch.int64)
    virtual_to_physical = torch.empty((1024,), dtype=torch.int64)
    compile_calls = []
    launches = []
    allocations = []
    real_empty = torch.empty

    def fake_compile(*args):
        compile_calls.append(args)
        return "one-q-head-artifact"

    def fake_launch(graph_key, operands, cuda_stream):
        launches.append((graph_key, operands, cuda_stream))

    def recording_empty(*args, **kwargs):
        result = real_empty(*args, **kwargs)
        allocations.append(result)
        return result

    monkeypatch.setattr(attention, "compile_paged_decode_for", fake_compile)
    monkeypatch.setattr(attention, "launch_graph", fake_launch)
    monkeypatch.setattr(torch, "empty", recording_empty)
    result = attention.paged_attention_decode(
        query,
        key_cache,
        value_cache,
        req_to_token,
        request_index,
        valid_tokens,
        virtual_to_physical,
        kv_heads=kv_heads,
        bucket_tokens=16,
        stream=SimpleNamespace(cuda_stream=123),
    )

    assert compile_calls == [
        (
            batch_size,
            1,
            1,
            16,
            head_dim,
            1024,
            65,
            4096,
            cache_row_stride,
            1024,
            query_row_stride,
            query_width,
        )
    ]
    assert len(allocations) == 1
    assert result is allocations[0]
    assert tuple(result.shape) == (batch_size, query_width)
    assert len(launches) == q_heads
    assert {launch[0] for launch in launches} == {"one-q-head-artifact"}
    assert {launch[2] for launch in launches} == {123}
    for q_head, (_graph_key, operands, _stream) in enumerate(launches):
        kv_head = q_head // (q_heads // kv_heads)
        query_group, key_group, value_group, *_metadata, result_group = operands
        assert tuple(query_group.shape) == (batch_size, 1, head_dim)
        assert tuple(query_group.stride()) == (query_row_stride, head_dim, 1)
        assert query_group.storage_offset() == 7 + q_head * head_dim
        assert query_group.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()
        for cache_group, cache, base_offset in (
            (key_group, key_cache, 11),
            (value_group, value_cache, 13),
        ):
            assert tuple(cache_group.shape) == (1024, head_dim)
            assert tuple(cache_group.stride()) == (cache_row_stride, 1)
            assert cache_group.storage_offset() == base_offset + kv_head * head_dim
            assert (
                cache_group.untyped_storage().data_ptr()
                == cache.untyped_storage().data_ptr()
            )
        assert tuple(result_group.shape) == (batch_size, 1, head_dim)
        assert tuple(result_group.stride()) == (query_width, head_dim, 1)
        assert result_group.storage_offset() == q_head * head_dim
        assert (
            result_group.untyped_storage().data_ptr()
            == result.untyped_storage().data_ptr()
        )


def test_paged_attention_decode_result_stride_is_in_artifact_identity(monkeypatch):
    builds = []
    compilations = []

    def fake_build(*shape):
        builds.append(shape)
        return shape

    def fake_compile(program, tiles, **metadata):
        compilations.append((program, tiles, metadata))
        return f"decode-artifact-{len(compilations)}"

    monkeypatch.setattr(attention, "_paged_decode_cache", {})
    monkeypatch.setattr(attention, "build_paged_decode", fake_build)
    monkeypatch.setattr(attention, "compile_graph", fake_compile)
    common = (1, 1, 1, 16, 256, 1024, 65, 4096, 1024, 1024, 4096)
    dense_result = attention.compile_paged_decode_for(*common, 1024)
    pitched_result = attention.compile_paged_decode_for(*common, 4096)
    reused_pitched_result = attention.compile_paged_decode_for(*common, 4096)

    assert dense_result == "decode-artifact-1"
    assert pitched_result == reused_pitched_result == "decode-artifact-2"
    assert [shape[-1] for shape in builds] == [1024, 4096]
    assert len(compilations) == 2
    assert [tiles for _program, tiles, _metadata in compilations] == [[64], [64]]


def test_paged_cache_layout_accepts_static_row_pitch() -> None:
    key_cache = torch.empty_strided(
        (1024, 512), (2048, 1), dtype=torch.bfloat16, device="meta"
    )
    value_cache = torch.empty_strided(
        (1024, 512), (2048, 1), dtype=torch.bfloat16, device="meta"
    )
    assert (
        attention._paged_cache_row_stride(key_cache, value_cache, operation="test")
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
    pitched_updates = str(
        attention.build_paged_cache_write(
            1024,
            13,
            512,
            key_row_stride=1536,
            value_row_stride=2048,
        )
    )
    assert "pl.TensorView(stride=[1536, 1]" in pitched_updates
    assert "pl.TensorView(stride=[2048, 1]" in pitched_updates


def test_paged_prefill_is_one_causal_gqa_graph():
    assert attention._paged_prefill_partition_count(2) == 1
    assert attention._paged_prefill_partition_count(4) == 4
    with pytest.raises(ValueError, match="positive KV heads"):
        attention._paged_prefill_partition_count(0)
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
    pitched_query = str(
        attention.build_paged_prefill(
            13, 8, 2, 16, 256, 1024, 65, 4096, query_row_stride=4096
        )
    )
    assert "pl.TensorView(stride=[256, 4096, 1]" in pitched_query
    pitched_result = str(
        attention.build_paged_prefill(
            13,
            8,
            2,
            16,
            256,
            1024,
            65,
            4096,
            query_row_stride=4096,
            result_row_stride=8192,
        )
    )
    assert "pl.TensorView(stride=[256, 8192, 1]" in pitched_result

    large_model = attention.build_paged_prefill(13, 16, 4, 16, 256, 1024, 65, 4096)
    large_rendered = str(large_model)
    assert "kv_heads: pl.Scalar[pl.INDEX] = 4" in large_rendered
    assert "queries_per_kv: pl.Scalar[pl.INDEX] = 16 // kv_heads" in large_rendered
    assert large_rendered.count("pl.tile.gather_row") == 2


def test_paged_prefill_tiles_follow_canonical_iteration_rank():
    assert attention._paged_prefill_tiles(1, 1, 1) == [128]
    assert attention._paged_prefill_tiles(19, 1, 1) == [1, 128]
    assert attention._paged_prefill_tiles(19, 4, 1) == [1, 1, 128]
    assert attention._paged_prefill_tiles(19, 8, 2) == [1, 1, 1, 128]
    with pytest.raises(ValueError, match="positive divisible GQA geometry"):
        attention._paged_prefill_tiles(19, 7, 2)


def test_paged_prefill_launches_head_major_zero_copy_views(monkeypatch):
    query_rows, q_heads, kv_heads, head_dim = 19, 8, 2, 256
    query_width = q_heads * head_dim
    query = torch.empty((query_rows, query_width), dtype=torch.bfloat16)
    key_cache = torch.empty((257, kv_heads * head_dim), dtype=torch.bfloat16)
    value_cache = torch.empty_like(key_cache)
    req_to_token = torch.empty((2, 260), dtype=torch.int32)
    request_index = torch.empty((1,), dtype=torch.int64)
    prefix_tokens = torch.empty((1,), dtype=torch.int32)
    virtual_to_physical = torch.empty((257,), dtype=torch.int64)
    compile_calls = []
    launches = []

    monkeypatch.setattr(
        attention,
        "compile_paged_prefill_for",
        lambda *shape: compile_calls.append(shape) or "head-major-prefill",
    )
    monkeypatch.setattr(
        attention,
        "_launch_paged_prefill_graph",
        lambda graph, operands, stream: launches.append((graph, operands, stream)),
    )

    result = attention.paged_attention_prefill(
        query,
        key_cache,
        value_cache,
        req_to_token,
        request_index,
        prefix_tokens,
        virtual_to_physical,
        kv_heads=kv_heads,
        bucket_tokens=32,
        stream=SimpleNamespace(cuda_stream=123),
    )

    assert compile_calls == [
        (
            query_rows,
            q_heads,
            kv_heads,
            32,
            head_dim,
            257,
            2,
            260,
            kv_heads * head_dim,
            257,
            query_width,
            query_width,
        )
    ]
    assert len(launches) == 1
    graph, operands, stream = launches[0]
    assert graph == "head-major-prefill"
    assert stream.cuda_stream == 123
    query_view, *_metadata, result_view = operands
    assert tuple(query_view.shape) == (q_heads, query_rows, head_dim)
    assert tuple(query_view.stride()) == (head_dim, query_width, 1)
    assert query_view.data_ptr() == query.data_ptr()
    assert tuple(result_view.shape) == (q_heads, query_rows, head_dim)
    assert tuple(result_view.stride()) == (head_dim, query_width, 1)
    assert tuple(result.shape) == (query_rows, query_width)
    assert result_view.untyped_storage().data_ptr() == result.untyped_storage().data_ptr()


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
    assert linear._tiles(1) == [128]
    assert linear._tiles(19) == [1, 128]
    float_program = linear.build(1, 128, 256, "float32")
    float_rendered = str(float_program)
    assert float_rendered.count("pl.tile.cast") == 2
    assert "pl.Out[pl.Tensor[[1, 256], pl.FP32]]" in float_rendered
    pitched = str(
        linear.build(19, 1024, 128, output_row_stride=8192)
    )
    assert "pl.TensorView(stride=[8192, 1]" in pitched


@pytest.mark.parametrize(
    ("rows", "hidden", "intermediate"),
    ((1, 1024, 3584), (19, 1024, 3584), (1, 4096, 12288), (19, 4096, 12288)),
)
def test_linear_and_lm_head_real_qwen35_geometries(
    rows: int, hidden: int, intermediate: int
) -> None:
    dense = str(linear.build(rows, hidden, 2 * intermediate))
    lm_head = str(linear.build(rows, hidden, 248320, "float32"))
    assert f"pl.Tensor[[{rows}, {hidden}], pl.BF16]" in dense
    assert f"pl.Out[pl.Tensor[[{rows}, {2 * intermediate}], pl.BF16]]" in dense
    assert f"pl.Out[pl.Tensor[[{rows}, 248320], pl.FP32]]" in lm_head
    assert lm_head.count("pl.tile.cast") == 2


def test_gdn_recurrent_is_one_mutation_declared_graph():
    program = gdn.build_recurrent(2, 1, 8, 16, 128, 128, 65, 16 * 128 * 128 + 4096)
    rendered = str(program)
    assert len(_one_program(program).body.stmts) == 2  # one scope + return
    assert rendered.count("pl.InOut[") == 1
    assert "pl.range(2)" in rendered and "pl.range(16)" in rendered
    assert rendered.count("pl.tile.row_sum") == 2  # Q and K L2 norms
    assert rendered.count("pl.tile.cast") >= 8  # includes BF16 dt_bias -> FP32
    assert rendered.count("pl.tile.matmul") == 2  # state*K and Q*state
    assert "pl.tile.abs" in rendered and "pl.tile.maximums" in rendered
    assert "pl.tile.row_expand_mul" in rendered
    assert "pl.tile.row_expand" in rendered
    assert "pl.tile.col_expand_mul" in rendered
    assert rendered.count("pl.tile.store") == 2  # output plus final FP32 state
    prefill = str(gdn.build_recurrent(1, 3, 8, 16, 128, 128, 65))
    assert "pl.range(3)" in prefill
    with pytest.raises(ValueError, match="bounded to one"):
        gdn.compile_recurrent(1, 2, 8, 16, 128, 128, 65, 16 * 128 * 128, "int32")


def test_gdn_recurrent_schedule_omits_unit_iteration_dimensions():
    assert gdn._recurrent_tiles(1, 1, 16, 16, 128) == [1, 128]
    assert gdn._recurrent_tiles(1, 1, 8, 16, 128) == [1, 1, 128]
    assert gdn._recurrent_tiles(2, 1, 8, 16, 128) == [1, 1, 1, 128]
    assert gdn._recurrent_tiles(1, 1, 1, 1, 48) == [32]


@pytest.mark.parametrize(
    ("value_heads", "channels"), ((16, 6144), (32, 8192))
)
def test_stateful_real_qwen35_geometries_are_single_token_primitives(
    value_heads: int, channels: int
) -> None:
    conv = str(causal_conv1d.build(1, 1, channels))
    recurrent = str(
        gdn.build_recurrent(1, 1, 16, value_heads, 128, 128, 65)
    )
    projection = str(gdn_projection.build(19, 16, value_heads, 128, 128))
    assert f"pl.Tensor[[1, 1, {channels}], pl.BF16]" in conv
    assert f"pl.range({value_heads})" in recurrent
    assert "pl.range(19)" in projection
    assert causal_conv1d._MAX_FUSED_TOKENS == 1
    assert gdn._MAX_FUSED_TOKENS == 1


def test_gdn_projection_split_is_one_packed_output_graph():
    program = gdn_projection.build(3, 8, 16, 128, 128)
    rendered = str(program)
    function = _one_program(program)
    assert len(function.return_types) == 1
    assert rendered.count("pl.tile.load") == 4
    assert rendered.count("pl.tile.store") == 4
    assert rendered.count("pl.Out[") == 1
    assert "pl.Tensor[[1, 18528]" in rendered
    pitched = str(
        gdn_projection.build(
            3,
            8,
            16,
            128,
            128,
            qkvz_row_stride=8192,
            ba_row_stride=128,
        )
    )
    assert "pl.TensorView(stride=[8192, 1]" in pitched
    assert "pl.TensorView(stride=[128, 1]" in pitched


def test_broadcast_dependencies_are_closed():
    assert rmsnorm.STATUS.endswith("executable")
    assert rope.STATUS.endswith("executable")
    assert not hasattr(rmsnorm, "BLOCKED_ON")
    assert not hasattr(rope, "BLOCKED_ON")
    assert not hasattr(gdn, "BLOCKED_ON")
    assert causal_conv1d.STATUS.endswith("executable")
    assert gdn.STATUS.endswith("executable")
    assert gdn_projection.STATUS.endswith("executable")


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
    test_paged_attention_decode_builds_one_q_head_graph()
    test_paged_cache_write_declares_mutation_and_one_graph()
    test_paged_prefill_is_one_causal_gqa_graph()
    test_linear_is_one_native_tile_graph()
    test_gdn_recurrent_is_one_mutation_declared_graph()
    test_gdn_projection_split_is_one_packed_output_graph()
    test_broadcast_dependencies_are_closed()
    print("ALL OPERATOR STRUCTURE TESTS PASSED")
