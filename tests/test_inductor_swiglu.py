"""CPU-only contracts for the revision-bound Inductor SwiGLU route."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from pypto_plugins.errors import StrictCoverageError
from pypto_plugins.torch import inductor_swiglu, pointwise_codegen, runtime_bridge
from pypto_plugins.torch.scheduling import REGISTRY, _OpsRecorder


def _call_key() -> inductor_swiglu.SwiGLUCallableKey:
    compiler = pointwise_codegen.BackendRevisionIdentity(
        "pypto-revision",
        "tensor-ir-revision",
        "cuda-tile-revision",
        "llvm-revision",
        "13.3",
        "a" * 64,
        "b" * 64,
    )
    revisions = inductor_swiglu.CallableRevisionIdentity(
        "torch-revision",
        "pypto-backend-hash",
        compiler,
    )
    gate = inductor_swiglu.TensorCallIdentity(
        (19, 3584),
        (7168, 1),
        0,
        "bfloat16",
        "cuda",
        0,
    )
    up = inductor_swiglu.TensorCallIdentity(
        (19, 3584),
        (7168, 1),
        3584,
        "bfloat16",
        "cuda",
        0,
    )
    return inductor_swiglu.SwiGLUCallableKey(
        gate,
        up,
        (3584, 1),
        revisions,
    )


def test_fp32_swiglu_formula_casts_only_the_result_to_bf16() -> None:
    torch.manual_seed(9)
    gate = torch.randn((5, 128), dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    expected = (
        gate.float() * torch.sigmoid(gate.float()) * up.float()
    ).to(torch.bfloat16)
    assert torch.equal(inductor_swiglu.fp32_swiglu_subgraph(gate, up), expected)


def test_standard_torch_compile_contract_is_fullgraph_static(monkeypatch) -> None:
    observed = []
    sentinel = object()

    def compile_contract(function, **kwargs):
        observed.append((function, kwargs))
        return sentinel

    monkeypatch.setattr(torch, "compile", compile_contract)
    assert inductor_swiglu._compile_callable() is sentinel
    assert observed == [
        (
            inductor_swiglu.fp32_swiglu_subgraph,
            {"backend": "pypto", "fullgraph": True, "dynamic": False},
        )
    ]


def test_row_pitched_specialization_uses_empty_strided_metadata() -> None:
    shape = (19, 3584)
    row_pitched = pointwise_codegen.PointwiseTensorSpec(
        shape,
        (7168, 1),
        "bfloat16",
        "cuda",
        0,
    )
    output = pointwise_codegen.PointwiseTensorSpec.dense(shape, "bfloat16")
    builder = pointwise_codegen.PointwiseProgramBuilder(
        shape,
        "bfloat16",
        output_spec=output,
    )
    gate = builder.add_input("gate", specialization=row_pitched)
    up = builder.add_input("up", specialization=row_pitched)
    gate_wide = builder.emit(
        "tensor.cast", [gate, builder.dtype("float32")]
    )
    up_wide = builder.emit("tensor.cast", [up, builder.dtype("float32")])
    result = builder.emit("tensor.mul", [gate_wide, up_wide])
    result = builder.emit(
        "tensor.cast", [result, builder.dtype("bfloat16")]
    )
    builder.mark_output(result)
    program = builder.build()
    samples = program.specialization_samples()
    assert tuple(samples[0].stride()) == (7168, 1)
    assert tuple(samples[1].stride()) == (7168, 1)
    assert tuple(samples[2].stride()) == (3584, 1)
    assert all(sample.device.type == "meta" for sample in samples)

    contiguous_builder = pointwise_codegen.PointwiseProgramBuilder(
        shape,
        "bfloat16",
        output_spec=output,
    )
    contiguous_gate = contiguous_builder.add_input("gate")
    contiguous_result = contiguous_builder.emit("tensor.neg", [contiguous_gate])
    contiguous_builder.mark_output(contiguous_result)
    assert program != contiguous_builder.build()


def test_pointwise_cast_recorder_is_explicit_and_bitcast_fails_closed() -> None:
    recorder = _OpsRecorder(loop_arity=2)
    value = recorder.load("arg0", torch.tensor(0).new_tensor(0))
    wide = recorder.to_dtype(
        value,
        torch.float32,
        src_dtype=torch.bfloat16,
        use_compute_types=True,
    )
    recorder.to_dtype(
        wide,
        torch.bfloat16,
        src_dtype=torch.float32,
        use_compute_types=True,
    )
    assert [event[0] for event in recorder.events] == ["load", "cast", "cast"]
    assert recorder.events[1][2] == "float32"
    assert recorder.events[2][2] == "bfloat16"
    with pytest.raises(StrictCoverageError, match="reinterpret"):
        recorder.to_dtype_bitcast(value, torch.int16, torch.bfloat16)
    with pytest.raises(StrictCoverageError, match="src_dtype"):
        recorder.to_dtype(value, torch.float32, src_dtype=torch.float16)
    with pytest.raises(StrictCoverageError, match="use_compute_types=True"):
        recorder.to_dtype(
            value,
            torch.float32,
            src_dtype=torch.bfloat16,
            use_compute_types=False,
        )


def test_callable_cache_compiles_once_and_reuses_prewarmed_artifact(monkeypatch) -> None:
    key = _call_key()
    calls = []
    source = "@pl.jit\ndef generated_pointwise_kernel():\n    pass\n"
    artifact = pointwise_codegen.PointwiseArtifact(
        kernel_name="pypto_inductor_4444444444444444",
        entry_name="stable_entry",
        build_spec_sha256="1" * 64,
        artifact_sha256="2" * 64,
        cubin_sha256="3" * 64,
        cubin_bytes=1,
        grid=(1, 1, 1),
        argument_count=3,
        workspace_bytes=0,
        fallback_used=False,
        pypto_source=source,
        pypto_source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        cache_identity_sha256="4" * 64,
        source_node="torch-inductor:4444444444444444",
        dso_sha256="5" * 64,
    )
    REGISTRY.clear()
    inductor_swiglu.clear_callable_cache_for_testing()

    def compiled(_gate, _up):
        calls.append("launch")
        REGISTRY.register(artifact.kernel_name, artifact)
        pointwise_codegen._record_active_capture(artifact)
        return object()

    monkeypatch.setattr(inductor_swiglu, "callable_key", lambda *_args: key)
    monkeypatch.setattr(inductor_swiglu, "_compile_callable", lambda: compiled)
    monkeypatch.setattr(inductor_swiglu, "_validate_output", lambda *_args: None)
    monkeypatch.setattr(inductor_swiglu, "trace_window_active", lambda: False)
    monkeypatch.setattr(
        inductor_swiglu.runtime_bridge,
        "kernel_is_prewarmed",
        lambda name: name == artifact.kernel_name,
    )
    first = inductor_swiglu.run_fp32_swiglu(object(), object())
    second = inductor_swiglu.run_fp32_swiglu(object(), object())
    assert first is not second
    assert calls == ["launch", "launch"]
    snapshot = inductor_swiglu.callable_cache_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0][1:] == (artifact.kernel_name, artifact.source_node)


def test_callable_cache_miss_during_trace_fails_before_compile(monkeypatch) -> None:
    inductor_swiglu.clear_callable_cache_for_testing()
    monkeypatch.setattr(inductor_swiglu, "callable_key", lambda *_args: _call_key())
    monkeypatch.setattr(inductor_swiglu, "trace_window_active", lambda: True)
    monkeypatch.setattr(
        inductor_swiglu,
        "_compile_callable",
        lambda: pytest.fail("compile must not run inside a trace window"),
    )
    with pytest.raises(StrictCoverageError, match="not prepared before trace"):
        inductor_swiglu.run_fp32_swiglu(object(), object())


def test_artifact_cache_miss_during_trace_fails_before_compiler(monkeypatch) -> None:
    import pypto_plugins.activity_trace as activity_trace

    info = SimpleNamespace(
        compiled=True,
        pypto_revision="pypto",
        tensor_ir_revision="tensor-ir",
        cuda_tile_revision="cuda-tile",
        llvm_revision="llvm",
        cuda_toolkit_version="13.3",
        tileiras_sha256="b" * 64,
    )

    class Compiler:
        @staticmethod
        def get_nvidia_backend_build_info():
            return info

        @staticmethod
        def compile_structured_strict(*_args):
            pytest.fail("compiler must not run inside a trace window")

    builder = pointwise_codegen.PointwiseProgramBuilder((128,), "float32")
    value = builder.add_input("value")
    result = builder.emit("tensor.neg", [value])
    builder.mark_output(result)
    pointwise_codegen.clear_caches_for_testing()
    monkeypatch.setattr(
        pointwise_codegen,
        "bootstrap_pypto",
        lambda: {"compiler": Compiler, "pypto": object()},
    )
    monkeypatch.setattr(pointwise_codegen, "pypto_dso_sha256", lambda: "c" * 64)
    monkeypatch.setattr(activity_trace, "trace_window_active", lambda: True)
    with pytest.raises(StrictCoverageError, match="not compiled before the trace"):
        pointwise_codegen.compile_pointwise(builder.build(), tile=128)


def test_compiler_callable_and_registry_caches_fail_after_fork(monkeypatch) -> None:
    monkeypatch.setattr(
        inductor_swiglu,
        "_OWNER_PID",
        inductor_swiglu.os.getpid() + 1,
    )
    with pytest.raises(StrictCoverageError, match="inherited across fork"):
        inductor_swiglu.callable_cache_snapshot()


def test_registry_is_locked_and_pid_bound(monkeypatch) -> None:
    artifact = SimpleNamespace(cache_identity_sha256="a", marker="stable")
    REGISTRY.clear()
    REGISTRY.register("one", artifact)
    REGISTRY.register("one", artifact)
    assert REGISTRY.snapshot() == (("one", artifact),)
    monkeypatch.setattr(REGISTRY, "_owner_pid", REGISTRY._owner_pid + 1)
    with pytest.raises(StrictCoverageError, match="inherited across fork"):
        REGISTRY.snapshot()


def test_compile_and_prewarm_share_one_locked_transaction(monkeypatch) -> None:
    import pypto_plugins.activity_trace as activity_trace

    info = SimpleNamespace(
        compiled=True,
        pypto_revision="pypto",
        tensor_ir_revision="tensor-ir",
        cuda_tile_revision="cuda-tile",
        llvm_revision="llvm",
        cuda_toolkit_version="13.3",
        tileiras_sha256="6" * 64,
    )
    kernel_abi = SimpleNamespace(
        entry_function_name="entry",
        grid_abi=SimpleNamespace(static_dimensions=(1, 1, 1)),
        argument_layout=SimpleNamespace(total_kernel_argument_count=2),
        workspace_abi=SimpleNamespace(size_bytes=0),
    )

    class Artifact:
        device_code = b"cubin"
        fallback_used = False

        @staticmethod
        def serialize():
            return b"artifact"

    Artifact.kernel_abi = kernel_abi
    result = SimpleNamespace(
        artifact=Artifact(),
        build_spec=SimpleNamespace(serialize=lambda: b"build-spec"),
    )

    class Compiler:
        @staticmethod
        def get_nvidia_backend_build_info():
            return info

        @staticmethod
        def compile_structured_strict(*_args):
            assert pointwise_codegen._COMPILE_LOCK._is_owned()
            return result

    builder = pointwise_codegen.PointwiseProgramBuilder((128,), "float32")
    value = builder.add_input("value")
    output = builder.emit("tensor.neg", [value])
    builder.mark_output(output)
    pointwise_codegen.clear_caches_for_testing()
    monkeypatch.setattr(
        pointwise_codegen,
        "bootstrap_pypto",
        lambda: {"compiler": Compiler, "pypto": object()},
    )
    monkeypatch.setattr(pointwise_codegen, "pypto_dso_sha256", lambda: "7" * 64)
    monkeypatch.setattr(pointwise_codegen, "_reference_request", lambda *_a, **_k: object())
    monkeypatch.setattr(pointwise_codegen, "_reference_schedule", lambda *_a: object())
    monkeypatch.setattr(
        pointwise_codegen.NativePointwiseProgram,
        "specialize",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        pointwise_codegen.NativePointwiseProgram,
        "native_source",
        lambda *_a, **_k: "@pl.jit\ndef generated_pointwise_kernel():\n    pass\n",
    )
    monkeypatch.setattr(activity_trace, "trace_window_active", lambda: False)
    prewarmed = []

    def prewarm(name):
        assert pointwise_codegen._COMPILE_LOCK._is_owned()
        prewarmed.append(name)

    monkeypatch.setattr(runtime_bridge, "prewarm_kernel", prewarm)
    artifact = pointwise_codegen.compile_pointwise(
        builder.build(),
        tile=128,
        prewarm_runtime=True,
    )
    assert prewarmed == [artifact.kernel_name]
    assert artifact.dso_sha256 == "7" * 64
