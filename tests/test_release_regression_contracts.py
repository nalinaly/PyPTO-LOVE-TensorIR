from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unicodedata

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import (  # noqa: E402
    controllers,
    correctness_runtime,
    lanes,
    performance_runtime,
    profile_runtime,
    sglang_compat,
    workload,
)


def _compile_snapshot_record(disposition: str) -> dict[str, str]:
    return {
        "source_node": "pypto_kernels.attention:paged_decode",
        "provider": "pypto.attention",
        "cache_key": "a" * 64,
        "build_spec_identity": "b" * 64,
        "artifact_identity": "c" * 64,
        "disposition": disposition,
    }


def test_candidate_compile_cache_evidence_is_strict_but_hit_is_observational() -> None:
    dispositions = (
        "Uncached",
        "CacheHit",
        "CompiledAndPublished",
        "CompiledAndValidatedExisting",
    )
    source = [_compile_snapshot_record(value) for value in dispositions]
    evidence = correctness_runtime._compile_cache_evidence(source)
    assert evidence["record_count"] == 4
    assert evidence["cache_hit_observed"] is True
    assert evidence["cache_hit_required"] is False
    assert evidence["coverage_identity_includes_snapshot"] is False
    assert evidence["disposition_counts"] == {value: 1 for value in dispositions}
    evidence["records"][0]["cache_key"] = "mutated"
    assert source[0]["cache_key"] == "a" * 64

    no_hit = correctness_runtime._compile_cache_evidence(
        [_compile_snapshot_record("CompiledAndPublished")]
    )
    assert no_hit["cache_hit_observed"] is False
    assert no_hit["cache_hit_required"] is False


def test_candidate_compile_cache_evidence_rejects_invalid_records() -> None:
    with pytest.raises(
        workload.ReleaseContractError, match="no PyPTO compile snapshot"
    ):
        correctness_runtime._compile_cache_evidence([])
    with pytest.raises(workload.ReleaseContractError, match="unknown disposition"):
        correctness_runtime._compile_cache_evidence(
            [_compile_snapshot_record("CacheHitAfterWait")]
        )
    malformed = _compile_snapshot_record("CacheHit")
    malformed["cache_key"] = "a" * 16
    with pytest.raises(workload.ReleaseContractError, match="invalid cache_key"):
        correctness_runtime._compile_cache_evidence([malformed])


def load_tool(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(f"release_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


operator_tool = load_tool("run_operator_regression.py")
model_tool = load_tool("run_model_correctness.py")
performance_tool = load_tool("run_performance_regression.py")
profile_tool = load_tool("profile_qwen35.py")
render_tool = load_tool("render_release_results.py")


def display_width_for_test(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in value
    )


def test_exact_workload_has_one_machine_readable_source_of_truth() -> None:
    manifest = json.loads(
        (ROOT / "benchmarks/release/workload.json").read_text(encoding="utf-8")
    )
    assert manifest == workload.workload_record()
    assert manifest["prompt"] == (
        "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
    )
    assert len(manifest["prompt_token_ids"]) == 31
    assert manifest["prompt_tokens"] == 31
    assert len(manifest["raw_prompt_token_ids"]) == 19
    assert manifest["raw_prompt_tokens"] == 19
    assert manifest["workload_kind"] == "qwen35-chat-template-nonthinking"
    assert manifest["output_tokens"] == 64
    assert manifest["concurrency"] == 1
    assert manifest["greedy"] is True
    assert manifest["ignore_eos"] is True


def test_compact_ablation_output_fits_one_terminal_view() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/print_inductor_ablation.py"),
            "--compact",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.splitlines()
    assert max(map(len, lines)) <= 64
    assert any("[prefill 19x24576]" in line for line in lines)
    assert any("[decode 1x24576]" in line for line in lines)
    text = completed.stdout
    for marker in ("E=eager", "N=NV", "P=PyPTO", "cold/launch", "LR="):
        assert marker in text


def test_compact_model_gate_replay_fits_one_terminal_view() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/print_qwen35_model_gate.py"),
            "--compact",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.splitlines()
    assert len(lines) == 7
    assert max(map(display_width_for_test, lines)) <= 66
    text = completed.stdout
    for marker in (
        "not a live rerun",
        "3 starts x 10 Engine requests",
        "coverage=29728/29728",
        "handwritten=27680",
        "Inductor=2048",
        workload.PROMPT,
        "output:",
    ):
        assert marker in text


@pytest.mark.parametrize(
    ("gate", "markers"),
    (
        ("build", ("wheel build=pass", "CTest=13/13", "same artifact set")),
        ("operator", ("8/8 suites", "all_correct=true", "Inductor-SwiGLU=4")),
    ),
)
def test_compact_release_gate_replay_fits_one_terminal_view(
    gate: str, markers: tuple[str, ...]
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/print_release_gate.py"),
            gate,
            "--compact",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.splitlines()
    assert len(lines) <= 7
    assert max(map(display_width_for_test, lines)) <= 66
    assert "not a live rerun" in completed.stdout
    for marker in markers:
        assert marker in completed.stdout


def test_chat_workload_is_reproducible_from_both_pinned_tokenizers() -> None:
    for name in ("Qwen3.5-0.8B", "Qwen3.5-9B"):
        model_path = ROOT / "models" / name
        spec = workload.resolve_qwen35_model_spec(ROOT, model_path)
        record, resolution = workload.verify_chat_workload(model_path, spec)
        assert resolution["verified"] is True
        assert resolution["input_token_count"] == 31
        assert record["prompt_tokens"] == 31
        assert record["raw_prompt_tokens"] == 19
        assert record["rendered_input"].endswith("</think>\n\n")


def test_workload_schema_freezes_every_field() -> None:
    schema = json.loads(
        (ROOT / "benchmarks/release/workload.schema.json").read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(workload.workload_record())
    for key, value in workload.workload_record().items():
        assert schema["properties"][key]["const"] == value


def test_release_parallelism_and_sample_counts_are_frozen() -> None:
    assert workload.CPU_JOBS == 24
    assert workload.MEASURED_REQUESTS == 10
    assert model_tool.FRESH_STARTS == 3
    assert profile_runtime.PROFILE_REQUESTS == 5
    assert correctness_runtime.TEACHER_FORCED_REQUESTS == 1


def test_model_correctness_all_mode_is_one_public_closed_loop() -> None:
    args = model_tool.parser().parse_args(
        [
            "all",
            "--model-path",
            "models/Qwen3.5-9B",
            "--semantic-oracle",
            "runs/semantic-oracle-qwen35-0.8b-chat-nonthinking.json",
            "--dry-run",
        ]
    )
    assert args.mode == "all"
    assert args.reference_report is None


def _write_model_spec_fixture(root: Path) -> dict[str, Path]:
    layers = {"Qwen3.5-0.8B": 24, "Qwen3.5-9B": 32}
    paths = {}
    records = {}
    for name, layer_count in layers.items():
        path = root / "models" / name
        path.mkdir(parents=True)
        (path / "config.json").write_text(
            json.dumps({"text_config": {"num_hidden_layers": layer_count}}),
            encoding="utf-8",
        )
        paths[name] = path
        records[name] = {
            "destination": f"models/{name}",
            "repository_id": f"Qwen/{name}",
        }
    (root / "models/MANIFEST.json").write_text(
        json.dumps({"schema": 1, "models": records}), encoding="utf-8"
    )
    return paths


def test_model_correctness_spec_drives_both_model_contracts(tmp_path: Path) -> None:
    paths = _write_model_spec_fixture(tmp_path)
    small = workload.resolve_qwen35_model_spec(tmp_path, paths["Qwen3.5-0.8B"])
    large = workload.resolve_qwen35_model_spec(tmp_path, paths["Qwen3.5-9B"])
    assert small.record() == {
        "manifest_name": "Qwen3.5-0.8B",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_size": "0.8B",
        "report_stem": "qwen35-0.8b",
        "num_hidden_layers": 24,
        "expected_inductor_calls": 24 * 64,
    }
    assert large.record() == {
        "manifest_name": "Qwen3.5-9B",
        "model_id": "Qwen/Qwen3.5-9B",
        "model_size": "9B",
        "report_stem": "qwen35-9b",
        "num_hidden_layers": 32,
        "expected_inductor_calls": 32 * 64,
    }
    assert model_tool._report_name(small, "reference") == ("qwen35-0.8b-reference.json")
    assert model_tool._report_name(small, "candidate") == (
        "qwen35-0.8b-correctness.json"
    )
    assert model_tool._report_name(large, "reference") == "qwen35-9b-reference.json"
    assert model_tool._report_name(large, "candidate") == ("qwen35-9b-correctness.json")
    assert workload.workload_record(small)["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert workload.workload_record(large) == workload.workload_record()


def test_model_correctness_spec_rejects_config_layer_drift(tmp_path: Path) -> None:
    paths = _write_model_spec_fixture(tmp_path)
    (paths["Qwen3.5-0.8B"] / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 32}}),
        encoding="utf-8",
    )
    with pytest.raises(workload.ReleaseContractError, match="layer count differs"):
        workload.resolve_qwen35_model_spec(tmp_path, paths["Qwen3.5-0.8B"])


def test_performance_surface_has_no_numerical_acceptance_inputs() -> None:
    parser = performance_tool.parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--lane",
                "pypto",
                "--model-path",
                "model",
                "--reference-logits",
                "value.pt",
            ]
        )
    texts = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "tools/run_performance_regression.py",
            "benchmarks/release/performance_runtime.py",
        )
    )
    forbidden_imports = (
        "correctness_runtime",
        "run_model_correctness",
        "compare_qwen35_logits",
    )
    assert all(name not in texts for name in forbidden_imports)
    assert "torch.allclose" not in texts
    assert "cosine_similarity" not in texts
    assert "torch.topk" not in texts


def test_timing_request_discards_generated_values() -> None:
    class FakeEngine:
        def generate(self, **_kwargs):
            for completion in range(1, 65):
                yield {
                    "output_ids": [1000 + completion],
                    "meta_info": {"completion_tokens": completion},
                }

    result = performance_runtime._stream_request(FakeEngine(), 0)
    assert result["completion_tokens"] == 64
    assert "output_ids" not in result
    assert "output_text" not in result
    assert "passed" not in result
    assert len(result["chunk_timestamps"]) == 64
    assert len(result["itl_ms"]) == 63


def test_effective_compilation_requires_scheduler_counters_and_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    after = {
        "SGLANG_CACHE_DIR/torch_compile_cache/computation_graph_1.py": {
            "bytes": 4,
            "suffix": ".py",
        },
        "SGLANG_CACHE_DIR/torch_compile_cache/kernel.so": {
            "bytes": 4,
            "suffix": ".so",
        },
    }
    observed = performance_runtime._compilation_observation(
        {},
        after,
        requested=True,
        scheduler_counter={"num_graphs_seen": 1, "num_inductor_compiles": 1},
    )
    flag_only = performance_runtime._compilation_observation(
        {}, after, requested=True, scheduler_counter={}
    )
    assert observed["effective"] is True
    assert flag_only["effective"] is False
    pypto_cache = performance_runtime._compilation_observation(
        {},
        {
            "TORCHINDUCTOR_CACHE_DIR/aa/generated.py": {
                "bytes": 4,
                "suffix": ".py",
                "contains_pypto_launch": True,
            }
        },
        requested=True,
        scheduler_counter={},
        lane="pypto",
    )
    assert pypto_cache["effective"] is True
    assert pypto_cache["backend_invocation_evidence"][
        "pypto_torchinductor_cache_wrapper"
    ] is True
    optimized_cache = performance_runtime._compilation_observation(
        {},
        {
            "TORCHINDUCTOR_CACHE_DIR/aa/generated.py": {
                "bytes": 4,
                "suffix": ".py",
                "is_torch_inductor_wrapper": True,
            }
        },
        requested=True,
        scheduler_counter={},
        lane="sglang-optimized",
    )
    assert optimized_cache["effective"] is True
    assert optimized_cache["backend_invocation_evidence"][
        "optimized_torchinductor_cache_wrapper"
    ] is True

    cache = tmp_path / "torchinductor"
    wrapper = cache / "aa/generated.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "\n".join(
            (
                "from torch._inductor.async_compile import AsyncCompile",
                "from torch._inductor.runtime.triton_heuristics import start_graph",
                "async_compile = AsyncCompile()",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache))
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(tmp_path / "sglang"))
    snapshot = performance_runtime._cache_snapshot()
    assert snapshot["TORCHINDUCTOR_CACHE_DIR/aa/generated.py"][
        "is_torch_inductor_wrapper"
    ] is True


def test_engine_shutdown_retries_only_the_sglang_reap_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Engine:
        calls = 0

        def shutdown(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("2 process(es) not reaped within 60s")

    engine = Engine()
    monkeypatch.setattr(performance_runtime.time, "sleep", lambda _seconds: None)
    attempts = performance_runtime._shutdown_engine_with_retry(engine)
    assert [record["status"] for record in attempts] == ["failed", "complete"]
    assert engine.calls == 2

    class OtherFailure:
        def shutdown(self) -> None:
            raise RuntimeError("unrelated failure")

    with pytest.raises(RuntimeError, match="unrelated failure"):
        performance_runtime._shutdown_engine_with_retry(OtherFailure())


def test_matched_performance_control_may_record_disabled_compile_effective() -> None:
    source = (ROOT / "benchmarks/release/performance_runtime.py").read_text(
        encoding="utf-8"
    )
    assert 'if lane != "sglang-matched"' in source
    assert "Matched control deliberately disables CUDA graphs" in source


def test_performance_matrix_is_the_frozen_balanced_twelve_start_order() -> None:
    expected = (
        "pypto",
        "sglang-matched",
        "sglang-matched",
        "pypto",
        "sglang-optimized",
        "pypto",
        "pypto",
        "sglang-optimized",
        "sglang-matched",
        "sglang-optimized",
        "sglang-optimized",
        "sglang-matched",
    )
    assert performance_tool.MATRIX_SCHEDULE == expected
    assert {lane: expected.count(lane) for lane in workload.LANES} == {
        lane: 4 for lane in workload.LANES
    }
    pair = (
        "pypto",
        "sglang-matched",
        "sglang-matched",
        "pypto",
        "sglang-matched",
        "pypto",
        "pypto",
        "sglang-matched",
    )
    assert workload.PAIR_PERFORMANCE_SCHEDULE == pair
    assert pair.count("pypto") == pair.count("sglang-matched") == 4
    assert "sglang-optimized" not in pair


def test_performance_pair_matrix_is_an_independent_public_mode() -> None:
    args = performance_tool.parser().parse_args(
        [
            "--pair-matrix",
            "--model-path",
            "models/Qwen3.5-9B",
            "--optimized-memory-mode",
            "matched",
            "--dry-run",
        ]
    )
    assert args.pair_matrix is True
    assert args.matrix is False
    assert args.lane is None


def test_profile_matrix_is_three_starts_and_five_requests_per_lane() -> None:
    expected = (
        "pypto",
        "sglang-matched",
        "sglang-optimized",
        "sglang-matched",
        "sglang-optimized",
        "pypto",
        "sglang-optimized",
        "pypto",
        "sglang-matched",
    )
    assert profile_tool.PROFILE_SCHEDULE == expected
    assert all(expected.count(lane) == 3 for lane in workload.LANES)
    assert profile_runtime.PROFILE_REQUESTS == 5


def test_lane_memory_and_provider_qualifications_are_explicit() -> None:
    model = Path("model")
    pypto = lanes.server_kwargs("pypto", model)
    matched = lanes.server_kwargs("sglang-matched", model)
    optimized = lanes.server_kwargs("sglang-optimized", model)
    qualified_optimized = lanes.server_kwargs(
        "sglang-optimized", model, optimized_memory_mode="matched"
    )
    large_pypto = lanes.server_kwargs("pypto", Path("Qwen3.5-9B"))
    large_matched = lanes.server_kwargs("sglang-matched", Path("Qwen3.5-9B"))
    performance_pypto = lanes.performance_server_kwargs(
        "pypto", Path("Qwen3.5-9B")
    )
    performance_matched = lanes.performance_server_kwargs(
        "sglang-matched", Path("Qwen3.5-9B")
    )
    small_pypto = lanes.server_kwargs("pypto", Path("Qwen3.5-0.8B"))
    small_matched = lanes.server_kwargs("sglang-matched", Path("Qwen3.5-0.8B"))
    for config in (pypto, matched, optimized, qualified_optimized):
        assert "language_model_only" not in config
        assert config["json_model_override_args"] == ('{"language_model_only":true}')
    assert pypto["cpu_offload_gb"] == matched["cpu_offload_gb"] == 2
    assert pypto["mem_fraction_static"] == matched["mem_fraction_static"] == 0.78
    assert large_pypto["cpu_offload_gb"] == 0
    assert large_matched["cpu_offload_gb"] == 2
    assert large_pypto["mem_fraction_static"] == 0.78
    assert large_matched["mem_fraction_static"] == 0.78
    assert performance_pypto["cpu_offload_gb"] == 2
    assert performance_matched["cpu_offload_gb"] == 2
    assert performance_pypto["mem_fraction_static"] == 0.78
    assert performance_matched["mem_fraction_static"] == 0.78
    assert lanes.matched_lane_comparability(
        performance_pypto, performance_matched
    )["matched_claim_allowed"] is False  # resolved controls are required
    performance_resolved_pypto = {
        "sampling_backend": "pytorch",
        "torch_compile_requested": True,
        "cuda_graph_enabled_by_server_args": False,
        "overlap_schedule_enabled_by_server_args": False,
        "radix_cache_enabled_by_server_args": False,
    }
    performance_resolved_matched = dict(performance_resolved_pypto)
    assert lanes.matched_lane_comparability(
        performance_pypto,
        performance_matched,
        performance_resolved_pypto,
        performance_resolved_matched,
    )["matched_claim_allowed"] is True
    assert small_pypto["cpu_offload_gb"] == small_matched["cpu_offload_gb"] == 0
    assert (
        small_pypto["mem_fraction_static"]
        == small_matched["mem_fraction_static"]
        == 0.78
    )
    assert optimized["cpu_offload_gb"] == 0
    assert "mem_fraction_static" not in optimized
    assert qualified_optimized["cpu_offload_gb"] == 2
    assert qualified_optimized["mem_fraction_static"] == 0.69
    for config in (matched, optimized):
        assert config["linear_attn_backend"] == "triton"
        assert config["linear_attn_decode_backend"] == "triton"
        assert config["linear_attn_prefill_backend"] == "triton"
        assert config["mamba_ssm_dtype"] == "float32"
    assert pypto["sampling_backend"] == matched["sampling_backend"] == "pytorch"
    assert optimized["sampling_backend"] == "flashinfer"
    assert matched["disable_cuda_graph"] is True
    assert matched["disable_overlap_schedule"] is True
    assert "disable_cuda_graph" not in optimized
    assert "disable_overlap_schedule" not in optimized


def test_requested_compile_flag_is_not_reported_as_effective() -> None:
    fake = SimpleNamespace(
        get_attention_backends=lambda: ("flashinfer", "flashinfer"),
        linear_attn_backend="triton",
        linear_attn_prefill_backend="triton",
        linear_attn_decode_backend="triton",
        mamba_backend="triton",
        mamba_ssm_dtype="float32",
        sampling_backend="flashinfer",
        enable_torch_compile=True,
        disable_cuda_graph=False,
        disable_overlap_schedule=False,
        disable_radix_cache=False,
        cuda_graph_config={"decode": {"backend": "full"}},
    )
    record = lanes.resolved_backend_record(fake)
    assert record["torch_compile_requested"] is True
    assert "torch_compile" not in record
    lanes.validate_resolved_backends("sglang-optimized", record)


def test_runtime_contract_uses_portable_release_prefixes() -> None:
    runtime = json.loads(
        (ROOT / "benchmarks/release/runtime.json").read_text(encoding="utf-8")
    )
    assert runtime["control_prefix"] == "envs/pypto-release"
    assert runtime["profiles"]["pypto"] == {
        "environment": "pypto-release",
        "prefix": "envs/pypto-release",
    }
    assert runtime["profiles"]["baseline"] == {
        "environment": "sglang-baseline",
        "prefix": "envs/sglang-baseline",
    }


def test_formal_worker_rejects_diagnostic_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYPTO_FRAMEWORK_PROFILE", "pypto")
    monkeypatch.setenv("PYPTO_KERNEL_CUDART", "/diagnostic/libcudart.so")
    with pytest.raises(workload.ReleaseContractError, match="diagnostic runtime"):
        lanes.prepare_worker_environment("pypto")


def test_shared_gemma_offload_compatibility_is_strict_and_backend_neutral() -> None:
    class FakeDevice:
        def __init__(self, kind: str):
            self.type = kind

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeDevice) and self.type == other.type

        def __str__(self) -> str:
            return self.type

    class FakeTensor:
        def __init__(self, device: str):
            self.device = FakeDevice(device)

        def to(self, *, device: FakeDevice):
            return FakeTensor(device.type)

    same = SimpleNamespace(gemma_weight=FakeTensor("cpu"))
    assert sglang_compat._colocate_gemma_weight(same, FakeTensor("cpu")) is False

    offloaded = SimpleNamespace(gemma_weight=FakeTensor("cuda"))
    assert sglang_compat._colocate_gemma_weight(offloaded, FakeTensor("cpu")) is True
    assert offloaded.gemma_weight.device.type == "cpu"

    unsupported = SimpleNamespace(gemma_weight=FakeTensor("cpu"))
    with pytest.raises(workload.ReleaseContractError, match="unsupported device split"):
        sglang_compat._colocate_gemma_weight(unsupported, FakeTensor("cuda"))

    class FakeGemmaRMSNorm:
        def _weight_loader(self, param, loaded_weight):
            assert self.gemma_weight.device == param.device
            return loaded_weight

    first = sglang_compat._install_on_class(FakeGemmaRMSNorm)
    second = sglang_compat._install_on_class(FakeGemmaRMSNorm)
    assert first["disposition"] == "installed"
    assert second["disposition"] == "already-installed"
    layer = FakeGemmaRMSNorm()
    layer.gemma_weight = FakeTensor("cuda")
    param = FakeTensor("cpu")
    assert layer._weight_loader(param, "loaded") == "loaded"
    assert layer.gemma_weight.device.type == "cpu"

    calls = []

    def functional_call(module, state, *, args=None, kwargs=None, tie_weights=True):
        calls.append((module, state, args, kwargs, tie_weights))
        return "called"

    offloader = SimpleNamespace(functional_call=functional_call)
    offloader_first = sglang_compat._install_offloader_functional_call(offloader)
    offloader_second = sglang_compat._install_offloader_functional_call(offloader)
    assert offloader_first["disposition"] == "installed"
    assert offloader_second["disposition"] == "already-installed"

    original = SimpleNamespace(
        data_ptr=lambda: 1234,
        _version=7,
        shape=(2, 3),
    )
    replacement = SimpleNamespace(shape=(2, 3))
    module = SimpleNamespace(
        named_parameters=lambda *, remove_duplicate: [("alias", original)]
    )
    assert offloader.functional_call(module, {"alias": replacement}) == "called"
    assert calls[-1][-1] is False
    assert replacement._pypto_offload_source_signature == (
        "offloaded",
        1234,
        7,
        (2, 3),
    )
    with pytest.raises(workload.ReleaseContractError, match="tie_weights=False"):
        offloader.functional_call(module, {}, tie_weights=True)

    class FakeViewTensor(FakeTensor):
        def __init__(self, device: str, shape):
            super().__init__(device)
            self.shape = shape

        def view(self, *shape):
            return FakeViewTensor(self.device.type, shape)

    original_view = FakeViewTensor("cuda", (4, 3))
    replacement_weight = FakeViewTensor("cuda", (4, 1, 3))
    gdn_module = SimpleNamespace(
        linear_attn=SimpleNamespace(
            conv1d=SimpleNamespace(weight=FakeViewTensor("cpu", (4, 1, 3))),
            attn=SimpleNamespace(conv_weights=original_view),
        ),
        named_parameters=lambda *, remove_duplicate: [],
    )

    observed_view = []

    def functional_call_with_view(module, state, *, args=None, kwargs=None, tie_weights=True):
        observed_view.append(module.linear_attn.attn.conv_weights)
        assert observed_view[-1].shape == (4, 3)
        return "view-called"

    view_offloader = SimpleNamespace(functional_call=functional_call_with_view)
    sglang_compat._install_offloader_functional_call(view_offloader)
    assert (
        view_offloader.functional_call(
            gdn_module, {"linear_attn.conv1d.weight": replacement_weight}
        )
        == "view-called"
    )
    assert gdn_module.linear_attn.attn.conv_weights is original_view

    record = {
        "applies_equally_to_lanes": [
            "pypto",
            "sglang-matched",
            "sglang-optimized",
        ],
        "components": [first, offloader_first],
    }
    assert record["applies_equally_to_lanes"] == [
        "pypto",
        "sglang-matched",
        "sglang-optimized",
    ]
    assert [component["scope"] for component in record["components"]] == [
        "model-weight-load-only",
        "cpu-offloaded-module-forward",
    ]


def test_controller_commands_route_through_generalized_bounded_controls(
    tmp_path: Path,
) -> None:
    root = tmp_path
    for relative in (
        "envs/pypto-release/bin/python",
        "envs/sglang-baseline/bin/python",
        "tools/run_pypto_gpu_bounded.py",
        "tools/run_pypto_cpu_bounded.py",
        "tools/worker.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    runtime = json.loads(
        (ROOT / "benchmarks/release/runtime.json").read_text(encoding="utf-8")
    )
    target = root / "benchmarks/release/runtime.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(runtime), encoding="utf-8")
    command = controllers.isolated_command(
        root,
        root / "tools/worker.py",
        ("--_worker",),
        tmp_path / "pointer.json",
        framework_profile="baseline",
        timeout_seconds=10,
    )
    assert str(root / "tools/run_pypto_gpu_bounded.py") in command
    assert "sglang-baseline" in command
    assert "baseline" in command
    floor_index = command.index("--gpu-free-floor-mib")
    assert command[floor_index + 1] == str(4 * 1024)
    optimized_command = controllers.isolated_command(
        root,
        root / "tools/worker.py",
        ("--_worker",),
        tmp_path / "optimized-pointer.json",
        framework_profile="baseline",
        timeout_seconds=10,
        gpu_free_floor_mib=0,
    )
    optimized_floor_index = optimized_command.index("--gpu-free-floor-mib")
    assert optimized_command[optimized_floor_index + 1] == "0"
    cpu_command = controllers.isolated_command(
        root,
        root / "tools/worker.py",
        ("--_worker", "--_jobs", "24"),
        tmp_path / "cpu-pointer.json",
        framework_profile="pypto",
        timeout_seconds=10,
        cpu_only=True,
    )
    assert str(root / "tools/run_pypto_cpu_bounded.py") in cpu_command
    assert "pypto-release" in cpu_command
    assert cpu_command[-2:] == ("--_jobs", "24")


def test_operator_defaults_to_portable_package_and_explicit_output_contract(
    tmp_path: Path,
) -> None:
    args = operator_tool.parser().parse_args(["--stage", "structure"])
    assert args.kernel_root == ROOT / "packages/pypto-kernels"
    accepted = tmp_path / "accepted.py"
    accepted.write_text('parser.add_argument("--output")\n', encoding="utf-8")
    operator_tool._require_portable_output_contract(accepted)
    rejected = tmp_path / "rejected.py"
    rejected.write_text("result.write_text('x')\n", encoding="utf-8")
    with pytest.raises(workload.ReleaseContractError, match="--output"):
        operator_tool._require_portable_output_contract(rejected)


def test_operator_manifest_covers_compile_numerical_and_graph_gates() -> None:
    manifest = json.loads(
        (ROOT / "benchmarks/release/operator_manifest.json").read_text(encoding="utf-8")
    )
    suites = manifest["gpu_suites"]
    assert [Path(item["path"]).name for item in suites] == [
        "classify_sm120.py",
        "exec_sm120.py",
        "stateful_sm120.py",
        "paged_attention_sm120.py",
        "qk_sm120.py",
        "linear_sm120.py",
        "cuda_graph_stateful_sm120.py",
        "qwen35_swiglu_torch_compile_sm120.py",
    ]
    assert all("success_field" in item for item in suites)
    assert manifest["schema"] == 2
    assert manifest["structure"] == {
        "jobs": 24,
        "paths": [
            "packages/pypto-kernels/tests/test_operators.py",
            "packages/pypto-kernels/tests/test_portable_release.py",
        ],
    }
    assert [
        path.name
        for path in operator_tool._structure_sources(ROOT / "packages/pypto-kernels")
    ] == ["test_operators.py", "test_portable_release.py"]
    resolved = operator_tool._operator_suites(ROOT / "packages/pypto-kernels")
    assert [item[0]["id"] for item in resolved] == [item["id"] for item in suites]
    assert all(source.is_file() for _raw, source, _scope, _args in resolved)


def test_operator_manifest_locks_real_model_shape_and_launch_contracts() -> None:
    manifest = json.loads(
        (ROOT / "benchmarks/release/operator_manifest.json").read_text(encoding="utf-8")
    )
    suites = {item["id"]: item for item in manifest["gpu_suites"]}
    compiled = suites["handwritten-compile-classification"]["case_expectations"]
    assert len(compiled) == 20
    assert len({item["where"]["case"] for item in compiled}) == 20
    assert all(item["equals"]["status"] == "compiled" for item in compiled)
    compiled_by_case = {item["where"]["case"]: item["equals"] for item in compiled}
    assert {
        case: {
            "compiled_artifacts": compiled_by_case[case]["compiled_artifacts"],
            "launch_count": compiled_by_case[case]["launch_count"],
            "attention_launches": compiled_by_case[case]["attention_launches"],
        }
        for case in (
            "attention_paged_decode_0_8b",
            "attention_paged_decode_9b",
            "attention_paged_decode_batch2_strided_0_8b",
        )
    } == {
        "attention_paged_decode_0_8b": {
            "compiled_artifacts": 1,
            "launch_count": 8,
            "attention_launches": 8,
        },
        "attention_paged_decode_9b": {
            "compiled_artifacts": 1,
            "launch_count": 16,
            "attention_launches": 16,
        },
        "attention_paged_decode_batch2_strided_0_8b": {
            "compiled_artifacts": 1,
            "launch_count": 8,
            "attention_launches": 8,
        },
    }
    numerical = suites["handwritten-numerical"]["case_expectations"]
    assert {
        item["where"]["case"]
        for item in numerical
        if item["where"]["case"].startswith("embedding_bf16_rows")
    } == {
        "embedding_bf16_rows1_248320x1024",
        "embedding_bf16_rows19_248320x1024",
    }
    stateful = suites["stateful-real-model-shapes"]["case_expectations"]
    assert any(
        item["where"] == {"case": "gdn_recurrent_Qwen3.5-9B_rows19"}
        and item["equals"]["q_heads"] == 16
        and item["equals"]["value_heads"] == 32
        and item["equals"]["launches"] == 19
        for item in stateful
    )
    paged = suites["paged-attention"]["case_expectations"]
    assert suites["paged-attention"]["expected_case_count"] == 14
    decode_launches = {
        item["where"]["case"]: item["equals"]
        for item in paged
        if item["where"]["case"].startswith("decode_")
    }
    assert {
        case: (
            values["kv_heads"],
            values["launches"],
            values["attention_launches"],
            values["compiled_artifacts"],
        )
        for case, values in decode_launches.items()
    } == {
        "decode_0_8b_valid13": (2, 8, 8, 1),
        "decode_9b_valid16": (4, 16, 16, 1),
        "decode_batch2_0_8b_valid13_7_strided": (2, 8, 8, 1),
        "decode_0_8b_valid83_bucket96": (2, 8, 8, 1),
    }
    assert any(
        item["where"] == {"case": "prefill_9b_prefix2_extend13"}
        and item["equals"]
        == {
            "correct": True,
            "kv_heads": 4,
            "launches": 5,
            "launch_count": 5,
            "cache_write_launches": 1,
            "attention_launches": 4,
        }
        for item in paged
    )
    swiglu = suites["inductor-swiglu-real-model-shapes"]
    assert swiglu["expected_case_count"] == 4
    assert {
        (item["where"]["model"], item["where"]["rows"])
        for item in swiglu["case_expectations"]
    } == {
        ("Qwen3.5-0.8B", 1),
        ("Qwen3.5-0.8B", 19),
        ("Qwen3.5-9B", 1),
        ("Qwen3.5-9B", 19),
    }


def test_operator_case_expectations_fail_closed() -> None:
    suite = {
        "case_expectations": [
            {
                "where": {"model": "Qwen3.5-9B", "rows": 19},
                "equals": {"correct": True, "launches": 19},
            }
        ],
        "expected_case_count": 1,
    }
    operator_tool._validate_case_expectations(
        {
            "cases": [
                {
                    "model": "Qwen3.5-9B",
                    "rows": 19,
                    "correct": True,
                    "launches": 19,
                }
            ]
        },
        suite,
    )
    with pytest.raises(workload.ReleaseContractError, match="expectation failed"):
        operator_tool._validate_case_expectations(
            {
                "cases": [
                    {
                        "model": "Qwen3.5-9B",
                        "rows": 19,
                        "correct": True,
                        "launches": 1,
                    }
                ]
            },
            suite,
        )


def test_engine_token_mismatch_evidence_is_exact_and_length_aware() -> None:
    assert correctness_runtime._token_sequence_mismatch([1, 2, 3], [1, 9, 3]) == {
        "first_mismatch_step": 1,
        "expected_token_id": 2,
        "observed_token_id": 9,
    }
    assert correctness_runtime._token_sequence_mismatch([1, 2], [1]) == {
        "first_mismatch_step": 1,
        "expected_token_id": 2,
        "observed_token_id": None,
    }
    assert correctness_runtime._token_sequence_mismatch([1], [1, 2]) == {
        "first_mismatch_step": 1,
        "expected_token_id": None,
        "observed_token_id": 2,
    }
    assert correctness_runtime._token_sequence_mismatch([1, 2], [1, 2]) == {
        "first_mismatch_step": None,
        "expected_token_id": None,
        "observed_token_id": None,
    }


def test_step_parity_accepts_only_an_exact_candidate_maximum_tie() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([0.0, 2.0, 1.999, 0.0, -1.0])
    candidate = torch.tensor([0.0, 2.0, 2.0, 0.0, -1.0])
    tied = correctness_runtime._step_parity(
        torch, [1], 2, reference, candidate
    )
    assert tied["passed"] is True
    assert tied["exact_greedy_token_id"] is False
    assert tied["metrics"]["candidate_max_tie_count"] == 2
    assert tied["checks"]["reference_token_at_candidate_maximum"] is True
    assert tied["checks"]["sampled_token_at_candidate_maximum"] is True

    below_maximum = candidate.clone()
    below_maximum[1] = 1.999
    rejected = correctness_runtime._step_parity(
        torch, [1], 2, reference, below_maximum
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["reference_token_at_candidate_maximum"] is False


def test_candidate_numerical_gate_uses_reference_prefix_teacher_forcing() -> None:
    source = (ROOT / "benchmarks/release/correctness_runtime.py").read_text(
        encoding="utf-8"
    )
    candidate = source[source.index("def run_candidate(") :]
    assert "_generate_teacher_forced(" in candidate
    assert '"evaluation_mode": "teacher-forced-reference-prefixes"' in candidate
    assert "end-to-end SGLang Engine output is incomplete or unstable" in candidate
    assert '"explained_by_teacher_forced_maximum_tie"' in candidate
    teacher_forced = source[
        source.index("def _generate_teacher_forced(") : source.index(
            "def _shutdown_runner("
        )
    ]
    assert teacher_forced.index("forced_tokens = [") < teacher_forced.index(
        "def traced_forward("
    )
    assert teacher_forced.index("torch.cuda.synchronize()") < teacher_forced.index(
        "def traced_forward("
    )


def test_engine_acceptance_isolated_before_parent_cupti_start() -> None:
    source = (ROOT / "benchmarks/release/correctness_runtime.py").read_text(
        encoding="utf-8"
    )
    candidate = source[source.index("def run_candidate(") :]
    assert candidate.index("_run_engine_sequences_isolated(") < candidate.index(
        "from torch.profiler import _cupti_monitor"
    )
    isolated = source[
        source.index("def _run_engine_sequences_isolated(") : source.index(
            "def run_reference("
        )
    ]
    assert 'multiprocessing.get_context("spawn")' in isolated


@dataclass(frozen=True)
class _FakeCoverageEvent:
    activity_id: str
    provider: str
    call_count: int
    gpu_time_ns: int


def test_cross_window_coverage_events_aggregate_by_activity_id() -> None:
    aggregates: dict[str, object] = {}
    correctness_runtime._merge_coverage_event(
        aggregates, _FakeCoverageEvent("same", "pypto", 2, 11)
    )
    correctness_runtime._merge_coverage_event(
        aggregates, _FakeCoverageEvent("same", "pypto", 3, 17)
    )
    assert aggregates == {"same": _FakeCoverageEvent("same", "pypto", 5, 28)}
    with pytest.raises(workload.ReleaseContractError, match="conflicting activity"):
        correctness_runtime._merge_coverage_event(
            aggregates, _FakeCoverageEvent("same", "external", 1, 1)
        )


def test_cupti_profiler_overlay_is_hash_locked_without_prefix_install() -> None:
    artifacts = json.loads(
        (ROOT / "environment/python-artifacts.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (ROOT / "environment/release-runtime.json").read_text(encoding="utf-8")
    )
    overlay = artifacts["profiling_overlay"]
    wheel = artifacts["artifacts"]["cupti_python"]
    assert overlay == {
        "artifact": "cupti_python",
        "destination": "caches/cupti-python-overlay/13.2.0",
        "distribution": "cupti-python",
        "libcupti_relative": "targets/x86_64-linux/lib/libcupti.so",
        "python_abi": "cp314",
        "toolkit": "13.3.73",
        "toolkit_root_label": "cuda-13.3",
        "version": "13.2.0",
    }
    assert wheel["sha256"] == (
        "d57ad6cad9a757dcda2fdd4b7eaa9994991c739d569073e6d78820a25dee0ab7"
    )
    assert runtime["cuda"]["cupti_python_distribution_required"] is False
    assert runtime["cuda"]["cupti_python_mode"] == "hash-locked-overlay"
    source = (ROOT / "benchmarks/release/correctness_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "monitor.flush(forced=True)" in source
    assert 'monitor.stats()["buffers_completed"]' in source


def test_operator_manifest_paths_cannot_escape_or_own_output() -> None:
    for value in ("../escape.py", "/absolute.py", "a/./b.py"):
        with pytest.raises(workload.ReleaseContractError, match="relative path"):
            operator_tool._manifest_relative_path(value, "test path")
    with pytest.raises(workload.ReleaseContractError, match="output ownership"):
        operator_tool._suite_arguments({"arguments": ["--output", "foreign.json"]})
    with pytest.raises(workload.ReleaseContractError, match="unknown schema"):
        operator_tool._suite_arguments(
            {"path_arguments": [{"flag": "--model-root", "path": "models", "x": 1}]}
        )


def test_release_sources_have_no_workstation_absolute_paths() -> None:
    paths = [
        *sorted((ROOT / "benchmarks/release").glob("*")),
        ROOT / "tools/run_operator_regression.py",
        ROOT / "tools/run_model_correctness.py",
        ROOT / "tools/run_performance_regression.py",
        ROOT / "tools/profile_qwen35.py",
        ROOT / "tools/render_release_results.py",
    ]
    for path in paths:
        if not path.is_file() or path.suffix == ".pyc":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "/" + "home" + "/" not in text, path
        assert "/" + "Users" + "/" not in text, path
        assert "projects" + "/pypto-kernels" not in text, path


def test_run_directory_is_controller_owned_and_cannot_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYPTO_RUN_ID", raising=False)
    with pytest.raises(workload.ReleaseContractError):
        workload.require_run_directory(tmp_path)
    monkeypatch.setenv("PYPTO_RUN_ID", "../escape")
    with pytest.raises(workload.ReleaseContractError):
        workload.require_run_directory(tmp_path)
    monkeypatch.setenv("PYPTO_RUN_ID", "release-test-1")
    run_id, run_dir = workload.require_run_directory(tmp_path)
    assert run_id == "release-test-1"
    assert run_dir == tmp_path / "runs/release-test-1"


def test_logical_phase_aggregation_uses_annotations_and_artifacts() -> None:
    phase = json.dumps(
        {
            "kind": profile_runtime.PHASE_ANNOTATION_KIND,
            "schema_version": 1,
            "phase": "full_attention_qkv",
            "module": "model.layers.0.self_attn.q_proj",
        }
    )
    artifact = json.dumps(
        {
            "kind": profile_runtime.ARTIFACT_ANNOTATION_KIND,
            "schema_version": 1,
            "artifact": {
                "provider": "pypto.tensorir",
                "source_node": "pypto_kernels.qk_rmsnorm_rope:kernel",
            },
        }
    )
    window = {
        "user_annotations": {"7": phase, "8": artifact},
        "events": [
            {"kind": "external_correlation", "correlation_id": 10, "external_id": 7},
            {"kind": "external_correlation", "correlation_id": 11, "external_id": 8},
            {
                "kind": "kernel",
                "correlation_id": 10,
                "name": "gemm",
                "start_ns": 10,
                "end_ns": 110,
            },
            {
                "kind": "kernel",
                "correlation_id": 11,
                "name": "tile",
                "start_ns": 120,
                "end_ns": 320,
            },
            {"kind": "gpu_memcpy", "name": "copy", "start_ns": 330, "end_ns": 380},
        ],
    }
    rules = json.loads(
        (ROOT / "benchmarks/release/logical_phases.json").read_text(encoding="utf-8")
    )
    result = profile_runtime.aggregate_windows([window], rules)
    assert result["compute_gpu_time_ns"] == 300
    assert result["runtime_memcpy_gpu_time_ns"] == 50
    assert result["phase_totals"]["full_attention_qkv"]["gpu_time_ns"] == 100
    assert result["phase_totals"]["qk_rmsnorm_rope"]["gpu_time_ns"] == 200


def _profile(lane: str, compute_ns: int, phase_ns: int) -> dict[str, object]:
    requested = lanes.performance_server_kwargs(lane, Path("model"))
    resolved = {
        "sampling_backend": requested["sampling_backend"],
        "torch_compile_requested": requested["enable_torch_compile"],
        "cuda_graph_enabled_by_server_args": not requested.get(
            "disable_cuda_graph", False
        ),
        "overlap_schedule_enabled_by_server_args": not requested.get(
            "disable_overlap_schedule", False
        ),
        "radix_cache_enabled_by_server_args": not requested.get(
            "disable_radix_cache", False
        ),
    }
    request_aggregation = {
        "compute_gpu_time_ns": compute_ns / 5,
        "runtime_memcpy_gpu_time_ns": 0,
        "phase_totals": {"mlp_down": {"calls": 1, "gpu_time_ns": phase_ns / 5}},
    }
    return {
        "status": "complete",
        "lane": lane,
        "workload": workload.workload_record(),
        "profile_requests": 5,
        "requested_server_config": requested,
        "resolved_backends": resolved,
        "compilation": {"requested": True, "effective": True},
        "resources": {
            "nvml_error": None,
            "gpu_identity": {
                "name": "NVIDIA GeForce RTX 5090",
                "uuid": "GPU-test",
                "driver": "test",
                "total_memory_bytes": 24 * 1024**3,
            },
            "summary": {
                "minimum_gpu_memory_free_bytes": 4 * 1024**3,
                "peak_gpu_memory_used_bytes": 20 * 1024**3,
                "minimum_mem_available_kib": 16 * 1024**2,
                "peak_owned_pgid_rss_kib": 8 * 1024**2,
                "sample_count": 100,
                "thermal_throttle_observed": False,
            },
        },
        "requests": [
            {
                "request_index": index,
                "trace_attempts": 1,
                "aggregation": request_aggregation,
            }
            for index in range(5)
        ],
        "aggregation": {
            "compute_gpu_time_ns": compute_ns,
            "runtime_memcpy_gpu_time_ns": 0,
            "phase_totals": {"mlp_down": {"calls": 5, "gpu_time_ns": phase_ns}},
        },
    }


def test_profile_gap_reconciliation_is_per_request_and_closes() -> None:
    profiles = {
        "pypto": [_profile("pypto", 5_000_000, 5_000_000)] * 3,
        "sglang-matched": [_profile("sglang-matched", 2_500_000, 2_500_000)] * 3,
        "sglang-optimized": [_profile("sglang-optimized", 2_000_000, 2_000_000)] * 3,
    }
    result = profile_runtime.reconcile(profiles)
    matched = result["comparisons"]["sglang-matched"]
    assert matched["model_compute_gap_ms"] == pytest.approx(0.5)
    assert matched["phase_gap_sum_ms"] == pytest.approx(0.5)
    assert matched["phase_reconciliation_residual_ms"] == pytest.approx(0.0)
    assert result["profile_inputs"]["pypto"] == {
        "fresh_starts": 3,
        "profile_requests": 15,
    }


def test_renderer_defines_headline_ratio_from_median_output_rate() -> None:
    payload = {
        "lanes": {
            lane: {
                "fresh_starts": 4,
                "measured_requests": 40,
                "ttft_ms": {"p50": 1.0},
                "e2e_ms": {"p50": 2.0},
                "tpot_ms": {"p50": 0.1},
                "input_tokens_per_second": {"p50": 19.0},
                "decode_tokens_per_second": {"p50": 630.0},
                "output_tokens_per_second": {"p50": 100.0},
                "resources": {"peak_gpu_memory_used_bytes": 1024**3},
            }
            for lane in workload.LANES
        },
        "pypto_percent_of_stock": {
            "sglang-matched": 100.0,
            "sglang-optimized": 100.0,
        },
    }
    rendered = render_tool._performance_markdown(payload)
    assert "Output tok/s p50" in rendered
    assert "PyPTO / stock" in rendered
    assert rendered.count("100.00%") == 3
