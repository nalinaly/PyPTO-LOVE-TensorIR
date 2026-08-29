from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import (  # noqa: E402
    controllers,
    correctness_runtime,
    lanes,
    performance_runtime,
    profile_runtime,
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
    assert evidence["disposition_counts"] == {
        value: 1 for value in dispositions
    }
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


def test_exact_workload_has_one_machine_readable_source_of_truth() -> None:
    manifest = json.loads(
        (ROOT / "benchmarks/release/workload.json").read_text(encoding="utf-8")
    )
    assert manifest == workload.workload_record()
    assert manifest["prompt"] == (
        "为什么说鞠婧祎主演的《月鳞绮纪》是国产电视剧的巅峰之作？"
    )
    assert len(manifest["prompt_token_ids"]) == 19
    assert manifest["prompt_tokens"] == 19
    assert manifest["output_tokens"] == 64
    assert manifest["concurrency"] == 1
    assert manifest["greedy"] is True
    assert manifest["ignore_eos"] is True


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


def test_model_correctness_all_mode_is_one_public_closed_loop() -> None:
    args = model_tool.parser().parse_args(
        ["all", "--model-path", "models/Qwen3.5-9B", "--dry-run"]
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
    small = workload.resolve_qwen35_model_spec(
        tmp_path, paths["Qwen3.5-0.8B"]
    )
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
    assert model_tool._report_name(small, "reference") == (
        "qwen35-0.8b-reference.json"
    )
    assert model_tool._report_name(small, "candidate") == (
        "qwen35-0.8b-correctness.json"
    )
    assert model_tool._report_name(large, "reference") == "qwen35-9b-reference.json"
    assert model_tool._report_name(large, "candidate") == (
        "qwen35-9b-correctness.json"
    )
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


def test_effective_compilation_requires_scheduler_counters_and_code() -> None:
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
    small_pypto = lanes.server_kwargs("pypto", Path("Qwen3.5-0.8B"))
    small_matched = lanes.server_kwargs(
        "sglang-matched", Path("Qwen3.5-0.8B")
    )
    for config in (pypto, matched, optimized, qualified_optimized):
        assert "language_model_only" not in config
        assert config["json_model_override_args"] == (
            '{"language_model_only":true}'
        )
    assert pypto["cpu_offload_gb"] == matched["cpu_offload_gb"] == 2
    assert pypto["mem_fraction_static"] == matched["mem_fraction_static"] == 0.78
    assert small_pypto["cpu_offload_gb"] == small_matched["cpu_offload_gb"] == 0
    assert (
        small_pypto["mem_fraction_static"]
        == small_matched["mem_fraction_static"]
        == 0.78
    )
    assert optimized["cpu_offload_gb"] == 0
    assert "mem_fraction_static" not in optimized
    assert qualified_optimized["cpu_offload_gb"] == 2
    assert qualified_optimized["mem_fraction_static"] == 0.78
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
    assert [path.name for path in operator_tool._structure_sources(
        ROOT / "packages/pypto-kernels"
    )] == ["test_operators.py", "test_portable_release.py"]
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
    assert suites["paged-attention"]["expected_case_count"] == 12
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
    requested = lanes.server_kwargs(lane, Path("model"))
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
