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
    lanes,
    performance_runtime,
    profile_runtime,
    workload,
)


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
    assert pypto["cpu_offload_gb"] == matched["cpu_offload_gb"] == 2
    assert pypto["mem_fraction_static"] == matched["mem_fraction_static"] == 0.78
    assert optimized["cpu_offload_gb"] == 0
    assert "mem_fraction_static" not in optimized
    assert qualified_optimized["cpu_offload_gb"] == 2
    assert qualified_optimized["mem_fraction_static"] == 0.78
    for config in (matched, optimized):
        assert config["linear_attn_backend"] == "flashinfer"
        assert config["linear_attn_decode_backend"] == "flashinfer"
        assert config["linear_attn_prefill_backend"] == "flashinfer"
        assert config["mamba_ssm_dtype"] == "bfloat16"
    assert matched["disable_cuda_graph"] is True
    assert matched["disable_overlap_schedule"] is True
    assert "disable_cuda_graph" not in optimized
    assert "disable_overlap_schedule" not in optimized


def test_requested_compile_flag_is_not_reported_as_effective() -> None:
    fake = SimpleNamespace(
        get_attention_backends=lambda: ("flashinfer", "flashinfer"),
        linear_attn_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
        linear_attn_decode_backend="flashinfer",
        mamba_backend="triton",
        mamba_ssm_dtype="bfloat16",
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
        "cuda_graph_stateful_sm120.py",
    ]
    assert all("success_field" in item for item in suites)


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
    return {
        "status": "complete",
        "lane": lane,
        "workload": workload.workload_record(),
        "profile_requests": 5,
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
