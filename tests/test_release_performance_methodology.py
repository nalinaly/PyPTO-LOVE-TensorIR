from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release import (  # noqa: E402
    lanes,
    operator_performance_runtime,
    performance_runtime,
    profile_runtime,
    workload,
)


def _load_tool(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(f"methodology_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


operator_performance_tool = _load_tool("run_operator_performance.py")


def _resolved_record(lane: str) -> dict[str, object]:
    requested = lanes.server_kwargs(lane, Path("model"))
    return {
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


def _request(value: float) -> dict[str, object]:
    return {
        "e2e_ms": value,
        "ttft_ms": value,
        "tpot_ms": value,
        "itl_ms": [value, value + 0.25],
        "output_tokens_per_second": 1_000.0 / value,
        "decode_tokens_per_second": 900.0 / value,
        "input_tokens_per_second": 800.0 / value,
        "total_tokens_per_second": 700.0 / value,
        "requests_per_second": 1.0 / value,
    }


def _performance_report(lane: str, start: int, value: float) -> dict[str, object]:
    return {
        "status": "complete",
        "kind": "qwen35-9b-performance-only",
        "lane": lane,
        "run_id": f"{lane}-{start}",
        "workload": workload.workload_record(),
        "requested_server_config": lanes.server_kwargs(lane, Path("model")),
        "resolved_backends": _resolved_record(lane),
        "compilation": {
            "requested": True,
            "backend_invocation_observed": True,
            "effective": True,
        },
        "resources": {
            "nvml_error": None,
            "gpu_identity": {
                "name": "NVIDIA GeForce RTX 5090",
                "uuid": "GPU-test",
                "driver": "test",
                "total_memory_bytes": 24 * 1024**3,
            },
            "summary": {
                "peak_gpu_memory_used_bytes": 20 * 1024**3,
                "minimum_gpu_memory_free_bytes": 4 * 1024**3,
                "minimum_mem_available_kib": 16 * 1024**2,
                "peak_owned_pgid_rss_kib": 8 * 1024**2,
                "thermal_throttle_observed": False,
            },
        },
        "cold_engine_start_ms": 100.0 + start,
        "first_compile_trigger_request_ms": 200.0 + start,
        "raw_requests": [_request(value + request / 100.0) for request in range(10)],
    }


def test_fresh_start_bootstrap_is_deterministic_and_retains_tails() -> None:
    first = workload.fresh_start_summary([1.0, 2.0, 9.0, 10.0], salt="test")
    second = workload.fresh_start_summary([1.0, 2.0, 9.0, 10.0], salt="test")
    assert first == second
    assert first["p50"] == pytest.approx(5.5)
    assert first["p90"] == 10.0
    assert first["p90_nearest_rank"] == 10.0
    assert first["p99"] == 10.0
    assert first["p99_nearest_rank"] == 10.0
    assert first["sample_unit"] == "fresh_process_start"
    assert first["median_bootstrap_95ci"]["resamples"] == 10_000


def test_performance_summary_uses_start_medians_and_matched_pytorch_sampler() -> None:
    reports = {
        lane: [
            _performance_report(lane, index, value)
            for index, value in enumerate((1.0, 2.0, 9.0, 10.0))
        ]
        for lane in workload.LANES
    }
    summary = performance_runtime.summarize_fresh_starts(reports)
    assert summary["status"] == "complete"
    assert summary["methodology"]["pooling_requests_across_starts"] is False
    e2e = summary["lanes"]["pypto"]["e2e_ms"]
    assert e2e["sample_count"] == 4
    assert len(summary["lanes"]["pypto"]["per_start"]) == 4
    comparability = summary["matched_comparability"]
    assert comparability["matched_claim_allowed"] is True
    assert not comparability["control_mismatches"]
    assert (
        reports["pypto"][0]["requested_server_config"]["sampling_backend"]
        == reports["sglang-matched"][0]["requested_server_config"]["sampling_backend"]
        == "pytorch"
    )


def test_matched_label_fails_closed_on_sampler_drift() -> None:
    candidate = lanes.server_kwargs("pypto", Path("model"))
    baseline = lanes.server_kwargs("sglang-matched", Path("model"))
    baseline["sampling_backend"] = "flashinfer"
    candidate_resolved = _resolved_record("pypto")
    baseline_resolved = _resolved_record("sglang-matched")
    baseline_resolved["sampling_backend"] = "flashinfer"
    record = lanes.matched_lane_comparability(
        candidate, baseline, candidate_resolved, baseline_resolved
    )
    assert record["status"] == "unmatched_controls"
    assert record["matched_claim_allowed"] is False
    assert [item["field"] for item in record["control_mismatches"]] == [
        "sampling_backend",
        "resolved.sampling_backend",
    ]


def test_graph_and_overlap_records_do_not_claim_runtime_execution() -> None:
    requested = lanes.server_kwargs("sglang-optimized", Path("model"))
    resolved = {
        "cuda_graph_enabled_by_server_args": True,
        "overlap_schedule_enabled_by_server_args": True,
    }
    features = lanes.execution_feature_record(requested, resolved)
    assert features["cuda_graph"] == {
        "requested": True,
        "enabled": True,
        "replay_runtime_observed": None,
        "evidence_boundary": (
            "requested/enabled are configuration facts; this record does not prove "
            "CUDA Graph replay"
        ),
    }
    assert features["overlap_schedule"]["runtime_overlap_observed"] is None
    graph = performance_runtime._graph_observation(
        {"internal_states": [{"memory_usage": {"graph": {"decode": 1.0}}}]}
    )
    assert graph["capture_memory_metadata_observed"] is True
    assert graph["replay_runtime_observed"] is None
    assert "capture_observed" not in graph


def test_inductor_hash_and_shared_linear_attribution_are_explicit() -> None:
    rules = json.loads(
        (ROOT / "benchmarks/release/logical_phases.json").read_text(encoding="utf-8")
    )
    phase = json.dumps(
        {
            "kind": profile_runtime.ARTIFACT_ANNOTATION_KIND,
            "schema_version": 1,
            "artifact": {
                "provider": "pypto.tensorir",
                "source_node": "torch-inductor:0123456789abcdef",
            },
        }
    )
    linear = json.dumps(
        {
            "kind": profile_runtime.ARTIFACT_ANNOTATION_KIND,
            "schema_version": 1,
            "artifact": {
                "provider": "pypto.tensorir",
                "source_node": "pypto_kernels.linear:linear",
            },
        }
    )
    window = {
        "user_annotations": {"7": phase, "8": linear},
        "events": [
            {"kind": "external_correlation", "correlation_id": 10, "external_id": 7},
            {"kind": "external_correlation", "correlation_id": 11, "external_id": 8},
            {
                "kind": "kernel",
                "correlation_id": 10,
                "name": "generated_a",
                "start_ns": 10,
                "end_ns": 20,
            },
            {
                "kind": "kernel",
                "correlation_id": 11,
                "name": "shared_linear",
                "start_ns": 30,
                "end_ns": 50,
            },
        ],
    }
    groups = profile_runtime.aggregate_windows([window], rules)["kernel_groups"]
    by_source = {group["source"]: group for group in groups}
    assert by_source["torch-inductor:0123456789abcdef"]["phase"] == "mlp_swiglu"
    assert (
        by_source["torch-inductor:0123456789abcdef"]["attribution"]
        == "artifact_source_identity"
    )
    assert by_source["pypto_kernels.linear:linear"]["phase"] == ("unattributed_compute")
    assert by_source["pypto_kernels.linear:linear"]["attribution"] == (
        "explicit_unattributed_shared_artifact"
    )


def _profile_report(lane: str, start: int, request_ns: float) -> dict[str, object]:
    request_aggregations = [
        {
            "compute_gpu_time_ns": request_ns + request * 100.0,
            "runtime_memcpy_gpu_time_ns": 10.0,
            "phase_totals": {
                "mlp_swiglu": {
                    "calls": 32,
                    "gpu_time_ns": request_ns + request * 100.0,
                }
            },
        }
        for request in range(5)
    ]
    return {
        "status": "complete",
        "kind": "qwen35-9b-logical-phase-profile",
        "lane": lane,
        "run_id": f"profile-{lane}-{start}",
        "workload": workload.workload_record(),
        "requested_server_config": lanes.server_kwargs(lane, Path("model")),
        "resolved_backends": _resolved_record(lane),
        "compilation": {"requested": True, "effective": True},
        "profile_requests": 5,
        "requests": [
            {
                "request_index": index,
                "trace_attempts": 1,
                "aggregation": aggregation,
            }
            for index, aggregation in enumerate(request_aggregations)
        ],
        "aggregation": {
            "compute_gpu_time_ns": sum(
                item["compute_gpu_time_ns"] for item in request_aggregations
            ),
            "runtime_memcpy_gpu_time_ns": 50.0,
            "phase_totals": {
                "mlp_swiglu": {
                    "calls": 160,
                    "gpu_time_ns": sum(
                        item["phase_totals"]["mlp_swiglu"]["gpu_time_ns"]
                        for item in request_aggregations
                    ),
                }
            },
        },
    }


def test_profile_reconciliation_uses_the_same_fresh_start_unit() -> None:
    base = {
        "pypto": 1_000_000.0,
        "sglang-matched": 500_000.0,
        "sglang-optimized": 400_000.0,
    }
    profiles = {
        lane: [
            _profile_report(lane, start, value + start * 1_000.0) for start in range(3)
        ]
        for lane, value in base.items()
    }
    result = profile_runtime.reconcile(profiles)
    assert result["status"] == "complete"
    assert result["methodology"]["experimental_unit"] == "fresh_process_start"
    compute = result["lane_summaries"]["pypto"]["compute_gpu_time_ns_per_request"]
    assert compute["sample_count"] == 3
    assert "p90_nearest_rank" in compute and "p99_nearest_rank" in compute
    matched = result["comparisons"]["sglang-matched"]
    assert matched["model_compute_gap_ms"] == pytest.approx(0.5)
    assert "model_compute_gap_bootstrap_95ci_ms" in matched
    assert result["matched_comparability"]["matched_claim_allowed"] is True


def test_profile_reconciliation_records_unmatched_controls_before_failing() -> None:
    profiles = {
        lane: [_profile_report(lane, start, 1_000_000.0) for start in range(3)]
        for lane in workload.LANES
    }
    for report in profiles["sglang-matched"]:
        report["requested_server_config"]["sampling_backend"] = "flashinfer"
        report["resolved_backends"]["sampling_backend"] = "flashinfer"
    result = profile_runtime.reconcile(profiles)
    assert result["status"] == "failed"
    assert result["comparisons"] == {}
    assert result["matched_comparability"]["matched_claim_allowed"] is False
    assert result["matched_comparability"]["control_mismatches"][0]["field"] == (
        "sampling_backend"
    )


def _operator_report(lane: str, start: int) -> dict[str, object]:
    return {
        "status": "complete",
        "kind": "qwen35-swiglu-operator-performance-only",
        "lane": lane,
        "run_id": f"operator-{lane}-{start}",
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
                "minimum_mem_available_kib": 16 * 1024**2,
                "peak_owned_pgid_rss_kib": 1024,
                "thermal_throttle_observed": False,
            },
        },
        "cases": [
            {
                **case,
                "first_compile_trigger_call_wall_ms": 100.0 + start,
                "raw_batch_average_ms_per_call": [float(start + 1) for _ in range(30)],
            }
            for case in operator_performance_runtime.SWIGLU_CASES
        ],
    }


def test_operator_ab_is_a_separate_performance_only_entrypoint() -> None:
    args = operator_performance_tool.parser().parse_args(["--matrix", "--dry-run"])
    assert args.matrix is True
    assert operator_performance_runtime.OPERATOR_SCHEDULE.count("pypto") == 4
    assert operator_performance_runtime.OPERATOR_SCHEDULE.count("sglang-matched") == 4
    reports = {
        lane: [_operator_report(lane, start) for start in range(4)]
        for lane in operator_performance_runtime.OPERATOR_LANES
    }
    summary = operator_performance_runtime.summarize_fresh_starts(reports)
    assert summary["status"] == "complete"
    assert summary["correctness_evaluated"] is False
    source = (ROOT / "benchmarks/release/operator_performance_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "correctness_runtime" not in source
    assert "torch.allclose" not in source
    assert "torch.equal" not in source
