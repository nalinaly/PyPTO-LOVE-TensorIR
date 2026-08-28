"""Performance-only A/B workload for the Qwen3.5 SwiGLU callsite."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time
import traceback

from .evidence_identity import collect_run_identity
from .lanes import prepare_worker_environment
from .performance_runtime import ResourceSampler, _resource_summary
from .workload import (
    SAMPLE_INTERVAL_MS,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    bootstrap_median_comparison_ci,
    distribution,
    fresh_start_methodology,
    fresh_start_summary,
    sha256_file,
)


OPERATOR_LANES = ("pypto", "sglang-matched")
ROOT = Path(__file__).resolve().parents[2]
OPERATOR_SCHEDULE = (
    "pypto",
    "sglang-matched",
    "sglang-matched",
    "pypto",
    "pypto",
    "sglang-matched",
    "sglang-matched",
    "pypto",
)
WARMUP_CALLS = 20
TIMED_BATCHES = 30
CALLS_PER_BATCH = 100
SWIGLU_CASES = (
    {
        "name": "decode-1x24576",
        "phase": "decode",
        "rows": 1,
        "intermediate_size": 12_288,
        "packed_width": 24_576,
    },
    {
        "name": "prefill-19x24576",
        "phase": "prefill",
        "rows": 19,
        "intermediate_size": 12_288,
        "packed_width": 24_576,
    },
)


def _candidate_provenance() -> dict[str, object]:
    from pypto_plugins.torch.inductor_swiglu import callable_cache_snapshot

    snapshot = callable_cache_snapshot()
    if len(snapshot) != len(SWIGLU_CASES) or any(
        not source.startswith("torch-inductor:") for _key, _kernel, source in snapshot
    ):
        raise ReleaseContractError(
            "operator A/B candidate did not populate the PyPTO Inductor cache"
        )
    return {
        "provider": "pypto.inductor",
        "compiled_callable_count": len(snapshot),
        "source_nodes": sorted({source for _key, _kernel, source in snapshot}),
    }


def _measure_case(torch, operator, case: dict[str, object]) -> dict[str, object]:
    rows = int(case["rows"])
    packed_width = int(case["packed_width"])
    torch.manual_seed(19 + rows)
    value = torch.randn((rows, packed_width), dtype=torch.bfloat16, device="cuda")

    torch.cuda.synchronize()
    trigger_start_ns = time.perf_counter_ns()
    output = operator.forward_cuda(value)
    torch.cuda.synchronize()
    first_trigger_ms = (time.perf_counter_ns() - trigger_start_ns) / 1e6

    for _ in range(WARMUP_CALLS):
        output = operator.forward_cuda(value)
    torch.cuda.synchronize()

    batch_ms = []
    for _ in range(TIMED_BATCHES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(CALLS_PER_BATCH):
            output = operator.forward_cuda(value)
        end.record()
        end.synchronize()
        batch_ms.append(float(start.elapsed_time(end)) / CALLS_PER_BATCH)
    del output
    return {
        **case,
        "dtype": "bfloat16",
        "first_compile_trigger_call_wall_ms": first_trigger_ms,
        "warmup_calls": WARMUP_CALLS,
        "timed_batches": TIMED_BATCHES,
        "calls_per_batch": CALLS_PER_BATCH,
        "latency_ms_per_call": distribution(batch_ms),
        "raw_batch_average_ms_per_call": batch_ms,
    }


def run(lane: str, model_path: Path, run_id: str, run_dir: Path) -> int:
    if lane not in OPERATOR_LANES:
        raise ReleaseContractError(f"unknown operator performance lane: {lane}")
    model_path = model_path.resolve(strict=True)
    prepare_worker_environment(lane)
    report_path = run_dir / f"qwen35-swiglu-performance-{lane}.json"
    resources_path = run_dir / f"qwen35-swiglu-resources-{lane}.json"
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-swiglu-operator-performance-only",
        "lane": lane,
        "run_id": run_id,
        "operator": "sglang.srt.layers.activation.SiluAndMul.forward_cuda",
        "shape_source": (
            "Qwen3.5-9B text_config.intermediate_size=12288 and frozen 19-token "
            "release prompt; packed gate/up width is 2*12288"
        ),
        "measurement_boundary": (
            "Warm latency is CUDA-event stream device time across the public SGLang "
            "operator call; host dispatch/allocation overhead is excluded unless it "
            "enqueues device work. The first compile-trigger call is synchronized wall "
            "time and includes compilation. No correctness oracle runs in this process."
        ),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "status": "starting",
    }
    sampler = ResourceSampler(os.getpgid(0))
    sampler.start()
    try:
        import torch
        from sglang.srt.layers.activation import SiluAndMul

        if not torch.cuda.is_available():
            raise ReleaseContractError("CUDA is unavailable")
        operator = SiluAndMul()
        report["cases"] = [
            _measure_case(torch, operator, dict(case)) for case in SWIGLU_CASES
        ]
        report["provider"] = (
            _candidate_provenance()
            if lane == "pypto"
            else {"provider": "sglang.stock.silu_and_mul"}
        )
        sampler.stop()
        report["identity_collection_boundary"] = (
            "Full model/package hashing runs after CUDA-event timing and after "
            "resource sampling stops."
        )
        identity_profile = "pypto" if lane == "pypto" else "baseline"
        report["evidence_identity"] = collect_run_identity(
            ROOT, identity_profile, model_path
        )
        report["status"] = "complete"
        return_code = 0
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    finally:
        sampler.stop()
        resource_payload = {
            "schema": SCHEMA_VERSION,
            "kind": "qwen35-swiglu-operator-resource-samples",
            "lane": lane,
            "run_id": run_id,
            "sample_interval_ms": SAMPLE_INTERVAL_MS,
            "nvml_error": sampler.nvml_error,
            "gpu_identity": sampler.identity,
            "samples": sampler.samples,
        }
        atomic_json(resources_path, resource_payload)
        report["resources"] = {
            "path": str(resources_path),
            "sha256": sha256_file(resources_path),
            "summary": _resource_summary(sampler.samples),
            "nvml_error": sampler.nvml_error,
            "gpu_identity": sampler.identity,
        }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "kind": report["kind"],
                "status": report["status"],
                "lane": lane,
                "run_id": run_id,
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return return_code


def summarize_fresh_starts(
    reports_by_lane: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    if set(reports_by_lane) != set(OPERATOR_LANES):
        raise ReleaseContractError("operator A/B summary requires candidate and stock")
    starts = {}
    summaries: dict[str, object] = {}
    gpu_identities = []
    for lane in OPERATOR_LANES:
        reports = reports_by_lane[lane]
        if len(reports) != 4:
            raise ReleaseContractError(f"operator lane {lane} requires four starts")
        case_starts: dict[str, list[float]] = {
            str(case["name"]): [] for case in SWIGLU_CASES
        }
        trigger_starts: dict[str, list[float]] = {
            str(case["name"]): [] for case in SWIGLU_CASES
        }
        for report in reports:
            if (
                report.get("status") != "complete"
                or report.get("kind") != "qwen35-swiglu-operator-performance-only"
                or report.get("lane") != lane
            ):
                raise ReleaseContractError(
                    f"operator report is not accepted for {lane}"
                )
            cases = report.get("cases")
            if type(cases) is not list or [item.get("name") for item in cases] != list(
                case_starts
            ):
                raise ReleaseContractError(f"operator cases drifted for {lane}")
            resources = report.get("resources")
            if (
                not isinstance(resources, dict)
                or resources.get("nvml_error") is not None
            ):
                raise ReleaseContractError("operator NVML sampling is incomplete")
            gpu_identity = resources.get("gpu_identity")
            if not isinstance(gpu_identity, dict):
                raise ReleaseContractError("operator GPU identity is absent")
            gpu_identities.append(gpu_identity)
            resource_summary = resources.get("summary")
            if (
                not isinstance(resource_summary, dict)
                or resource_summary.get("thermal_throttle_observed") is not False
            ):
                raise ReleaseContractError("operator run observed thermal throttling")
            if any(
                type(resource_summary.get(field)) is not int
                for field in (
                    "minimum_gpu_memory_free_bytes",
                    "minimum_mem_available_kib",
                    "peak_owned_pgid_rss_kib",
                )
            ):
                raise ReleaseContractError("operator resource telemetry is incomplete")
            if int(resource_summary["minimum_gpu_memory_free_bytes"]) < 4 * 1024**3:
                raise ReleaseContractError("operator run crossed the 4 GiB GPU floor")
            if int(resource_summary["minimum_mem_available_kib"]) < 12 * 1024**2:
                raise ReleaseContractError("operator run crossed the 12 GiB host floor")
            for case in cases:
                raw = case.get("raw_batch_average_ms_per_call")
                if type(raw) is not list or len(raw) != TIMED_BATCHES:
                    raise ReleaseContractError("operator timed batch count drifted")
                name = str(case["name"])
                case_starts[name].append(distribution(raw)["p50"])
                trigger_starts[name].append(
                    float(case["first_compile_trigger_call_wall_ms"])
                )
        starts[lane] = case_starts
        summaries[lane] = {
            "fresh_starts": len(reports),
            "cases": {
                name: {
                    "latency_ms_per_call": fresh_start_summary(
                        values, salt=f"operator:{lane}:{name}:latency"
                    ),
                    "first_compile_trigger_call_wall_ms": fresh_start_summary(
                        trigger_starts[name],
                        salt=f"operator:{lane}:{name}:first-trigger",
                    ),
                }
                for name, values in case_starts.items()
            },
        }
    if any(identity != gpu_identities[0] for identity in gpu_identities[1:]):
        raise ReleaseContractError("operator A/B runs used different GPUs or drivers")
    comparisons = {}
    for case in SWIGLU_CASES:
        name = str(case["name"])
        candidate = starts["pypto"][name]
        baseline = starts["sglang-matched"][name]
        candidate_median = float(
            summaries["pypto"]["cases"][name]["latency_ms_per_call"]["p50"]
        )
        baseline_median = float(
            summaries["sglang-matched"]["cases"][name]["latency_ms_per_call"]["p50"]
        )
        interval = bootstrap_median_comparison_ci(
            candidate,
            baseline,
            operation="ratio",
            salt=f"operator:pypto-vs-stock:{name}:latency",
        )
        comparisons[name] = {
            "pypto_latency_percent_of_stock": 100.0
            * candidate_median
            / baseline_median,
            "median_ratio_bootstrap_95ci_percent": {
                **interval,
                "lower": float(interval["lower"]) * 100.0,
                "upper": float(interval["upper"]) * 100.0,
            },
        }
    return {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-swiglu-operator-ab-performance-summary",
        "methodology": fresh_start_methodology(
            "CUDA-event timed batch averages within each operator case"
        ),
        "lanes": summaries,
        "gpu_identity": gpu_identities[0],
        "comparisons": comparisons,
        "correctness_evaluated": False,
        "status": "complete",
    }
