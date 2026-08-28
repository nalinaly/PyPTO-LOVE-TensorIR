#!/usr/bin/env python3
"""Validate release evidence and render data-only Markdown fragments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.workload import (  # noqa: E402
    LANES,
    MEASURED_REQUESTS,
    PERFORMANCE_SCHEDULE,
    PROFILE_SCHEDULE,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    distribution,
    read_json,
    require_path_below_runs,
    sha256_file,
    workload_record,
)


def _selectors(values: list[str], label: str) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = defaultdict(list)
    for raw in values:
        lane, separator, path = raw.partition("=")
        if separator != "=" or lane not in LANES:
            raise ReleaseContractError(f"invalid {label} selector: {raw}")
        selected[lane].append(Path(path).resolve(strict=True))
    return dict(selected)


def _controller_evidence(report_path: Path, expected_profile: str) -> dict[str, object]:
    controller_path = report_path.parent / "process.json"
    initial_path = report_path.parent / "initial-audit.json"
    controller = read_json(controller_path)
    initial = read_json(initial_path)
    post = controller.get("post_audit")
    if (
        controller.get("mode") != "gpu-bounded"
        or controller.get("framework_profile") != expected_profile
        or controller.get("status") != "exited"
        or controller.get("return_code") != 0
        or controller.get("abort_reason") is not None
        or not isinstance(post, dict)
    ):
        raise ReleaseContractError(
            f"bounded GPU controller did not accept the run: {controller_path}"
        )
    policy = controller.get("policy") or {}
    if (
        policy.get("launch_admission_floor_kib") is not None
        or policy.get("host_abort_floor_kib") != 16 * 1024**2
        or policy.get("host_emergency_abort_floor_kib") != 15 * 1024**2
        or policy.get("gpu_free_floor_mib") != 4 * 1024
    ):
        raise ReleaseContractError(
            f"bounded resource policy drifted: {controller_path}"
        )
    for label, audit in (("initial", initial), ("post", post)):
        gpu = audit.get("gpu") or {}
        if (
            gpu.get("compute_capability") != "12.0"
            or audit.get("external_compute_pids")
            or audit.get("protected_compute_pids")
            or audit.get("protected_runtime_mapping_pids")
            or audit.get("unreadable_protected_maps")
        ):
            raise ReleaseContractError(
                f"{label} NVIDIA ownership/identity audit failed: {controller_path}"
            )
    return {
        "controller": str(controller_path),
        "controller_sha256": sha256_file(controller_path),
        "initial_audit": str(initial_path),
        "initial_audit_sha256": sha256_file(initial_path),
    }


def _load_performance(paths: dict[str, list[Path]]) -> dict[str, object]:
    if set(paths) != set(LANES) or any(len(items) != 4 for items in paths.values()):
        raise ReleaseContractError(
            "performance rendering requires four fresh starts for every lane"
        )
    result = {}
    for lane in LANES:
        reports = [read_json(path) for path in paths[lane]]
        requests = []
        cold = []
        compile_warmup = []
        peak_gpu = []
        minimum_host = []
        peak_rss = []
        resolved = []
        memory_qualifications = []
        inputs = []
        for path, report in zip(paths[lane], reports):
            if (
                report.get("status") != "complete"
                or report.get("kind") != "qwen35-9b-performance-only"
                or report.get("lane") != lane
                or report.get("workload") != workload_record()
            ):
                raise ReleaseContractError(
                    f"performance report is not accepted: {path}"
                )
            expected_profile = "pypto" if lane == "pypto" else "baseline"
            controller_evidence = _controller_evidence(path, expected_profile)
            raw = report.get("raw_requests")
            if type(raw) is not list or len(raw) != MEASURED_REQUESTS:
                raise ReleaseContractError(f"performance request count drifted: {path}")
            requests.extend(raw)
            cold.append(float(report["cold_engine_start_ms"]))
            compile_warmup.append(float(report["first_compile_trigger_request_ms"]))
            resources = report["resources"]
            if resources.get("nvml_error") is not None:
                raise ReleaseContractError(f"NVML sampling failed: {path}")
            summary = resources["summary"]
            if summary.get("thermal_throttle_observed") is not False:
                raise ReleaseContractError(f"thermal throttling observed: {path}")
            if int(summary.get("minimum_gpu_memory_free_bytes", 0)) < 4 * 1024**3:
                raise ReleaseContractError(
                    f"GPU free memory fell below the 4 GiB release floor: {path}"
                )
            if int(summary.get("minimum_mem_available_kib", 0)) < 16 * 1024**2:
                raise ReleaseContractError(
                    f"host MemAvailable fell below the 16 GiB release floor: {path}"
                )
            peak_gpu.append(int(summary["peak_gpu_memory_used_bytes"]))
            minimum_host.append(int(summary["minimum_mem_available_kib"]))
            peak_rss.append(int(summary["peak_owned_pgid_rss_kib"]))
            allocator = report.get("torch_allocator")
            if not isinstance(allocator, dict) or any(
                type(allocator.get(key)) is not int
                for key in (
                    "allocated_bytes",
                    "reserved_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            ):
                raise ReleaseContractError(
                    f"scheduler allocator telemetry is incomplete: {path}"
                )
            resolved.append(report["resolved_backends"])
            memory_qualifications.append(report["memory_qualification"])
            if lane == "sglang-optimized" and (
                report.get("compilation", {}).get("effective") is not True
                or report.get("cuda_graph", {}).get("capture_observed") is not True
            ):
                raise ReleaseContractError(
                    f"optimized compile/graph execution was not proven: {path}"
                )
            inputs.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    **controller_evidence,
                }
            )
        if any(item != resolved[0] for item in resolved[1:]):
            raise ReleaseContractError(f"resolved backend drift across {lane} starts")
        if any(item != memory_qualifications[0] for item in memory_qualifications[1:]):
            raise ReleaseContractError(
                f"memory qualification drift across {lane} starts"
            )
        result[lane] = {
            "fresh_starts": len(reports),
            "measured_requests": len(requests),
            "cold_engine_start_ms": distribution(cold),
            "first_compile_trigger_request_ms": distribution(compile_warmup),
            "e2e_ms": distribution(item["e2e_ms"] for item in requests),
            "ttft_ms": distribution(item["ttft_ms"] for item in requests),
            "tpot_ms": distribution(item["tpot_ms"] for item in requests),
            "itl_ms": distribution(
                value for item in requests for value in item["itl_ms"]
            ),
            "output_tokens_per_second": distribution(
                item["output_tokens_per_second"] for item in requests
            ),
            "decode_tokens_per_second": distribution(
                item["decode_tokens_per_second"] for item in requests
            ),
            "input_tokens_per_second": distribution(
                item["input_tokens_per_second"] for item in requests
            ),
            "total_tokens_per_second": distribution(
                item["total_tokens_per_second"] for item in requests
            ),
            "requests_per_second": distribution(
                item["requests_per_second"] for item in requests
            ),
            "resources": {
                "peak_gpu_memory_used_bytes": max(peak_gpu),
                "minimum_mem_available_kib": min(minimum_host),
                "peak_owned_pgid_rss_kib": max(peak_rss),
            },
            "resolved_backends": resolved[0],
            "memory_qualification": memory_qualifications[0],
            "inputs": inputs,
        }
    pypto_rate = result["pypto"]["output_tokens_per_second"]["p50"]
    ratios = {
        lane: 100.0 * pypto_rate / result[lane]["output_tokens_per_second"]["p50"]
        for lane in ("sglang-matched", "sglang-optimized")
    }
    return {"lanes": result, "pypto_percent_of_stock": ratios}


def _load_correctness(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 3:
        raise ReleaseContractError("correctness rendering requires three fresh starts")
    reports = [read_json(path) for path in paths]
    identities = set()
    sequences = set()
    coverage_calls = []
    inductor_calls = []
    handwritten_calls = []
    inputs = []
    for path, report in zip(paths, reports):
        if (
            report.get("status") != "complete"
            or report.get("all_passed") is not True
            or report.get("kind") != "qwen35-9b-multitoken-correctness"
            or report.get("workload") != workload_record()
        ):
            raise ReleaseContractError(f"correctness report is not accepted: {path}")
        controller_evidence = _controller_evidence(path, "pypto")
        requests = report.get("requests")
        if type(requests) is not list or len(requests) != MEASURED_REQUESTS:
            raise ReleaseContractError(f"correctness request count drifted: {path}")
        engine = report.get("engine") or {}
        if (
            engine.get("all_passed") is not True
            or engine.get("stable_output") is not True
            or len(engine.get("requests", [])) != MEASURED_REQUESTS
        ):
            raise ReleaseContractError(
                f"end-to-end SGLang Engine stability is not accepted: {path}"
            )
        identities.add(report["reference"]["identity"])
        for request in requests:
            if request.get("passed") is not True:
                raise ReleaseContractError(f"request failed in {path}")
            if request.get("compilation_execution", {}).get("effective") is not True:
                raise ReleaseContractError(
                    f"executed Inductor-to-PyPTO artifacts were not proven in {path}"
                )
            inductor_calls.append(
                int(request["compilation_execution"]["inductor_compute_calls"])
            )
            handwritten_calls.append(
                int(request["compilation_execution"]["handwritten_compute_calls"])
            )
            sequences.add(request["output_sequence_sha256"])
            coverage = request["coverage"]
            if (
                coverage.get("strict_policy_passed") is not True
                or coverage.get("covered_calls") != coverage.get("total_calls")
                or coverage.get("fallback_event_groups") != 0
                or coverage.get("violation_count") != 0
            ):
                raise ReleaseContractError(f"coverage is not strict in {path}")
            coverage_calls.append(int(coverage["total_calls"]))
        inputs.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                **controller_evidence,
            }
        )
    if len(identities) != 1 or len(sequences) != 1:
        raise ReleaseContractError("correctness starts did not share one stable result")
    return {
        "fresh_starts": len(reports),
        "accepted_requests": len(reports) * MEASURED_REQUESTS,
        "output_tokens_per_request": 64,
        "unique_output_sequences": 1,
        "reference_identity": next(iter(identities)),
        "coverage_calls": distribution(coverage_calls),
        "inductor_compute_calls": distribution(inductor_calls),
        "handwritten_compute_calls": distribution(handwritten_calls),
        "output_text": reports[0]["output_text"],
        "inputs": inputs,
    }


def _performance_markdown(performance: dict[str, object]) -> str:
    rows = [
        "| Lane | Fresh starts | Requests | TTFT p50 (ms) | E2E p50 (ms) | TPOT p50 (ms) | Input tok/s p50 | Decode tok/s p50 | Output tok/s p50 | PyPTO / stock | Peak GPU GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lanes = performance["lanes"]
    ratios = performance["pypto_percent_of_stock"]
    for lane in LANES:
        item = lanes[lane]
        ratio = "100.00%" if lane == "pypto" else f"{ratios[lane]:.2f}%"
        rows.append(
            "| {lane} | {starts} | {requests} | {ttft:.3f} | {e2e:.3f} | "
            "{tpot:.3f} | {input_rate:.3f} | {decode_rate:.3f} | {rate:.3f} | {ratio} | {gpu:.3f} |".format(
                lane=lane,
                starts=item["fresh_starts"],
                requests=item["measured_requests"],
                ttft=item["ttft_ms"]["p50"],
                e2e=item["e2e_ms"]["p50"],
                tpot=item["tpot_ms"]["p50"],
                input_rate=item["input_tokens_per_second"]["p50"],
                decode_rate=item["decode_tokens_per_second"]["p50"],
                rate=item["output_tokens_per_second"]["p50"],
                ratio=ratio,
                gpu=item["resources"]["peak_gpu_memory_used_bytes"] / (1024**3),
            )
        )
    return "\n".join(rows) + "\n"


def _correctness_markdown(correctness: dict[str, object]) -> str:
    return (
        "| Fresh starts | Requests/start | Generated tokens/request | Exact stable sequences | Strict coverage |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"| {correctness['fresh_starts']} | {MEASURED_REQUESTS} | 64 | "
        f"{correctness['unique_output_sequences']} | 100% |\n"
    )


def _profile_markdown(reconciliation: dict[str, object]) -> str:
    rows = [
        "| Baseline | Logical phase | PyPTO GPU ms/request | Baseline GPU ms/request | Gap ms/request |",
        "|---|---|---:|---:|---:|",
    ]
    for baseline, comparison in reconciliation["comparisons"].items():
        for phase in comparison["phases"]:
            rows.append(
                "| {baseline} | {phase} | {pypto:.6f} | {stock:.6f} | {gap:.6f} |".format(
                    baseline=baseline,
                    phase=phase["phase"],
                    pypto=phase["pypto_gpu_ms"],
                    stock=phase["baseline_gpu_ms"],
                    gap=phase["gap_ms"],
                )
            )
        rows.append(
            "| {baseline} | **reconciliation residual** |  |  | {gap:.9f} |".format(
                baseline=baseline,
                gap=comparison["phase_reconciliation_residual_ms"],
            )
        )
    return "\n".join(rows) + "\n"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    performance = value.add_mutually_exclusive_group(required=True)
    performance.add_argument("--performance", action="append")
    performance.add_argument("--performance-matrix", type=Path)
    correctness = value.add_mutually_exclusive_group(required=True)
    correctness.add_argument("--correctness-report", action="append", type=Path)
    correctness.add_argument("--correctness-summary", type=Path)
    profiles = value.add_mutually_exclusive_group(required=True)
    profiles.add_argument("--reconciliation", type=Path)
    profiles.add_argument("--profile-matrix", type=Path)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def _matrix_performance_paths(path: Path) -> dict[str, list[Path]]:
    summary = read_json(path.resolve(strict=True))
    if (
        summary.get("status") != "complete"
        or summary.get("kind") != "qwen35-9b-performance-matrix-control"
        or len(summary.get("runs", [])) != 12
        or [item.get("lane") for item in summary.get("runs", [])]
        != list(PERFORMANCE_SCHEDULE)
    ):
        raise ReleaseContractError("performance matrix is not complete")
    selected: dict[str, list[Path]] = defaultdict(list)
    for item in summary["runs"]:
        selected[str(item["lane"])].append(Path(item["report"]).resolve(strict=True))
    return dict(selected)


def _summary_correctness_paths(path: Path) -> list[Path]:
    summary = read_json(path.resolve(strict=True))
    if (
        summary.get("status") != "complete"
        or summary.get("kind") != "qwen35-9b-candidate-control"
        or len(summary.get("runs", [])) != 3
    ):
        raise ReleaseContractError("candidate correctness summary is not complete")
    return [Path(item["report"]).resolve(strict=True) for item in summary["runs"]]


def _matrix_reconciliation_path(path: Path) -> Path:
    summary = read_json(path.resolve(strict=True))
    if (
        summary.get("status") != "complete"
        or summary.get("kind") != "qwen35-9b-profile-matrix-control"
        or len(summary.get("runs", [])) != 9
        or [item.get("lane") for item in summary.get("runs", [])]
        != list(PROFILE_SCHEDULE)
    ):
        raise ReleaseContractError("profile matrix is not complete")
    return Path(summary["reconciliation"]).resolve(strict=True)


def main() -> int:
    args = parser().parse_args()
    output = require_path_below_runs(ROOT, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    performance_paths = (
        _matrix_performance_paths(args.performance_matrix)
        if args.performance_matrix is not None
        else _selectors(args.performance, "performance")
    )
    performance = _load_performance(performance_paths)
    correctness_paths = (
        _summary_correctness_paths(args.correctness_summary)
        if args.correctness_summary is not None
        else [path.resolve(strict=True) for path in args.correctness_report]
    )
    correctness = _load_correctness(correctness_paths)
    reconciliation_path = (
        _matrix_reconciliation_path(args.profile_matrix)
        if args.profile_matrix is not None
        else args.reconciliation.resolve(strict=True)
    )
    reconciliation = read_json(reconciliation_path)
    if (
        reconciliation.get("status") != "complete"
        or reconciliation.get("workload") != workload_record()
    ):
        raise ReleaseContractError("profile reconciliation is not accepted")
    summary = {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-9b-release-results",
        "workload": workload_record(),
        "correctness": correctness,
        "performance": performance,
        "profile_reconciliation": {
            "path": str(reconciliation_path),
            "sha256": sha256_file(reconciliation_path),
            "comparisons": reconciliation["comparisons"],
        },
        "status": "complete",
    }
    summary_path = output / "release-summary.json"
    atomic_json(summary_path, summary)
    (output / "performance-table.md").write_text(
        _performance_markdown(performance), encoding="utf-8"
    )
    (output / "correctness-table.md").write_text(
        _correctness_markdown(correctness), encoding="utf-8"
    )
    (output / "profile-table.md").write_text(
        _profile_markdown(reconciliation), encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "complete", "summary": str(summary_path)},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
