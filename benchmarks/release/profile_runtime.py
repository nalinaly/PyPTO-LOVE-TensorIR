"""CUPTI/NVTX collection and logical-phase aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import traceback

from .correctness_runtime import _generate, _load_runner, _shutdown_runner
from .lanes import memory_qualification, prepare_worker_environment
from .workload import (
    OUTPUT_TOKENS,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    distribution,
    sha256_file,
    workload_record,
)


PHASE_ANNOTATION_KIND = "pypto.release-phase.v1"
ARTIFACT_ANNOTATION_KIND = "pypto.artifact-launch.v1"
PROFILE_REQUESTS = 5
MAX_TRACE_ATTEMPTS = 10


def _load_rules(root: Path) -> dict[str, object]:
    path = root / "benchmarks/release/logical_phases.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict or value.get("schema") != SCHEMA_VERSION:
        raise ReleaseContractError("logical phase rules have an unknown schema")
    return value


def _phase_for(value: str, rules: list[list[str]]) -> str | None:
    for pattern, phase in rules:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return phase
    return None


def _install_phase_hooks(model: object, monitor: object, rules: dict[str, object]):
    import torch

    handles = []
    stacks: dict[str, list[int]] = defaultdict(list)
    module_rules = rules["module_name_rules"]
    for name, module in model.named_modules():
        phase = _phase_for(name, module_rules)
        if phase is None:
            continue

        def before(_module, _inputs, *, module_name=name, logical_phase=phase):
            payload = json.dumps(
                {
                    "kind": PHASE_ANNOTATION_KIND,
                    "schema_version": SCHEMA_VERSION,
                    "phase": logical_phase,
                    "module": module_name,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            torch.cuda.nvtx.range_push(
                f"pypto-release-phase:{logical_phase}:{module_name}"
            )
            external_id = monitor.push_user_annotation(payload)
            if type(external_id) is not int or external_id <= 0:
                torch.cuda.nvtx.range_pop()
                raise ReleaseContractError("CUPTI rejected a release phase annotation")
            stacks[module_name].append(external_id)

        def after(
            _module,
            _inputs,
            output,
            *,
            module_name=name,
        ):
            expected = stacks[module_name].pop()
            observed = monitor.pop_user_annotation()
            torch.cuda.nvtx.range_pop()
            if observed != expected:
                raise ReleaseContractError(
                    f"phase annotation stack mismatch for {module_name}"
                )
            return output

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    if not handles:
        raise ReleaseContractError("no Qwen modules matched the logical phase rules")
    return handles


def _annotation(value: object) -> dict[str, object] | None:
    if not isinstance(value, str):
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if type(payload) is dict else None


def aggregate_windows(
    windows: list[dict[str, object]], rules: dict[str, object]
) -> dict[str, object]:
    groups: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    stage_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for window_index, window in enumerate(windows):
        stage = "prefill" if window_index % OUTPUT_TOKENS == 0 else "decode"
        annotations = {
            int(key): _annotation(value)
            for key, value in window.get("user_annotations", {}).items()
        }
        correlations = {}
        for event in window.get("events", []):
            if event.get("kind") == "external_correlation":
                correlations[int(event["correlation_id"])] = int(event["external_id"])
        for event in window.get("events", []):
            kind = event.get("kind")
            if kind not in {"kernel", "gpu_memcpy", "gpu_memset"}:
                continue
            start_ns = event.get("start_ns")
            end_ns = event.get("end_ns")
            if (
                type(start_ns) is not int
                or type(end_ns) is not int
                or end_ns < start_ns
            ):
                raise ReleaseContractError("CUPTI activity has invalid timestamps")
            duration = end_ns - start_ns
            if kind != "kernel":
                phase = "runtime_memcpy"
                provider = "cuda.runtime"
                source = kind
                kernel = str(event.get("name") or kind)
            else:
                kernel = str(event.get("name") or "")
                correlation = event.get("correlation_id")
                external_id = correlations.get(correlation)
                payload = annotations.get(external_id)
                phase = None
                provider = "stock.cuda"
                source = "unannotated"
                if payload and payload.get("kind") == PHASE_ANNOTATION_KIND:
                    phase = str(payload.get("phase"))
                    source = str(payload.get("module"))
                elif payload and payload.get("kind") == ARTIFACT_ANNOTATION_KIND:
                    artifact = payload.get("artifact")
                    if isinstance(artifact, dict):
                        provider = str(artifact.get("provider"))
                        source = str(artifact.get("source_node"))
                        phase = _phase_for(source, rules["source_node_rules"])
                if phase is None:
                    phase = _phase_for(kernel, rules["kernel_name_rules"])
                if phase is None:
                    phase = "unattributed_compute"
            key = (stage, phase, provider, source, kernel)
            groups[key][0] += 1
            groups[key][1] += duration
            stage_totals[stage][0] += 1
            stage_totals[stage][1] += duration

    phase_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (_stage, phase, _provider, _source, _kernel), totals in groups.items():
        phase_totals[phase][0] += totals[0]
        phase_totals[phase][1] += totals[1]
    compute_ns = sum(
        totals[1]
        for (stage, phase, provider, source, kernel), totals in groups.items()
        if phase != "runtime_memcpy"
    )
    runtime_ns = phase_totals["runtime_memcpy"][1]
    return {
        "forward_count": len(windows),
        "compute_gpu_time_ns": compute_ns,
        "runtime_memcpy_gpu_time_ns": runtime_ns,
        "stage_totals": {
            stage: {"calls": totals[0], "gpu_time_ns": totals[1]}
            for stage, totals in sorted(stage_totals.items())
        },
        "phase_totals": {
            phase: {"calls": totals[0], "gpu_time_ns": totals[1]}
            for phase, totals in sorted(phase_totals.items())
        },
        "kernel_groups": [
            {
                "stage": key[0],
                "phase": key[1],
                "provider": key[2],
                "source": key[3],
                "kernel": key[4],
                "calls": totals[0],
                "gpu_time_ns": totals[1],
            }
            for key, totals in sorted(
                groups.items(), key=lambda item: item[1][1], reverse=True
            )
        ],
    }


def run(
    lane: str,
    model_path: Path,
    run_id: str,
    run_dir: Path,
    root: Path,
    optimized_memory_mode: str = "zero-offload",
) -> int:
    prepare_worker_environment(lane)
    model_path = model_path.resolve(strict=True)
    report_path = run_dir / f"qwen35-9b-profile-{lane}.json"
    raw_path = run_dir / f"qwen35-9b-cupti-{lane}.json"
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-9b-logical-phase-profile",
        "lane": lane,
        "run_id": run_id,
        "workload": workload_record(),
        "entrypoint": "sglang.benchmark.one_batch ModelRunner with CUPTI/NVTX",
        "memory_qualification": memory_qualification(lane, optimized_memory_mode),
        "status": "starting",
    }
    runner = None
    monitor = None
    monitor_api = None
    handles = []
    try:
        import torch
        from torch.profiler import _cupti_monitor as monitor_api

        if torch.cuda.is_initialized():
            raise ReleaseContractError("CUPTI must start before CUDA initialization")
        monitor = monitor_api.start_collection(run_dir / "cupti-monitor")
        torch, one_batch, runner, requested, resolved = _load_runner(
            lane, model_path, optimized_memory_mode
        )
        report["requested_server_config"] = requested
        report["resolved_backends"] = resolved
        warm_ids, _warm_logits, _warm_windows = _generate(torch, one_batch, runner)
        if len(warm_ids) != OUTPUT_TOKENS:
            raise ReleaseContractError("profile warmup did not complete")
        rules = _load_rules(root)
        handles = _install_phase_hooks(runner.torch_runner.model, monitor, rules)
        requests = []
        all_windows = []
        for request_index in range(PROFILE_REQUESTS):
            accepted = None
            for attempt in range(1, MAX_TRACE_ATTEMPTS + 1):
                output_ids, _profile_logits, windows = _generate(
                    torch, one_batch, runner, monitor
                )
                torch.cuda.synchronize()
                if (
                    len(output_ids) == OUTPUT_TOKENS
                    and len(windows) == OUTPUT_TOKENS
                    and all(
                        any(event.get("kind") == "kernel" for event in window["events"])
                        for window in windows
                    )
                ):
                    accepted = (windows, attempt)
                    break
            if accepted is None:
                raise ReleaseContractError(
                    "profile did not capture all 64 nonempty model-forward windows"
                )
            windows, attempt = accepted
            all_windows.extend(windows)
            requests.append(
                {
                    "request_index": request_index,
                    "trace_attempts": attempt,
                    "aggregation": aggregate_windows(windows, rules),
                }
            )
        stats = monitor_api.stop_collection()
        monitor = None
        if int(stats.get("dropped_records", 0)) != 0:
            raise ReleaseContractError("CUPTI dropped profile activity records")
        atomic_json(
            raw_path,
            {
                "schema": SCHEMA_VERSION,
                "stats": stats,
                "profile_requests": PROFILE_REQUESTS,
                "windows": all_windows,
            },
        )
        aggregation = aggregate_windows(all_windows, rules)
        from sglang.srt.compilation.compilation_counter import compilation_counter

        counter = asdict(compilation_counter)
        graph_memory = {
            str(key): float(value)
            for key, value in getattr(
                runner.torch_runner, "graph_memory_usage", {}
            ).items()
        }
        compilation = {
            "requested": bool(requested.get("enable_torch_compile")),
            "sglang_counter": counter,
            "backend_invocation_observed": bool(
                counter.get("num_graphs_seen", 0) > 0
                and counter.get("num_inductor_compiles", 0) > 0
            ),
            "graph_capture_observed": any(value > 0 for value in graph_memory.values()),
            "graph_memory_gb": graph_memory,
        }
        inductor_calls = sum(
            int(group["calls"])
            for group in aggregation["kernel_groups"]
            if str(group["source"]).startswith("torch-inductor:")
        )
        compilation["pypto_inductor_cupti_calls"] = inductor_calls
        compilation["pypto_inductor_expected_calls"] = (
            PROFILE_REQUESTS * OUTPUT_TOKENS * 32
        )
        compilation["effective"] = bool(
            compilation["backend_invocation_observed"]
            if lane != "pypto"
            else inductor_calls == PROFILE_REQUESTS * OUTPUT_TOKENS * 32
        )
        if lane in {"pypto", "sglang-optimized"} and not compilation["effective"]:
            raise ReleaseContractError(
                f"{lane} requested compilation but execution was not observed"
            )
        if lane == "sglang-optimized" and not compilation["graph_capture_observed"]:
            raise ReleaseContractError(
                "optimized profile did not observe CUDA Graph capture"
            )
        report.update(
            {
                "status": "complete",
                "profiler_perturbs_latency": True,
                "annotations": ["CUPTI external correlation", "NVTX"],
                "profile_requests": PROFILE_REQUESTS,
                "requests": requests,
                "raw_trace": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
                "collector_stats": stats,
                "aggregation": aggregation,
                "compilation": compilation,
                "limitations": [
                    "Latency claims must come from run_performance_regression.py.",
                    "A PyPTO artifact annotation is more specific than its enclosing module annotation; shared linear artifacts remain unattributed unless their source identity is phase-specific.",
                    "Kernel-name fallback is descriptive and is never used as PyPTO coverage evidence.",
                ],
            }
        )
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
        for handle in reversed(handles):
            handle.remove()
        if monitor is not None and monitor_api is not None:
            try:
                report["collector_stats"] = monitor_api.stop_collection()
            except BaseException as error:
                report["collector_stop_error"] = f"{type(error).__name__}: {error}"
        if runner is not None:
            _shutdown_runner()
        atomic_json(report_path, report)
        print(
            json.dumps(
                {
                    "kind": report["kind"],
                    "status": report["status"],
                    "lane": lane,
                    "run_id": run_id,
                    "report": str(report_path),
                    "compute_gpu_time_ns": (
                        report.get("aggregation", {}).get("compute_gpu_time_ns")
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return return_code


def reconcile(
    profiles: dict[str, object],
    performance: dict[str, object] | None = None,
) -> dict[str, object]:
    required = {"pypto", "sglang-matched", "sglang-optimized"}
    if set(profiles) != required:
        raise ReleaseContractError(
            "reconciliation requires exactly three profile lanes"
        )
    averaged = {}
    input_counts = {}
    for lane, raw_reports in profiles.items():
        reports = raw_reports if isinstance(raw_reports, list) else [raw_reports]
        if not reports:
            raise ReleaseContractError(f"profile list is empty for lane {lane}")
        phase_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        compute_ns = 0.0
        memcpy_ns = 0.0
        profile_requests = 0
        for report in reports:
            if report.get("status") != "complete" or report.get("lane") != lane:
                raise ReleaseContractError(f"profile is not accepted for lane {lane}")
            if report.get("workload") != workload_record():
                raise ReleaseContractError(f"profile workload drifted for lane {lane}")
            count = int(report.get("profile_requests", 0))
            if count <= 0:
                raise ReleaseContractError(
                    f"profile request count is invalid for {lane}"
                )
            aggregation = report["aggregation"]
            compute_ns += float(aggregation["compute_gpu_time_ns"])
            memcpy_ns += float(aggregation["runtime_memcpy_gpu_time_ns"])
            profile_requests += count
            for phase, totals in aggregation["phase_totals"].items():
                phase_totals[phase][0] += float(totals["calls"])
                phase_totals[phase][1] += float(totals["gpu_time_ns"])
        averaged[lane] = {
            "compute_gpu_time_ns": compute_ns / profile_requests,
            "runtime_memcpy_gpu_time_ns": memcpy_ns / profile_requests,
            "phase_totals": {
                phase: {
                    "calls": totals[0] / profile_requests,
                    "gpu_time_ns": totals[1] / profile_requests,
                }
                for phase, totals in phase_totals.items()
            },
        }
        input_counts[lane] = {
            "fresh_starts": len(reports),
            "profile_requests": profile_requests,
        }
    candidate = averaged["pypto"]
    comparisons = {}
    for baseline_lane in ("sglang-matched", "sglang-optimized"):
        baseline = averaged[baseline_lane]
        phase_names = sorted(
            set(candidate["phase_totals"]) | set(baseline["phase_totals"])
        )
        phase_names = [name for name in phase_names if name != "runtime_memcpy"]
        phase_gaps = []
        for phase in phase_names:
            candidate_ns = float(
                candidate["phase_totals"].get(phase, {}).get("gpu_time_ns", 0)
            )
            baseline_ns = float(
                baseline["phase_totals"].get(phase, {}).get("gpu_time_ns", 0)
            )
            phase_gaps.append(
                {
                    "phase": phase,
                    "pypto_gpu_ms": candidate_ns / 1e6,
                    "baseline_gpu_ms": baseline_ns / 1e6,
                    "gap_ms": (candidate_ns - baseline_ns) / 1e6,
                }
            )
        candidate_total = float(candidate["compute_gpu_time_ns"])
        baseline_total = float(baseline["compute_gpu_time_ns"])
        total_gap_ms = (candidate_total - baseline_total) / 1e6
        explained_ms = sum(item["gap_ms"] for item in phase_gaps)
        comparison: dict[str, object] = {
            "baseline_lane": baseline_lane,
            "model_compute_gap_ms": total_gap_ms,
            "phase_gap_sum_ms": explained_ms,
            "phase_reconciliation_residual_ms": total_gap_ms - explained_ms,
            "phases": phase_gaps,
        }
        if performance is not None:
            for lane in ("pypto", baseline_lane):
                raw_measured = performance.get(lane)
                measured_reports = (
                    raw_measured if isinstance(raw_measured, list) else [raw_measured]
                )
                if not measured_reports or any(
                    measured is None or measured.get("status") != "complete"
                    for measured in measured_reports
                ):
                    raise ReleaseContractError(f"performance report missing for {lane}")
                if any(
                    measured.get("workload") != workload_record()
                    for measured in measured_reports
                ):
                    raise ReleaseContractError(
                        f"performance workload drifted for {lane}"
                    )

            def pooled_e2e(raw):
                reports = raw if isinstance(raw, list) else [raw]
                values = [
                    float(request["e2e_ms"])
                    for measured in reports
                    for request in measured["raw_requests"]
                ]
                return distribution(values)["p50"]

            candidate_e2e = pooled_e2e(performance["pypto"])
            baseline_e2e = pooled_e2e(performance[baseline_lane])
            e2e_gap = candidate_e2e - baseline_e2e
            comparison["median_e2e_gap_ms"] = e2e_gap
            comparison["host_scheduler_memcpy_graph_gap_ms"] = e2e_gap - total_gap_ms
        comparisons[baseline_lane] = comparison
    return {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-9b-profile-gap-reconciliation",
        "workload": workload_record(),
        "profile_inputs": input_counts,
        "comparisons": comparisons,
        "status": "complete",
    }
