"""CUPTI/NVTX collection and logical-phase aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import traceback

from .correctness_runtime import _generate, _load_runner, _shutdown_runner
from .cupti_overlay import activate_overlay
from .evidence_identity import collect_run_identity
from .lanes import (
    execution_feature_record,
    matched_lane_comparability,
    memory_qualification,
    prepare_worker_environment,
)
from .workload import (
    OUTPUT_TOKENS,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    bootstrap_median_comparison_ci,
    distribution,
    fresh_start_methodology,
    fresh_start_summary,
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
    groups: dict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(
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
                attribution = "runtime_activity_kind"
            else:
                kernel = str(event.get("name") or "")
                correlation = event.get("correlation_id")
                external_id = correlations.get(correlation)
                payload = annotations.get(external_id)
                phase = None
                provider = "stock.cuda"
                source = "unannotated"
                attribution = "none"
                if payload and payload.get("kind") == PHASE_ANNOTATION_KIND:
                    phase = str(payload.get("phase"))
                    source = str(payload.get("module"))
                    attribution = "callsite_external_correlation"
                elif payload and payload.get("kind") == ARTIFACT_ANNOTATION_KIND:
                    artifact = payload.get("artifact")
                    if isinstance(artifact, dict):
                        provider = str(artifact.get("provider"))
                        source = str(artifact.get("source_node"))
                        phase = _phase_for(source, rules["source_node_rules"])
                        attribution = (
                            "explicit_unattributed_shared_artifact"
                            if phase == "unattributed_compute"
                            else "artifact_source_identity"
                        )
                if phase is None:
                    phase = _phase_for(kernel, rules["kernel_name_rules"])
                    if phase is not None:
                        attribution = "kernel_name_heuristic"
                if phase is None:
                    phase = "unattributed_compute"
                    attribution = "unattributed"
            key = (stage, phase, provider, source, kernel, attribution)
            groups[key][0] += 1
            groups[key][1] += duration
            stage_totals[stage][0] += 1
            stage_totals[stage][1] += duration

    phase_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (
        _stage,
        phase,
        _provider,
        _source,
        _kernel,
        _attribution,
    ), totals in groups.items():
        phase_totals[phase][0] += totals[0]
        phase_totals[phase][1] += totals[1]
    compute_ns = sum(
        totals[1]
        for (
            stage,
            phase,
            provider,
            source,
            kernel,
            attribution,
        ), totals in groups.items()
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
                "attribution": key[5],
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
        report["cupti_overlay"] = activate_overlay()
        import torch
        from torch.profiler import _cupti_monitor as monitor_api

        if torch.cuda.is_initialized():
            raise ReleaseContractError("CUPTI must start before CUDA initialization")
        monitor = monitor_api.start_collection(run_dir / "cupti-monitor")
        (
            torch,
            one_batch,
            runner,
            requested,
            resolved,
            compatibility,
            workload,
            workload_resolution,
        ) = _load_runner(
            lane, model_path, optimized_memory_mode
        )
        report["requested_server_config"] = requested
        report["resolved_backends"] = resolved
        report["shared_runtime_compatibility"] = compatibility
        report["workload"] = workload
        report["workload_resolution"] = workload_resolution
        prompt_token_ids = workload["prompt_token_ids"]
        report["execution_features"] = execution_feature_record(requested, resolved)
        warm_ids, _warm_logits, _warm_windows = _generate(
            torch, one_batch, runner, prompt_token_ids=prompt_token_ids
        )
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
                    torch,
                    one_batch,
                    runner,
                    monitor,
                    prompt_token_ids=prompt_token_ids,
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
        }
        report["execution_features"]["cuda_graph"].update(
            {
                "capture_memory_metadata_gb": graph_memory,
                "capture_memory_metadata_observed": any(
                    value > 0 for value in graph_memory.values()
                ),
                "evidence_boundary": (
                    "Nonzero graph memory is capture metadata, not proof that "
                    "profiled requests replayed a CUDA Graph."
                ),
            }
        )
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
        if compilation["requested"] and not compilation["effective"]:
            raise ReleaseContractError(
                f"{lane} requested compilation but execution was not observed"
            )
        if lane == "sglang-optimized" and not all(
            report["execution_features"][feature]["requested"]
            and report["execution_features"][feature]["enabled"]
            for feature in ("cuda_graph", "overlap_schedule")
        ):
            raise ReleaseContractError(
                "optimized profile did not keep CUDA Graph and overlap configured"
            )
        identity_profile = "pypto" if lane == "pypto" else "baseline"
        evidence_identity = collect_run_identity(root, identity_profile, model_path)
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
                "evidence_identity": evidence_identity,
                "limitations": [
                    "Latency claims must come from run_performance_regression.py.",
                    "Shared linear artifacts are explicitly unattributed when CUPTI exposes only their artifact identity and no callsite correlation.",
                    "Kernel-name fallback is descriptive and is never used as PyPTO coverage evidence.",
                    "CUDA Graph replay and runtime overlap are not inferred from configuration or graph-memory metadata.",
                    "GPU time is the sum of CUPTI activity durations; overlapping kernels can make it exceed elapsed critical-path time.",
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


def _profile_start_estimate(report: dict[str, object]) -> dict[str, object]:
    requests = report.get("requests")
    if type(requests) is not list or len(requests) != PROFILE_REQUESTS:
        raise ReleaseContractError("profile report lacks five request aggregations")
    aggregations = []
    for request in requests:
        if not isinstance(request, dict) or not isinstance(
            request.get("aggregation"), dict
        ):
            raise ReleaseContractError("profile request aggregation is malformed")
        aggregations.append(request["aggregation"])
    aggregate = report.get("aggregation")
    if not isinstance(aggregate, dict):
        raise ReleaseContractError("profile run aggregation is absent")
    compute_total = sum(float(item["compute_gpu_time_ns"]) for item in aggregations)
    memcpy_total = sum(
        float(item["runtime_memcpy_gpu_time_ns"]) for item in aggregations
    )
    if compute_total != float(
        aggregate["compute_gpu_time_ns"]
    ) or memcpy_total != float(aggregate["runtime_memcpy_gpu_time_ns"]):
        raise ReleaseContractError("profile request totals do not reconcile to the run")
    phase_names = sorted(
        {
            *(str(phase) for item in aggregations for phase in item["phase_totals"]),
            *(str(phase) for phase in aggregate["phase_totals"]),
        }
    )
    for phase in phase_names:
        calls = sum(
            float(item["phase_totals"].get(phase, {}).get("calls", 0))
            for item in aggregations
        )
        gpu_time_ns = sum(
            float(item["phase_totals"].get(phase, {}).get("gpu_time_ns", 0))
            for item in aggregations
        )
        aggregate_phase = aggregate["phase_totals"].get(phase, {})
        if calls != float(aggregate_phase.get("calls", 0)) or gpu_time_ns != float(
            aggregate_phase.get("gpu_time_ns", 0)
        ):
            raise ReleaseContractError(
                f"profile phase {phase!r} does not reconcile to the run"
            )
    return {
        "compute_gpu_time_ns": distribution(
            float(item["compute_gpu_time_ns"]) for item in aggregations
        )["p50"],
        "runtime_memcpy_gpu_time_ns": distribution(
            float(item["runtime_memcpy_gpu_time_ns"]) for item in aggregations
        )["p50"],
        "phase_totals": {
            phase: {
                "calls": distribution(
                    float(item["phase_totals"].get(phase, {}).get("calls", 0))
                    for item in aggregations
                )["p50"],
                "gpu_time_ns": distribution(
                    float(item["phase_totals"].get(phase, {}).get("gpu_time_ns", 0))
                    for item in aggregations
                )["p50"],
            }
            for phase in phase_names
        },
    }


def reconcile(
    profiles: dict[str, object],
    performance: dict[str, object] | None = None,
) -> dict[str, object]:
    required = {"pypto", "sglang-matched", "sglang-optimized"}
    if set(profiles) != required:
        raise ReleaseContractError(
            "reconciliation requires exactly three profile lanes"
        )
    lane_starts: dict[str, list[dict[str, object]]] = {}
    lane_summaries: dict[str, dict[str, object]] = {}
    input_counts = {}
    requested_configs: dict[str, dict[str, object]] = {}
    resolved_configs: dict[str, dict[str, object]] = {}
    for lane, raw_reports in profiles.items():
        reports = raw_reports if isinstance(raw_reports, list) else [raw_reports]
        if len(reports) != 3:
            raise ReleaseContractError(
                f"profile reconciliation requires three fresh starts for {lane}"
            )
        for report in reports:
            if report.get("status") != "complete" or report.get("lane") != lane:
                raise ReleaseContractError(f"profile is not accepted for lane {lane}")
            if report.get("workload") != workload_record():
                raise ReleaseContractError(f"profile workload drifted for lane {lane}")
            if int(report.get("profile_requests", 0)) != PROFILE_REQUESTS:
                raise ReleaseContractError(
                    f"profile request count is invalid for {lane}"
                )
            compilation = report.get("compilation")
            if (
                not isinstance(compilation, dict)
                or compilation.get("requested") is not True
                or compilation.get("effective") is not True
            ):
                raise ReleaseContractError(
                    f"profile did not prove compiled execution for {lane}"
                )
        configs = [report.get("requested_server_config") for report in reports]
        if any(type(config) is not dict for config in configs):
            raise ReleaseContractError(f"profile configuration is absent for {lane}")
        if any(config != configs[0] for config in configs[1:]):
            raise ReleaseContractError(
                f"profile configuration drifted across {lane} starts"
            )
        requested_configs[lane] = configs[0]
        resolved = [report.get("resolved_backends") for report in reports]
        if any(type(config) is not dict for config in resolved):
            raise ReleaseContractError(
                f"profile resolved configuration is absent for {lane}"
            )
        if any(config != resolved[0] for config in resolved[1:]):
            raise ReleaseContractError(
                f"profile resolved configuration drifted across {lane} starts"
            )
        resolved_configs[lane] = resolved[0]
        starts = [_profile_start_estimate(report) for report in reports]
        lane_starts[lane] = starts
        phase_names = sorted(
            {phase for start in starts for phase in start["phase_totals"]}
        )
        lane_summaries[lane] = {
            "compute_gpu_time_ns_per_request": fresh_start_summary(
                (float(start["compute_gpu_time_ns"]) for start in starts),
                salt=f"profile:{lane}:compute",
            ),
            "runtime_memcpy_gpu_time_ns_per_request": fresh_start_summary(
                (float(start["runtime_memcpy_gpu_time_ns"]) for start in starts),
                salt=f"profile:{lane}:memcpy",
            ),
            "phase_totals": {
                phase: {
                    "calls_per_request": fresh_start_summary(
                        (
                            float(start["phase_totals"].get(phase, {}).get("calls", 0))
                            for start in starts
                        ),
                        salt=f"profile:{lane}:{phase}:calls",
                    ),
                    "gpu_time_ns_per_request": fresh_start_summary(
                        (
                            float(
                                start["phase_totals"]
                                .get(phase, {})
                                .get("gpu_time_ns", 0)
                            )
                            for start in starts
                        ),
                        salt=f"profile:{lane}:{phase}:gpu-time",
                    ),
                }
                for phase in phase_names
            },
        }
        input_counts[lane] = {
            "fresh_starts": len(reports),
            "profile_requests": len(reports) * PROFILE_REQUESTS,
        }

    comparability = matched_lane_comparability(
        requested_configs["pypto"],
        requested_configs["sglang-matched"],
        resolved_configs["pypto"],
        resolved_configs["sglang-matched"],
    )
    if not comparability["matched_claim_allowed"]:
        return {
            "schema": SCHEMA_VERSION,
            "kind": "qwen35-9b-profile-gap-reconciliation",
            "workload": workload_record(),
            "methodology": fresh_start_methodology(),
            "gpu_time_definition": (
                "sum of CUPTI activity durations, not elapsed critical-path time"
            ),
            "profile_inputs": input_counts,
            "lane_summaries": lane_summaries,
            "matched_comparability": comparability,
            "comparisons": {},
            "status": "failed",
            "error": "matched profile controls drifted",
        }
    performance_summary = None
    if performance is not None:
        from .performance_runtime import summarize_fresh_starts

        performance_summary = summarize_fresh_starts(performance)
        if performance_summary["status"] != "complete":
            raise ReleaseContractError("performance controls are not comparable")
    candidate = lane_summaries["pypto"]
    comparisons = {}
    for baseline_lane in ("sglang-matched", "sglang-optimized"):
        baseline = lane_summaries[baseline_lane]
        phase_names = sorted(
            set(candidate["phase_totals"]) | set(baseline["phase_totals"])
        )
        phase_names = [name for name in phase_names if name != "runtime_memcpy"]
        phase_gaps = []
        for phase in phase_names:
            candidate_values = [
                float(start["phase_totals"].get(phase, {}).get("gpu_time_ns", 0))
                for start in lane_starts["pypto"]
            ]
            baseline_values = [
                float(start["phase_totals"].get(phase, {}).get("gpu_time_ns", 0))
                for start in lane_starts[baseline_lane]
            ]
            candidate_ns = float(
                candidate["phase_totals"]
                .get(phase, {})
                .get("gpu_time_ns_per_request", {"p50": 0})["p50"]
            )
            baseline_ns = float(
                baseline["phase_totals"]
                .get(phase, {})
                .get("gpu_time_ns_per_request", {"p50": 0})["p50"]
            )
            interval = bootstrap_median_comparison_ci(
                candidate_values,
                baseline_values,
                operation="difference",
                salt=f"profile:pypto-vs-{baseline_lane}:{phase}",
            )
            phase_gaps.append(
                {
                    "phase": phase,
                    "pypto_gpu_ms": candidate_ns / 1e6,
                    "baseline_gpu_ms": baseline_ns / 1e6,
                    "gap_ms": (candidate_ns - baseline_ns) / 1e6,
                    "gap_bootstrap_95ci_ms": {
                        **interval,
                        "lower": float(interval["lower"]) / 1e6,
                        "upper": float(interval["upper"]) / 1e6,
                    },
                }
            )
        candidate_values = [
            float(start["compute_gpu_time_ns"]) for start in lane_starts["pypto"]
        ]
        baseline_values = [
            float(start["compute_gpu_time_ns"]) for start in lane_starts[baseline_lane]
        ]
        candidate_total = float(candidate["compute_gpu_time_ns_per_request"]["p50"])
        baseline_total = float(baseline["compute_gpu_time_ns_per_request"]["p50"])
        total_gap_ms = (candidate_total - baseline_total) / 1e6
        total_interval = bootstrap_median_comparison_ci(
            candidate_values,
            baseline_values,
            operation="difference",
            salt=f"profile:pypto-vs-{baseline_lane}:compute-total",
        )
        explained_ms = sum(float(item["gap_ms"]) for item in phase_gaps)
        comparison: dict[str, object] = {
            "baseline_lane": baseline_lane,
            "model_compute_gap_ms": total_gap_ms,
            "model_compute_gap_bootstrap_95ci_ms": {
                **total_interval,
                "lower": float(total_interval["lower"]) / 1e6,
                "upper": float(total_interval["upper"]) / 1e6,
            },
            "phase_gap_sum_ms": explained_ms,
            "phase_reconciliation_residual_ms": total_gap_ms - explained_ms,
            "phase_reconciliation_boundary": (
                "The residual can be nonzero because each phase and the total use "
                "separate medians across fresh starts."
            ),
            "phases": phase_gaps,
        }
        if performance_summary is not None:
            candidate_starts = [
                float(start["metric_p50"]["e2e_ms"])
                for start in performance_summary["lanes"]["pypto"]["per_start"]
            ]
            baseline_starts = [
                float(start["metric_p50"]["e2e_ms"])
                for start in performance_summary["lanes"][baseline_lane]["per_start"]
            ]
            candidate_e2e = float(
                performance_summary["lanes"]["pypto"]["e2e_ms"]["p50"]
            )
            baseline_e2e = float(
                performance_summary["lanes"][baseline_lane]["e2e_ms"]["p50"]
            )
            e2e_gap = candidate_e2e - baseline_e2e
            e2e_interval = bootstrap_median_comparison_ci(
                candidate_starts,
                baseline_starts,
                operation="difference",
                salt=f"reconcile:pypto-vs-{baseline_lane}:e2e",
            )
            comparison["median_e2e_gap_ms"] = e2e_gap
            comparison["median_e2e_gap_bootstrap_95ci_ms"] = e2e_interval
            comparison["non_profiled_e2e_residual_ms"] = e2e_gap - total_gap_ms
            comparison["non_profiled_e2e_residual_boundary"] = (
                "Residual includes host scheduling, sampling, synchronization, runtime "
                "copies and profiler/method differences; it is not labeled as CUDA "
                "Graph or overlap time without replay/overlap runtime evidence."
            )
        comparisons[baseline_lane] = comparison
    return {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-9b-profile-gap-reconciliation",
        "workload": workload_record(),
        "methodology": fresh_start_methodology(),
        "gpu_time_definition": (
            "sum of CUPTI activity durations, not elapsed critical-path time"
        ),
        "profile_inputs": input_counts,
        "lane_summaries": lane_summaries,
        "matched_comparability": comparability,
        "comparisons": comparisons,
        "status": "complete",
    }
