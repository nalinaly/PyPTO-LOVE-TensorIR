"""Timing-only Qwen3.5 engine workload.

This module records completion timing and resource telemetry.  It deliberately
contains no numerical acceptance machinery and persists no generated token
values or text.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import platform
import threading
import time
import traceback
from typing import Any

from .evidence_identity import collect_run_identity
from .lanes import (
    execution_feature_record,
    matched_lane_comparability,
    memory_qualification,
    prepare_worker_environment,
    resolved_backend_record,
    server_kwargs,
    validate_resolved_backends,
)
from .workload import (
    COMPILE_WARMUPS,
    MEASURED_REQUESTS,
    OUTPUT_TOKENS,
    PROMPT_TOKENS,
    PROMPT_TOKEN_IDS,
    SAMPLE_INTERVAL_MS,
    SCHEMA_VERSION,
    UNTIMED_WARMUPS,
    ReleaseContractError,
    atomic_json,
    bootstrap_median_comparison_ci,
    distribution,
    fresh_start_methodology,
    fresh_start_summary,
    model_revision,
    sha256_file,
    workload_record,
)


ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_METRICS = (
    "e2e_ms",
    "ttft_ms",
    "tpot_ms",
    "itl_ms",
    "output_tokens_per_second",
    "decode_tokens_per_second",
    "input_tokens_per_second",
    "total_tokens_per_second",
    "requests_per_second",
)


def run_scheduler_with_release_metrics(*args, **kwargs):
    """Inject read-only allocator/compiler counters into get_server_info."""

    import dataclasses
    import torch
    from sglang.srt.compilation.compilation_counter import compilation_counter
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    original = Scheduler.get_internal_state

    def measured_internal_state(scheduler, request):
        response = original(scheduler, request)
        response.internal_state["release_torch_allocator"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        response.internal_state["release_compilation_counter"] = dataclasses.asdict(
            compilation_counter
        )
        return response

    Scheduler.get_internal_state = measured_internal_state
    return run_scheduler_process(*args, **kwargs)


def _mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise ReleaseContractError("/proc/meminfo has no MemAvailable field")


def _pgid_rss_kib(pgid: int) -> int:
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat.rpartition(")")[2].split()
            if int(fields[2]) != pgid:
                continue
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total


class ResourceSampler:
    def __init__(self, pgid: int) -> None:
        self._pgid = pgid
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.samples: list[dict[str, object]] = []
        self.identity: dict[str, object] | None = None
        self.nvml_error: str | None = None
        self._nvml: Any | None = None
        self._handle: Any | None = None
        self._stopped = False

    def start(self) -> None:
        try:
            self._nvml = importlib.import_module("pynvml")
            self._nvml.nvmlInit()
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
            name = self._nvml.nvmlDeviceGetName(self._handle)
            uuid = self._nvml.nvmlDeviceGetUUID(self._handle)
            driver = self._nvml.nvmlSystemGetDriverVersion()
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.identity = {
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "uuid": uuid.decode() if isinstance(uuid, bytes) else str(uuid),
                "driver": driver.decode() if isinstance(driver, bytes) else str(driver),
                "total_memory_bytes": int(memory.total),
            }
        except BaseException as error:
            self.nvml_error = f"{type(error).__name__}: {error}"
            self._nvml = None
            self._handle = None
        self._thread.start()

    def _nvml_value(self, name: str, *args) -> object:
        if self._nvml is None or self._handle is None:
            return None
        try:
            return getattr(self._nvml, name)(self._handle, *args)
        except BaseException:
            return None

    def _sample(self) -> dict[str, object]:
        sample: dict[str, object] = {
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "mem_available_kib": _mem_available_kib(),
            "owned_pgid_rss_kib": _pgid_rss_kib(self._pgid),
        }
        if self._nvml is not None:
            memory = self._nvml_value("nvmlDeviceGetMemoryInfo")
            utilization = self._nvml_value("nvmlDeviceGetUtilizationRates")
            throttle_reasons = self._nvml_value(
                "nvmlDeviceGetCurrentClocksThrottleReasons"
            )
            thermal_mask = int(
                getattr(self._nvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0)
            ) | int(getattr(self._nvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0))
            sample["gpu"] = {
                "memory_used_bytes": getattr(memory, "used", None),
                "memory_free_bytes": getattr(memory, "free", None),
                "utilization_percent": getattr(utilization, "gpu", None),
                "memory_utilization_percent": getattr(utilization, "memory", None),
                "temperature_c": self._nvml_value(
                    "nvmlDeviceGetTemperature", self._nvml.NVML_TEMPERATURE_GPU
                ),
                "power_mw": self._nvml_value("nvmlDeviceGetPowerUsage"),
                "sm_clock_mhz": self._nvml_value(
                    "nvmlDeviceGetClockInfo", self._nvml.NVML_CLOCK_SM
                ),
                "memory_clock_mhz": self._nvml_value(
                    "nvmlDeviceGetClockInfo", self._nvml.NVML_CLOCK_MEM
                ),
                "pstate": self._nvml_value("nvmlDeviceGetPowerState"),
                "throttle_reasons": throttle_reasons,
                "thermal_throttle": bool(
                    isinstance(throttle_reasons, int)
                    and thermal_mask
                    and throttle_reasons & thermal_mask
                ),
            }
        return sample

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._sample())
            except BaseException as error:
                self.samples.append(
                    {
                        "wall_time_ns": time.time_ns(),
                        "sample_error": f"{type(error).__name__}: {error}",
                    }
                )
            self._stop.wait(SAMPLE_INTERVAL_MS / 1000.0)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        self._thread.join(timeout=2)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except BaseException:
                pass


def _model_record(model_path: Path) -> dict[str, object]:
    config = model_path / "config.json"
    if not config.is_file():
        raise ReleaseContractError(f"model config is missing: {config}")
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise ReleaseContractError(f"model has no safetensors: {model_path}")
    return {
        "path": str(model_path),
        "revision": model_revision(model_path),
        "config_sha256": sha256_file(config),
        "shards": [
            {"name": path.name, "bytes": path.stat().st_size} for path in shards
        ],
    }


def _cache_snapshot() -> dict[str, dict[str, object]]:
    roots = []
    for name in ("SGLANG_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        raw = os.environ.get(name)
        if raw:
            roots.append((name, Path(raw)))
    files = {}
    for label, root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = f"{label}/{path.relative_to(root)}"
            files[relative] = {
                "bytes": path.stat().st_size,
                "suffix": path.suffix,
            }
    return files


def _compilation_observation(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
    *,
    requested: bool,
    scheduler_counter: dict[str, object] | None,
) -> dict[str, object]:
    new_files = sorted(set(after) - set(before))
    graph_files = [
        path for path in new_files if "computation_graph_" in Path(path).name
    ]
    compiled_files = [
        path
        for path in new_files
        if after[path]["suffix"] in {".so", ".cubin", ".ptx", ".cpp", ".py"}
        and "torch_compile" in path.lower()
    ]
    backend_invoked = bool(
        scheduler_counter
        and int(scheduler_counter.get("num_graphs_seen", 0)) > 0
        and int(scheduler_counter.get("num_inductor_compiles", 0)) > 0
    )
    return {
        "requested": requested,
        "backend_invocation_observed": backend_invoked,
        "compiled_code_file_count": len(compiled_files),
        "new_cache_file_count": len(new_files),
        "graph_records": graph_files,
        "compiled_code_files": compiled_files,
        "scheduler_counter": scheduler_counter,
        "effective": bool(backend_invoked and compiled_files),
        "evidence_boundary": (
            "Effective means the scheduler observed Dynamo graphs and Inductor compile "
            "calls, generated code appeared in the isolated cache, and subsequent "
            "measured requests completed."
        ),
    }


def _graph_observation(server_info: object) -> dict[str, object]:
    graph: dict[str, float] = {}
    if isinstance(server_info, dict):
        states = server_info.get("internal_states")
        if isinstance(states, list) and states and isinstance(states[0], dict):
            memory = states[0].get("memory_usage")
            if isinstance(memory, dict) and isinstance(memory.get("graph"), dict):
                graph = {
                    str(key): float(value)
                    for key, value in memory["graph"].items()
                    if isinstance(value, (int, float))
                }
    return {
        "capture_memory_metadata_gb": graph,
        "capture_memory_metadata_observed": any(value > 0 for value in graph.values()),
        "replay_runtime_observed": None,
        "evidence_boundary": (
            "Nonzero graph memory is capture metadata, not proof that measured "
            "requests replayed a CUDA Graph."
        ),
    }


def _completion_count(chunk: dict[str, object], previous: int) -> int:
    meta = chunk.get("meta_info")
    if isinstance(meta, dict):
        value = meta.get("completion_tokens")
        if type(value) is int and value >= previous:
            return value
    ids = chunk.get("output_ids")
    if isinstance(ids, list):
        return previous + len(ids)
    return previous


def _stream_request(engine: object, request_index: int) -> dict[str, object]:
    start = time.perf_counter_ns()
    iterator = engine.generate(
        input_ids=list(PROMPT_TOKEN_IDS),
        sampling_params={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
        },
        stream=True,
        rid=f"release-performance-{request_index}-{start}",
    )
    completion = 0
    token_arrivals: list[int] = []
    chunk_timestamps: list[dict[str, int]] = []
    for chunk in iterator:
        now = time.perf_counter_ns()
        if type(chunk) is not dict:
            raise ReleaseContractError("streaming engine returned a non-object chunk")
        updated = _completion_count(chunk, completion)
        if updated < completion or updated > OUTPUT_TOKENS:
            raise ReleaseContractError("stream completion count is not monotonic")
        if updated - completion > 1:
            raise ReleaseContractError(
                "stream_interval=1 was not effective; exact ITL cannot be reconstructed"
            )
        for _ in range(updated - completion):
            token_arrivals.append(now)
        completion = updated
        chunk_timestamps.append({"arrival_ns": now, "completion_tokens": completion})
    end = time.perf_counter_ns()
    if completion != OUTPUT_TOKENS or len(token_arrivals) != OUTPUT_TOKENS:
        raise ReleaseContractError(
            f"request completed {completion} tokens; expected {OUTPUT_TOKENS}"
        )
    e2e_ms = (end - start) / 1e6
    ttft_ms = (token_arrivals[0] - start) / 1e6
    itl_ms = [
        (right - left) / 1e6 for left, right in zip(token_arrivals, token_arrivals[1:])
    ]
    return {
        "request_index": request_index,
        "request_start_ns": start,
        "request_end_ns": end,
        "chunk_timestamps": chunk_timestamps,
        "completion_tokens": completion,
        "e2e_ms": e2e_ms,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "tpot_ms": (token_arrivals[-1] - token_arrivals[0])
        / max(1, OUTPUT_TOKENS - 1)
        / 1e6,
        "output_tokens_per_second": OUTPUT_TOKENS * 1e9 / (end - start),
        "decode_tokens_per_second": (
            (OUTPUT_TOKENS - 1) * 1e9 / max(1, token_arrivals[-1] - token_arrivals[0])
        ),
        "input_tokens_per_second": PROMPT_TOKENS
        * 1e9
        / max(1, token_arrivals[0] - start),
        "total_tokens_per_second": (PROMPT_TOKENS + OUTPUT_TOKENS)
        * 1e9
        / (end - start),
        "requests_per_second": 1e9 / (end - start),
    }


def _resource_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    host = [
        int(sample["mem_available_kib"])
        for sample in samples
        if type(sample.get("mem_available_kib")) is int
    ]
    rss = [
        int(sample["owned_pgid_rss_kib"])
        for sample in samples
        if type(sample.get("owned_pgid_rss_kib")) is int
    ]
    gpu_used = [
        int(gpu["memory_used_bytes"])
        for sample in samples
        if isinstance((gpu := sample.get("gpu")), dict)
        and type(gpu.get("memory_used_bytes")) is int
    ]
    gpu_free = [
        int(gpu["memory_free_bytes"])
        for sample in samples
        if isinstance((gpu := sample.get("gpu")), dict)
        and type(gpu.get("memory_free_bytes")) is int
    ]
    thermal_throttle = any(
        bool(gpu.get("thermal_throttle"))
        for sample in samples
        if isinstance((gpu := sample.get("gpu")), dict)
    )
    return {
        "sample_count": len(samples),
        "minimum_mem_available_kib": min(host) if host else None,
        "peak_owned_pgid_rss_kib": max(rss) if rss else None,
        "peak_gpu_memory_used_bytes": max(gpu_used) if gpu_used else None,
        "minimum_gpu_memory_free_bytes": min(gpu_free) if gpu_free else None,
        "thermal_throttle_observed": thermal_throttle,
    }


def _start_metric_estimate(report: dict[str, object], metric: str) -> float:
    requests = report.get("raw_requests")
    if type(requests) is not list or len(requests) != MEASURED_REQUESTS:
        raise ReleaseContractError("performance report request count drifted")
    if metric == "itl_ms":
        request_values = [
            distribution(request[metric])["p50"]
            for request in requests
            if isinstance(request, dict)
        ]
    else:
        request_values = [
            float(request[metric]) for request in requests if isinstance(request, dict)
        ]
    if len(request_values) != MEASURED_REQUESTS:
        raise ReleaseContractError(f"performance metric {metric!r} is incomplete")
    return distribution(request_values)["p50"]


def summarize_fresh_starts(
    reports_by_lane: dict[str, list[dict[str, object]]],
    *,
    expected_starts: int = 4,
) -> dict[str, object]:
    """Summarize performance without pooling requests across process starts."""

    expected_lanes = {"pypto", "sglang-matched", "sglang-optimized"}
    if set(reports_by_lane) != expected_lanes:
        raise ReleaseContractError("performance summary requires exactly three lanes")
    lane_summaries: dict[str, dict[str, object]] = {}
    per_start_metrics: dict[str, dict[str, list[float]]] = {}
    requested_configs: dict[str, dict[str, object]] = {}
    resolved_configs: dict[str, dict[str, object]] = {}
    gpu_identities = []
    for lane in sorted(expected_lanes):
        reports = reports_by_lane[lane]
        if len(reports) != expected_starts:
            raise ReleaseContractError(
                f"{lane} requires {expected_starts} fresh performance starts"
            )
        for report in reports:
            if (
                report.get("status") != "complete"
                or report.get("kind") != "qwen35-9b-performance-only"
                or report.get("lane") != lane
                or report.get("workload") != workload_record()
            ):
                raise ReleaseContractError(
                    f"performance report is not accepted for {lane}"
                )
            compilation = report.get("compilation")
            if (
                not isinstance(compilation, dict)
                or compilation.get("requested") is not True
                or compilation.get("backend_invocation_observed") is not True
            ):
                raise ReleaseContractError(
                    f"{lane} did not prove a requested torch.compile backend invocation"
                )
        configs = [report.get("requested_server_config") for report in reports]
        if any(type(config) is not dict for config in configs):
            raise ReleaseContractError(f"{lane} requested configuration is absent")
        if any(config != configs[0] for config in configs[1:]):
            raise ReleaseContractError(f"{lane} configuration drifted across starts")
        requested_configs[lane] = configs[0]
        resolved = [report.get("resolved_backends") for report in reports]
        if any(type(config) is not dict for config in resolved):
            raise ReleaseContractError(f"{lane} resolved configuration is absent")
        if any(config != resolved[0] for config in resolved[1:]):
            raise ReleaseContractError(
                f"{lane} resolved configuration drifted across starts"
            )
        resolved_configs[lane] = resolved[0]
        resource_summaries = []
        for report in reports:
            resources = report.get("resources")
            if (
                not isinstance(resources, dict)
                or resources.get("nvml_error") is not None
            ):
                raise ReleaseContractError(f"{lane} NVML sampling is incomplete")
            identity = resources.get("gpu_identity")
            resource_summary = resources.get("summary")
            if not isinstance(identity, dict) or not isinstance(resource_summary, dict):
                raise ReleaseContractError(f"{lane} resource identity is incomplete")
            required_resource_fields = (
                "peak_gpu_memory_used_bytes",
                "minimum_gpu_memory_free_bytes",
                "minimum_mem_available_kib",
                "peak_owned_pgid_rss_kib",
            )
            if any(
                type(resource_summary.get(field)) is not int
                for field in required_resource_fields
            ):
                raise ReleaseContractError(f"{lane} resource telemetry is incomplete")
            if resource_summary.get("thermal_throttle_observed") is not False:
                raise ReleaseContractError(f"{lane} observed thermal throttling")
            if (
                int(resource_summary.get("minimum_gpu_memory_free_bytes", 0))
                < 4 * 1024**3
            ):
                raise ReleaseContractError(f"{lane} crossed the 4 GiB GPU free floor")
            if int(resource_summary.get("minimum_mem_available_kib", 0)) < 12 * 1024**2:
                raise ReleaseContractError(
                    f"{lane} crossed the 12 GiB host MemAvailable floor"
                )
            gpu_identities.append(identity)
            resource_summaries.append(resource_summary)
        estimates = {
            metric: [_start_metric_estimate(report, metric) for report in reports]
            for metric in PERFORMANCE_METRICS
        }
        per_start_metrics[lane] = estimates
        summary: dict[str, object] = {
            "fresh_starts": len(reports),
            "measured_requests": len(reports) * MEASURED_REQUESTS,
            "per_start": [
                {
                    "run_id": report.get("run_id"),
                    "request_count": MEASURED_REQUESTS,
                    "metric_p50": {
                        metric: estimates[metric][index]
                        for metric in PERFORMANCE_METRICS
                    },
                }
                for index, report in enumerate(reports)
            ],
            "cold_engine_start_ms": fresh_start_summary(
                (float(report["cold_engine_start_ms"]) for report in reports),
                salt=f"performance:{lane}:cold_engine_start_ms",
            ),
            "first_compile_trigger_request_ms": fresh_start_summary(
                (
                    float(report["first_compile_trigger_request_ms"])
                    for report in reports
                ),
                salt=f"performance:{lane}:first_compile_trigger_request_ms",
            ),
            "resources": {
                "peak_gpu_memory_used_bytes": max(
                    int(item["peak_gpu_memory_used_bytes"])
                    for item in resource_summaries
                ),
                "minimum_gpu_memory_free_bytes": min(
                    int(item["minimum_gpu_memory_free_bytes"])
                    for item in resource_summaries
                ),
                "minimum_mem_available_kib": min(
                    int(item["minimum_mem_available_kib"])
                    for item in resource_summaries
                ),
                "peak_owned_pgid_rss_kib": max(
                    int(item["peak_owned_pgid_rss_kib"]) for item in resource_summaries
                ),
                "thermal_throttle_observed": False,
            },
        }
        for metric, values in estimates.items():
            summary[metric] = fresh_start_summary(
                values, salt=f"performance:{lane}:{metric}"
            )
        lane_summaries[lane] = summary

    if any(identity != gpu_identities[0] for identity in gpu_identities[1:]):
        raise ReleaseContractError(
            "performance matrix spans different GPUs, drivers, or memory sizes"
        )
    comparability = matched_lane_comparability(
        requested_configs["pypto"],
        requested_configs["sglang-matched"],
        resolved_configs["pypto"],
        resolved_configs["sglang-matched"],
    )
    comparisons: dict[str, object] = {}
    candidate_rates = per_start_metrics["pypto"]["output_tokens_per_second"]
    candidate_median = float(lane_summaries["pypto"]["output_tokens_per_second"]["p50"])
    for baseline_lane in ("sglang-matched", "sglang-optimized"):
        baseline_rates = per_start_metrics[baseline_lane]["output_tokens_per_second"]
        baseline_median = float(
            lane_summaries[baseline_lane]["output_tokens_per_second"]["p50"]
        )
        ratio = candidate_median / baseline_median
        interval = bootstrap_median_comparison_ci(
            candidate_rates,
            baseline_rates,
            operation="ratio",
            salt=f"performance:pypto-vs-{baseline_lane}:output-rate",
        )
        comparisons[baseline_lane] = {
            "metric": "output_tokens_per_second",
            "pypto_percent_of_baseline": ratio * 100.0,
            "median_ratio_bootstrap_95ci_percent": {
                **interval,
                "lower": float(interval["lower"]) * 100.0,
                "upper": float(interval["upper"]) * 100.0,
            },
        }
    return {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-9b-fresh-start-performance-summary",
        "workload": workload_record(),
        "methodology": fresh_start_methodology(),
        "lanes": lane_summaries,
        "matched_comparability": comparability,
        "comparisons": comparisons,
        "gpu_identity": gpu_identities[0],
        "status": ("complete" if comparability["matched_claim_allowed"] else "failed"),
    }


def run(
    lane: str,
    model_path: Path,
    run_id: str,
    run_dir: Path,
    optimized_memory_mode: str = "zero-offload",
) -> int:
    prepare_worker_environment(lane)
    model_path = model_path.resolve(strict=True)
    report_path = run_dir / f"qwen35-9b-performance-{lane}.json"
    resources_path = run_dir / f"qwen35-9b-resources-{lane}.json"
    memory = memory_qualification(lane, optimized_memory_mode)
    requested = server_kwargs(lane, model_path, optimized_memory_mode)
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-9b-performance-only",
        "lane": lane,
        "run_id": run_id,
        "workload": workload_record(),
        "measurement": {
            "entrypoint": "sglang.Engine offline streaming API",
            "first_compile_trigger_requests": COMPILE_WARMUPS,
            "untimed_warmups": UNTIMED_WARMUPS,
            "measured_requests": MEASURED_REQUESTS,
            "profiler_enabled": False,
            "resource_sample_interval_ms": SAMPLE_INTERVAL_MS,
        },
        "requested_server_config": requested,
        "memory_qualification": memory,
        "model": _model_record(model_path),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "status": "starting",
    }
    sampler = ResourceSampler(os.getpgid(0))
    engine = None
    sampler.start()
    startup_start = time.perf_counter_ns()
    cache_before = _cache_snapshot()
    try:
        import sglang as sgl
        from sglang.srt.entrypoints.engine import Engine

        class ReleaseMeasurementEngine(Engine):
            run_scheduler_process_func = staticmethod(
                run_scheduler_with_release_metrics
            )

        engine = ReleaseMeasurementEngine(**requested)
        engine_ready = time.perf_counter_ns()
        resolved = resolved_backend_record(engine.server_args)
        validate_resolved_backends(lane, resolved)
        report["sglang"] = {
            "version": sgl.__version__,
            "source": str(Path(sgl.__file__).resolve()),
        }
        report["resolved_backends"] = resolved
        report["execution_features"] = execution_feature_record(requested, resolved)
        report["cold_engine_start_ms"] = (engine_ready - startup_start) / 1e6

        compile_start = time.perf_counter_ns()
        _stream_request(engine, -3)
        report["first_compile_trigger_request_ms"] = (
            time.perf_counter_ns() - compile_start
        ) / 1e6
        cache_after_compile = _cache_snapshot()
        warmup_ms = []
        for index in range(UNTIMED_WARMUPS):
            started = time.perf_counter_ns()
            _stream_request(engine, -2 + index)
            warmup_ms.append((time.perf_counter_ns() - started) / 1e6)
        report["untimed_warmup_ms"] = warmup_ms

        requests = [
            _stream_request(engine, index) for index in range(MEASURED_REQUESTS)
        ]
        report["raw_requests"] = requests
        report["metrics"] = {
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
        }
        server_info = engine.get_server_info()
        internal = (
            server_info.get("internal_states")
            if isinstance(server_info, dict)
            else None
        )
        internal_zero = (
            internal[0]
            if isinstance(internal, list) and internal and isinstance(internal[0], dict)
            else {}
        )
        compilation = _compilation_observation(
            cache_before,
            cache_after_compile,
            requested=bool(requested.get("enable_torch_compile")),
            scheduler_counter=internal_zero.get("release_compilation_counter"),
        )
        report["compilation"] = compilation
        report["compilation"]["timing_boundary"] = (
            "first_compile_trigger_request_ms includes compilation plus one full "
            "19+64 request and is not compiler-only time"
        )
        report["torch_allocator"] = internal_zero.get("release_torch_allocator")
        graph = _graph_observation(server_info)
        report["execution_features"]["cuda_graph"].update(graph)
        if compilation["requested"] and not compilation["backend_invocation_observed"]:
            raise ReleaseContractError(
                f"{lane} requested torch.compile but no backend invocation was observed"
            )
        if lane == "sglang-optimized" and not all(
            report["execution_features"][feature]["requested"]
            and report["execution_features"][feature]["enabled"]
            for feature in ("cuda_graph", "overlap_schedule")
        ):
            raise ReleaseContractError(
                "optimized lane did not keep CUDA Graph and overlap configuration enabled"
            )
        report["sglang_memory_state"] = internal
        report["sglang_startup_time"] = (
            server_info.get("startup_time") if isinstance(server_info, dict) else None
        )
        sampler.stop()
        report["identity_collection_boundary"] = (
            "Full model/package hashing runs after every timed request and after "
            "resource sampling stops, so it cannot warm cold-load inputs or enter "
            "latency/resource summaries."
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
        if engine is not None:
            try:
                engine.shutdown()
            except BaseException as error:
                report["shutdown_error"] = f"{type(error).__name__}: {error}"
                report["status"] = "failed"
                return_code = 1
        sampler.stop()
        resource_payload = {
            "schema": SCHEMA_VERSION,
            "kind": "qwen35-9b-resource-samples",
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
                    "metrics": report.get("metrics"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return return_code
