#!/usr/bin/env python3
"""Validate one complete evidence set and render immutable release fragments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.evidence_identity import comparable_identity  # noqa: E402
from benchmarks.release.operator_performance_runtime import (  # noqa: E402
    OPERATOR_LANES,
    OPERATOR_SCHEDULE,
    summarize_fresh_starts as summarize_operator_fresh_starts,
)
from benchmarks.release.performance_runtime import (  # noqa: E402
    summarize_fresh_starts,
)
from benchmarks.release.profile_runtime import reconcile  # noqa: E402
from benchmarks.release.workload import (  # noqa: E402
    CPU_JOBS,
    LANES,
    MEASURED_REQUESTS,
    PERFORMANCE_SCHEDULE,
    PROFILE_SCHEDULE,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    canonical_json_sha256,
    distribution,
    read_json,
    require_path_below_runs,
    sha256_file,
    workload_record,
)


_OUTPUT_NAME = re.compile(r"release-results-[A-Za-z0-9][A-Za-z0-9._-]{0,160}\Z")
_PROFILE_FOR_LANE = {
    "pypto": "pypto",
    "sglang-matched": "baseline",
    "sglang-optimized": "baseline",
}
_PLACEHOLDERS = (
    "待正式",
    "待补",
    "tbd",
    "todo",
    "placeholder",
    "pending formal",
    "xx%",
)


def _relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ReleaseContractError(f"release evidence escaped the workspace: {resolved}")
    return resolved.relative_to(ROOT).as_posix()


def _evidence_path(raw: str | Path) -> Path:
    path = require_path_below_runs(ROOT, raw).resolve(strict=True)
    if not path.is_file():
        raise ReleaseContractError(f"release evidence is not a file: {path}")
    return path


def _input(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": _relative(path), "sha256": sha256_file(path)}


def _bound_artifact(
    owner_report: Path,
    raw_path: object,
    expected_sha256: object,
    role: str,
) -> dict[str, str]:
    if type(raw_path) is not str or type(expected_sha256) is not str:
        raise ReleaseContractError(f"{role} artifact record is incomplete")
    path = _evidence_path(raw_path)
    if path.parent != owner_report.parent:
        raise ReleaseContractError(f"{role} artifact escaped its owned run directory")
    if sha256_file(path) != expected_sha256:
        raise ReleaseContractError(f"{role} artifact SHA-256 differs: {path}")
    return _input(path, role)


def _selectors(values: list[str], label: str) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = defaultdict(list)
    for raw in values:
        lane, separator, path = raw.partition("=")
        if separator != "=" or lane not in LANES:
            raise ReleaseContractError(f"invalid {label} selector: {raw}")
        selected[lane].append(_evidence_path(path))
    return dict(selected)


def _immutable_controller_gpu(gpu: object) -> dict[str, object]:
    if not isinstance(gpu, dict):
        raise ReleaseContractError("controller GPU identity is missing")
    result = {
        "name": gpu.get("name"),
        "compute_capability": gpu.get("compute_capability"),
        "total_memory_mib": int(str(gpu.get("memory_mib", "-1"))),
        "driver": gpu.get("driver"),
    }
    if (
        type(result["name"]) is not str
        or result["compute_capability"] != "12.0"
        or int(result["total_memory_mib"]) <= 0
        or type(result["driver"]) is not str
    ):
        raise ReleaseContractError("controller GPU identity is incomplete")
    return result


def _controller_evidence(
    report_path: Path, expected_profile: str
) -> tuple[list[dict[str, str]], dict[str, object]]:
    controller_path = _evidence_path(report_path.parent / "process.json")
    initial_path = _evidence_path(report_path.parent / "initial-audit.json")
    locked_path = _evidence_path(report_path.parent / "locked-audit.json")
    identity_path = _evidence_path(report_path.parent / "post-identity-audit.json")
    controller = read_json(controller_path)
    initial = read_json(initial_path)
    locked = read_json(locked_path)
    post_identity = read_json(identity_path)
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
        or policy.get("formal_identity_verified") is not True
    ):
        raise ReleaseContractError(f"bounded resource policy drifted: {controller_path}")
    immutable = None
    for label, audit in (
        ("initial", initial),
        ("locked", locked),
        ("post-identity", post_identity),
        ("post", post),
    ):
        gpu = audit.get("gpu") or {}
        if (
            audit.get("external_compute_pids")
            or audit.get("protected_compute_pids")
            or audit.get("protected_runtime_mapping_pids")
            or audit.get("unreadable_protected_maps")
        ):
            raise ReleaseContractError(
                f"{label} NVIDIA ownership audit failed: {controller_path}"
            )
        observed = _immutable_controller_gpu(gpu)
        if immutable is None:
            immutable = observed
        elif immutable != observed:
            raise ReleaseContractError(
                f"GPU identity changed during bounded run: {controller_path}"
            )
    if immutable is None:
        raise ReleaseContractError("controller has no immutable GPU identity")
    return (
        [
            _input(controller_path, "gpu-controller"),
            _input(initial_path, "gpu-audit:initial"),
            _input(locked_path, "gpu-audit:locked"),
            _input(identity_path, "gpu-audit:post-identity"),
        ],
        immutable,
    )


def _cpu_controller_evidence(report_path: Path) -> list[dict[str, str]]:
    controller_path = _evidence_path(report_path.parent / "process.json")
    controller = read_json(controller_path)
    if (
        controller.get("mode") != "cpu-bounded"
        or controller.get("framework_profile") != "pypto"
        or controller.get("status") != "exited"
        or controller.get("return_code") != 0
        or controller.get("abort_reason") is not None
        or controller.get("policy", {}).get("parallelism") != CPU_JOBS
        or controller.get("policy", {}).get("launch_admission_floor_kib") is not None
        or controller.get("policy", {}).get("formal_identity_verified") is not True
    ):
        raise ReleaseContractError(
            f"bounded CPU controller did not accept the run: {controller_path}"
        )
    return [_input(controller_path, "cpu-controller")]


class IdentityAudit:
    """Cross-compare the full byte identity across every selected report."""

    def __init__(self) -> None:
        self._global: dict[str, object] | None = None
        self._runtime_by_profile: dict[str, dict[str, object]] = {}
        self._compiler: dict[str, object] | None = None
        self.records: list[dict[str, str]] = []

    def add(
        self,
        report: dict[str, object],
        *,
        report_path: Path,
        expected_profile: str,
        controller_gpu: dict[str, object] | None,
    ) -> None:
        identity = report.get("evidence_identity")
        if not isinstance(identity, dict):
            raise ReleaseContractError(f"report has no evidence identity: {report_path}")
        claimed = identity.get("identity_sha256")
        unsigned = copy.deepcopy(identity)
        unsigned.pop("identity_sha256", None)
        if claimed != canonical_json_sha256(unsigned):
            raise ReleaseContractError(f"evidence identity digest differs: {report_path}")
        selected = identity.get("selected_environment_lock")
        runtime = identity.get("runtime")
        if selected != expected_profile or not isinstance(runtime, dict):
            raise ReleaseContractError(
                f"report selected the wrong formal profile: {report_path}"
            )
        if runtime.get("profile") != expected_profile:
            raise ReleaseContractError(f"runtime profile differs: {report_path}")
        global_identity = comparable_identity(identity)
        locks = identity["environment_locks"]
        selected_lock = locks[expected_profile]["identity"]
        runtime_torch = runtime.get("torch")
        runtime_sglang = runtime.get("sglang")
        if not isinstance(runtime_torch, dict) or any(
            selected_lock.get(lock_field) != runtime_torch.get(runtime_field)
            for lock_field, runtime_field in (
                ("torch", "version"),
                ("torch_git", "git"),
                ("cuda", "cuda_toolkit"),
                ("hip", "hip"),
            )
        ):
            raise ReleaseContractError(
                f"runtime Torch differs from selected identity lock: {report_path}"
            )
        source_sglang = identity["sources"]["sglang"]
        if not isinstance(runtime_sglang, dict) or any(
            runtime_sglang.get(field) != source_sglang.get(source_field)
            for field, source_field in (
                ("version", "version"),
                ("source_commit", "commit"),
                ("source_tree", "tree"),
            )
        ):
            raise ReleaseContractError(
                f"runtime SGLang differs from locked source: {report_path}"
            )
        if self._global is None:
            self._global = global_identity
        elif self._global != global_identity:
            raise ReleaseContractError(
                f"model/package/source/GPU identity drifted: {report_path}"
            )
        stable_runtime = copy.deepcopy(runtime)
        stable_runtime.pop("prefix", None)
        torch = stable_runtime.get("torch")
        if isinstance(torch, dict):
            torch.pop("module", None)
        previous_runtime = self._runtime_by_profile.setdefault(
            expected_profile, stable_runtime
        )
        if previous_runtime != stable_runtime:
            raise ReleaseContractError(
                f"{expected_profile} runtime identity drifted: {report_path}"
            )
        compiler = identity.get("compiler")
        if expected_profile == "pypto":
            if not isinstance(compiler, dict):
                raise ReleaseContractError(
                    f"candidate report has no compiler identity: {report_path}"
                )
            if self._compiler is None:
                self._compiler = compiler
            elif self._compiler != compiler:
                raise ReleaseContractError(
                    f"candidate compiler identity drifted: {report_path}"
                )
        elif compiler is not None:
            raise ReleaseContractError(
                f"baseline report imported the candidate compiler: {report_path}"
            )
        gpu = identity.get("gpu")
        if not isinstance(gpu, dict) or not gpu.get("uuid"):
            raise ReleaseContractError(f"GPU UUID is missing: {report_path}")
        if controller_gpu is not None:
            expected_gpu = {
                "name": gpu.get("name"),
                "compute_capability": gpu.get("compute_capability"),
                "total_memory_mib": gpu.get("total_memory_mib"),
                "driver": gpu.get("driver"),
            }
            if controller_gpu != expected_gpu:
                raise ReleaseContractError(
                    f"worker/controller GPU identity differs: {report_path}"
                )
        self.records.append(
            {
                "path": _relative(report_path),
                "profile": expected_profile,
                "identity_sha256": str(claimed),
            }
        )

    def result(self) -> dict[str, object]:
        if self._global is None or set(self._runtime_by_profile) != {"pypto", "baseline"}:
            raise ReleaseContractError(
                "release evidence does not contain both candidate and baseline identities"
            )
        if self._compiler is None:
            raise ReleaseContractError("release evidence has no observed PyPTO compiler")
        candidate = self._runtime_by_profile["pypto"]
        baseline = self._runtime_by_profile["baseline"]
        for family in ("torch", "sglang"):
            if candidate.get(family) != baseline.get(family):
                raise ReleaseContractError(
                    f"candidate/baseline {family} semantic identity differs"
                )
        compiler = copy.deepcopy(self._compiler)
        compiler["cuda_toolkit_root"] = Path(
            str(compiler["cuda_toolkit_root"])
        ).name
        compiler["tileiras_real_path"] = Path(
            str(compiler["tileiras_real_path"])
        ).name
        result = copy.deepcopy(self._global)
        result["compiler"] = compiler
        result["runtimes"] = copy.deepcopy(self._runtime_by_profile)
        result["report_count"] = len(self.records)
        result["reports"] = sorted(self.records, key=lambda item: item["path"])
        result["identity_sha256"] = canonical_json_sha256(result)
        return result


def _matrix_performance_paths(
    path: Path,
) -> tuple[dict[str, list[Path]], Path, list[dict[str, str]]]:
    summary_path = _evidence_path(path)
    summary = read_json(summary_path)
    if (
        summary.get("status") != "complete"
        or summary.get("kind") != "qwen35-9b-performance-matrix-control"
        or len(summary.get("runs", [])) != len(PERFORMANCE_SCHEDULE)
        or [item.get("lane") for item in summary.get("runs", [])]
        != list(PERFORMANCE_SCHEDULE)
    ):
        raise ReleaseContractError("performance matrix is not complete")
    selected: dict[str, list[Path]] = defaultdict(list)
    for item in summary["runs"]:
        if item.get("return_code") != 0:
            raise ReleaseContractError("performance matrix contains a failed start")
        selected[str(item["lane"])].append(_evidence_path(str(item["report"])))
    aggregation_path = _evidence_path(str(summary.get("aggregation")))
    return (
        dict(selected),
        aggregation_path,
        [
            _input(summary_path, "performance-matrix"),
            _input(aggregation_path, "performance-aggregation"),
        ],
    )


def _load_performance(
    paths: dict[str, list[Path]], audit: IdentityAudit
) -> tuple[dict[str, object], list[dict[str, str]], dict[str, list[dict[str, object]]]]:
    if set(paths) != set(LANES) or any(len(items) != 4 for items in paths.values()):
        raise ReleaseContractError(
            "performance rendering requires four fresh starts for every lane"
        )
    result: dict[str, object] = {}
    all_inputs: list[dict[str, str]] = []
    raw_reports: dict[str, list[dict[str, object]]] = {}
    for lane in LANES:
        reports = [read_json(path) for path in paths[lane]]
        raw_reports[lane] = reports
        peak_gpu = []
        minimum_host = []
        peak_rss = []
        resolved = []
        memory_qualifications = []
        report_inputs = []
        for path, report in zip(paths[lane], reports):
            if (
                report.get("status") != "complete"
                or report.get("kind") != "qwen35-9b-performance-only"
                or report.get("lane") != lane
                or report.get("workload") != workload_record()
            ):
                raise ReleaseContractError(f"performance report is not accepted: {path}")
            expected_profile = _PROFILE_FOR_LANE[lane]
            controller_inputs, controller_gpu = _controller_evidence(path, expected_profile)
            audit.add(
                report,
                report_path=path,
                expected_profile=expected_profile,
                controller_gpu=controller_gpu,
            )
            raw = report.get("raw_requests")
            if type(raw) is not list or len(raw) != MEASURED_REQUESTS:
                raise ReleaseContractError(f"performance request count drifted: {path}")
            if any(
                request.get("completion_tokens") != 64
                or type(request.get("chunk_timestamps")) is not list
                or len(request["chunk_timestamps"]) != 64
                or type(request.get("itl_ms")) is not list
                or len(request["itl_ms"]) != 63
                for request in raw
            ):
                raise ReleaseContractError(
                    f"performance request completion/timestamps drifted: {path}"
                )
            resources = report["resources"]
            if resources.get("nvml_error") is not None:
                raise ReleaseContractError(f"NVML sampling failed: {path}")
            resource_gpu = resources.get("gpu_identity")
            identity_gpu = report["evidence_identity"]["gpu"]
            if not isinstance(resource_gpu, dict) or any(
                resource_gpu.get(key) != identity_gpu.get(identity_key)
                for key, identity_key in (
                    ("name", "name"),
                    ("uuid", "uuid"),
                    ("driver", "driver"),
                )
            ):
                raise ReleaseContractError(f"NVML/report GPU identity differs: {path}")
            if int(resource_gpu.get("total_memory_bytes", -1)) // (1024**2) != int(
                identity_gpu["total_memory_mib"]
            ):
                raise ReleaseContractError(f"NVML GPU memory identity differs: {path}")
            resource_summary = resources["summary"]
            if resource_summary.get("thermal_throttle_observed") is not False:
                raise ReleaseContractError(f"thermal throttling observed: {path}")
            if int(resource_summary.get("minimum_gpu_memory_free_bytes", 0)) < 4 * 1024**3:
                raise ReleaseContractError(
                    f"GPU free memory fell below the 4 GiB release floor: {path}"
                )
            if int(resource_summary.get("minimum_mem_available_kib", 0)) < 16 * 1024**2:
                raise ReleaseContractError(
                    f"host MemAvailable fell below the 16 GiB release floor: {path}"
                )
            peak_gpu.append(int(resource_summary["peak_gpu_memory_used_bytes"]))
            minimum_host.append(int(resource_summary["minimum_mem_available_kib"]))
            peak_rss.append(int(resource_summary["peak_owned_pgid_rss_kib"]))
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
            if lane == "sglang-optimized":
                features = report.get("execution_features", {})
                if report.get("compilation", {}).get("effective") is not True or any(
                    features.get(feature, {}).get(field) is not True
                    for feature in ("cuda_graph", "overlap_schedule")
                    for field in ("requested", "enabled")
                ):
                    raise ReleaseContractError(
                        f"optimized compile/configuration evidence is incomplete: {path}"
                    )
            current_inputs = [_input(path, f"performance:{lane}"), *controller_inputs]
            current_inputs.append(
                _bound_artifact(
                    path,
                    resources.get("path"),
                    resources.get("sha256"),
                    f"performance-resources:{lane}",
                )
            )
            report_inputs.extend(current_inputs)
            all_inputs.extend(current_inputs)
        if any(item != resolved[0] for item in resolved[1:]):
            raise ReleaseContractError(f"resolved backend drift across {lane} starts")
        if any(item != memory_qualifications[0] for item in memory_qualifications[1:]):
            raise ReleaseContractError(f"memory qualification drift across {lane} starts")
        result[lane] = {
            "fresh_starts": len(reports),
            "measured_requests": len(reports) * MEASURED_REQUESTS,
            "resources": {
                "peak_gpu_memory_used_bytes": max(peak_gpu),
                "minimum_mem_available_kib": min(minimum_host),
                "peak_owned_pgid_rss_kib": max(peak_rss),
            },
            "resolved_backends": resolved[0],
            "memory_qualification": memory_qualifications[0],
            "inputs": report_inputs,
        }
    fresh_start = summarize_fresh_starts(raw_reports)
    if fresh_start.get("status") != "complete":
        raise ReleaseContractError("fresh-start performance controls are not comparable")
    for lane in LANES:
        fresh_start["lanes"][lane].update(
            {
                "resolved_backends": result[lane]["resolved_backends"],
                "memory_qualification": result[lane]["memory_qualification"],
                "inputs": result[lane]["inputs"],
            }
        )
    fresh_start["pypto_percent_of_stock"] = {
        lane: fresh_start["comparisons"][lane]["pypto_percent_of_baseline"]
        for lane in ("sglang-matched", "sglang-optimized")
    }
    return fresh_start, all_inputs, raw_reports


def _summary_correctness_paths(
    path: Path,
) -> tuple[Path | None, list[Path], list[dict[str, str]]]:
    summary_path = _evidence_path(path)
    summary = read_json(summary_path)
    kind = summary.get("kind")
    runs = summary.get("runs")
    if summary.get("status") != "complete" or type(runs) is not list:
        raise ReleaseContractError("correctness summary is not complete")
    if kind == "qwen35-9b-all-control":
        if (
            len(runs) != 4
            or [item.get("phase") for item in runs]
            != ["reference", "candidate", "candidate", "candidate"]
        ):
            raise ReleaseContractError("all-mode correctness schedule differs")
        reference = _evidence_path(str(runs[0]["report"]))
        candidates = [_evidence_path(str(item["report"])) for item in runs[1:]]
    elif kind == "qwen35-9b-candidate-control":
        if len(runs) != 3:
            raise ReleaseContractError("candidate correctness schedule differs")
        reference = None
        candidates = [_evidence_path(str(item["report"])) for item in runs]
    else:
        raise ReleaseContractError(f"unsupported correctness summary kind: {kind}")
    if any(item.get("return_code") != 0 for item in runs):
        raise ReleaseContractError("correctness summary contains a failed start")
    return reference, candidates, [_input(summary_path, "correctness-control")]


def _reference_from_candidates(paths: list[Path]) -> Path:
    candidates = [read_json(path) for path in paths]
    references = {
        str(report.get("reference", {}).get("path")) for report in candidates
    }
    if len(references) != 1 or "None" in references:
        raise ReleaseContractError("candidate reports do not share one reference")
    return _evidence_path(next(iter(references)))


def _load_correctness(
    reference_path: Path,
    paths: list[Path],
    audit: IdentityAudit,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    if len(paths) != 3:
        raise ReleaseContractError("correctness rendering requires three fresh starts")
    reference = read_json(reference_path)
    if (
        reference.get("status") != "complete"
        or reference.get("kind") != "qwen35-9b-multitoken-reference"
        or reference.get("workload") != workload_record()
    ):
        raise ReleaseContractError("stock correctness reference is not accepted")
    reference_controller, reference_gpu = _controller_evidence(reference_path, "baseline")
    audit.add(
        reference,
        report_path=reference_path,
        expected_profile="baseline",
        controller_gpu=reference_gpu,
    )
    reports = [read_json(path) for path in paths]
    identities = set()
    sequences = set()
    coverage_calls = []
    inductor_calls = []
    handwritten_calls = []
    logits = reference.get("logits")
    if not isinstance(logits, dict):
        raise ReleaseContractError("correctness reference logits are missing")
    inputs = [
        _input(reference_path, "correctness:reference"),
        _bound_artifact(
            reference_path,
            logits.get("path"),
            logits.get("file_sha256"),
            "correctness-reference-logits",
        ),
        *reference_controller,
    ]
    expected_output_ids = reference.get("output_token_ids")
    if type(expected_output_ids) is not list or len(expected_output_ids) != 64:
        raise ReleaseContractError("correctness reference token sequence is incomplete")
    for path, report in zip(paths, reports):
        if (
            report.get("status") != "complete"
            or report.get("all_passed") is not True
            or report.get("kind") != "qwen35-9b-multitoken-correctness"
            or report.get("workload") != workload_record()
            or report.get("thresholds") != reference.get("thresholds")
        ):
            raise ReleaseContractError(f"correctness report is not accepted: {path}")
        controller_inputs, controller_gpu = _controller_evidence(path, "pypto")
        audit.add(
            report,
            report_path=path,
            expected_profile="pypto",
            controller_gpu=controller_gpu,
        )
        if Path(str(report.get("reference", {}).get("path"))).resolve() != reference_path:
            raise ReleaseContractError(f"candidate used a different reference: {path}")
        if report.get("reference", {}).get("sha256") != sha256_file(reference_path):
            raise ReleaseContractError(f"candidate reference SHA-256 differs: {path}")
        requests = report.get("requests")
        if type(requests) is not list or len(requests) != MEASURED_REQUESTS:
            raise ReleaseContractError(f"correctness request count drifted: {path}")
        engine = report.get("engine") or {}
        if (
            engine.get("all_passed") is not True
            or engine.get("stable_output") is not True
            or len(engine.get("requests", [])) != MEASURED_REQUESTS
        ):
            raise ReleaseContractError(f"SGLang Engine stability is not accepted: {path}")
        identities.add(report["reference"]["identity"])
        for request in requests:
            if (
                request.get("passed") is not True
                or request.get("exact_output_sequence") is not True
                or request.get("output_token_ids") != expected_output_ids
                or request.get("output_sequence_sha256")
                != canonical_json_sha256(expected_output_ids)
                or type(request.get("steps")) is not list
                or len(request["steps"]) != 64
                or any(
                    step.get("passed") is not True
                    or type(step.get("checks")) is not dict
                    or len(step["checks"]) != 7
                    or not all(value is True for value in step["checks"].values())
                    for step in request["steps"]
                )
            ):
                raise ReleaseContractError(f"request failed in {path}")
            execution = request.get("compilation_execution", {})
            if execution.get("effective") is not True:
                raise ReleaseContractError(
                    f"executed Inductor-to-PyPTO artifacts were not proven in {path}"
                )
            inductor_calls.append(int(execution["inductor_compute_calls"]))
            handwritten_calls.append(int(execution["handwritten_compute_calls"]))
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
            inputs.extend(
                (
                    _bound_artifact(
                        path,
                        request.get("coverage_path"),
                        request.get("coverage_sha256"),
                        "correctness-coverage",
                    ),
                    _bound_artifact(
                        path,
                        request.get("trace_path"),
                        request.get("trace_sha256"),
                        "correctness-trace",
                    ),
                )
            )
        for engine_request in engine["requests"]:
            if (
                engine_request.get("passed") is not True
                or engine_request.get("exact_output_sequence") is not True
                or engine_request.get("output_token_ids") != expected_output_ids
                or engine_request.get("output_sequence_sha256")
                != canonical_json_sha256(expected_output_ids)
            ):
                raise ReleaseContractError(f"SGLang Engine request failed in {path}")
        collector = report.get("collector_stats")
        if not isinstance(collector, dict) or int(collector.get("dropped_records", -1)) != 0:
            raise ReleaseContractError(f"correctness CUPTI records were dropped: {path}")
        inputs.extend([_input(path, "correctness:candidate"), *controller_inputs])
    if (
        len(identities) != 1
        or next(iter(identities)) != reference.get("reference_identity")
        or len(sequences) != 1
    ):
        raise ReleaseContractError("correctness starts did not share one stable result")
    return (
        {
            "fresh_starts": len(reports),
            "accepted_requests": len(reports) * MEASURED_REQUESTS,
            "output_tokens_per_request": 64,
            "unique_output_sequences": 1,
            "output_sequence_sha256": next(iter(sequences)),
            "reference_identity": next(iter(identities)),
            "coverage_calls": distribution(coverage_calls),
            "inductor_compute_calls": distribution(inductor_calls),
            "handwritten_compute_calls": distribution(handwritten_calls),
        },
        inputs,
    )


def _matrix_profile_paths(
    path: Path,
) -> tuple[dict[str, list[Path]], Path, list[dict[str, str]]]:
    summary_path = _evidence_path(path)
    summary = read_json(summary_path)
    runs = summary.get("runs")
    if (
        summary.get("status") != "complete"
        or summary.get("kind") != "qwen35-9b-profile-matrix-control"
        or type(runs) is not list
        or len(runs) != len(PROFILE_SCHEDULE)
        or [item.get("lane") for item in runs] != list(PROFILE_SCHEDULE)
        or any(item.get("return_code") != 0 for item in runs)
    ):
        raise ReleaseContractError("profile matrix is not complete")
    selected: dict[str, list[Path]] = defaultdict(list)
    for item in runs:
        selected[str(item["lane"])].append(_evidence_path(str(item["report"])))
    reconciliation_path = _evidence_path(str(summary.get("reconciliation")))
    return dict(selected), reconciliation_path, [_input(summary_path, "profile-matrix")]


def _load_profiles(
    paths: dict[str, list[Path]],
    reconciliation_path: Path,
    performance_reports: dict[str, list[dict[str, object]]],
    audit: IdentityAudit,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    if set(paths) != set(LANES) or any(len(items) != 3 for items in paths.values()):
        raise ReleaseContractError("profile rendering requires three starts per lane")
    reports: dict[str, list[dict[str, object]]] = {}
    inputs = []
    for lane in LANES:
        reports[lane] = []
        for path in paths[lane]:
            report = read_json(path)
            if (
                report.get("status") != "complete"
                or report.get("kind") != "qwen35-9b-logical-phase-profile"
                or report.get("lane") != lane
                or report.get("workload") != workload_record()
                or int(report.get("profile_requests", 0)) != 5
                or int(report.get("collector_stats", {}).get("dropped_records", -1))
                != 0
            ):
                raise ReleaseContractError(f"profile report is not accepted: {path}")
            expected_profile = _PROFILE_FOR_LANE[lane]
            controller_inputs, controller_gpu = _controller_evidence(path, expected_profile)
            audit.add(
                report,
                report_path=path,
                expected_profile=expected_profile,
                controller_gpu=controller_gpu,
            )
            reports[lane].append(report)
            raw_trace = report.get("raw_trace")
            if not isinstance(raw_trace, dict):
                raise ReleaseContractError(f"profile raw trace is missing: {path}")
            inputs.extend(
                [
                    _input(path, f"profile:{lane}"),
                    _bound_artifact(
                        path,
                        raw_trace.get("path"),
                        raw_trace.get("sha256"),
                        f"profile-trace:{lane}",
                    ),
                    *controller_inputs,
                ]
            )
    observed = read_json(reconciliation_path)
    expected = reconcile(reports, performance_reports)
    observed_comparable = copy.deepcopy(observed)
    observed_comparable.pop("inputs", None)
    if observed_comparable != expected:
        raise ReleaseContractError(
            "profile reconciliation does not reproduce from selected raw reports"
        )
    inputs.append(_input(reconciliation_path, "profile-reconciliation"))
    return observed_comparable, inputs


def _load_operator(
    path: Path, audit: IdentityAudit
) -> tuple[dict[str, object], list[dict[str, str]]]:
    summary_path = _evidence_path(path)
    summary = read_json(summary_path)
    stages = summary.get("stages")
    if (
        summary.get("status") != "complete"
        or summary.get("kind") != "pypto-release-operator-regression-control"
        or type(stages) is not list
        or [item.get("stage") for item in stages] != ["structure", "gpu"]
        or any(item.get("return_code") != 0 for item in stages)
    ):
        raise ReleaseContractError("operator regression summary is not complete")
    inputs = [_input(summary_path, "operator-control")]
    reports = {}
    for item in stages:
        stage = str(item["stage"])
        report_path = _evidence_path(str(item.get("report")))
        report = read_json(report_path)
        reports[stage] = report
        if stage == "structure":
            if (
                report.get("kind") != "pypto-release-operator-structure"
                or report.get("status") != "complete"
                or report.get("return_code") != 0
                or report.get("jobs") != CPU_JOBS
            ):
                raise ReleaseContractError("operator structure report is not accepted")
            controller_inputs = _cpu_controller_evidence(report_path)
            controller_gpu = None
        else:
            if (
                report.get("kind") != "pypto-release-operator-numerical"
                or report.get("status") != "complete"
                or report.get("all_correct") is not True
                or not report.get("suites")
                or any(suite.get("passed") is not True for suite in report["suites"])
            ):
                raise ReleaseContractError("operator numerical report is not accepted")
            controller_inputs, controller_gpu = _controller_evidence(report_path, "pypto")
            dso = report.get("dso")
            packages = report.get("evidence_identity", {}).get("candidate_packages", {})
            if (
                not isinstance(dso, dict)
                or dso.get("sha256") != packages.get("dso", {}).get("sha256")
            ):
                raise ReleaseContractError("operator DSO differs from release identity")
        audit.add(
            report,
            report_path=report_path,
            expected_profile="pypto",
            controller_gpu=controller_gpu,
        )
        if stage == "gpu":
            for suite in report["suites"]:
                inputs.append(
                    _bound_artifact(
                        report_path,
                        suite.get("result"),
                        suite.get("result_sha256"),
                        f"operator-result:{suite.get('suite_id')}",
                    )
                )
        inputs.extend([_input(report_path, f"operator:{stage}"), *controller_inputs])
    numerical = reports["gpu"]
    return (
        {
            "structure": {
                "jobs": reports["structure"]["jobs"],
                "return_code": reports["structure"]["return_code"],
            },
            "suite_count": len(numerical["suites"]),
            "case_count": sum(
                int(suite.get("case_count", 0)) for suite in numerical["suites"]
            ),
            "suites": [
                {
                    "source": suite["source"],
                    "case_count": int(suite.get("case_count", 0)),
                    "passed": suite["passed"],
                }
                for suite in numerical["suites"]
            ],
            "all_correct": True,
        },
        inputs,
    )


def _load_operator_performance(
    path: Path, audit: IdentityAudit
) -> tuple[dict[str, object], list[dict[str, str]]]:
    summary_path = _evidence_path(path)
    summary = read_json(summary_path)
    runs = summary.get("runs")
    if (
        summary.get("status") != "complete"
        or summary.get("kind")
        != "qwen35-swiglu-operator-performance-matrix-control"
        or type(runs) is not list
        or len(runs) != len(OPERATOR_SCHEDULE)
        or [item.get("lane") for item in runs] != list(OPERATOR_SCHEDULE)
        or any(item.get("return_code") != 0 for item in runs)
    ):
        raise ReleaseContractError("operator performance matrix is not complete")
    grouped: dict[str, list[dict[str, object]]] = {
        lane: [] for lane in OPERATOR_LANES
    }
    inputs = [_input(summary_path, "operator-performance-matrix")]
    for item in runs:
        lane = str(item["lane"])
        report_path = _evidence_path(str(item["report"]))
        report = read_json(report_path)
        if (
            report.get("status") != "complete"
            or report.get("kind") != "qwen35-swiglu-operator-performance-only"
            or report.get("lane") != lane
        ):
            raise ReleaseContractError(
                f"operator performance report is not accepted: {report_path}"
            )
        profile = "pypto" if lane == "pypto" else "baseline"
        controller_inputs, controller_gpu = _controller_evidence(report_path, profile)
        audit.add(
            report,
            report_path=report_path,
            expected_profile=profile,
            controller_gpu=controller_gpu,
        )
        resources = report.get("resources")
        if not isinstance(resources, dict):
            raise ReleaseContractError(
                f"operator performance resources are missing: {report_path}"
            )
        grouped[lane].append(report)
        inputs.extend(
            [
                _input(report_path, f"operator-performance:{lane}"),
                _bound_artifact(
                    report_path,
                    resources.get("path"),
                    resources.get("sha256"),
                    f"operator-performance-resources:{lane}",
                ),
                *controller_inputs,
            ]
        )
    reproduced = summarize_operator_fresh_starts(grouped)
    aggregation_path = _evidence_path(str(summary.get("aggregation")))
    if read_json(aggregation_path) != reproduced:
        raise ReleaseContractError(
            "operator performance aggregation does not reproduce from raw reports"
        )
    inputs.append(_input(aggregation_path, "operator-performance-aggregation"))
    return reproduced, inputs


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


def _operator_markdown(operator: dict[str, object]) -> str:
    rows = ["| Suite | Cases | Result |", "|---|---:|---|"]
    for item in operator["suites"]:
        rows.append(
            f"| `{item['source']}` | {item['case_count']} | {'PASS' if item['passed'] else 'FAIL'} |"
        )
    rows.append(
        f"| **Total** | **{operator['case_count']}** | **{'PASS' if operator['all_correct'] else 'FAIL'}** |"
    )
    return "\n".join(rows) + "\n"


def _coverage_markdown(correctness: dict[str, object]) -> str:
    return (
        "| Coverage | Total compute calls/request p50 | Inductor calls/request p50 | Handwritten calls/request p50 | Fallback / unknown |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"| 100% | {correctness['coverage_calls']['p50']:.0f} | "
        f"{correctness['inductor_compute_calls']['p50']:.0f} | "
        f"{correctness['handwritten_compute_calls']['p50']:.0f} | 0 |\n"
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


def _operator_performance_markdown(performance: dict[str, object]) -> str:
    rows = [
        "| SwiGLU case | PyPTO ms/call p50 | Stock ms/call p50 | PyPTO latency / stock | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for case, comparison in performance["comparisons"].items():
        candidate = performance["lanes"]["pypto"]["cases"][case][
            "latency_ms_per_call"
        ]["p50"]
        baseline = performance["lanes"]["sglang-matched"]["cases"][case][
            "latency_ms_per_call"
        ]["p50"]
        interval = comparison["median_ratio_bootstrap_95ci_percent"]
        rows.append(
            f"| {case} | {candidate:.6f} | {baseline:.6f} | "
            f"{comparison['pypto_latency_percent_of_stock']:.2f}% | "
            f"[{interval['lower']:.2f}%, {interval['upper']:.2f}%] |"
        )
    return "\n".join(rows) + "\n"


def _summary_markdown(
    correctness: dict[str, object],
    operator: dict[str, object],
    performance: dict[str, object],
    *,
    language: str,
) -> str:
    matched = performance["pypto_percent_of_stock"]["sglang-matched"]
    optimized = performance["pypto_percent_of_stock"]["sglang-optimized"]
    if language == "zh":
        return (
            "| 项目 | Qwen3.5-9B release-v1 |\n"
            "|---|---:|\n"
            f"| 64-token greedy 正确性 | PASS（{correctness['fresh_starts']} 次 fresh start，"
            f"{correctness['accepted_requests']} 个请求） |\n"
            "| model-forward PyPTO compute coverage | 100% |\n"
            f"| 算子 regression | PASS（{operator['suite_count']} suites，"
            f"{operator['case_count']} cases） |\n"
            f"| PyPTO / matched SGLang | {matched:.2f}% |\n"
            f"| PyPTO / optimized SGLang | {optimized:.2f}% |\n"
            "| 性能瓶颈归因 | CUPTI/NVTX reconciliation complete |\n"
        )
    return (
        "| Item | Qwen3.5-9B release-v1 |\n"
        "|---|---:|\n"
        f"| 64-token greedy correctness | PASS ({correctness['fresh_starts']} fresh starts, "
        f"{correctness['accepted_requests']} requests) |\n"
        "| model-forward PyPTO compute coverage | 100% |\n"
        f"| Operator regression | PASS ({operator['suite_count']} suites, "
        f"{operator['case_count']} cases) |\n"
        f"| PyPTO / matched SGLang | {matched:.2f}% |\n"
        f"| PyPTO / optimized SGLang | {optimized:.2f}% |\n"
        "| Performance bottleneck attribution | CUPTI/NVTX reconciliation complete |\n"
    )


def _conclusion_markdown(performance: dict[str, object]) -> str:
    matched = performance["pypto_percent_of_stock"]["sglang-matched"]
    optimized = performance["pypto_percent_of_stock"]["sglang-optimized"]
    return (
        "在固定的 19+64、greedy、并发 1 workload 下，Qwen3.5-9B 已通过三次 fresh "
        "start 的多 token 正确性与 100% model-forward PyPTO compute coverage 门。"
        f"PyPTO 的 median output throughput 分别达到 matched 和 optimized stock "
        f"SGLang 的 {matched:.2f}% 与 {optimized:.2f}%；阶段差距和未解释残差见上表。\n"
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _require_sanitized(value: object, label: str = "summary") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _require_sanitized(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_sanitized(item, f"{label}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if (
            value.startswith("/")
            or re.match(r"^[A-Za-z]:[\\/]", value)
            or value.startswith("file://")
            or str(ROOT) in value
            or "\x00" in value
        ):
            raise ReleaseContractError(f"unsanitized absolute/control value at {label}")
        if any(token in lowered for token in _PLACEHOLDERS):
            raise ReleaseContractError(f"placeholder survived at {label}: {value!r}")


def _write_fragments(
    output: Path,
    summary_path: Path,
    fragments: dict[str, str],
) -> Path:
    fragment_dir = output / "markers"
    records = {}
    for name, text in fragments.items():
        lowered = text.lower()
        if not text.strip() or any(token in lowered for token in _PLACEHOLDERS):
            raise ReleaseContractError(f"marker fragment is empty or placeholder: {name}")
        filename = name.lower().replace("_", "-") + ".md"
        path = fragment_dir / filename
        _atomic_text(path, text.rstrip() + "\n")
        records[name] = {
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-release-marker-fragments",
        "release_summary": {
            "path": summary_path.relative_to(output).as_posix(),
            "sha256": sha256_file(summary_path),
        },
        "fragments": records,
        "status": "complete",
    }
    manifest_path = output / "marker-fragments.json"
    atomic_json(manifest_path, manifest)
    return manifest_path


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
    value.add_argument(
        "--profile",
        action="append",
        default=[],
        help="lane=report; required with --reconciliation",
    )
    value.add_argument("--operator-summary", type=Path, required=True)
    value.add_argument("--operator-performance-matrix", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def _unique_inputs(inputs: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    observed = set()
    for item in inputs:
        key = (item["path"], item["sha256"])
        if key not in observed:
            observed.add(key)
            result.append(item)
    return sorted(result, key=lambda item: (item["role"], item["path"]))


def main() -> int:
    args = parser().parse_args()
    output = require_path_below_runs(ROOT, args.output_dir)
    if output.parent != (ROOT / "runs").resolve() or not _OUTPUT_NAME.fullmatch(output.name):
        raise ReleaseContractError(
            "output must be a direct runs/release-results-<unique-id> directory"
        )
    if output.exists() and any(output.iterdir()):
        raise ReleaseContractError(f"refusing to overwrite release results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit = IdentityAudit()
    inputs: list[dict[str, str]] = []

    if args.performance_matrix is not None:
        performance_paths, performance_aggregation_path, matrix_inputs = _matrix_performance_paths(
            args.performance_matrix
        )
        inputs.extend(matrix_inputs)
    else:
        performance_paths = _selectors(args.performance, "performance")
        performance_aggregation_path = None
    performance, performance_inputs, performance_reports = _load_performance(
        performance_paths, audit
    )
    if performance_aggregation_path is not None:
        reproduced_performance = copy.deepcopy(performance)
        reproduced_performance.pop("pypto_percent_of_stock")
        for lane in LANES:
            for field in ("resolved_backends", "memory_qualification", "inputs"):
                reproduced_performance["lanes"][lane].pop(field)
        if read_json(performance_aggregation_path) != reproduced_performance:
            raise ReleaseContractError(
                "performance aggregation does not reproduce from selected raw reports"
            )
    inputs.extend(performance_inputs)

    if args.correctness_summary is not None:
        reference_path, correctness_paths, control_inputs = _summary_correctness_paths(
            args.correctness_summary
        )
        inputs.extend(control_inputs)
    else:
        correctness_paths = [_evidence_path(path) for path in args.correctness_report]
        reference_path = None
    if reference_path is None:
        reference_path = _reference_from_candidates(correctness_paths)
    correctness, correctness_inputs = _load_correctness(
        reference_path, correctness_paths, audit
    )
    inputs.extend(correctness_inputs)

    if args.profile_matrix is not None:
        profile_paths, reconciliation_path, profile_control_inputs = (
            _matrix_profile_paths(args.profile_matrix)
        )
        if args.profile:
            raise ReleaseContractError("--profile cannot accompany --profile-matrix")
        inputs.extend(profile_control_inputs)
    else:
        if not args.profile:
            raise ReleaseContractError(
                "--reconciliation requires every raw --profile lane=report"
            )
        profile_paths = _selectors(args.profile, "profile")
        reconciliation_path = _evidence_path(args.reconciliation)
    reconciliation, profile_inputs = _load_profiles(
        profile_paths,
        reconciliation_path,
        performance_reports,
        audit,
    )
    inputs.extend(profile_inputs)

    operator, operator_inputs = _load_operator(args.operator_summary, audit)
    inputs.extend(operator_inputs)
    operator_performance, operator_performance_inputs = _load_operator_performance(
        args.operator_performance_matrix, audit
    )
    inputs.extend(operator_performance_inputs)
    release_identity = audit.result()
    summary = {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-9b-release-results",
        "workload": workload_record(),
        "release_identity": release_identity,
        "operator_correctness": operator,
        "operator_performance": operator_performance,
        "model_correctness": correctness,
        "performance": performance,
        "profile_reconciliation": reconciliation,
        "inputs": _unique_inputs(inputs),
        "status": "complete",
    }
    _require_sanitized(summary)
    summary_path = output / "release-summary.json"
    atomic_json(summary_path, summary)
    fragments = {
        "SUMMARY_ZH": _summary_markdown(
            correctness, operator, performance, language="zh"
        ),
        "SUMMARY_EN": _summary_markdown(
            correctness, operator, performance, language="en"
        ),
        "OPERATOR_CORRECTNESS_ZH": _operator_markdown(operator),
        "MODEL_CORRECTNESS_ZH": _correctness_markdown(correctness),
        "COVERAGE_ZH": _coverage_markdown(correctness),
        "PERFORMANCE_ZH": _performance_markdown(performance),
        "BREAKDOWN_ZH": (
            _operator_performance_markdown(operator_performance)
            + "\n"
            + _profile_markdown(reconciliation)
        ),
        "CONCLUSION_ZH": _conclusion_markdown(performance),
    }
    manifest_path = _write_fragments(output, summary_path, fragments)
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": _relative(summary_path),
                "marker_fragments": _relative(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
