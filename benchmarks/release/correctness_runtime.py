"""Independent multi-token reference and PyPTO numerical acceptance."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback

from .cupti_overlay import activate_overlay
from .evidence_identity import collect_run_identity
from .lanes import (
    prepare_worker_environment,
    resolved_backend_record,
    server_kwargs,
    validate_resolved_backends,
)
from .workload import (
    MEASURED_REQUESTS,
    OUTPUT_TOKENS,
    PROMPT,
    PROMPT_TOKEN_IDS,
    Qwen35ModelSpec,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    canonical_json_sha256,
    model_revision,
    read_json,
    resolve_qwen35_model_spec,
    sha256_file,
    validate_workload,
    workload_record,
)


MAX_TRACE_ATTEMPTS = 10
CUPTI_BUFFER_COMPLETION_TIMEOUT_SECONDS = 1.0
ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = {
    "cosine_similarity_min": 0.98,
    "max_abs_error_max": 3.0,
    "max_relative_error_max": 2.1,
    "max_relative_error_reference_floor": 1.0,
    "mean_abs_error_max": 0.45,
    "minimum_candidate_top1_margin": 0.05,
    "top5_token_overlap_min": 3,
    "exact_greedy_token_ids": True,
}
_COMPILE_CACHE_DISPOSITIONS = frozenset(
    {
        "Uncached",
        "CacheHit",
        "CompiledAndPublished",
        "CompiledAndValidatedExisting",
    }
)
_COMPILE_SNAPSHOT_FIELDS = frozenset(
    {
        "source_node",
        "provider",
        "cache_key",
        "build_spec_identity",
        "artifact_identity",
        "disposition",
    }
)


def _model_record(model_path: Path, model_spec: Qwen35ModelSpec) -> dict[str, object]:
    config = model_path / "config.json"
    if not config.is_file():
        raise ReleaseContractError(f"model config is missing: {config}")
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise ReleaseContractError(f"model has no safetensors: {model_path}")
    return {
        "path": str(model_path),
        **model_spec.record(),
        "revision": model_revision(model_path),
        "config_sha256": sha256_file(config),
        "shards": [
            {"name": path.name, "bytes": path.stat().st_size} for path in shards
        ],
    }


def _load_runner(
    lane: str,
    model_path: Path,
    optimized_memory_mode: str = "zero-offload",
):
    prepare_worker_environment(lane)
    import torch
    from sglang.benchmark import one_batch
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import PortArgs, ServerArgs

    load_plugins()
    requested = server_kwargs(lane, model_path, optimized_memory_mode)
    args = ServerArgs(**requested)
    _set_envs_and_config(args)
    initialize_moe_config(args)
    initialize_fp8_gemm_config(args)
    initialize_fp4_gemm_config(args)
    resolved = resolved_backend_record(args)
    validate_resolved_backends(lane, resolved)
    ports = PortArgs.init_new(args)
    one_batch.get_tokenizer = lambda *_args, **_kwargs: None
    runner, _tokenizer = one_batch.load_model(args, ports, gpu_id=0, tp_rank=0)
    return torch, one_batch, runner, requested, resolved


def _generate(torch, one_batch, runner, monitor=None):
    windows: list[dict[str, object]] = []
    torch_runner = runner.torch_runner
    original_forward = torch_runner.forward

    if monitor is not None:

        def traced_forward(*args, **kwargs):
            window = None
            monitor.begin_trace_window()
            try:
                return original_forward(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                completed_before = int(monitor.stats()["buffers_completed"])
                monitor.flush(forced=True)
                deadline = time.monotonic() + CUPTI_BUFFER_COMPLETION_TIMEOUT_SECONDS
                while (
                    int(monitor.stats()["buffers_completed"]) <= completed_before
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.001)
                window = monitor.end_trace_window()
                windows.append(window)

        torch_runner.forward = traced_forward

    batch = None
    cpu_logits = []
    output_ids: list[int] = []
    try:
        reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
            1, len(PROMPT_TOKEN_IDS), [list(PROMPT_TOKEN_IDS)]
        )
        next_ids, logits, batch = runner.extend(reqs)
        runner.synchronize()
        output_ids.append(int(next_ids.cpu()[0]))
        cpu_logits.append(logits.detach().float().cpu().contiguous())
        for _ in range(OUTPUT_TOKENS - 1):
            next_ids, logits = runner.decode(next_ids, batch)
            runner.synchronize()
            output_ids.append(int(next_ids.cpu()[0]))
            cpu_logits.append(logits.detach().float().cpu().contiguous())
        return output_ids, torch.cat(cpu_logits, dim=0), windows
    finally:
        torch_runner.forward = original_forward
        if batch is not None:
            runner.cleanup(batch)
        runner.clear()


def _shutdown_runner() -> None:
    try:
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except (ImportError, RuntimeError):
        pass


def _tensor_raw_sha256(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _token_sequence_mismatch(
    expected: list[int], observed: list[int]
) -> dict[str, int | None]:
    mismatch = next(
        (
            index
            for index, (expected_token, observed_token) in enumerate(
                zip(expected, observed, strict=False)
            )
            if expected_token != observed_token
        ),
        min(len(expected), len(observed)) if len(expected) != len(observed) else None,
    )
    return {
        "first_mismatch_step": mismatch,
        "expected_token_id": (
            expected[mismatch]
            if mismatch is not None and mismatch < len(expected)
            else None
        ),
        "observed_token_id": (
            observed[mismatch]
            if mismatch is not None and mismatch < len(observed)
            else None
        ),
    }


def _is_lowercase_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _compile_cache_evidence(
    snapshot: object | None = None,
) -> dict[str, object]:
    """Validate main-process compile records without changing coverage identity."""

    if snapshot is None:
        from pypto_kernels._boot import artifact_compile_snapshot

        snapshot = artifact_compile_snapshot()
    if type(snapshot) is not list or not snapshot:
        raise ReleaseContractError(
            "candidate main process produced no PyPTO compile snapshot"
        )
    records: list[dict[str, str]] = []
    disposition_counts = {
        disposition: 0 for disposition in sorted(_COMPILE_CACHE_DISPOSITIONS)
    }
    for index, raw in enumerate(snapshot):
        if type(raw) is not dict or set(raw) != _COMPILE_SNAPSHOT_FIELDS:
            raise ReleaseContractError(
                f"compile snapshot record {index} has an invalid schema"
            )
        if any(
            type(raw[field]) is not str or not str(raw[field]).strip()
            for field in ("source_node", "provider")
        ):
            raise ReleaseContractError(
                f"compile snapshot record {index} has empty provenance"
            )
        for field in (
            "cache_key",
            "build_spec_identity",
            "artifact_identity",
        ):
            if not _is_lowercase_sha256(raw[field]):
                raise ReleaseContractError(
                    f"compile snapshot record {index} has invalid {field}"
                )
        disposition = raw["disposition"]
        if disposition not in _COMPILE_CACHE_DISPOSITIONS:
            raise ReleaseContractError(
                f"compile snapshot record {index} has unknown disposition "
                f"{disposition!r}"
            )
        disposition_counts[disposition] += 1
        records.append(dict(raw))
    return {
        "scope": "candidate-main-process",
        "records": records,
        "record_count": len(records),
        "disposition_counts": disposition_counts,
        "cache_hit_observed": disposition_counts["CacheHit"] > 0,
        "cache_hit_required": False,
        "cache_hit_requirement_reason": (
            "SGLang Engine warmup runs in another process and may compile a "
            "different static graph set; cross-process reuse is reported but "
            "is not a correctness acceptance gate"
        ),
        "coverage_identity_includes_snapshot": False,
    }


def _run_engine_sequences(
    model_path: Path,
    expected_output_ids: list[int],
    run_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    prepare_worker_environment("pypto")
    import sglang as sgl

    requested = server_kwargs("pypto", model_path)
    progress_path = run_dir / "qwen35-engine-progress.json"

    def publish_progress(stage: str, request_index: int | None = None) -> None:
        payload: dict[str, object] = {
            "schema": SCHEMA_VERSION,
            "kind": "qwen35-engine-correctness-progress",
            "stage": stage,
        }
        if request_index is not None:
            payload["request_index"] = request_index
        atomic_json(progress_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)

    engine = None
    try:
        publish_progress("engine-construction-start")
        engine = sgl.Engine(**requested)
        publish_progress("engine-construction-complete")
        resolved = resolved_backend_record(engine.server_args)
        validate_resolved_backends("pypto", resolved)

        def generate(request_index: int) -> list[int]:
            response = engine.generate(
                input_ids=list(PROMPT_TOKEN_IDS),
                sampling_params={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": OUTPUT_TOKENS,
                    "ignore_eos": True,
                },
                rid=f"release-correctness-{request_index}",
            )
            if (
                type(response) is not dict
                or type(response.get("output_ids")) is not list
            ):
                raise ReleaseContractError("SGLang Engine returned no output token IDs")
            return [int(value) for value in response["output_ids"]]

        publish_progress("warmup-start", -1)
        warmup = generate(-1)
        publish_progress("warmup-complete", -1)
        if len(warmup) != OUTPUT_TOKENS:
            raise ReleaseContractError(
                "SGLang Engine warmup did not generate 64 tokens"
            )
        requests = []
        for request_index in range(MEASURED_REQUESTS):
            publish_progress("request-start", request_index)
            output_ids = generate(request_index)
            publish_progress("request-complete", request_index)
            exact = output_ids == expected_output_ids
            requests.append(
                {
                    "request_index": request_index,
                    "output_token_ids": output_ids,
                    "output_sequence_sha256": canonical_json_sha256(output_ids),
                    "exact_output_sequence": exact,
                    **_token_sequence_mismatch(expected_output_ids, output_ids),
                    "passed": exact,
                }
            )
        return requests, requested, resolved
    finally:
        if engine is not None:
            publish_progress("shutdown-start")
            engine.shutdown()
            publish_progress("shutdown-complete")


def _engine_sequence_process(
    model_path: str,
    expected_output_ids: list[int],
    run_dir: str,
    sender,
) -> None:
    try:
        sender.send(
            {
                "ok": True,
                "value": _run_engine_sequences(
                    Path(model_path), expected_output_ids, Path(run_dir)
                ),
            }
        )
    except BaseException as error:
        sender.send(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        sender.close()


def _run_engine_sequences_isolated(
    model_path: Path,
    expected_output_ids: list[int],
    run_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_engine_sequence_process,
        args=(str(model_path), expected_output_ids, str(run_dir), sender),
        daemon=False,
    )
    process.start()
    sender.close()
    try:
        payload = receiver.recv()
    except EOFError as error:
        process.join()
        raise ReleaseContractError(
            "SGLang Engine correctness child exited without a result: "
            f"exit_code={process.exitcode}"
        ) from error
    finally:
        receiver.close()
    process.join()
    if process.exitcode != 0:
        raise ReleaseContractError(
            f"SGLang Engine correctness child failed: exit_code={process.exitcode}"
        )
    if payload.get("ok") is not True:
        raise ReleaseContractError(
            "SGLang Engine correctness child failed: "
            f"{payload.get('error_type')}: {payload.get('error')}\n"
            f"{payload.get('traceback')}"
        )
    value = payload.get("value")
    if not isinstance(value, tuple) or len(value) != 3:
        raise ReleaseContractError(
            "SGLang Engine correctness child returned an invalid payload"
        )
    return value


def run_reference(model_path: Path, run_id: str, run_dir: Path) -> int:
    lane = "sglang-matched"
    model_path = model_path.resolve(strict=True)
    model_spec = resolve_qwen35_model_spec(ROOT, model_path)
    report_path = run_dir / f"{model_spec.report_stem}-reference.json"
    tensor_path = run_dir / f"{model_spec.report_stem}-reference-logits.pt"
    model = _model_record(model_path, model_spec)
    workload = workload_record(model_spec)
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": f"{model_spec.report_stem}-multitoken-reference",
        "lane": lane,
        "run_id": run_id,
        "workload": workload,
        "entrypoint": "sglang.benchmark.one_batch ModelRunner",
        "thresholds": THRESHOLDS,
        "model": model,
        "status": "starting",
    }
    runner = None
    try:
        torch, one_batch, runner, requested, resolved = _load_runner(lane, model_path)
        report["requested_server_config"] = requested
        report["resolved_backends"] = resolved
        warm_ids, _warm_logits, _windows = _generate(torch, one_batch, runner)
        if len(warm_ids) != OUTPUT_TOKENS:
            raise ReleaseContractError("reference warmup did not complete")
        output_ids, logits, _windows = _generate(torch, one_batch, runner)
        if list(logits.shape[:1]) != [OUTPUT_TOKENS]:
            raise ReleaseContractError("reference tensor does not contain 64 steps")
        torch.save(logits, tensor_path)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        encoded_prompt = tokenizer.encode(PROMPT, add_special_tokens=False)
        if encoded_prompt != list(PROMPT_TOKEN_IDS):
            raise ReleaseContractError(
                "tokenizer revision does not encode the frozen 19-token prompt"
            )
        evidence_identity = collect_run_identity(ROOT, "baseline", model_path)
        report.update(
            {
                "status": "complete",
                "output_token_ids": output_ids,
                "output_text": tokenizer.decode(output_ids, skip_special_tokens=False),
                "logits": {
                    "path": str(tensor_path),
                    "file_sha256": sha256_file(tensor_path),
                    "raw_sha256": _tensor_raw_sha256(logits),
                    "shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                },
                "reference_identity": canonical_json_sha256(
                    {
                        "model": model,
                        "workload": workload,
                        "thresholds": THRESHOLDS,
                        "output_token_ids": output_ids,
                        "logits_raw_sha256": _tensor_raw_sha256(logits),
                    }
                ),
                "evidence_identity": evidence_identity,
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
        if runner is not None:
            _shutdown_runner()
        atomic_json(report_path, report)
        print(
            json.dumps(
                {
                    "kind": report["kind"],
                    "status": report["status"],
                    "run_id": run_id,
                    "report": str(report_path),
                    "output_token_count": len(report.get("output_token_ids", [])),
                    "output_text": report.get("output_text"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return return_code


def _step_parity(torch, expected_ids, reference, candidate) -> dict[str, object]:
    difference = (candidate - reference).abs()
    floor = float(THRESHOLDS["max_relative_error_reference_floor"])
    mask = reference.abs() >= floor
    relative = difference[mask] / reference.abs()[mask]
    cosine = torch.nn.functional.cosine_similarity(
        candidate.double(), reference.double(), dim=0
    )
    reference_values, reference_ids = torch.topk(reference, 5)
    candidate_values, candidate_ids = torch.topk(candidate, 5)
    margin = float(candidate_values[0] - candidate_values[1])
    overlap = len(set(reference_ids.tolist()) & set(candidate_ids.tolist()))
    metrics = {
        "candidate_top1_margin": margin,
        "cosine_similarity": float(cosine),
        "max_abs_error": float(difference.max()),
        "max_relative_error": float(relative.max()),
        "mean_abs_error": float(difference.mean()),
        "top5_token_overlap": overlap,
    }
    checks = {
        "candidate_top1_margin": margin
        >= float(THRESHOLDS["minimum_candidate_top1_margin"]),
        "cosine_similarity": metrics["cosine_similarity"]
        >= float(THRESHOLDS["cosine_similarity_min"]),
        "exact_greedy_token_id": candidate_ids[:1].tolist() == expected_ids,
        "max_abs_error": metrics["max_abs_error"]
        <= float(THRESHOLDS["max_abs_error_max"]),
        "max_relative_error": metrics["max_relative_error"]
        <= float(THRESHOLDS["max_relative_error_max"]),
        "mean_abs_error": metrics["mean_abs_error"]
        <= float(THRESHOLDS["mean_abs_error_max"]),
        "top5_token_overlap": overlap >= int(THRESHOLDS["top5_token_overlap_min"]),
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def _device_fingerprint(torch) -> str:
    properties = torch.cuda.get_device_properties(0)
    return canonical_json_sha256(
        {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory": properties.total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    )


def _merge_coverage_event(event_aggregates: dict[str, object], event) -> None:
    previous = event_aggregates.get(event.activity_id)
    if previous is None:
        event_aggregates[event.activity_id] = event
        return
    if (
        replace(
            event,
            call_count=previous.call_count,
            gpu_time_ns=previous.gpu_time_ns,
        )
        != previous
    ):
        raise ReleaseContractError("conflicting activity identity across trace windows")
    event_aggregates[event.activity_id] = replace(
        previous,
        call_count=previous.call_count + event.call_count,
        gpu_time_ns=previous.gpu_time_ns + event.gpu_time_ns,
    )


def _coverage_request(
    *,
    torch,
    run_id: str,
    request_index: int,
    windows: list[dict[str, object]],
    monitor_stats: dict[str, object],
    model_spec: Qwen35ModelSpec,
    model_revision: str,
    run_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    from pypto_plugins.activity_trace import normalize_cupti_window
    from pypto_plugins.coverage import (
        FRAMEWORK_PROFILE,
        TRACE_COLLECTOR,
        TRACE_COLLECTOR_REVISION,
        CoverageAuditor,
        CoverageMode,
        TraceManifest,
        compute_artifact_registry_digest,
        compute_trace_digest,
    )

    artifacts = {}
    event_aggregates = {}
    dropped = int(monitor_stats.get("dropped_records", 0))
    closed_world = True
    for window in windows:
        normalized = normalize_cupti_window(window, dropped_records=dropped)
        closed_world = closed_world and bool(normalized.closed_world)
        for artifact in normalized.artifacts:
            previous = artifacts.setdefault(artifact.artifact_id, artifact)
            if previous != artifact:
                raise ReleaseContractError("conflicting artifact across trace windows")
        for event in normalized.events:
            _merge_coverage_event(event_aggregates, event)
    artifact_values = list(artifacts.values())
    events = list(event_aggregates.values())
    if not closed_world:
        raise ReleaseContractError("CUPTI request trace is not closed-world")
    manifest = TraceManifest(
        run_id=f"{run_id}:request:{request_index}",
        model_id=model_spec.model_id,
        model_revision=model_revision,
        device_fingerprint=_device_fingerprint(torch),
        collector=TRACE_COLLECTOR,
        collector_revision=TRACE_COLLECTOR_REVISION,
        framework_profile=FRAMEWORK_PROFILE,
        artifact_registry_digest=compute_artifact_registry_digest(artifact_values),
        trace_digest=compute_trace_digest(events),
        activity_count=len(events),
        closed_world=closed_world,
    )
    coverage_path = run_dir / f"coverage-request-{request_index:02d}.json"
    with CoverageAuditor(
        mode=CoverageMode.STRICT,
        report_path=coverage_path,
        manifest=manifest,
        artifacts=artifact_values,
    ) as auditor:
        for event in events:
            auditor.record(event)
        summary = auditor.finalize(event_stream_complete=True)
    coverage_payload = read_json(coverage_path)
    registry = {
        item["artifact_id"]: item
        for item in coverage_payload.get("artifact_registry", [])
    }
    inductor_artifacts = {
        artifact_id
        for artifact_id, artifact in registry.items()
        if str(artifact.get("source_node", "")).startswith("torch-inductor:")
    }
    handwritten_artifacts = {
        artifact_id
        for artifact_id, artifact in registry.items()
        if str(artifact.get("source_node", "")).startswith(
            ("pypto_kernels.", "pypto-kernels:")
        )
    }
    unknown_artifacts = sorted(
        set(registry) - inductor_artifacts - handwritten_artifacts
    )
    inductor_calls = 0
    handwritten_calls = 0
    for event in coverage_payload.get("events", []):
        provenance = event.get("provenance") or {}
        if provenance.get("artifact_id") in inductor_artifacts:
            inductor_calls += int(event.get("call_count", 0))
        if provenance.get("artifact_id") in handwritten_artifacts:
            handwritten_calls += int(event.get("call_count", 0))
    compilation_execution = {
        "inductor_artifact_count": len(inductor_artifacts),
        "inductor_compute_calls": inductor_calls,
        "handwritten_artifact_count": len(handwritten_artifacts),
        "handwritten_compute_calls": handwritten_calls,
        "unknown_artifacts": unknown_artifacts,
        "expected_calls": model_spec.expected_inductor_calls,
        "num_hidden_layers": model_spec.num_hidden_layers,
        "effective": bool(
            inductor_artifacts
            and inductor_calls == model_spec.expected_inductor_calls
            and handwritten_artifacts
            and handwritten_calls > 0
            and not unknown_artifacts
        ),
        "evidence": "CUPTI-correlated immutable PyPTO artifacts",
    }
    trace_path = run_dir / f"trace-request-{request_index:02d}.json"
    atomic_json(
        trace_path,
        {"schema": SCHEMA_VERSION, "windows": windows, "stats": monitor_stats},
    )
    return asdict(summary), {
        "coverage_path": str(coverage_path),
        "coverage_sha256": sha256_file(coverage_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "compilation_execution": compilation_execution,
    }


def run_candidate(
    model_path: Path,
    reference_report_path: Path,
    run_id: str,
    run_dir: Path,
) -> int:
    lane = "pypto"
    model_path = model_path.resolve(strict=True)
    model_spec = resolve_qwen35_model_spec(ROOT, model_path)
    reference_report_path = reference_report_path.resolve(strict=True)
    report_path = run_dir / f"{model_spec.report_stem}-correctness.json"
    model = _model_record(model_path, model_spec)
    workload = workload_record(model_spec)
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": f"{model_spec.report_stem}-multitoken-correctness",
        "lane": lane,
        "run_id": run_id,
        "workload": workload,
        "model": model,
        "entrypoint": "sglang.benchmark.one_batch ModelRunner",
        "request_count": MEASURED_REQUESTS,
        "fresh_process": True,
        "status": "starting",
    }
    runner = None
    monitor = None
    monitor_api = None
    try:
        reference_report = read_json(reference_report_path)
        if reference_report.get("status") != "complete":
            raise ReleaseContractError("reference report is not complete")
        expected_reference_kind = f"{model_spec.report_stem}-multitoken-reference"
        if reference_report.get("kind") != expected_reference_kind:
            raise ReleaseContractError(
                "reference report kind differs for selected model"
            )
        validate_workload(reference_report.get("workload", {}), model_spec)
        if reference_report.get("thresholds") != THRESHOLDS:
            raise ReleaseContractError(
                "reference thresholds differ from release policy"
            )
        reference_model = reference_report.get("model")
        if not isinstance(reference_model, dict) or any(
            reference_model.get(field) != model[field]
            for field in (
                "manifest_name",
                "model_id",
                "model_size",
                "num_hidden_layers",
                "expected_inductor_calls",
                "revision",
                "config_sha256",
            )
        ):
            raise ReleaseContractError(
                "candidate and reference model specifications differ"
            )
        logits_record = reference_report.get("logits")
        if not isinstance(logits_record, dict):
            raise ReleaseContractError("reference report has no tensor record")
        reference_tensor_path = Path(str(logits_record["path"])).resolve(strict=True)
        if sha256_file(reference_tensor_path) != logits_record.get("file_sha256"):
            raise ReleaseContractError("reference tensor file identity changed")

        expected_output_ids = list(reference_report["output_token_ids"])
        if len(expected_output_ids) != OUTPUT_TOKENS:
            raise ReleaseContractError(
                "reference output token sequence does not contain 64 steps"
            )
        engine_requests, engine_requested, engine_resolved = (
            _run_engine_sequences_isolated(model_path, expected_output_ids, run_dir)
        )
        engine_progress_path = run_dir / "qwen35-engine-progress.json"
        report["engine"] = {
            "entrypoint": "sglang.Engine offline API",
            "requested_server_config": engine_requested,
            "resolved_backends": engine_resolved,
            "requests": engine_requests,
            "all_passed": all(item["passed"] for item in engine_requests),
            "stable_output": len(
                {item["output_sequence_sha256"] for item in engine_requests}
            )
            == 1,
            "progress": {
                "path": str(engine_progress_path),
                "sha256": sha256_file(engine_progress_path),
            },
        }
        if report["engine"]["all_passed"] is not True:
            raise ReleaseContractError(
                "end-to-end SGLang Engine output differs from the frozen reference"
            )

        cupti_overlay = activate_overlay()
        report["cupti_overlay"] = cupti_overlay
        import torch
        from torch.profiler import _cupti_monitor as monitor_api

        if torch.cuda.is_initialized():
            raise ReleaseContractError("CUPTI must start before CUDA initialization")
        monitor = monitor_api.start_collection(run_dir / "cupti-monitor")
        torch, one_batch, runner, requested, resolved = _load_runner(lane, model_path)
        report["requested_server_config"] = requested
        report["resolved_backends"] = resolved
        reference = (
            torch.load(reference_tensor_path, map_location="cpu", weights_only=True)
            .float()
            .contiguous()
        )
        if list(reference.shape[:1]) != [OUTPUT_TOKENS]:
            raise ReleaseContractError("reference tensor does not contain 64 steps")
        if _tensor_raw_sha256(reference) != logits_record.get("raw_sha256"):
            raise ReleaseContractError("reference tensor payload identity changed")

        warm_ids, _warm_logits, _windows = _generate(torch, one_batch, runner)
        if len(warm_ids) != OUTPUT_TOKENS:
            raise ReleaseContractError("candidate warmup did not complete")
        torch.cuda.synchronize()
        requests = []
        for request_index in range(MEASURED_REQUESTS):
            accepted = None
            for attempt in range(1, MAX_TRACE_ATTEMPTS + 1):
                output_ids, candidate, windows = _generate(
                    torch, one_batch, runner, monitor
                )
                torch.cuda.synchronize()
                if len(windows) == OUTPUT_TOKENS and all(
                    any(event.get("kind") == "kernel" for event in window["events"])
                    for window in windows
                ):
                    accepted = (output_ids, candidate, windows, attempt)
                    break
            if accepted is None:
                raise ReleaseContractError(
                    f"request {request_index} produced no complete CUPTI trace"
                )
            output_ids, candidate, windows, attempt = accepted
            step_results = [
                _step_parity(
                    torch,
                    [expected_output_ids[step]],
                    reference[step],
                    candidate[step],
                )
                for step in range(OUTPUT_TOKENS)
            ]
            monitor_stats = monitor.stats()
            coverage, paths = _coverage_request(
                torch=torch,
                run_id=run_id,
                request_index=request_index,
                windows=windows,
                monitor_stats=monitor_stats,
                model_spec=model_spec,
                model_revision=str(model["revision"]),
                run_dir=run_dir,
            )
            exact_sequence = output_ids == expected_output_ids
            passed = bool(
                exact_sequence
                and all(item["passed"] for item in step_results)
                and coverage.get("strict_policy_passed") is True
                and paths["compilation_execution"]["effective"] is True
            )
            requests.append(
                {
                    "request_index": request_index,
                    "trace_attempts": attempt,
                    "output_token_ids": output_ids,
                    "output_sequence_sha256": canonical_json_sha256(output_ids),
                    "exact_output_sequence": exact_sequence,
                    "steps": step_results,
                    "coverage": coverage,
                    **paths,
                    "passed": passed,
                }
            )
        stats = monitor_api.stop_collection()
        monitor = None
        all_passed = all(item["passed"] for item in requests)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        if tokenizer.encode(PROMPT, add_special_tokens=False) != list(PROMPT_TOKEN_IDS):
            raise ReleaseContractError(
                "candidate tokenizer does not encode the frozen 19-token prompt"
            )
        evidence_identity = collect_run_identity(ROOT, "pypto", model_path)
        compile_cache = _compile_cache_evidence()
        report.update(
            {
                "status": "complete" if all_passed else "failed",
                "all_passed": all_passed,
                "reference": {
                    "path": str(reference_report_path),
                    "sha256": sha256_file(reference_report_path),
                    "identity": reference_report.get("reference_identity"),
                },
                "thresholds": THRESHOLDS,
                "requests": requests,
                "stable_output": len(
                    {item["output_sequence_sha256"] for item in requests}
                )
                == 1,
                "output_text": tokenizer.decode(
                    expected_output_ids, skip_special_tokens=False
                ),
                "collector_stats": stats,
                "compile_cache": compile_cache,
                "evidence_identity": evidence_identity,
            }
        )
        return_code = 0 if all_passed else 1
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "all_passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    finally:
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
                    "run_id": run_id,
                    "report": str(report_path),
                    "accepted_requests": sum(
                        bool(item.get("passed")) for item in report.get("requests", [])
                    ),
                    "output_text": report.get("output_text"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return return_code
