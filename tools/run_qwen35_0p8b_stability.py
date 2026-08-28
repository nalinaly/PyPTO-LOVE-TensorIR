#!/usr/bin/env python3
"""Run ten warm exact-prompt requests with per-request strict evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import traceback

from run_qwen35_0p8b_model_runner_smoke import (
    _device_fingerprint,
    _model_revision,
)
from run_qwen35_0p8b_pypto_smoke import (
    PROMPT,
    PROMPT_TOKEN_IDS,
    ROOT,
    configure_environment,
)


REQUEST_COUNT = 10
MAX_CUPTI_ATTEMPTS = 10


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parity(policy: dict, reference, candidate) -> dict[str, object]:
    import torch

    thresholds = policy["candidate_requirements"]
    difference = (candidate - reference).abs()
    floor = float(thresholds["max_relative_error_reference_floor"])
    mask = reference.abs() >= floor
    relative = difference[mask] / reference.abs()[mask]
    cosine = torch.nn.functional.cosine_similarity(
        candidate.double(), reference.double(), dim=1
    )
    reference_top_values, reference_top_ids = torch.topk(reference[0], 5)
    candidate_top_values, candidate_top_ids = torch.topk(candidate[0], 5)
    margin = float(candidate_top_values[0] - candidate_top_values[1])
    overlap = len(
        set(reference_top_ids.tolist()) & set(candidate_top_ids.tolist())
    )
    metrics = {
        "candidate_top1_margin": margin,
        "cosine_similarity": float(cosine.min()),
        "max_abs_error": float(difference.max()),
        "max_relative_error": float(relative.max()),
        "mean_abs_error": float(difference.mean()),
        "top5_token_overlap": overlap,
    }
    checks = {
        "candidate_top1_margin": margin
        >= float(thresholds["minimum_candidate_top1_margin"]),
        "cosine_similarity": metrics["cosine_similarity"]
        >= float(thresholds["cosine_similarity_min"]),
        "exact_greedy_token_ids": candidate_top_ids[:1].tolist()
        == policy["reference"]["next_token_ids"],
        "max_abs_error": metrics["max_abs_error"]
        <= float(thresholds["max_abs_error_max"]),
        "max_relative_error": metrics["max_relative_error"]
        <= float(thresholds["max_relative_error_max"]),
        "mean_abs_error": metrics["mean_abs_error"]
        <= float(thresholds["mean_abs_error_max"]),
        "top5_token_overlap": overlap
        >= int(thresholds["top5_token_overlap_min"]),
    }
    return {"checks": checks, "metrics": metrics, "passed": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument(
        "--model-size", choices=("0.8B", "9B"), default="0.8B"
    )
    args = parser.parse_args()
    model_slug = "0p8b" if args.model_size == "0.8B" else "9b"
    model_path = ROOT / f"models/Qwen3.5-{args.model_size}"
    model_id = f"Qwen/Qwen3.5-{args.model_size}"
    mem_fraction_static = 0.55 if args.model_size == "0.8B" else 0.78
    cpu_offload_gb = 0 if args.model_size == "0.8B" else 2
    configure_environment()
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / f"qwen35-{model_slug}-stability.json"
    policy_bytes = args.policy.read_bytes()
    policy = json.loads(policy_bytes)
    report: dict[str, object] = {
        "kind": f"qwen35-{model_slug}-strict-stability",
        "model_id": model_id,
        "cpu_offload_gb": cpu_offload_gb,
        "mem_fraction_static": mem_fraction_static,
        "policy": str(args.policy.resolve()),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "prompt": PROMPT,
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "request_count": REQUEST_COUNT,
        "run_id": run_id,
        "schema": 1,
        "status": "starting",
    }
    model_runner = None
    monitor = None
    monitor_api = None
    try:
        if (
            policy.get("status")
            != "frozen-before-final-candidate-measurement"
            or policy.get("model", {}).get("id") != model_id
            or policy.get("prompt") != PROMPT
            or policy.get("prompt_token_ids") != PROMPT_TOKEN_IDS
        ):
            raise ValueError("stability policy does not match the selected model/prompt")
        reference_file_sha256 = hashlib.sha256(
            args.reference_logits.read_bytes()
        ).hexdigest()
        if (
            reference_file_sha256
            != policy["reference"]["logits_tensor_file_sha256"]
        ):
            raise ValueError("reference logits file differs from the frozen policy")
        import torch
        from torch.profiler import _cupti_monitor as monitor_api

        if torch.cuda.is_initialized():
            raise RuntimeError("CUPTI stability collection must start before CUDA")
        monitor = monitor_api.start_collection(run_dir / "cupti-monitor")

        from sglang.srt.entrypoints.engine import _set_envs_and_config
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import (
            initialize_fp4_gemm_config,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            initialize_fp8_gemm_config,
        )
        from sglang.srt.plugins import load_plugins
        from sglang.benchmark import one_batch
        from sglang.srt.server_args import PortArgs, ServerArgs
        from transformers import AutoTokenizer

        load_plugins()
        server_args = ServerArgs(
            model_path=str(model_path),
            tokenizer_path=str(model_path),
            skip_tokenizer_init=True,
            enable_multimodal=False,
            language_model_only=True,
            load_format="safetensors",
            model_loader_extra_config='{"enable_multithread_load": false}',
            weight_loader_drop_cache_after_load=True,
            dtype="bfloat16",
            attention_backend="pypto",
            decode_attention_backend="pypto",
            prefill_attention_backend="pypto",
            linear_attn_backend="pypto",
            linear_attn_decode_backend="pypto",
            linear_attn_prefill_backend="pypto",
            disable_radix_cache=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            disable_custom_all_reduce=True,
            page_size=1,
            context_length=64,
            max_total_tokens=64,
            max_prefill_tokens=64,
            chunked_prefill_size=64,
            prefill_max_requests=1,
            max_running_requests=1,
            mem_fraction_static=mem_fraction_static,
            cpu_offload_gb=cpu_offload_gb,
            enable_torch_compile=True,
            torch_compile_max_bs=1,
            sampling_backend="pytorch",
            grammar_backend="none",
            random_seed=19,
            log_level="error",
            mm_processor_worker_num=1,
            mm_io_worker_num=1,
        )
        _set_envs_and_config(server_args)
        initialize_moe_config(server_args)
        initialize_fp8_gemm_config(server_args)
        initialize_fp4_gemm_config(server_args)
        port_args = PortArgs.init_new(server_args)
        one_batch.get_tokenizer = lambda *_args, **_kwargs: None
        model_runner, _tokenizer = one_batch.load_model(
            server_args, port_args, gpu_id=0, tp_rank=0
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        expected_ids = policy["reference"]["next_token_ids"]
        expected_text = tokenizer.decode(expected_ids, skip_special_tokens=False)
        reference = torch.load(
            args.reference_logits, map_location="cpu", weights_only=True
        ).float().contiguous()
        if (
            hashlib.sha256(reference.numpy().tobytes()).hexdigest()
            != policy["reference"]["logits_raw_sha256"]
        ):
            raise ValueError("reference logits payload differs from the frozen policy")

        warmup_reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
            1, len(PROMPT_TOKEN_IDS), [PROMPT_TOKEN_IDS]
        )
        _warmup_ids, _warmup_logits, warmup_batch = model_runner.extend(warmup_reqs)
        model_runner.synchronize()
        model_runner.cleanup(warmup_batch)
        model_runner.clear()
        torch.cuda.synchronize()

        windows = []
        raw_results = []
        torch_runner = model_runner.torch_runner
        original_forward = torch_runner.forward

        calibration_attempts = 0
        while calibration_attempts < MAX_CUPTI_ATTEMPTS:
            calibration_attempts += 1
            reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
                1, len(PROMPT_TOKEN_IDS), [PROMPT_TOKEN_IDS]
            )
            calibration_window = None

            def calibration_forward(*forward_args, **forward_kwargs):
                nonlocal calibration_window
                monitor.begin_trace_window()
                try:
                    return original_forward(*forward_args, **forward_kwargs)
                finally:
                    torch.cuda.synchronize()
                    calibration_window = monitor.end_trace_window()

            torch_runner.forward = calibration_forward
            try:
                _ids, _logits, calibration_batch = model_runner.extend(reqs)
            finally:
                torch_runner.forward = original_forward
            model_runner.synchronize()
            model_runner.cleanup(calibration_batch)
            model_runner.clear()
            torch.cuda.synchronize()
            if calibration_window is not None and any(
                event.get("kind") == "kernel"
                for event in calibration_window["events"]
            ):
                break
        else:
            raise RuntimeError("CUPTI calibration failed to capture GPU activity")
        report["cupti_calibration_attempts"] = calibration_attempts
        request_attempts = []

        for request_index in range(REQUEST_COUNT):
            attempt = 0
            while attempt < MAX_CUPTI_ATTEMPTS:
                attempt += 1
                reqs = one_batch.prepare_synthetic_inputs_for_latency_test(
                    1, len(PROMPT_TOKEN_IDS), [PROMPT_TOKEN_IDS]
                )
                window = None

                def traced_forward(*forward_args, **forward_kwargs):
                    nonlocal window
                    monitor.begin_trace_window()
                    try:
                        return original_forward(*forward_args, **forward_kwargs)
                    finally:
                        torch.cuda.synchronize()
                        window = monitor.end_trace_window()

                torch_runner.forward = traced_forward
                try:
                    next_ids, logits, batch = model_runner.extend(reqs)
                finally:
                    torch_runner.forward = original_forward
                model_runner.synchronize()
                model_runner.cleanup(batch)
                model_runner.clear()
                torch.cuda.synchronize()
                if window is not None and any(
                    event.get("kind") == "kernel"
                    for event in window["events"]
                ):
                    break
            else:
                raise RuntimeError(
                    f"request {request_index} failed CUPTI capture "
                    f"{MAX_CUPTI_ATTEMPTS} times"
                )
            request_attempts.append(attempt)
            logits_cpu = logits.detach().float().cpu().contiguous()
            ids = next_ids.cpu().tolist()
            text = tokenizer.decode(ids, skip_special_tokens=False)
            raw_results.append(
                {
                    "decoded_text": text,
                    "expected_decoded_text": expected_text,
                    "exact_decoded_text": text == expected_text,
                    "exact_token_ids": ids == expected_ids,
                    "logits_sha256": hashlib.sha256(
                        logits_cpu.numpy().tobytes()
                    ).hexdigest(),
                    "next_token_ids": ids,
                    "parity": _parity(policy, reference, logits_cpu),
                    "request_index": request_index,
                }
            )
            windows.append(window)

        report["cupti_request_attempts"] = request_attempts

        monitor_stats = monitor_api.stop_collection()
        monitor = None
        if int(monitor_stats["dropped_records"]) != 0:
            raise RuntimeError("CUPTI dropped stability activity records")

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

        device_fingerprint = _device_fingerprint(torch)
        model_revision = _model_revision(model_path)
        for request_index, (window, result) in enumerate(
            zip(windows, raw_results, strict=True)
        ):
            normalized = normalize_cupti_window(window, dropped_records=0)
            artifacts = list(normalized.artifacts)
            events = list(normalized.events)
            coverage_path = run_dir / f"coverage-request-{request_index:02d}.json"
            manifest = TraceManifest(
                run_id=f"{run_id}:request:{request_index}",
                model_id=model_id,
                model_revision=model_revision,
                device_fingerprint=device_fingerprint,
                collector=TRACE_COLLECTOR,
                collector_revision=TRACE_COLLECTOR_REVISION,
                framework_profile=FRAMEWORK_PROFILE,
                artifact_registry_digest=compute_artifact_registry_digest(
                    artifacts
                ),
                trace_digest=compute_trace_digest(events),
                activity_count=len(events),
                closed_world=normalized.closed_world,
            )
            with CoverageAuditor(
                mode=CoverageMode.STRICT,
                report_path=coverage_path,
                manifest=manifest,
                artifacts=artifacts,
            ) as auditor:
                for event in events:
                    auditor.record(event)
                summary = auditor.finalize(event_stream_complete=True)
            result["coverage_report"] = str(coverage_path)
            result["coverage_summary"] = asdict(summary)
            result["passed"] = bool(
                result["exact_token_ids"]
                and result["exact_decoded_text"]
                and result["parity"]["passed"]
                and summary.strict_policy_passed
            )

        report["requests"] = raw_results
        report["expected_next_token_ids"] = expected_ids
        report["expected_decoded_text"] = expected_text
        report["all_passed"] = all(result["passed"] for result in raw_results)
        report["unique_logits_sha256"] = sorted(
            {result["logits_sha256"] for result in raw_results}
        )
        report["status"] = "complete" if report["all_passed"] else "failed"
        _write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if report["all_passed"] else 1
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 1
    finally:
        if monitor is not None and monitor_api is not None:
            try:
                monitor_api.stop_collection()
            except BaseException:
                pass
        try:
            from sglang.srt.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except (ImportError, RuntimeError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
