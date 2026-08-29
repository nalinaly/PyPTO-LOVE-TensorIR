"""Performance-only aligned operator A/B for real Qwen3.5-9B shapes."""

from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import platform
import time
import traceback

from .evidence_identity import collect_run_identity, comparable_identity
from .lanes import prepare_worker_environment
from .performance_runtime import ResourceSampler, _resource_summary
from .workload import (
    PROMPT_TOKENS,
    SAMPLE_INTERVAL_MS,
    SCHEMA_VERSION,
    ReleaseContractError,
    atomic_json,
    bootstrap_median_comparison_ci,
    canonical_json_sha256,
    distribution,
    fresh_start_methodology,
    fresh_start_summary,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
OPERATOR_LANES = ("pypto", "sglang-matched")
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
DEFAULT_CALLS_PER_BATCH = 100
LM_HEAD_CALLS_PER_BATCH = 1
EXPECTED_MODEL_FIELDS = {
    "model_type": "qwen3_5_text",
    "dtype": "bfloat16",
    "hidden_size": 4_096,
    "intermediate_size": 12_288,
    "vocab_size": 248_320,
    "num_hidden_layers": 32,
}
CASE_CONTRACT_FIELDS = (
    "name",
    "operator",
    "phase",
    "rows",
    "input_shape",
    "weight_shape",
    "output_shape",
    "input_dtype",
    "weight_dtype",
    "output_dtype",
    "warmup_calls",
    "timed_batches",
    "calls_per_batch",
    "calls_adjustment_reason",
    "semantic_contract",
    "callsite_note",
    "candidate_implementation",
    "stock_implementation",
)


def load_model_contract(model_path: Path) -> dict[str, object]:
    model_path = model_path.resolve(strict=True)
    if not model_path.is_dir():
        raise ReleaseContractError(f"model path is not a directory: {model_path}")
    if ROOT not in model_path.parents:
        raise ReleaseContractError("Qwen3.5-9B model path escaped the workspace")
    config_path = (model_path / "config.json").resolve(strict=True)
    if model_path not in config_path.parents or not config_path.is_file():
        raise ReleaseContractError("Qwen3.5-9B config escaped the model directory")
    raw = config_path.read_bytes()
    payload = json.loads(raw)
    text = payload.get("text_config", payload)
    if type(text) is not dict:
        raise ReleaseContractError("Qwen3.5-9B text_config is not an object")
    observed = {field: text.get(field) for field in EXPECTED_MODEL_FIELDS}
    if observed != EXPECTED_MODEL_FIELDS or any(
        type(observed[field]) is not type(expected)
        for field, expected in EXPECTED_MODEL_FIELDS.items()
    ):
        raise ReleaseContractError(
            "Qwen3.5-9B operator geometry drifted: "
            f"expected={EXPECTED_MODEL_FIELDS!r}, observed={observed!r}"
        )
    return {
        "model_id": "Qwen/Qwen3.5-9B",
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "tensor_parallel_size": 1,
        "frozen_prompt_tokens": PROMPT_TOKENS,
        **observed,
    }


def case_specs(model: dict[str, object]) -> tuple[dict[str, object], ...]:
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    vocab = int(model["vocab_size"])
    cases = []
    for phase, rows in (("decode", 1), ("prefill", PROMPT_TOKENS)):
        cases.extend(
            (
                {
                    "name": f"swiglu-{phase}-{rows}x{2 * intermediate}",
                    "operator": "swiglu",
                    "phase": phase,
                    "rows": rows,
                    "input_shape": [rows, 2 * intermediate],
                    "weight_shape": None,
                    "output_shape": [rows, intermediate],
                    "input_dtype": "bfloat16",
                    "weight_dtype": None,
                    "output_dtype": "bfloat16",
                    "semantic_contract": "packed BF16 SiLU(gate FP32) * up FP32, cast BF16",
                    "callsite_note": None,
                    "candidate_implementation": "inductor_generated_pypto",
                    "stock_implementation": "sglang_jit_silu_and_mul",
                },
                {
                    "name": f"gate-up-linear-{phase}-{rows}x{hidden}x{2 * intermediate}",
                    "operator": "gate_up_linear",
                    "phase": phase,
                    "rows": rows,
                    "input_shape": [rows, hidden],
                    "weight_shape": [2 * intermediate, hidden],
                    "output_shape": [rows, 2 * intermediate],
                    "input_dtype": "bfloat16",
                    "weight_dtype": "bfloat16",
                    "output_dtype": "bfloat16",
                    "semantic_contract": "bias-free BF16 linear projection",
                    "callsite_note": None,
                    "candidate_implementation": "handwritten_pypto",
                    "stock_implementation": "torch_functional_linear",
                },
                {
                    "name": f"down-linear-{phase}-{rows}x{intermediate}x{hidden}",
                    "operator": "down_linear",
                    "phase": phase,
                    "rows": rows,
                    "input_shape": [rows, intermediate],
                    "weight_shape": [hidden, intermediate],
                    "output_shape": [rows, hidden],
                    "input_dtype": "bfloat16",
                    "weight_dtype": "bfloat16",
                    "output_dtype": "bfloat16",
                    "semantic_contract": "bias-free BF16 linear projection",
                    "callsite_note": None,
                    "candidate_implementation": "handwritten_pypto",
                    "stock_implementation": "torch_functional_linear",
                },
            )
        )
    cases.append(
        {
            "name": (f"fp32-lm-head-decode-and-pruned-prefill-1x{hidden}x{vocab}"),
            "operator": "fp32_lm_head",
            "phase": "decode_and_pruned_prefill",
            "rows": 1,
            "input_shape": [1, hidden],
            "weight_shape": [vocab, hidden],
            "output_shape": [1, vocab],
            "input_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "output_dtype": "float32",
            "semantic_contract": (
                "use_fp32_lm_head=false: BF16 F.linear logits widened to FP32"
            ),
            "callsite_note": (
                "Plain prefill zero-copy selects the final hidden-state row before "
                f"the LM head, so decode and the frozen {PROMPT_TOKENS}-token prefill "
                "both call "
                "the projection with rows=1."
            ),
            "candidate_implementation": "handwritten_pypto",
            "stock_implementation": "torch_functional_linear_then_float",
        }
    )
    result = []
    for case in cases:
        lm_head = case["operator"] == "fp32_lm_head"
        result.append(
            {
                **case,
                "warmup_calls": WARMUP_CALLS,
                "timed_batches": TIMED_BATCHES,
                "calls_per_batch": (
                    LM_HEAD_CALLS_PER_BATCH if lm_head else DEFAULT_CALLS_PER_BATCH
                ),
                "calls_adjustment_reason": (
                    "The 248320x4096 BF16 vocabulary weight is about 1.9 GiB; one "
                    "call per timed batch preserves 30 independent CUDA-event "
                    "samples without turning the A/B into a multi-terabyte read loop."
                    if lm_head
                    else None
                ),
            }
        )
    return tuple(result)


def _artifact_snapshot() -> dict[str, dict[str, object]]:
    from pypto_plugins.activity_trace import artifact_registry_snapshot

    return {
        record.artifact_id: record.to_dict() for record in artifact_registry_snapshot()
    }


def _new_case_artifact(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
    case: dict[str, object],
) -> dict[str, object]:
    created = [after[key] for key in sorted(set(after) - set(before))]
    if len(created) != 1:
        raise ReleaseContractError(
            f"{case['name']} produced {len(created)} new PyPTO artifacts; expected one"
        )
    artifact = created[0]
    expected_provider = (
        "pypto.generic" if case["operator"] == "swiglu" else "pypto.matmul"
    )
    source = str(artifact.get("source_node"))
    expected_source = {
        "swiglu": "torch-inductor:",
        "gate_up_linear": "pypto_kernels.linear:linear_kernel",
        "down_linear": "pypto_kernels.linear:linear_kernel",
        "fp32_lm_head": "pypto_kernels.linear:linear_to_float_kernel",
    }[str(case["operator"])]
    source_matches = (
        source.startswith(expected_source)
        if expected_source.endswith(":")
        else source == expected_source
    )
    if artifact.get("provider") != expected_provider or not source_matches:
        raise ReleaseContractError(
            f"{case['name']} artifact provenance drifted: {artifact!r}"
        )
    return artifact


def _expected_public_callable(lane: str, operator: str) -> str:
    if operator == "swiglu":
        return (
            "pypto_plugins.torch.inductor_swiglu.run_fp32_swiglu"
            if lane == "pypto"
            else "sglang.srt.layers.activation.silu_and_mul"
        )
    if lane == "pypto":
        return (
            "pypto_kernels.linear.linear_to_float"
            if operator == "fp32_lm_head"
            else "pypto_kernels.linear.linear"
        )
    return (
        "torch.nn.functional.linear(...).float"
        if operator == "fp32_lm_head"
        else "torch.nn.functional.linear"
    )


def _validate_provider_record(
    lane: str, case: dict[str, object], provider: dict[str, object]
) -> None:
    expected_callable = _expected_public_callable(lane, str(case["operator"]))
    if provider.get("public_callable") != expected_callable:
        raise ReleaseContractError(
            f"operator public callable drifted for {case['name']}: {provider!r}"
        )
    expected_stream_policy = (
        "pypto_stream_current_ordering"
        if lane == "pypto" and case["operator"] != "swiglu"
        else "caller_current_stream"
    )
    if provider.get("stream_policy") != expected_stream_policy:
        raise ReleaseContractError(
            f"operator stream policy drifted for {case['name']}: {provider!r}"
        )
    artifact = provider.get("artifact")
    if lane == "pypto":
        if provider.get("kind") != "pypto_artifact" or not isinstance(artifact, dict):
            raise ReleaseContractError(
                f"candidate artifact provenance is absent for {case['name']}"
            )
        artifact_id = artifact.get("artifact_id")
        if type(artifact_id) is not str or not artifact_id:
            raise ReleaseContractError("candidate artifact identity is empty")
        _new_case_artifact({}, {artifact_id: artifact}, case)
    elif provider.get("kind") != "stock_public_api" or artifact is not None:
        raise ReleaseContractError(
            f"stock provider provenance drifted for {case['name']}: {provider!r}"
        )


def _prepare_case(torch, lane: str, case: dict[str, object]):
    input_ = torch.full(
        tuple(case["input_shape"]),
        0.125,
        dtype=torch.bfloat16,
        device="cuda",
    )
    operator = str(case["operator"])
    if operator == "swiglu":
        if lane == "pypto":
            from pypto_plugins.torch.inductor_swiglu import run_fp32_swiglu

            half = int(case["output_shape"][-1])
            gate = input_[:, :half]
            up = input_[:, half:]

            def invoke():
                return run_fp32_swiglu(gate, up)

            allocations = (input_, gate, up)
        else:
            from sglang.srt.layers.activation import silu_and_mul

            def invoke():
                output = torch.empty(
                    tuple(case["output_shape"]),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                silu_and_mul(input_, output)
                return output

            allocations = (input_,)
        callable_name = _expected_public_callable(lane, operator)
    else:
        weight = torch.full(
            tuple(case["weight_shape"]),
            0.03125,
            dtype=torch.bfloat16,
            device="cuda",
        )
        if lane == "pypto":
            from pypto_kernels import linear
            from pypto_plugins.sglang.stream import pypto_stream

            if operator == "fp32_lm_head":

                def invoke():
                    with pypto_stream(input_.device) as stream:
                        return linear.linear_to_float(input_, weight, stream=stream)

                callable_name = _expected_public_callable(lane, operator)
            else:

                def invoke():
                    with pypto_stream(input_.device) as stream:
                        return linear.linear(input_, weight, stream=stream)

                callable_name = _expected_public_callable(lane, operator)
        elif operator == "fp32_lm_head":

            def invoke():
                return torch.nn.functional.linear(input_, weight).float()

            callable_name = _expected_public_callable(lane, operator)
        else:

            def invoke():
                return torch.nn.functional.linear(input_, weight)

            callable_name = _expected_public_callable(lane, operator)
        allocations = (input_, weight)
    return invoke, callable_name, allocations


def _measure_case(torch, lane: str, case: dict[str, object]) -> dict[str, object]:
    before = _artifact_snapshot() if lane == "pypto" else {}
    invoke, callable_name, allocations = _prepare_case(torch, lane, case)
    torch.cuda.synchronize()
    trigger_start_ns = time.perf_counter_ns()
    output = invoke()
    torch.cuda.synchronize()
    first_trigger_ms = (time.perf_counter_ns() - trigger_start_ns) / 1e6
    after = _artifact_snapshot() if lane == "pypto" else {}
    provider = (
        {
            "kind": "pypto_artifact",
            "public_callable": callable_name,
            "stream_policy": (
                "pypto_stream_current_ordering"
                if case["operator"] != "swiglu"
                else "caller_current_stream"
            ),
            "artifact": _new_case_artifact(before, after, case),
        }
        if lane == "pypto"
        else {
            "kind": "stock_public_api",
            "public_callable": callable_name,
            "stream_policy": "caller_current_stream",
            "artifact": None,
        }
    )

    for _ in range(int(case["warmup_calls"])):
        output = invoke()
    torch.cuda.synchronize()
    batch_ms = []
    for _ in range(int(case["timed_batches"])):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(case["calls_per_batch"])):
            output = invoke()
        end.record()
        end.synchronize()
        batch_ms.append(float(start.elapsed_time(end)) / int(case["calls_per_batch"]))
    del output, invoke, allocations
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {
        **case,
        "provider": provider,
        "first_compile_trigger_call_wall_ms": first_trigger_ms,
        "total_timed_calls": int(case["timed_batches"]) * int(case["calls_per_batch"]),
        "latency_ms_per_call": distribution(batch_ms),
        "raw_batch_average_ms_per_call": batch_ms,
    }


def _record_failure(report: dict[str, object], error: BaseException) -> None:
    report.update(
        {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    )


def _validated_resource_identity(resources: object) -> dict[str, object]:
    if not isinstance(resources, dict) or resources.get("nvml_error") is not None:
        raise ReleaseContractError("operator NVML sampling is incomplete")
    gpu_identity = resources.get("gpu_identity")
    resource_summary = resources.get("summary")
    if not isinstance(gpu_identity, dict) or not isinstance(resource_summary, dict):
        raise ReleaseContractError("operator resource identity is incomplete")
    if resource_summary.get("thermal_throttle_observed") is not False:
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
    return gpu_identity


def run(lane: str, model_path: Path, run_id: str, run_dir: Path) -> int:
    if lane not in OPERATOR_LANES:
        raise ReleaseContractError(f"unknown operator performance lane: {lane}")
    prepare_worker_environment(lane)
    report_path = run_dir / f"qwen35-9b-operator-performance-{lane}.json"
    resources_path = run_dir / f"qwen35-9b-operator-resources-{lane}.json"
    report: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "kind": "qwen35-9b-aligned-operator-performance-only",
        "lane": lane,
        "run_id": run_id,
        "measurement_boundary": (
            "Warm latency is CUDA-event stream device time across equivalent public "
            "operator calls. Host dispatch/allocation overhead is excluded unless it "
            "enqueues device work. First-trigger is synchronized wall time and includes "
            "compilation initiated by the callable; module imports and input/weight "
            "allocation happen before that timer. This process runs no numerical "
            "correctness oracle."
        ),
        "allocation_policy": (
            "Only one case's inputs and weight are resident; each case is synchronized, "
            "released, garbage-collected and removed from the CUDA allocator cache "
            "before the next case."
        ),
        "tensor_content_policy": (
            "Inputs and weights use fixed finite constants with exact frozen model "
            "shapes and dtypes. Initialization finishes before first-trigger timing; "
            "no model weight contents or generated values are inspected."
        ),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "status": "starting",
    }
    sampler = ResourceSampler(os.getpgid(0))
    sampler.start()
    return_code = 1
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (
            12,
            0,
        ):
            raise ReleaseContractError("aligned operator A/B requires one SM120 GPU")
        model = load_model_contract(model_path)
        report["model_contract"] = model
        with torch.inference_mode():
            report["cases"] = [
                _measure_case(torch, lane, case) for case in case_specs(model)
            ]
        return_code = 0
    except BaseException as error:
        _record_failure(report, error)
    finally:
        sampler.stop()
        resource_payload = {
            "schema": SCHEMA_VERSION,
            "kind": "qwen35-9b-aligned-operator-resource-samples",
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

    if return_code == 0:
        try:
            _validated_resource_identity(report["resources"])
            identity_profile = "pypto" if lane == "pypto" else "baseline"
            report["evidence_identity"] = collect_run_identity(
                ROOT, identity_profile, model_path.resolve(strict=True)
            )
            report["status"] = "complete"
        except BaseException as error:
            _record_failure(report, error)
            return_code = 1
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


def _validated_global_identity(
    report: dict[str, object], lane: str
) -> dict[str, object]:
    identity = report.get("evidence_identity")
    if not isinstance(identity, dict):
        raise ReleaseContractError("operator A/B report has no evidence identity")
    expected_profile = "pypto" if lane == "pypto" else "baseline"
    if identity.get("selected_environment_lock") != expected_profile:
        raise ReleaseContractError("operator A/B report selected the wrong environment")
    unsigned = copy.deepcopy(identity)
    claimed = unsigned.pop("identity_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise ReleaseContractError("operator A/B evidence identity digest differs")
    return comparable_identity(identity)


def summarize_fresh_starts(
    reports_by_lane: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    if set(reports_by_lane) != set(OPERATOR_LANES):
        raise ReleaseContractError("operator A/B summary requires candidate and stock")
    starts: dict[str, dict[str, list[float]]] = {}
    summaries: dict[str, object] = {}
    gpu_identities = []
    global_identities = []
    model_contract = None
    case_contracts: dict[str, dict[str, object]] = {}
    for lane in OPERATOR_LANES:
        reports = reports_by_lane[lane]
        if len(reports) != 4:
            raise ReleaseContractError(f"operator lane {lane} requires four starts")
        case_starts: dict[str, list[float]] = {}
        trigger_starts: dict[str, list[float]] = {}
        provider_records: dict[str, list[dict[str, object]]] = {}
        for report in reports:
            if (
                report.get("status") != "complete"
                or report.get("kind") != "qwen35-9b-aligned-operator-performance-only"
                or report.get("lane") != lane
            ):
                raise ReleaseContractError(
                    f"operator report is not accepted for {lane}"
                )
            current_model = report.get("model_contract")
            if not isinstance(current_model, dict):
                raise ReleaseContractError("operator model contract is absent")
            if model_contract is None:
                model_contract = current_model
            elif current_model != model_contract:
                raise ReleaseContractError(
                    "operator model geometry drifted across starts"
                )
            global_identities.append(_validated_global_identity(report, lane))
            gpu_identity = _validated_resource_identity(report.get("resources"))
            gpu_identities.append(gpu_identity)
            cases = report.get("cases")
            if type(cases) is not list or len(cases) != 7:
                raise ReleaseContractError("operator report must contain seven cases")
            for case in cases:
                if not isinstance(case, dict):
                    raise ReleaseContractError("operator case is malformed")
                name = str(case.get("name"))
                contract = {field: case.get(field) for field in CASE_CONTRACT_FIELDS}
                previous_contract = case_contracts.setdefault(name, contract)
                if previous_contract != contract:
                    raise ReleaseContractError(
                        f"operator case contract drifted: {name}"
                    )
                raw = case.get("raw_batch_average_ms_per_call")
                if type(raw) is not list or len(raw) != int(case["timed_batches"]):
                    raise ReleaseContractError("operator timed batch count drifted")
                if int(case.get("total_timed_calls", 0)) != int(
                    case["timed_batches"]
                ) * int(case["calls_per_batch"]):
                    raise ReleaseContractError("operator timed call count drifted")
                provider = case.get("provider")
                if not isinstance(provider, dict):
                    raise ReleaseContractError("operator provider record is absent")
                _validate_provider_record(lane, case, provider)
                case_starts.setdefault(name, []).append(distribution(raw)["p50"])
                trigger_starts.setdefault(name, []).append(
                    float(case["first_compile_trigger_call_wall_ms"])
                )
                provider_records.setdefault(name, []).append(provider)
        if set(case_starts) != set(case_contracts):
            raise ReleaseContractError("operator case set drifted between lanes")
        for name, records in provider_records.items():
            if any(record != records[0] for record in records[1:]):
                raise ReleaseContractError(
                    f"operator provider/artifact drifted across {lane} starts: {name}"
                )
        starts[lane] = case_starts
        summaries[lane] = {
            "fresh_starts": len(reports),
            "cases": {
                name: {
                    "contract": case_contracts[name],
                    "provider": provider_records[name][0],
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
    if any(identity != global_identities[0] for identity in global_identities[1:]):
        raise ReleaseContractError(
            "operator A/B model/package/source/GPU identity drifted"
        )
    comparisons = {}
    for name in sorted(case_contracts):
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
            "contract": case_contracts[name],
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
        "kind": "qwen35-9b-aligned-operator-ab-performance-summary",
        "model_contract": model_contract,
        "methodology": fresh_start_methodology(
            "CUDA-event timed batch averages within each aligned operator case"
        ),
        "lanes": summaries,
        "comparisons": comparisons,
        "gpu_identity": gpu_identities[0],
        "global_evidence_identity": global_identities[0],
        "correctness_evaluated": False,
        "status": "complete",
    }
