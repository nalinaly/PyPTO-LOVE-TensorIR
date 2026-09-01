"""Frozen SGLang configurations for the three release lanes."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

from .workload import LANES, ReleaseContractError


BASELINE_LANES = {"sglang-matched", "sglang-optimized"}
MATCHED_CONTROL_FIELDS = (
    "model_path",
    "tokenizer_path",
    "skip_tokenizer_init",
    "enable_multimodal",
    "json_model_override_args",
    "load_format",
    "dtype",
    "kv_cache_dtype",
    "tp_size",
    "context_length",
    "max_total_tokens",
    "max_prefill_tokens",
    "max_running_requests",
    "prefill_max_requests",
    "enable_torch_compile",
    "torch_compile_max_bs",
    "sampling_backend",
    "cpu_offload_gb",
    "mem_fraction_static",
    "disable_radix_cache",
    "disable_cuda_graph",
    "disable_overlap_schedule",
    "disable_custom_all_reduce",
    "page_size",
    "chunked_prefill_size",
    "random_seed",
    "stream_interval",
)
IMPLEMENTATION_FIELDS = (
    "attention_backend",
    "decode_attention_backend",
    "prefill_attention_backend",
    "linear_attn_backend",
    "linear_attn_decode_backend",
    "linear_attn_prefill_backend",
    "mamba_ssm_dtype",
)
MATCHED_RESOLVED_CONTROL_FIELDS = (
    "sampling_backend",
    "torch_compile_requested",
    "cuda_graph_enabled_by_server_args",
    "overlap_schedule_enabled_by_server_args",
    "radix_cache_enabled_by_server_args",
)
MATCHED_RESOLVED_IMPLEMENTATION_FIELDS = (
    "attention_prefill",
    "attention_decode",
    "linear_attention_default",
    "linear_attention_prefill",
    "linear_attention_decode",
    "mamba",
    "mamba_ssm_dtype",
)
FORBIDDEN_RELEASE_OVERRIDES = (
    "PYPTO_PLUGINS_PYPTO_DSO",
    "PYPTO_PLUGINS_CUDA_DRIVER_LABEL",
    "PYPTO_PLUGINS_CUDART",
    "PYPTO_KERNEL_DSO_PATH",
    "PYPTO_KERNEL_PACKAGE_PATH",
    "PYPTO_KERNEL_CUDA_DRIVER_LABEL",
    "PYPTO_KERNEL_CUDART",
)


def prepare_worker_environment(lane: str) -> None:
    if lane not in LANES:
        raise ValueError(f"unknown lane: {lane}")
    active_overrides = sorted(
        name for name in FORBIDDEN_RELEASE_OVERRIDES if os.environ.get(name)
    )
    if active_overrides:
        raise ReleaseContractError(
            "formal release workers forbid diagnostic runtime overrides: "
            + ", ".join(active_overrides)
        )
    expected_profile = "pypto" if lane == "pypto" else "baseline"
    actual_profile = os.environ.get("PYPTO_FRAMEWORK_PROFILE")
    if actual_profile != expected_profile:
        raise ReleaseContractError(
            f"lane {lane} requires profile {expected_profile}, got {actual_profile!r}"
        )
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
            "SGLANG_IO_WORKERS": "1",
            "SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS": "1",
        }
    )
    if lane == "pypto":
        os.environ.update(
            {
                "PYPTO_ALLOW_FALLBACK": "0",
                "PYPTO_STRICT_COVERAGE": "1",
                "SGLANG_PLUGINS": "pypto",
            }
        )


def memory_qualification(
    lane: str,
    optimized_memory_mode: str = "zero-offload",
    model_path: Path | None = None,
) -> dict[str, object]:
    if lane in {"pypto", "sglang-matched"}:
        if model_path is not None and model_path.name == "Qwen3.5-0.8B":
            return {
                "name": "candidate-matched-0p8b-zero-offload",
                "cpu_offload_gb": 0,
                "mem_fraction_static": 0.78,
            }
        if model_path is not None and model_path.name == "Qwen3.5-9B":
            if lane == "pypto":
                return {
                    "name": "candidate-9b-offload-2g",
                    "cpu_offload_gb": 2,
                    "mem_fraction_static": 0.78,
                }
            return {
                "name": "matched-9b-offload-2g",
                "cpu_offload_gb": 2,
                "mem_fraction_static": 0.78,
            }
        return {
            "name": "candidate-matched-offload-2g",
            "cpu_offload_gb": 2,
            "mem_fraction_static": 0.78,
        }
    if optimized_memory_mode == "zero-offload":
        return {
            "name": "optimized-zero-offload-auto-static",
            "cpu_offload_gb": 0,
            "mem_fraction_static": None,
        }
    if optimized_memory_mode == "matched":
        return {
            "name": "optimized-matched-memory",
            "cpu_offload_gb": 2,
            "mem_fraction_static": 0.69,
        }
    raise ValueError(f"unknown optimized memory mode: {optimized_memory_mode}")


def server_kwargs(
    lane: str,
    model_path: Path,
    optimized_memory_mode: str = "zero-offload",
) -> dict[str, Any]:
    if lane not in LANES:
        raise ValueError(f"unknown lane: {lane}")
    common: dict[str, Any] = {
        "model_path": str(model_path),
        "tokenizer_path": str(model_path),
        "skip_tokenizer_init": True,
        "enable_multimodal": False,
        # The pinned stock SGLang ServerArgs whitelist does not yet admit
        # Qwen3.5 for its --language-model-only flag even though the model
        # implementation already honors the equivalent config field.  Use
        # SGLang's public model-config override in every lane so the baseline
        # remains plugin-free and all lanes construct the same text-only model.
        "json_model_override_args": '{"language_model_only":true}',
        "load_format": "safetensors",
        "model_loader_extra_config": '{"enable_multithread_load": false}',
        "weight_loader_drop_cache_after_load": True,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "tp_size": 1,
        "context_length": 256,
        "max_total_tokens": 256,
        "max_prefill_tokens": 256,
        "max_running_requests": 1,
        "prefill_max_requests": 1,
        "enable_torch_compile": True,
        "torch_compile_max_bs": 1,
        "random_seed": 19,
        "grammar_backend": "none",
        "stream_interval": 1,
        "watchdog_timeout": 1200,
        "soft_watchdog_timeout": 600,
        "log_level": "error",
        "mm_processor_worker_num": 1,
        "mm_io_worker_num": 1,
    }
    memory = memory_qualification(lane, optimized_memory_mode, model_path)
    common["cpu_offload_gb"] = memory["cpu_offload_gb"]
    if memory["mem_fraction_static"] is not None:
        common["mem_fraction_static"] = memory["mem_fraction_static"]
    if lane == "pypto":
        common.update(
            {
                "attention_backend": "pypto",
                "decode_attention_backend": "pypto",
                "prefill_attention_backend": "pypto",
                "linear_attn_backend": "pypto",
                "linear_attn_decode_backend": "pypto",
                "linear_attn_prefill_backend": "pypto",
                "sampling_backend": "pytorch",
                "disable_radix_cache": True,
                "disable_cuda_graph": True,
                "disable_overlap_schedule": True,
                "disable_custom_all_reduce": True,
                "page_size": 1,
                "chunked_prefill_size": -1,
            }
        )
    else:
        # Keep the pinned model's FP32 recurrent-state contract and SGLang's
        # stock Triton GDN implementation. On this exact SM120 stack,
        # FlashInfer decode is admitted only with BF16 state while its selected
        # prefill kernel rejects BF16 state, so that combination is not a
        # runnable baseline. Full attention remains on FlashInfer.
        common.update(
            {
                "attention_backend": "flashinfer",
                "linear_attn_backend": "triton",
                "linear_attn_decode_backend": "triton",
                "linear_attn_prefill_backend": "triton",
                "mamba_ssm_dtype": "float32",
                "sampling_backend": (
                    "pytorch" if lane == "sglang-matched" else "flashinfer"
                ),
            }
        )
        if lane == "sglang-matched":
            common.update(
                {
                    "disable_radix_cache": True,
                    "disable_cuda_graph": True,
                    "disable_overlap_schedule": True,
                    "disable_custom_all_reduce": True,
                    "page_size": 1,
                    "chunked_prefill_size": -1,
                }
            )
        else:
            # The release workload has one request and a fixed 256-token
            # prefill. Restrict the official CUDA-graph lane to those exact
            # buckets so unrelated capture sizes cannot consume its safety
            # margin or contaminate the timing sample.
            common.update(
                {
                    "cuda_graph_bs_decode": [1],
                    "cuda_graph_bs_prefill": [32],
                    "cuda_graph_backend_prefill": "disabled",
                }
            )
    return common


def performance_memory_qualification(
    lane: str,
    optimized_memory_mode: str = "zero-offload",
    model_path: Path | None = None,
) -> dict[str, object]:
    """Return a cross-lane-comparable memory envelope for timing runs.

    Correctness keeps its established lane-specific memory qualification.  The
    9B timing pair instead uses a common envelope intended to restore margin on
    this 24 GiB GPU. The measured resource gate still decides acceptance.
    """

    if (
        model_path is not None
        and model_path.name == "Qwen3.5-9B"
        and lane in {"pypto", "sglang-matched"}
    ):
        return {
            "name": "performance-pair-9b-offload-2g",
            "cpu_offload_gb": 2,
            "mem_fraction_static": 0.78,
        }
    return memory_qualification(lane, optimized_memory_mode, model_path)


def performance_server_kwargs(
    lane: str,
    model_path: Path,
    optimized_memory_mode: str = "zero-offload",
) -> dict[str, Any]:
    """Apply the timing-only memory envelope to the normal lane contract."""

    values = server_kwargs(lane, model_path, optimized_memory_mode)
    memory = performance_memory_qualification(
        lane, optimized_memory_mode, model_path
    )
    values["cpu_offload_gb"] = memory["cpu_offload_gb"]
    if memory["mem_fraction_static"] is None:
        values.pop("mem_fraction_static", None)
    else:
        values["mem_fraction_static"] = memory["mem_fraction_static"]
    return values


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def resolved_backend_record(server_args: object) -> dict[str, object]:
    getter = getattr(server_args, "get_attention_backends", None)
    attention = (
        list(getter())
        if callable(getter)
        else [
            getattr(server_args, "prefill_attention_backend", None)
            or getattr(server_args, "attention_backend", None),
            getattr(server_args, "decode_attention_backend", None)
            or getattr(server_args, "attention_backend", None),
        ]
    )
    record = {
        "attention_prefill": attention[0],
        "attention_decode": attention[1],
        "linear_attention_default": getattr(server_args, "linear_attn_backend", None),
        "linear_attention_prefill": getattr(
            server_args, "linear_attn_prefill_backend", None
        )
        or getattr(server_args, "linear_attn_backend", None),
        "linear_attention_decode": getattr(
            server_args, "linear_attn_decode_backend", None
        )
        or getattr(server_args, "linear_attn_backend", None),
        "mamba": getattr(server_args, "mamba_backend", None),
        "mamba_ssm_dtype": getattr(server_args, "mamba_ssm_dtype", None),
        "sampling_backend": getattr(server_args, "sampling_backend", None),
        "torch_compile_requested": bool(
            getattr(server_args, "enable_torch_compile", False)
        ),
        "cuda_graph_enabled_by_server_args": not bool(
            getattr(server_args, "disable_cuda_graph", False)
        ),
        "overlap_schedule_enabled_by_server_args": not bool(
            getattr(server_args, "disable_overlap_schedule", False)
        ),
        "radix_cache_enabled_by_server_args": not bool(
            getattr(server_args, "disable_radix_cache", False)
        ),
        "cuda_graph_config": _jsonable(getattr(server_args, "cuda_graph_config", None)),
    }
    return record


def validate_resolved_backends(lane: str, record: dict[str, object]) -> None:
    if lane == "pypto":
        required = (
            "attention_prefill",
            "attention_decode",
            "linear_attention_default",
            "linear_attention_prefill",
            "linear_attention_decode",
        )
        drift = {key: record.get(key) for key in required if record.get(key) != "pypto"}
        if drift:
            raise ReleaseContractError(f"PyPTO backend resolution drifted: {drift}")
        if record.get("sampling_backend") != "pytorch":
            raise ReleaseContractError(
                "PyPTO lane did not resolve the frozen PyTorch sampler"
            )
        return
    expected_backends = {
        "attention_prefill": "flashinfer",
        "attention_decode": "flashinfer",
        "linear_attention_default": "triton",
        "linear_attention_prefill": "triton",
        "linear_attention_decode": "triton",
    }
    drift = {
        key: record.get(key)
        for key, expected in expected_backends.items()
        if record.get(key) != expected
    }
    if drift or record.get("mamba_ssm_dtype") != "float32":
        raise ReleaseContractError(
            "stock backend did not resolve to supported Triton/FP32 state: "
            f"drift={drift}, dtype={record.get('mamba_ssm_dtype')!r}"
        )
    expected_sampler = "pytorch" if lane == "sglang-matched" else "flashinfer"
    if record.get("sampling_backend") != expected_sampler:
        raise ReleaseContractError(
            f"{lane} sampler did not resolve to {expected_sampler!r}: "
            f"{record.get('sampling_backend')!r}"
        )


def execution_feature_record(
    requested: dict[str, object], resolved: dict[str, object]
) -> dict[str, object]:
    """Record configuration state without claiming replay or overlap execution."""

    cuda_graph_requested = not bool(requested.get("disable_cuda_graph", False))
    overlap_requested = not bool(requested.get("disable_overlap_schedule", False))
    return {
        "cuda_graph": {
            "requested": cuda_graph_requested,
            "enabled": bool(resolved.get("cuda_graph_enabled_by_server_args")),
            "replay_runtime_observed": None,
            "evidence_boundary": (
                "requested/enabled are configuration facts; this record does not "
                "prove CUDA Graph replay"
            ),
        },
        "overlap_schedule": {
            "requested": overlap_requested,
            "enabled": bool(resolved.get("overlap_schedule_enabled_by_server_args")),
            "runtime_overlap_observed": None,
            "evidence_boundary": (
                "requested/enabled are configuration facts; this record does not "
                "prove concurrent runtime overlap"
            ),
        },
    }


def matched_lane_comparability(
    candidate: dict[str, object],
    baseline: dict[str, object],
    candidate_resolved: dict[str, object] | None = None,
    baseline_resolved: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fail-closed accounting for the controls behind the ``matched`` label."""

    control_mismatches = [
        {
            "field": field,
            "pypto": candidate.get(field),
            "sglang_matched": baseline.get(field),
        }
        for field in MATCHED_CONTROL_FIELDS
        if candidate.get(field) != baseline.get(field)
    ]
    if candidate_resolved is None or baseline_resolved is None:
        control_mismatches.append(
            {
                "field": "resolved_server_args",
                "pypto": "missing" if candidate_resolved is None else "present",
                "sglang_matched": (
                    "missing" if baseline_resolved is None else "present"
                ),
            }
        )
    else:
        control_mismatches.extend(
            {
                "field": f"resolved.{field}",
                "pypto": candidate_resolved.get(field),
                "sglang_matched": baseline_resolved.get(field),
            }
            for field in MATCHED_RESOLVED_CONTROL_FIELDS
            if candidate_resolved.get(field) != baseline_resolved.get(field)
        )
    implementation_differences = [
        {
            "field": field,
            "pypto": candidate.get(field),
            "sglang_matched": baseline.get(field),
            "reason": "implementation under test",
        }
        for field in IMPLEMENTATION_FIELDS
        if candidate.get(field) != baseline.get(field)
    ]
    if candidate_resolved is not None and baseline_resolved is not None:
        implementation_differences.extend(
            {
                "field": f"resolved.{field}",
                "pypto": candidate_resolved.get(field),
                "sglang_matched": baseline_resolved.get(field),
                "reason": "implementation under test",
            }
            for field in MATCHED_RESOLVED_IMPLEMENTATION_FIELDS
            if candidate_resolved.get(field) != baseline_resolved.get(field)
        )
    claim_allowed = not control_mismatches
    return {
        "status": (
            "matched_controls_with_implementation_differences"
            if claim_allowed
            else "unmatched_controls"
        ),
        "matched_claim_allowed": claim_allowed,
        "matched_control_fields": list(MATCHED_CONTROL_FIELDS),
        "matched_resolved_control_fields": list(MATCHED_RESOLVED_CONTROL_FIELDS),
        "control_mismatches": control_mismatches,
        "intentionally_unmatched_implementation_fields": implementation_differences,
        "evidence_boundary": (
            "The matched label covers listed controls only. Backend/provider fields "
            "are intentionally different because they are the implementation under test."
        ),
    }
