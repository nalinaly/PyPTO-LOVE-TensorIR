"""Frozen SGLang configurations for the three release lanes."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

from .workload import LANES, ReleaseContractError


BASELINE_LANES = {"sglang-matched", "sglang-optimized"}


def prepare_worker_environment(lane: str) -> None:
    if lane not in LANES:
        raise ValueError(f"unknown lane: {lane}")
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
    lane: str, optimized_memory_mode: str = "zero-offload"
) -> dict[str, object]:
    if lane in {"pypto", "sglang-matched"}:
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
            "mem_fraction_static": 0.78,
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
        "language_model_only": True,
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
    memory = memory_qualification(lane, optimized_memory_mode)
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
        # The pinned Triton GDN path calls a removed tl.make_block_ptr API.
        # FlashInfer is SGLang's supported SM100+ GDN route and requires BF16
        # state.  Both requested and post-resolution values are recorded.
        common.update(
            {
                "attention_backend": "flashinfer",
                "linear_attn_backend": "flashinfer",
                "linear_attn_decode_backend": "flashinfer",
                "linear_attn_prefill_backend": "flashinfer",
                "mamba_ssm_dtype": "bfloat16",
                "sampling_backend": "flashinfer",
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
    return common


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
        "torch_compile_requested": bool(
            getattr(server_args, "enable_torch_compile", False)
        ),
        "disable_cuda_graph": bool(getattr(server_args, "disable_cuda_graph", False)),
        "disable_overlap_schedule": bool(
            getattr(server_args, "disable_overlap_schedule", False)
        ),
        "disable_radix_cache": bool(getattr(server_args, "disable_radix_cache", False)),
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
        return
    required = (
        "attention_prefill",
        "attention_decode",
        "linear_attention_default",
        "linear_attention_prefill",
        "linear_attention_decode",
    )
    drift = {
        key: record.get(key) for key in required if record.get(key) != "flashinfer"
    }
    if drift or record.get("mamba_ssm_dtype") != "bfloat16":
        raise ReleaseContractError(
            "stock backend did not resolve to supported FlashInfer/BF16 state: "
            f"drift={drift}, dtype={record.get('mamba_ssm_dtype')!r}"
        )
