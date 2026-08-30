"""SGLang general-plugin entry point.

Registration is fail-closed and happens before ServerArgs construction in the
pinned SGLang release. Actual Attention/GDN classes are added after the
standalone operator ABI exists.
"""

from __future__ import annotations

import importlib
import atexit
import functools
import inspect
import json
import os
from pathlib import Path
import threading
import weakref

from .errors import BackendNotReadyError
from .operator_library import assert_operator_library_compatible
from .torch_inductor import assert_backend_executable_ready, install
from .versions import assert_sglang_compatible, assert_torch_compatible


LINEAR_BACKEND_RESOLVER_TARGET = (
    "sglang.srt.layers.attention.linear.utils.resolve_linear_attn_backends"
)
ATTENTION_WRAPPER_TARGET = (
    "sglang.srt.layers.attention.attention_registry.attn_backend_wrapper"
)
GDN_PROJECTION_TARGET = (
    "sglang.srt.models.qwen3_5.fused_qkvzba_split_reshape_cat_contiguous"
)
TRITON_SUPPORT_TARGETS = (
    "sglang.srt.utils.common.support_triton",
    "sglang.srt.mem_cache.allocation.support_triton",
    "sglang.srt.model_executor.forward_batch_info.support_triton",
    "sglang.srt.layers.rotary_embedding.mrope.support_triton",
)
GEMMA_RMSNORM_TARGET = "sglang.srt.layers.layernorm.GemmaRMSNorm._forward_impl"
GEMMA_RMSNORM_WEIGHT_LOADER_TARGET = (
    "sglang.srt.layers.layernorm.GemmaRMSNorm._weight_loader"
)
FLA_GATED_RMSNORM_TARGET = (
    "sglang.kernels.ops.attention.fla.layernorm_gated.layernorm_fn"
)
QK_RMSNORM_ROPE_GATE_TARGET = (
    "sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate."
    "fused_qk_gemma_rmsnorm_rope_gate"
)
FUSED_SIGMOID_MUL_TARGET = (
    "sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul"
)
UNQUANTIZED_LINEAR_TARGET = (
    "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply"
)
SILU_AND_MUL_TARGET = "sglang.srt.layers.activation.silu_and_mul"
SILU_FORWARD_CUDA_TARGET = "sglang.srt.layers.activation.SiluAndMul.forward_cuda"
LM_HEAD_TARGET = "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head"
EMBEDDING_TARGET = (
    "sglang.srt.layers.quantization.unquant.UnquantizedEmbeddingMethod.embedding"
)
PRUNED_STATES_TARGET = (
    "sglang.srt.layers.logits_processor.LogitsProcessor._get_pruned_states"
)
MAMBA_INDICES_TARGET = (
    "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.get_mamba_indices"
)
UNIFIED_MAMBA_TRANSLATE_TARGET = (
    "sglang.srt.mem_cache.unified_memory_pool.UnifiedMambaSlotAllocator.translate"
)
INPUT_BUFFER_STAGE_TARGETS = (
    "sglang.srt.model_executor.runner_utils.buffers."
    "DecodeInputBuffers.populate_from_forward_batch",
    "sglang.srt.model_executor.runner_utils.buffers."
    "PrefillInputBuffers.populate_from_forward_batch",
    "sglang.srt.model_executor.cuda_graph_buffer_registry."
    "CudaGraphBufferRegistry.fill_from",
)
POSITION_STAGE_TARGET = "sglang.srt.model_executor.forward_batch_info.compute_position"
QWEN_LANGUAGE_MODEL_ONLY_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration",)

_linear_prepack_lock = threading.RLock()
_linear_prepack_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_differential_lock = threading.RLock()
_differential_records: list[dict[str, object]] = []
_differential_writer_registered = False
_registration_lock = threading.RLock()
_registration_pid = os.getpid()
_registered = False


def _pypto_compute_selected() -> bool:
    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    return "pypto" in {
        getattr(mamba, "linear_attn_backend", None),
        getattr(mamba, "linear_attn_decode_backend", None),
        getattr(mamba, "linear_attn_prefill_backend", None),
        getattr(mamba, "linear_attn_verify_backend", None),
    }


def _vendor_reference_mode() -> bool:
    return os.environ.get("PYPTO_REFERENCE_VENDOR_COMPUTE") == "1"


def _record_differential(kind: str, name: str, candidate, reference) -> None:
    path = os.environ.get("PYPTO_DIFFERENTIAL_REPORT")
    if not path:
        return
    import torch

    torch.cuda.synchronize(candidate.device)
    difference = (candidate.detach().float() - reference.detach().float()).abs()
    record = {
        "candidate_dtype": str(candidate.dtype),
        "kind": kind,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "name": name,
        "reference_dtype": str(reference.dtype),
        "shape": list(candidate.shape),
    }
    global _differential_writer_registered
    with _differential_lock:
        _differential_records.append(record)
        if not _differential_writer_registered:

            def write_report() -> None:
                output = Path(path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        _differential_records,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            atexit.register(write_report)
            _differential_writer_registered = True


def _attention_factory(runner):
    from pypto_kernels import attention

    if attention.PAGED_DECODE_STATUS != "native-tile executable":
        raise BackendNotReadyError(
            "PyPTO paged attention remains a source candidate; refusing SGLang "
            "compute fallback before its compiler and numerical gates pass."
        )
    from .sglang.attention_backend import create_attention_backend

    return create_attention_backend(runner)


def _resolve_linear_backends_around(original_fn, prefill_default=None):
    """Force every selected GDN phase, including verify, to PyPTO."""

    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    selections = {
        mamba.linear_attn_backend,
        mamba.linear_attn_decode_backend,
        mamba.linear_attn_prefill_backend,
        mamba.linear_attn_verify_backend,
    }
    uses_pypto = "pypto" in selections
    decode_backend = mamba.linear_attn_decode_backend or mamba.linear_attn_backend
    prefill_backend = mamba.linear_attn_prefill_backend or mamba.linear_attn_backend
    verify_backend = mamba.linear_attn_verify_backend
    if uses_pypto and (decode_backend != "pypto" or prefill_backend != "pypto"):
        raise ValueError(
            "PyPTO linear attention requires both decode and prefill backends "
            f"to be 'pypto', got decode={decode_backend!r}, "
            f"prefill={prefill_backend!r}"
        )
    if uses_pypto and verify_backend not in (None, "pypto"):
        raise ValueError(
            "PyPTO linear attention requires verify backend 'pypto', "
            f"got {verify_backend!r}"
        )
    backends = original_fn(prefill_default=None if uses_pypto else prefill_default)
    if not uses_pypto:
        return backends
    return type(backends)(
        decode=type(backends.decode).CUSTOM,
        prefill=type(backends.prefill).CUSTOM,
        verify=type(backends.verify).CUSTOM,
    )


def _attention_wrapper_around(original_fn, runner, full_attn_backend):
    """Replace only a user-selected Qwen GDN side with the PyPTO adapter."""

    from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
    from sglang.srt.runtime_context import get_exec

    config = hybrid_gdn_config(runner.model_config)
    if config is None:
        return original_fn(runner, full_attn_backend)
    mamba = get_exec().mamba
    selected = {
        mamba.linear_attn_backend,
        mamba.linear_attn_decode_backend,
        mamba.linear_attn_prefill_backend,
        mamba.linear_attn_verify_backend,
    }
    if "pypto" not in selected:
        return original_fn(runner, full_attn_backend)

    from pypto_kernels import causal_conv1d, gdn

    if (
        causal_conv1d.STATUS != "native-tile stateful executable"
        or gdn.STATUS != "native-tile recurrent executable"
    ):
        raise BackendNotReadyError(
            "PyPTO stateful conv/GDN remain source candidates; refusing SGLang "
            "linear-attention fallback before compiler and numerical gates pass."
        )
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.utils import (
        resolve_linear_attn_backends,
    )

    from .sglang.gdn_backend import create_gdn_backend

    runner.linear_attn_backends = resolve_linear_attn_backends(prefill_default=None)
    full_attention_layers = (
        [0] if runner.is_draft_worker else config.full_attention_layer_ids
    )
    return HybridLinearAttnBackend(
        full_attn_backend,
        create_gdn_backend(runner),
        full_attention_layers,
    )


def _gdn_projection_around(
    original_fn,
    projected_qkvz,
    projected_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    """Route Qwen3.5's four-output projection copy through one PyPTO graph."""

    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    selected = {
        mamba.linear_attn_backend,
        mamba.linear_attn_decode_backend,
        mamba.linear_attn_prefill_backend,
    }
    if "pypto" not in selected:
        return original_fn(
            projected_qkvz,
            projected_ba,
            num_heads_qk,
            num_heads_v,
            head_qk,
            head_v,
        )
    from pypto_kernels import gdn_projection
    from .sglang.stream import pypto_stream

    if gdn_projection.STATUS != "native-tile packed executable":
        raise BackendNotReadyError(
            "PyPTO GDN projection split remains a source candidate; refusing "
            "the Qwen3.5 Triton copy fallback."
        )
    with pypto_stream(projected_qkvz.device) as stream:
        result = gdn_projection.split_projection(
            projected_qkvz,
            projected_ba,
            q_heads=num_heads_qk,
            value_heads=num_heads_v,
            key_dim=head_qk,
            value_dim=head_v,
            stream=stream,
        )
    if os.environ.get("PYPTO_DIFFERENTIAL_REPORT"):
        rows = int(projected_qkvz.shape[0])
        mixed_width = 2 * num_heads_qk * head_qk + num_heads_v * head_v
        reference = (
            projected_qkvz[:, :mixed_width].contiguous(),
            projected_qkvz[:, mixed_width:]
            .contiguous()
            .view(rows, num_heads_v, head_v),
            projected_ba[:, :num_heads_v].contiguous(),
            projected_ba[:, num_heads_v:].contiguous(),
        )
        for index, (candidate_value, reference_value) in enumerate(
            zip(result, reference, strict=True)
        ):
            _record_differential(
                "gdn_projection",
                f"projection_output_{index}",
                candidate_value,
                reference_value,
            )
    return result


def _support_triton_around(original_fn, backend):
    """Keep PyPTO scheduler metadata away from Triton-only dispatch."""

    if backend == "pypto":
        return False
    return original_fn(backend)


def _gemma_rmsnorm_around(
    original_fn,
    layer,
    x,
    residual=None,
    post_residual_addition=None,
):
    """Route Qwen Gemma RMSNorm and fused residual RMSNorm through PyPTO."""

    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    selected = {
        mamba.linear_attn_backend,
        mamba.linear_attn_decode_backend,
        mamba.linear_attn_prefill_backend,
    }
    if "pypto" not in selected:
        return original_fn(layer, x, residual, post_residual_addition)
    if post_residual_addition is not None:
        raise BackendNotReadyError(
            "PyPTO Gemma RMSNorm does not yet fuse post_residual_addition."
        )
    from pypto_kernels import fused_add_rmsnorm, rmsnorm
    from .sglang.stream import pypto_stream

    if (
        rmsnorm.STATUS != "native-tile executable"
        or fused_add_rmsnorm.STATUS != "native-tile executable"
    ):
        raise BackendNotReadyError("PyPTO Gemma RMSNorm operators are not executable.")
    original_shape = tuple(x.shape)
    flat = x if x.ndim == 2 else x.contiguous().reshape(-1, original_shape[-1])
    differential = bool(os.environ.get("PYPTO_DIFFERENTIAL_REPORT"))
    reference_x = x.clone() if differential and residual is not None else x
    reference_residual = (
        residual.clone() if differential and residual is not None else residual
    )

    def torch_reference(value):
        import torch

        wide = value.float()
        inverse = torch.rsqrt(
            wide.square().mean(dim=-1, keepdim=True) + layer.variance_epsilon
        )
        return (wide * inverse * (1.0 + layer.weight.data.float())).to(value.dtype)

    with pypto_stream(x.device) as stream:
        if residual is None:
            output = rmsnorm.rmsnorm(
                flat,
                layer.weight.data,
                layer.variance_epsilon,
                stream=stream,
            )
            result = output.reshape(original_shape)
            if differential:
                reference = torch_reference(x)
                _record_differential("gemma_rmsnorm", "rmsnorm", result, reference)
            return result
        if x.ndim != 2 or residual.ndim != 2:
            raise BackendNotReadyError(
                "PyPTO fused Gemma RMSNorm requires rank-2 residual inputs."
            )
        result = fused_add_rmsnorm.fused_add_rmsnorm(
            x,
            residual,
            layer.weight.data,
            layer.variance_epsilon,
            stream=stream,
        )
        if differential:
            reference_sum = reference_x + reference_residual
            reference = (torch_reference(reference_sum), reference_sum)
            for index, (candidate_value, reference_value) in enumerate(
                zip(result, reference, strict=True)
            ):
                _record_differential(
                    "gemma_rmsnorm",
                    f"fused_output_{index}",
                    candidate_value,
                    reference_value,
                )
        return result


def _gemma_rmsnorm_weight_loader_around(original_fn, layer, param, loaded_weight):
    """Keep Gemma's derived buffer colocated with an offloaded parameter."""

    derived = getattr(layer, "gemma_weight", None)
    param_device = getattr(param, "device", None)
    derived_device = getattr(derived, "device", None)
    if param_device is None or derived_device is None:
        raise BackendNotReadyError(
            "pinned Gemma RMSNorm weight-loader tensor contract changed"
        )
    if param_device != derived_device:
        if (
            getattr(param_device, "type", None) != "cpu"
            or getattr(derived_device, "type", None) != "cuda"
        ):
            raise BackendNotReadyError(
                "Gemma RMSNorm offload produced an unsupported device split"
            )
        layer.gemma_weight = derived.to(device=param_device)
    result = original_fn(layer, param, loaded_weight)
    if layer.gemma_weight.device != param.device:
        raise BackendNotReadyError(
            "Gemma RMSNorm derived weight did not follow its parameter"
        )
    return result


def _fla_gated_rmsnorm_around(
    original_fn,
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation="swish",
):
    """Route Qwen GDN output RMSNorm and SiLU gate through PyPTO."""

    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    selected = {
        mamba.linear_attn_backend,
        mamba.linear_attn_decode_backend,
        mamba.linear_attn_prefill_backend,
    }
    if "pypto" not in selected:
        return original_fn(
            x,
            weight,
            bias,
            z,
            eps,
            group_size,
            norm_before_gate,
            is_rms_norm,
            activation,
        )
    if (
        bias is not None
        or z is None
        or group_size is not None
        or not norm_before_gate
        or not is_rms_norm
        or activation != "swish"
        or float(eps) != 1.0e-6
    ):
        raise BackendNotReadyError(
            "PyPTO FLA gated RMSNorm requires bias=None, a gate tensor, "
            "group_size=None, norm_before_gate=True, is_rms_norm=True, "
            "activation='swish', and eps=1e-6."
        )
    if (
        x.ndim < 2
        or tuple(z.shape) != tuple(x.shape)
        or not x.is_contiguous()
        or not z.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise BackendNotReadyError(
            "PyPTO FLA gated RMSNorm requires matching contiguous x/z and "
            "a contiguous weight."
        )
    from pypto_kernels import gated_rmsnorm
    from .sglang.stream import pypto_stream

    if gated_rmsnorm.STATUS != "native-tile executable":
        raise BackendNotReadyError("PyPTO gated RMSNorm operator is not executable.")
    original_shape = tuple(x.shape)
    x_flat = x.reshape(-1, original_shape[-1])
    z_flat = z.reshape(-1, original_shape[-1])
    with pypto_stream(x.device) as stream:
        output = gated_rmsnorm.gated_rmsnorm(
            x_flat,
            z_flat,
            weight,
            eps=float(eps),
            stream=stream,
        )
    result = output.reshape(original_shape)
    if os.environ.get("PYPTO_DIFFERENTIAL_REPORT"):
        import torch

        wide = x.float()
        normalized = wide * torch.rsqrt(
            wide.square().mean(dim=-1, keepdim=True) + float(eps)
        )
        gate_wide = z.float()
        reference = (
            normalized * weight.float() * (gate_wide * torch.sigmoid(gate_wide))
        ).to(x.dtype)
        _record_differential("gated_rmsnorm", "gdn_output_norm", result, reference)
    return result


def _qk_rmsnorm_rope_gate_around(
    original_fn,
    q_gate,
    key,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate=True,
):
    """Route Qwen full-attention Q/K preparation through one PyPTO graph."""

    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    selected = {
        mamba.linear_attn_backend,
        mamba.linear_attn_decode_backend,
        mamba.linear_attn_prefill_backend,
    }
    if "pypto" not in selected:
        return original_fn(
            q_gate,
            key,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            has_gate=has_gate,
        )
    if (
        not has_gate
        or float(eps) != 1.0e-6
        or head_dim <= 0
        or head_dim % 128
        or rotary_dim <= 0
        or rotary_dim % 2
        or rotary_dim >= head_dim
        or q_weight.numel() != head_dim
        or k_weight.numel() != head_dim
        or cos_sin_cache.ndim != 2
        or int(cos_sin_cache.shape[1]) != rotary_dim
    ):
        raise BackendNotReadyError(
            "PyPTO fused Q/K preparation requires has_gate=True, eps=1e-6, "
            "head_dim divisible by 128, an even partial rotary_dim, matching "
            "head weights, and a matching rank-2 cos/sin cache."
        )
    from pypto_kernels import qk_rmsnorm_rope
    from .sglang.stream import pypto_stream

    if qk_rmsnorm_rope.STATUS != "native-tile executable":
        raise BackendNotReadyError(
            "PyPTO fused Q/K RMSNorm/RoPE operator is not executable."
        )
    if positions.ndim == 2:
        if (
            tuple(positions.shape) != (3, int(q_gate.shape[0]))
            or positions.stride(1) != 1
        ):
            raise BackendNotReadyError(
                "PyPTO fused Q/K preparation requires M-RoPE positions "
                "with shape [3, tokens] and unit inner stride."
            )
        positions = positions[0]
    differential = bool(os.environ.get("PYPTO_DIFFERENTIAL_REPORT"))
    reference_inputs = (
        (q_gate.clone(), key.clone(), positions.clone()) if differential else None
    )
    with pypto_stream(q_gate.device) as stream:
        result = qk_rmsnorm_rope.qk_rmsnorm_rope_gate(
            q_gate,
            key,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            q_heads=num_q_heads,
            kv_heads=num_kv_heads,
            stream=stream,
        )
    if differential:
        reference_q_gate, reference_key, reference_positions = reference_inputs
        import torch

        tokens = int(reference_q_gate.shape[0])
        q_interleaved = reference_q_gate.view(tokens, num_q_heads, 2, head_dim)
        q_source = q_interleaved[:, :, 0, :]
        gate_reference = q_interleaved[:, :, 1, :].contiguous()
        k_source = reference_key.view(tokens, num_kv_heads, head_dim)

        def normalize(value, weight):
            wide = value.float()
            inverse = torch.rsqrt(wide.square().mean(dim=-1, keepdim=True) + float(eps))
            return (wide * inverse * (1.0 + weight.float())).to(value.dtype).float()

        cache = cos_sin_cache.index_select(
            0, reference_positions.to(torch.int64)
        ).float()
        half = rotary_dim // 2
        cos = cache[:, :half].unsqueeze(1)
        sin = cache[:, half:].unsqueeze(1)

        def rotate(value):
            low = value[..., :half]
            high = value[..., half:rotary_dim]
            tail = value[..., rotary_dim:]
            return torch.cat(
                (low * cos - high * sin, high * cos + low * sin, tail),
                dim=-1,
            ).to(reference_q_gate.dtype)

        q_reference = rotate(normalize(q_source, q_weight)).reshape(
            tokens, num_q_heads * head_dim
        )
        k_reference = rotate(normalize(k_source, k_weight)).reshape(
            tokens, num_kv_heads * head_dim
        )
        reference = (q_reference, k_reference, gate_reference)
        for index, (candidate_value, reference_value) in enumerate(
            zip(result, reference, strict=True)
        ):
            _record_differential(
                "qk_rmsnorm_rope",
                f"qkv_output_{index}",
                candidate_value,
                reference_value,
            )
    return result


def _fused_sigmoid_mul_around(
    original_fn,
    attn_output,
    gate,
    inplace=False,
):
    """Route the full-attention output gate through PyPTO."""

    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    selected = {
        mamba.linear_attn_backend,
        mamba.linear_attn_decode_backend,
        mamba.linear_attn_prefill_backend,
    }
    if "pypto" not in selected:
        return original_fn(attn_output, gate, inplace=inplace)
    if attn_output.ndim != 2:
        raise BackendNotReadyError(
            "PyPTO fused sigmoid-mul requires rank-2 attention output."
        )
    if gate.ndim == 3:
        tokens, heads, head_dim = map(int, gate.shape)
        if (
            tuple(attn_output.shape) != (tokens, heads * head_dim)
            or gate.stride(2) != 1
            or gate.stride(1) != head_dim
            or gate.stride(0) < heads * head_dim
        ):
            raise BackendNotReadyError(
                "PyPTO fused sigmoid-mul received an incompatible rank-3 gate."
            )
        gate = gate.view(tokens, heads * head_dim)
    elif gate.ndim != 2 or tuple(gate.shape) != tuple(attn_output.shape):
        raise BackendNotReadyError(
            "PyPTO fused sigmoid-mul requires matching rank-2 values or a "
            "head-dense rank-3 gate."
        )
    from pypto_kernels import sigmoid_mul
    from .sglang.stream import pypto_stream

    if sigmoid_mul.STATUS != "native-tile executable":
        raise BackendNotReadyError("PyPTO sigmoid-mul operator is not executable.")
    differential = bool(os.environ.get("PYPTO_DIFFERENTIAL_REPORT"))
    reference_output = attn_output.clone() if differential else None
    reference_gate = gate.clone() if differential else None
    with pypto_stream(attn_output.device) as stream:
        result = sigmoid_mul.sigmoid_mul(
            attn_output,
            gate,
            stream=stream,
            inplace=bool(inplace),
        )
    if differential:
        import torch

        reference = (
            reference_output.float() * torch.sigmoid(reference_gate.float())
        ).to(reference_output.dtype)
        _record_differential("sigmoid_mul", "attention_output_gate", result, reference)
    return result


def _unquantized_linear_around(original_fn, method, layer, x, bias=None):
    """Route every selected BF16 SGLang linear through PyPTO matmul."""

    if not _pypto_compute_selected() or _vendor_reference_mode():
        return original_fn(method, layer, x, bias)
    import torch

    weight = getattr(layer, "weight", None)
    if (
        bias is not None
        or type(x) is not torch.Tensor
        or type(weight) not in (torch.Tensor, torch.nn.Parameter)
        or x.dtype is not torch.bfloat16
        or weight.dtype is not torch.bfloat16
        or not x.is_cuda
        or not weight.is_cuda
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise BackendNotReadyError(
            "PyPTO unquantized linear requires bias=None and exact contiguous "
            "CUDA BF16 input/weight tensors."
        )
    from pypto_kernels import linear
    from .sglang.stream import pypto_stream

    if linear.STATUS != "native-tile executable":
        raise BackendNotReadyError("PyPTO linear operator is not executable.")
    logical_features = int(weight.shape[0])
    launch_weight = weight.data
    if logical_features % 128:
        padded_features = ((logical_features + 127) // 128) * 128
        source_signature = getattr(weight, "_pypto_offload_source_signature", None)
        if source_signature is None:
            signature = (
                "resident",
                int(weight.data_ptr()),
                int(weight._version),
                tuple(weight.shape),
            )
        elif (
            not isinstance(source_signature, tuple)
            or len(source_signature) != 4
            or source_signature[0] != "offloaded"
            or source_signature[3] != tuple(weight.shape)
        ):
            raise BackendNotReadyError(
                "PyPTO linear received an invalid offloaded-weight identity."
            )
        else:
            signature = source_signature
        with _linear_prepack_lock:
            cached = _linear_prepack_cache.get(layer)
            if cached is None or cached[0] != signature:
                from .activity_trace import trace_window_active

                if trace_window_active():
                    raise BackendNotReadyError(
                        "PyPTO linear weight prepack was not completed before "
                        "the model-forward trace window."
                    )
                padded = torch.zeros(
                    (padded_features, int(weight.shape[1])),
                    dtype=weight.dtype,
                    device=weight.device,
                )
                padded[:logical_features].copy_(weight)
                cached = (signature, padded)
                _linear_prepack_cache[layer] = cached
            launch_weight = cached[1]
    with pypto_stream(x.device) as stream:
        output = linear.linear(x, launch_weight, stream=stream)
    output = output[..., :logical_features]
    if os.environ.get("PYPTO_DIFFERENTIAL_REPORT"):
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        try:
            reference = original_fn(method, layer, x, bias)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        _record_differential(
            "linear",
            str(getattr(layer, "prefix", type(layer).__name__)),
            output,
            reference,
        )
    return output


def _dispatch_swiglu(gate, up, out=None):
    """Select functional Inductor or mutation-declared handwritten execution."""

    if out is None:
        from .torch.inductor_swiglu import run_fp32_swiglu

        return run_fp32_swiglu(gate, up)
    # SGLang's caller-owned output contract cannot be represented by this
    # functional Inductor graph without an extra copy.  Keep the existing
    # mutation-declared handwritten PyPTO operator for that exact ABI.
    from pypto_kernels import silu_and_mul
    from .sglang.stream import pypto_stream

    if silu_and_mul.STATUS != "native-tile executable":
        raise BackendNotReadyError("PyPTO SiLU-and-mul operator is not executable.")
    with pypto_stream(gate.device) as stream:
        return silu_and_mul.silu_and_mul(
            gate,
            up,
            stream=stream,
            out=out,
        )


def _validated_swiglu_views(input, out=None):
    """Validate the pinned packed ABI and return its two row-pitched views."""

    import torch

    if (
        type(input) is not torch.Tensor
        or input.dtype is not torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or input.ndim != 2
        or int(input.shape[-1]) % 256
    ):
        raise BackendNotReadyError(
            "PyPTO SiLU-and-mul requires a rank-2 contiguous CUDA BF16 input "
            "whose packed width is divisible by 256."
        )
    half = int(input.shape[-1]) // 2
    if out is not None and (
        type(out) is not torch.Tensor
        or tuple(out.shape) != (int(input.shape[0]), half)
        or out.dtype is not input.dtype
        or not out.is_contiguous()
    ):
        raise BackendNotReadyError(
            "PyPTO SiLU-and-mul received an incompatible caller-owned output."
        )
    return input[:, :half], input[:, half:]


def _silu_and_mul_around(original_fn, input, out=None):
    """Route packed SwiGLU through Inductor; preserve caller-owned output ABI."""

    if not _pypto_compute_selected() or _vendor_reference_mode():
        return original_fn(input, out)
    gate, up = _validated_swiglu_views(input, out)
    result = _dispatch_swiglu(gate, up, out)
    if os.environ.get("PYPTO_DIFFERENTIAL_REPORT"):
        reference = original_fn(input, None)
        _record_differential("silu_and_mul", "packed_swiglu", result, reference)
    return result


def _silu_forward_cuda_around(original_fn, operator, input):
    """Bypass SGLang's preallocated output so real Qwen uses Inductor.

    The pinned ``SiluAndMul.forward_cuda`` always allocates ``out`` before
    calling the module-level primitive.  Hooking only that primitive would
    therefore select the handwritten mutation path for every model call.  This
    class-method hook retains the public functional return ABI and sends the
    real model call through the standard ``torch.compile(backend='pypto')``
    route without adding an ATen copy kernel.
    """

    if not _pypto_compute_selected() or _vendor_reference_mode():
        return original_fn(operator, input)
    gate, up = _validated_swiglu_views(input)
    result = _dispatch_swiglu(gate, up, None)
    if os.environ.get("PYPTO_DIFFERENTIAL_REPORT"):
        import torch

        reference = (torch.nn.functional.silu(gate.float()) * up.float()).to(
            torch.bfloat16
        )
        _record_differential("silu_and_mul", "packed_swiglu", result, reference)
    return result


def _lm_head_around(
    original_fn,
    processor,
    hidden_states,
    lm_head,
    embedding_bias=None,
):
    """Route the unquantized Qwen vocabulary projection through PyPTO."""

    if not _pypto_compute_selected() or _vendor_reference_mode():
        return original_fn(processor, hidden_states, lm_head, embedding_bias)
    import torch

    weight = getattr(lm_head, "weight", None)
    quant_method = getattr(lm_head, "quant_method", None)
    use_fp32_lm_head = bool(getattr(processor, "use_fp32_lm_head", False))
    if (
        embedding_bias is not None
        or type(hidden_states) is not torch.Tensor
        or type(weight) is not torch.nn.Parameter
        or type(quant_method).__name__ != "UnquantizedEmbeddingMethod"
        or getattr(processor, "rl_on_policy_target", None) is not None
        or hasattr(lm_head, "set_lora")
        or hidden_states.dtype is not torch.bfloat16
        or weight.dtype is not torch.bfloat16
        or not hidden_states.is_cuda
        or not weight.is_cuda
        or not hidden_states.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise BackendNotReadyError(
            "PyPTO LM head requires the plain contiguous CUDA BF16 Qwen "
            "embedding weight with no bias, LoRA, or quantization."
        )
    from pypto_kernels import linear
    from .sglang.stream import pypto_stream

    with pypto_stream(hidden_states.device) as stream:
        # Match SGLang's default ``use_fp32_lm_head=false`` contract: the
        # regular path returns BF16 logits, while an explicit FP32 request
        # uses the widened graph.  Keeping this branch here also makes the
        # correctness-only FP32 mode auditable instead of silently rejecting it.
        result = (
            linear.linear_to_float(hidden_states, weight.data, stream=stream)
            if use_fp32_lm_head
            else linear.linear(hidden_states, weight.data, stream=stream)
        )
    if os.environ.get("PYPTO_DIFFERENTIAL_REPORT"):
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        try:
            reference = original_fn(processor, hidden_states, lm_head, embedding_bias)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        _record_differential("lm_head", "lm_head", result, reference.float())
    return result


def _embedding_around(original_fn, method, layer, input_):
    """Route the unquantized token embedding gather through PyPTO."""

    if not _pypto_compute_selected() or _vendor_reference_mode():
        return original_fn(method, layer, input_)
    import torch

    weight = getattr(layer, "weight", None)
    if (
        type(input_) is not torch.Tensor
        or type(weight) is not torch.nn.Parameter
        or input_.ndim != 1
        or input_.dtype is not torch.int64
        or weight.ndim != 2
        or weight.dtype is not torch.bfloat16
        or not input_.is_cuda
        or not weight.is_cuda
        or not input_.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise BackendNotReadyError(
            "PyPTO embedding requires contiguous CUDA INT64 token ids and a "
            "contiguous CUDA BF16 weight."
        )
    from pypto_kernels import embedding
    from .sglang.stream import pypto_stream

    if embedding.STATUS != "native-tile executable":
        raise BackendNotReadyError("PyPTO embedding operator is not executable.")
    with pypto_stream(input_.device) as stream:
        return embedding.embedding(input_, weight.data, stream=stream)


def _pruned_states_around(
    original_fn,
    processor,
    hidden_states,
    hidden_states_before_norm,
    aux_hidden_states,
    logits_metadata,
):
    """Keep decode states or use a zero-copy last row for batch-one prefill."""

    if not _pypto_compute_selected() or _vendor_reference_mode():
        return original_fn(
            processor,
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            logits_metadata,
        )
    mode = logits_metadata.forward_mode
    if mode.is_decode_or_idle():
        if getattr(logits_metadata, "draft_extend_select_index", None) is not None:
            raise BackendNotReadyError(
                "PyPTO logits pruning does not support draft-select decode."
            )
        aux_pruned_states = (
            aux_hidden_states
            if aux_hidden_states is None or hasattr(aux_hidden_states, "shape")
            else list(aux_hidden_states)
        )
        return (
            hidden_states,
            hidden_states_before_norm,
            aux_pruned_states,
            None,
            None,
            [],
        )
    lengths = logits_metadata.extend_seq_lens_cpu
    if (
        not mode.is_extend()
        or logits_metadata.extend_return_logprob
        or lengths is None
        or len(lengths) != 1
        or int(lengths[0]) != int(hidden_states.shape[0])
        or hidden_states_before_norm is not None
        or aux_hidden_states is not None
    ):
        raise BackendNotReadyError(
            "PyPTO logits pruning requires one plain prefill request without "
            "input logprobs or auxiliary hidden states."
        )
    return hidden_states[-1:], None, None, None, None, []


def _integer_gather(table, indices):
    import torch

    if (
        type(table) is not torch.Tensor
        or type(indices) is not torch.Tensor
        or table.ndim != 1
        or table.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 1
        or indices.dtype is not torch.int64
        or not table.is_cuda
        or not indices.is_cuda
        or not table.is_contiguous()
        or not indices.is_contiguous()
        or table.device != indices.device
    ):
        raise BackendNotReadyError(
            "PyPTO Mamba slot translation requires a contiguous CUDA integer "
            "table and contiguous CUDA INT64 indices."
        )
    from pypto_kernels import embedding
    from .sglang.stream import pypto_stream

    with pypto_stream(table.device) as stream:
        return embedding.integer_gather(table, indices, stream=stream)


def _mamba_indices_around(original_fn, pool, req_indices):
    if not _pypto_compute_selected() or _vendor_reference_mode():
        return original_fn(pool, req_indices)
    return _integer_gather(pool.req_index_to_mamba_index_mapping, req_indices)


def _unified_mamba_translate_around(original_fn, allocator, virtual_ids):
    if not _pypto_compute_selected() or _vendor_reference_mode():
        return original_fn(allocator, virtual_ids)
    return _integer_gather(allocator.virtual_to_physical, virtual_ids)


def _input_buffer_stage_around(original_fn, *args, **kwargs):
    from .activity_trace import annotate_framework_activity

    with annotate_framework_activity("sglang.input-buffer-staging"):
        return original_fn(*args, **kwargs)


def _position_stage_around(original_fn, *args, **kwargs):
    from .activity_trace import annotate_framework_activity

    with annotate_framework_activity("sglang.position-staging"):
        return original_fn(*args, **kwargs)


def _require_callable_hook_target(target: str) -> None:
    parts = target.split(".")
    for count in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:count])
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                continue
            raise
        for attribute in parts[count:]:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if callable(value):
            return
        break
    raise BackendNotReadyError(f"pinned SGLang hook target is not callable: {target}")


def _enable_qwen_language_model_only(server_args_class=None) -> tuple[str, ...]:
    """Enable SGLang's implemented text-only construction for Qwen3.5.

    The pinned Qwen3.5 model honors ``config.language_model_only`` by omitting
    its vision tower, but SGLang's command-line admission whitelist has not yet
    listed the architecture. Keep that compatibility adaptation outside the
    vendored SGLang tree and fail closed if its class-level contract changes.
    """

    if server_args_class is None:
        from sglang.srt.server_args import ServerArgs

        server_args_class = ServerArgs
    current = getattr(server_args_class, "LANGUAGE_MODEL_ONLY_ARCHITECTURES", None)
    if not isinstance(current, tuple) or not all(
        isinstance(name, str) and name for name in current
    ):
        raise BackendNotReadyError(
            "pinned SGLang language-model-only architecture contract changed"
        )
    additions = tuple(
        name for name in QWEN_LANGUAGE_MODEL_ONLY_ARCHITECTURES if name not in current
    )
    updated = (*current, *additions)
    server_args_class.LANGUAGE_MODEL_ONLY_ARCHITECTURES = updated
    return updated


def _enable_deterministic_pypto_attention(server_args_module=None) -> tuple[str, ...]:
    """Admit PyPTO deterministic attention without claiming radix support."""

    if server_args_module is None:
        server_args_module = importlib.import_module("sglang.srt.server_args")
    deterministic = getattr(
        server_args_module,
        "DETERMINISTIC_ATTENTION_BACKEND_CHOICES",
        None,
    )
    radix = getattr(
        server_args_module,
        "RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND",
        None,
    )
    add_choice = getattr(
        server_args_module,
        "add_deterministic_attention_backend_choices",
        None,
    )
    if (
        not isinstance(deterministic, list)
        or not all(isinstance(name, str) and name for name in deterministic)
        or not isinstance(radix, list)
        or not all(isinstance(name, str) and name for name in radix)
        or not callable(add_choice)
    ):
        raise BackendNotReadyError(
            "pinned SGLang deterministic attention registration contract changed"
        )
    if "pypto" in radix:
        raise BackendNotReadyError(
            "PyPTO must not be registered for deterministic radix attention"
        )
    if "pypto" not in deterministic:
        add_choice(["pypto"])
    if deterministic.count("pypto") != 1 or "pypto" in radix:
        raise BackendNotReadyError(
            "PyPTO deterministic attention registration is inconsistent"
        )
    return tuple(deterministic)


def _qwen_text_only_weight_name(name: str) -> bool:
    return not (
        name.startswith(("model.visual.", "visual.", "model.mtp.", "mtp."))
        or ".mtp." in name
    )


def _enable_qwen_text_only_weight_filter(loader_module=None):
    """Skip optional Qwen tensors before safetensors materializes them."""

    if loader_module is None:
        loader_module = importlib.import_module("sglang.srt.model_loader.loader")
    current = getattr(loader_module, "safetensors_weights_iterator", None)
    if not callable(current):
        raise BackendNotReadyError(
            "pinned SGLang safetensors iterator contract changed"
        )
    if getattr(current, "_pypto_qwen_text_only_filter", False):
        return current

    @functools.wraps(current)
    def filtered_iterator(
        hf_weights_files,
        disable_mmap=False,
        prefetch=False,
        prefetch_num_threads=4,
        drop_cache_after_load=False,
    ):
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
        architectures = tuple(
            getattr(server_args.get_model_config().hf_config, "architectures", ())
        )
        enabled = bool(
            server_args.language_model_only
            and any(
                architecture in QWEN_LANGUAGE_MODEL_ONLY_ARCHITECTURES
                for architecture in architectures
            )
        )
        if not enabled or disable_mmap or prefetch:
            yield from current(
                hf_weights_files,
                disable_mmap=disable_mmap,
                prefetch=prefetch,
                prefetch_num_threads=prefetch_num_threads,
                drop_cache_after_load=drop_cache_after_load,
            )
            return

        import safetensors
        from sglang.srt.model_loader import weight_utils

        for st_file in hf_weights_files:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if _qwen_text_only_weight_name(name):
                        yield name, handle.get_tensor(name)
            if drop_cache_after_load:
                weight_utils._drop_file_cache_after_load(st_file)

    filtered_iterator._pypto_qwen_text_only_filter = True
    loader_module.safetensors_weights_iterator = filtered_iterator
    return filtered_iterator


def _enable_offloader_tied_parameter_support(offloader_module=None):
    """Make OffloaderV1 functional calls explicit about Qwen parameter aliases."""

    if offloader_module is None:
        offloader_module = importlib.import_module("sglang.srt.utils.offloader")
    current = getattr(offloader_module, "functional_call", None)
    if not callable(current):
        raise BackendNotReadyError(
            "pinned SGLang offloader functional_call contract changed"
        )
    if getattr(current, "_pypto_untied_parameter_compatible", False):
        return current
    try:
        parameters = inspect.signature(current).parameters
    except (TypeError, ValueError) as error:
        raise BackendNotReadyError(
            "pinned SGLang offloader functional_call is not inspectable"
        ) from error
    if "tie_weights" not in parameters:
        raise BackendNotReadyError(
            "pinned SGLang offloader functional_call lacks tie_weights"
        )

    @functools.wraps(current)
    def functional_call_with_explicit_aliases(*args, **kwargs):
        if "tie_weights" in kwargs and kwargs["tie_weights"] is not False:
            raise BackendNotReadyError(
                "PyPTO Qwen offload requires functional_call tie_weights=False"
            )
        kwargs["tie_weights"] = False
        if len(args) >= 2 and isinstance(args[1], dict):
            module = args[0]
            replacements = args[1]
            try:
                parameters = dict(module.named_parameters(remove_duplicate=False))
            except (AttributeError, TypeError) as error:
                raise BackendNotReadyError(
                    "pinned SGLang offloader module parameter contract changed"
                ) from error
            for name, replacement in replacements.items():
                original = parameters.get(name)
                if original is None or not hasattr(replacement, "shape"):
                    continue
                replacement._pypto_offload_source_signature = (
                    "offloaded",
                    int(original.data_ptr()),
                    int(original._version),
                    tuple(original.shape),
                )
        return current(*args, **kwargs)

    functional_call_with_explicit_aliases._pypto_untied_parameter_compatible = True
    offloader_module.functional_call = functional_call_with_explicit_aliases
    return functional_call_with_explicit_aliases


def _register_impl() -> None:
    assert_operator_library_compatible()
    assert_backend_executable_ready()
    assert_torch_compatible()
    assert_sglang_compatible()
    _enable_qwen_language_model_only()
    _enable_deterministic_pypto_attention()
    _enable_qwen_text_only_weight_filter()
    _enable_offloader_tied_parameter_support()
    install()

    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )
    from sglang.srt.server_args import (
        add_attention_backend_choices,
        add_linear_attn_kernel_backend_choices,
    )
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    hook_targets = (
        LINEAR_BACKEND_RESOLVER_TARGET,
        ATTENTION_WRAPPER_TARGET,
        GDN_PROJECTION_TARGET,
        GEMMA_RMSNORM_TARGET,
        GEMMA_RMSNORM_WEIGHT_LOADER_TARGET,
        FLA_GATED_RMSNORM_TARGET,
        QK_RMSNORM_ROPE_GATE_TARGET,
        FUSED_SIGMOID_MUL_TARGET,
        UNQUANTIZED_LINEAR_TARGET,
        SILU_AND_MUL_TARGET,
        SILU_FORWARD_CUDA_TARGET,
        LM_HEAD_TARGET,
        EMBEDDING_TARGET,
        PRUNED_STATES_TARGET,
        MAMBA_INDICES_TARGET,
        UNIFIED_MAMBA_TRANSLATE_TARGET,
        POSITION_STAGE_TARGET,
        *TRITON_SUPPORT_TARGETS,
    )
    for target in hook_targets:
        _require_callable_hook_target(target)

    add_attention_backend_choices(["pypto"])
    add_linear_attn_kernel_backend_choices(["pypto"])
    register_attention_backend("pypto")(_attention_factory)
    HookRegistry.register(
        LINEAR_BACKEND_RESOLVER_TARGET,
        _resolve_linear_backends_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        ATTENTION_WRAPPER_TARGET,
        _attention_wrapper_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        GDN_PROJECTION_TARGET,
        _gdn_projection_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        GEMMA_RMSNORM_TARGET,
        _gemma_rmsnorm_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        GEMMA_RMSNORM_WEIGHT_LOADER_TARGET,
        _gemma_rmsnorm_weight_loader_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        FLA_GATED_RMSNORM_TARGET,
        _fla_gated_rmsnorm_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        QK_RMSNORM_ROPE_GATE_TARGET,
        _qk_rmsnorm_rope_gate_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        FUSED_SIGMOID_MUL_TARGET,
        _fused_sigmoid_mul_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        UNQUANTIZED_LINEAR_TARGET,
        _unquantized_linear_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        SILU_AND_MUL_TARGET,
        _silu_and_mul_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        SILU_FORWARD_CUDA_TARGET,
        _silu_forward_cuda_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        LM_HEAD_TARGET,
        _lm_head_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        EMBEDDING_TARGET,
        _embedding_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        PRUNED_STATES_TARGET,
        _pruned_states_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        MAMBA_INDICES_TARGET,
        _mamba_indices_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        UNIFIED_MAMBA_TRANSLATE_TARGET,
        _unified_mamba_translate_around,
        HookType.AROUND,
    )
    for target in INPUT_BUFFER_STAGE_TARGETS:
        HookRegistry.register(target, _input_buffer_stage_around, HookType.AROUND)
    HookRegistry.register(
        POSITION_STAGE_TARGET,
        _position_stage_around,
        HookType.AROUND,
    )
    for target in TRITON_SUPPORT_TARGETS:
        HookRegistry.register(target, _support_triton_around, HookType.AROUND)


def register() -> None:
    """Register the pinned SGLang adapter or terminate fail-closed.

    SGLang's general-plugin loader deliberately catches ordinary
    :class:`Exception` instances so one optional plugin cannot take down the
    server. That policy is unsafe for a user-selected strict compute backend:
    logging a compatibility error and continuing could silently select a
    default provider. ``SystemExit`` derives from ``BaseException`` rather than
    ``Exception``, so it crosses that loader boundary and stops the worker.
    """
    global _registered
    current_pid = os.getpid()
    if current_pid != _registration_pid:
        raise SystemExit(
            "PyPTO SGLang registration state was inherited across fork; use spawn/exec"
        )
    with _registration_lock:
        if _registered:
            return
        try:
            _register_impl()
        except Exception as exc:
            raise SystemExit(f"PyPTO SGLang plugin registration failed: {exc}") from exc
        _registered = True
