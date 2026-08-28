"""SGLang general-plugin entry point.

Registration is fail-closed and happens before ServerArgs construction in the
pinned SGLang release. Actual Attention/GDN classes are added after the
standalone operator ABI exists.
"""

from __future__ import annotations

import importlib

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
        return gdn_projection.split_projection(
            projected_qkvz,
            projected_ba,
            q_heads=num_heads_qk,
            value_heads=num_heads_v,
            key_dim=head_qk,
            value_dim=head_v,
            stream=stream,
        )


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
    with pypto_stream(x.device) as stream:
        if residual is None:
            output = rmsnorm.rmsnorm(
                flat,
                layer.weight.data,
                layer.variance_epsilon,
                stream=stream,
            )
            return output.reshape(original_shape)
        if x.ndim != 2 or residual.ndim != 2:
            raise BackendNotReadyError(
                "PyPTO fused Gemma RMSNorm requires rank-2 residual inputs."
            )
        return fused_add_rmsnorm.fused_add_rmsnorm(
            x,
            residual,
            layer.weight.data,
            layer.variance_epsilon,
            stream=stream,
        )


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
        raise BackendNotReadyError(
            "PyPTO gated RMSNorm operator is not executable."
        )
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
    return output.reshape(original_shape)


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
    with pypto_stream(q_gate.device) as stream:
        return qk_rmsnorm_rope.qk_rmsnorm_rope_gate(
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
        raise BackendNotReadyError(
            "PyPTO sigmoid-mul operator is not executable."
        )
    with pypto_stream(attn_output.device) as stream:
        return sigmoid_mul.sigmoid_mul(
            attn_output,
            gate,
            stream=stream,
            inplace=bool(inplace),
        )


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


def _register_impl() -> None:
    assert_operator_library_compatible()
    assert_backend_executable_ready()
    assert_torch_compatible()
    assert_sglang_compatible()
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
        FLA_GATED_RMSNORM_TARGET,
        QK_RMSNORM_ROPE_GATE_TARGET,
        FUSED_SIGMOID_MUL_TARGET,
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
    try:
        _register_impl()
    except Exception as exc:
        raise SystemExit(f"PyPTO SGLang plugin registration failed: {exc}") from exc
