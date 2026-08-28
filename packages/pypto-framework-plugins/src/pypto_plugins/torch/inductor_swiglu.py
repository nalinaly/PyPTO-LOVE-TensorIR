"""Revision-bound Inductor execution for packed FP32 SwiGLU pointwise work."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from typing import Any, Callable

from ..activity_trace import trace_window_active
from ..errors import BackendNotReadyError, StrictCoverageError
from ..versions import EXPECTED_TORCH_COMMIT
from . import pointwise_codegen, runtime_bridge
from .registration import PYPTO_BACKEND_HASH
from .scheduling import REGISTRY


@dataclass(frozen=True, slots=True)
class TensorCallIdentity:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    storage_offset: int
    dtype_name: str
    device_type: str
    device_index: int


@dataclass(frozen=True, slots=True)
class CallableRevisionIdentity:
    torch_revision: str
    backend_hash: str
    compiler: pointwise_codegen.BackendRevisionIdentity


@dataclass(frozen=True, slots=True)
class SwiGLUCallableKey:
    gate: TensorCallIdentity
    up: TensorCallIdentity
    output_strides: tuple[int, ...]
    revisions: CallableRevisionIdentity


@dataclass(frozen=True, slots=True)
class _CallableEntry:
    function: Callable[[Any, Any], Any]
    kernel_name: str
    source_node: str


_lock = threading.RLock()
_OWNER_PID = os.getpid()
_revision_identity: CallableRevisionIdentity | None = None
_callable_cache: dict[SwiGLUCallableKey, _CallableEntry] = {}


def _require_owner_process() -> None:
    current = os.getpid()
    if current != _OWNER_PID:
        raise StrictCoverageError(
            "PyPTO Inductor callable cache was inherited across fork; use spawn/exec"
        )


def fp32_swiglu_subgraph(gate: Any, up: Any) -> Any:
    """Compute SiLU(gate) * up in FP32 and materialize BF16 output."""

    import torch

    gate_wide = gate.to(torch.float32)
    up_wide = up.to(torch.float32)
    return (gate_wide * torch.sigmoid(gate_wide) * up_wide).to(torch.bfloat16)


def _tensor_identity(value: Any) -> TensorCallIdentity:
    import torch

    if type(value) is not torch.Tensor:
        raise BackendNotReadyError("PyPTO Inductor SwiGLU requires exact tensors")
    device_index = value.device.index
    if device_index is None and value.device.type == "cuda":
        device_index = int(torch.cuda.current_device())
    if device_index is None:
        device_index = 0
    return TensorCallIdentity(
        tuple(int(extent) for extent in value.shape),
        tuple(int(stride) for stride in value.stride()),
        int(value.storage_offset()),
        str(value.dtype).removeprefix("torch."),
        value.device.type,
        int(device_index),
    )


def _validate_operands(gate: Any, up: Any) -> tuple[TensorCallIdentity, TensorCallIdentity]:
    import torch

    gate_identity = _tensor_identity(gate)
    up_identity = _tensor_identity(up)
    if (
        gate_identity.shape != up_identity.shape
        or len(gate_identity.shape) != 2
        or any(extent <= 0 for extent in gate_identity.shape)
        or gate_identity.dtype_name != "bfloat16"
        or up_identity.dtype_name != "bfloat16"
        or gate_identity.device_type != "cuda"
        or up_identity.device_type != "cuda"
        or gate_identity.device_index != up_identity.device_index
        or gate_identity.strides[-1] != 1
        or up_identity.strides[-1] != 1
        or gate_identity.shape[-1] % 128
        or gate.requires_grad
        or up.requires_grad
        or gate.dtype is not torch.bfloat16
        or up.dtype is not torch.bfloat16
    ):
        raise BackendNotReadyError(
            "PyPTO Inductor SwiGLU requires matching rank-2 CUDA BF16 operands "
            "with unit inner stride and width divisible by 128"
        )
    return gate_identity, up_identity


def _current_revisions() -> CallableRevisionIdentity:
    _require_owner_process()
    global _revision_identity
    with _lock:
        if _revision_identity is not None:
            return _revision_identity
        if trace_window_active():
            raise StrictCoverageError(
                "PyPTO Inductor compiler identity was not prepared before trace"
            )
        import torch

        torch_revision = str(torch.version.git_version)
        if torch_revision != EXPECTED_TORCH_COMMIT:
            raise BackendNotReadyError(
                "PyPTO Inductor SwiGLU observed an unpinned Torch revision"
            )
        _revision_identity = CallableRevisionIdentity(
            torch_revision,
            PYPTO_BACKEND_HASH,
            pointwise_codegen.current_backend_revision_identity(),
        )
        return _revision_identity


def callable_key(gate: Any, up: Any) -> SwiGLUCallableKey:
    _require_owner_process()
    gate_identity, up_identity = _validate_operands(gate, up)
    columns = gate_identity.shape[-1]
    return SwiGLUCallableKey(
        gate_identity,
        up_identity,
        (columns, 1),
        _current_revisions(),
    )


def _compile_callable() -> Callable[[Any, Any], Any]:
    import torch

    return torch.compile(
        fp32_swiglu_subgraph,
        backend="pypto",
        fullgraph=True,
        dynamic=False,
    )


def _validate_output(result: Any, key: SwiGLUCallableKey) -> None:
    import torch

    if (
        type(result) is not torch.Tensor
        or tuple(result.shape) != key.gate.shape
        or tuple(result.stride()) != key.output_strides
        or result.dtype is not torch.bfloat16
        or result.device.type != key.gate.device_type
        or int(result.device.index or 0) != key.gate.device_index
    ):
        raise StrictCoverageError(
            "PyPTO Inductor SwiGLU returned an incompatible output"
        )


def run_fp32_swiglu(gate: Any, up: Any) -> Any:
    """Run one cached, prewarmed, fail-closed Inductor-generated kernel."""

    _require_owner_process()
    key = callable_key(gate, up)
    with _lock:
        entry = _callable_cache.get(key)
        if entry is not None:
            if entry.kernel_name not in REGISTRY:
                raise StrictCoverageError(
                    "cached PyPTO Inductor callable lost its artifact registration"
                )
            if not runtime_bridge.kernel_is_prewarmed(entry.kernel_name):
                raise StrictCoverageError(
                    "cached PyPTO Inductor callable lost its executable prewarm"
                )
            result = entry.function(gate, up)
            _validate_output(result, key)
            return result

        if trace_window_active():
            raise StrictCoverageError(
                "PyPTO Inductor SwiGLU callable was not prepared before trace"
            )
        with pointwise_codegen.capture_pointwise_artifacts() as capture:
            compiled = _compile_callable()
            result = compiled(gate, up)
        _validate_output(result, key)
        artifact = capture.single_artifact()
        kernel_name = artifact.kernel_name
        if kernel_name not in REGISTRY:
            raise StrictCoverageError(
                "captured PyPTO artifact has no stable registry binding"
            )
        registered = REGISTRY.get(kernel_name)
        if registered != artifact or (
            not artifact.source_node.startswith("torch-inductor:")
            or artifact.fallback_used
            or not runtime_bridge.kernel_is_prewarmed(kernel_name)
        ):
            raise StrictCoverageError(
                "PyPTO Inductor SwiGLU artifact is not native and prewarmed"
            )
        _callable_cache[key] = _CallableEntry(
            compiled,
            kernel_name,
            artifact.source_node,
        )
        return result


def callable_cache_snapshot() -> tuple[tuple[SwiGLUCallableKey, str, str], ...]:
    _require_owner_process()
    with _lock:
        return tuple(
            (key, entry.kernel_name, entry.source_node)
            for key, entry in sorted(
                _callable_cache.items(),
                key=lambda item: repr(item[0]),
            )
        )


def callable_source_evidence(
    gate: Any,
    up: Any,
) -> pointwise_codegen.PointwiseSourceEvidence:
    """Return fail-closed source evidence for the exact cached callable."""

    _require_owner_process()
    key = callable_key(gate, up)
    with _lock:
        entry = _callable_cache.get(key)
        if entry is None:
            raise StrictCoverageError(
                "PyPTO Inductor SwiGLU source was requested before compilation"
            )
        if entry.kernel_name not in REGISTRY:
            raise StrictCoverageError(
                "PyPTO Inductor SwiGLU source lost its artifact registration"
            )
        artifact = REGISTRY.get(entry.kernel_name)
        if (
            artifact.kernel_name != entry.kernel_name
            or artifact.source_node != entry.source_node
        ):
            raise StrictCoverageError(
                "PyPTO Inductor SwiGLU callable and artifact source bindings differ"
            )
        return pointwise_codegen.pointwise_source_evidence(artifact)


def clear_callable_cache_for_testing() -> None:
    _require_owner_process()
    global _revision_identity
    with _lock:
        _callable_cache.clear()
        _revision_identity = None


__all__ = (
    "CallableRevisionIdentity",
    "SwiGLUCallableKey",
    "TensorCallIdentity",
    "callable_cache_snapshot",
    "callable_key",
    "callable_source_evidence",
    "clear_callable_cache_for_testing",
    "fp32_swiglu_subgraph",
    "run_fp32_swiglu",
)
