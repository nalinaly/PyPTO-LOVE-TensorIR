"""Wrapper-side launch bridge for Inductor-generated PyPTO kernels.

The generated Inductor Python wrapper imports ``pypto_launch`` from this
module and calls it with the kernel name, the launch arguments and the
raw current stream. The bridge owns the plugin side of the PyPTO
``NvidiaExecutable`` lifecycle: it observes the live NVIDIA runtime once,
binds each registered artifact to a process/device context, prewarms it
outside graph capture, prepares an allocation-free launch packet from
pointer/shape/stride arguments and launches on the caller's current
stream. No kernel algorithm lives here.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

from ..activity_trace import (
    annotate_artifact_launch,
    artifact_record_from_runtime,
    trace_window_active,
)
from ..errors import StrictCoverageError
from .runtime_identity import resolve_live_runtime_expectation

_lock = threading.RLock()
_OWNER_PID = os.getpid()
_observations: dict[int, Any] = {}
_executables: dict[str, tuple[str, str, int, Any]] = {}
_kernel_executable_identities: dict[str, tuple[str, str, int]] = {}
_artifact_records: dict[str, Any] = {}


def _require_owner_process() -> None:
    current = os.getpid()
    if current != _OWNER_PID:
        raise StrictCoverageError(
            "PyPTO runtime caches were inherited across fork; use spawn/exec "
            f"(owner_pid={_OWNER_PID}, current_pid={current})"
        )


def _ensure_observation(runtime: Any, device_index: int) -> Any:
    _require_owner_process()
    with _lock:
        observation = _observations.get(device_index)
        if observation is None:
            expected = resolve_live_runtime_expectation()
            observation = runtime.observe_current_nvidia_runtime(
                expected.driver_label,
                expected.cuda_runtime_library_path,
            )
            _observations[device_index] = observation
        return observation


def _ensure_executable(
    runtime: Any,
    name: str,
    artifact: Any,
    request: Any,
    dso_sha256: str,
    device_index: int,
) -> Any:
    _require_owner_process()
    identity = str(artifact.identity_digest)
    artifact_sha = hashlib.sha256(bytes(artifact.serialize())).hexdigest()
    with _lock:
        executable_identity = (identity, dso_sha256, device_index)
        previous_identity = _kernel_executable_identities.setdefault(
            name,
            executable_identity,
        )
        if previous_identity != executable_identity:
            raise StrictCoverageError(
                f"PyPTO kernel name {name!r} was rebound to another artifact"
            )
        retained = _executables.get(identity)
        if retained is None:
            if trace_window_active():
                raise StrictCoverageError(
                    f"PyPTO executable {name!r} was not prewarmed before trace"
                )
            observation = _ensure_observation(runtime, device_index)
            executable = runtime.NvidiaExecutable(artifact, request)
            executable.prewarm(observation.cuda_runtime_api_version)
            _executables[identity] = (
                artifact_sha,
                dso_sha256,
                device_index,
                executable,
            )
            return executable
        previous_sha, previous_dso_sha, previous_device, executable = retained
        if (
            previous_sha != artifact_sha
            or previous_dso_sha != dso_sha256
            or previous_device != device_index
        ):
            raise StrictCoverageError(
                f"PyPTO artifact identity collision for {identity!r}"
            )
        return executable


def _ensure_artifact_record(
    name: str,
    artifact: Any,
    source_node: str,
    dso_sha256: str,
) -> Any:
    _require_owner_process()
    identity = str(artifact.identity_digest)
    with _lock:
        record = _artifact_records.get(identity)
        if record is None:
            if trace_window_active():
                raise StrictCoverageError(
                    f"PyPTO artifact provenance {identity!r} was not prewarmed"
                )
            record = artifact_record_from_runtime(
                artifact,
                provider="pypto.generic",
                source_node=source_node,
                kernels_revision=f"pypto-dso-sha256:{dso_sha256}",
            )
            _artifact_records[identity] = record
        elif (
            record.source_node != source_node
            or record.provider != "pypto.generic"
            or record.kernels_revision != f"pypto-dso-sha256:{dso_sha256}"
        ):
            raise StrictCoverageError(
                f"conflicting provenance for PyPTO artifact {identity!r}"
            )
        return record


def prewarm_kernel(kernel_name: str) -> None:
    """Create and prewarm one retained executable outside a trace window."""

    _require_owner_process()
    from . import pointwise_codegen

    from pypto.runtime import nvidia as runtime
    import torch

    with _lock:
        if trace_window_active():
            raise StrictCoverageError(
                "PyPTO executable prewarm is forbidden during trace"
            )
        retained = pointwise_codegen.runtime_objects(kernel_name)
        if retained is None:
            raise StrictCoverageError(
                f"PyPTO kernel {kernel_name!r} has no retained runtime objects"
            )
        artifact, request = retained
        source_node = pointwise_codegen.runtime_source_node(kernel_name)
        if source_node is None:
            raise StrictCoverageError(
                f"PyPTO kernel {kernel_name!r} has no retained source node"
            )
        device_index = pointwise_codegen.runtime_device_index(kernel_name)
        if device_index is None:
            raise StrictCoverageError(
                f"PyPTO kernel {kernel_name!r} has no retained CUDA device"
            )
        dso_sha256 = pointwise_codegen.runtime_dso_sha256(kernel_name)
        if dso_sha256 is None:
            raise StrictCoverageError(
                f"PyPTO kernel {kernel_name!r} has no retained DSO identity"
            )
        with torch.cuda.device(device_index):
            _ensure_executable(
                runtime,
                kernel_name,
                artifact,
                request,
                dso_sha256,
                device_index,
            )
        _ensure_artifact_record(kernel_name, artifact, source_node, dso_sha256)
        if trace_window_active():
            raise StrictCoverageError(
                "a trace window began during PyPTO runtime prewarm transaction"
            )


def kernel_is_prewarmed(kernel_name: str) -> bool:
    """Return whether ``kernel_name`` resolves to a ready cached executable."""

    _require_owner_process()
    with _lock:
        identity_pair = _kernel_executable_identities.get(kernel_name)
        if identity_pair is None:
            return False
        identity, dso_sha256, device_index = identity_pair
        executable = _executables.get(identity)
        record = _artifact_records.get(identity)
        return (
            executable is not None
            and executable[1] == dso_sha256
            and executable[2] == device_index
            and record is not None
            and record.kernels_revision == f"pypto-dso-sha256:{dso_sha256}"
        )


def pypto_launch(kernel_name: str, args: tuple[Any, ...], stream: int) -> None:
    """Launch a registered PyPTO kernel on the caller's current stream."""

    _require_owner_process()
    from . import pointwise_codegen
    from .scheduling import REGISTRY

    if kernel_name not in REGISTRY:
        raise StrictCoverageError(
            f"PyPTO kernel {kernel_name!r} is not registered in this process"
        )
    retained = pointwise_codegen.runtime_objects(kernel_name)
    if retained is None:
        raise StrictCoverageError(
            f"PyPTO kernel {kernel_name!r} has no retained runtime objects"
        )
    artifact, request = retained
    source_node = pointwise_codegen.runtime_source_node(kernel_name)
    if source_node is None or not source_node.startswith("torch-inductor:"):
        raise StrictCoverageError(
            f"PyPTO kernel {kernel_name!r} has no stable Inductor source node"
        )
    expected_device_index = pointwise_codegen.runtime_device_index(kernel_name)
    if expected_device_index is None:
        raise StrictCoverageError(
            f"PyPTO kernel {kernel_name!r} has no retained CUDA device"
        )
    dso_sha256 = pointwise_codegen.runtime_dso_sha256(kernel_name)
    if dso_sha256 is None:
        raise StrictCoverageError(
            f"PyPTO kernel {kernel_name!r} has no retained DSO identity"
        )
    from pypto.runtime import nvidia as runtime
    import torch

    current_device_index = int(torch.cuda.current_device())
    if current_device_index != expected_device_index:
        raise StrictCoverageError(
            f"PyPTO kernel {kernel_name!r} expects cuda:{expected_device_index}, "
            f"current device is cuda:{current_device_index}"
        )
    for tensor in args:
        device = getattr(tensor, "device", None)
        index = getattr(device, "index", None)
        if getattr(device, "type", None) != "cuda" or int(index or 0) != expected_device_index:
            raise StrictCoverageError(
                f"PyPTO kernel {kernel_name!r} received an operand on {device!r}"
            )
    caller_stream = torch.cuda.current_stream()
    if int(caller_stream.cuda_stream) != int(stream):
        raise StrictCoverageError(
            "generated PyPTO wrapper did not pass the caller's current stream"
        )
    executable = _ensure_executable(
        runtime,
        kernel_name,
        artifact,
        request,
        dso_sha256,
        expected_device_index,
    )
    arguments = [
        runtime.NvidiaLaunchArgument.tensor(
            int(tensor.data_ptr()),
            list(tensor.shape),
            list(tensor.stride()),
        )
        for tensor in args
    ]
    try:
        packet = executable.prepare_launch(arguments)
    except Exception as error:
        descriptors = []
        try:
            for descriptor in artifact.kernel_abi.argument_layout.operand_descriptors:
                descriptors.append(
                    (str(descriptor.kind).rsplit(".", 1)[-1], list(descriptor.shape))
                )
        except Exception:  # pragma: no cover - diagnostics only
            pass
        raise type(error)(
            f"{kernel_name}: {error}; args="
            + repr([(tuple(t.shape), tuple(t.stride())) for t in args])
            + "; abi="
            + repr(descriptors)
        ) from error
    artifact_record = _ensure_artifact_record(
        kernel_name,
        artifact,
        source_node,
        dso_sha256,
    )
    with annotate_artifact_launch(artifact_record):
        executable.launch(packet, int(stream))
    del packet
