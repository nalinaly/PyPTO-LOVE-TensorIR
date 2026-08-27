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

import threading
from typing import Any

from ..errors import StrictCoverageError

_EXPECTED_DRIVER_RELEASE = "610.74"
_EXPECTED_RUNTIME_LIBRARY = (
    "/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/lib/"
    "python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13"
)

_lock = threading.RLock()
_observation: Any = None
_executables: dict[str, Any] = {}


def _ensure_observation(runtime: Any) -> Any:
    global _observation
    with _lock:
        if _observation is None:
            _observation = runtime.observe_current_nvidia_runtime(
                _EXPECTED_DRIVER_RELEASE, _EXPECTED_RUNTIME_LIBRARY
            )
        return _observation


def _ensure_executable(
    runtime: Any, name: str, artifact: Any, request: Any
) -> Any:
    global _executables
    with _lock:
        executable = _executables.get(name)
        if executable is None:
            observation = _ensure_observation(runtime)
            executable = runtime.NvidiaExecutable(artifact, request)
            executable.prewarm(observation.cuda_runtime_api_version)
            _executables[name] = executable
        return executable


def pypto_launch(kernel_name: str, args: tuple[Any, ...], stream: int) -> None:
    """Launch a registered PyPTO kernel on the caller's current stream."""

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
    from pypto.runtime import nvidia as runtime

    executable = _ensure_executable(runtime, kernel_name, artifact, request)
    arguments = [
        runtime.NvidiaLaunchArgument.tensor(
            int(tensor.data_ptr()),
            list(tensor.shape),
            list(tensor.stride()),
        )
        for tensor in args
    ]
    packet = executable.prepare_launch(arguments)
    executable.launch(packet, stream)
    del packet
