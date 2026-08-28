"""Self-contained exact-DSO bootstrap for the operator library."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import threading
from typing import Any

DSO_PATH = os.environ.get(
    "PYPTO_KERNEL_DSO_PATH",
    (
        "/home/zhaosiying/pypto-love-tensor-ir/builds/"
        "pypto-opext-on-a589f79/product/"
        "pypto_core.cpython-314-x86_64-linux-gnu.so"
    ),
)
PYPTO_PACKAGE = os.environ.get(
    "PYPTO_KERNEL_PACKAGE_PATH",
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto/python/pypto",
)

_lock = threading.RLock()
_modules: dict[str, Any] | None = None


def bootstrap() -> dict[str, Any]:
    """Bind the exact-DSO pypto package once per process."""

    global _modules
    with _lock:
        if _modules is not None:
            return _modules
    resolved = pathlib.Path(DSO_PATH).resolve(strict=True)
    package_root = pathlib.Path(PYPTO_PACKAGE).resolve(strict=True)
    occupied = sorted(
        name for name in sys.modules if name == "pypto" or name.startswith("pypto.")
    )
    if occupied:
        raise RuntimeError(
            "foreign pypto modules occupy exact-bootstrap slots: " + ", ".join(occupied)
        )
    removed_finders: list[tuple[int, Any]] = []
    probe_path = [str(package_root / "jit")]
    for index, finder in tuple(enumerate(sys.meta_path)):
        try:
            candidate = finder.find_spec("pypto.jit.decorator", probe_path, None)
        except (AttributeError, ImportError, TypeError, ValueError):
            continue
        origin = getattr(candidate, "origin", None)
        if not origin:
            continue
        try:
            resolved_origin = pathlib.Path(origin).resolve(strict=True)
        except OSError:
            continue
        if package_root not in resolved_origin.parents:
            removed_finders.append((index, finder))
    if removed_finders:
        rejected = {id(finder) for _, finder in removed_finders}
        sys.meta_path[:] = [
            finder for finder in sys.meta_path if id(finder) not in rejected
        ]
    spec = importlib.util.spec_from_file_location(
        "pypto",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    assert spec is not None and spec.loader is not None
    pypto = importlib.util.module_from_spec(spec)
    core_spec = importlib.util.spec_from_file_location("pypto.pypto_core", resolved)
    assert core_spec is not None and core_spec.loader is not None
    core = importlib.util.module_from_spec(core_spec)
    sys.modules["pypto"] = pypto
    sys.modules["pypto.pypto_core"] = core
    try:
        core_spec.loader.exec_module(core)
        spec.loader.exec_module(pypto)
    except BaseException:
        for name in tuple(sys.modules):
            if name == "pypto" or name.startswith("pypto."):
                sys.modules.pop(name, None)
        for index, finder in removed_finders:
            sys.meta_path.insert(min(index, len(sys.meta_path)), finder)
        raise
    with _lock:
        _modules = {"pypto": pypto, "ir": pypto.ir, "compiler": pypto.compiler}
    return _modules


EXPECTED_DRIVER = "610.74"
EXPECTED_RUNTIME = (
    "/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/lib/"
    "python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13"
)


def compile_graph(program: Any, tiles: list[int]) -> str:
    """Compile one graph through the strict facade; return its cache key.

    The library premise is "one operator = one graph", so compilation is
    per-graph and cached by (program identity, tiles).
    """

    import hashlib

    from pypto.compiler import (
        CanonicalSchedule,
        CompileRequest,
        ScheduleParameter,
        ScheduleValueKind,
    )

    modules = bootstrap()
    compiler = modules["compiler"]
    info = compiler.get_nvidia_backend_build_info()
    if not info.compiled:
        raise RuntimeError("PyPTO exact DSO backend is not compiled")

    import torch

    torch.zeros(1, device="cuda")
    from pypto.runtime import nvidia as runtime

    observation = runtime.observe_current_nvidia_runtime(
        EXPECTED_DRIVER, EXPECTED_RUNTIME
    )
    from pypto.compiler import ToolchainIdentity

    toolchain = ToolchainIdentity(
        info.pypto_revision,
        info.tensor_ir_revision,
        info.cuda_tile_revision,
        info.llvm_revision,
        info.cuda_toolkit_root,
        info.cuda_toolkit_version,
        info.tileiras_real_path,
        info.tileiras_version,
        info.tileiras_sha256,
    )
    request = CompileRequest(observation.target_info, toolchain)
    parameter = ScheduleParameter
    unsigned = ScheduleValueKind.UnsignedInteger
    tile_parameters = [
        parameter(f"dim_{index:03d}", unsigned, str(tile))
        for index, tile in enumerate(tiles)
    ]
    schedule = CanonicalSchedule(
        [parameter("codegen_strategy", ScheduleValueKind.Text, "layout-propagation")],
        tile_parameters,
        [],
        [],
        [parameter("count", unsigned, "1")],
        [parameter("count", unsigned, "4")],
        [],
        [
            parameter("bytecode_major", unsigned, "13"),
            parameter("bytecode_minor", unsigned, "3"),
            parameter("bytecode_tag", unsigned, "0"),
            parameter("max_candidates", unsigned, "0"),
            parameter("uniform_signature", ScheduleValueKind.Boolean, "false"),
        ],
    )
    result = compiler.compile_structured_strict(program, request, schedule)
    artifact = result.artifact
    key = hashlib.sha256(bytes(artifact.device_code)).hexdigest()[:16]
    with _lock:
        _GRAPHS[key] = (artifact, request)
    return key


def compile_jit_kernel(kernel: Any, samples: tuple[Any, ...], tiles: list[int]) -> str:
    """Specialize a native ``@pl.jit`` tile kernel and compile that IR.

    This boundary preserves the user's
    ``pl.at`` / ``pl.range`` / ``tile.load`` / ``tile.store`` source.  The
    NVIDIA frontend lifts the statically complete tile loop nest into one
    TensorIR graph and the resulting operator still launches exactly once.
    """

    return compile_graph(kernel.specialize(*samples), tiles)


_GRAPHS: dict[str, tuple[Any, Any]] = {}
_EXECUTABLES: dict[str, Any] = {}


def _ready_executable(key: str) -> Any:
    """Return one process/context-bound executable retained for server life."""

    from pypto.runtime import nvidia as runtime

    with _lock:
        executable = _EXECUTABLES.get(key)
        if executable is not None:
            return executable
        artifact, request = _GRAPHS[key]
        executable = runtime.NvidiaExecutable(artifact, request)
        observation = runtime.observe_current_nvidia_runtime(
            EXPECTED_DRIVER, EXPECTED_RUNTIME
        )
        executable.prewarm(observation.cuda_runtime_api_version)
        _EXECUTABLES[key] = executable
        return executable


def acquire_cuda_graph_leases(
    excluded_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Lease every newly warmed module for a framework CUDA-graph owner."""

    excluded = excluded_keys or set()
    with _lock:
        return {
            key: executable.acquire_cuda_graph_lease()
            for key, executable in sorted(_EXECUTABLES.items())
            if key not in excluded
        }


def launch_graph(key: str, tensors: tuple[Any, ...], stream: Any) -> None:
    """Launch one compiled graph (the single launch of one operator)."""

    from pypto.runtime import nvidia as runtime

    artifact, request = _GRAPHS[key]
    del request
    executable = _ready_executable(key)
    descriptors = artifact.kernel_abi.argument_layout.operand_descriptors
    if len(descriptors) != len(tensors):
        raise ValueError(
            "PyPTO launch tensor count differs from the static Artifact ABI: "
            f"expected={len(descriptors)} actual={len(tensors)}"
        )
    for index, (descriptor, tensor) in enumerate(zip(descriptors, tensors)):
        expected_shape = list(descriptor.shape)
        expected_stride = list(descriptor.strides)
        actual_shape = list(tensor.shape)
        actual_stride = list(tensor.stride())
        if expected_shape != actual_shape or expected_stride != actual_stride:
            raise ValueError(
                f"PyPTO launch operand {index} differs from static Artifact ABI: "
                f"expected_shape={expected_shape} actual_shape={actual_shape} "
                f"expected_stride={expected_stride} actual_stride={actual_stride}"
            )
    arguments = [
        runtime.NvidiaLaunchArgument.tensor(
            int(t.data_ptr()), list(t.shape), list(t.stride())
        )
        for t in tensors
    ]
    packet = executable.prepare_launch(arguments)
    executable.launch(packet, stream)
    del packet


def classify(program: Any, tiles: list[int]) -> dict[str, str]:
    """Classify a single-graph operator for status reporting.

    'compiled'        — the graph lowers and is executable today.
    'producer-blocked'— the HIR graph is VALID (emission accepted it) and
                        only the pinned producer lowering rejects it; this
                        is exactly the codex L0 (broadcast lowering) marker.
    'hir-rejected'    — the graph itself is invalid.
    """

    try:
        key = compile_graph(program, tiles)
        return {"status": "compiled", "key": key}
    except RuntimeError as error:  # producer-stage failure
        return {"status": "producer-blocked", "error": str(error)[:200]}
    except ValueError as error:  # HIR emission rejection
        return {"status": "hir-rejected", "error": str(error)[:200]}
