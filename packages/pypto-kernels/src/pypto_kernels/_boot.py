"""Self-contained exact-DSO bootstrap for the operator library."""

from __future__ import annotations

from contextlib import nullcontext
import ctypes
import hashlib
import importlib
import importlib.util
import os
import pathlib
import sys
import threading
from typing import Any

DSO_PATH = os.environ.get("PYPTO_KERNEL_DSO_PATH")
PYPTO_PACKAGE = os.environ.get("PYPTO_KERNEL_PACKAGE_PATH")
_DRIVER_LABEL_ENV = "PYPTO_KERNEL_CUDA_DRIVER_LABEL"
_CUDART_PATH_ENV = "PYPTO_KERNEL_CUDART"

_lock = threading.RLock()
_modules: dict[str, Any] | None = None
_sources_revision: str | None = None
_runtime_expectation: tuple[str, str] | None = None


def _kernel_sources_revision() -> str:
    """Return an exact digest of the deployed operator-library sources."""

    global _sources_revision
    with _lock:
        if _sources_revision is not None:
            return _sources_revision
        package_root = pathlib.Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for source in sorted(package_root.glob("*.py"), key=lambda path: path.name):
            digest.update(source.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
        _sources_revision = "sha256:" + digest.hexdigest()
        return _sources_revision


def bootstrap() -> dict[str, Any]:
    """Bind installed PyPTO or one explicitly requested diagnostic DSO."""

    global _modules
    with _lock:
        if _modules is not None:
            return _modules
    if DSO_PATH is None and PYPTO_PACKAGE is None:
        pypto = importlib.import_module("pypto")
        core = importlib.import_module("pypto.pypto_core")
        modules = {
            "pypto": pypto,
            "ir": pypto.ir,
            "compiler": pypto.compiler,
            "core": core,
        }
        with _lock:
            if _modules is None:
                _modules = modules
            return _modules
    if DSO_PATH is None:
        raise RuntimeError(
            "PYPTO_KERNEL_PACKAGE_PATH requires PYPTO_KERNEL_DSO_PATH"
        )
    resolved = pathlib.Path(DSO_PATH).resolve(strict=True)
    if resolved.is_dir():
        matches = sorted(resolved.glob("pypto_core*.so"))
        if len(matches) != 1:
            raise RuntimeError(
                "PYPTO_KERNEL_DSO_PATH directory must contain exactly one "
                "pypto_core shared library"
            )
        resolved = matches[0].resolve(strict=True)
    if PYPTO_PACKAGE is None:
        package_spec = importlib.util.find_spec("pypto")
        if package_spec is None or package_spec.origin is None:
            raise RuntimeError(
                "explicit PyPTO DSO bootstrap cannot locate the installed package"
            )
        package_root = pathlib.Path(package_spec.origin).resolve(strict=True).parent
    else:
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
        _modules = {
            "pypto": pypto,
            "ir": pypto.ir,
            "compiler": pypto.compiler,
            "core": core,
        }
    return _modules


def loaded_dso_path() -> pathlib.Path:
    """Return the concrete extension selected by :func:`bootstrap`."""

    value = getattr(bootstrap()["core"], "__file__", None)
    if not isinstance(value, str):
        raise RuntimeError("loaded pypto_core has no concrete file")
    return pathlib.Path(value).resolve(strict=True)


class _DlInfo(ctypes.Structure):
    _fields_ = (
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    )


def _loaded_symbol_provider(symbol_name: str) -> str:
    process = ctypes.CDLL(None)
    try:
        symbol = getattr(process, symbol_name)
        dladdr = process.dladdr
    except AttributeError as error:
        raise RuntimeError(
            f"the process has no loaded {symbol_name} provider"
        ) from error
    dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(_DlInfo))
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info)) != 1:
        raise RuntimeError(f"dladdr failed for loaded symbol {symbol_name}")
    if not info.dli_fname:
        raise RuntimeError(f"dladdr returned no provider for {symbol_name}")
    return str(pathlib.Path(os.fsdecode(info.dli_fname)).resolve(strict=True))


def _driver_api_label() -> str:
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        query = driver.cuDriverGetVersion
    except (OSError, AttributeError) as error:
        raise RuntimeError("CUDA Driver API version is unavailable") from error
    query.argtypes = (ctypes.POINTER(ctypes.c_int),)
    query.restype = ctypes.c_int
    value = ctypes.c_int()
    status = int(query(ctypes.byref(value)))
    if status != 0 or value.value <= 0:
        raise RuntimeError(
            f"cuDriverGetVersion failed with status={status}, value={value.value}"
        )
    return f"cuda-driver-api-{value.value}"


def _live_runtime_expectation() -> tuple[str, str]:
    """Resolve strict expectations from live providers or explicit overrides."""

    global _runtime_expectation
    with _lock:
        if _runtime_expectation is not None:
            return _runtime_expectation
    driver_label = os.environ.get(_DRIVER_LABEL_ENV)
    if driver_label is None:
        driver_label = _driver_api_label()
    elif not driver_label or driver_label != driver_label.strip():
        raise RuntimeError(f"{_DRIVER_LABEL_ENV} must be non-empty and trimmed")
    runtime_override = os.environ.get(_CUDART_PATH_ENV)
    runtime_path = (
        str(pathlib.Path(runtime_override).resolve(strict=True))
        if runtime_override
        else _loaded_symbol_provider("cudaRuntimeGetVersion")
    )
    value = (driver_label, runtime_path)
    with _lock:
        _runtime_expectation = value
    return value


def compile_graph(
    program: Any,
    tiles: list[int],
    *,
    provider: str = "pypto.tensorir",
    source_node: str | None = None,
) -> str:
    """Compile one graph through the strict facade; return its cache key.

    The library premise is "one operator = one graph", so compilation is
    per-graph and cached by (program identity, tiles).
    """

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

    observation = runtime.observe_current_nvidia_runtime(*_live_runtime_expectation())
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
    try:
        from pypto_plugins.activity_trace import artifact_record_from_runtime
    except ImportError:
        if os.environ.get("PYPTO_STRICT_COVERAGE") == "1":
            raise RuntimeError(
                "strict coverage requires pypto-framework-plugins"
            ) from None
        artifact_record = None
    else:
        artifact_record = artifact_record_from_runtime(
            artifact,
            provider=provider,
            source_node=source_node
            or f"pypto-kernels:{artifact.kernel_abi.entry_function_name}",
            kernels_revision=_kernel_sources_revision(),
        )
    with _lock:
        _GRAPHS[key] = (artifact, request)
        _GRAPH_RECORDS[key] = artifact_record
    return key


def compile_jit_kernel(
    kernel: Any,
    samples: tuple[Any, ...],
    tiles: list[int],
    *,
    provider: str = "pypto.tensorir",
    source_node: str | None = None,
) -> str:
    """Specialize a native ``@pl.jit`` tile kernel and compile that IR.

    This boundary preserves the user's
    ``pl.at`` / ``pl.range`` / ``tile.load`` / ``tile.store`` source.  The
    NVIDIA frontend lifts the statically complete tile loop nest into one
    TensorIR graph and the resulting operator still launches exactly once.
    """

    return compile_graph(
        kernel.specialize(*samples),
        tiles,
        provider=provider,
        source_node=source_node
        or f"{getattr(kernel, '__module__', 'pypto_kernels')}:{getattr(kernel, '__name__', type(kernel).__name__)}",
    )


_GRAPHS: dict[str, tuple[Any, Any]] = {}
_EXECUTABLES: dict[str, Any] = {}
_GRAPH_RECORDS: dict[str, Any] = {}


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
            *_live_runtime_expectation()
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
    artifact_record = _GRAPH_RECORDS.get(key)
    annotation = nullcontext()
    if artifact_record is not None:
        from pypto_plugins.activity_trace import annotate_artifact_launch

        annotation = annotate_artifact_launch(artifact_record)
    with annotation:
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
