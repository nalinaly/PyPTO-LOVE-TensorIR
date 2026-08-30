"""Dependency-free, read-only NVML queries for the bounded GPU controller."""

from __future__ import annotations

import ctypes
from ctypes import c_char_p, c_int, c_uint, c_ulonglong, c_void_p
from ctypes.util import find_library
from pathlib import Path


NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 6
_NAME_BYTES = 96
_DRIVER_BYTES = 80
_MAX_PROCESS_ENTRIES = 4096
_MIB = 1024 * 1024


class NvmlError(RuntimeError):
    """Raised when the local NVML library cannot answer a query."""


class _Memory(ctypes.Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pid", c_uint),
        ("used_gpu_memory", c_ulonglong),
        ("gpu_instance_id", c_uint),
        ("compute_instance_id", c_uint),
    ]


def _library_candidates() -> tuple[str, ...]:
    candidates = [
        "/usr/lib/wsl/lib/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    ]
    discovered = find_library("nvidia-ml")
    if discovered:
        candidates.append(discovered)
    return tuple(dict.fromkeys(candidates))


def _load_library() -> ctypes.CDLL:
    errors: list[str] = []
    for candidate in _library_candidates():
        if "/" in candidate and not Path(candidate).is_file():
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    detail = "; ".join(errors) if errors else "no libnvidia-ml candidate"
    raise NvmlError(f"cannot load NVML: {detail}")


def _function(lib: ctypes.CDLL, name: str, argtypes: list[object]):
    try:
        function = getattr(lib, name)
    except AttributeError as error:
        raise NvmlError(f"NVML symbol is missing: {name}") from error
    function.argtypes = argtypes
    function.restype = c_int
    return function


def _check(status: int, operation: str) -> None:
    if int(status) != NVML_SUCCESS:
        raise NvmlError(f"{operation} returned NVML status {int(status)}")


class _Session:
    def __enter__(self) -> ctypes.CDLL:
        self.lib = _load_library()
        init = getattr(self.lib, "nvmlInit_v2", None) or getattr(
            self.lib, "nvmlInit", None
        )
        if init is None:
            raise NvmlError("NVML initialization symbol is missing")
        init.restype = c_int
        _check(init(), "nvmlInit")
        return self.lib

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        shutdown = getattr(self.lib, "nvmlShutdown", None)
        if shutdown is not None:
            shutdown.restype = c_int
            shutdown()


def _device_handle(lib: ctypes.CDLL) -> c_void_p:
    symbol = (
        "nvmlDeviceGetHandleByIndex_v2"
        if hasattr(lib, "nvmlDeviceGetHandleByIndex_v2")
        else "nvmlDeviceGetHandleByIndex"
    )
    function = _function(lib, symbol, [c_uint, ctypes.POINTER(c_void_p)])
    handle = c_void_p()
    _check(function(0, ctypes.byref(handle)), symbol)
    return handle


def _read_text(lib: ctypes.CDLL, symbol: str, handle: c_void_p, size: int) -> str:
    function = _function(lib, symbol, [c_void_p, c_char_p, c_uint])
    buffer = ctypes.create_string_buffer(size)
    _check(function(handle, buffer, size), symbol)
    return buffer.value.decode("utf-8", errors="replace")


def query_identity() -> dict[str, str]:
    """Return the identity fields consumed by the frozen preflight contract."""

    with _Session() as lib:
        handle = _device_handle(lib)
        name = _read_text(lib, "nvmlDeviceGetName", handle, _NAME_BYTES)
        driver = _function(lib, "nvmlSystemGetDriverVersion", [c_char_p, c_uint])
        driver_buffer = ctypes.create_string_buffer(_DRIVER_BYTES)
        _check(driver(driver_buffer, _DRIVER_BYTES), "nvmlSystemGetDriverVersion")
        memory = _Memory()
        memory_function = _function(
            lib, "nvmlDeviceGetMemoryInfo", [c_void_p, ctypes.POINTER(_Memory)]
        )
        _check(memory_function(handle, ctypes.byref(memory)), "nvmlDeviceGetMemoryInfo")
        capability = _function(
            lib,
            "nvmlDeviceGetCudaComputeCapability",
            [c_void_p, ctypes.POINTER(c_int), ctypes.POINTER(c_int)],
        )
        major = c_int()
        minor = c_int()
        _check(
            capability(handle, ctypes.byref(major), ctypes.byref(minor)),
            "nvmlDeviceGetCudaComputeCapability",
        )
        return {
            "name": name,
            "compute_capability": f"{major.value}.{minor.value}",
            "memory_mib": str(int(memory.total) // _MIB),
            "used_mib": str(int(memory.used) // _MIB),
            "driver": driver_buffer.value.decode("utf-8", errors="replace"),
        }


def query_compute_pids() -> set[int]:
    """Return active compute PIDs through NVML's read-only process query."""

    with _Session() as lib:
        handle = _device_handle(lib)
        symbol = (
            "nvmlDeviceGetComputeRunningProcesses_v3"
            if hasattr(lib, "nvmlDeviceGetComputeRunningProcesses_v3")
            else "nvmlDeviceGetComputeRunningProcesses_v2"
        )
        function = _function(
            lib,
            symbol,
            [c_void_p, ctypes.POINTER(c_uint), ctypes.POINTER(_ProcessInfo)],
        )
        count = c_uint(0)
        status = int(function(handle, ctypes.byref(count), None))
        if status == NVML_SUCCESS:
            return set()
        if status != NVML_ERROR_INSUFFICIENT_SIZE:
            raise NvmlError(f"{symbol} returned NVML status {status}")
        capacity = min(max(int(count.value) * 2 + 5, 5), _MAX_PROCESS_ENTRIES)
        entries = (_ProcessInfo * capacity)()
        count = c_uint(capacity)
        _check(function(handle, ctypes.byref(count), entries), symbol)
        if int(count.value) > capacity:
            raise NvmlError(f"{symbol} returned too many process entries")
        return {int(entries[index].pid) for index in range(int(count.value))}
