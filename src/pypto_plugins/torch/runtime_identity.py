"""Resolve the already-loaded CUDA runtime without machine-local defaults."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path

from ..errors import StrictCoverageError


_DRIVER_LABEL_ENV = "PYPTO_PLUGINS_CUDA_DRIVER_LABEL"
_CUDART_PATH_ENV = "PYPTO_PLUGINS_CUDART"


class _DlInfo(ctypes.Structure):
    _fields_ = (
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    )


@dataclass(frozen=True, slots=True)
class LiveRuntimeExpectation:
    driver_label: str
    cuda_runtime_library_path: str


def _symbol_provider_path(symbol_name: str) -> str:
    process = ctypes.CDLL(None)
    try:
        symbol = getattr(process, symbol_name)
        dladdr = process.dladdr
    except AttributeError as error:
        raise StrictCoverageError(
            f"the process has no loaded {symbol_name} provider"
        ) from error
    dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(_DlInfo))
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info)) != 1:
        raise StrictCoverageError(f"dladdr failed for loaded symbol {symbol_name}")
    if not info.dli_fname:
        raise StrictCoverageError(
            f"dladdr returned no provider path for {symbol_name}"
        )
    return str(Path(os.fsdecode(info.dli_fname)).resolve(strict=True))


def _cuda_driver_api_version() -> int:
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        query = driver.cuDriverGetVersion
    except (OSError, AttributeError) as error:
        raise StrictCoverageError("CUDA Driver API version is unavailable") from error
    query.argtypes = (ctypes.POINTER(ctypes.c_int),)
    query.restype = ctypes.c_int
    version = ctypes.c_int()
    status = int(query(ctypes.byref(version)))
    if status != 0 or version.value <= 0:
        raise StrictCoverageError(
            f"cuDriverGetVersion failed with status={status}, value={version.value}"
        )
    return int(version.value)


def resolve_live_runtime_expectation() -> LiveRuntimeExpectation:
    """Return strict expectations derived from live providers or explicit overrides."""

    runtime_override = os.environ.get(_CUDART_PATH_ENV)
    if runtime_override:
        runtime_path = str(Path(runtime_override).resolve(strict=True))
    else:
        runtime_path = _symbol_provider_path("cudaRuntimeGetVersion")
    driver_override = os.environ.get(_DRIVER_LABEL_ENV)
    if driver_override is not None:
        if not driver_override or driver_override != driver_override.strip():
            raise StrictCoverageError(
                f"{_DRIVER_LABEL_ENV} must be a non-empty trimmed label"
            )
        driver_label = driver_override
    else:
        driver_label = f"cuda-driver-api-{_cuda_driver_api_version()}"
    return LiveRuntimeExpectation(driver_label, runtime_path)


__all__ = ("LiveRuntimeExpectation", "resolve_live_runtime_expectation")
