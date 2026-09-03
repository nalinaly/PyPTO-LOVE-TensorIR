"""FX pointwise chains lowered to native PyPTO tile-DSL graphs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import linecache
import os
import pathlib
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Iterator

from ..errors import StrictCoverageError

_DSO_ENVIRONMENT = "PYPTO_PLUGINS_PYPTO_DSO"
_ARTIFACT_CACHE_ENVIRONMENT = "PYPTO_CACHE_DIR"
_STRICT_COVERAGE_ENVIRONMENT = "PYPTO_STRICT_COVERAGE"
POINTWISE_CODEGEN_REVISION = "audited-generated-source-v2-20260829"
_PERSISTENT_CACHE_DISPOSITIONS = frozenset(
    {
        "Uncached",
        "CacheHit",
        "CompiledAndPublished",
        "CompiledAndValidatedExisting",
    }
)

_lock = threading.Lock()
_pypto_modules: dict[str, ModuleType] | None = None
_OWNER_PID = os.getpid()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_owner_process() -> None:
    current = os.getpid()
    if current != _OWNER_PID:
        raise StrictCoverageError(
            "PyPTO compiler caches were inherited across fork; use spawn/exec "
            f"(owner_pid={_OWNER_PID}, current_pid={current})"
        )


def _resolve_dso_override(value: str | pathlib.Path) -> pathlib.Path:
    resolved = pathlib.Path(value).resolve(strict=True)
    if resolved.is_dir():
        candidates = sorted(
            path.resolve(strict=True)
            for path in resolved.glob("pypto_core*.so")
            if path.is_file()
        )
        if len(candidates) != 1:
            raise StrictCoverageError(
                "PyPTO DSO directory must contain exactly one pypto_core*.so; "
                f"found {len(candidates)} under {resolved}"
            )
        resolved = candidates[0]
    if not resolved.is_file():
        raise StrictCoverageError(f"PyPTO DSO is not a regular file: {resolved}")
    return resolved


def bootstrap_pypto(
    dso_path: str | pathlib.Path | None = None,
) -> dict[str, ModuleType]:
    """Bind installed ``pypto`` or one explicitly requested diagnostic DSO."""

    _require_owner_process()
    global _pypto_modules
    with _lock:
        if _pypto_modules is not None:
            return _pypto_modules
    override = dso_path or os.environ.get(_DSO_ENVIRONMENT)
    if override is None:
        pypto = importlib.import_module("pypto")
        pypto_core = importlib.import_module("pypto.pypto_core")
        modules = {
            "pypto": pypto,
            "ir": pypto.ir,
            "compiler": pypto.compiler,
            "core": pypto_core,
        }
        with _lock:
            if _pypto_modules is None:
                _pypto_modules = modules
            return _pypto_modules

    resolved = _resolve_dso_override(override)
    package_spec = importlib.util.find_spec("pypto")
    if package_spec is None or package_spec.origin is None:
        raise StrictCoverageError("installed pypto package metadata is unavailable")
    package_root = pathlib.Path(package_spec.origin).resolve(strict=True).parent
    existing_package = sys.modules.get("pypto")
    existing_core = sys.modules.get("pypto.pypto_core")
    if existing_package is not None or existing_core is not None:
        package_file = getattr(existing_package, "__file__", None)
        core_file = getattr(existing_core, "__file__", None)
        if (
            not isinstance(existing_package, ModuleType)
            or not isinstance(existing_core, ModuleType)
            or not isinstance(package_file, str)
            or not isinstance(core_file, str)
            or pathlib.Path(package_file).resolve(strict=True).parent
            != package_root
            or pathlib.Path(core_file).resolve(strict=True) != resolved
        ):
            raise StrictCoverageError(
                "foreign pypto modules occupy the exact bootstrap slots"
            )
        modules = {
            "pypto": existing_package,
            "ir": existing_package.ir,
            "compiler": existing_package.compiler,
            "core": existing_core,
        }
        with _lock:
            if _pypto_modules is None:
                _pypto_modules = modules
            return _pypto_modules
    spec = importlib.util.spec_from_file_location(
        "pypto",
        pathlib.Path(package_spec.origin).resolve(strict=True),
        submodule_search_locations=[str(package_root)],
    )
    assert spec is not None and spec.loader is not None
    pypto = importlib.util.module_from_spec(spec)
    core_spec = importlib.util.spec_from_file_location("pypto.pypto_core", resolved)
    assert core_spec is not None and core_spec.loader is not None
    pypto_core = importlib.util.module_from_spec(core_spec)
    with _lock:
        if _pypto_modules is None:
            sys.modules["pypto.pypto_core"] = pypto_core
            core_spec.loader.exec_module(pypto_core)
            spec.loader.exec_module(pypto)
            sys.modules["pypto"] = pypto
            _pypto_modules = {
                "pypto": pypto,
                "ir": pypto.ir,
                "compiler": pypto.compiler,
                "core": pypto_core,
            }
    return _pypto_modules


def pypto_dso_path() -> pathlib.Path:
    """Return the loaded extension path for diagnostics and evidence."""

    modules = bootstrap_pypto()
    value = getattr(modules["core"], "__file__", None)
    if not isinstance(value, str):
        raise StrictCoverageError("loaded pypto_core has no concrete file")
    path = pathlib.Path(value).resolve(strict=True)
    candidates = sorted(
        candidate.resolve(strict=True)
        for candidate in path.parent.glob("pypto_core*.so")
        if candidate.is_file()
    )
    if candidates != [path]:
        raise StrictCoverageError(
            "loaded PyPTO extension directory has an ambiguous pypto_core*.so set"
        )
    return path


_DSO_DIGEST_LOCK = threading.Lock()
_DSO_DIGEST_CACHE: tuple[int, pathlib.Path, int, int, str] | None = None


def pypto_dso_sha256() -> str:
    """Hash the real loaded extension once per process and file identity."""

    _require_owner_process()
    path = pypto_dso_path()
    stat = path.stat()
    identity = (_OWNER_PID, path, stat.st_size, stat.st_mtime_ns)
    global _DSO_DIGEST_CACHE
    with _DSO_DIGEST_LOCK:
        if _DSO_DIGEST_CACHE is not None and _DSO_DIGEST_CACHE[:4] == identity:
            return _DSO_DIGEST_CACHE[4]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            file_digest = getattr(hashlib, "file_digest", None)
            if file_digest is not None:
                digest_hex = file_digest(stream, "sha256").hexdigest()
            else:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                digest_hex = digest.hexdigest()
        _DSO_DIGEST_CACHE = (*identity, digest_hex)
        return digest_hex


@dataclass(frozen=True, slots=True)
class _Value:
    key: int


@dataclass(frozen=True, slots=True)
class _Scalar:
    value: float


@dataclass(frozen=True, slots=True)
class _DType:
    name: str


def _dense_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    running = 1
    reversed_strides: list[int] = []
    for extent in reversed(shape):
        reversed_strides.append(running)
        running *= extent
    return tuple(reversed(reversed_strides))


@dataclass(frozen=True, slots=True)
class PointwiseTensorSpec:
    """Static tensor identity used by PyPTO specialization and caching.

    ``device_type``/``device_index`` describe the real tensor even though the
    frontend sample itself is allocated on the meta device.  In particular,
    row-pitched packed views retain their physical stride instead of silently
    becoming contiguous during specialization.
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype_name: str
    device_type: str
    device_index: int | None

    def __post_init__(self) -> None:
        if not self.shape or any(type(value) is not int or value <= 0 for value in self.shape):
            raise StrictCoverageError("pointwise tensor shape must be positive and static")
        if len(self.strides) != len(self.shape) or any(
            type(value) is not int or value < 0 for value in self.strides
        ):
            raise StrictCoverageError(
                "pointwise tensor strides must be non-negative static integers"
            )
        if self.dtype_name not in ("float32", "bfloat16"):
            raise StrictCoverageError(
                f"unsupported pointwise tensor dtype: {self.dtype_name!r}"
            )
        if type(self.device_type) is not str or not self.device_type:
            raise StrictCoverageError("pointwise tensor device type is missing")
        if self.device_index is not None and (
            type(self.device_index) is not int or self.device_index < 0
        ):
            raise StrictCoverageError("pointwise tensor device index is invalid")

    @classmethod
    def dense(
        cls,
        shape: tuple[int, ...],
        dtype_name: str,
        *,
        device_type: str = "cuda",
        device_index: int | None = 0,
    ) -> "PointwiseTensorSpec":
        normalized = tuple(int(extent) for extent in shape)
        return cls(
            normalized,
            _dense_strides(normalized),
            dtype_name,
            device_type,
            device_index,
        )

    def meta_tensor(self) -> Any:
        """Create the exact strided meta sample consumed by ``specialize``."""

        import torch

        dtype = torch.float32 if self.dtype_name == "float32" else torch.bfloat16
        return torch.empty_strided(
            self.shape,
            self.strides,
            dtype=dtype,
            device="meta",
        )


@dataclass(frozen=True, slots=True)
class _Input:
    name: str
    logical_shape: tuple[int, ...]
    specialization: PointwiseTensorSpec
    value: _Value


@dataclass(frozen=True, slots=True)
class _Instruction:
    op_name: str
    arguments: tuple[_Value | _Scalar | _DType, ...]
    result: _Value


@dataclass(frozen=True, slots=True)
class NativePointwiseProgram:
    """Immutable native tile program waiting for its schedule tile."""

    shape: tuple[int, ...]
    dtype_name: str
    output_spec: PointwiseTensorSpec
    inputs: tuple[_Input, ...]
    instructions: tuple[_Instruction, ...]
    output: _Value

    _OPS = {
        "tensor.add": "add",
        "tensor.adds": "add",
        "tensor.sub": "sub",
        "tensor.subs": "sub",
        "tensor.mul": "mul",
        "tensor.muls": "mul",
        "tensor.div": "div",
        "tensor.divs": "div",
        "tensor.neg": "neg",
        "tensor.exp": "exp",
        "tensor.recip": "recip",
        "tensor.rsqrt": "rsqrt",
        "tensor.abs": "abs",
        "tensor.sqrt": "sqrt",
        "tensor.log": "log",
        "tensor.sin": "sin",
        "tensor.cos": "cos",
        "tensor.maximum": "maximum",
        "tensor.minimum": "minimum",
        "tensor.row_expand": "row_expand",
        "tensor.row_expand_add": "row_expand_add",
        "tensor.row_expand_sub": "row_expand_sub",
        "tensor.row_expand_mul": "row_expand_mul",
        "tensor.row_expand_div": "row_expand_div",
        "tensor.row_expand_max": "row_expand_max",
        "tensor.row_expand_min": "row_expand_min",
    }

    def __post_init__(self) -> None:
        if self.output_spec.shape != self.shape:
            raise StrictCoverageError(
                "native pointwise output specialization shape disagrees with program"
            )
        if self.output_spec.dtype_name != self.dtype_name:
            raise StrictCoverageError(
                "native pointwise output specialization dtype disagrees with program"
            )

    def specialization_samples(self) -> tuple[Any, ...]:
        """Return stride-exact meta inputs and output in launch ABI order."""

        return tuple(
            [item.specialization.meta_tensor() for item in self.inputs]
            + [self.output_spec.meta_tensor()]
        )

    def native_source(self, tile: int) -> str:
        """Return the generated ``@pl.jit`` source for inspection/evidence."""

        key = (self, tile)
        if key not in _NATIVE_SOURCE_CACHE:
            self.specialize(tile)
        return _NATIVE_SOURCE_CACHE[key]

    def specialize(self, tile: int) -> Any:
        if tile <= 0 or tile & (tile - 1):
            raise StrictCoverageError("native pointwise tile must be a power of two")
        if self.shape[-1] % tile:
            raise StrictCoverageError(
                f"native pointwise trailing extent {self.shape[-1]} "
                f"is not divisible by tile {tile}"
            )
        bootstrap_pypto()
        import pypto.language as pl

        parameters = [f"input_{index}: pl.Tensor" for index in range(len(self.inputs))]
        parameters.append("out: pl.Out[pl.Tensor]")
        lines = [
            "@pl.jit",
            "def generated_pointwise_kernel(" + ", ".join(parameters) + "):",
            "    with pl.at(level=pl.Level.CORE_GROUP):",
        ]
        indent = "        "
        offsets: list[str] = []
        for dimension, extent in enumerate(self.shape[:-1]):
            loop_name = f"index_{dimension}"
            lines.append(f"{indent}for {loop_name} in pl.range({extent}):")
            offsets.append(loop_name)
            indent += "    "
        lines.append(f"{indent}for block in pl.range({self.shape[-1] // tile}):")
        indent += "    "
        output_offsets = [*offsets, f"block * {tile}"]
        output_tile = [1] * (len(self.shape) - 1) + [tile]
        names: dict[int, str] = {}
        for index, input_spec in enumerate(self.inputs):
            input_offsets = [
                offset if input_spec.logical_shape[axis] != 1 else "0"
                for axis, offset in enumerate(output_offsets)
            ]
            input_tile = [
                output_tile[axis] if input_spec.logical_shape[axis] != 1 else 1
                for axis in range(len(self.shape))
            ]
            name = f"value_{input_spec.value.key}"
            lines.append(
                f"{indent}{name} = pl.load(input_{index}, "
                f"[{', '.join(input_offsets)}], {input_tile})"
            )
            names[input_spec.value.key] = name
        for instruction in self.instructions:
            if instruction.op_name == "tensor.cast":
                if (
                    len(instruction.arguments) != 2
                    or not isinstance(instruction.arguments[0], _Value)
                    or not isinstance(instruction.arguments[1], _DType)
                ):
                    raise StrictCoverageError("native pointwise cast is malformed")
                dtype_expr = {
                    "float32": "pl.FP32",
                    "bfloat16": "pl.BF16",
                }.get(instruction.arguments[1].name)
                if dtype_expr is None:
                    raise StrictCoverageError(
                        "native pointwise cast target is unsupported"
                    )
                name = f"value_{instruction.result.key}"
                source_name = names[instruction.arguments[0].key]
                lines.append(f"{indent}{name} = pl.cast({source_name}, {dtype_expr})")
                names[instruction.result.key] = name
                continue
            op = self._OPS.get(instruction.op_name)
            if op is None:
                raise StrictCoverageError(
                    f"native pointwise has no tile op for {instruction.op_name!r}"
                )
            arguments: list[str] = []
            for argument in instruction.arguments:
                if isinstance(argument, _Scalar):
                    arguments.append(repr(argument.value))
                elif isinstance(argument, _DType):
                    raise StrictCoverageError(
                        "native pointwise dtype operand is only valid for cast"
                    )
                else:
                    arguments.append(names[argument.key])
            name = f"value_{instruction.result.key}"
            lines.append(f"{indent}{name} = pl.{op}({', '.join(arguments)})")
            names[instruction.result.key] = name
        lines.append(
            f"{indent}pl.store({names[self.output.key]}, "
            f"[{', '.join(output_offsets)}], out)"
        )
        lines.append("    return out")
        source = "\n".join(lines) + "\n"
        _NATIVE_SOURCE_CACHE[(self, tile)] = source
        filename = "<pypto_inductor_native_pointwise>"
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        namespace: dict[str, Any] = {"pl": pl}
        exec(compile(source, filename, "exec"), namespace)
        kernel = namespace["generated_pointwise_kernel"]
        return kernel.specialize(*self.specialization_samples())


_NATIVE_SOURCE_CACHE: dict[tuple[NativePointwiseProgram, int], str] = {}


@dataclass(frozen=True, slots=True)
class NativeReductionProgram:
    """Trailing-axis row sum/max expressed as a native tile graph."""

    input_shape: tuple[int, ...]
    dtype_name: str
    mode: str
    device_type: str = "cuda"
    device_index: int = 0

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (*self.input_shape[:-1], 1)

    @property
    def row_count(self) -> int:
        count = 1
        for extent in self.input_shape[:-1]:
            count *= extent
        return count

    @property
    def row_tile(self) -> int:
        return min(128, 1 << (self.row_count.bit_length() - 1))

    def native_source(self) -> str:
        cached = _REDUCTION_SOURCE_CACHE.get(self)
        if cached is not None:
            return cached
        if self.mode not in ("sum", "max"):
            raise StrictCoverageError(f"unsupported native reduction {self.mode!r}")
        if not self.input_shape or any(extent <= 0 for extent in self.input_shape):
            raise StrictCoverageError("native reduction shape must be positive")
        dtype_expr = "pl.FP32" if self.dtype_name == "float32" else "pl.BF16"
        lines = [
            "@pl.jit",
            "def generated_reduction_kernel(input_0: pl.Tensor, out: pl.Out[pl.Tensor]):",
            "    with pl.at(level=pl.Level.CORE_GROUP):",
        ]
        indent = "        "
        offsets: list[str] = []
        for dimension, extent in enumerate(self.input_shape[:-1]):
            name = f"index_{dimension}"
            lines.append(f"{indent}for {name} in pl.range({extent}):")
            offsets.append(name)
            indent += "    "
        input_tile = [1] * (len(self.input_shape) - 1) + [self.input_shape[-1]]
        input_offsets = [*offsets, "0"]
        output_offsets = [*offsets, "0"]
        lines.extend(
            [
                f"{indent}input_tile = pl.load(input_0, "
                f"[{', '.join(input_offsets)}], {input_tile})",
                f"{indent}scratch = pl.create_tile({input_tile}, "
                f"dtype={dtype_expr}, target_memory=pl.MemorySpace.Vec)",
                f"{indent}reduced = pl.row_{self.mode}(input_tile, scratch)",
                f"{indent}pl.store(reduced, [{', '.join(output_offsets)}], out)",
                "    return out",
            ]
        )
        source = "\n".join(lines) + "\n"
        _REDUCTION_SOURCE_CACHE[self] = source
        return source

    def specialize(self) -> Any:
        bootstrap_pypto()
        import torch
        import pypto.language as pl

        source = self.native_source()
        filename = "<pypto_inductor_native_reduction>"
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        namespace: dict[str, Any] = {"pl": pl}
        exec(compile(source, filename, "exec"), namespace)
        kernel = namespace["generated_reduction_kernel"]
        dtype = torch.float32 if self.dtype_name == "float32" else torch.bfloat16
        input_sample = torch.empty(self.input_shape, dtype=dtype, device="meta")
        output_sample = torch.empty(self.output_shape, dtype=dtype, device="meta")
        return kernel.specialize(input_sample, output_sample)


_REDUCTION_SOURCE_CACHE: dict[NativeReductionProgram, str] = {}


class PointwiseProgramBuilder:
    """Record a bounded FX chain for later native tile specialization."""

    MAX_ASSIGNMENTS = 64
    MAX_INPUTS = 16
    _DTYPE_NAMES = ("float32", "bfloat16")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype_name: str,
        *,
        output_spec: PointwiseTensorSpec | None = None,
    ) -> None:
        if dtype_name not in self._DTYPE_NAMES:
            raise StrictCoverageError(
                f"unsupported fused pointwise dtype: {dtype_name!r}"
            )
        self._shape_tuple = tuple(int(dim) for dim in shape)
        if not self._shape_tuple or any(dim <= 0 for dim in self._shape_tuple):
            raise StrictCoverageError("native pointwise shape must be positive")
        self._dtype_name = dtype_name
        self._output_spec = output_spec or PointwiseTensorSpec.dense(
            self._shape_tuple,
            dtype_name,
        )
        if (
            self._output_spec.shape != self._shape_tuple
            or self._output_spec.dtype_name != dtype_name
        ):
            raise StrictCoverageError(
                "pointwise output specialization disagrees with builder"
            )
        self._instructions: list[_Instruction] = []
        self._next_key = 0
        self.inputs: list[_Input] = []
        self.outputs: list[_Value] = []

    def _value(self) -> _Value:
        value = _Value(self._next_key)
        self._next_key += 1
        return value

    def add_input(
        self,
        name: str,
        *,
        specialization: PointwiseTensorSpec | None = None,
    ) -> Any:
        if len(self.inputs) >= self.MAX_INPUTS:
            raise StrictCoverageError("fused pointwise chain exceeds 16 inputs")
        specialization = specialization or PointwiseTensorSpec.dense(
            self._shape_tuple,
            self._dtype_name,
            device_type=self._output_spec.device_type,
            device_index=self._output_spec.device_index,
        )
        if specialization.shape != self._shape_tuple:
            raise StrictCoverageError(
                "non-broadcast pointwise input shape must match the output"
            )
        self._require_compatible_input(specialization)
        value = self._value()
        self.inputs.append(_Input(name, self._shape_tuple, specialization, value))
        return value

    def add_broadcast_input(
        self,
        name: str,
        *,
        specialization: PointwiseTensorSpec | None = None,
    ) -> Any:
        """Add a [M,1,...] row input for row-expand fused operands."""
        if len(self.inputs) >= self.MAX_INPUTS:
            raise StrictCoverageError("fused pointwise chain exceeds 16 inputs")
        row_shape = [dim if index == 0 else 1 for index, dim in enumerate(self.shape)]
        if specialization is None:
            row_spec = PointwiseTensorSpec.dense(
                tuple(row_shape),
                self._dtype_name,
                device_type=self._output_spec.device_type,
                device_index=self._output_spec.device_index,
            )
        else:
            row_spec = specialization
        if row_spec.shape != tuple(row_shape):
            raise StrictCoverageError(
                "broadcast pointwise input must have [M,1,...] logical shape"
            )
        self._require_compatible_input(row_spec)
        expanded_spec = PointwiseTensorSpec(
            self._shape_tuple,
            (row_spec.strides[0], *([0] * (len(self._shape_tuple) - 1))),
            row_spec.dtype_name,
            row_spec.device_type,
            row_spec.device_index,
        )
        value = self._value()
        self.inputs.append(_Input(name, tuple(row_shape), expanded_spec, value))
        return value

    def _require_compatible_input(self, value: PointwiseTensorSpec) -> None:
        if value.dtype_name != self._dtype_name:
            raise StrictCoverageError(
                "pointwise mixed input dtypes require an explicit supported cast"
            )
        if (
            value.device_type != self._output_spec.device_type
            or value.device_index != self._output_spec.device_index
        ):
            raise StrictCoverageError(
                "pointwise inputs and output must use one static device"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape_tuple

    def scalar(self, value: float) -> Any:
        return _Scalar(float(value))

    def dtype(self, name: str) -> Any:
        if name not in self._DTYPE_NAMES:
            raise StrictCoverageError(
                f"unsupported fused pointwise cast dtype: {name!r}"
            )
        return _DType(name)

    def emit(self, op_name: str, arguments: list[Any]) -> Any:
        if len(self._instructions) >= self.MAX_ASSIGNMENTS:
            raise StrictCoverageError("fused pointwise chain exceeds 64 assignments")
        if not all(isinstance(item, (_Value, _Scalar, _DType)) for item in arguments):
            raise StrictCoverageError(
                "native pointwise operand is not a recorded value"
            )
        value = self._value()
        self._instructions.append(_Instruction(op_name, tuple(arguments), value))
        return value

    def mark_output(self, var: Any) -> None:
        self.outputs.append(var)

    def build(self) -> Any:
        if not self.inputs or not self.outputs or not self._instructions:
            raise StrictCoverageError(
                "fused pointwise program needs at least one input, one op "
                "and one output"
            )
        if len(self.outputs) != 1:
            raise StrictCoverageError("native pointwise currently requires one output")
        return NativePointwiseProgram(
            self._shape_tuple,
            self._dtype_name,
            self._output_spec,
            tuple(self.inputs),
            tuple(self._instructions),
            self.outputs[0],
        )


_RUNTIME_OBJECTS: dict[str, tuple[Any, Any]] = {}
_RUNTIME_SOURCE_NODES: dict[str, str] = {}
_RUNTIME_DEVICE_INDICES: dict[str, int] = {}
_RUNTIME_DSO_SHA256: dict[str, str] = {}
_RUNTIME_WRAPPER_SOURCES: dict[
    str,
    dict[tuple[str, str], "WrapperLaunchSource"],
] = {}
_RUNTIME_LOCK = threading.RLock()


def retain_runtime_objects(
    name: str,
    artifact: Any,
    request: Any,
    *,
    source_node: str,
    device_index: int,
    dso_sha256: str,
) -> None:
    """Keep the live artifact/request pair for the wrapper-side bridge."""

    if type(name) is not str or not name:
        raise StrictCoverageError("runtime object name must be non-empty")
    if not source_node.startswith("torch-inductor:"):
        raise StrictCoverageError(
            "Inductor runtime source node must use the torch-inductor namespace"
        )
    if type(device_index) is not int or device_index < 0:
        raise StrictCoverageError("runtime object CUDA device index is invalid")
    if not _is_sha256(dso_sha256):
        raise StrictCoverageError("runtime object DSO SHA256 is invalid")
    _require_owner_process()
    with _RUNTIME_LOCK:
        previous = _RUNTIME_OBJECTS.setdefault(name, (artifact, request))
        previous_source = _RUNTIME_SOURCE_NODES.setdefault(name, source_node)
        previous_device = _RUNTIME_DEVICE_INDICES.setdefault(name, device_index)
        previous_dso = _RUNTIME_DSO_SHA256.setdefault(name, dso_sha256)
        if (
            previous != (artifact, request)
            or previous_source != source_node
            or previous_device != device_index
            or previous_dso != dso_sha256
        ):
            raise StrictCoverageError(
                f"conflicting PyPTO runtime object registration for {name!r}"
            )


def runtime_objects(name: str) -> tuple[Any, Any] | None:
    _require_owner_process()
    with _RUNTIME_LOCK:
        return _RUNTIME_OBJECTS.get(name)


def runtime_source_node(name: str) -> str | None:
    _require_owner_process()
    with _RUNTIME_LOCK:
        return _RUNTIME_SOURCE_NODES.get(name)


def runtime_device_index(name: str) -> int | None:
    _require_owner_process()
    with _RUNTIME_LOCK:
        return _RUNTIME_DEVICE_INDICES.get(name)


def runtime_dso_sha256(name: str) -> str | None:
    _require_owner_process()
    with _RUNTIME_LOCK:
        return _RUNTIME_DSO_SHA256.get(name)


@dataclass(frozen=True, slots=True)
class PointwiseArtifact:
    """The compiled native tile artifact identity."""

    kernel_name: str
    entry_name: str
    build_spec_sha256: str
    artifact_sha256: str
    cubin_sha256: str
    cubin_bytes: int
    grid: tuple[int, int, int]
    argument_count: int
    workspace_bytes: int
    fallback_used: bool
    pypto_source: str
    pypto_source_sha256: str
    cache_identity_sha256: str
    artifact_cache_key_sha256: str
    artifact_cache_disposition: str
    source_node: str
    dso_sha256: str

    def __post_init__(self) -> None:
        digests = (
            self.build_spec_sha256,
            self.artifact_sha256,
            self.cubin_sha256,
            self.pypto_source_sha256,
            self.cache_identity_sha256,
            self.artifact_cache_key_sha256,
            self.dso_sha256,
        )
        if not all(_is_sha256(value) for value in digests):
            raise StrictCoverageError("pointwise artifact has an invalid SHA256")
        if type(self.pypto_source) is not str:
            raise StrictCoverageError("pointwise generated source must be exact text")
        source_digest = hashlib.sha256(self.pypto_source.encode("utf-8")).hexdigest()
        if not self.pypto_source.startswith("@pl.jit\n"):
            raise StrictCoverageError("pointwise artifact lacks generated @pl.jit source")
        if self.pypto_source_sha256 != source_digest:
            raise StrictCoverageError("pointwise generated source SHA256 differs")
        if self.source_node != f"torch-inductor:{self.cache_identity_sha256[:16]}":
            raise StrictCoverageError("pointwise source node is not cache-identity bound")
        if self.kernel_name != f"pypto_inductor_{self.cache_identity_sha256[:16]}":
            raise StrictCoverageError("pointwise kernel name is not cache-identity bound")
        if self.artifact_cache_disposition not in _PERSISTENT_CACHE_DISPOSITIONS:
            raise StrictCoverageError(
                "pointwise artifact has an invalid persistent-cache disposition"
            )


@dataclass(frozen=True, slots=True)
class WrapperLaunchSource:
    """Exact source lines emitted into an Inductor Python wrapper."""

    registry_name: str
    kernel_name: str
    source_node: str
    artifact_id: str
    artifact_sha256: str
    header_source: str
    header_source_sha256: str
    launch_source: str
    launch_source_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "registry_name": self.registry_name,
            "kernel_name": self.kernel_name,
            "source_node": self.source_node,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "header_source": self.header_source,
            "header_source_sha256": self.header_source_sha256,
            "launch_source": self.launch_source,
            "launch_source_sha256": self.launch_source_sha256,
        }


@dataclass(frozen=True, slots=True)
class PointwiseSourceEvidence:
    """Audited generated-source and native-artifact binding."""

    kernel_name: str
    entry_name: str
    source_node: str
    cache_identity_sha256: str
    artifact_cache_key_sha256: str
    artifact_cache_disposition: str
    build_spec_sha256: str
    artifact_id: str
    artifact_sha256: str
    cubin_sha256: str
    dso_sha256: str
    pypto_source: str
    pypto_source_sha256: str
    wrapper_launch_sources: tuple[WrapperLaunchSource, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_name": self.kernel_name,
            "entry_name": self.entry_name,
            "source_node": self.source_node,
            "cache_identity_sha256": self.cache_identity_sha256,
            "artifact_cache_observation": {
                "cache_key_sha256": self.artifact_cache_key_sha256,
                "disposition": self.artifact_cache_disposition,
            },
            "build_spec_sha256": self.build_spec_sha256,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "cubin_sha256": self.cubin_sha256,
            "dso_sha256": self.dso_sha256,
            "pypto_source": self.pypto_source,
            "pypto_source_sha256": self.pypto_source_sha256,
            "wrapper_launch_sources": [
                record.to_dict() for record in self.wrapper_launch_sources
            ],
        }


def record_wrapper_launch_source(
    registry_name: str,
    artifact: PointwiseArtifact,
    *,
    header_lines: tuple[str, ...],
    launch_line: str,
) -> WrapperLaunchSource:
    """Bind exact emitted wrapper lines to their retained native artifact."""

    _require_owner_process()
    if type(registry_name) is not str or not registry_name:
        raise StrictCoverageError("wrapper registry name must be non-empty")
    if type(artifact) is not PointwiseArtifact:
        raise StrictCoverageError("wrapper source requires an exact pointwise artifact")
    if (
        type(header_lines) is not tuple
        or not header_lines
        or any(type(line) is not str or not line or "\n" in line for line in header_lines)
    ):
        raise StrictCoverageError("wrapper header lines must be exact non-empty lines")
    if type(launch_line) is not str or not launch_line or "\n" in launch_line:
        raise StrictCoverageError("wrapper launch line must be one exact source line")
    if not launch_line.startswith(f"pypto_launch({registry_name!r}, "):
        raise StrictCoverageError("wrapper launch line is not registry-name bound")

    with _RUNTIME_LOCK:
        retained = _RUNTIME_OBJECTS.get(registry_name)
        retained_source_node = _RUNTIME_SOURCE_NODES.get(registry_name)
        retained_dso = _RUNTIME_DSO_SHA256.get(registry_name)
        if retained is None:
            raise StrictCoverageError(
                f"wrapper source has no retained runtime object for {registry_name!r}"
            )
        native_artifact, _request = retained
        serialized = bytes(native_artifact.serialize())
        artifact_sha256 = hashlib.sha256(serialized).hexdigest()
        cubin_sha256 = hashlib.sha256(bytes(native_artifact.device_code)).hexdigest()
        if (
            artifact_sha256 != artifact.artifact_sha256
            or cubin_sha256 != artifact.cubin_sha256
            or str(native_artifact.kernel_abi.entry_function_name)
            != artifact.entry_name
            or bool(native_artifact.fallback_used) != artifact.fallback_used
            or str(native_artifact.cache_key_digest)
            != artifact.artifact_cache_key_sha256
            or retained_source_node != artifact.source_node
            or retained_dso != artifact.dso_sha256
        ):
            raise StrictCoverageError(
                "wrapper source and retained native artifact identities differ"
            )
        artifact_id = f"pypto-artifact-v1:{native_artifact.identity_digest}"
        header_source = "\n".join(header_lines) + "\n"
        launch_source = launch_line + "\n"
        header_sha256 = hashlib.sha256(header_source.encode("utf-8")).hexdigest()
        launch_sha256 = hashlib.sha256(launch_source.encode("utf-8")).hexdigest()
        record = WrapperLaunchSource(
            registry_name=registry_name,
            kernel_name=artifact.kernel_name,
            source_node=artifact.source_node,
            artifact_id=artifact_id,
            artifact_sha256=artifact.artifact_sha256,
            header_source=header_source,
            header_source_sha256=header_sha256,
            launch_source=launch_source,
            launch_source_sha256=launch_sha256,
        )
        records = _RUNTIME_WRAPPER_SOURCES.setdefault(
            artifact.cache_identity_sha256,
            {},
        )
        previous = records.setdefault((header_sha256, launch_sha256), record)
        if previous != record:
            raise StrictCoverageError(
                "conflicting wrapper source for one pointwise artifact identity"
            )
        return previous


def pointwise_source_evidence(
    artifact: PointwiseArtifact,
    *,
    require_wrapper_source: bool = True,
) -> PointwiseSourceEvidence:
    """Audit generated DSL, wrapper source and the retained native artifact."""

    _require_owner_process()
    if type(artifact) is not PointwiseArtifact:
        raise StrictCoverageError("source evidence requires an exact pointwise artifact")
    with _RUNTIME_LOCK:
        retained = _RUNTIME_OBJECTS.get(artifact.kernel_name)
        if retained is None:
            raise StrictCoverageError("source evidence has no canonical runtime object")
        native_artifact, _request = retained
        artifact_sha256 = hashlib.sha256(
            bytes(native_artifact.serialize())
        ).hexdigest()
        cubin_sha256 = hashlib.sha256(bytes(native_artifact.device_code)).hexdigest()
        source_sha256 = hashlib.sha256(
            artifact.pypto_source.encode("utf-8")
        ).hexdigest()
        if (
            artifact_sha256 != artifact.artifact_sha256
            or cubin_sha256 != artifact.cubin_sha256
            or source_sha256 != artifact.pypto_source_sha256
            or str(native_artifact.kernel_abi.entry_function_name)
            != artifact.entry_name
            or bool(native_artifact.fallback_used) != artifact.fallback_used
            or str(native_artifact.cache_key_digest)
            != artifact.artifact_cache_key_sha256
            or _RUNTIME_SOURCE_NODES.get(artifact.kernel_name)
            != artifact.source_node
            or _RUNTIME_DSO_SHA256.get(artifact.kernel_name) != artifact.dso_sha256
        ):
            raise StrictCoverageError(
                "generated source and retained native artifact identities differ"
            )
        artifact_id = f"pypto-artifact-v1:{native_artifact.identity_digest}"
        wrapper_sources = tuple(
            sorted(
                _RUNTIME_WRAPPER_SOURCES.get(
                    artifact.cache_identity_sha256,
                    {},
                ).values(),
                key=lambda record: (
                    record.registry_name,
                    record.header_source_sha256,
                    record.launch_source_sha256,
                ),
            )
        )
        if require_wrapper_source and not wrapper_sources:
            raise StrictCoverageError("pointwise artifact has no emitted wrapper source")
        for record in wrapper_sources:
            if (
                record.kernel_name != artifact.kernel_name
                or record.source_node != artifact.source_node
                or record.artifact_id != artifact_id
                or record.artifact_sha256 != artifact.artifact_sha256
                or hashlib.sha256(record.header_source.encode("utf-8")).hexdigest()
                != record.header_source_sha256
                or hashlib.sha256(record.launch_source.encode("utf-8")).hexdigest()
                != record.launch_source_sha256
            ):
                raise StrictCoverageError(
                    "emitted wrapper source lost its native artifact binding"
                )
        return PointwiseSourceEvidence(
            kernel_name=artifact.kernel_name,
            entry_name=artifact.entry_name,
            source_node=artifact.source_node,
            cache_identity_sha256=artifact.cache_identity_sha256,
            artifact_cache_key_sha256=artifact.artifact_cache_key_sha256,
            artifact_cache_disposition=artifact.artifact_cache_disposition,
            build_spec_sha256=artifact.build_spec_sha256,
            artifact_id=artifact_id,
            artifact_sha256=artifact.artifact_sha256,
            cubin_sha256=artifact.cubin_sha256,
            dso_sha256=artifact.dso_sha256,
            pypto_source=artifact.pypto_source,
            pypto_source_sha256=artifact.pypto_source_sha256,
            wrapper_launch_sources=wrapper_sources,
        )


def _live_target_or_none(device_index: int) -> Any:
    """Observe the live NVIDIA target when CUDA is available in-process."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        with torch.cuda.device(device_index):
            torch.empty((), device=f"cuda:{device_index}")
            from pypto.runtime import nvidia as runtime
            from .runtime_identity import resolve_live_runtime_expectation

            expected = resolve_live_runtime_expectation()
            observation = runtime.observe_current_nvidia_runtime(
                expected.driver_label,
                expected.cuda_runtime_library_path,
            )
        return observation.target_info
    except Exception:
        return None


def _reference_request(
    compiler: Any,
    pypto: Any,
    info: Any,
    *,
    require_live_target: bool,
    device_index: int,
) -> Any:
    """Build the SM120 CompileRequest from live backend build info."""

    from pypto.compiler import (
        CompileRequest,
        NvidiaTargetInfo,
        TargetTraits,
        ToolchainIdentity,
    )

    live_target = _live_target_or_none(device_index)
    if live_target is None and require_live_target:
        raise StrictCoverageError(
            "Inductor PyPTO compilation requires a live CUDA target"
        )

    traits = TargetTraits(
        compute_capability=120,
        multiprocessor_count=82,
        warp_size=32,
        max_threads_per_block=1024,
        max_threads_per_multiprocessor=1536,
        max_blocks_per_multiprocessor=24,
        max_block_dim_x=1024,
        max_block_dim_y=1024,
        max_block_dim_z=64,
        max_grid_dim_x=(1 << 31) - 1,
        max_grid_dim_y=65535,
        max_grid_dim_z=65535,
        l1_cache_line_bytes=128,
        default_shared_memory_per_cta_bytes=48 * 1024,
        max_shared_memory_per_cta_bytes=101376,
        shared_memory_per_multiprocessor_bytes=131072,
        registers_per_cta=65536,
        max_registers_per_thread=255,
        registers_per_multiprocessor=65536,
        l2_cache_size_bytes=96 * 1024 * 1024,
        total_global_memory_bytes=24 * 1024 * 1024 * 1024,
    )
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
    target = live_target
    if target is None:
        target = NvidiaTargetInfo(
            0,
            "PyPTO plugins pointwise device",
            "GPU-pypto-plugins",
            "0000:01:00.0",
            traits,
            info.cuda_toolkit_version,
            "offline-sm120-reference",
            info.tensor_ir_revision,
            info.cuda_tile_revision,
            [pypto.DataType.BF16, pypto.DataType.FP32],
        )
    return CompileRequest(target, toolchain)


def _reference_schedule(tile_shape: tuple[int, ...]) -> Any:
    from pypto.compiler import (
        CanonicalSchedule,
        ScheduleParameter,
        ScheduleValueKind,
    )
    if not tile_shape or any(
        type(extent) is not int or extent <= 0 for extent in tile_shape
    ):
        raise StrictCoverageError("pointwise schedule tile shape must be positive")

    parameter = ScheduleParameter
    unsigned = ScheduleValueKind.UnsignedInteger
    return CanonicalSchedule(
        [parameter("codegen_strategy", ScheduleValueKind.Text, "layout-propagation")],
        [
            parameter(f"dim_{index:03d}", unsigned, str(extent))
            for index, extent in enumerate(tile_shape)
        ],
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


def _pointwise_tile_shape(shape: tuple[int, ...], tile: int) -> tuple[int, ...]:
    """Match TensorIR's canonical removal of unit iteration dimensions."""

    if not shape or tile <= 0 or shape[-1] % tile:
        raise StrictCoverageError("pointwise shape is incompatible with its tile")
    # Ada TensorIR collapses a dense multi-d pointwise nest to rank-1 numel.
    # Keep a 1-D schedule tile so CanonicalSchedule matches that space.
    # SM120 evidence still uses the leading-ones form below.
    try:
        import torch

        if (
            torch.cuda.is_available()
            and tuple(torch.cuda.get_device_capability(0)) == (8, 9)
        ):
            numel = 1
            for extent in shape:
                numel *= int(extent)
            if numel % tile:
                raise StrictCoverageError(
                    "pointwise numel is incompatible with its tile"
                )
            return (tile,)
    except StrictCoverageError:
        raise
    except Exception:
        pass
    return (*([1] * sum(extent != 1 for extent in shape[:-1])), tile)


@dataclass(frozen=True, slots=True)
class BackendRevisionIdentity:
    pypto_revision: str
    tensor_ir_revision: str
    cuda_tile_revision: str
    llvm_revision: str
    cuda_toolkit_version: str
    tileiras_sha256: str
    pypto_dso_sha256: str
    pointwise_codegen_revision: str = POINTWISE_CODEGEN_REVISION


def _backend_revision_identity(info: Any, dso_sha256: str) -> BackendRevisionIdentity:
    return BackendRevisionIdentity(
        str(info.pypto_revision),
        str(info.tensor_ir_revision),
        str(info.cuda_tile_revision),
        str(info.llvm_revision),
        str(info.cuda_toolkit_version),
        str(info.tileiras_sha256),
        dso_sha256,
        POINTWISE_CODEGEN_REVISION,
    )


def current_backend_revision_identity() -> BackendRevisionIdentity:
    """Return the loaded compiler stack identity used in callable cache keys."""

    compiler = bootstrap_pypto()["compiler"]
    info = compiler.get_nvidia_backend_build_info()
    if not info.compiled:
        raise StrictCoverageError("PyPTO exact DSO backend is not compiled")
    return _backend_revision_identity(info, pypto_dso_sha256())


_ARTIFACT_CACHE_LOCK = threading.Lock()
_ARTIFACT_CACHE_STATE: tuple[int, str, Any, Any] | None = None
_COMPILE_LOCK = threading.RLock()
_COMPILE_CACHE: dict[
    tuple[NativePointwiseProgram | NativeReductionProgram, int, BackendRevisionIdentity],
    PointwiseArtifact,
] = {}
_CAPTURE_LOCK = threading.RLock()
_ACTIVE_CAPTURE: "PointwiseArtifactCapture | None" = None


def _artifact_cache_root() -> str | None:
    value = os.environ.get(_ARTIFACT_CACHE_ENVIRONMENT)
    if value is None:
        if os.environ.get(_STRICT_COVERAGE_ENVIRONMENT) == "1":
            raise StrictCoverageError(
                "strict coverage requires an absolute PYPTO_CACHE_DIR"
            )
        return None
    if not value or value != value.strip() or not os.path.isabs(value):
        raise StrictCoverageError(
            "PYPTO_CACHE_DIR must be one non-empty absolute canonical path"
        )
    try:
        resolved = pathlib.Path(value).resolve(strict=True)
    except OSError as error:
        raise StrictCoverageError(
            f"PYPTO_CACHE_DIR is missing or inaccessible: {value}"
        ) from error
    if str(resolved) != value or not resolved.is_dir():
        raise StrictCoverageError(
            "PYPTO_CACHE_DIR must be an existing canonical directory without symlinks"
        )
    return value


def _artifact_cache_for(compiler: Any) -> Any | None:
    """Return one strict ArtifactCache handle per process and absolute root."""

    global _ARTIFACT_CACHE_LOCK, _ARTIFACT_CACHE_STATE
    root = _artifact_cache_root()
    if root is None:
        return None
    cache_type = getattr(compiler, "ArtifactCache", None)
    cached_compile = getattr(compiler, "compile_structured_strict_cached", None)
    if not callable(cache_type) or not callable(cached_compile):
        raise StrictCoverageError(
            "configured ArtifactCache requires compile_structured_strict_cached; "
            "the legacy compiler API cannot service this cache"
        )
    process_id = os.getpid()
    observed = _ARTIFACT_CACHE_STATE
    if observed is not None and observed[0] != process_id:
        # A fork can strand a Python lock in the acquired state. The child must
        # never reuse that lock or the creator-bound C++ ArtifactCache handle.
        _ARTIFACT_CACHE_LOCK = threading.Lock()
    with _ARTIFACT_CACHE_LOCK:
        observed = _ARTIFACT_CACHE_STATE
        if observed is not None:
            owner_pid, configured_root, owner_compiler, handle = observed
            if configured_root != root:
                raise StrictCoverageError(
                    "PYPTO_CACHE_DIR changed after ArtifactCache initialization"
                )
            if owner_pid == process_id:
                if owner_compiler is not compiler:
                    raise StrictCoverageError(
                        "PyPTO compiler module changed after ArtifactCache initialization"
                    )
                return handle
        try:
            handle = cache_type(root)
        except Exception as error:
            raise StrictCoverageError(
                f"PYPTO_CACHE_DIR was rejected by ArtifactCache: {error}"
            ) from error
        _ARTIFACT_CACHE_STATE = (process_id, root, compiler, handle)
        return handle


def _compile_structured(
    compiler: Any,
    program: Any,
    request: Any,
    schedule: Any,
) -> Any:
    cache = _artifact_cache_for(compiler)
    if cache is None:
        uncached_compile = getattr(compiler, "compile_structured_strict", None)
        if not callable(uncached_compile):
            raise StrictCoverageError(
                "PyPTO compiler lacks compile_structured_strict"
            )
        return uncached_compile(program, request, schedule)
    return compiler.compile_structured_strict_cached(
        program,
        request,
        schedule,
        cache,
    )


def _persistent_cache_disposition(disposition: Any) -> str:
    name = getattr(disposition, "name", None)
    if not isinstance(name, str):
        name = str(disposition).rsplit(".", 1)[-1]
    if name not in _PERSISTENT_CACHE_DISPOSITIONS:
        raise StrictCoverageError(
            f"structured cached compile returned invalid disposition {name!r}"
        )
    return name


def _persistent_cache_key(artifact: Any) -> str:
    value = str(artifact.cache_key_digest)
    if not _is_sha256(value):
        raise StrictCoverageError(
            "structured cached compile returned an invalid full cache key"
        )
    return value


class PointwiseArtifactCapture:
    """One exclusive callable-compilation capture, independent of registry order."""

    def __init__(self) -> None:
        self.owner_pid = os.getpid()
        self._artifacts: dict[str, PointwiseArtifact] = {}

    def _record(self, artifact: PointwiseArtifact) -> None:
        if os.getpid() != self.owner_pid:
            raise StrictCoverageError("pointwise artifact capture crossed a fork")
        previous = self._artifacts.setdefault(
            artifact.cache_identity_sha256,
            artifact,
        )
        if previous != artifact:
            raise StrictCoverageError(
                "pointwise artifact capture observed a cache identity conflict"
            )

    def single_artifact(self) -> PointwiseArtifact:
        if len(self._artifacts) != 1:
            raise StrictCoverageError(
                "one Inductor callable must capture exactly one PyPTO artifact; "
                f"captured={sorted(self._artifacts)}"
            )
        return next(iter(self._artifacts.values()))


@contextmanager
def capture_pointwise_artifacts() -> Iterator[PointwiseArtifactCapture]:
    """Capture artifacts compiled by one synchronous ``torch.compile`` call."""

    _require_owner_process()
    capture = PointwiseArtifactCapture()
    global _ACTIVE_CAPTURE
    with _CAPTURE_LOCK:
        if _ACTIVE_CAPTURE is not None:
            raise StrictCoverageError("a PyPTO artifact capture is already active")
        _ACTIVE_CAPTURE = capture
        try:
            # Hold the re-entrant lock for the complete synchronous compiler
            # transaction. Same-thread artifact callbacks may re-enter, while
            # unrelated compile threads cannot contaminate this capture.
            yield capture
        finally:
            if _ACTIVE_CAPTURE is not capture:
                raise StrictCoverageError("PyPTO artifact capture ownership changed")
            _ACTIVE_CAPTURE = None


def _record_active_capture(artifact: PointwiseArtifact) -> None:
    with _CAPTURE_LOCK:
        if _ACTIVE_CAPTURE is not None:
            _ACTIVE_CAPTURE._record(artifact)


def compile_cache_snapshot() -> tuple[tuple[str, str], ...]:
    """Return immutable cache/source identities for diagnostics and tests."""

    _require_owner_process()
    with _COMPILE_LOCK:
        return tuple(
            sorted(
                (artifact.cache_identity_sha256, artifact.source_node)
                for artifact in _COMPILE_CACHE.values()
            )
        )


def artifact_cache_snapshot() -> tuple[tuple[str, str, str], ...]:
    """Return persistent-cache outcomes without changing coverage identity."""

    _require_owner_process()
    with _COMPILE_LOCK:
        return tuple(
            sorted(
                (
                    artifact.cache_identity_sha256,
                    artifact.artifact_cache_key_sha256,
                    artifact.artifact_cache_disposition,
                )
                for artifact in _COMPILE_CACHE.values()
            )
        )


def clear_caches_for_testing() -> None:
    """Clear process caches; only tests may call this without process restart."""

    global _ARTIFACT_CACHE_STATE
    _require_owner_process()
    with _CAPTURE_LOCK:
        if _ACTIVE_CAPTURE is not None:
            raise StrictCoverageError("cannot clear caches during artifact capture")
    with _COMPILE_LOCK, _RUNTIME_LOCK:
        _COMPILE_CACHE.clear()
        _RUNTIME_OBJECTS.clear()
        _RUNTIME_SOURCE_NODES.clear()
        _RUNTIME_DEVICE_INDICES.clear()
        _RUNTIME_DSO_SHA256.clear()
        _RUNTIME_WRAPPER_SOURCES.clear()
    with _ARTIFACT_CACHE_LOCK:
        _ARTIFACT_CACHE_STATE = None
    _NATIVE_SOURCE_CACHE.clear()
    _REDUCTION_SOURCE_CACHE.clear()


def compile_pointwise(
    program: Any,
    *,
    tile: int = 128,
    registry_name: str | None = None,
    prewarm_runtime: bool = False,
) -> PointwiseArtifact:
    """Specialize a native tile program and compile through the strict facade."""

    _require_owner_process()
    modules = bootstrap_pypto()
    compiler = modules["compiler"]
    pypto = modules["pypto"]
    info = compiler.get_nvidia_backend_build_info()
    if not info.compiled:
        raise StrictCoverageError("PyPTO exact DSO backend is not compiled")
    if not isinstance(program, (NativePointwiseProgram, NativeReductionProgram)):
        raise StrictCoverageError("compile_pointwise requires a native program")
    dso_sha256 = pypto_dso_sha256()
    revision_identity = _backend_revision_identity(info, dso_sha256)
    cache_key = (program, tile, revision_identity)
    if isinstance(program, NativePointwiseProgram):
        if program.output_spec.device_type != "cuda":
            raise StrictCoverageError("native pointwise compilation requires CUDA")
        device_index = program.output_spec.device_index
        if device_index is None:
            raise StrictCoverageError(
                "native pointwise compilation requires a static CUDA device index"
            )
    else:
        if program.device_type != "cuda":
            raise StrictCoverageError("native reduction compilation requires CUDA")
        device_index = program.device_index

    from ..activity_trace import trace_window_active

    with _COMPILE_LOCK:
        trace_was_active = trace_window_active()
        cached = _COMPILE_CACHE.get(cache_key)
        if cached is None:
            if trace_was_active:
                raise StrictCoverageError(
                    "PyPTO Inductor artifact was not compiled before the trace window"
                )
            request = _reference_request(
                compiler,
                pypto,
                info,
                require_live_target=prewarm_runtime,
                device_index=device_index,
            )
            tile_shape = (
                _pointwise_tile_shape(program.shape, tile)
                if isinstance(program, NativePointwiseProgram)
                else (tile,)
            )
            schedule = _reference_schedule(tile_shape)
            specialized = (
                program.specialize(tile)
                if isinstance(program, NativePointwiseProgram)
                else program.specialize()
            )
            pypto_source = (
                program.native_source(tile)
                if isinstance(program, NativePointwiseProgram)
                else program.native_source()
            )
            pypto_source_sha = hashlib.sha256(
                pypto_source.encode("utf-8")
            ).hexdigest()
            result = _compile_structured(
                compiler,
                specialized,
                request,
                schedule,
            )
            if trace_window_active():
                raise StrictCoverageError(
                    "a trace window began during PyPTO artifact compilation"
                )
            artifact = result.artifact
            artifact_cache_key = _persistent_cache_key(artifact)
            artifact_cache_disposition = _persistent_cache_disposition(
                result.disposition
            )
            kernel = artifact.kernel_abi
            cubin = bytes(artifact.device_code)
            cubin_sha = hashlib.sha256(cubin).hexdigest()
            build_spec_sha = hashlib.sha256(
                result.build_spec.serialize()
            ).hexdigest()
            artifact_sha = hashlib.sha256(artifact.serialize()).hexdigest()
            identity_sha = hashlib.sha256(
                b"pypto-inductor-source-artifact-v2\0"
                + repr(cache_key).encode("utf-8")
                + b"\0"
                + pypto_source_sha.encode("ascii")
                + b"\0"
                + build_spec_sha.encode("ascii")
                + b"\0"
                + artifact_sha.encode("ascii")
                + b"\0"
                + cubin_sha.encode("ascii")
            ).hexdigest()
            kernel_name = f"pypto_inductor_{identity_sha[:16]}"
            source_node = f"torch-inductor:{identity_sha[:16]}"
            cached = PointwiseArtifact(
                kernel_name=kernel_name,
                entry_name=kernel.entry_function_name,
                build_spec_sha256=build_spec_sha,
                artifact_sha256=artifact_sha,
                cubin_sha256=cubin_sha,
                cubin_bytes=len(cubin),
                grid=tuple(kernel.grid_abi.static_dimensions),
                argument_count=kernel.argument_layout.total_kernel_argument_count,
                workspace_bytes=kernel.workspace_abi.size_bytes,
                fallback_used=bool(artifact.fallback_used),
                pypto_source=pypto_source,
                pypto_source_sha256=pypto_source_sha,
                cache_identity_sha256=identity_sha,
                artifact_cache_key_sha256=artifact_cache_key,
                artifact_cache_disposition=artifact_cache_disposition,
                source_node=source_node,
                dso_sha256=dso_sha256,
            )
            retain_runtime_objects(
                kernel_name,
                artifact,
                request,
                source_node=source_node,
                device_index=device_index,
                dso_sha256=dso_sha256,
            )
            _COMPILE_CACHE[cache_key] = cached
        retained = runtime_objects(cached.kernel_name)
        if retained is None:
            raise StrictCoverageError(
                f"cached PyPTO artifact {cached.kernel_name!r} lost runtime objects"
            )
        artifact, request = retained
        if registry_name is not None:
            if trace_was_active and runtime_objects(registry_name) is None:
                raise StrictCoverageError(
                    "PyPTO Inductor runtime alias was not prepared before trace"
                )
            retain_runtime_objects(
                registry_name,
                artifact,
                request,
                source_node=cached.source_node,
                device_index=device_index,
                dso_sha256=cached.dso_sha256,
            )
        if prewarm_runtime:
            from . import runtime_bridge

            if trace_was_active:
                if not runtime_bridge.kernel_is_prewarmed(cached.kernel_name):
                    raise StrictCoverageError(
                        "PyPTO executable was not prewarmed before trace"
                    )
            else:
                runtime_bridge.prewarm_kernel(cached.kernel_name)
        if not trace_was_active and trace_window_active():
            raise StrictCoverageError(
                "a trace window began during PyPTO compile/prewarm transaction"
            )
        _record_active_capture(cached)
        return cached


__all__ = (
    "PointwiseArtifact",
    "PointwiseArtifactCapture",
    "PointwiseSourceEvidence",
    "WrapperLaunchSource",
    "NativePointwiseProgram",
    "NativeReductionProgram",
    "BackendRevisionIdentity",
    "PointwiseProgramBuilder",
    "PointwiseTensorSpec",
    "artifact_cache_snapshot",
    "bootstrap_pypto",
    "capture_pointwise_artifacts",
    "clear_caches_for_testing",
    "compile_cache_snapshot",
    "compile_pointwise",
    "current_backend_revision_identity",
    "pointwise_source_evidence",
    "pypto_dso_path",
    "pypto_dso_sha256",
    "record_wrapper_launch_source",
    "retain_runtime_objects",
    "runtime_objects",
    "runtime_device_index",
    "runtime_dso_sha256",
    "runtime_source_node",
)
