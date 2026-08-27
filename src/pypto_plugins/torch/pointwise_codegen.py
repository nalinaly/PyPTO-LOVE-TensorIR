"""FX pointwise chains lowered to native PyPTO tile-DSL graphs."""

from __future__ import annotations

import hashlib
import importlib.util
import linecache
import os
import pathlib
import sys
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from ..errors import StrictCoverageError

_DSO_ENVIRONMENT = "PYPTO_PLUGINS_PYPTO_DSO"
_DEFAULT_DSO = (
    "/home/zhaosiying/pypto-love-tensor-ir/builds/"
    "pypto-opext-on-a589f79/product/"
    "pypto_core.cpython-314-x86_64-linux-gnu.so"
)
_PYPTO_PACKAGE = "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto/python/pypto"

_lock = threading.Lock()
_pypto_modules: dict[str, ModuleType] | None = None


def bootstrap_pypto(
    dso_path: str | pathlib.Path | None = None,
) -> dict[str, ModuleType]:
    """Bind the exact-DSO ``pypto`` package into this process once."""

    global _pypto_modules
    with _lock:
        if _pypto_modules is not None:
            return _pypto_modules
    resolved = pathlib.Path(
        dso_path or os.environ.get(_DSO_ENVIRONMENT) or _DEFAULT_DSO
    ).resolve(strict=True)
    if resolved.is_dir():
        resolved = next(resolved.glob("pypto_core*.so")).resolve(strict=True)
    package_root = pathlib.Path(_PYPTO_PACKAGE).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        "pypto",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    assert spec is not None and spec.loader is not None
    pypto = importlib.util.module_from_spec(spec)
    core_spec = importlib.util.spec_from_file_location("pypto.pypto_core", resolved)
    assert core_spec is not None and core_spec.loader is not None
    pypto_core = importlib.util.module_from_spec(core_spec)
    with _lock:
        if _pypto_modules is None:
            existing = sys.modules.get("pypto.pypto_core")
            if existing is not None and existing is not pypto_core:
                raise StrictCoverageError(
                    "a foreign pypto_core module already occupies the import slot"
                )
            sys.modules["pypto.pypto_core"] = pypto_core
            core_spec.loader.exec_module(pypto_core)
            spec.loader.exec_module(pypto)
            sys.modules["pypto"] = pypto
            _pypto_modules = {
                "pypto": pypto,
                "ir": pypto.ir,
                "compiler": pypto.compiler,
            }
    return _pypto_modules


@dataclass(frozen=True, slots=True)
class _Value:
    key: int


@dataclass(frozen=True, slots=True)
class _Scalar:
    value: float


@dataclass(frozen=True, slots=True)
class _Input:
    name: str
    shape: tuple[int, ...]
    value: _Value


@dataclass(frozen=True, slots=True)
class _Instruction:
    op_name: str
    arguments: tuple[_Value | _Scalar, ...]
    result: _Value


@dataclass(frozen=True, slots=True)
class NativePointwiseProgram:
    """Immutable native tile program waiting for its schedule tile."""

    shape: tuple[int, ...]
    dtype_name: str
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
        import torch
        import pypto.language as pl

        dtype = torch.float32 if self.dtype_name == "float32" else torch.bfloat16
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
                offset if input_spec.shape[axis] != 1 else "0"
                for axis, offset in enumerate(output_offsets)
            ]
            input_tile = [
                output_tile[axis] if input_spec.shape[axis] != 1 else 1
                for axis in range(len(self.shape))
            ]
            name = f"value_{input_spec.value.key}"
            lines.append(
                f"{indent}{name} = pl.load(input_{index}, "
                f"[{', '.join(input_offsets)}], {input_tile})"
            )
            names[input_spec.value.key] = name
        for instruction in self.instructions:
            op = self._OPS.get(instruction.op_name)
            if op is None:
                raise StrictCoverageError(
                    f"native pointwise has no tile op for {instruction.op_name!r}"
                )
            arguments: list[str] = []
            for argument in instruction.arguments:
                if isinstance(argument, _Scalar):
                    arguments.append(repr(argument.value))
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
        samples = [
            torch.empty(item.shape, dtype=dtype, device="meta") for item in self.inputs
        ]
        samples.append(torch.empty(self.shape, dtype=dtype, device="meta"))
        return kernel.specialize(*samples)


_NATIVE_SOURCE_CACHE: dict[tuple[NativePointwiseProgram, int], str] = {}


@dataclass(frozen=True, slots=True)
class NativeReductionProgram:
    """Trailing-axis row sum/max expressed as a native tile graph."""

    input_shape: tuple[int, ...]
    dtype_name: str
    mode: str

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

    def __init__(self, shape: tuple[int, ...], dtype_name: str) -> None:
        if dtype_name not in self._DTYPE_NAMES:
            raise StrictCoverageError(
                f"unsupported fused pointwise dtype: {dtype_name!r}"
            )
        self._shape_tuple = tuple(int(dim) for dim in shape)
        if not self._shape_tuple or any(dim <= 0 for dim in self._shape_tuple):
            raise StrictCoverageError("native pointwise shape must be positive")
        self._dtype_name = dtype_name
        self._instructions: list[_Instruction] = []
        self._next_key = 0
        self.inputs: list[_Input] = []
        self.outputs: list[_Value] = []

    def _value(self) -> _Value:
        value = _Value(self._next_key)
        self._next_key += 1
        return value

    def add_input(self, name: str) -> Any:
        if len(self.inputs) >= self.MAX_INPUTS:
            raise StrictCoverageError("fused pointwise chain exceeds 16 inputs")
        value = self._value()
        self.inputs.append(_Input(name, self._shape_tuple, value))
        return value

    def add_broadcast_input(self, name: str) -> Any:
        """Add a [M,1,...] row input for row-expand fused operands."""
        if len(self.inputs) >= self.MAX_INPUTS:
            raise StrictCoverageError("fused pointwise chain exceeds 16 inputs")
        row_shape = [dim if index == 0 else 1 for index, dim in enumerate(self.shape)]
        value = self._value()
        self.inputs.append(_Input(name, tuple(row_shape), value))
        return value

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape_tuple

    def scalar(self, value: float) -> Any:
        return _Scalar(float(value))

    def emit(self, op_name: str, arguments: list[Any]) -> Any:
        if len(self._instructions) >= self.MAX_ASSIGNMENTS:
            raise StrictCoverageError("fused pointwise chain exceeds 64 assignments")
        if not all(isinstance(item, (_Value, _Scalar)) for item in arguments):
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
            tuple(self.inputs),
            tuple(self._instructions),
            self.outputs[0],
        )


_RUNTIME_OBJECTS: dict[str, tuple[Any, Any]] = {}


def retain_runtime_objects(name: str, artifact: Any, request: Any) -> None:
    """Keep the live artifact/request pair for the wrapper-side bridge."""

    _RUNTIME_OBJECTS.setdefault(name, (artifact, request))


def runtime_objects(name: str) -> tuple[Any, Any] | None:
    return _RUNTIME_OBJECTS.get(name)


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
    launcher_source: str


def _live_target_or_none(compiler: Any) -> Any:
    """Observe the live NVIDIA target when CUDA is available in-process."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.zeros(1, device="cuda")  # force the primary context current
        from pypto.runtime import nvidia as runtime

        observation = runtime.observe_current_nvidia_runtime(
            "610.74",
            "/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/lib/"
            "python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13",
        )
        return observation.target_info
    except Exception:
        return None


def _reference_request(compiler: Any, pypto: Any, info: Any) -> Any:
    """Build the SM120 CompileRequest from live backend build info."""

    from pypto.compiler import (
        CompileRequest,
        NvidiaTargetInfo,
        TargetTraits,
        ToolchainIdentity,
    )

    live_target = _live_target_or_none(compiler)

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
            "610.74",
            info.tensor_ir_revision,
            info.cuda_tile_revision,
            [pypto.DataType.BF16, pypto.DataType.FP32],
        )
    return CompileRequest(target, toolchain)


def _reference_schedule(tile: int) -> Any:
    from pypto.compiler import (
        CanonicalSchedule,
        ScheduleParameter,
        ScheduleValueKind,
    )

    parameter = ScheduleParameter
    unsigned = ScheduleValueKind.UnsignedInteger
    return CanonicalSchedule(
        [parameter("codegen_strategy", ScheduleValueKind.Text, "layout-propagation")],
        [parameter("dim_000", unsigned, str(tile))],
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


def compile_pointwise(
    program: Any, *, tile: int = 128, registry_name: str | None = None
) -> PointwiseArtifact:
    """Specialize a native tile program and compile through the strict facade."""

    modules = bootstrap_pypto()
    compiler = modules["compiler"]
    pypto = modules["pypto"]
    info = compiler.get_nvidia_backend_build_info()
    if not info.compiled:
        raise StrictCoverageError("PyPTO exact DSO backend is not compiled")
    request = _reference_request(compiler, pypto, info)
    schedule = _reference_schedule(tile)
    if isinstance(program, NativePointwiseProgram):
        program = program.specialize(tile)
    elif isinstance(program, NativeReductionProgram):
        program = program.specialize()
    result = compiler.compile_structured_strict(program, request, schedule)
    artifact = result.artifact
    kernel_name = (
        "pypto_pointwise_"
        + hashlib.sha256(bytes(artifact.device_code)).hexdigest()[:12]
    )
    kernel = artifact.kernel_abi
    cubin = bytes(artifact.device_code)
    cubin_sha = hashlib.sha256(cubin).hexdigest()
    retain_runtime_objects(kernel_name, artifact, request)
    if registry_name is not None:
        retain_runtime_objects(registry_name, artifact, request)
    return PointwiseArtifact(
        kernel_name=kernel_name,
        entry_name=kernel.entry_function_name,
        build_spec_sha256=hashlib.sha256(result.build_spec.serialize()).hexdigest(),
        artifact_sha256=hashlib.sha256(artifact.serialize()).hexdigest(),
        cubin_sha256=cubin_sha,
        cubin_bytes=len(cubin),
        grid=tuple(kernel.grid_abi.static_dimensions),
        argument_count=kernel.argument_layout.total_kernel_argument_count,
        workspace_bytes=kernel.workspace_abi.size_bytes,
        fallback_used=bool(artifact.fallback_used),
        launcher_source=(
            f"def launch(stream, *tensors):\n"
            f"    return _PYPTO_REGISTRY[{cubin_sha!r}].launch(stream, tensors)\n"
        ),
    )


__all__ = (
    "PointwiseArtifact",
    "NativePointwiseProgram",
    "NativeReductionProgram",
    "PointwiseProgramBuilder",
    "bootstrap_pypto",
    "compile_pointwise",
    "retain_runtime_objects",
    "runtime_objects",
)
