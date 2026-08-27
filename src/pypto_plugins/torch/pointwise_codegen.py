"""FX-graph-to-PyPTO FusedPointwiseV2 code generation core.

This module owns the plugin's first real Inductor-facing capability: it
translates a bounded chain of pointwise elementwise operators into the
exact PyPTO ``FusedPointwiseV2`` HIR program, compiles it through
``pypto.compiler.compile_structured_strict`` against the pinned exact-DSO
backend, and exposes the immutable artifact identity plus a generated
Python launcher source for the wrapper layer. The plugin contains no
kernel algorithms; all code generation and execution semantics live in
the PyPTO compiler product.
"""

from __future__ import annotations

import hashlib
import importlib.util
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
_PYPTO_PACKAGE = (
    "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto/python/pypto"
)

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


class PointwiseProgramBuilder:
    """Build the bounded FusedPointwiseV2 HIR chain from a flat op list."""

    MAX_ASSIGNMENTS = 64
    MAX_INPUTS = 16
    _DTYPE_NAMES = ("float32", "bfloat16")

    def __init__(self, shape: tuple[int, ...], dtype_name: str) -> None:
        if dtype_name not in self._DTYPE_NAMES:
            raise StrictCoverageError(
                f"unsupported fused pointwise dtype: {dtype_name!r}"
            )
        modules = bootstrap_pypto()
        self._pypto = modules["pypto"]
        self._ir = modules["ir"]
        self._dtype = {
            "float32": self._pypto.DataType.FP32,
            "bfloat16": self._pypto.DataType.BF16,
        }[dtype_name]
        self._tensor_type = self._ir.TensorType(list(shape), self._dtype)
        self._shape_tuple = tuple(int(dim) for dim in shape)
        self._span = self._ir.Span("pypto_plugins.pointwise_codegen", 1, 1)
        self._statements: list[Any] = []
        self._previous: Any | None = None
        self.inputs: list[Any] = []
        self.outputs: list[Any] = []

    def add_input(self, name: str) -> Any:
        if len(self.inputs) >= self.MAX_INPUTS:
            raise StrictCoverageError("fused pointwise chain exceeds 16 inputs")
        var = self._ir.Var(name, self._tensor_type, self._span)
        self.inputs.append(var)
        return var

    def add_broadcast_input(self, name: str) -> Any:
        """Add a [M,1,...] row input for row-expand fused operands."""
        if len(self.inputs) >= self.MAX_INPUTS:
            raise StrictCoverageError("fused pointwise chain exceeds 16 inputs")
        row_shape = [dim if index == 0 else 1 for index, dim in enumerate(self.shape)]
        row_type = self._ir.TensorType(row_shape, self._dtype)
        var = self._ir.Var(name, row_type, self._span)
        self.inputs.append(var)
        return var

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape_tuple

    def scalar(self, value: float) -> Any:
        return self._ir.ConstFloat(value, self._dtype, self._span)

    def emit(self, op_name: str, arguments: list[Any]) -> Any:
        if len(self._statements) >= self.MAX_ASSIGNMENTS:
            raise StrictCoverageError("fused pointwise chain exceeds 64 assignments")
        value = self._ir.Var("ignored", self._tensor_type, self._span)
        call = self._ir.Call(
            self._ir.get_op(op_name), arguments, self._tensor_type, self._span
        )
        self._statements.append(self._ir.AssignStmt(value, call, self._span))
        self._previous = value
        return value

    def mark_output(self, var: Any) -> None:
        self.outputs.append(var)

    def build(self) -> Any:
        if not self.inputs or not self.outputs or not self._statements:
            raise StrictCoverageError(
                "fused pointwise program needs at least one input, one op "
                "and one output"
            )
        statements = list(self._statements)
        statements.append(self._ir.ReturnStmt(list(self.outputs), self._span))
        function = self._ir.Function(
            "ignored_fused_pointwise",
            list(self.inputs),
            [self._tensor_type for _ in self.outputs],
            self._ir.SeqStmts(statements, self._span),
            self._span,
        )
        return self._ir.Program(
            [function], "pypto_plugins_pointwise", self._span
        )


_RUNTIME_OBJECTS: dict[str, tuple[Any, Any]] = {}


def retain_runtime_objects(name: str, artifact: Any, request: Any) -> None:
    """Keep the live artifact/request pair for the wrapper-side bridge."""

    _RUNTIME_OBJECTS.setdefault(name, (artifact, request))


def runtime_objects(name: str) -> tuple[Any, Any] | None:
    return _RUNTIME_OBJECTS.get(name)


@dataclass(frozen=True, slots=True)
class PointwiseArtifact:
    """The compiled FusedPointwiseV2 artifact identity."""

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
    """Compile an HIR program through the exact-DSO strict facade."""

    modules = bootstrap_pypto()
    compiler = modules["compiler"]
    pypto = modules["pypto"]
    info = compiler.get_nvidia_backend_build_info()
    if not info.compiled:
        raise StrictCoverageError("PyPTO exact DSO backend is not compiled")
    request = _reference_request(compiler, pypto, info)
    schedule = _reference_schedule(tile)
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
    "PointwiseProgramBuilder",
    "bootstrap_pypto",
    "compile_pointwise",
    "retain_runtime_objects",
    "runtime_objects",
)
