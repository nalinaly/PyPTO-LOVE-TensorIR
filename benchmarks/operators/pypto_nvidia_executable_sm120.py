#!/usr/bin/env python3
"""Correctness-only real-SM120 smoke for the exact PyPTO NvidiaExecutable.

This is deliberately not a performance benchmark.  The module imports only
the standard library before the parent-owned start barrier is authenticated.
Torch and the exact PyPTO DSO are loaded afterwards without site processing.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any


RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")

STATIC_SOURCE = b"""
module {
  nv_tensor_ir.graph @add_op_simple(
    %a: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"},
    %b: tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}
  ) -> (tensor<8x8xf32> {nv_tensor_ir.stride = "(8,1)"}) {
    %add = add %a, %b : tensor<8x8xf32>
    results %add : tensor<8x8xf32>
  }
}
"""

DYNAMIC_SOURCE = b"""
module {
  nv_tensor_ir.graph @add_dynamic_shape(
    %a: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"},
    %b: tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}
  ) -> (tensor<?x?xf32> {nv_tensor_ir.stride = "(?,1)"}) {
    %add = add %a, %b : tensor<?x?xf32>
    results %add : tensor<?x?xf32>
  }
}
"""

SCALAR_SOURCE = b"""
module {
  nv_tensor_ir.graph @add_tensor_scalar(
      %in: tensor<4x4x4xf16> {nv_tensor_ir.alignment = 16 : i64},
      %scalar: f32)
      -> (tensor<4x4x4xf16> {nv_tensor_ir.alignment = 16 : i64}) {
    %convert = convert %in : tensor<4x4x4xf16> -> tensor<4x4x4xf32>
    %splat = splat %scalar : tensor<4x4x4xf32>
    %add = add %convert, %splat : tensor<4x4x4xf32>
    %result = convert %add : tensor<4x4x4xf32> -> tensor<4x4x4xf16>
    results %result : tensor<4x4x4xf16>
  }
}
"""


class SmokeError(RuntimeError):
    """A fail-closed correctness-smoke error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def duplicate_key_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise SmokeError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_canonical_json(
    path: Path, description: str
) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise SmokeError(f"{description} must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"{description} is not valid JSON") from error
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise SmokeError(f"{description} is not canonical JSON")
    return value, raw


def publish_no_replace(path: Path, payload: bytes, mode: int = 0o444) -> str:
    if path.exists() or path.is_symlink():
        raise SmokeError(f"refusing to replace smoke evidence: {path}")
    parent = path.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SmokeError(f"refusing to replace smoke evidence: {path}") from error
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(payload)


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise SmokeError(f"cannot load exact module {name} from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _process_start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    return int(fields[19])


def _workspace_from_environment() -> tuple[Path, str]:
    raw_workspace = os.environ.get("PYPTO_WORKSPACE_ROOT", "")
    workspace = Path(raw_workspace)
    if not workspace.is_absolute() or workspace.resolve(strict=True) != workspace:
        raise SmokeError("PYPTO_WORKSPACE_ROOT is not one canonical directory")
    run_id = os.environ.get("PYPTO_RUN_ID", "")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise SmokeError("PYPTO_RUN_ID is malformed")
    if os.environ.get("PYPTO_RUN_MODE") != "gpu-smoke":
        raise SmokeError("runner requires gpu-smoke mode")
    if os.environ.get("PYPTO_ALLOW_FALLBACK") != "0":
        raise SmokeError("runner requires PYPTO_ALLOW_FALLBACK=0")
    if os.environ.get("PYPTO_STRICT_COVERAGE") != "1":
        raise SmokeError("runner requires PYPTO_STRICT_COVERAGE=1")
    if os.environ.get("PYTHONPATH", ""):
        raise SmokeError("runner requires an empty PYTHONPATH")
    if os.environ.get("SGLANG_PLUGINS") != "__pypto_exact_nvidia_smoke_no_plugins__":
        raise SmokeError("runner requires the no-plugin SGLang policy")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise SmokeError("runner requires Python -I -B -S")
    forbidden = {"torch", "pypto", "triton", "sglang", "flashinfer"}
    if forbidden & set(sys.modules):
        raise SmokeError("GPU/framework modules loaded before the start barrier")
    return workspace, run_id


def wait_for_start_barrier(workspace: Path, run_id: str) -> dict[str, object]:
    expected = workspace / "runs" / run_id / "gpu-smoke-start-barrier.json"
    raw_path = os.environ.get("PYPTO_GPU_SMOKE_START_BARRIER", "")
    if Path(raw_path) != expected:
        raise SmokeError("GPU-smoke start-barrier path differs from the run identity")
    deadline = time.monotonic() + 60.0
    while not expected.exists():
        if time.monotonic() >= deadline:
            raise SmokeError("timed out before parent GPU-smoke gate release")
        time.sleep(0.05)
    barrier, _raw = load_canonical_json(expected, "GPU-smoke start barrier")
    if (
        barrier.get("schema") != 1
        or barrier.get("run_id") != run_id
        or barrier.get("pid") != os.getpid()
        or barrier.get("pgid") != os.getpgrp()
        or barrier.get("start_ticks") != _process_start_ticks(os.getpid())
    ):
        raise SmokeError("GPU-smoke start barrier does not identify this process")
    gate_path = Path(str(barrier.get("gate_path", "")))
    expected_gate = workspace / "runs" / run_id / "gpu-smoke-gate.json"
    if gate_path != expected_gate:
        raise SmokeError("GPU-smoke gate path differs from the run identity")
    gate, gate_raw = load_canonical_json(gate_path, "GPU-smoke gate")
    if sha256_bytes(gate_raw) != barrier.get("gate_sha256"):
        raise SmokeError("GPU-smoke gate digest join failed")
    if (
        gate.get("run_id") != run_id
        or gate.get("pid") != os.getpid()
        or gate.get("pgid") != os.getpgrp()
        or gate.get("start_ticks") != _process_start_ticks(os.getpid())
    ):
        raise SmokeError("GPU-smoke gate does not identify this process")
    return {"barrier": barrier, "gate": gate}


def load_contract_and_child_gate(
    workspace: Path, run_id: str, parent_gate: dict[str, object]
) -> tuple[Any, dict[str, object]]:
    contract_path = workspace / "tools" / "_pypto_nvidia_executable_sm120_contract.py"
    contract = _load_module("_pypto_nvidia_executable_sm120_contract", contract_path)
    control = _load_module(
        "_pypto_nvidia_sm120_control_manifest",
        workspace / "tools" / "_pypto_nvidia_sm120_control_manifest.py",
    )
    control_identity = control.validate_control_manifest(workspace)
    if parent_gate.get("control_manifest") != control_identity:
        raise SmokeError("parent/child control-manifest identity differs")
    runner = Path(__file__).resolve(strict=True)
    if (
        runner != workspace / contract.RUNNER_RELATIVE_PATH
        or runner.stat().st_size != contract.RUNNER_SIZE
        or sha256_file(runner) != contract.RUNNER_SHA256
    ):
        raise SmokeError("runner bytes differ from the fixed smoke contract")
    if contract.fixed_child_command(workspace) != sys.orig_argv:
        raise SmokeError("live Python argv differs from the fixed smoke command")
    preflight = _load_module("preflight", workspace / "tools" / "preflight.py")
    static_identity = preflight.static_torch_identity()
    if static_identity.get("static_identity_error"):
        raise SmokeError(str(static_identity["static_identity_error"]))
    if preflight.mem_available_kib() < preflight.GPU_SMOKE_MEMORY_FLOOR_KIB:
        raise SmokeError("child admission host-memory floor failed")
    gpu = preflight.nvidia_identity()
    if gpu.get("compute_capability") != "12.0":
        raise SmokeError("child admission target is not SM120")
    free_memory_mib = int(gpu["memory_mib"]) - int(gpu["used_mib"])
    if free_memory_mib < preflight.GPU_SMOKE_FREE_MEMORY_FLOOR_MIB:
        raise SmokeError("child admission GPU-memory floor failed")
    compute_pids = preflight.nvidia_compute_pids()
    if compute_pids:
        raise SmokeError(f"child admission found NVIDIA compute PIDs: {compute_pids}")
    _all, protected, _workspace = preflight.process_table()
    protected_runtime, unreadable = preflight.protected_nvidia_runtime_mappings(
        protected
    )
    if protected_runtime or unreadable:
        raise SmokeError(
            "child admission cannot prove protected NVIDIA isolation: "
            f"runtime={protected_runtime}, unreadable={unreadable}"
        )
    protected_heavy = [
        process for process in protected if preflight.is_heavy_command(process.command)
    ]
    requested = os.environ.get("PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED") == "1"
    if protected_heavy and not requested:
        raise SmokeError("protected CPU lane exists without GPU-smoke authorization")
    if requested and os.environ.get("PYPTO_GPU_SMOKE_AUTHORIZATION") != (
        contract.GPU_SMOKE_AUTHORIZATION
    ):
        raise SmokeError("GPU-smoke authorization identity differs")
    return contract, {
        "static_identity": static_identity,
        "gpu": gpu,
        "free_memory_mib": free_memory_mib,
        "protected_heavy_pids": [process.pid for process in protected_heavy],
        "protected_runtime_pids": protected_runtime,
        "unreadable_protected_maps": unreadable,
        "nvidia_compute_pids": sorted(compute_pids),
        "control_manifest": control_identity,
    }


def bootstrap_exact_pypto(workspace: Path, product: Path) -> ModuleType:
    package = workspace / "projects" / "pypto" / "python" / "pypto"
    cores = sorted(product.glob("pypto_core*.so"))
    if len(cores) != 1:
        raise SmokeError("exact product directory does not contain one PyPTO DSO")
    core = cores[0].resolve(strict=True)
    package_spec = importlib.util.spec_from_file_location(
        "pypto",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    if package_spec is None or package_spec.loader is None:
        raise SmokeError("cannot create exact PyPTO package specification")
    package_module = importlib.util.module_from_spec(package_spec)
    sys.modules["pypto"] = package_module
    core_spec = importlib.util.spec_from_file_location("pypto.pypto_core", core)
    if core_spec is None or core_spec.loader is None:
        raise SmokeError("cannot create exact PyPTO DSO specification")
    core_module = importlib.util.module_from_spec(core_spec)
    sys.modules["pypto.pypto_core"] = core_module
    core_spec.loader.exec_module(core_module)
    package_spec.loader.exec_module(package_module)
    return package_module


def mapped_library_paths(marker: str) -> list[str]:
    paths: set[str] = set()
    for line in Path("/proc/self/maps").read_text(errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        candidate = fields[-1]
        if marker not in candidate or not candidate.startswith("/"):
            continue
        paths.add(str(Path(candidate).resolve(strict=True)))
    return sorted(paths)


def toolchain_identity(compiler: Any, info: Any) -> Any:
    return compiler.ToolchainIdentity(
        pypto_revision=info.pypto_revision,
        tensor_ir_revision=info.tensor_ir_revision,
        cuda_tile_revision=info.cuda_tile_revision,
        llvm_revision=info.llvm_revision,
        cuda_toolkit_root=info.cuda_toolkit_root,
        cuda_toolkit_version=info.cuda_toolkit_version,
        tileiras_real_path=info.tileiras_real_path,
        tileiras_version=info.tileiras_version,
        tileiras_sha256=info.tileiras_sha256,
    )


def parameter(compiler: Any, name: str, kind: Any, value: str) -> Any:
    return compiler.ScheduleParameter(name, kind, value)


def schedule(compiler: Any, tile_sizes: tuple[int, ...]) -> Any:
    unsigned = compiler.ScheduleValueKind.UnsignedInteger
    return compiler.CanonicalSchedule(
        schedule=[
            parameter(
                compiler,
                "codegen_strategy",
                compiler.ScheduleValueKind.Text,
                "layout-propagation",
            )
        ],
        tile=[
            parameter(compiler, f"dim_{index:03d}", unsigned, str(value))
            for index, value in enumerate(tile_sizes)
        ],
        layout=[],
        persistence=[],
        cta=[parameter(compiler, "count", unsigned, "1")],
        warp=[parameter(compiler, "count", unsigned, "4")],
        stage=[],
        options=[
            parameter(compiler, "bytecode_major", unsigned, "13"),
            parameter(compiler, "bytecode_minor", unsigned, "3"),
            parameter(compiler, "bytecode_tag", unsigned, "0"),
            parameter(compiler, "max_candidates", unsigned, "0"),
            parameter(
                compiler,
                "uniform_signature",
                compiler.ScheduleValueKind.Boolean,
                "false",
            ),
        ],
    )


def tensor_descriptor(
    compiler: Any,
    shape: list[int | None],
    strides: list[int | None],
    dynamic_sizes: int,
    dynamic_strides: int,
    explicit_strides: bool,
) -> Any:
    return compiler.ArtifactArgumentDescriptor(
        kind=compiler.ArtifactOperandKind.Tensor,
        rank=len(shape),
        shape=shape,
        strides=strides,
        dynamic_size_count=dynamic_sizes,
        dynamic_stride_count=dynamic_strides,
        explicit_strides=explicit_strides,
        scalar_size_bytes=0,
    )


def scalar_descriptor(compiler: Any) -> Any:
    return compiler.ArtifactArgumentDescriptor(
        kind=compiler.ArtifactOperandKind.Scalar,
        rank=0,
        shape=[],
        strides=[],
        dynamic_size_count=0,
        dynamic_stride_count=0,
        explicit_strides=False,
        scalar_size_bytes=4,
    )


def kernel_abi(
    compiler: Any,
    *,
    entry: str,
    descriptors: list[Any],
    input_count: int,
    argument_count: int,
    packing: Any,
    grid_policy: Any,
    grid: tuple[int, int, int],
    tile_sizes: tuple[int, ...],
) -> Any:
    return compiler.ArtifactKernelAbi(
        runtime_kernel_name="tensor_ir_rtk",
        entry_function_name=entry,
        argument_layout=compiler.ArtifactArgumentLayout(
            operand_descriptors=descriptors,
            input_operand_count=input_count,
            total_kernel_argument_count=argument_count,
            uniform_signature=False,
        ),
        argument_packing_policy=packing,
        grid_abi=compiler.ArtifactGridAbi(
            policy=grid_policy,
            shape_operand_index=0,
            static_dimensions=list(grid),
            tile_sizes=list(tile_sizes),
        ),
        workspace_abi=compiler.ArtifactWorkspaceAbi(
            compiler.ArtifactWorkspaceKind.Static, 0, 1
        ),
        launch_abi=compiler.ArtifactLaunchAbi(
            block_dimensions=[1, 1, 1],
            cluster_scheduling_policy=(compiler.ArtifactClusterSchedulingPolicy.Spread),
            dynamic_shared_memory_bytes=0,
            kernel_argument_slot_bytes=8,
        ),
        loader_abi=compiler.ArtifactLoaderAbi(13_000, 13_000),
    )


def make_case_contracts(compiler: Any) -> dict[str, tuple[bytes, Any, tuple[int, ...]]]:
    static_descriptor = tensor_descriptor(compiler, [8, 8], [8, 1], 0, 0, True)
    static = kernel_abi(
        compiler,
        entry="add_op_simple",
        descriptors=[static_descriptor, static_descriptor, static_descriptor],
        input_count=2,
        argument_count=3,
        packing=compiler.ArtifactArgumentPackingPolicy.PointerOnly,
        grid_policy=compiler.ArtifactGridPolicy.Static,
        grid=(4, 1, 1),
        tile_sizes=(16,),
    )
    dynamic_descriptor = tensor_descriptor(
        compiler, [None, None], [None, 1], 2, 1, True
    )
    dynamic = kernel_abi(
        compiler,
        entry="add_dynamic_shape",
        descriptors=[dynamic_descriptor, dynamic_descriptor, dynamic_descriptor],
        input_count=2,
        argument_count=12,
        packing=compiler.ArtifactArgumentPackingPolicy.Flat,
        grid_policy=compiler.ArtifactGridPolicy.TileBasedRuntime,
        grid=(1, 1, 1),
        tile_sizes=(8, 8),
    )
    scalar_tensor = tensor_descriptor(compiler, [4, 4, 4], [], 0, 0, False)
    scalar = kernel_abi(
        compiler,
        entry="add_tensor_scalar",
        descriptors=[scalar_tensor, scalar_descriptor(compiler), scalar_tensor],
        input_count=2,
        argument_count=3,
        packing=compiler.ArtifactArgumentPackingPolicy.PointerOnly,
        grid_policy=compiler.ArtifactGridPolicy.Static,
        grid=(4, 1, 1),
        tile_sizes=(16,),
    )
    return {
        "static": (STATIC_SOURCE, static, (16,)),
        "dynamic": (DYNAMIC_SOURCE, dynamic, (8, 8)),
        "scalar": (SCALAR_SOURCE, scalar, (16,)),
    }


def build_spec(
    compiler: Any,
    request: Any,
    source: bytes,
    abi: Any,
    tiles: tuple[int, ...],
    pipeline: str,
) -> Any:
    return compiler.KernelBuildSpec(
        source_ir_digest=compiler.Artifact.compute_source_ir_digest(source),
        callable_abi_digest=abi.identity_digest,
        semantic_route=compiler.SemanticRoute.StructuredTensorIr,
        pipeline_revision=pipeline,
        resolved_schedule=schedule(compiler, tiles),
        static_specialization_digest="a" * 64,
        symbolic_specialization_digest="b" * 64,
        argument_abi_digest="c" * 64,
        result_abi_digest="d" * 64,
        mutation_abi_digest="e" * 64,
        compile_request_byte_identity_digest=request.byte_compile_identity_digest,
        catalog_provenance=None,
    )


def logical_tensor_bytes(torch: Any, tensor: Any) -> bytes:
    return bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())


def tensor_argument(runtime: Any, tensor: Any) -> Any:
    return runtime.NvidiaLaunchArgument.tensor(
        int(tensor.data_ptr()), list(tensor.shape), list(tensor.stride())
    )


def execute_case(
    torch: Any,
    runtime: Any,
    artifact: Any,
    request: Any,
    observation: Any,
    stream: Any,
    case: Any,
    repetition: int,
) -> dict[str, object]:
    scalar_value = 0.75
    padding_indices: list[int] = []
    input_reference_tensors: list[Any]
    input_device_tensors: list[Any]
    with torch.cuda.stream(stream):
        if case.name == "static":
            left_cpu = (torch.arange(64, dtype=torch.float32) - 17).reshape(8, 8) / 7
            right_cpu = (torch.arange(64, dtype=torch.float32) + 3).reshape(8, 8) / 11
            reference = left_cpu + right_cpu
            left = left_cpu.to("cuda")
            right = right_cpu.to("cuda")
            output = torch.empty((8, 8), dtype=torch.float32, device="cuda")
            output_storage = output
            input_reference_tensors = [left_cpu, right_cpu]
            input_device_tensors = [left, right]
            arguments = [
                tensor_argument(runtime, left),
                tensor_argument(runtime, right),
                tensor_argument(runtime, output),
            ]
        elif case.name == "dynamic":
            storage_count = case.shape[0] * case.strides[0]
            sentinel = -12345.5
            left_cpu_storage = torch.full(
                (storage_count,), sentinel, dtype=torch.float32
            )
            right_cpu_storage = torch.full(
                (storage_count,), sentinel, dtype=torch.float32
            )
            output_cpu_storage = torch.full(
                (storage_count,), sentinel, dtype=torch.float32
            )
            left_cpu = left_cpu_storage.as_strided(case.shape, case.strides)
            right_cpu = right_cpu_storage.as_strided(case.shape, case.strides)
            values = torch.arange(case.shape[0] * case.shape[1], dtype=torch.float32)
            left_cpu.copy_((values.reshape(case.shape) - 31) / 13)
            right_cpu.copy_((values.reshape(case.shape) + 7) / 17)
            reference = left_cpu + right_cpu
            left_storage = left_cpu_storage.to("cuda")
            right_storage = right_cpu_storage.to("cuda")
            output_storage = output_cpu_storage.to("cuda")
            left = left_storage.as_strided(case.shape, case.strides)
            right = right_storage.as_strided(case.shape, case.strides)
            output = output_storage.as_strided(case.shape, case.strides)
            input_reference_tensors = [left_cpu_storage, right_cpu_storage]
            input_device_tensors = [left_storage, right_storage]
            logical_indices = {
                row * case.strides[0] + column
                for row in range(case.shape[0])
                for column in range(case.shape[1])
            }
            padding_indices = [
                index for index in range(storage_count) if index not in logical_indices
            ]
            arguments = [
                tensor_argument(runtime, left),
                tensor_argument(runtime, right),
                tensor_argument(runtime, output),
            ]
        elif case.name == "scalar":
            input_cpu = (
                torch.linspace(-2.0, 2.0, 64, dtype=torch.float32)
                .to(torch.float16)
                .reshape(case.shape)
            )
            reference = (input_cpu.float() + scalar_value).to(torch.float16)
            left = input_cpu.to("cuda")
            output = torch.empty(case.shape, dtype=torch.float16, device="cuda")
            output_storage = output
            input_reference_tensors = [input_cpu]
            input_device_tensors = [left]
            if int(left.data_ptr()) % 16 or int(output.data_ptr()) % 16:
                raise SmokeError("scalar fixture tensors are not 16-byte aligned")
            arguments = [
                tensor_argument(runtime, left),
                runtime.NvidiaLaunchArgument.scalar(struct.pack("<f", scalar_value)),
                tensor_argument(runtime, output),
            ]
        else:
            raise SmokeError(f"unknown smoke case: {case.name}")

        executable = runtime.NvidiaExecutable(artifact, request)
        executable.prewarm(observation.cuda_runtime_api_version)
        if not executable.ready:
            raise SmokeError("NvidiaExecutable did not become ready")
        packet = executable.prepare_launch(arguments)
        if tuple(packet.grid_dimensions) != case.expected_grid:
            raise SmokeError(f"{case.name} runtime grid differs")
        if packet.kernel_argument_count != case.expected_kernel_arguments:
            raise SmokeError(f"{case.name} runtime argument count differs")
        raw_stream = int(torch._C._cuda_getCurrentRawStream(0))
        public_current = int(torch.cuda.current_stream(0).cuda_stream)
        selected_stream = int(stream.cuda_stream)
        default_stream = int(torch.cuda.default_stream(0).cuda_stream)
        if (
            raw_stream != public_current
            or raw_stream != selected_stream
            or raw_stream in (0, 1, 2)
            or raw_stream == default_stream
        ):
            raise SmokeError(
                "PyPTO launch stream is not the selected non-default stream"
            )
        executable.launch(packet, raw_stream)
    stream.synchronize()
    actual_storage = output_storage.cpu()
    input_actual_tensors = [tensor.cpu() for tensor in input_device_tensors]
    actual = (
        actual_storage.as_strided(case.shape, case.strides)
        if case.name == "dynamic"
        else actual_storage
    )
    if not torch.equal(actual, reference):
        difference = (actual.float() - reference.float()).abs()
        raise SmokeError(
            f"{case.name} numerical mismatch: max={float(difference.max())}"
        )
    input_reference_bytes = b"".join(
        logical_tensor_bytes(torch, tensor) for tensor in input_reference_tensors
    )
    input_actual_bytes = b"".join(
        logical_tensor_bytes(torch, tensor) for tensor in input_actual_tensors
    )
    if input_actual_bytes != input_reference_bytes:
        raise SmokeError(f"{case.name} kernel modified a read-only input")
    padding_unchanged = True
    if case.name == "dynamic":
        padding_unchanged = all(
            float(actual_storage[index]) == -12345.5 for index in padding_indices
        )
        if not padding_unchanged:
            raise SmokeError("dynamic kernel modified output padding")
    expected_sha256 = sha256_bytes(logical_tensor_bytes(torch, reference))
    actual_sha256 = sha256_bytes(logical_tensor_bytes(torch, actual))
    if expected_sha256 != actual_sha256:
        raise SmokeError(f"{case.name} logical output bytes differ")
    bound_context = executable.bound_context_address
    bound_context_id = executable.bound_context_id
    if (
        bound_context != observation.context_address
        or bound_context_id != observation.context_id
    ):
        raise SmokeError("NvidiaExecutable context differs from observation")
    del packet
    gc.collect()
    executable.unload()
    if (
        executable.ready
        or executable.state != runtime.NvidiaExecutableState.Unloaded
        or executable.bound_context_address != 0
        or executable.bound_context_id != 0
    ):
        raise SmokeError("NvidiaExecutable did not terminally unload")
    return {
        "case": case.name,
        "repetition": repetition,
        "artifact_identity_digest": artifact.identity_digest,
        "dtype": case.dtype,
        "shape": list(case.shape),
        "strides": list(case.strides),
        "grid": list(case.expected_grid),
        "kernel_argument_count": case.expected_kernel_arguments,
        "raw_current_stream": raw_stream,
        "non_default_stream": True,
        "external_stream_synchronized": True,
        "expected_logical_bytes_sha256": expected_sha256,
        "actual_logical_bytes_sha256": actual_sha256,
        "input_bytes_sha256": sha256_bytes(input_reference_bytes),
        "input_unchanged": True,
        "torch_equal": True,
        "padding_unchanged": padding_unchanged,
        "packet_released_after_synchronization": True,
        "explicit_unload": True,
        "terminal_state": "Unloaded",
        "bound_context_before_unload": bound_context,
        "bound_context_id_before_unload": bound_context_id,
        "bound_context_after_unload": executable.bound_context_address,
    }


def target_traits_document(traits: Any) -> dict[str, int]:
    names = (
        "compute_capability",
        "multiprocessor_count",
        "warp_size",
        "max_threads_per_block",
        "max_threads_per_multiprocessor",
        "max_blocks_per_multiprocessor",
        "max_block_dim_x",
        "max_block_dim_y",
        "max_block_dim_z",
        "max_grid_dim_x",
        "max_grid_dim_y",
        "max_grid_dim_z",
        "l1_cache_line_bytes",
        "default_shared_memory_per_cta_bytes",
        "max_shared_memory_per_cta_bytes",
        "shared_memory_per_multiprocessor_bytes",
        "registers_per_cta",
        "max_registers_per_thread",
        "registers_per_multiprocessor",
        "l2_cache_size_bytes",
        "total_global_memory_bytes",
    )
    return {name: int(getattr(traits, name)) for name in names}


def supported_dtype_names(pypto: Any, values: Any) -> list[str]:
    names: list[str] = []
    for value in values:
        if value == pypto.DataType.BF16:
            names.append("BF16")
        elif value == pypto.DataType.FP32:
            names.append("FP32")
        else:
            raise SmokeError(f"unexpected observed compute dtype: {value}")
    return names


def git_identity(repository: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return {"head": head, "tree": tree, "clean": not dirty}


def validate_pypto_python_source(workspace: Path) -> None:
    repository = workspace / "projects/pypto"
    ignored = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "python/pypto",
        ],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if ignored:
        raise SmokeError("PyPTO Python package contains ignored shadow files")


def run_smoke() -> tuple[dict[str, object], Path, str]:
    workspace, run_id = _workspace_from_environment()
    barrier_evidence = wait_for_start_barrier(workspace, run_id)
    contract, child_gate = load_contract_and_child_gate(
        workspace, run_id, barrier_evidence["gate"]
    )
    validate_pypto_python_source(workspace)
    site = workspace / "envs/pypto-nvidia/lib/python3.14/site-packages"
    if site.is_symlink() or not site.is_dir() or site.resolve(strict=True) != site:
        raise SmokeError("selected site-packages path is not canonical")
    sys.path.insert(0, str(site))
    import torch

    if (
        str(torch.__version__) != contract.EXPECTED_TORCH_VERSION
        or str(torch.version.git_version) != contract.EXPECTED_TORCH_GIT
        or torch.version.cuda != contract.EXPECTED_TORCH_CUDA
        or torch.version.hip is not None
    ):
        raise SmokeError("live Torch identity differs from the fixed smoke contract")
    forbidden_imports = {"triton", "sglang", "flashinfer"} & set(sys.modules)
    if forbidden_imports:
        raise SmokeError(f"forbidden framework imported: {sorted(forbidden_imports)}")
    torch.cuda.set_device(0)
    torch.cuda.init()
    if (
        torch.cuda.get_device_name(0) != contract.EXPECTED_DEVICE_NAME
        or tuple(torch.cuda.get_device_capability(0))
        != contract.EXPECTED_COMPUTE_CAPABILITY
    ):
        raise SmokeError("live Torch CUDA target differs from the fixed contract")
    runtime_paths = mapped_library_paths("libcudart.so")
    expected_runtime = str(
        (workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True)
    )
    if runtime_paths != [expected_runtime]:
        raise SmokeError(f"live libcudart provider set differs: {runtime_paths}")
    maps_lower = Path("/proc/self/maps").read_text(errors="replace").lower()
    for marker in ("libamdhip64", "libhsa-runtime64", "gemsim"):
        if marker in maps_lower:
            raise SmokeError(f"forbidden runtime mapping after Torch import: {marker}")

    product = (workspace / contract.PYPTO_DSO_RELATIVE_PATH).resolve(strict=True).parent
    pypto_module = bootstrap_exact_pypto(workspace, product)
    from pypto import compiler
    from pypto.runtime import nvidia as runtime

    info = compiler.get_nvidia_backend_build_info()
    if (
        not info.compiled
        or not info.compiler_factory_available
        or info.pypto_revision != contract.PYPTO_HEAD
        or info.tensor_ir_revision != contract.TENSOR_IR_HEAD
        or info.cuda_tile_revision != contract.CUDA_TILE_HEAD
        or info.llvm_revision != contract.LLVM_HEAD
    ):
        raise SmokeError("exact PyPTO DSO build identity differs")
    observation = runtime.observe_current_nvidia_runtime(
        contract.EXPECTED_DRIVER_RELEASE, expected_runtime
    )
    target = observation.target_info
    if (
        target.device_name != contract.EXPECTED_DEVICE_NAME
        or target.traits.compute_capability != 120
        or target.traits.multiprocessor_count != contract.EXPECTED_SM_COUNT
        or observation.cuda_runtime_library_path != expected_runtime
        or observation.cuda_driver_api_version
        < contract.MINIMUM_CUDA_DRIVER_API_VERSION
        or observation.cuda_runtime_api_version
        < contract.MINIMUM_CUDA_RUNTIME_API_VERSION
    ):
        raise SmokeError("PyPTO runtime observation differs from the fixed target")
    request = compiler.CompileRequest(target, toolchain_identity(compiler, info))
    contracts = make_case_contracts(compiler)
    case_by_name = {case.name: case for case in contract.CASE_SPECS}
    artifacts: dict[str, Any] = {}
    artifact_records: list[dict[str, object]] = []
    replay = contract.replay_directory(workspace, run_id)
    replay.mkdir(mode=0o700, parents=False, exist_ok=False)
    replay_files: list[dict[str, object]] = []

    def replay_file(name: str, payload: bytes) -> None:
        path = replay / name
        digest = publish_no_replace(path, payload)
        replay_files.append(
            {
                "path": path.relative_to(workspace).as_posix(),
                "bytes": len(payload),
                "sha256": digest,
            }
        )

    replay_file("compile-request.msgpack", request.serialize())
    for case in contract.CASE_SPECS:
        source, abi, tiles = contracts[case.name]
        spec = build_spec(
            compiler, request, source, abi, tiles, contract.PIPELINE_REVISION
        )
        artifact = compiler.Artifact.compile_strict(source, request, spec)
        if (
            artifact.fallback_used
            or artifact.kernel_abi.identity_digest != abi.identity_digest
            or artifact.actual_target.compute_capability != 120
            or len(artifact.device_code) != case.expected_device_code_bytes
            or artifact.device_code_sha256 != case.expected_device_code_sha256
        ):
            raise SmokeError(f"{case.name} Artifact violates the strict contract")
        artifacts[case.name] = artifact
        replay_file(f"{case.name}.build-spec.msgpack", spec.serialize())
        replay_file(f"{case.name}.artifact.msgpack", artifact.serialize())
        artifact_records.append(
            {
                "case": case.name,
                "source_sha256": sha256_bytes(source),
                "build_spec_identity_digest": spec.identity_digest,
                "artifact_identity_digest": artifact.identity_digest,
                "cache_key_digest": artifact.cache_key_digest,
                "loader_compatibility_digest": artifact.loader_compatibility_digest,
                "device_code_bytes": len(artifact.device_code),
                "device_code_sha256": artifact.device_code_sha256,
                "kernel_abi_identity_digest": artifact.kernel_abi.identity_digest,
                "entry_function_name": artifact.kernel_abi.entry_function_name,
                "fallback_used": False,
                "expected_grid": list(case.expected_grid),
                "expected_kernel_arguments": case.expected_kernel_arguments,
                "expected_device_code_bytes": case.expected_device_code_bytes,
                "expected_device_code_sha256": case.expected_device_code_sha256,
            }
        )

    stream = torch.cuda.Stream(device=0)
    executions: list[dict[str, object]] = []
    for case in contract.CASE_SPECS:
        for repetition in range(case.repetitions):
            executions.append(
                execute_case(
                    torch,
                    runtime,
                    artifacts[case.name],
                    request,
                    observation,
                    stream,
                    case_by_name[case.name],
                    repetition,
                )
            )
    torch.cuda.synchronize(0)
    if {"triton", "sglang", "flashinfer"} & set(sys.modules):
        raise SmokeError("forbidden provider imported during exact smoke")

    integrity_paths = {
        "contract": workspace / "tools/_pypto_nvidia_executable_sm120_contract.py",
        "runner": Path(__file__).resolve(strict=True),
        "environment_lock": workspace / "ENVIRONMENT.lock",
        "versions_lock": workspace / "VERSIONS.lock",
        "workspace_lock": workspace / "WORKSPACE.lock",
        "pypto_dso": workspace / contract.PYPTO_DSO_RELATIVE_PATH,
        "cuda_runtime": workspace / contract.CUDA_RUNTIME_RELATIVE_PATH,
    }
    integrity = {
        name: {
            "path": path.relative_to(workspace).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in integrity_paths.items()
    }
    pypto_identity = git_identity(workspace / "projects/pypto")
    if pypto_identity != {
        "head": contract.PYPTO_HEAD,
        "tree": contract.PYPTO_TREE,
        "clean": True,
    }:
        raise SmokeError("PyPTO source changed during smoke")
    preflight_path = Path(os.environ["PYPTO_PREFLIGHT_REPORT_PATH"])
    preflight_sha256 = os.environ["PYPTO_PREFLIGHT_REPORT_SHA256"]
    if sha256_file(preflight_path) != preflight_sha256:
        raise SmokeError("preflight sidecar changed during smoke")
    gate = barrier_evidence["barrier"]
    gate_document = barrier_evidence["gate"]
    provisional = {
        "schema_version": contract.SMOKE_SCHEMA_VERSION,
        "smoke": contract.SMOKE_NAME,
        "acceptance": "gpu-execution-complete-awaiting-run-finalization",
        "scope": {
            "provider": "pypto.tensorir",
            "runtime_object": "NvidiaExecutable",
            "operator_correctness": True,
            "model_forward": False,
            "strict_coverage_result": False,
            "performance_result": False,
            "cuda_graph_result": False,
        },
        "inputs": {
            "integrity": integrity,
            "pypto": pypto_identity,
            "tensor_ir_head": contract.TENSOR_IR_HEAD,
            "cuda_tile_head": contract.CUDA_TILE_HEAD,
            "llvm_head": contract.LLVM_HEAD,
            "replay_files": replay_files,
            "control_manifest": child_gate["control_manifest"],
        },
        "run_context": {
            "run_id": run_id,
            "mode": "gpu-smoke",
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "start_ticks": _process_start_ticks(os.getpid()),
            "preflight": {
                "path": preflight_path.relative_to(workspace).as_posix(),
                "sha256": preflight_sha256,
            },
            "gate": {
                "path": str(gate["gate_path"]),
                "sha256": str(gate["gate_sha256"]),
                "document": gate_document,
            },
            "start_barrier_sha256": sha256_file(
                Path(os.environ["PYPTO_GPU_SMOKE_START_BARRIER"])
            ),
            "protected_zero_nvidia_policy": (
                os.environ.get("PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED") == "1"
            ),
        },
        "runtime": {
            "torch": {
                "version": str(torch.__version__),
                "git_version": str(torch.version.git_version),
                "cuda": torch.version.cuda,
                "hip": torch.version.hip,
                "module_path": str(Path(torch.__file__).resolve(strict=True)),
            },
            "child_pre_cuda_gate": child_gate,
            "libcudart_paths": runtime_paths,
            "observation": {
                "device_ordinal": target.device_ordinal,
                "device_name": target.device_name,
                "device_uuid": target.device_uuid,
                "pci_device_id": target.pci_device_id,
                "traits": target_traits_document(target.traits),
                "cuda_toolkit_version": target.cuda_toolkit_version,
                "cuda_driver_version": target.cuda_driver_version,
                "tensor_ir_revision": target.tensor_ir_revision,
                "cuda_tile_revision": target.cuda_tile_revision,
                "supported_compute_dtypes": supported_dtype_names(
                    pypto_module, target.supported_compute_dtypes
                ),
                "cuda_driver_release_provenance": target.cuda_driver_version,
                "cuda_driver_api_version": observation.cuda_driver_api_version,
                "cuda_runtime_api_version": observation.cuda_runtime_api_version,
                "cuda_runtime_library_path": observation.cuda_runtime_library_path,
                "context_address": observation.context_address,
                "context_id": observation.context_id,
            },
            "compile_request": {
                "byte_identity_digest": request.byte_compile_identity_digest,
                "loader_compatibility_input_digest": (
                    request.loader_compatibility_input_digest
                ),
                "device_autotune_identity_digest": (
                    request.device_autotune_identity_digest
                ),
            },
            "artifacts": artifact_records,
            "executions": executions,
            "case_order": [case.name for case in contract.CASE_SPECS],
            "repetitions_per_case": 2,
            "module_lifetimes": len(executions),
            "explicit_unloads": len(executions),
            "non_default_current_stream": True,
            "external_synchronization": True,
            "fallback_used": False,
            "forbidden_provider_imports": [],
        },
    }
    provisional_path = contract.provisional_path(workspace, run_id)
    provisional_bytes = canonical_json(provisional)
    provisional_sha256 = publish_no_replace(provisional_path, provisional_bytes)
    return provisional, provisional_path, provisional_sha256


def main() -> int:
    document, path, digest = run_smoke()
    print(
        json.dumps(
            {
                "acceptance": document["acceptance"],
                "path": str(path),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
