#!/usr/bin/env python3
"""Correctness-only HIR-to-Artifact-to-SM120 vector-add smoke.

Only standard-library modules are imported before the parent-owned start
barrier is authenticated.  Torch and the exact PyPTO DSO are exposed only
after the child repeats the v4 admission audit.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any


RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")


class SmokeError(RuntimeError):
    """A fail-closed frontend correctness-smoke error."""


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
    workspace = Path(os.environ.get("PYPTO_WORKSPACE_ROOT", ""))
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
    if Path(os.environ.get("PYPTO_GPU_SMOKE_START_BARRIER", "")) != expected:
        raise SmokeError("GPU-smoke start-barrier path differs from the run identity")
    deadline = time.monotonic() + 60.0
    while not expected.exists():
        if time.monotonic() >= deadline:
            raise SmokeError("timed out before parent GPU-smoke gate release")
        time.sleep(0.05)
    barrier, _ = load_canonical_json(expected, "GPU-smoke start barrier")
    identity = {
        "schema": 1,
        "run_id": run_id,
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "start_ticks": _process_start_ticks(os.getpid()),
    }
    if any(barrier.get(name) != value for name, value in identity.items()):
        raise SmokeError("GPU-smoke start barrier does not identify this process")
    gate_path = Path(str(barrier.get("gate_path", "")))
    expected_gate = workspace / "runs" / run_id / "gpu-smoke-gate.json"
    if gate_path != expected_gate:
        raise SmokeError("GPU-smoke gate path differs from the run identity")
    gate, gate_raw = load_canonical_json(gate_path, "GPU-smoke gate")
    if sha256_bytes(gate_raw) != barrier.get("gate_sha256"):
        raise SmokeError("GPU-smoke gate digest join failed")
    if any(gate.get(name) != value for name, value in identity.items()):
        raise SmokeError("GPU-smoke gate does not identify this process")
    return {"barrier": barrier, "gate": gate}


def load_contract_and_child_gate(
    workspace: Path, parent_gate: dict[str, object]
) -> tuple[Any, dict[str, object]]:
    contract = _load_module(
        "_pypto_frontend_vector_add_sm120_contract",
        workspace / "tools/_pypto_frontend_vector_add_sm120_contract.py",
    )
    control = _load_module(
        "_pypto_frontend_sm120_control_manifest",
        workspace / "tools/_pypto_frontend_sm120_control_manifest.py",
    )
    control_identity = control.validate_control_manifest(workspace)
    if parent_gate.get("control_manifest") != control_identity:
        raise SmokeError("parent/child control-manifest identity differs")
    runner = Path(__file__).resolve(strict=True)
    if (
        runner != workspace / contract.RUNNER_RELATIVE_PATH
        or runner.stat().st_size != contract.RUNNER_SIZE
        or sha256_file(runner) != contract.RUNNER_SHA256
        or contract.fixed_child_command(workspace) != sys.orig_argv
    ):
        raise SmokeError("live runner identity differs from the fixed command")

    # preflight.py is an accepted v4 blob whose one contract import is used only
    # for static environment constants.  Supply this versioned contract without
    # editing that primitive.
    sys.modules["_pypto_nvidia_executable_sm120_contract"] = contract
    preflight = _load_module("preflight", workspace / "tools/preflight.py")
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
    package = workspace / "projects/pypto/python/pypto"
    cores = sorted(product.glob("pypto_core*.so"))
    if len(cores) != 1:
        raise SmokeError("exact product directory does not contain one PyPTO DSO")
    core = cores[0].resolve(strict=True)
    package_spec = importlib.util.spec_from_file_location(
        "pypto", package / "__init__.py", submodule_search_locations=[str(package)]
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
        if fields and marker in fields[-1] and fields[-1].startswith("/"):
            paths.add(str(Path(fields[-1]).resolve(strict=True)))
    return sorted(paths)


def toolchain_identity(compiler: Any, info: Any) -> Any:
    return compiler.ToolchainIdentity(
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


def schedule(compiler: Any, tile_sizes: tuple[int, ...]) -> Any:
    parameter = compiler.ScheduleParameter
    unsigned = compiler.ScheduleValueKind.UnsignedInteger
    return compiler.CanonicalSchedule(
        [
            parameter(
                "codegen_strategy",
                compiler.ScheduleValueKind.Text,
                "layout-propagation",
            )
        ],
        [
            parameter(f"dim_{index:03d}", unsigned, str(value))
            for index, value in enumerate(tile_sizes)
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
            parameter("uniform_signature", compiler.ScheduleValueKind.Boolean, "false"),
        ],
    )


def make_program(pypto: Any, ir: Any, case: Any) -> Any:
    span = ir.Span(f"pypto_frontend_vector_add_sm120:{case.name}", 1, 1)
    dtype = pypto.DataType.FP32 if case.dtype == "float32" else pypto.DataType.BF16
    tensor_type = ir.TensorType(list(case.shape), dtype)
    lhs = ir.Var("lhs", tensor_type, span)
    rhs = ir.Var("rhs", tensor_type, span)
    result = ir.Var("sum", tensor_type, span)
    call = ir.Call(ir.get_op("tensor.add"), [lhs, rhs], tensor_type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(result, call, span), ir.ReturnStmt([result], span)], span
    )
    function = ir.Function(
        "add_main",
        [(lhs, ir.ParamDirection.In), (rhs, ir.ParamDirection.In)],
        [tensor_type],
        body,
        span,
    )
    return ir.Program([function], f"frontend_smoke_{case.name}", span)


def require_sm120_artifact_target(artifact: Any) -> None:
    """Validate the real ArtifactTarget Python surface used by both replayers."""

    if artifact.actual_target.compute_capability != 120:
        raise SmokeError("structured Artifact actual target is not SM120")


def validate_compiled_artifact(
    compiler: Any, build_spec: Any, artifact: Any, request: Any, case: Any
) -> dict[str, object]:
    require_sm120_artifact_target(artifact)
    kernel = artifact.kernel_abi
    expected_descriptor_strides = list(case.strides) if len(case.shape) > 1 else []
    expected_explicit_strides = len(case.shape) > 1
    descriptors = list(kernel.argument_layout.operand_descriptors)
    if (
        artifact.fallback_used
        or build_spec.source_ir_digest != case.expected_source_ir_digest
        or build_spec.static_specialization_digest
        != case.expected_static_specialization_digest
        or build_spec.symbolic_specialization_digest
        != case.expected_symbolic_specialization_digest
        or build_spec.argument_abi_digest != case.expected_argument_abi_digest
        or build_spec.result_abi_digest != case.expected_result_abi_digest
        or build_spec.mutation_abi_digest != case.expected_mutation_abi_digest
        or build_spec.callable_abi_digest != case.expected_callable_abi_digest
        or build_spec.callable_abi_digest != kernel.identity_digest
        or artifact.identities.kernel_build_spec_digest != build_spec.identity_digest
        or artifact.identities.source_ir_digest != build_spec.source_ir_digest
        or artifact.identities.argument_abi_digest != build_spec.argument_abi_digest
        or artifact.identities.result_abi_digest != build_spec.result_abi_digest
        or artifact.identities.mutation_abi_digest != build_spec.mutation_abi_digest
        or kernel.runtime_kernel_name != "tensor_ir_rtk"
        or kernel.entry_function_name != "pypto_vector_add"
        or kernel.argument_packing_policy
        != compiler.ArtifactArgumentPackingPolicy.PointerOnly
        or kernel.grid_abi.policy != compiler.ArtifactGridPolicy.Static
        or tuple(kernel.grid_abi.static_dimensions) != case.expected_grid
        or tuple(kernel.grid_abi.tile_sizes) != case.tile_sizes
        or kernel.argument_layout.input_operand_count != 2
        or kernel.argument_layout.total_kernel_argument_count
        != case.expected_kernel_arguments
        or kernel.argument_layout.uniform_signature
        or len(descriptors) != 3
        or len(artifact.device_code) != case.expected_device_code_bytes
        or artifact.device_code_sha256 != case.expected_device_code_sha256
        or sha256_bytes(bytes(artifact.device_code)) != case.expected_device_code_sha256
    ):
        raise SmokeError(f"{case.name} structured Artifact contract differs")
    for descriptor in descriptors:
        if (
            descriptor.kind != compiler.ArtifactOperandKind.Tensor
            or list(descriptor.shape) != list(case.shape)
            or list(descriptor.strides) != expected_descriptor_strides
            or descriptor.dynamic_size_count != 0
            or descriptor.dynamic_stride_count != 0
            or descriptor.explicit_strides is not expected_explicit_strides
            or descriptor.scalar_size_bytes != 0
        ):
            raise SmokeError(f"{case.name} producer tensor descriptor differs")
    if (
        type(build_spec).deserialize(build_spec.serialize()).serialize()
        != build_spec.serialize()
        or compiler.Artifact.deserialize(
            artifact.serialize(), request, build_spec
        ).serialize()
        != artifact.serialize()
    ):
        raise SmokeError(f"{case.name} compiler wire round-trip differs")
    return {
        "case": case.name,
        "compile_api": "pypto.compiler.compile_structured_strict",
        "compiler_invocations": 1,
        "build_spec_identity_digest": build_spec.identity_digest,
        "source_ir_digest": build_spec.source_ir_digest,
        "callable_abi_digest": build_spec.callable_abi_digest,
        "static_specialization_digest": build_spec.static_specialization_digest,
        "symbolic_specialization_digest": build_spec.symbolic_specialization_digest,
        "argument_abi_digest": build_spec.argument_abi_digest,
        "result_abi_digest": build_spec.result_abi_digest,
        "mutation_abi_digest": build_spec.mutation_abi_digest,
        "artifact_identity_digest": artifact.identity_digest,
        "cache_key_digest": artifact.cache_key_digest,
        "loader_compatibility_digest": artifact.loader_compatibility_digest,
        "device_code_bytes": len(artifact.device_code),
        "device_code_sha256": artifact.device_code_sha256,
        "kernel_abi_identity_digest": kernel.identity_digest,
        "entry_function_name": kernel.entry_function_name,
        "fallback_used": False,
        "expected_grid": list(case.expected_grid),
        "expected_kernel_arguments": case.expected_kernel_arguments,
        "expected_device_code_bytes": case.expected_device_code_bytes,
        "expected_device_code_sha256": case.expected_device_code_sha256,
    }


def validate_structured_result(
    compiler: Any, result: Any, request: Any, case: Any
) -> dict[str, object]:
    if not isinstance(result, compiler.StructuredCompileResult):
        raise SmokeError(f"{case.name} did not return StructuredCompileResult")
    build_spec = result.build_spec
    artifact = result.artifact
    record = validate_compiled_artifact(compiler, build_spec, artifact, request, case)
    return {"build_spec": build_spec, "artifact": artifact, "record": record}


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
    capture_free_before = not bool(torch.cuda.is_current_stream_capturing())
    if not capture_free_before:
        raise SmokeError("frontend smoke cannot begin during CUDA Graph capture")
    with torch.cuda.stream(stream):
        count = 1
        for dimension in case.shape:
            count *= dimension
        base = torch.arange(count, dtype=torch.float32).reshape(case.shape)
        if case.dtype == "float32":
            left_cpu = (base - 17 + repetition) / 7
            right_cpu = (base + 3 - repetition) / 11
        else:
            left_cpu = (base - 64 + repetition).to(torch.bfloat16)
            right_cpu = ((base.remainder(7)) - 3 - repetition).to(torch.bfloat16)
        reference = (
            left_cpu + right_cpu
            if case.dtype == "float32"
            else (left_cpu.float() + right_cpu.float()).to(torch.bfloat16)
        )
        left = left_cpu.to("cuda")
        right = right_cpu.to("cuda")
        output = torch.empty(case.shape, dtype=left.dtype, device="cuda")
        arguments = [
            tensor_argument(runtime, left),
            tensor_argument(runtime, right),
            tensor_argument(runtime, output),
        ]
        executable = runtime.NvidiaExecutable(artifact, request)
        executable.prewarm(observation.cuda_runtime_api_version)
        if not executable.ready:
            raise SmokeError("NvidiaExecutable did not become ready")
        packet = executable.prepare_launch(arguments)
        if (
            tuple(packet.grid_dimensions) != case.expected_grid
            or packet.kernel_argument_count != case.expected_kernel_arguments
        ):
            raise SmokeError(f"{case.name} launch packet differs")
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
        capture_free_at_launch = not bool(torch.cuda.is_current_stream_capturing())
        if not capture_free_at_launch:
            raise SmokeError("frontend smoke cannot launch during CUDA Graph capture")
        executable.launch(packet, raw_stream)

    # Packet storage is retained until this explicit external synchronization.
    stream.synchronize()
    actual = output.cpu()
    left_after = left.cpu()
    right_after = right.cpu()
    if not torch.equal(actual, reference):
        difference = (actual.float() - reference.float()).abs()
        raise SmokeError(
            f"{case.name} numerical mismatch: max={float(difference.max())}"
        )
    expected_sha256 = sha256_bytes(logical_tensor_bytes(torch, reference))
    actual_sha256 = sha256_bytes(logical_tensor_bytes(torch, actual))
    input_before = logical_tensor_bytes(torch, left_cpu) + logical_tensor_bytes(
        torch, right_cpu
    )
    input_after = logical_tensor_bytes(torch, left_after) + logical_tensor_bytes(
        torch, right_after
    )
    if expected_sha256 != actual_sha256 or input_before != input_after:
        raise SmokeError(f"{case.name} byte correctness differs")
    bound_context = executable.bound_context_address
    bound_context_id = executable.bound_context_id
    if (
        bound_context != observation.context_address
        or bound_context_id != observation.context_id
    ):
        raise SmokeError("NvidiaExecutable context differs from observation")

    # NvidiaLaunchPacket has a move-only native lifetime and no public release
    # method.  Deleting it after synchronization is the explicit Python release;
    # unload succeeding immediately afterwards proves no packet lease remains.
    del packet
    gc.collect()
    executable.unload()
    terminal_state = executable.state
    bound_context_after = executable.bound_context_address
    bound_context_id_after = executable.bound_context_id
    if (
        executable.ready
        or terminal_state != runtime.NvidiaExecutableState.Unloaded
        or bound_context_after != 0
        or bound_context_id_after != 0
    ):
        raise SmokeError("NvidiaExecutable did not terminally unload")
    del executable
    gc.collect()
    return {
        "case": case.name,
        "repetition": repetition,
        "lifetime_ordinal": repetition,
        "fresh_executable": True,
        "artifact_identity_digest": artifact.identity_digest,
        "dtype": case.dtype,
        "shape": list(case.shape),
        "strides": list(case.strides),
        "grid": list(case.expected_grid),
        "kernel_argument_count": case.expected_kernel_arguments,
        "raw_current_stream": raw_stream,
        "non_default_stream": True,
        "capture_free_before": capture_free_before,
        "capture_free_at_launch": capture_free_at_launch,
        "external_stream_synchronized": True,
        "expected_logical_bytes_sha256": expected_sha256,
        "actual_logical_bytes_sha256": actual_sha256,
        "input_bytes_sha256": sha256_bytes(input_before),
        "input_unchanged": True,
        "torch_equal": True,
        "packet_released_after_synchronization": True,
        "explicit_unload": True,
        "terminal_state": "Unloaded",
        "bound_context_before_unload": bound_context,
        "bound_context_id_before_unload": bound_context_id,
        "bound_context_after_unload": bound_context_after,
        "bound_context_id_after_unload": bound_context_id_after,
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
        workspace, barrier_evidence["gate"]
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

    ir = pypto_module.ir

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
    if torch.cuda.is_current_stream_capturing():
        raise SmokeError("frontend compilation cannot begin during CUDA Graph capture")
    request = compiler.CompileRequest(target, toolchain_identity(compiler, info))
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
    artifacts: dict[str, Any] = {}
    artifact_records: list[dict[str, object]] = []
    hir_records: list[dict[str, object]] = []
    for case in contract.CASE_SPECS:
        original = make_program(pypto_module, ir, case)
        hir_bytes = bytes(pypto_module.ir.serialize(original))
        restored = pypto_module.ir.deserialize(hir_bytes)
        restored_bytes = bytes(pypto_module.ir.serialize(restored))
        if (
            len(hir_bytes) != case.expected_hir_bytes
            or sha256_bytes(hir_bytes) != case.expected_hir_sha256
            or not isinstance(restored, ir.Program)
            or not ir.structural_equal(original, restored, enable_auto_mapping=True)
            or restored_bytes != hir_bytes
        ):
            raise SmokeError(f"{case.name} HIR serialization round-trip differs")
        restored_function = restored.get_function("add_main")
        if restored_function is None or list(restored_function.param_directions) != [
            ir.ParamDirection.In,
            ir.ParamDirection.In,
        ]:
            raise SmokeError(f"{case.name} HIR input directions differ")
        replay_file(f"{case.name}.hir.msgpack", hir_bytes)

        # This is the one and only producer invocation for this case.
        result = compiler.compile_structured_strict(
            restored, request, schedule(compiler, case.tile_sizes)
        )
        validated = validate_structured_result(compiler, result, request, case)
        build_spec = validated["build_spec"]
        artifact = validated["artifact"]
        record = validated["record"]
        assert isinstance(record, dict)
        record["hir_sha256"] = sha256_bytes(hir_bytes)
        record["hir_bytes"] = len(hir_bytes)
        record["hir_roundtrip_exact"] = True
        artifacts[case.name] = artifact
        replay_file(f"{case.name}.build-spec.msgpack", build_spec.serialize())
        replay_file(f"{case.name}.artifact.msgpack", artifact.serialize())
        artifact_records.append(record)
        hir_records.append(
            {
                "case": case.name,
                "bytes": len(hir_bytes),
                "sha256": sha256_bytes(hir_bytes),
                "serialized_once": True,
                "deserialized_before_compile": True,
                "canonical_reserialization_equal": True,
                "structural_equal": True,
                "parameter_directions": ["In", "In"],
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
                    case,
                    repetition,
                )
            )
    torch.cuda.synchronize(0)
    if {"triton", "sglang", "flashinfer"} & set(sys.modules):
        raise SmokeError("forbidden provider imported during exact smoke")

    integrity_paths = {
        "contract": workspace / "tools/_pypto_frontend_vector_add_sm120_contract.py",
        "runner": Path(__file__).resolve(strict=True),
        "controller": workspace / "tools/run_pypto_frontend_sm120_isolated.py",
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
    barrier = barrier_evidence["barrier"]
    gate_document = barrier_evidence["gate"]
    provisional = {
        "schema_version": contract.SMOKE_SCHEMA_VERSION,
        "smoke": contract.SMOKE_NAME,
        "acceptance": "gpu-execution-complete-awaiting-run-finalization",
        "scope": {
            "provider": "pypto.tensorir",
            "frontend_hir": True,
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
                "path": str(barrier["gate_path"]),
                "sha256": str(barrier["gate_sha256"]),
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
            "hir_programs": hir_records,
            "artifacts": artifact_records,
            "executions": executions,
            "case_order": [case.name for case in contract.CASE_SPECS],
            "compile_invocations_per_case": 1,
            "repetitions_per_case": 2,
            "module_lifetimes": len(executions),
            "explicit_packet_releases": len(executions),
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
