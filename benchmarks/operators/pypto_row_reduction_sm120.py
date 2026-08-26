#!/usr/bin/env python3
"""Correctness-only RowReductionV3 Artifact-to-SM120 gate."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
BASE_RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_fused_pointwise_sm120.py")
BASE_RUNNER_SIZE = 66_999
BASE_RUNNER_SHA256 = "b7960cc894834b3ba05476943e774cfc8602891faa5b9137b3d97a6aac40ab15"
COMPARISON_MODE_EXACT = "exact-word"
COMPARISON_MODE_TOLERANCE = "sum-tolerance"
COMPARISON_MODE_SPECIAL = "special-classification-sign"


class SmokeError(RuntimeError):
    """The fixed row-reduction transaction is invalid."""


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
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def load_exact(
    name: str,
    path: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise SmokeError(f"exact row control is noncanonical: {path}")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if expected_size is not None and len(raw) != expected_size:
        raise SmokeError(f"exact row control size differs: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise SmokeError(f"exact row control hash differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = digest
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


base = load_exact(
    "_pypto_row_reduction_runner_base",
    ROOT / BASE_RUNNER_RELATIVE_PATH,
    BASE_RUNNER_SIZE,
    BASE_RUNNER_SHA256,
)


def workspace_from_environment() -> tuple[Path, str]:
    workspace = Path(os.environ.get("PYPTO_WORKSPACE_ROOT", ""))
    if (
        not workspace.is_absolute()
        or workspace.resolve(strict=True) != workspace
        or workspace != ROOT
    ):
        raise SmokeError("PYPTO_WORKSPACE_ROOT is not the canonical root")
    run_id = os.environ.get("PYPTO_RUN_ID", "")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise SmokeError("PYPTO_RUN_ID is malformed")
    if os.environ.get("PYPTO_RUN_MODE") != "gpu-smoke":
        raise SmokeError("row runner requires gpu-smoke mode")
    if (
        os.environ.get("PYPTO_ALLOW_FALLBACK") != "0"
        or os.environ.get("PYPTO_STRICT_COVERAGE") != "1"
    ):
        raise SmokeError("row runner requires strict coverage with no fallback")
    if os.environ.get("PYTHONPATH", ""):
        raise SmokeError("row runner requires an empty PYTHONPATH")
    if os.environ.get("SGLANG_PLUGINS") != "__pypto_exact_nvidia_smoke_no_plugins__":
        raise SmokeError("row runner requires the no-plugin policy")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise SmokeError("row runner requires Python -I -B -S")
    if {"torch", "pypto", "triton", "sglang", "flashinfer"} & set(sys.modules):
        raise SmokeError("GPU/framework modules loaded before row admission")
    return workspace, run_id


def wait_for_start_barrier(workspace: Path, run_id: str) -> dict[str, object]:
    path = workspace / "runs" / run_id / "gpu-smoke-start-barrier.json"
    if Path(os.environ.get("PYPTO_GPU_SMOKE_START_BARRIER", "")) != path:
        raise SmokeError("row start-barrier path differs")
    deadline = time.monotonic() + 60
    while not path.exists():
        if time.monotonic() >= deadline:
            raise SmokeError("timed out before row start-barrier release")
        time.sleep(0.05)
    barrier, _ = base.load_canonical_json(path, "row start barrier")
    identity = {
        "schema": 2,
        "run_id": run_id,
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "start_ticks": base._process_start_ticks(os.getpid()),
    }
    if any(barrier.get(name) != value for name, value in identity.items()):
        raise SmokeError("row start-barrier identity differs")
    gate_path = workspace / "runs" / run_id / "gpu-smoke-gate.json"
    gate, gate_raw = base.load_canonical_json(gate_path, "row GPU-smoke gate")
    if (
        Path(str(barrier.get("gate_path", ""))) != gate_path
        or sha256_bytes(gate_raw) != barrier.get("gate_sha256")
        or any(gate.get(name) != value for name, value in identity.items())
    ):
        raise SmokeError("row gate/barrier join differs")
    return {"barrier": barrier, "gate": gate}


def load_contract_and_child_gate(
    workspace: Path, parent_gate: dict[str, object]
) -> tuple[Any, dict[str, object]]:
    control = load_exact(
        "_pypto_row_reduction_sm120_control_manifest",
        workspace / "tools/_pypto_row_reduction_sm120_control_manifest.py",
    )
    control.reject_control_bytecode_cache(workspace)
    contract = load_exact(
        "_pypto_row_reduction_sm120_contract",
        workspace / "tools/_pypto_row_reduction_sm120_contract.py",
    )
    if (
        contract.COMPARISON_MODE_EXACT != COMPARISON_MODE_EXACT
        or contract.COMPARISON_MODE_TOLERANCE != COMPARISON_MODE_TOLERANCE
        or contract.COMPARISON_MODE_SPECIAL != COMPARISON_MODE_SPECIAL
    ):
        raise SmokeError("row comparison-mode contract differs")
    identity = control.validate_control_manifest(workspace)
    if parent_gate.get("control_manifest") != identity:
        raise SmokeError("parent/child row control identity differs")
    runner = Path(__file__).resolve(strict=True)
    if (
        runner != workspace / contract.RUNNER_RELATIVE_PATH
        or runner.stat().st_size != contract.RUNNER_SIZE
        or sha256_file(runner) != contract.RUNNER_SHA256
        or contract.fixed_child_command(workspace) != sys.orig_argv
    ):
        raise SmokeError("live row runner identity differs")
    preflight = load_exact(
        "preflight_gpu_smoke_v2_row_child",
        workspace / contract.PREFLIGHT_ADAPTER_RELATIVE_PATH,
    )
    static_identity = preflight.static_torch_identity()
    if static_identity.get("static_identity_error"):
        raise SmokeError(str(static_identity["static_identity_error"]))
    requested = os.environ.get("PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED") == "1"
    floor = (
        contract.PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB
        if requested
        else contract.EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB
    )
    available = preflight.mem_available_kib()
    if available < floor:
        raise SmokeError("row child host-memory floor failed")
    gpu = preflight.nvidia_identity()
    free_memory = int(gpu["memory_mib"]) - int(gpu["used_mib"])
    if (
        gpu.get("compute_capability") != "12.0"
        or free_memory < contract.GPU_FREE_MEMORY_FLOOR_MIB
    ):
        raise SmokeError("row child GPU identity or memory floor differs")
    compute_pids = preflight.nvidia_compute_pids()
    if compute_pids:
        raise SmokeError(f"row child found NVIDIA compute PIDs: {compute_pids}")
    _all, protected, _workspace = preflight.process_table()
    protected_runtime, unreadable = preflight.protected_nvidia_runtime_mappings(
        protected
    )
    if protected_runtime or unreadable:
        raise SmokeError("row child cannot prove protected NVIDIA isolation")
    protected_heavy = [
        item for item in protected if preflight.is_heavy_command(item.command)
    ]
    if protected_heavy and not requested:
        raise SmokeError("protected CPU lane exists without row authorization")
    if (
        requested
        and os.environ.get("PYPTO_GPU_SMOKE_AUTHORIZATION")
        != contract.GPU_SMOKE_AUTHORIZATION
    ):
        raise SmokeError("row GPU-smoke authorization differs")
    return contract, {
        "static_identity": static_identity,
        "gpu": gpu,
        "free_memory_mib": free_memory,
        "mem_available_kib": available,
        "host_memory_floor_kib": floor,
        "admission_policy": preflight.policy_document(),
        "protected_heavy_pids": [item.pid for item in protected_heavy],
        "protected_runtime_pids": protected_runtime,
        "unreadable_protected_maps": unreadable,
        "nvidia_compute_pids": sorted(compute_pids),
        "control_manifest": identity,
        "base_runner": {
            "path": BASE_RUNNER_RELATIVE_PATH.as_posix(),
            "bytes": BASE_RUNNER_SIZE,
            "sha256": BASE_RUNNER_SHA256,
        },
    }


def load_anchors(
    workspace: Path, contract: Any
) -> tuple[dict[str, object], dict[str, object]]:
    path = workspace / contract.COMPILE_ANCHORS_RELATIVE_PATH
    raw = path.read_bytes()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != contract.COMPILE_ANCHORS_SIZE
        or sha256_bytes(raw) != contract.COMPILE_ANCHORS_SHA256
    ):
        raise SmokeError("row compile anchors differ")
    value = json.loads(raw)
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "pypto-row-reduction-compile-anchors-v1"
        or value.get("matrix_policy", {}).get("case_count") != 10
        or value.get("matrix_policy", {}).get("executions") != 20
        or [item.get("case") for item in value.get("records", [])]
        != list(contract.CASE_ORDER)
    ):
        raise SmokeError("row compile-anchor schema differs")
    return value, {item["case"]: item for item in value["records"]}


def element_count(shape: tuple[int, ...]) -> int:
    value = 1
    for extent in shape:
        value *= extent
    return value


def input_values(case: Any, repetition: int) -> list[float]:
    values: list[float] = []
    tolerance_rows = set(case.tolerance_output_indices(repetition))
    for row in range(case.rows):
        if (
            repetition == 0
            and case.dtype == "bfloat16"
            and case.op_name == "tensor.row_sum"
            and row == 0
        ):
            values.extend([1.0, *([2.0**-8] * (case.contraction - 1))])
            continue
        if row in tolerance_rows:
            values.extend(
                (1.0 + ((row * 13 + column * 11) % 29)) / 7.0
                for column in range(case.contraction)
            )
            continue
        if repetition == 0 and case.op_name == "tensor.row_sum":
            values.extend(
                float(((row * 3 + column * 5) % 9) - 4) / 4
                for column in range(case.contraction)
            )
            continue
        if repetition == 0:
            row_values = [
                float(((row * 11 + column * 7) % 127) - 90)
                for column in range(case.contraction)
            ]
            row_values[-1] = float(32 + row)
            values.extend(row_values)
            continue
        mode = row % 5
        if case.op_name == "tensor.row_sum":
            row_values = [-0.0] * case.contraction
            if mode == 1:
                row_values[0] = math.inf
                row_values[1:] = [1.0] * (case.contraction - 1)
            elif mode == 2:
                row_values[0] = -math.inf
                row_values[1:] = [-1.0] * (case.contraction - 1)
            elif mode == 3 and case.contraction > 1:
                row_values[0], row_values[1] = math.inf, -math.inf
            elif mode == 4:
                row_values[0] = math.nan
            values.extend(row_values)
        else:
            if mode == 0:
                values.extend([-0.0, 0.0, *([-1.0] * (case.contraction - 2))])
            elif mode == 1:
                values.extend([-0.0] * case.contraction)
            elif mode == 2:
                values.extend([math.inf, *([1.0] * (case.contraction - 1))])
            elif mode == 3:
                values.extend([-math.inf] * case.contraction)
            else:
                values.extend([math.nan] * case.contraction)
    if len(values) != element_count(case.shape):
        raise SmokeError(f"{case.name} input cardinality differs")
    return values


def input_tensor(torch: Any, case: Any, repetition: int) -> Any:
    dtype = torch.float32 if case.dtype == "float32" else torch.bfloat16
    return (
        torch.tensor(input_values(case, repetition), dtype=dtype)
        .reshape(case.shape)
        .contiguous()
    )


def eager_reference(torch: Any, case: Any, value: Any) -> Any:
    if case.op_name == "tensor.row_sum":
        reduced = value.float().sum(dim=-1, keepdim=True)
        return reduced if case.dtype == "float32" else reduced.to(torch.bfloat16)
    reduced = value.float().amax(dim=-1, keepdim=True)
    return reduced if case.dtype == "float32" else reduced.to(torch.bfloat16)


def words(torch: Any, tensor: Any, dtype: str) -> list[int]:
    code = torch.int32 if dtype == "float32" else torch.int16
    mask = 0xFFFFFFFF if dtype == "float32" else 0xFFFF
    return [
        int(item) & mask for item in tensor.contiguous().view(code).reshape(-1).tolist()
    ]


def float32_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bfloat16_word(value: float) -> int:
    bits = float32_word(value)
    if bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF:
        return 0x7FC0
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def encode_word(value: float, dtype: str) -> int:
    return float32_word(value) if dtype == "float32" else bfloat16_word(value)


def word_value(word: int, dtype: str) -> float:
    bits = word if dtype == "float32" else word << 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def pack_words(values: list[int], dtype: str) -> bytes:
    code = "I" if dtype == "float32" else "H"
    return struct.pack(f"<{len(values)}{code}", *values)


def cpu_reference_words(case: Any, repetition: int) -> list[int]:
    inputs = [
        encode_word(value, case.dtype) for value in input_values(case, repetition)
    ]
    output: list[int] = []
    for row in range(case.rows):
        row_words = inputs[row * case.contraction : (row + 1) * case.contraction]
        row_values = [word_value(word, case.dtype) for word in row_words]
        if any(math.isnan(value) for value in row_values):
            output.append(encode_word(math.nan, case.dtype))
            continue
        if case.op_name == "tensor.row_sum":
            positive_inf = any(value == math.inf for value in row_values)
            negative_inf = any(value == -math.inf for value in row_values)
            if positive_inf and negative_inf:
                result = math.nan
            elif positive_inf:
                result = math.inf
            elif negative_inf:
                result = -math.inf
            elif all(value == 0.0 for value in row_values):
                result = 0.0
            else:
                result = math.fsum(row_values)
        else:
            result = max(row_values)
            if result == 0.0:
                result = (
                    0.0
                    if any(
                        value == 0.0 and math.copysign(1.0, value) > 0
                        for value in row_values
                    )
                    else -0.0
                )
        output.append(encode_word(result, case.dtype))
    return output


def compare_word_sequences(
    case: Any,
    actual_words: list[int],
    reference_words: list[int],
    *,
    comparison_modes: tuple[str, ...],
) -> dict[str, object]:
    if (
        len(actual_words) != case.rows
        or len(reference_words) != case.rows
        or len(comparison_modes) != case.rows
    ):
        raise SmokeError(f"{case.name} comparison cardinality differs")
    max_ulp = 0
    max_relative = 0.0
    max_absolute = 0.0
    for index, (actual_word, reference_word, mode) in enumerate(
        zip(actual_words, reference_words, comparison_modes, strict=True)
    ):
        actual_class = base._classification(actual_word, case.dtype)
        reference_class = base._classification(reference_word, case.dtype)
        if mode == COMPARISON_MODE_EXACT:
            if actual_word != reference_word:
                raise SmokeError(f"{case.name} exact word differs at {index}")
            continue
        if mode == COMPARISON_MODE_SPECIAL:
            if reference_class[0] not in {"nan", "inf", "zero"}:
                raise SmokeError(
                    f"{case.name} special policy reached a nonspecial row at {index}"
                )
            if reference_class[0] == "nan":
                if actual_class[0] != "nan":
                    raise SmokeError(
                        f"{case.name} NaN classification differs at {index}"
                    )
            elif actual_class != reference_class:
                raise SmokeError(
                    f"{case.name} special sign/classification differs at {index}"
                )
            continue
        if mode != COMPARISON_MODE_TOLERANCE:
            raise SmokeError(f"{case.name} comparison mode differs at {index}")
        if reference_class[0] not in {"finite", "subnormal"} or actual_class[0] not in {
            "finite",
            "subnormal",
        }:
            raise SmokeError(f"{case.name} tolerance class differs at {index}")
        distance = abs(
            base._ordered_word(actual_word, case.dtype)
            - base._ordered_word(reference_word, case.dtype)
        )
        actual_value = word_value(actual_word, case.dtype)
        reference_value = word_value(reference_word, case.dtype)
        absolute = abs(actual_value - reference_value)
        denominator = abs(reference_value)
        relative = (
            0.0
            if denominator == 0.0 and absolute == 0.0
            else (math.inf if denominator == 0.0 else absolute / denominator)
        )
        max_ulp = max(max_ulp, distance)
        max_relative = max(max_relative, relative)
        max_absolute = max(max_absolute, absolute)
        if distance > case.max_ulp or absolute > case.rtol * denominator:
            raise SmokeError(f"{case.name} sum tolerance differs at {index}")
    return {
        "observed_max_ulp": max_ulp,
        "observed_max_relative_error": max_relative,
        "observed_max_absolute_error": max_absolute,
    }


def compare_output(
    torch: Any, case: Any, repetition: int, actual: Any, reference: Any
) -> tuple[dict[str, object], bytes]:
    actual_words = words(torch, actual, case.dtype)
    reference_words = words(torch, reference, case.dtype)
    cpu_words = cpu_reference_words(case, repetition)
    comparison_modes = case.output_comparison_modes(repetition)
    exact_indices = case.exact_output_indices(repetition)
    tolerance_indices = case.tolerance_output_indices(repetition)
    special_indices = case.special_output_indices(repetition)
    if (
        set(exact_indices) & set(tolerance_indices)
        or set(exact_indices) & set(special_indices)
        or set(tolerance_indices) & set(special_indices)
        or set(exact_indices) | set(tolerance_indices) | set(special_indices)
        != set(range(case.rows))
    ):
        raise SmokeError(f"{case.name} comparison partition differs")
    discriminator = (
        repetition == 0
        and case.dtype == "bfloat16"
        and case.op_name == "tensor.row_sum"
    )
    expected_discriminator_word: int | None = None
    sequential_bf16_word: int | None = None
    if discriminator:
        expected = 0x3FC0 if case.contraction == 129 else 0x4000
        accumulator = bfloat16_word(1.0)
        increment = word_value(bfloat16_word(2.0**-8), "bfloat16")
        for _ in range(case.contraction - 1):
            accumulator = bfloat16_word(word_value(accumulator, "bfloat16") + increment)
        if (
            actual_words[0] != expected
            or reference_words[0] != expected
            or cpu_words[0] != expected
            or accumulator != 0x3F80
            or accumulator == expected
        ):
            raise SmokeError(f"{case.name} FP32 accumulation discriminator differs")
        expected_discriminator_word = expected
        sequential_bf16_word = accumulator
    return (
        {
            "policy": case.comparison,
            "repetition0_policy": case.repetition0_policy,
            "comparison_modes": list(comparison_modes),
            "exact_output_indices": list(exact_indices),
            "tolerance_output_indices": list(tolerance_indices),
            "special_output_indices": list(special_indices),
            "max_ulp_limit": case.max_ulp,
            "rtol": case.rtol,
            "atol": 0.0,
            "candidate_vs_torch": compare_word_sequences(
                case,
                actual_words,
                reference_words,
                comparison_modes=comparison_modes,
            ),
            "candidate_vs_cpu": compare_word_sequences(
                case, actual_words, cpu_words, comparison_modes=comparison_modes
            ),
            "torch_vs_cpu": compare_word_sequences(
                case, reference_words, cpu_words, comparison_modes=comparison_modes
            ),
            "special_classification_and_sign_passed": True,
            "bf16_fp32_accumulation_discriminator_passed": discriminator,
            "bf16_expected_output_word": expected_discriminator_word,
            "bf16_sequential_accumulator_word": sequential_bf16_word,
        },
        pack_words(cpu_words, case.dtype),
    )


def guarded_cuda_tensor(
    torch: Any,
    logical: Any,
    *,
    prefix_value: float,
    suffix_value: float,
    guard_elements: int,
) -> tuple[Any, Any, bytes, bytes]:
    count = logical.numel()
    storage = torch.empty(
        count + 2 * guard_elements, dtype=logical.dtype, device="cuda"
    )
    storage[:guard_elements].fill_(prefix_value)
    storage[-guard_elements:].fill_(suffix_value)
    view = storage[guard_elements : guard_elements + count].view(logical.shape)
    view.copy_(logical, non_blocking=False)
    prefix = base.logical_tensor_bytes(torch, storage[:guard_elements].cpu())
    suffix = base.logical_tensor_bytes(torch, storage[-guard_elements:].cpu())
    return storage, view, prefix, suffix


def validate_compiled(
    compiler: Any,
    result: Any,
    request: Any,
    case: Any,
    anchor: dict[str, object],
) -> tuple[Any, dict[str, object]]:
    source = contract_module.canonical_tensor_ir_source(case)
    build = result.build_spec
    artifact = result.artifact
    build_raw = build.serialize()
    artifact_raw = artifact.serialize()
    device_code = bytes(artifact.device_code)
    kernel = artifact.kernel_abi
    descriptors = list(kernel.argument_layout.operand_descriptors)
    semantic_abi = contract_module.artifact_semantic_abi(
        compiler, request, build, artifact, case
    )
    if (
        sha256_bytes(source) != anchor["source_sha256"]
        or sha256_bytes(build_raw) != anchor["build_spec_sha256"]
        or sha256_bytes(artifact_raw) != anchor["artifact_sha256"]
        or sha256_bytes(device_code) != anchor["device_code_sha256"]
        or artifact.fallback_used
        or artifact.actual_target.compute_capability != 120
        or build.source_ir_digest != anchor["source_ir_digest"]
        or artifact.identities.kernel_build_spec_digest
        != anchor["kernel_build_spec_digest"]
        or kernel.entry_function_name != "pypto_row_reduction_v3"
        or tuple(kernel.grid_abi.static_dimensions) != case.grid
        or list(kernel.grid_abi.tile_sizes) != [case.row_tile]
        or kernel.argument_layout.input_operand_count != 1
        or kernel.argument_layout.total_kernel_argument_count != 2
        or len(descriptors) != 2
        or list(descriptors[0].shape) != list(case.shape)
        or list(descriptors[1].shape) != list(case.result_shape)
        or kernel.workspace_abi.size_bytes != 0
        or type(build).deserialize(build_raw).serialize() != build_raw
        or compiler.Artifact.deserialize(artifact_raw, request, build).serialize()
        != artifact_raw
        or semantic_abi != anchor["semantic_abi"]
        or anchor["comparison_modes"]
        != [
            list(case.output_comparison_modes(repetition))
            for repetition in range(contract_module.REPETITIONS)
        ]
    ):
        raise SmokeError(f"{case.name} compiled anchor differs")
    return artifact, {
        "case": case.name,
        "hir_bytes": anchor["hir_bytes"],
        "hir_sha256": anchor["hir_sha256"],
        "source_ir_bytes": len(source),
        "source_ir_sha256": sha256_bytes(source),
        "source_ir_digest": build.source_ir_digest,
        "build_spec_bytes": len(build_raw),
        "build_spec_identity_digest": build.identity_digest,
        "artifact_bytes": len(artifact_raw),
        "artifact_identity_digest": artifact.identity_digest,
        "device_code_bytes": len(device_code),
        "device_code_sha256": artifact.device_code_sha256,
        "grid": list(case.grid),
        "row_tile": case.row_tile,
        "semantic_abi": semantic_abi,
        "fallback_used": False,
    }


def execute_case(
    torch: Any,
    runtime: Any,
    artifact: Any,
    request: Any,
    observation: Any,
    candidate_stream: Any,
    reference_stream: Any,
    case: Any,
    repetition: int,
    numerical_oracle: dict[str, object],
    replay_file: Any,
) -> dict[str, object]:
    if torch.cuda.is_current_stream_capturing():
        raise SmokeError("row smoke cannot begin during CUDA Graph capture")
    logical = input_tensor(torch, case, repetition)
    input_raw = base.logical_tensor_bytes(torch, logical)
    oracle_keys = {
        "repetition",
        "input_elements",
        "input_word_bytes",
        "input_word_sha256",
        "cpu_reference_words",
        "cpu_reference_word_bytes",
        "cpu_reference_word_sha256",
        "cpu_reference_class_sign",
        "comparison_modes",
        "element_width_bytes",
    }
    if (
        set(numerical_oracle) != oracle_keys
        or numerical_oracle.get("repetition") != repetition
        or numerical_oracle.get("input_elements") != logical.numel()
        or numerical_oracle.get("input_word_bytes") != len(input_raw)
        or numerical_oracle.get("input_word_sha256") != sha256_bytes(input_raw)
        or numerical_oracle.get("comparison_modes")
        != list(case.output_comparison_modes(repetition))
    ):
        raise SmokeError(f"{case.name} frozen input oracle differs")
    replay_file(f"{case.name}.r{repetition}.input.bin", input_raw)
    with torch.cuda.stream(reference_stream):
        reference_input = logical.to("cuda")
        reference_cuda = eager_reference(torch, case, reference_input)
    reference_stream.synchronize()
    reference = reference_cuda.cpu()
    with torch.cuda.stream(candidate_stream):
        input_storage, input_view, input_prefix, input_suffix = guarded_cuda_tensor(
            torch,
            logical,
            prefix_value=contract_module.INPUT_GUARD_PREFIX,
            suffix_value=contract_module.INPUT_GUARD_SUFFIX,
            guard_elements=contract_module.INPUT_GUARD_ELEMENTS,
        )
        output_template = torch.full(case.result_shape, 19.0, dtype=logical.dtype)
        output_storage, output, output_prefix, output_suffix = guarded_cuda_tensor(
            torch,
            output_template,
            prefix_value=contract_module.OUTPUT_GUARD_PREFIX,
            suffix_value=contract_module.OUTPUT_GUARD_SUFFIX,
            guard_elements=contract_module.OUTPUT_GUARD_ELEMENTS,
        )
        arguments = [
            base.tensor_argument(runtime, input_view),
            base.tensor_argument(runtime, output),
        ]
        executable = runtime.NvidiaExecutable(artifact, request)
        executable.prewarm(observation.cuda_runtime_api_version)
        packet = executable.prepare_launch(arguments)
        raw_stream = int(torch._C._cuda_getCurrentRawStream(0))
        if (
            not executable.ready
            or tuple(packet.grid_dimensions) != case.grid
            or packet.kernel_argument_count != 2
            or raw_stream != int(candidate_stream.cuda_stream)
            or raw_stream == int(torch.cuda.default_stream(0).cuda_stream)
            or raw_stream == int(reference_stream.cuda_stream)
            or torch.cuda.is_current_stream_capturing()
        ):
            raise SmokeError(f"{case.name} launch stream/packet differs")
        executable.launch(packet, raw_stream)
    candidate_stream.synchronize()
    actual = output.cpu()
    comparison, cpu_reference_raw = compare_output(
        torch, case, repetition, actual, reference
    )
    width, code = (4, "I") if case.dtype == "float32" else (2, "H")
    cpu_words = list(
        struct.unpack(f"<{len(cpu_reference_raw) // width}{code}", cpu_reference_raw)
    )
    cpu_class_sign = [
        {"class": classification, "sign": sign}
        for classification, sign in (
            base._classification(word, case.dtype) for word in cpu_words
        )
    ]
    if (
        numerical_oracle.get("element_width_bytes") != width
        or numerical_oracle.get("cpu_reference_words") != cpu_words
        or numerical_oracle.get("cpu_reference_word_bytes")
        != len(cpu_reference_raw)
        or numerical_oracle.get("cpu_reference_word_sha256")
        != sha256_bytes(cpu_reference_raw)
        or numerical_oracle.get("cpu_reference_class_sign") != cpu_class_sign
    ):
        raise SmokeError(f"{case.name} frozen CPU oracle differs")
    reference_raw = base.logical_tensor_bytes(torch, reference)
    actual_raw = base.logical_tensor_bytes(torch, actual)
    replay_file(f"{case.name}.r{repetition}.reference.bin", reference_raw)
    replay_file(f"{case.name}.r{repetition}.actual.bin", actual_raw)
    replay_file(f"{case.name}.r{repetition}.cpu-reference.bin", cpu_reference_raw)
    input_storage_after = input_storage.cpu()
    input_after = base.logical_tensor_bytes(torch, input_view.cpu())
    input_prefix_after = base.logical_tensor_bytes(
        torch, input_storage_after[: contract_module.INPUT_GUARD_ELEMENTS]
    )
    input_suffix_after = base.logical_tensor_bytes(
        torch, input_storage_after[-contract_module.INPUT_GUARD_ELEMENTS :]
    )
    output_storage_after = output_storage.cpu()
    output_prefix_after = base.logical_tensor_bytes(
        torch, output_storage_after[: contract_module.OUTPUT_GUARD_ELEMENTS]
    )
    output_suffix_after = base.logical_tensor_bytes(
        torch, output_storage_after[-contract_module.OUTPUT_GUARD_ELEMENTS :]
    )
    if (
        input_after != input_raw
        or input_prefix_after != input_prefix
        or input_suffix_after != input_suffix
        or output_prefix_after != output_prefix
        or output_suffix_after != output_suffix
    ):
        raise SmokeError(f"{case.name} input/canary mutation")
    bound_context = executable.bound_context_address
    bound_context_id = executable.bound_context_id
    if (
        bound_context != observation.context_address
        or bound_context_id != observation.context_id
    ):
        raise SmokeError("row executable context differs")
    del packet
    gc.collect()
    executable.unload()
    if (
        executable.ready
        or executable.state != runtime.NvidiaExecutableState.Unloaded
        or executable.bound_context_address != 0
        or executable.bound_context_id != 0
    ):
        raise SmokeError("row executable did not terminally unload")
    return {
        "case": case.name,
        "repetition": repetition,
        "fresh_executable": True,
        "artifact_identity_digest": artifact.identity_digest,
        "expected_logical_bytes_sha256": sha256_bytes(reference_raw),
        "actual_logical_bytes_sha256": sha256_bytes(actual_raw),
        "cpu_reference_bytes_sha256": sha256_bytes(cpu_reference_raw),
        "input_before_sha256": sha256_bytes(input_raw),
        "input_after_sha256": sha256_bytes(input_after),
        "input_unchanged": True,
        "input_guard_elements": contract_module.INPUT_GUARD_ELEMENTS,
        "output_guard_elements": contract_module.OUTPUT_GUARD_ELEMENTS,
        "input_prefix_before_sha256": sha256_bytes(input_prefix),
        "input_prefix_after_sha256": sha256_bytes(input_prefix_after),
        "input_suffix_before_sha256": sha256_bytes(input_suffix),
        "input_suffix_after_sha256": sha256_bytes(input_suffix_after),
        "output_prefix_before_sha256": sha256_bytes(output_prefix),
        "output_prefix_after_sha256": sha256_bytes(output_prefix_after),
        "output_suffix_before_sha256": sha256_bytes(output_suffix),
        "output_suffix_after_sha256": sha256_bytes(output_suffix_after),
        "guards_unchanged": True,
        "comparison": comparison,
        "frozen_numerical_oracle_passed": True,
        "comparison_passed": True,
        "non_default_stream": True,
        "current_stream_launch": True,
        "raw_current_stream": raw_stream,
        "raw_reference_stream": int(reference_stream.cuda_stream),
        "distinct_nondefault_reference_stream": True,
        "reference_stream_synchronized_before_candidate": True,
        "reference_stream_policy": contract_module.REFERENCE_STREAM_POLICY,
        "candidate_stream_policy": contract_module.CANDIDATE_STREAM_POLICY,
        "reference_compute_boundary": contract_module.REFERENCE_COMPUTE_BOUNDARY,
        "capture_free_before": True,
        "capture_free_at_launch": True,
        "external_stream_synchronized": True,
        "packet_released_after_synchronization": True,
        "explicit_unload": True,
        "terminal_state": "Unloaded",
        "bound_context_before_unload": bound_context,
        "bound_context_id_before_unload": bound_context_id,
        "bound_context_after_unload": 0,
        "bound_context_id_after_unload": 0,
    }


def integrity_record(path: Path, workspace: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(workspace).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def reject_preexisting_pypto_modules() -> None:
    preexisting = sorted(
        name for name in sys.modules if name == "pypto" or name.startswith("pypto.")
    )
    if preexisting:
        raise SmokeError(
            "PyPTO modules existed before exact bootstrap: " + ", ".join(preexisting)
        )


def validate_exact_module_origins(
    workspace: Path,
    dso: Path,
    pypto: ModuleType,
    compiler: ModuleType,
    runtime: ModuleType,
    torch: ModuleType,
) -> None:
    package = workspace / "projects/pypto/python/pypto"
    expected = {
        "pypto": package / "__init__.py",
        "pypto.pypto_core": dso,
        "pypto.compiler": package / "compiler/__init__.py",
        "pypto.runtime.nvidia": package / "runtime/nvidia.py",
        "torch": workspace
        / "envs/pypto-nvidia/lib/python3.14/site-packages/torch/__init__.py",
    }
    modules = {
        "pypto": pypto,
        "pypto.pypto_core": sys.modules.get("pypto.pypto_core"),
        "pypto.compiler": compiler,
        "pypto.runtime.nvidia": runtime,
        "torch": torch,
    }
    for name, module in modules.items():
        path = expected[name]
        if (
            not isinstance(module, ModuleType)
            or sys.modules.get(name) is not module
            or getattr(module, "__file__", None) != str(path)
            or path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise SmokeError(f"exact module origin differs: {name}")
    if list(getattr(pypto, "__path__", ())) != [str(package)]:
        raise SmokeError("exact PyPTO package search path differs")


def run_smoke() -> tuple[dict[str, object], Path, str]:
    workspace, run_id = workspace_from_environment()
    barrier_evidence = wait_for_start_barrier(workspace, run_id)
    global contract_module
    contract_module, child_gate = load_contract_and_child_gate(
        workspace, barrier_evidence["gate"]
    )
    anchors, anchor_records = load_anchors(workspace, contract_module)
    base.validate_pypto_python_source(workspace)
    site = workspace / "envs/pypto-nvidia/lib/python3.14/site-packages"
    if site.is_symlink() or not site.is_dir() or site.resolve(strict=True) != site:
        raise SmokeError("selected row site-packages path is noncanonical")
    sys.path.insert(0, str(site))
    reject_preexisting_pypto_modules()
    import torch

    if (
        str(torch.__version__) != contract_module.EXPECTED_TORCH_VERSION
        or str(torch.version.git_version) != contract_module.EXPECTED_TORCH_GIT
        or torch.version.cuda != contract_module.EXPECTED_TORCH_CUDA
        or torch.version.hip is not None
    ):
        raise SmokeError("row Torch identity differs")
    torch.cuda.set_device(0)
    torch.cuda.init()
    if (
        torch.cuda.get_device_name(0) != contract_module.EXPECTED_DEVICE_NAME
        or tuple(torch.cuda.get_device_capability(0))
        != contract_module.EXPECTED_COMPUTE_CAPABILITY
    ):
        raise SmokeError("row live Torch CUDA target differs")
    forbidden_imports = {"triton", "sglang", "flashinfer"} & set(sys.modules)
    if forbidden_imports:
        raise SmokeError(f"forbidden row provider imported: {sorted(forbidden_imports)}")
    maps_lower = Path("/proc/self/maps").read_text(errors="replace").lower()
    for marker in ("libamdhip64", "libhsa-runtime64", "gemsim"):
        if marker in maps_lower:
            raise SmokeError(f"forbidden row runtime mapping: {marker}")
    expected_runtime = str(
        (workspace / contract_module.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True)
    )
    if base.mapped_library_paths("libcudart.so") != [expected_runtime]:
        raise SmokeError("row libcudart provider differs")
    reject_preexisting_pypto_modules()
    dso = workspace / contract_module.PYPTO_DSO_RELATIVE_PATH
    if (
        dso.is_symlink()
        or not dso.is_file()
        or dso.resolve(strict=True) != dso
        or dso.stat().st_size != contract_module.PYPTO_DSO_SIZE
        or sha256_file(dso) != contract_module.PYPTO_DSO_SHA256
    ):
        raise SmokeError("row PyPTO DSO differs before exact bootstrap")
    pypto = base.bootstrap_exact_pypto(workspace, dso.parent)
    from pypto import compiler
    from pypto.runtime import nvidia as runtime

    validate_exact_module_origins(workspace, dso, pypto, compiler, runtime, torch)
    info = compiler.get_nvidia_backend_build_info()
    if (
        not info.compiled
        or not info.compiler_factory_available
        or info.pypto_revision != contract_module.PYPTO_HEAD
        or info.tensor_ir_revision != contract_module.TENSOR_IR_HEAD
        or info.cuda_tile_revision != contract_module.CUDA_TILE_HEAD
        or info.llvm_revision != contract_module.LLVM_HEAD
    ):
        raise SmokeError("row compiler identity differs")
    observation = runtime.observe_current_nvidia_runtime(
        contract_module.EXPECTED_DRIVER_RELEASE, expected_runtime
    )
    target = observation.target_info
    if (
        target.device_name != contract_module.EXPECTED_DEVICE_NAME
        or target.traits.compute_capability != 120
        or target.traits.multiprocessor_count != contract_module.EXPECTED_SM_COUNT
    ):
        raise SmokeError("row runtime target differs")
    request = compiler.CompileRequest(target, base.toolchain_identity(compiler, info))
    request_raw = request.serialize()
    if sha256_bytes(request_raw) != anchors["anchor_request"]["derived_sha256"]:
        raise SmokeError("row runtime CompileRequest differs from anchors")
    replay = contract_module.replay_directory(workspace, run_id)
    replay.mkdir(mode=0o700, parents=False, exist_ok=False)
    replay_files: list[dict[str, object]] = []

    def replay_file(name: str, payload: bytes) -> None:
        path = replay / name
        digest = base.publish_no_replace(path, payload)
        replay_files.append(
            {
                "path": path.relative_to(workspace).as_posix(),
                "bytes": len(payload),
                "sha256": digest,
            }
        )

    replay_file("compile-request.msgpack", request_raw)
    artifacts: dict[str, Any] = {}
    artifact_records: list[dict[str, object]] = []
    hir_records: list[dict[str, object]] = []
    for case in contract_module.CASE_SPECS:
        program = contract_module.make_program(pypto, pypto.ir, case)
        hir = bytes(pypto.ir.serialize(program))
        restored = pypto.ir.deserialize(hir)
        anchor = anchor_records[case.name]
        if (
            bytes(pypto.ir.serialize(restored)) != hir
            or len(hir) != anchor["hir_bytes"]
            or sha256_bytes(hir) != anchor["hir_sha256"]
        ):
            raise SmokeError(f"{case.name} HIR anchor differs")
        source = contract_module.canonical_tensor_ir_source(case)
        result = compiler.compile_structured_strict(
            restored, request, contract_module.schedule(compiler, case.row_tile)
        )
        artifact, record = validate_compiled(compiler, result, request, case, anchor)
        artifacts[case.name] = artifact
        replay_file(f"{case.name}.hir.msgpack", hir)
        replay_file(f"{case.name}.source.mlir", source)
        replay_file(f"{case.name}.build-spec.msgpack", result.build_spec.serialize())
        replay_file(f"{case.name}.artifact.msgpack", artifact.serialize())
        replay_file(f"{case.name}.cubin", bytes(artifact.device_code))
        artifact_records.append(record)
        hir_records.append(
            {
                "case": case.name,
                "bytes": len(hir),
                "sha256": sha256_bytes(hir),
                "canonical_roundtrip": True,
                "input_count": 1,
                "operator": case.op_name,
            }
        )
    candidate_stream = torch.cuda.Stream(device=0)
    reference_stream = torch.cuda.Stream(device=0)
    if int(candidate_stream.cuda_stream) == int(reference_stream.cuda_stream):
        raise SmokeError("row candidate/reference streams are not distinct")
    executions: list[dict[str, object]] = []
    for case in contract_module.CASE_SPECS:
        for repetition in range(contract_module.REPETITIONS):
            executions.append(
                execute_case(
                    torch,
                    runtime,
                    artifacts[case.name],
                    request,
                    observation,
                    candidate_stream,
                    reference_stream,
                    case,
                    repetition,
                    anchor_records[case.name]["numerical_oracles"][repetition],
                    replay_file,
                )
            )
    if {"triton", "sglang", "flashinfer"} & set(sys.modules):
        raise SmokeError("forbidden provider imported during row smoke")
    integrity_paths = {
        "runner": Path(__file__).resolve(strict=True),
        "contract": workspace / "tools/_pypto_row_reduction_sm120_contract.py",
        "anchor_generator": workspace / contract_module.ANCHOR_GENERATOR_RELATIVE_PATH,
        "compile_anchors": workspace / contract_module.COMPILE_ANCHORS_RELATIVE_PATH,
        "controller": workspace / contract_module.CONTROLLER_RELATIVE_PATH,
        "control_validator": workspace
        / contract_module.CONTROL_VALIDATOR_RELATIVE_PATH,
        "preflight": workspace / contract_module.PREFLIGHT_ADAPTER_RELATIVE_PATH,
        "python": workspace / contract_module.PYTHON_REAL_RELATIVE_PATH,
        "cuda_runtime": workspace / contract_module.CUDA_RUNTIME_RELATIVE_PATH,
        "pypto_dso": workspace / contract_module.PYPTO_DSO_RELATIVE_PATH,
        "environment_lock": workspace / "ENVIRONMENT.lock",
        "versions_lock": workspace / "VERSIONS.lock",
        "workspace_lock": workspace / "WORKSPACE.lock",
    }
    initial_path = Path(os.environ["PYPTO_INITIAL_PREFLIGHT_REPORT_PATH"])
    preflight_path = Path(os.environ["PYPTO_PREFLIGHT_REPORT_PATH"])
    initial_sha = os.environ["PYPTO_INITIAL_PREFLIGHT_REPORT_SHA256"]
    preflight_sha = os.environ["PYPTO_PREFLIGHT_REPORT_SHA256"]
    if (
        sha256_file(initial_path) != initial_sha
        or sha256_file(preflight_path) != preflight_sha
    ):
        raise SmokeError("row preflight sidecar changed")
    barrier = barrier_evidence["barrier"]
    gate = barrier_evidence["gate"]
    provisional = {
        "schema_version": 1,
        "smoke": contract_module.SMOKE_NAME,
        "acceptance": "gpu-execution-complete-awaiting-run-finalization",
        "scope": {
            "frontend_family": "RowReductionV3",
            "fixed_case_count": 10,
            "fixed_case_correctness": True,
            "general_reduction_correctness": False,
            "performance_result": False,
            "framework_or_model_result": False,
        },
        "inputs": {
            "integrity": {
                name: integrity_record(path, workspace)
                for name, path in integrity_paths.items()
            },
            "control_manifest": child_gate["control_manifest"],
            "pypto": base.git_identity(workspace / "projects/pypto"),
            "tensor_ir_head": contract_module.TENSOR_IR_HEAD,
            "cuda_tile_head": contract_module.CUDA_TILE_HEAD,
            "llvm_head": contract_module.LLVM_HEAD,
            "replay_files": replay_files,
        },
        "run_context": {
            "run_id": run_id,
            "mode": "gpu-smoke",
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "start_ticks": base._process_start_ticks(os.getpid()),
            "initial_preflight": {
                "path": initial_path.relative_to(workspace).as_posix(),
                "sha256": initial_sha,
            },
            "preflight": {
                "path": preflight_path.relative_to(workspace).as_posix(),
                "sha256": preflight_sha,
            },
            "gate": {
                "path": str(barrier["gate_path"]),
                "sha256": str(barrier["gate_sha256"]),
                "document": gate,
            },
            "start_barrier_sha256": sha256_file(
                Path(os.environ["PYPTO_GPU_SMOKE_START_BARRIER"])
            ),
            "protected_zero_nvidia_policy": (
                os.environ.get("PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED") == "1"
            ),
            "admission_policy": child_gate["admission_policy"],
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
            "libcudart_paths": base.mapped_library_paths("libcudart.so"),
            "observation": {
                "device_ordinal": target.device_ordinal,
                "device_name": target.device_name,
                "device_uuid": target.device_uuid,
                "pci_device_id": target.pci_device_id,
                "traits": base.target_traits_document(target.traits),
                "cuda_toolkit_version": target.cuda_toolkit_version,
                "cuda_driver_version": target.cuda_driver_version,
                "tensor_ir_revision": target.tensor_ir_revision,
                "cuda_tile_revision": target.cuda_tile_revision,
                "supported_compute_dtypes": base.supported_dtype_names(
                    pypto, target.supported_compute_dtypes
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
                "loader_compatibility_input_digest": request.loader_compatibility_input_digest,
                "device_autotune_identity_digest": request.device_autotune_identity_digest,
            },
            "hir_programs": hir_records,
            "artifacts": artifact_records,
            "executions": executions,
            "case_order": list(contract_module.CASE_ORDER),
            "case_count": 10,
            "compile_invocations_per_case": 1,
            "repetitions_per_case": 2,
            "module_lifetimes": 20,
            "explicit_packet_releases": 20,
            "explicit_unloads": 20,
            "non_default_current_stream": True,
            "distinct_nondefault_reference_stream": True,
            "reference_compute_outside_candidate_coverage": True,
            "external_reference_synchronizations": 20,
            "external_synchronization": True,
            "fallback_used": False,
            "forbidden_provider_imports": [],
        },
    }
    path = contract_module.provisional_path(workspace, run_id)
    payload = canonical_json(provisional)
    digest = base.publish_no_replace(path, payload)
    return provisional, path, digest


contract_module: Any


def main() -> int:
    document, path, digest = run_smoke()
    print(
        json.dumps(
            {"acceptance": document["acceptance"], "path": str(path), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
