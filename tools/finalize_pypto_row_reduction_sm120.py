#!/usr/bin/env python3
"""CPU-only no-replace finalizer for RowReductionV3 SM120 correctness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
BASE_FINALIZER_RELATIVE_PATH = Path("tools/finalize_pypto_fused_pointwise_sm120_v2.py")
BASE_FINALIZER_SIZE = 55_922
BASE_FINALIZER_SHA256 = (
    "f1007836af671a47a75133e0b215f0b2abf2e84060af028295928d4e24d745a8"
)
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FinalizeError(RuntimeError):
    """The row-reduction provisional cannot be promoted."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_exact(
    name: str, path: Path, size: int | None = None, digest: str | None = None
) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise FinalizeError(f"exact row finalizer source is noncanonical: {path}")
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if size is not None and len(raw) != size:
        raise FinalizeError(f"exact row finalizer source size differs: {path}")
    if digest is not None and actual != digest:
        raise FinalizeError(f"exact row finalizer source hash differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


base = load_exact(
    "_pypto_row_reduction_finalizer_base",
    ROOT / BASE_FINALIZER_RELATIVE_PATH,
    BASE_FINALIZER_SIZE,
    BASE_FINALIZER_SHA256,
)
contract = load_exact(
    "_pypto_row_reduction_sm120_contract",
    ROOT / "tools/_pypto_row_reduction_sm120_contract.py",
)
control = load_exact(
    "_pypto_row_reduction_sm120_control_manifest",
    ROOT / "tools/_pypto_row_reduction_sm120_control_manifest.py",
)
control.reject_control_bytecode_cache(ROOT)


def require_exact_keys(
    value: object, expected: set[str], description: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise FinalizeError(f"{description} key set differs")
    return value


def require_sha(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise FinalizeError(f"{description} is not a SHA-256")
    return value


def load_canonical(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise FinalizeError(f"{description} is not a canonical regular file")
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=base.base.duplicate_key_guard)
    if not isinstance(value, dict) or base.base.canonical_json(value) != raw:
        raise FinalizeError(f"{description} is not canonical JSON")
    return value, raw


def anchors() -> tuple[dict[str, object], dict[str, object]]:
    value, raw = load_canonical(
        ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH, "row compile anchors"
    )
    if (
        len(raw) != contract.COMPILE_ANCHORS_SIZE
        or sha256_bytes(raw) != contract.COMPILE_ANCHORS_SHA256
    ):
        raise FinalizeError("row compile-anchor bytes differ")
    records = value.get("records")
    if (
        not isinstance(records, list)
        or len(records) != 10
        or [record.get("case") for record in records] != list(contract.CASE_ORDER)
        or value.get("matrix_policy", {}).get("executions") != 20
    ):
        raise FinalizeError("row compile-anchor schema differs")
    return value, {record["case"]: record for record in records}


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
    return struct.unpack(
        "<f", struct.pack("<I", word if dtype == "float32" else word << 16)
    )[0]


def unpack_words(raw: bytes, dtype: str) -> list[int]:
    width, code = (4, "I") if dtype == "float32" else (2, "H")
    return list(struct.unpack(f"<{len(raw) // width}{code}", raw))


def pack_words(words: list[int], dtype: str) -> bytes:
    code = "I" if dtype == "float32" else "H"
    return struct.pack(f"<{len(words)}{code}", *words)


def input_values(case: object, repetition: int) -> list[float]:
    values: list[float] = []
    for row in range(case.rows):
        if (
            repetition == 0
            and case.dtype == "bfloat16"
            and case.op_name == "tensor.row_sum"
        ):
            values.extend([1.0, *([2.0**-8] * (case.contraction - 1))])
        elif repetition == 0 and case.op_name == "tensor.row_sum":
            values.extend(
                float(((row * 3 + column * 5) % 9) - 4) / 4
                for column in range(case.contraction)
            )
        elif repetition == 0:
            row_values = [
                float(((row * 11 + column * 7) % 127) - 90)
                for column in range(case.contraction)
            ]
            row_values[-1] = float(32 + row)
            values.extend(row_values)
        else:
            mode = row % 5
            if case.op_name == "tensor.row_sum":
                row_values = [-0.0] * case.contraction
                if mode == 1:
                    row_values[0], row_values[1:] = (
                        math.inf,
                        [1.0] * (case.contraction - 1),
                    )
                elif mode == 2:
                    row_values[0], row_values[1:] = (
                        -math.inf,
                        [-1.0] * (case.contraction - 1),
                    )
                elif mode == 3 and case.contraction > 1:
                    row_values[0], row_values[1] = math.inf, -math.inf
                elif mode == 4:
                    row_values[0] = math.nan
                values.extend(row_values)
            elif mode == 0:
                values.extend([-0.0, 0.0, *([-1.0] * (case.contraction - 2))])
            elif mode == 1:
                values.extend([-0.0] * case.contraction)
            elif mode == 2:
                values.extend([math.inf, *([1.0] * (case.contraction - 1))])
            elif mode == 3:
                values.extend([-math.inf] * case.contraction)
            else:
                values.extend([math.nan] * case.contraction)
    return values


def cpu_reference(case: object, repetition: int) -> list[int]:
    inputs = [
        encode_word(value, case.dtype) for value in input_values(case, repetition)
    ]
    output: list[int] = []
    for row in range(case.rows):
        row_values = [
            word_value(word, case.dtype)
            for word in inputs[row * case.contraction : (row + 1) * case.contraction]
        ]
        if any(math.isnan(value) for value in row_values):
            result = math.nan
        elif case.op_name == "tensor.row_sum":
            pos, neg = (
                any(v == math.inf for v in row_values),
                any(v == -math.inf for v in row_values),
            )
            if pos and neg:
                result = math.nan
            elif pos:
                result = math.inf
            elif neg:
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
                    if any(v == 0.0 and math.copysign(1.0, v) > 0 for v in row_values)
                    else -0.0
                )
        output.append(encode_word(result, case.dtype))
    return output


def compare(
    case: object, actual: list[int], expected: list[int], *, exact: bool
) -> dict[str, object]:
    max_ulp, max_rel, max_abs = 0, 0.0, 0.0
    for lhs, rhs in zip(actual, expected, strict=True):
        lhs_class = base.base._classification(lhs, case.dtype)
        rhs_class = base.base._classification(rhs, case.dtype)
        if rhs_class[0] == "nan":
            if lhs_class[0] != "nan":
                raise FinalizeError("NaN classification differs")
            continue
        if rhs_class[0] != "finite":
            if lhs_class != rhs_class:
                raise FinalizeError("special sign/classification differs")
            continue
        if case.op_name == "tensor.row_max" or exact:
            if lhs != rhs:
                raise FinalizeError("exact reduction output differs")
            continue
        ulp = abs(
            base.base._ordered_word(lhs, case.dtype)
            - base.base._ordered_word(rhs, case.dtype)
        )
        left, right = word_value(lhs, case.dtype), word_value(rhs, case.dtype)
        absolute = abs(left - right)
        relative = (
            0.0
            if right == 0.0 and absolute == 0.0
            else (math.inf if right == 0.0 else absolute / abs(right))
        )
        if ulp > case.max_ulp or absolute > case.rtol * abs(right):
            raise FinalizeError("sum tolerance differs")
        max_ulp, max_rel, max_abs = (
            max(max_ulp, ulp),
            max(max_rel, relative),
            max(max_abs, absolute),
        )
    return {
        "observed_max_ulp": max_ulp,
        "observed_max_relative_error": max_rel,
        "observed_max_absolute_error": max_abs,
    }


def comparison_metadata(case: object, repetition: int) -> dict[str, object]:
    discriminator = (
        repetition == 0
        and case.dtype == "bfloat16"
        and case.op_name == "tensor.row_sum"
    )
    expected: int | None = None
    negative: int | None = None
    if discriminator:
        expected = 0x3FC0 if case.contraction == 129 else 0x4000
        accumulator = bfloat16_word(1.0)
        increment = word_value(bfloat16_word(2.0**-8), "bfloat16")
        for _ in range(case.contraction - 1):
            accumulator = bfloat16_word(word_value(accumulator, "bfloat16") + increment)
        negative = accumulator
        if negative != 0x3F80 or negative == expected:
            raise FinalizeError("BF16 sequential-accumulator negative control differs")
    return {
        "policy": case.comparison,
        "max_ulp_limit": case.max_ulp,
        "rtol": case.rtol,
        "atol": 0.0,
        "special_classification_and_sign_passed": True,
        "bf16_fp32_accumulation_discriminator_passed": discriminator,
        "bf16_expected_output_word": expected,
        "bf16_sequential_accumulator_word": negative,
    }


def validate_comparison_metadata(
    value: object, case: object, repetition: int
) -> dict[str, object]:
    comparison = require_exact_keys(
        value,
        {
            "policy",
            "max_ulp_limit",
            "rtol",
            "atol",
            "candidate_vs_torch",
            "candidate_vs_cpu",
            "torch_vs_cpu",
            "special_classification_and_sign_passed",
            "bf16_fp32_accumulation_discriminator_passed",
            "bf16_expected_output_word",
            "bf16_sequential_accumulator_word",
        },
        "row comparison",
    )
    expected = comparison_metadata(case, repetition)
    if any(comparison.get(name) != item for name, item in expected.items()):
        raise FinalizeError("row comparison metadata differs")
    return comparison


def validate_frontend(
    provisional: dict[str, object], anchor_records: dict[str, object]
) -> None:
    runtime = provisional["runtime"]
    if (
        runtime.get("case_order") != list(contract.CASE_ORDER)
        or runtime.get("case_count") != 10
        or runtime.get("compile_invocations_per_case") != 1
        or runtime.get("repetitions_per_case") != 2
        or runtime.get("module_lifetimes") != 20
        or runtime.get("explicit_packet_releases") != 20
        or runtime.get("explicit_unloads") != 20
        or runtime.get("external_reference_synchronizations") != 20
        or runtime.get("fallback_used") is not False
        or runtime.get("forbidden_provider_imports") != []
    ):
        raise FinalizeError("row runtime aggregate differs")
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 10:
        raise FinalizeError("row Artifact set differs")
    for record, case in zip(artifacts, contract.CASE_SPECS, strict=True):
        anchor = anchor_records[case.name]
        if (
            record.get("case") != case.name
            or record.get("build_spec_identity_digest") != anchor["build_spec_sha256"]
            or record.get("artifact_identity_digest") != anchor["artifact_sha256"]
            or record.get("device_code_sha256") != anchor["device_code_sha256"]
            or record.get("fallback_used") is not False
        ):
            raise FinalizeError("row Artifact anchor differs")
    executions = runtime.get("executions")
    if not isinstance(executions, list) or len(executions) != 20:
        raise FinalizeError("row execution set differs")
    observation = runtime.get("observation")
    if not isinstance(observation, dict):
        raise FinalizeError("row runtime observation differs")
    execution_index = 0
    for case in contract.CASE_SPECS:
        anchor = anchor_records[case.name]
        sentinel = contract.SENTINEL_WORDS[case.dtype]
        input_prefix = sha256_bytes(
            pack_words([sentinel[0]] * contract.INPUT_GUARD_ELEMENTS, case.dtype)
        )
        input_suffix = sha256_bytes(
            pack_words([sentinel[1]] * contract.INPUT_GUARD_ELEMENTS, case.dtype)
        )
        output_prefix = sha256_bytes(
            pack_words([sentinel[2]] * contract.OUTPUT_GUARD_ELEMENTS, case.dtype)
        )
        output_suffix = sha256_bytes(
            pack_words([sentinel[3]] * contract.OUTPUT_GUARD_ELEMENTS, case.dtype)
        )
        for repetition in range(2):
            execution = executions[execution_index]
            execution_index += 1
            if not isinstance(execution, dict):
                raise FinalizeError("row execution record is malformed")
            expected_guards = {
                "input_prefix_before_sha256": input_prefix,
                "input_prefix_after_sha256": input_prefix,
                "input_suffix_before_sha256": input_suffix,
                "input_suffix_after_sha256": input_suffix,
                "output_prefix_before_sha256": output_prefix,
                "output_prefix_after_sha256": output_prefix,
                "output_suffix_before_sha256": output_suffix,
                "output_suffix_after_sha256": output_suffix,
            }
            if any(
                execution.get(name) != value for name, value in expected_guards.items()
            ):
                raise FinalizeError("row canary hash differs")
            if (
                execution.get("case") != case.name
                or execution.get("repetition") != repetition
                or execution.get("artifact_identity_digest")
                != anchor["artifact_sha256"]
                or execution.get("fresh_executable") is not True
                or execution.get("input_unchanged") is not True
                or execution.get("input_before_sha256")
                != execution.get("input_after_sha256")
                or execution.get("guards_unchanged") is not True
                or execution.get("input_guard_elements") != 4096
                or execution.get("output_guard_elements") != 16
                or execution.get("comparison_passed") is not True
                or execution.get("non_default_stream") is not True
                or execution.get("current_stream_launch") is not True
                or isinstance(execution.get("raw_current_stream"), bool)
                or not isinstance(execution.get("raw_current_stream"), int)
                or execution.get("raw_current_stream") in {0, 1, 2}
                or isinstance(execution.get("raw_reference_stream"), bool)
                or not isinstance(execution.get("raw_reference_stream"), int)
                or execution.get("raw_reference_stream")
                in {
                    0,
                    1,
                    2,
                    execution.get("raw_current_stream"),
                }
                or execution.get("distinct_nondefault_reference_stream") is not True
                or execution.get("reference_stream_synchronized_before_candidate")
                is not True
                or execution.get("reference_stream_policy")
                != contract.REFERENCE_STREAM_POLICY
                or execution.get("candidate_stream_policy")
                != contract.CANDIDATE_STREAM_POLICY
                or execution.get("reference_compute_boundary")
                != contract.REFERENCE_COMPUTE_BOUNDARY
                or execution.get("capture_free_before") is not True
                or execution.get("capture_free_at_launch") is not True
                or execution.get("external_stream_synchronized") is not True
                or execution.get("packet_released_after_synchronization") is not True
                or execution.get("explicit_unload") is not True
                or execution.get("terminal_state") != "Unloaded"
                or execution.get("bound_context_before_unload")
                != observation.get("context_address")
                or execution.get("bound_context_id_before_unload")
                != observation.get("context_id")
                or execution.get("bound_context_after_unload") != 0
                or execution.get("bound_context_id_after_unload") != 0
            ):
                raise FinalizeError("row execution lifecycle differs")


def expected_replay_names() -> list[str]:
    names = ["compile-request.msgpack"]
    for case in contract.CASE_SPECS:
        names.extend(
            [
                f"{case.name}.hir.msgpack",
                f"{case.name}.source.mlir",
                f"{case.name}.build-spec.msgpack",
                f"{case.name}.artifact.msgpack",
                f"{case.name}.cubin",
            ]
        )
    for case in contract.CASE_SPECS:
        for repetition in range(2):
            names.extend(
                [
                    f"{case.name}.r{repetition}.input.bin",
                    f"{case.name}.r{repetition}.reference.bin",
                    f"{case.name}.r{repetition}.actual.bin",
                    f"{case.name}.r{repetition}.cpu-reference.bin",
                ]
            )
    return names


def validate_replay(
    provisional: dict[str, object], run_id: str
) -> list[dict[str, object]]:
    records = provisional["inputs"]["replay_files"]
    names = expected_replay_names()
    replay = contract.replay_directory(ROOT, run_id)
    if not isinstance(records, list) or len(records) != len(names):
        raise FinalizeError("row replay set differs")
    if sorted(path.name for path in replay.iterdir()) != sorted(
        [*names, contract.PROVISIONAL_NAME]
    ):
        raise FinalizeError("row replay directory differs")
    for record, name in zip(records, names, strict=True):
        path = replay / name
        if (
            record.get("path") != path.relative_to(ROOT).as_posix()
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or stat.S_IMODE(path.stat().st_mode) != 0o444
            or sha256_file(path) != record.get("sha256")
        ):
            raise FinalizeError(f"row replay bytes differ: {name}")
    return records


def audit_numerical(
    provisional: dict[str, object], run_id: str
) -> list[dict[str, object]]:
    replay = contract.replay_directory(ROOT, run_id)
    executions = provisional["runtime"]["executions"]
    output: list[dict[str, object]] = []
    index = 0
    for case in contract.CASE_SPECS:
        for repetition in range(2):
            execution = executions[index]
            index += 1
            input_raw = (replay / f"{case.name}.r{repetition}.input.bin").read_bytes()
            expected_input = pack_words(
                [
                    encode_word(value, case.dtype)
                    for value in input_values(case, repetition)
                ],
                case.dtype,
            )
            if (
                input_raw != expected_input
                or sha256_bytes(input_raw) != execution.get("input_before_sha256")
                or sha256_bytes(input_raw) != execution.get("input_after_sha256")
            ):
                raise FinalizeError("row independent input reconstruction differs")
            reference_raw = (
                replay / f"{case.name}.r{repetition}.reference.bin"
            ).read_bytes()
            actual_raw = (replay / f"{case.name}.r{repetition}.actual.bin").read_bytes()
            cpu_raw = (
                replay / f"{case.name}.r{repetition}.cpu-reference.bin"
            ).read_bytes()
            reconstructed_cpu = cpu_reference(case, repetition)
            if cpu_raw != pack_words(reconstructed_cpu, case.dtype):
                raise FinalizeError("row independent CPU reference differs")
            reference_words = unpack_words(reference_raw, case.dtype)
            actual_words = unpack_words(actual_raw, case.dtype)
            cpu_words = unpack_words(cpu_raw, case.dtype)
            exact = repetition == 0 and case.op_name == "tensor.row_sum"
            candidate_torch = compare(case, actual_words, reference_words, exact=exact)
            candidate_cpu = compare(case, actual_words, cpu_words, exact=exact)
            torch_cpu = compare(case, reference_words, cpu_words, exact=exact)
            recorded = validate_comparison_metadata(
                execution.get("comparison"), case, repetition
            )
            if (
                recorded.get("candidate_vs_torch") != candidate_torch
                or recorded.get("candidate_vs_cpu") != candidate_cpu
                or recorded.get("torch_vs_cpu") != torch_cpu
                or sha256_bytes(reference_raw)
                != execution.get("expected_logical_bytes_sha256")
                or sha256_bytes(actual_raw)
                != execution.get("actual_logical_bytes_sha256")
                or sha256_bytes(cpu_raw) != execution.get("cpu_reference_bytes_sha256")
            ):
                raise FinalizeError("row three-way numerical join differs")
            output.append(
                {
                    "case": case.name,
                    "repetition": repetition,
                    "candidate_vs_torch": candidate_torch,
                    "candidate_vs_cpu": candidate_cpu,
                    "torch_vs_cpu": torch_cpu,
                    "input_reconstructed": True,
                    "cpu_reference_reconstructed": True,
                }
            )
    return output


REPLAY_PROGRAM = r"""
import hashlib,json,sys
from pathlib import Path
from types import ModuleType
workspace=Path(sys.argv[1]).resolve(strict=True); replay=Path(sys.argv[2]).resolve(strict=True)
def load(name,path):
 raw=path.read_bytes(); m=ModuleType(name); m.__file__=str(path); m.__package__=""; sys.modules[name]=m; exec(compile(raw,str(path),"exec",dont_inherit=True),m.__dict__); return m
contract=load("row_contract",workspace/"tools/_pypto_row_reduction_sm120_contract.py")
base=load("row_base",workspace/"benchmarks/operators/pypto_fused_pointwise_sm120.py")
site=workspace/"envs/pypto-nvidia/lib/python3.14/site-packages"; sys.path.insert(0,str(site))
pypto=base.bootstrap_exact_pypto(workspace,(workspace/contract.PYPTO_DSO_RELATIVE_PATH).parent)
import torch
from pypto import compiler
assert not torch.cuda.is_initialized()
request=compiler.CompileRequest.deserialize((replay/"compile-request.msgpack").read_bytes())
records=[]
for case in contract.CASE_SPECS:
 hir=(replay/f"{case.name}.hir.msgpack").read_bytes(); program=pypto.ir.deserialize(hir); assert bytes(pypto.ir.serialize(program))==hir
 build_raw=(replay/f"{case.name}.build-spec.msgpack").read_bytes(); build=compiler.KernelBuildSpec.deserialize(build_raw); assert build.serialize()==build_raw
 artifact_raw=(replay/f"{case.name}.artifact.msgpack").read_bytes(); artifact=compiler.Artifact.deserialize(artifact_raw,request,build); assert artifact.serialize()==artifact_raw
 cubin=(replay/f"{case.name}.cubin").read_bytes(); assert cubin==bytes(artifact.device_code) and hashlib.sha256(cubin).hexdigest()==artifact.device_code_sha256 and not artifact.fallback_used
 records.append({"case":case.name,"hir_sha256":hashlib.sha256(hir).hexdigest(),"build_spec_sha256":hashlib.sha256(build_raw).hexdigest(),"artifact_sha256":hashlib.sha256(artifact_raw).hexdigest(),"device_code_sha256":hashlib.sha256(cubin).hexdigest()})
assert not torch.cuda.is_initialized()
print(json.dumps({"torch_cuda_initialized":False,"records":records},sort_keys=True))
"""


def replay_semantics(
    run_id: str, anchor_records: dict[str, object]
) -> dict[str, object]:
    python = (ROOT / contract.PYTHON_REAL_RELATIVE_PATH).resolve(strict=True)
    replay = contract.replay_directory(ROOT, run_id)
    command = [
        str(python),
        "-I",
        "-B",
        "-S",
        "-c",
        REPLAY_PROGRAM,
        str(ROOT),
        str(replay),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode:
        raise FinalizeError(
            "row CPU-only DSO replay failed: " + completed.stderr[-2048:]
        )
    value = json.loads(completed.stdout)
    if value.get("torch_cuda_initialized") is not False:
        raise FinalizeError("row CPU-only replay initialized CUDA")
    for record in value["records"]:
        anchor = anchor_records[record["case"]]
        if any(record[name] != anchor[name] for name in record if name != "case"):
            raise FinalizeError("row CPU-only replay anchor differs")
    return {
        "command_sha256": sha256_bytes("\0".join(command).encode()),
        "stdout_sha256": sha256_bytes(completed.stdout.encode()),
        **value,
    }


def validate_run_documents(
    *,
    run_id: str,
    process: dict[str, object],
    initial: dict[str, object],
    initial_raw: bytes,
    preflight: dict[str, object],
    preflight_raw: bytes,
    gate: dict[str, object],
    gate_raw: bytes,
    barrier: dict[str, object],
    barrier_raw: bytes,
    provisional: dict[str, object],
    control_identity: dict[str, object],
) -> bool:
    require_exact_keys(
        process,
        {
            "schema",
            "run_id",
            "workspace",
            "environment",
            "environment_access_lock",
            "framework_profile",
            "framework_launch",
            "mode",
            "coexistence",
            "gpu_smoke",
            "initial_preflight",
            "preflight",
            "resource_policy",
            "command",
            "pid",
            "pgid",
            "start_ticks",
            "started_at",
            "status",
            "gpu_smoke_pre_release_audit",
            "gpu_smoke_last_audit",
            "gpu_smoke_post_exit_audit",
            "return_code",
            "finished_at",
        },
        "row process metadata",
    )
    require_exact_keys(
        gate,
        {
            "schema",
            "run_id",
            "pid",
            "pgid",
            "start_ticks",
            "command",
            "initial_preflight",
            "preflight",
            "static_identity",
            "control_manifest",
            "runtime_isolation",
            "admission_policy",
        },
        "row pre-release gate",
    )
    require_exact_keys(
        barrier,
        {"schema", "run_id", "pid", "pgid", "start_ticks", "gate_path", "gate_sha256"},
        "row start barrier",
    )
    try:
        initial_requested = base.validate_preflight(
            initial, description="initial row preflight"
        )
        requested = base.validate_preflight(
            preflight, description="action row preflight"
        )
        base.validate_process_policy(
            process, run_id=run_id, requested=requested, preflight=preflight
        )
        base.validate_audit(
            process["gpu_smoke_pre_release_audit"],
            description="row pre-release",
            authorized=requested,
            require_zero_owned=True,
        )
        base.validate_audit(
            process["gpu_smoke_last_audit"],
            description="row periodic",
            authorized=requested,
            require_zero_owned=False,
        )
        base.validate_audit(
            process["gpu_smoke_post_exit_audit"],
            description="row post-exit",
            authorized=requested,
            require_zero_owned=True,
        )
    except base.FinalizeV2Error as error:
        raise FinalizeError(str(error)) from error
    if initial_requested is not requested:
        raise FinalizeError("row initial/action authorization differs")
    initial_path = ROOT / "runs" / run_id / "initial-preflight.json"
    preflight_path = ROOT / "runs" / run_id / "preflight.json"
    gate_path = ROOT / "runs" / run_id / "gpu-smoke-gate.json"
    resource = require_exact_keys(
        process.get("resource_policy"),
        {"timeout_seconds", "minimum_free_disk_bytes", "owned_run_pause_memory_kib"},
        "row resource policy",
    )
    gpu_smoke = process.get("gpu_smoke")
    identity = {
        "schema": 2,
        "run_id": run_id,
        "pid": process.get("pid"),
        "pgid": process.get("pgid"),
        "start_ticks": process.get("start_ticks"),
    }
    if (
        process.get("schema") != 4
        or process.get("status") != "exited"
        or process.get("return_code") != 0
        or process.get("command") != contract.fixed_child_command(ROOT)
        or resource.get("timeout_seconds") != contract.GPU_SMOKE_TIMEOUT_SECONDS
        or resource.get("minimum_free_disk_bytes")
        != contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB << 30
        or resource.get("owned_run_pause_memory_kib")
        != contract.OWNED_RUN_ABORT_MEMORY_FLOOR_KIB
        or not isinstance(gpu_smoke, dict)
        or process.get("initial_preflight")
        != {"path": str(initial_path), "sha256": sha256_bytes(initial_raw)}
        or process.get("preflight")
        != {"path": str(preflight_path), "sha256": sha256_bytes(preflight_raw)}
        or any(gate.get(name) != value for name, value in identity.items())
        or any(barrier.get(name) != value for name, value in identity.items())
        or gate.get("control_manifest") != control_identity
        or gate.get("runtime_isolation") != process.get("gpu_smoke_pre_release_audit")
        or gate.get("initial_preflight") != process.get("initial_preflight")
        or gate.get("preflight") != process.get("preflight")
        or gate.get("admission_policy") != preflight.get("gpu_smoke_admission_policy")
        or barrier.get("gate_path") != str(gate_path)
        or barrier.get("gate_sha256") != sha256_bytes(gate_raw)
        or gpu_smoke.get("gate_sha256") != sha256_bytes(gate_raw)
        or gpu_smoke.get("start_barrier_sha256") != sha256_bytes(barrier_raw)
    ):
        raise FinalizeError("row process/gate/barrier identity differs")
    run = provisional["run_context"]
    if (
        run.get("initial_preflight")
        != {
            "path": initial_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(initial_raw),
        }
        or run.get("preflight")
        != {
            "path": preflight_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(preflight_raw),
        }
        or run.get("gate")
        != {
            "path": str(gate_path),
            "sha256": sha256_bytes(gate_raw),
            "document": gate,
        }
        or run.get("start_barrier_sha256") != sha256_bytes(barrier_raw)
        or run.get("protected_zero_nvidia_policy") is not requested
        or run.get("admission_policy") != preflight.get("gpu_smoke_admission_policy")
    ):
        raise FinalizeError("row provisional run-context joins differ")
    try:
        base.validate_run_context_identity(run, process, gate)
        base.validate_child_gate(
            provisional["runtime"]["child_pre_cuda_gate"],
            requested=requested,
            control_identity=control_identity,
        )
        base.validate_child_parent_joins(
            provisional["runtime"]["child_pre_cuda_gate"], preflight, process
        )
        base.validate_runtime_identity(provisional, preflight, gate)
    except base.FinalizeV2Error as error:
        raise FinalizeError(str(error)) from error
    return requested


def validate_integrity(provisional: dict[str, object]) -> None:
    expected = {
        "runner": ROOT / contract.RUNNER_RELATIVE_PATH,
        "contract": ROOT / "tools/_pypto_row_reduction_sm120_contract.py",
        "anchor_generator": ROOT / contract.ANCHOR_GENERATOR_RELATIVE_PATH,
        "compile_anchors": ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH,
        "controller": ROOT / contract.CONTROLLER_RELATIVE_PATH,
        "control_validator": ROOT / contract.CONTROL_VALIDATOR_RELATIVE_PATH,
        "preflight": ROOT / contract.PREFLIGHT_ADAPTER_RELATIVE_PATH,
        "pypto_dso": ROOT / contract.PYPTO_DSO_RELATIVE_PATH,
        "environment_lock": ROOT / "ENVIRONMENT.lock",
        "versions_lock": ROOT / "VERSIONS.lock",
        "workspace_lock": ROOT / "WORKSPACE.lock",
    }
    integrity = provisional["inputs"]["integrity"]
    if not isinstance(integrity, dict) or set(integrity) != set(expected):
        raise FinalizeError("row integrity set differs")
    for name, path in expected.items():
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise FinalizeError(f"row integrity path differs: {name}")
        record = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if integrity.get(name) != record:
            raise FinalizeError(f"row integrity record differs: {name}")


def validate_provisional_schema(provisional: dict[str, object]) -> None:
    require_exact_keys(
        provisional,
        {
            "schema_version",
            "smoke",
            "acceptance",
            "scope",
            "inputs",
            "run_context",
            "runtime",
        },
        "row provisional",
    )
    scope = require_exact_keys(
        provisional.get("scope"),
        {
            "frontend_family",
            "fixed_case_count",
            "fixed_case_correctness",
            "general_reduction_correctness",
            "performance_result",
            "framework_or_model_result",
        },
        "row scope",
    )
    if scope != {
        "frontend_family": "RowReductionV3",
        "fixed_case_count": 10,
        "fixed_case_correctness": True,
        "general_reduction_correctness": False,
        "performance_result": False,
        "framework_or_model_result": False,
    }:
        raise FinalizeError("row scope exceeds fixed correctness")
    require_exact_keys(
        provisional.get("inputs"),
        {
            "integrity",
            "control_manifest",
            "pypto",
            "tensor_ir_head",
            "cuda_tile_head",
            "llvm_head",
            "replay_files",
        },
        "row inputs",
    )
    require_exact_keys(
        provisional.get("run_context"),
        {
            "run_id",
            "mode",
            "pid",
            "pgid",
            "start_ticks",
            "initial_preflight",
            "preflight",
            "gate",
            "start_barrier_sha256",
            "protected_zero_nvidia_policy",
            "admission_policy",
        },
        "row run context",
    )
    require_exact_keys(
        provisional.get("runtime"),
        {
            "torch",
            "child_pre_cuda_gate",
            "libcudart_paths",
            "observation",
            "compile_request",
            "hir_programs",
            "artifacts",
            "executions",
            "case_order",
            "case_count",
            "compile_invocations_per_case",
            "repetitions_per_case",
            "module_lifetimes",
            "explicit_packet_releases",
            "explicit_unloads",
            "non_default_current_stream",
            "distinct_nondefault_reference_stream",
            "reference_compute_outside_candidate_coverage",
            "external_reference_synchronizations",
            "external_synchronization",
            "fallback_used",
            "forbidden_provider_imports",
        },
        "row runtime",
    )


def finalize(
    *, workspace: Path, run_id: str, expected_provisional_sha256: str
) -> tuple[dict[str, object], Path, str]:
    if workspace.resolve(strict=True) != ROOT or workspace.absolute() != ROOT:
        raise FinalizeError("workspace must be the exact root")
    base.base.require_no_site_finalizer()
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise FinalizeError("row run ID is malformed")
    require_sha(expected_provisional_sha256, "external provisional anchor")
    control_identity = control.validate_control_manifest(ROOT)
    anchor_manifest, anchor_records = anchors()
    run_dir = ROOT / "runs" / run_id
    paths = {
        "process": run_dir / "process.json",
        "initial": run_dir / "initial-preflight.json",
        "preflight": run_dir / "preflight.json",
        "gate": run_dir / "gpu-smoke-gate.json",
        "barrier": run_dir / "gpu-smoke-start-barrier.json",
        "provisional": contract.provisional_path(ROOT, run_id),
    }
    loaded = {name: load_canonical(path, name) for name, path in paths.items()}
    for name in ("process", "initial", "preflight", "gate", "barrier"):
        if stat.S_IMODE(paths[name].stat().st_mode) != 0o600:
            raise FinalizeError(f"row sidecar mode differs: {name}")
    if stat.S_IMODE(paths["provisional"].stat().st_mode) != 0o444:
        raise FinalizeError("row provisional mode differs")
    provisional, provisional_raw = loaded["provisional"]
    validate_provisional_schema(provisional)
    if (
        sha256_bytes(provisional_raw) != expected_provisional_sha256
        or provisional.get("schema_version") != 1
        or provisional.get("smoke") != contract.SMOKE_NAME
        or provisional.get("acceptance")
        != "gpu-execution-complete-awaiting-run-finalization"
        or provisional["inputs"].get("control_manifest") != control_identity
        or provisional["inputs"].get("pypto")
        != {"head": contract.PYPTO_HEAD, "tree": contract.PYPTO_TREE, "clean": True}
        or provisional["inputs"].get("tensor_ir_head") != contract.TENSOR_IR_HEAD
        or provisional["inputs"].get("cuda_tile_head") != contract.CUDA_TILE_HEAD
        or provisional["inputs"].get("llvm_head") != contract.LLVM_HEAD
    ):
        raise FinalizeError("row provisional identity differs")
    validate_run_documents(
        run_id=run_id,
        process=loaded["process"][0],
        initial=loaded["initial"][0],
        initial_raw=loaded["initial"][1],
        preflight=loaded["preflight"][0],
        preflight_raw=loaded["preflight"][1],
        gate=loaded["gate"][0],
        gate_raw=loaded["gate"][1],
        barrier=loaded["barrier"][0],
        barrier_raw=loaded["barrier"][1],
        provisional=provisional,
        control_identity=control_identity,
    )
    validate_integrity(provisional)
    validate_frontend(provisional, anchor_records)
    replay_files = validate_replay(provisional, run_id)
    numerical = audit_numerical(provisional, run_id)
    semantics = replay_semantics(run_id, anchor_records)
    if base.base.git_identity(ROOT / "projects/pypto") != {
        "head": contract.PYPTO_HEAD,
        "tree": contract.PYPTO_TREE,
        "clean": True,
    }:
        raise FinalizeError("row PyPTO identity differs at finalization")
    report = {
        "schema_version": 1,
        "smoke": contract.SMOKE_NAME,
        "status": "accepted-real-sm120-row-reduction-ten-case-correctness-gate",
        "scope": provisional["scope"],
        "not_claimed": [
            "general RowReductionV3 correctness",
            "other shapes dtypes operators or special-value domains",
            "performance CUDA Graph framework model or strict coverage",
            "any reinterpretation of CP47 or CP48",
        ],
        "run": {
            "run_id": run_id,
            "process_sha256": sha256_bytes(loaded["process"][1]),
            "initial_preflight_sha256": sha256_bytes(loaded["initial"][1]),
            "preflight_sha256": sha256_bytes(loaded["preflight"][1]),
            "gate_sha256": sha256_bytes(loaded["gate"][1]),
            "start_barrier_sha256": sha256_bytes(loaded["barrier"][1]),
            "provisional_sha256": expected_provisional_sha256,
            "command": contract.fixed_child_command(ROOT),
            "zero_nvidia_interference": True,
        },
        "inputs": {
            "control_manifest": control_identity,
            "compile_anchors": {
                "path": contract.COMPILE_ANCHORS_RELATIVE_PATH.as_posix(),
                "bytes": contract.COMPILE_ANCHORS_SIZE,
                "sha256": contract.COMPILE_ANCHORS_SHA256,
                "records_sha256": anchor_manifest["records_sha256"],
            },
            "replay_files": replay_files,
            "numerical_replay": numerical,
            "replay_semantics": semantics,
        },
        "result": provisional["runtime"],
        "finalizer": {
            "path": Path(__file__).resolve(strict=True).relative_to(ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "cpu_only_deserialization": True,
            "torch_cuda_initialized": False,
        },
    }
    output_parent = ROOT / contract.FINAL_REPORT_DIRECTORY
    output_parent.mkdir(parents=True, exist_ok=True)
    output = contract.final_report_path(ROOT, run_id)
    try:
        digest = base.base.publish_no_replace(output, report)
    except base.base.FinalizeError as error:
        raise FinalizeError(str(error)) from error
    return report, output, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-provisional-sha256", required=True)
    args = parser.parse_args()
    report, output, digest = finalize(
        workspace=args.workspace,
        run_id=args.run_id,
        expected_provisional_sha256=args.expected_provisional_sha256,
    )
    print(
        json.dumps({"path": str(output), "sha256": digest, "status": report["status"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
