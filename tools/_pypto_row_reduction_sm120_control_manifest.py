#!/usr/bin/env python3
"""Validate the separately published RowReductionV3 SM120 control manifest."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "pypto-row-reduction-sm120-controls-v1"
MANIFEST_RELATIVE_PATH = Path("state/contracts/pypto_row_reduction_sm120_v1.json")
BASE_ADMISSION_MANIFEST_RELATIVE_PATH = Path(
    "state/contracts/pypto_fused_pointwise_sm120_v2.json"
)
BASE_ADMISSION_MANIFEST_SIZE = 1_553
BASE_ADMISSION_MANIFEST_SHA256 = (
    "d3b16079c811dd2fbe610ba264d81117e8c4a44886b74caaddb684df2d467036"
)
BASE_ADMISSION_VALIDATOR_RELATIVE_PATH = Path(
    "tools/_pypto_fused_pointwise_sm120_control_manifest_v2.py"
)
BASE_ADMISSION_VALIDATOR_SIZE = 10_807
BASE_ADMISSION_VALIDATOR_SHA256 = (
    "6c2737daf653ac237a2da0081ad05b9d3e14593e2862582e74898f92c7c94ebf"
)
CONTROL_PATHS = (
    "benchmarks/operators/pypto_row_reduction_sm120.py",
    "tools/_pypto_row_reduction_sm120_contract.py",
    "tools/generate_pypto_row_reduction_anchors.py",
    "state/contracts/pypto_row_reduction_compile_anchors_v1.json",
    "tools/_pypto_row_reduction_sm120_control_manifest.py",
    "tools/run_pypto_row_reduction_sm120_isolated.py",
    "tools/finalize_pypto_row_reduction_sm120.py",
)
PYTHON_SOURCE_PATHS = (
    *[path for path in CONTROL_PATHS if path.endswith(".py")],
    "tests/test_pypto_row_reduction_sm120.py",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")


class ControlManifestError(RuntimeError):
    """The row-reduction controls do not match reviewed bytes."""


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


def duplicate_key_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ControlManifestError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_exact(name: str, path: Path, size: int, digest: str) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise ControlManifestError(
            f"exact base admission source is noncanonical: {path}"
        )
    raw = path.read_bytes()
    if len(raw) != size or sha256_bytes(raw) != digest:
        raise ControlManifestError(f"exact base admission source differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


base_admission = load_exact(
    "_pypto_row_reduction_base_admission_validator",
    ROOT / BASE_ADMISSION_VALIDATOR_RELATIVE_PATH,
    BASE_ADMISSION_VALIDATOR_SIZE,
    BASE_ADMISSION_VALIDATOR_SHA256,
)


def control_bytecode_cache_entries(root: Path) -> list[str]:
    entries: set[str] = set()
    for relative in PYTHON_SOURCE_PATHS:
        source = root / relative
        for suffix in (".pyc", ".pyo"):
            candidate = source.with_suffix(suffix)
            if candidate.exists() or candidate.is_symlink():
                entries.add(candidate.relative_to(root).as_posix())
        cache = source.parent / "__pycache__"
        if not cache.is_dir():
            continue
        for candidate in cache.iterdir():
            if candidate.name.startswith(source.stem + ".") and candidate.name.endswith(
                (".pyc", ".pyo")
            ):
                entries.add(candidate.relative_to(root).as_posix())
    return sorted(entries)


def reject_control_bytecode_cache(root: Path) -> None:
    entries = control_bytecode_cache_entries(root)
    if entries:
        raise ControlManifestError(
            "row-reduction control bytecode/cache entries are forbidden: "
            + ", ".join(entries)
        )


def git(root: Path, *arguments: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=text, capture_output=True
    ).stdout


def load_canonical(
    path: Path, size: int, digest: str, description: str
) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise ControlManifestError(f"{description} bytes differ")
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ControlManifestError(f"{description} is not canonical JSON")
    return value


def load_exact_json(
    path: Path, size: int, digest: str, description: str
) -> dict[str, object]:
    """Load accepted JSON whose immutable producer used a different layout."""

    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise ControlManifestError(f"{description} bytes differ")
    value = json.loads(path.read_bytes(), object_pairs_hook=duplicate_key_guard)
    if not isinstance(value, dict):
        raise ControlManifestError(f"{description} is not a JSON object")
    return value


def require_exact_keys(
    value: object, expected: set[str], description: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ControlManifestError(f"{description} key set differs")
    return value


def classify_word(word: int, dtype: str) -> dict[str, object]:
    fraction_bits = 23 if dtype == "float32" else 7
    sign = word >> (fraction_bits + 8)
    exponent = (word >> fraction_bits) & 0xFF
    fraction = word & ((1 << fraction_bits) - 1)
    if exponent == 0xFF:
        classification = "nan" if fraction else "inf"
    elif exponent == 0:
        classification = "zero" if fraction == 0 else "subnormal"
    else:
        classification = "finite"
    return {"class": classification, "sign": sign}


def validate_numerical_oracles(
    value: object, case: object, contract: ModuleType
) -> None:
    if not isinstance(value, list) or len(value) != contract.REPETITIONS:
        raise ControlManifestError("row-reduction numerical oracle set differs")
    width = 4 if case.dtype == "float32" else 2
    code = "I" if case.dtype == "float32" else "H"
    mask = 0xFFFFFFFF if case.dtype == "float32" else 0xFFFF
    expected_keys = {
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
    for repetition, item in enumerate(value):
        item = require_exact_keys(
            item, expected_keys, "row-reduction numerical oracle"
        )
        words = item.get("cpu_reference_words")
        if (
            not isinstance(words, list)
            or len(words) != case.rows
            or any(
                isinstance(word, bool)
                or not isinstance(word, int)
                or word < 0
                or word > mask
                for word in words
            )
        ):
            raise ControlManifestError("row-reduction oracle output words differ")
        raw = struct.pack(f"<{len(words)}{code}", *words)
        expected_classes = [classify_word(word, case.dtype) for word in words]
        if (
            item.get("repetition") != repetition
            or item.get("input_elements") != case.rows * case.contraction
            or item.get("input_word_bytes")
            != case.rows * case.contraction * width
            or not isinstance(item.get("input_word_sha256"), str)
            or SHA256_PATTERN.fullmatch(item["input_word_sha256"]) is None
            or item.get("cpu_reference_word_bytes") != case.rows * width
            or item.get("cpu_reference_word_sha256") != sha256_bytes(raw)
            or item.get("cpu_reference_class_sign") != expected_classes
            or item.get("comparison_modes")
            != list(case.output_comparison_modes(repetition))
            or item.get("element_width_bytes") != width
        ):
            raise ControlManifestError("row-reduction numerical oracle projection differs")


def expected_anchor_replay_names(contract: ModuleType) -> list[str]:
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
    return names


def validate_anchor_replay_closure(
    root: Path, run_id: str, record: dict[str, object], contract: ModuleType
) -> None:
    names = expected_anchor_replay_names(contract)
    records = record.get("replay_files")
    replay = root / "runs" / run_id / "row-reduction-compile-anchor-replay"
    if (
        replay.is_symlink()
        or not replay.is_dir()
        or replay.resolve(strict=True) != replay
        or not isinstance(records, list)
        or len(records) != len(names)
        or sorted(path.name for path in replay.iterdir()) != sorted(names)
    ):
        raise ControlManifestError("row-reduction anchor replay set differs")
    replay_items: dict[str, dict[str, object]] = {}
    for item, name in zip(records, names, strict=True):
        item = require_exact_keys(
            item, {"path", "bytes", "sha256"}, "row-reduction anchor replay record"
        )
        path = replay / name
        expected_relative = path.relative_to(root).as_posix()
        if (
            item.get("path") != expected_relative
            or isinstance(item.get("bytes"), bool)
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] < 0
            or not isinstance(item.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(item["sha256"]) is None
            or path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or path.stat().st_size != item["bytes"]
            or stat.S_IMODE(path.stat().st_mode) != 0o444
            or sha256_file(path) != item["sha256"]
        ):
            raise ControlManifestError(
                f"row-reduction anchor replay bytes differ: {name}"
            )
        replay_items[name] = item
    anchor_request = require_exact_keys(
        record.get("anchor_request"),
        {"path", "bytes", "sha256", "derived_bytes", "derived_sha256"},
        "row-reduction anchor request",
    )
    request_replay = replay_items["compile-request.msgpack"]
    if (
        request_replay["bytes"] != anchor_request["derived_bytes"]
        or request_replay["sha256"] != anchor_request["derived_sha256"]
    ):
        raise ControlManifestError("row-reduction CompileRequest replay join differs")
    case_records = record.get("records")
    if not isinstance(case_records, list) or len(case_records) != len(
        contract.CASE_SPECS
    ):
        raise ControlManifestError("row-reduction replay case records differ")
    joins = (
        ("hir.msgpack", "hir_bytes", "hir_sha256"),
        ("source.mlir", "source_bytes", "source_sha256"),
        ("build-spec.msgpack", "build_spec_bytes", "build_spec_sha256"),
        ("artifact.msgpack", "artifact_bytes", "artifact_sha256"),
        ("cubin", "device_code_bytes", "device_code_sha256"),
    )
    for case_record, case in zip(case_records, contract.CASE_SPECS, strict=True):
        if not isinstance(case_record, dict) or case_record.get("case") != case.name:
            raise ControlManifestError("row-reduction replay case order differs")
        for suffix, bytes_name, digest_name in joins:
            replay_item = replay_items[f"{case.name}.{suffix}"]
            if (
                replay_item["bytes"] != case_record.get(bytes_name)
                or replay_item["sha256"] != case_record.get(digest_name)
            ):
                raise ControlManifestError(
                    f"row-reduction replay metadata join differs: {case.name}.{suffix}"
                )
        if (
            case_record.get("source_ir_digest")
            != case_record.get("source_sha256")
            or case_record.get("kernel_build_spec_digest")
            != case_record.get("build_spec_sha256")
        ):
            raise ControlManifestError(
                f"row-reduction replay identity join differs: {case.name}"
            )


def validate_semantic_abi_projection(
    value: object, record: dict[str, object], case: object, contract: ModuleType
) -> None:
    semantic = require_exact_keys(
        value,
        {
            "frontend_metadata_schema_version",
            "semantic_route",
            "runtime_kernel_name",
            "entry_function_name",
            "argument_packing_policy",
            "argument_layout",
            "grid_abi",
            "workspace_abi",
            "launch_abi",
            "lowered_access",
            "build_spec",
            "artifact_identities",
            "kernel_abi_identity_digest",
            "producer",
            "cache_key_digest",
            "loader_compatibility_digest",
            "fallback_used",
        },
        "row-reduction semantic ABI",
    )
    layout = require_exact_keys(
        semantic.get("argument_layout"),
        {
            "input_operand_count",
            "total_kernel_argument_count",
            "uniform_signature",
            "operand_descriptors",
        },
        "row-reduction argument layout",
    )
    grid = require_exact_keys(
        semantic.get("grid_abi"),
        {"policy", "shape_operand_index", "static_dimensions", "tile_sizes"},
        "row-reduction grid ABI",
    )
    workspace = require_exact_keys(
        semantic.get("workspace_abi"),
        {"version", "kind", "size_bytes", "alignment_bytes"},
        "row-reduction workspace ABI",
    )
    launch = require_exact_keys(
        semantic.get("launch_abi"),
        {
            "version",
            "block_dimensions",
            "cluster_scheduling_policy",
            "dynamic_shared_memory_bytes",
            "kernel_argument_slot_bytes",
        },
        "row-reduction launch ABI",
    )
    lowered_access = require_exact_keys(
        semantic.get("lowered_access"),
        {
            "reduction_tile_budget",
            "rows",
            "row_tile",
            "materialized_rows",
            "contraction",
            "contraction_tile",
            "contraction_chunks",
            "input_logical_elements",
            "input_guard_max_index",
            "required_input_guard_elements",
            "output_logical_elements",
            "output_guard_max_index",
            "required_output_guard_elements",
        },
        "row-reduction lowered access",
    )
    digest_names = {
        "static_specialization_digest",
        "symbolic_specialization_digest",
        "argument_abi_digest",
        "result_abi_digest",
        "mutation_abi_digest",
    }
    build = require_exact_keys(
        semantic.get("build_spec"),
        {
            "schema_version",
            "pipeline_revision",
            "source_ir_digest",
            "callable_abi_digest",
            "compile_request_byte_identity_digest",
            "catalog_provenance",
            *digest_names,
        },
        "row-reduction semantic BuildSpec",
    )
    identities = require_exact_keys(
        semantic.get("artifact_identities"),
        {
            "kernel_build_spec_digest",
            "source_ir_digest",
            "callable_abi_digest",
            "compile_request_byte_identity_digest",
            *digest_names,
        },
        "row-reduction Artifact identities",
    )
    producer = require_exact_keys(
        semantic.get("producer"),
        {
            "kind",
            "pipeline_revision",
            "producer_result_contract",
            "options_identity_digest",
            "environment_overrides_enabled",
            "artifact_fallback_allowed",
        },
        "row-reduction Artifact producer",
    )
    descriptors = layout.get("operand_descriptors")
    if not isinstance(descriptors, list) or len(descriptors) != 2:
        raise ControlManifestError("row-reduction descriptor set differs")
    expected_shapes = (case.shape, case.result_shape)
    for descriptor, shape in zip(descriptors, expected_shapes, strict=True):
        descriptor = require_exact_keys(
            descriptor,
            {
                "kind",
                "rank",
                "shape",
                "strides",
                "dynamic_size_count",
                "dynamic_stride_count",
                "explicit_strides",
                "scalar_size_bytes",
            },
            "row-reduction operand descriptor",
        )
        if descriptor != {
            "kind": "Tensor",
            "rank": len(shape),
            "shape": list(shape),
            "strides": [] if len(shape) == 1 else list(contract.dense_strides(shape)),
            "dynamic_size_count": 0,
            "dynamic_stride_count": 0,
            "explicit_strides": len(shape) > 1,
            "scalar_size_bytes": 0,
        }:
            raise ControlManifestError("row-reduction operand descriptor differs")
    if (
        semantic.get("frontend_metadata_schema_version") != 3
        or semantic.get("semantic_route") != "StructuredTensorIr"
        or semantic.get("runtime_kernel_name") != "tensor_ir_rtk"
        or semantic.get("entry_function_name") != "pypto_row_reduction_v3"
        or semantic.get("argument_packing_policy") != "PointerOnly"
        or layout.get("input_operand_count") != 1
        or layout.get("total_kernel_argument_count") != 2
        or layout.get("uniform_signature") is not False
        or grid
        != {
            "policy": "Static",
            "shape_operand_index": 0,
            "static_dimensions": list(case.grid),
            "tile_sizes": [case.row_tile],
        }
        or workspace
        != {"version": 1, "kind": "Static", "size_bytes": 0, "alignment_bytes": 1}
        or launch
        != {
            "version": 1,
            "block_dimensions": [1, 1, 1],
            "cluster_scheduling_policy": "Spread",
            "dynamic_shared_memory_bytes": 0,
            "kernel_argument_slot_bytes": 8,
        }
        or lowered_access != contract.lowered_access_projection(case)
        or build.get("schema_version") != 1
        or build.get("catalog_provenance") is not None
        or build.get("source_ir_digest") != record.get("source_ir_digest")
        or identities.get("source_ir_digest") != record.get("source_ir_digest")
        or build.get("callable_abi_digest")
        != semantic.get("kernel_abi_identity_digest")
        or identities.get("callable_abi_digest")
        != semantic.get("kernel_abi_identity_digest")
        or identities.get("kernel_build_spec_digest")
        != record.get("build_spec_sha256")
        or identities.get("compile_request_byte_identity_digest")
        != build.get("compile_request_byte_identity_digest")
        or build.get("compile_request_byte_identity_digest")
        != contract.EXPECTED_COMPILE_REQUEST_BYTE_IDENTITY_DIGEST
        or any(build.get(name) != identities.get(name) for name in digest_names)
        or producer.get("kind") != "TensorIrCudaTile"
        or producer.get("pipeline_revision") != build.get("pipeline_revision")
        or producer.get("producer_result_contract")
        != "tensorir.cuda_tile_compiled_artifact.v1"
        or producer.get("environment_overrides_enabled") is not False
        or producer.get("artifact_fallback_allowed") is not False
        or semantic.get("fallback_used") is not False
    ):
        raise ControlManifestError("row-reduction semantic ABI projection differs")


def validate_compile_anchors(root: Path, contract: ModuleType) -> dict[str, object]:
    path = root / contract.COMPILE_ANCHORS_RELATIVE_PATH
    anchors = load_canonical(
        path,
        contract.COMPILE_ANCHORS_SIZE,
        contract.COMPILE_ANCHORS_SHA256,
        "row-reduction compile anchors",
    )
    records = anchors.get("records")
    policy = anchors.get("matrix_policy")
    normalized_fields = (
        "schema_version",
        "kind",
        "pypto",
        "dso",
        "generator",
        "base_runner",
        "anchor_request",
        "cp48_report",
        "matrix_policy",
        "records",
        "records_sha256",
    )
    normalized = {name: anchors.get(name) for name in normalized_fields}
    normalized_raw = canonical_json(normalized)
    top_level_keys = {
        *normalized_fields,
        "anchor_runs",
        "normalized_records_bytes",
        "normalized_records_sha256",
    }
    policy_keys = {
        "case_count",
        "executions",
        "fresh_executable_lifetimes",
        "input_guard_elements_per_side",
        "output_guard_elements_per_side",
        "maximum_required_input_guard_elements",
        "maximum_required_output_guard_elements",
        "sentinel_words",
        "bf16_sum_accumulation",
    }
    expected_sentinel_words = {
        dtype: list(words) for dtype, words in contract.SENTINEL_WORDS.items()
    }
    if (
        set(anchors) != top_level_keys
        or anchors.get("schema_version") != 1
        or anchors.get("kind") != "pypto-row-reduction-compile-anchors-v1"
        or not isinstance(records, list)
        or len(records) != 10
        or sha256_bytes(canonical_json(records)) != anchors.get("records_sha256")
        or not isinstance(policy, dict)
        or set(policy) != policy_keys
        or policy
        != {
            "case_count": len(contract.CASE_SPECS),
            "executions": len(contract.CASE_SPECS) * contract.REPETITIONS,
            "fresh_executable_lifetimes": len(contract.CASE_SPECS)
            * contract.REPETITIONS,
            "input_guard_elements_per_side": contract.INPUT_GUARD_ELEMENTS,
            "output_guard_elements_per_side": contract.OUTPUT_GUARD_ELEMENTS,
            "maximum_required_input_guard_elements": (
                contract.MAXIMUM_REQUIRED_INPUT_GUARD_ELEMENTS
            ),
            "maximum_required_output_guard_elements": (
                contract.MAXIMUM_REQUIRED_OUTPUT_GUARD_ELEMENTS
            ),
            "sentinel_words": expected_sentinel_words,
            "bf16_sum_accumulation": (
                "bf16-input-fp32-reduce-one-rne-bf16-output"
            ),
        }
        or [record.get("case") for record in records] != list(contract.CASE_ORDER)
        or len(normalized_raw) != anchors.get("normalized_records_bytes")
        or sha256_bytes(normalized_raw) != anchors.get("normalized_records_sha256")
        or anchors.get("pypto")
        != {"head": contract.PYPTO_HEAD, "tree": contract.PYPTO_TREE, "clean": True}
        or anchors.get("dso")
        != {
            "path": contract.PYPTO_DSO_RELATIVE_PATH.as_posix(),
            "bytes": contract.PYPTO_DSO_SIZE,
            "sha256": contract.PYPTO_DSO_SHA256,
        }
        or anchors.get("generator")
        != {
            "path": contract.ANCHOR_GENERATOR_RELATIVE_PATH.as_posix(),
            "bytes": contract.ANCHOR_GENERATOR_SIZE,
            "sha256": contract.ANCHOR_GENERATOR_SHA256,
        }
        or anchors.get("base_runner")
        != {
            "path": "benchmarks/operators/pypto_fused_pointwise_sm120.py",
            "bytes": 66_999,
            "sha256": "b7960cc894834b3ba05476943e774cfc8602891faa5b9137b3d97a6aac40ab15",
        }
        or anchors.get("anchor_request")
        != {
            "path": contract.ANCHOR_REQUEST_RELATIVE_PATH.as_posix(),
            "bytes": contract.ANCHOR_REQUEST_SIZE,
            "sha256": contract.ANCHOR_REQUEST_SHA256,
            "derived_bytes": 1_583,
            "derived_sha256": "ec4141dee238e515cdbd266bc2e950e98a85717480d60c6a3fd6debd4a9e8d07",
        }
        or anchors.get("cp48_report")
        != {
            "path": contract.CP48_REPORT_RELATIVE_PATH.as_posix(),
            "bytes": contract.CP48_REPORT_SIZE,
            "sha256": contract.CP48_REPORT_SHA256,
        }
    ):
        raise ControlManifestError("row-reduction compile-anchor schema differs")
    anchor_record_keys = {
        "case",
        "dtype",
        "shape",
        "result_shape",
        "op_name",
        "row_tile",
        "grid",
        "contraction_tile",
        "contraction_chunks",
        "required_input_guard_elements",
        "required_output_guard_elements",
        "comparison",
        "repetition0_policy",
        "comparison_modes",
        "numerical_oracles",
        "max_ulp",
        "rtol",
        "atol",
        "hir_bytes",
        "hir_sha256",
        "source_bytes",
        "source_sha256",
        "build_spec_bytes",
        "build_spec_sha256",
        "artifact_bytes",
        "artifact_sha256",
        "device_code_bytes",
        "device_code_sha256",
        "source_ir_digest",
        "kernel_build_spec_digest",
        "semantic_abi",
        "cp48_case",
    }
    for item, case in zip(records, contract.CASE_SPECS, strict=True):
        item = require_exact_keys(item, anchor_record_keys, "row-reduction anchor record")
        if (
            item.get("case") != case.name
            or item.get("dtype") != case.dtype
            or item.get("shape") != list(case.shape)
            or item.get("result_shape") != list(case.result_shape)
            or item.get("op_name") != case.op_name
            or item.get("row_tile") != case.row_tile
            or item.get("grid") != list(case.grid)
            or item.get("contraction_tile") != contract.contraction_tile(case)
            or item.get("contraction_chunks") != contract.contraction_chunks(case)
            or item.get("required_input_guard_elements")
            != contract.required_input_guard_elements(case)
            or item.get("required_output_guard_elements")
            != contract.required_output_guard_elements(case)
            or item.get("comparison") != case.comparison
            or item.get("repetition0_policy") != case.repetition0_policy
            or item.get("comparison_modes")
            != [
                list(case.output_comparison_modes(repetition))
                for repetition in range(contract.REPETITIONS)
            ]
            or not isinstance(item.get("numerical_oracles"), list)
            or item.get("max_ulp") != case.max_ulp
            or item.get("rtol") != case.rtol
            or item.get("atol") != contract.REDUCTION_ATOL
            or not isinstance(item.get("semantic_abi"), dict)
        ):
            raise ControlManifestError("row-reduction anchor case projection differs")
        validate_numerical_oracles(item["numerical_oracles"], case, contract)
        validate_semantic_abi_projection(item["semantic_abi"], item, case, contract)
    cp48 = load_exact_json(
        root / contract.CP48_REPORT_RELATIVE_PATH,
        contract.CP48_REPORT_SIZE,
        contract.CP48_REPORT_SHA256,
        "accepted CP48 compiler/Cubin report",
    )
    cp48_records = {
        item["case"]: item for item in cp48.get("records", []) if isinstance(item, dict)
    }
    overlap = 0
    for item, case in zip(records, contract.CASE_SPECS, strict=True):
        if case.cp48_case is None:
            if item.get("cp48_case") is not None:
                raise ControlManifestError("row-reduction CP48 non-overlap differs")
            continue
        frozen = cp48_records.get(case.cp48_case)
        if (
            item.get("cp48_case") != case.cp48_case
            or not isinstance(frozen, dict)
            or item.get("source_ir_digest") != frozen.get("source_ir_digest")
            or item.get("device_code_bytes") != frozen.get("device_code_bytes")
            or item.get("device_code_sha256") != frozen.get("device_code_sha256")
            or item.get("grid") != frozen.get("grid")
            or item.get("row_tile") != frozen.get("row_tile")
        ):
            raise ControlManifestError("row-reduction CP48 overlap join differs")
        overlap += 1
    if overlap != 4:
        raise ControlManifestError("row-reduction CP48 overlap count differs")
    runs = anchors.get("anchor_runs")
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or runs[0]["run_id"] == runs[1]["run_id"]
    ):
        raise ControlManifestError("row-reduction anchor-run set differs")
    command = [
        "/usr/bin/env",
        "CUDA_VISIBLE_DEVICES=",
        "NVIDIA_VISIBLE_DEVICES=void",
        "PYTHONDONTWRITEBYTECODE=1",
        "envs/pypto-nvidia/bin/python",
        "-I",
        "-B",
        "-S",
        contract.ANCHOR_GENERATOR_RELATIVE_PATH.as_posix(),
    ]
    run_record_keys = {
        *normalized_fields,
        "run_id",
        "cuda_visible_devices",
        "nvidia_visible_devices",
        "torch_cuda_initialized_before_and_after",
        "replay_files",
    }
    for run in runs:
        run = require_exact_keys(
            run,
            {
                "run_id",
                "return_code",
                "cuda_visible_devices",
                "nvidia_visible_devices",
                "torch_cuda_initialized_before_and_after",
                "preflight",
                "process",
                "record",
            },
            "row-reduction anchor run",
        )
        run_id = run.get("run_id")
        if (
            not isinstance(run_id, str)
            or RUN_ID_PATTERN.fullmatch(run_id) is None
            or run.get("return_code") != 0
            or run.get("cuda_visible_devices") != ""
            or run.get("nvidia_visible_devices") != "void"
            or run.get("torch_cuda_initialized_before_and_after") is not False
        ):
            raise ControlManifestError("row-reduction anchor-run identity differs")
        documents: dict[str, dict[str, object]] = {}
        for name in ("preflight", "process", "record"):
            item = run.get(name)
            expected_keys = {"path", "bytes", "sha256", "mode"}
            if name == "record":
                expected_keys.add("records_sha256")
            item = require_exact_keys(
                item, expected_keys, "row-reduction anchor sidecar"
            )
            expected_path = root / "runs" / run_id / (
                "row-reduction-compile-anchor-record.json"
                if name == "record"
                else f"{name}.json"
            )
            if item.get("path") != expected_path.relative_to(root).as_posix():
                raise ControlManifestError("row-reduction anchor sidecar path differs")
            documents[name] = load_canonical(
                expected_path,
                int(item["bytes"]),
                str(item["sha256"]),
                f"row-reduction anchor {name}",
            )
            if stat.S_IMODE(expected_path.stat().st_mode) != item["mode"]:
                raise ControlManifestError("row-reduction anchor sidecar mode differs")
        preflight = documents["preflight"]
        process = documents["process"]
        record = documents["record"]
        if (
            preflight.get("ok") is not True
            or preflight.get("failures") != []
            or preflight.get("mode") != "heavy"
            or preflight.get("nvidia_compute_pids") != []
            or preflight.get("nvidia_compute_audit_ok") is not True
            or preflight.get("protected_nvidia_compute_pids") != []
            or preflight.get("protected_nvidia_runtime_mapping_pids") != []
            or preflight.get("unreadable_protected_maps") != []
            or preflight.get("protected_zero_nvidia_gpu_smoke_requested") is not False
            or preflight.get("torch", {}).get("cuda_available") is not False
            or process.get("run_id") != run_id
            or process.get("workspace") != str(root)
            or process.get("environment") != str(root / "envs/pypto-nvidia")
            or process.get("mode") != "heavy"
            or process.get("status") != "exited"
            or process.get("return_code") != 0
            or process.get("command") != command
            or process.get("preflight")
            != {
                "path": str(root / "runs" / run_id / "preflight.json"),
                "sha256": run["preflight"]["sha256"],
            }
            or process.get("gpu_smoke", {}).get("requested") is not False
            or process.get("gpu_smoke", {}).get("authorization") is not None
            or set(record) != run_record_keys
            or record.get("run_id") != run_id
            or record.get("cuda_visible_devices") != ""
            or record.get("nvidia_visible_devices") != "void"
            or record.get("torch_cuda_initialized_before_and_after") is not False
            or {name: record.get(name) for name in normalized_fields} != normalized
            or record.get("records_sha256") != run["record"]["records_sha256"]
        ):
            raise ControlManifestError("row-reduction anchor run joins differ")
        validate_anchor_replay_closure(root, run_id, record, contract)
    return {
        "path": contract.COMPILE_ANCHORS_RELATIVE_PATH.as_posix(),
        "bytes": contract.COMPILE_ANCHORS_SIZE,
        "sha256": contract.COMPILE_ANCHORS_SHA256,
        "records_sha256": anchors["records_sha256"],
        "normalized_records_sha256": anchors["normalized_records_sha256"],
        "anchor_run_ids": [run["run_id"] for run in runs],
    }


def validate_control_manifest(workspace: Path) -> dict[str, object]:
    root = workspace.resolve(strict=True)
    if workspace.absolute() != root:
        raise ControlManifestError("workspace contains a symlinked path")
    reject_control_bytecode_cache(root)
    base_manifest = root / BASE_ADMISSION_MANIFEST_RELATIVE_PATH
    if (
        base_manifest.is_symlink()
        or not base_manifest.is_file()
        or base_manifest.stat().st_size != BASE_ADMISSION_MANIFEST_SIZE
        or sha256_file(base_manifest) != BASE_ADMISSION_MANIFEST_SHA256
    ):
        raise ControlManifestError("accepted policy-2 admission manifest differs")
    base_identity = base_admission.validate_control_manifest(root)
    contract = load_exact(
        "_pypto_row_reduction_sm120_contract_for_control",
        root / "tools/_pypto_row_reduction_sm120_contract.py",
        (root / "tools/_pypto_row_reduction_sm120_contract.py").stat().st_size,
        sha256_file(root / "tools/_pypto_row_reduction_sm120_contract.py"),
    )
    load_exact_json(
        root / contract.CP48_REPORT_RELATIVE_PATH,
        contract.CP48_REPORT_SIZE,
        contract.CP48_REPORT_SHA256,
        "accepted CP48 compiler/Cubin report",
    )
    anchors = validate_compile_anchors(root, contract)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControlManifestError("reviewed row-reduction control manifest is missing")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    if not isinstance(manifest, dict) or canonical_json(manifest) != raw:
        raise ControlManifestError(
            "row-reduction control manifest is not canonical JSON"
        )
    if set(manifest) != {
        "schema_version",
        "kind",
        "implementation_commit",
        "implementation_tree",
        "base_admission_manifest_sha256",
        "cp48_report_sha256",
        "files",
    }:
        raise ControlManifestError("row-reduction control-manifest schema differs")
    commit = manifest.get("implementation_commit")
    tree = manifest.get("implementation_tree")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("base_admission_manifest_sha256")
        != BASE_ADMISSION_MANIFEST_SHA256
        or manifest.get("cp48_report_sha256") != contract.CP48_REPORT_SHA256
        or not isinstance(commit, str)
        or COMMIT_PATTERN.fullmatch(commit) is None
        or not isinstance(tree, str)
        or COMMIT_PATTERN.fullmatch(tree) is None
        or str(git(root, "rev-parse", f"{commit}^{{tree}}")).strip() != tree
    ):
        raise ControlManifestError("row-reduction implementation identity differs")
    current_head = str(git(root, "rev-parse", "HEAD")).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current_head],
        cwd=root,
        check=False,
    ).returncode:
        raise ControlManifestError("row-reduction implementation is not an ancestor")
    if str(git(root, "status", "--porcelain=v1", "--untracked-files=all")):
        raise ControlManifestError("root control repository is not clean")
    if subprocess.run(
        ["git", "diff", "--quiet", f"{commit}..{current_head}", "--", *CONTROL_PATHS],
        cwd=root,
        check=False,
    ).returncode:
        raise ControlManifestError(
            "row-reduction controls changed after implementation"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(CONTROL_PATHS):
        raise ControlManifestError("row-reduction control file set differs")
    normalized: list[dict[str, object]] = []
    for record, expected_path in zip(files, CONTROL_PATHS, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ControlManifestError("row-reduction control record is malformed")
        path = root / expected_path
        if (
            record.get("path") != expected_path
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or stat.S_IMODE(path.stat().st_mode) != record.get("mode")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ControlManifestError(
                f"live row-reduction control differs: {expected_path}"
            )
        committed = git(root, "show", f"{commit}:{expected_path}", text=False)
        assert isinstance(committed, bytes)
        if (
            len(committed) != record["bytes"]
            or sha256_bytes(committed) != record["sha256"]
        ):
            raise ControlManifestError(
                f"committed row control differs: {expected_path}"
            )
        normalized.append(dict(record))
    return {
        "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_bytes": len(raw),
        "manifest_sha256": sha256_bytes(raw),
        "implementation_commit": commit,
        "implementation_tree": tree,
        "current_head": current_head,
        "current_tree": str(git(root, "rev-parse", "HEAD^{tree}")).strip(),
        "root_clean": True,
        "base_admission": base_identity,
        "compile_anchors": anchors,
        "files": normalized,
    }
