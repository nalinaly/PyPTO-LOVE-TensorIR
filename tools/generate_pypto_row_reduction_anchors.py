#!/usr/bin/env python3
"""Generate no-replace CUDA-hidden compiler anchors for the row-reduction gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_fused_pointwise_sm120.py")
BASE_RUNNER_SIZE = 66_999
BASE_RUNNER_SHA256 = "b7960cc894834b3ba05476943e774cfc8602891faa5b9137b3d97a6aac40ab15"
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
RUN_RECORD_NAME = "row-reduction-compile-anchor-record.json"


class AnchorError(RuntimeError):
    """The fixed row-reduction compiler anchor transaction is invalid."""


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
    name: str, path: Path, size: int | None = None, digest: str | None = None
) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise AnchorError(f"exact anchor source is noncanonical: {path}")
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if size is not None and len(raw) != size:
        raise AnchorError(f"exact anchor source size differs: {path}")
    if digest is not None and actual != digest:
        raise AnchorError(f"exact anchor source hash differs: {path}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = actual
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def publish_no_replace(path: Path, payload: bytes, mode: int) -> str:
    if path.exists() or path.is_symlink():
        raise AnchorError(f"refusing to replace compiler anchors: {path}")
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
        os.link(temporary, path)
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(payload)


contract = load_exact(
    "_pypto_row_reduction_sm120_contract",
    ROOT / "tools/_pypto_row_reduction_sm120_contract.py",
)
base = load_exact(
    "_pypto_row_reduction_anchor_base",
    ROOT / BASE_RUNNER_RELATIVE_PATH,
    BASE_RUNNER_SIZE,
    BASE_RUNNER_SHA256,
)


def git_identity(repository: Path) -> dict[str, object]:
    return base.git_identity(repository)


def load_canonical(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise AnchorError(f"{description} is not a canonical regular file")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise AnchorError(f"{description} is not canonical JSON")
    return value, raw


def compile_run() -> int:
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise AnchorError("row-reduction anchor generator requires Python -I -B -S")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"
    ):
        raise AnchorError("row-reduction anchors require hidden CUDA devices")
    run_id = os.environ.get("PYPTO_RUN_ID", "")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise AnchorError("row-reduction anchor run ID is missing or malformed")
    pypto_identity = git_identity(ROOT / "projects/pypto")
    if pypto_identity != {
        "head": contract.PYPTO_HEAD,
        "tree": contract.PYPTO_TREE,
        "clean": True,
    }:
        raise AnchorError("PyPTO source identity differs")
    dso = (ROOT / contract.PYPTO_DSO_RELATIVE_PATH).resolve(strict=True)
    if (
        dso.stat().st_size != contract.PYPTO_DSO_SIZE
        or sha256_file(dso) != contract.PYPTO_DSO_SHA256
    ):
        raise AnchorError("row-reduction ON DSO differs")
    request_path = (ROOT / contract.ANCHOR_REQUEST_RELATIVE_PATH).resolve(strict=True)
    request_raw = request_path.read_bytes()
    if (
        len(request_raw) != contract.ANCHOR_REQUEST_SIZE
        or sha256_bytes(request_raw) != contract.ANCHOR_REQUEST_SHA256
    ):
        raise AnchorError("anchor CompileRequest differs")
    cp48_path = (ROOT / contract.CP48_REPORT_RELATIVE_PATH).resolve(strict=True)
    cp48_raw = cp48_path.read_bytes()
    if (
        len(cp48_raw) != contract.CP48_REPORT_SIZE
        or sha256_bytes(cp48_raw) != contract.CP48_REPORT_SHA256
    ):
        raise AnchorError("CP48 compiler/Cubin report differs")
    cp48 = json.loads(cp48_raw)
    cp48_records = {record["case"]: record for record in cp48["records"]}

    site = ROOT / "envs/pypto-nvidia/lib/python3.14/site-packages"
    if site.is_symlink() or not site.is_dir() or site.resolve(strict=True) != site:
        raise AnchorError("selected site-packages path is noncanonical")
    sys.path.insert(0, str(site))
    pypto = base.bootstrap_exact_pypto(ROOT, dso.parent)
    import torch
    from pypto import compiler

    if torch.cuda.is_initialized():
        raise AnchorError("Torch CUDA initialized before row-reduction compilation")
    info = compiler.get_nvidia_backend_build_info()
    if (
        not info.compiled
        or not info.compiler_factory_available
        or info.pypto_revision != contract.PYPTO_HEAD
        or info.tensor_ir_revision != contract.TENSOR_IR_HEAD
        or info.cuda_tile_revision != contract.CUDA_TILE_HEAD
        or info.llvm_revision != contract.LLVM_HEAD
    ):
        raise AnchorError("row-reduction DSO compiler identity differs")
    old_request = compiler.CompileRequest.deserialize(request_raw)
    request = compiler.CompileRequest(
        old_request.target_info, base.toolchain_identity(compiler, info)
    )
    derived_request = request.serialize()
    replay = ROOT / "runs" / run_id / "row-reduction-compile-anchor-replay"
    replay.mkdir(mode=0o700, parents=True, exist_ok=False)
    replay_files: list[dict[str, object]] = []

    def replay_file(name: str, payload: bytes) -> None:
        path = replay / name
        digest = publish_no_replace(path, payload, 0o444)
        replay_files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(payload),
                "sha256": digest,
            }
        )

    replay_file("compile-request.msgpack", derived_request)
    records: list[dict[str, object]] = []
    for case in contract.CASE_SPECS:
        program = contract.make_program(pypto, pypto.ir, case)
        hir = bytes(pypto.ir.serialize(program))
        restored = pypto.ir.deserialize(hir)
        if bytes(pypto.ir.serialize(restored)) != hir:
            raise AnchorError(f"{case.name} HIR round-trip differs")
        source = contract.canonical_tensor_ir_source(case)
        result = compiler.compile_structured_strict(
            restored, request, contract.schedule(compiler, case.row_tile)
        )
        build_raw = result.build_spec.serialize()
        artifact_object = result.artifact
        artifact_raw = artifact_object.serialize()
        kernel = artifact_object.kernel_abi
        descriptors = list(kernel.argument_layout.operand_descriptors)
        device_code = bytes(artifact_object.device_code)
        source_digest = sha256_bytes(source)
        build_digest = sha256_bytes(build_raw)
        if (
            artifact_object.fallback_used
            or artifact_object.actual_target.compute_capability != 120
            or result.build_spec.semantic_route
            != compiler.SemanticRoute.StructuredTensorIr
            or kernel.entry_function_name != "pypto_row_reduction_v3"
            or kernel.argument_layout.input_operand_count != 1
            or kernel.argument_layout.total_kernel_argument_count != 2
            or tuple(kernel.grid_abi.static_dimensions) != case.grid
            or list(kernel.grid_abi.tile_sizes) != [case.row_tile]
            or kernel.grid_abi.shape_operand_index != 0
            or kernel.workspace_abi.size_bytes != 0
            or len(descriptors) != 2
            or list(descriptors[0].shape) != list(case.shape)
            or list(descriptors[1].shape) != list(case.result_shape)
            or compiler.Artifact.compute_source_ir_digest(source) != source_digest
            or result.build_spec.source_ir_digest != source_digest
            or artifact_object.identities.source_ir_digest != source_digest
            or result.build_spec.identity_digest != build_digest
            or artifact_object.identities.kernel_build_spec_digest != build_digest
            or hashlib.sha256(device_code).hexdigest()
            != artifact_object.device_code_sha256
            or type(result.build_spec).deserialize(build_raw).serialize() != build_raw
            or compiler.Artifact.deserialize(
                artifact_raw, request, result.build_spec
            ).serialize()
            != artifact_raw
        ):
            raise AnchorError(f"{case.name} Artifact ABI differs")
        replay_file(f"{case.name}.hir.msgpack", hir)
        replay_file(f"{case.name}.source.mlir", source)
        replay_file(f"{case.name}.build-spec.msgpack", build_raw)
        replay_file(f"{case.name}.artifact.msgpack", artifact_raw)
        replay_file(f"{case.name}.cubin", device_code)
        record = {
            "case": case.name,
            "dtype": case.dtype,
            "shape": list(case.shape),
            "result_shape": list(case.result_shape),
            "op_name": case.op_name,
            "row_tile": case.row_tile,
            "grid": list(case.grid),
            "contraction_tile": contract.contraction_tile(case),
            "contraction_chunks": contract.contraction_chunks(case),
            "required_input_guard_elements": contract.required_input_guard_elements(
                case
            ),
            "required_output_guard_elements": contract.required_output_guard_elements(
                case
            ),
            "comparison": case.comparison,
            "max_ulp": case.max_ulp,
            "rtol": case.rtol,
            "atol": contract.REDUCTION_ATOL,
            "hir_bytes": len(hir),
            "hir_sha256": sha256_bytes(hir),
            "source_bytes": len(source),
            "source_sha256": source_digest,
            "build_spec_bytes": len(build_raw),
            "build_spec_sha256": build_digest,
            "artifact_bytes": len(artifact_raw),
            "artifact_sha256": sha256_bytes(artifact_raw),
            "device_code_bytes": len(device_code),
            "device_code_sha256": sha256_bytes(device_code),
            "source_ir_digest": artifact_object.identities.source_ir_digest,
            "kernel_build_spec_digest": (
                artifact_object.identities.kernel_build_spec_digest
            ),
            "cp48_case": case.cp48_case,
        }
        if case.cp48_case is not None:
            frozen = cp48_records[case.cp48_case]
            if (
                record["source_ir_digest"] != frozen["source_ir_digest"]
                or record["device_code_sha256"] != frozen["device_code_sha256"]
                or record["device_code_bytes"] != frozen["device_code_bytes"]
                or record["grid"] != frozen["grid"]
                or record["row_tile"] != frozen["row_tile"]
            ):
                raise AnchorError(f"{case.name} CP48 source/Cubin join differs")
        records.append(record)
    if torch.cuda.is_initialized():
        raise AnchorError("Torch CUDA initialized during row-reduction compilation")
    generator = Path(__file__).resolve(strict=True)
    output = {
        "schema_version": 1,
        "kind": "pypto-row-reduction-compile-anchors-v1",
        "run_id": run_id,
        "cuda_visible_devices": "",
        "nvidia_visible_devices": "void",
        "torch_cuda_initialized_before_and_after": False,
        "pypto": pypto_identity,
        "dso": {
            "path": contract.PYPTO_DSO_RELATIVE_PATH.as_posix(),
            "bytes": contract.PYPTO_DSO_SIZE,
            "sha256": contract.PYPTO_DSO_SHA256,
        },
        "generator": {
            "path": generator.relative_to(ROOT).as_posix(),
            "bytes": generator.stat().st_size,
            "sha256": sha256_file(generator),
        },
        "base_runner": {
            "path": BASE_RUNNER_RELATIVE_PATH.as_posix(),
            "bytes": BASE_RUNNER_SIZE,
            "sha256": BASE_RUNNER_SHA256,
        },
        "anchor_request": {
            "path": contract.ANCHOR_REQUEST_RELATIVE_PATH.as_posix(),
            "bytes": contract.ANCHOR_REQUEST_SIZE,
            "sha256": contract.ANCHOR_REQUEST_SHA256,
            "derived_bytes": len(derived_request),
            "derived_sha256": sha256_bytes(derived_request),
        },
        "cp48_report": {
            "path": contract.CP48_REPORT_RELATIVE_PATH.as_posix(),
            "bytes": contract.CP48_REPORT_SIZE,
            "sha256": contract.CP48_REPORT_SHA256,
        },
        "matrix_policy": {
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
            "sentinel_words": contract.SENTINEL_WORDS,
            "bf16_sum_accumulation": "bf16-input-fp32-reduce-one-rne-bf16-output",
        },
        "records": records,
        "records_sha256": sha256_bytes(canonical_json(records)),
        "replay_files": replay_files,
    }
    payload = canonical_json(output)
    target = ROOT / "runs" / run_id / RUN_RECORD_NAME
    digest = publish_no_replace(target, payload, 0o444)
    print(json.dumps({"path": str(target), "bytes": len(payload), "sha256": digest}))
    return 0


def publish_runs(run_ids: list[str]) -> int:
    if len(run_ids) != 2 or run_ids[0] == run_ids[1]:
        raise AnchorError("compile-anchor publication requires two distinct runs")
    anchor_runs: list[dict[str, object]] = []
    documents: list[dict[str, object]] = []
    for run_id in run_ids:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise AnchorError("compile-anchor publication run ID is malformed")
        run_dir = ROOT / "runs" / run_id
        preflight_path = run_dir / "preflight.json"
        process_path = run_dir / "process.json"
        record_path = run_dir / RUN_RECORD_NAME
        preflight, preflight_raw = load_canonical(preflight_path, "anchor preflight")
        process, process_raw = load_canonical(process_path, "anchor process")
        record, record_raw = load_canonical(record_path, "anchor record")
        if (
            preflight.get("ok") is not True
            or preflight.get("failures") != []
            or preflight.get("mode") not in {"heavy", "light"}
            or preflight.get("nvidia_compute_pids") != []
            or process.get("run_id") != run_id
            or process.get("status") != "exited"
            or process.get("return_code") != 0
            or process.get("preflight")
            != {"path": str(preflight_path), "sha256": sha256_bytes(preflight_raw)}
            or not isinstance(process.get("command"), list)
            or contract.ANCHOR_GENERATOR_RELATIVE_PATH.as_posix()
            not in process["command"]
            or record.get("run_id") != run_id
            or record.get("cuda_visible_devices") != ""
            or record.get("nvidia_visible_devices") != "void"
            or record.get("torch_cuda_initialized_before_and_after") is not False
        ):
            raise AnchorError("compile-anchor run/sidecar identity differs")
        for path, mode in (
            (preflight_path, 0o600),
            (process_path, 0o600),
            (record_path, 0o444),
        ):
            if stat.S_IMODE(path.stat().st_mode) != mode:
                raise AnchorError("compile-anchor sidecar mode differs")
        replay_records = record.get("replay_files")
        if not isinstance(replay_records, list) or len(replay_records) != 51:
            raise AnchorError("compile-anchor replay set differs")
        for replay_record in replay_records:
            if not isinstance(replay_record, dict) or set(replay_record) != {
                "path",
                "bytes",
                "sha256",
            }:
                raise AnchorError("compile-anchor replay record is malformed")
            path = ROOT / str(replay_record["path"])
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != replay_record["bytes"]
                or stat.S_IMODE(path.stat().st_mode) != 0o444
                or sha256_file(path) != replay_record["sha256"]
            ):
                raise AnchorError("compile-anchor replay bytes differ")
        anchor_runs.append(
            {
                "run_id": run_id,
                "return_code": 0,
                "cuda_visible_devices": "",
                "nvidia_visible_devices": "void",
                "torch_cuda_initialized_before_and_after": False,
                "preflight": {
                    "path": preflight_path.relative_to(ROOT).as_posix(),
                    "bytes": len(preflight_raw),
                    "sha256": sha256_bytes(preflight_raw),
                    "mode": 0o600,
                },
                "process": {
                    "path": process_path.relative_to(ROOT).as_posix(),
                    "bytes": len(process_raw),
                    "sha256": sha256_bytes(process_raw),
                    "mode": 0o600,
                },
                "record": {
                    "path": record_path.relative_to(ROOT).as_posix(),
                    "bytes": len(record_raw),
                    "sha256": sha256_bytes(record_raw),
                    "mode": 0o444,
                    "records_sha256": record["records_sha256"],
                },
            }
        )
        documents.append(record)
    first, second = documents
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
    normalized_first = {name: first[name] for name in normalized_fields}
    normalized_second = {name: second[name] for name in normalized_fields}
    normalized_raw = canonical_json(normalized_first)
    if (
        normalized_first != normalized_second
        or canonical_json(normalized_second) != normalized_raw
    ):
        raise AnchorError("independent compile-anchor normalized records differ")
    manifest = {
        **normalized_first,
        "kind": "pypto-row-reduction-compile-anchors-v1",
        "anchor_runs": anchor_runs,
        "normalized_records_bytes": len(normalized_raw),
        "normalized_records_sha256": sha256_bytes(normalized_raw),
    }
    payload = canonical_json(manifest)
    target = ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = publish_no_replace(target, payload, 0o644)
    print(json.dumps({"path": str(target), "bytes": len(payload), "sha256": digest}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-runs", nargs=2)
    args = parser.parse_args()
    if args.publish_runs is not None:
        return publish_runs(args.publish_runs)
    return compile_run()


if __name__ == "__main__":
    raise SystemExit(main())
