#!/usr/bin/env python3
"""CUDA-hidden, no-replace generator for fused-pointwise compile anchors."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r"pypto-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9a-f]{6}")
RECORD_NAME = "fused-pointwise-compile-anchor-record.json"


class AnchorError(RuntimeError):
    """The fixed CPU/compiler anchor transaction differs."""


def load_exact(name: str, path: Path) -> ModuleType:
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise AnchorError(f"exact anchor source is noncanonical: {path}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = hashlib.sha256(raw).hexdigest()
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> tuple[Path, str]:
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise AnchorError("anchor generator requires Python -I -B -S")
    run_id = os.environ.get("PYPTO_RUN_ID", "")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise AnchorError("anchor generator requires one isolated run ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AnchorError("anchor generator requires CUDA_VISIBLE_DEVICES empty")
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void":
        raise AnchorError("anchor generator requires NVIDIA_VISIBLE_DEVICES=void")
    if {"torch", "pypto", "triton", "sglang", "flashinfer"} & set(sys.modules):
        raise AnchorError("framework module loaded before exact anchor bootstrap")

    control = load_exact(
        "_pypto_fused_pointwise_sm120_control_manifest",
        ROOT / "tools/_pypto_fused_pointwise_sm120_control_manifest.py",
    )
    control.reject_control_bytecode_cache(ROOT)
    contract = load_exact(
        "_pypto_fused_pointwise_sm120_contract",
        ROOT / "tools/_pypto_fused_pointwise_sm120_contract.py",
    )
    runner = load_exact(
        "fused_pointwise_anchor_runner", ROOT / contract.RUNNER_RELATIVE_PATH
    )
    runner_path = (ROOT / contract.RUNNER_RELATIVE_PATH).resolve(strict=True)
    if (
        runner_path.stat().st_size != contract.RUNNER_SIZE
        or sha256_file(runner_path) != contract.RUNNER_SHA256
    ):
        raise AnchorError("live fused-pointwise runner identity differs")
    generator = Path(__file__).resolve(strict=True)
    if (
        generator != ROOT / contract.ANCHOR_GENERATOR_RELATIVE_PATH
        or generator.stat().st_size != contract.ANCHOR_GENERATOR_SIZE
        or sha256_file(generator) != contract.ANCHOR_GENERATOR_SHA256
    ):
        raise AnchorError("live anchor-generator identity differs")
    dso = (ROOT / contract.PYPTO_DSO_RELATIVE_PATH).resolve(strict=True)
    if (
        dso.stat().st_size != contract.PYPTO_DSO_SIZE
        or sha256_file(dso) != contract.PYPTO_DSO_SHA256
    ):
        raise AnchorError("exact PyPTO DSO bytes differ")
    request_path = (ROOT / contract.ANCHOR_REQUEST_RELATIVE_PATH).resolve(strict=True)
    if (
        request_path.stat().st_size != contract.ANCHOR_REQUEST_SIZE
        or sha256_file(request_path) != contract.ANCHOR_REQUEST_SHA256
    ):
        raise AnchorError("fixed CPU anchor CompileRequest bytes differ")

    site = ROOT / "envs/pypto-nvidia/lib/python3.14/site-packages"
    sys.path.insert(0, str(site))
    runner.validate_pypto_python_source(ROOT)
    pypto = runner.bootstrap_exact_pypto(ROOT, dso.parent)
    from pypto import compiler
    from pypto.pypto_core import ir
    import torch

    if torch.cuda.is_initialized():
        raise AnchorError("Torch CUDA initialized before compiler anchors")
    info = compiler.get_nvidia_backend_build_info()
    if (
        not info.compiled
        or not info.compiler_factory_available
        or info.pypto_revision != contract.PYPTO_HEAD
        or info.tensor_ir_revision != contract.TENSOR_IR_HEAD
        or info.cuda_tile_revision != contract.CUDA_TILE_HEAD
        or info.llvm_revision != contract.LLVM_HEAD
    ):
        raise AnchorError("exact PyPTO compiler identity differs")
    old_request = compiler.CompileRequest.deserialize(request_path.read_bytes())
    request = compiler.CompileRequest(
        old_request.target_info, runner.toolchain_identity(compiler, info)
    )
    replay = ROOT / "runs" / run_id / "fused-pointwise-compile-anchor-replay"
    replay.mkdir(mode=0o700, exist_ok=False)
    replay_files: list[dict[str, object]] = []

    def replay_file(name: str, payload: bytes) -> None:
        path = replay / name
        digest = runner.publish_no_replace(path, payload)
        replay_files.append(
            {
                "bytes": len(payload),
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": digest,
            }
        )

    replay_file("compile-request.msgpack", request.serialize())
    records: list[dict[str, object]] = []
    for case in contract.CASE_SPECS:
        program = runner.make_program(pypto, ir, case)
        hir = bytes(ir.serialize(program))
        restored = ir.deserialize(hir)
        if (
            len(hir) != case.expected_hir_bytes
            or runner.sha256_bytes(hir) != case.expected_hir_sha256
            or bytes(ir.serialize(restored)) != hir
            or not ir.structural_equal(program, restored, enable_auto_mapping=True)
        ):
            raise AnchorError(f"{case.name} HIR anchor differs")
        source = runner.canonical_tensor_ir_source(case)
        result = compiler.compile_structured_strict(
            restored, request, runner.schedule(compiler, case.tile_sizes)
        )
        validated = runner.validate_structured_result(
            compiler, result, request, case, source
        )
        record = validated["record"]
        build_spec = validated["build_spec"]
        artifact = validated["artifact"]
        kernel = artifact.kernel_abi
        replay_file(f"{case.name}.hir.msgpack", hir)
        replay_file(f"{case.name}.source.mlir", source)
        replay_file(f"{case.name}.build-spec.msgpack", build_spec.serialize())
        replay_file(f"{case.name}.artifact.msgpack", artifact.serialize())
        replay_file(f"{case.name}.cubin", bytes(artifact.device_code))
        records.append(
            {
                "argument_abi_digest": record["argument_abi_digest"],
                "artifact_identity_digest": artifact.identity_digest,
                "assignment_count": case.assignment_count,
                "build_spec_identity_digest": build_spec.identity_digest,
                "callable_abi_digest": record["callable_abi_digest"],
                "case": case.name,
                "device_code_bytes": record["device_code_bytes"],
                "device_code_sha256": record["device_code_sha256"],
                "dtype": case.dtype,
                "entry_function_name": kernel.entry_function_name,
                "expected_grid": list(case.expected_grid),
                "fallback_used": artifact.fallback_used,
                "high_precision": False,
                "hir_bytes": len(hir),
                "hir_sha256": runner.sha256_bytes(hir),
                "input_count": case.input_count,
                "kernel_argument_count": case.expected_kernel_arguments,
                "module_kind": "FusedPointwiseV2",
                "mutation_abi_digest": record["mutation_abi_digest"],
                "operator_sequence": list(case.operator_sequence),
                "pointer_only": True,
                "projection_schema_version": 2,
                "result_abi_digest": record["result_abi_digest"],
                "result_count": 1,
                "scalar_literals": list(case.scalar_literals),
                "shape": list(case.shape),
                "source_ir_bytes": len(source),
                "source_ir_digest": record["source_ir_digest"],
                "static_specialization_digest": record["static_specialization_digest"],
                "strides": list(case.strides),
                "symbolic_specialization_digest": record[
                    "symbolic_specialization_digest"
                ],
                "tile_sizes": list(case.tile_sizes),
                "workspace_bytes": kernel.workspace_abi.size_bytes,
            }
        )
    if torch.cuda.is_initialized():
        raise AnchorError("Torch CUDA initialized during compiler anchors")
    if {"triton", "sglang", "flashinfer"} & set(sys.modules):
        raise AnchorError("forbidden provider imported during compiler anchors")
    pypto_identity = runner.git_identity(ROOT / "projects/pypto")
    if pypto_identity != {
        "head": contract.PYPTO_HEAD,
        "tree": contract.PYPTO_TREE,
        "clean": True,
    }:
        raise AnchorError("PyPTO source identity differs")
    records_bytes = runner.canonical_json(records)
    document = {
        "cuda_visible_devices": "",
        "dso": {
            "bytes": contract.PYPTO_DSO_SIZE,
            "path": contract.PYPTO_DSO_RELATIVE_PATH.as_posix(),
            "sha256": contract.PYPTO_DSO_SHA256,
        },
        "generator": {
            "bytes": contract.ANCHOR_GENERATOR_SIZE,
            "path": contract.ANCHOR_GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": contract.ANCHOR_GENERATOR_SHA256,
        },
        "kind": "pypto-fused-pointwise-compile-anchor-run-v1",
        "nvidia_visible_devices": "void",
        "pypto": pypto_identity,
        "records": records,
        "records_sha256": runner.sha256_bytes(records_bytes),
        "request": {
            "bytes": contract.ANCHOR_REQUEST_SIZE,
            "path": contract.ANCHOR_REQUEST_RELATIVE_PATH.as_posix(),
            "sha256": contract.ANCHOR_REQUEST_SHA256,
        },
        "replay_files": replay_files,
        "runner": {
            "bytes": contract.RUNNER_SIZE,
            "path": contract.RUNNER_RELATIVE_PATH.as_posix(),
            "sha256": contract.RUNNER_SHA256,
        },
        "run_id": run_id,
        "schema_version": 1,
        "torch_cuda_initialized_before_and_after": False,
        "toolchain": {
            "cuda_tile": info.cuda_tile_revision,
            "llvm": info.llvm_revision,
            "pypto": info.pypto_revision,
            "tensor_ir": info.tensor_ir_revision,
        },
    }
    output = ROOT / "runs" / run_id / RECORD_NAME
    digest = runner.publish_no_replace(output, runner.canonical_json(document))
    return output, digest


def main() -> int:
    output, digest = run()
    print(json.dumps({"path": str(output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
