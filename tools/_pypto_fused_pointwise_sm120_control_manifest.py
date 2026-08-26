#!/usr/bin/env python3
"""Validate the reviewed v1 fused-pointwise-SM120 root-control manifest."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "pypto-fused-pointwise-sm120-controls-v1"
MANIFEST_RELATIVE_PATH = Path("state/contracts/pypto_fused_pointwise_sm120_v1.json")
PARENT_MANIFEST_RELATIVE_PATH = Path(
    "state/contracts/pypto_nvidia_executable_sm120_v4.json"
)
PARENT_MANIFEST_SIZE = 1_569
PARENT_MANIFEST_SHA256 = (
    "a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf"
)
LEGACY_MANIFEST_RELATIVE_PATH = Path(
    "state/contracts/pypto_frontend_vector_add_sm120_v1.json"
)
LEGACY_MANIFEST_SIZE = 1_776
LEGACY_MANIFEST_SHA256 = (
    "f16c4fbac14f4ec4d2a26fef9df3c4e7d1d3c412fbaf3c48f200a61a118d8eed"
)
LEGACY_REPORT_RELATIVE_PATH = Path(
    "reports/data/pypto-frontend-vector-add-sm120-"
    "pypto-20260825T145519Z-1142938-70ac73.json"
)
LEGACY_REPORT_SIZE = 32_676
LEGACY_REPORT_MODE = 0o444
LEGACY_REPORT_SHA256 = (
    "8dbbfbf3ed791cc38d552fbd8f37e34f60d1d0262e9626195ab084b530f228e8"
)
CONTROL_PATHS = (
    "benchmarks/operators/pypto_fused_pointwise_sm120.py",
    "tools/_pypto_fused_pointwise_sm120_contract.py",
    "tools/generate_pypto_fused_pointwise_anchors.py",
    "state/contracts/pypto_fused_pointwise_compile_anchors_v1.json",
    "tools/_pypto_fused_pointwise_sm120_control_manifest.py",
    "tools/run_pypto_fused_pointwise_sm120_isolated.py",
    "tools/finalize_pypto_fused_pointwise_sm120.py",
    "tools/preflight.py",
    "tools/run_isolated.py",
    "tools/stop_run.py",
)
NEW_PYTHON_SOURCE_PATHS = (
    "benchmarks/operators/pypto_fused_pointwise_sm120.py",
    "tools/_pypto_fused_pointwise_sm120_contract.py",
    "tools/_pypto_fused_pointwise_sm120_control_manifest.py",
    "tools/generate_pypto_fused_pointwise_anchors.py",
    "tools/run_pypto_fused_pointwise_sm120_isolated.py",
    "tools/finalize_pypto_fused_pointwise_sm120.py",
    "tests/test_pypto_fused_pointwise_sm120.py",
)
COMPILE_ANCHORS_RELATIVE_PATH = Path(
    "state/contracts/pypto_fused_pointwise_compile_anchors_v1.json"
)
COMPILE_ANCHORS_SIZE = 21_490
COMPILE_ANCHORS_SHA256 = (
    "584f6755bbd248de5bb6ddd3ff610da8082667bc892a6cff6583ea42d4c44c97"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class ControlManifestError(RuntimeError):
    """The checked-out smoke controls are not the reviewed implementation."""


def control_bytecode_cache_entries(root: Path) -> list[str]:
    """Return matching bytecode entries without importing any dependency."""

    entries: set[str] = set()
    for relative in NEW_PYTHON_SOURCE_PATHS:
        source = root / relative
        for suffix in (".pyc", ".pyo"):
            direct = source.with_suffix(suffix)
            if direct.exists() or direct.is_symlink():
                entries.add(direct.relative_to(root).as_posix())
        cache = source.parent / "__pycache__"
        if not cache.is_dir():
            continue
        stem = source.stem
        for candidate in cache.iterdir():
            if candidate.name == f"{stem}.pyc" or (
                candidate.name.startswith(f"{stem}.")
                and candidate.name.endswith((".pyc", ".pyo"))
            ):
                entries.add(candidate.relative_to(root).as_posix())
    return sorted(entries)


def reject_control_bytecode_cache(root: Path) -> None:
    entries = control_bytecode_cache_entries(root)
    if entries:
        raise ControlManifestError(
            "fused-pointwise control bytecode/cache entries are forbidden: "
            + ", ".join(entries)
        )


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
            raise ControlManifestError(f"duplicate control-manifest key: {key}")
        output[key] = value
    return output


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(repository: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=text,
        capture_output=True,
    )
    return result.stdout


def validate_parent_control_manifest(root: Path) -> dict[str, object]:
    """Bind the accepted v4 transaction used as this lane's primitive."""

    path = root / PARENT_MANIFEST_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise ControlManifestError("accepted v4 parent control manifest is missing")
    raw = path.read_bytes()
    if len(raw) != PARENT_MANIFEST_SIZE or sha256_bytes(raw) != PARENT_MANIFEST_SHA256:
        raise ControlManifestError("accepted v4 parent control manifest differs")
    try:
        parent = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlManifestError("accepted v4 parent manifest is not JSON") from error
    if (
        not isinstance(parent, dict)
        or canonical_json(parent) != raw
        or set(parent)
        != {
            "schema_version",
            "kind",
            "implementation_commit",
            "implementation_tree",
            "files",
        }
        or parent.get("schema_version") != 4
        or parent.get("kind") != "pypto-nvidia-executable-sm120-controls-v4"
        or not isinstance(parent.get("files"), list)
    ):
        raise ControlManifestError("accepted v4 parent manifest schema differs")
    records = parent["files"]
    assert isinstance(records, list)
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    primitive_paths = CONTROL_PATHS[-3:]
    if set(primitive_paths) - set(by_path):
        raise ControlManifestError("accepted v4 parent omits an isolation primitive")
    return {
        "path": PARENT_MANIFEST_RELATIVE_PATH.as_posix(),
        "bytes": len(raw),
        "sha256": PARENT_MANIFEST_SHA256,
        "schema_version": parent["schema_version"],
        "kind": parent["kind"],
        "implementation_commit": parent["implementation_commit"],
        "implementation_tree": parent["implementation_tree"],
        "primitive_files": [dict(by_path[name]) for name in primitive_paths],
    }


def _load_frozen_json(
    root: Path,
    relative_path: Path,
    expected_size: int,
    expected_digest: str,
    description: str,
) -> tuple[dict[str, object], bytes]:
    path = root / relative_path
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise ControlManifestError(f"{description} is missing or noncanonical")
    raw = path.read_bytes()
    if len(raw) != expected_size or sha256_bytes(raw) != expected_digest:
        raise ControlManifestError(f"{description} differs")
    try:
        value = json.loads(raw, object_pairs_hook=duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlManifestError(f"{description} is not JSON") from error
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ControlManifestError(f"{description} is not canonical JSON")
    return value, raw


def validate_legacy_controls(
    root: Path, parent_identity: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Bind CP43 controls and the accepted CP44 report without reinterpretation."""

    legacy, legacy_raw = _load_frozen_json(
        root,
        LEGACY_MANIFEST_RELATIVE_PATH,
        LEGACY_MANIFEST_SIZE,
        LEGACY_MANIFEST_SHA256,
        "accepted CP43 control manifest",
    )

    if (
        set(legacy)
        != {
            "schema_version",
            "kind",
            "implementation_commit",
            "implementation_tree",
            "files",
        }
        or legacy.get("schema_version") != 1
        or legacy.get("kind") != "pypto-frontend-vector-add-sm120-controls-v1"
        or not isinstance(legacy.get("files"), list)
        or len(legacy["files"]) != 8
    ):
        raise ControlManifestError("accepted CP43 control manifest schema differs")
    records = legacy["files"]
    assert isinstance(records, list)
    primitive_paths = CONTROL_PATHS[-3:]
    parent_primitives = {
        record["path"]: record for record in parent_identity["primitive_files"]
    }
    normalized: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ControlManifestError("accepted CP43 file record is malformed")
        path_text = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        mode = record.get("mode")
        if (
            not isinstance(path_text, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or isinstance(mode, bool)
            or mode not in {0o644, 0o755}
        ):
            raise ControlManifestError("accepted CP43 file identity is malformed")
        if (
            index >= len(records) - 3
            and record != parent_primitives[primitive_paths[index - 5]]
        ):
            raise ControlManifestError(
                "CP43 isolation primitive differs from accepted v4"
            )
        live = root / path_text
        identity = live.resolve(strict=True).stat()
        if (
            live.is_symlink()
            or live.resolve(strict=True) != live
            or not stat.S_ISREG(identity.st_mode)
            or identity.st_size != size
            or stat.S_IMODE(identity.st_mode) != mode
            or sha256_file(live) != digest
        ):
            raise ControlManifestError(
                f"accepted CP43 live control differs: {path_text}"
            )
        normalized.append(dict(record))

    report, report_raw = _load_frozen_json(
        root,
        LEGACY_REPORT_RELATIVE_PATH,
        LEGACY_REPORT_SIZE,
        LEGACY_REPORT_SHA256,
        "accepted CP44 report",
    )
    report_path = root / LEGACY_REPORT_RELATIVE_PATH
    if stat.S_IMODE(report_path.stat().st_mode) != LEGACY_REPORT_MODE:
        raise ControlManifestError("accepted CP44 report mode differs")
    if (
        report.get("schema_version") != 1
        or report.get("smoke") != "pypto-frontend-vector-add-sm120"
        or report.get("status")
        != "accepted-real-sm120-frontend-vector-add-correctness-smoke"
    ):
        raise ControlManifestError("accepted CP44 report identity differs")
    return (
        {
            "path": LEGACY_MANIFEST_RELATIVE_PATH.as_posix(),
            "bytes": len(legacy_raw),
            "sha256": LEGACY_MANIFEST_SHA256,
            "schema_version": legacy["schema_version"],
            "kind": legacy["kind"],
            "implementation_commit": legacy["implementation_commit"],
            "implementation_tree": legacy["implementation_tree"],
            "files": normalized,
        },
        {
            "path": LEGACY_REPORT_RELATIVE_PATH.as_posix(),
            "bytes": len(report_raw),
            "sha256": LEGACY_REPORT_SHA256,
            "mode": LEGACY_REPORT_MODE,
            "schema_version": report["schema_version"],
            "smoke": report["smoke"],
            "status": report["status"],
        },
    )


def validate_compile_anchors(root: Path) -> dict[str, object]:
    anchors, raw = _load_frozen_json(
        root,
        COMPILE_ANCHORS_RELATIVE_PATH,
        COMPILE_ANCHORS_SIZE,
        COMPILE_ANCHORS_SHA256,
        "fused-pointwise compile anchors",
    )
    if (
        set(anchors)
        != {
            "anchor_runs",
            "dso",
            "kind",
            "records",
            "records_sha256",
            "schema_version",
        }
        or anchors.get("schema_version") != 1
        or anchors.get("kind") != "pypto-fused-pointwise-compile-anchors-v1"
        or anchors.get("records_sha256")
        != "01e8f99dfb0a1aa0e5788177b7d41cc6ed62983037aa57ed8628bb0a1594b844"
        or not isinstance(anchors.get("records"), list)
        or len(anchors["records"]) != 9
        or not isinstance(anchors.get("anchor_runs"), list)
        or len(anchors["anchor_runs"]) != 2
    ):
        raise ControlManifestError("fused-pointwise compile-anchor schema differs")
    if sha256_bytes(canonical_json(anchors["records"])) != anchors["records_sha256"]:
        raise ControlManifestError("compile-anchor record digest differs")
    expected_run_ids = (
        "pypto-20260826T042728Z-1382280-ce1fa0",
        "pypto-20260826T042750Z-1382496-07c3e7",
    )
    expected_case_order = (
        "arith_fp32_tail",
        "arith_bf16_rank3_tail",
        "exp_fp32_tail1",
        "exp_bf16_exact_tile",
        "recip_fp32_tail",
        "recip_bf16_tail1",
        "rsqrt_fp32_rank3_tail",
        "rsqrt_bf16_exact_tile",
        "max16x64_fp32_tail1",
    )
    if [record.get("case") for record in anchors["records"]] != list(
        expected_case_order
    ):
        raise ControlManifestError("compile-anchor case order differs")
    if anchors.get("dso") != {
        "bytes": 784_043_568,
        "mode": 0o755,
        "path": (
            "builds/pypto-fused-pointwise-v2-on-b83fcd3-final/product/"
            "pypto_core.cpython-314-x86_64-linux-gnu.so"
        ),
        "sha256": "0e8f33c263e06777aec06263bf32ca59ac554868529f3fa085212cf27e2facbe",
    }:
        raise ControlManifestError("compile-anchor DSO identity differs")
    for record, expected_run_id in zip(
        anchors["anchor_runs"], expected_run_ids, strict=True
    ):
        if (
            not isinstance(record, dict)
            or record.get("run_id") != expected_run_id
            or record.get("return_code") != 0
            or record.get("cuda_visible_devices") != ""
            or record.get("nvidia_visible_devices") != "void"
            or record.get("torch_cuda_initialized_before_and_after") is not False
        ):
            raise ControlManifestError("compile-anchor run identity differs")
        for sidecar_name in ("preflight", "process", "record"):
            sidecar = record.get(sidecar_name)
            expected_sidecar_keys = {
                "path",
                "bytes",
                "sha256",
                "mode",
            }
            if sidecar_name == "record":
                expected_sidecar_keys.add("records_sha256")
            if not isinstance(sidecar, dict) or set(sidecar) != expected_sidecar_keys:
                raise ControlManifestError("compile-anchor sidecar record is malformed")
            path = root / str(sidecar["path"])
            identity = path.resolve(strict=True).stat()
            if (
                path.is_symlink()
                or path.resolve(strict=True) != path
                or not stat.S_ISREG(identity.st_mode)
                or identity.st_size != sidecar["bytes"]
                or stat.S_IMODE(identity.st_mode) != sidecar["mode"]
                or sha256_file(path) != sidecar["sha256"]
            ):
                raise ControlManifestError("compile-anchor sidecar bytes differ")
        preflight_document, _ = _load_frozen_json(
            root,
            Path(str(record["preflight"]["path"])),
            int(record["preflight"]["bytes"]),
            str(record["preflight"]["sha256"]),
            "compile-anchor preflight",
        )
        process_document, _ = _load_frozen_json(
            root,
            Path(str(record["process"]["path"])),
            int(record["process"]["bytes"]),
            str(record["process"]["sha256"]),
            "compile-anchor process",
        )
        expected_command = [
            "/usr/bin/env",
            "CUDA_VISIBLE_DEVICES=",
            "NVIDIA_VISIBLE_DEVICES=void",
            "PYPTO_CODEGEN_MAX_WORKERS=1",
            "OMP_NUM_THREADS=1",
            "/usr/bin/nice",
            "-n",
            "10",
            "/usr/bin/taskset",
            "-c",
            "0-3",
            "/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/bin/python",
            "-I",
            "-B",
            "-S",
            "/home/zhaosiying/pypto-love-tensor-ir/"
            "tools/generate_pypto_fused_pointwise_anchors.py",
        ]
        if (
            preflight_document.get("mode") != "heavy"
            or preflight_document.get("ok") is not True
            or preflight_document.get("failures") != []
            or preflight_document.get("nvidia_compute_pids") != []
            or preflight_document.get("protected_nvidia_compute_pids") != []
            or preflight_document.get("protected_nvidia_runtime_mapping_pids") != []
            or preflight_document.get("unreadable_protected_maps") != []
            or process_document.get("mode") != "heavy"
            or process_document.get("run_id") != expected_run_id
            or process_document.get("status") != "exited"
            or process_document.get("return_code") != 0
            or process_document.get("command") != expected_command
            or process_document.get("preflight")
            != {
                "path": (
                    "/home/zhaosiying/pypto-love-tensor-ir/"
                    + str(record["preflight"]["path"])
                ),
                "sha256": record["preflight"]["sha256"],
            }
        ):
            raise ControlManifestError("compile-anchor isolation evidence differs")
        record_path = root / str(record["record"]["path"])
        record_document, _ = _load_frozen_json(
            root,
            Path(str(record["record"]["path"])),
            int(record["record"]["bytes"]),
            str(record["record"]["sha256"]),
            "per-run fused-pointwise compile anchors",
        )
        if (
            record_document.get("schema_version") != 1
            or record_document.get("kind")
            != "pypto-fused-pointwise-compile-anchor-run-v1"
            or record_document.get("run_id") != expected_run_id
            or record_document.get("cuda_visible_devices") != ""
            or record_document.get("nvidia_visible_devices") != "void"
            or record_document.get("torch_cuda_initialized_before_and_after")
            is not False
            or record_document.get("generator")
            != {
                "bytes": 11_387,
                "path": "tools/generate_pypto_fused_pointwise_anchors.py",
                "sha256": "89f06a416622e1d78595c0a086db4dce66bebbf70f3867b2601885767e85c54e",
            }
            or record_document.get("dso")
            != {
                "bytes": 784_043_568,
                "path": (
                    "builds/pypto-fused-pointwise-v2-on-b83fcd3-final/product/"
                    "pypto_core.cpython-314-x86_64-linux-gnu.so"
                ),
                "sha256": "0e8f33c263e06777aec06263bf32ca59ac554868529f3fa085212cf27e2facbe",
            }
            or record_document.get("pypto")
            != {
                "head": "b83fcd3ddc497d585bcc45883eede179aff7d4d2",
                "tree": "49eda98f3ed8d72bfd14d5a5900cdc0e71ca699d",
                "clean": True,
            }
            or record_document.get("request")
            != {
                "bytes": 1_583,
                "path": (
                    "runs/pypto-20260825T080254Z-910620-c669d9/"
                    "pypto-nvidia-executable-sm120/compile-request.msgpack"
                ),
                "sha256": "13c319b832c51188678b51a32b155253a6f896bfd1395044832611df0843adda",
            }
            or record_document.get("runner")
            != {
                "bytes": 66_999,
                "path": "benchmarks/operators/pypto_fused_pointwise_sm120.py",
                "sha256": "b7960cc894834b3ba05476943e774cfc8602891faa5b9137b3d97a6aac40ab15",
            }
            or record_document.get("toolchain")
            != {
                "cuda_tile": "af2417041cc939b87ef56d92cfdcf61737c5457e",
                "llvm": "57109befac92811d2253109242ca6fa69c961fb2",
                "pypto": "b83fcd3ddc497d585bcc45883eede179aff7d4d2",
                "tensor_ir": "1dcb38c20e53d07c97d3781cae538e33901bae30",
            }
            or record_document.get("records_sha256") != anchors["records_sha256"]
            or record_document.get("records") != anchors["records"]
            or record["record"].get("records_sha256") != anchors["records_sha256"]
            or stat.S_IMODE(record_path.stat().st_mode) != 0o444
        ):
            raise ControlManifestError("per-run compile-anchor records differ")
        replay_records = record_document.get("replay_files")
        replay_names = ["compile-request.msgpack"]
        for case_name in expected_case_order:
            replay_names.extend(
                [
                    f"{case_name}.hir.msgpack",
                    f"{case_name}.source.mlir",
                    f"{case_name}.build-spec.msgpack",
                    f"{case_name}.artifact.msgpack",
                    f"{case_name}.cubin",
                ]
            )
        if not isinstance(replay_records, list) or len(replay_records) != len(
            replay_names
        ):
            raise ControlManifestError("compile-anchor replay set differs")
        replay_root = (
            root / "runs" / expected_run_id / "fused-pointwise-compile-anchor-replay"
        )
        for replay_record, replay_name in zip(
            replay_records, replay_names, strict=True
        ):
            if not isinstance(replay_record, dict) or set(replay_record) != {
                "bytes",
                "path",
                "sha256",
            }:
                raise ControlManifestError("compile-anchor replay record is malformed")
            replay_path = replay_root / replay_name
            if replay_record.get("path") != replay_path.relative_to(root).as_posix():
                raise ControlManifestError("compile-anchor replay path differs")
            identity = replay_path.resolve(strict=True).stat()
            if (
                replay_path.is_symlink()
                or replay_path.resolve(strict=True) != replay_path
                or not stat.S_ISREG(identity.st_mode)
                or identity.st_size != replay_record.get("bytes")
                or stat.S_IMODE(identity.st_mode) != 0o444
                or sha256_file(replay_path) != replay_record.get("sha256")
            ):
                raise ControlManifestError("compile-anchor replay bytes differ")
    return {
        "path": COMPILE_ANCHORS_RELATIVE_PATH.as_posix(),
        "bytes": len(raw),
        "sha256": COMPILE_ANCHORS_SHA256,
        "schema_version": anchors["schema_version"],
        "kind": anchors["kind"],
        "records_sha256": anchors["records_sha256"],
        "anchor_run_ids": list(expected_run_ids),
    }


def validate_control_manifest(workspace: Path) -> dict[str, object]:
    root = workspace.resolve(strict=True)
    if workspace.absolute() != root:
        raise ControlManifestError("workspace contains a symlinked path")
    reject_control_bytecode_cache(root)
    parent_identity = validate_parent_control_manifest(root)
    legacy_identity, legacy_report = validate_legacy_controls(root, parent_identity)
    compile_anchors = validate_compile_anchors(root)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControlManifestError(
            "reviewed fused-pointwise smoke control manifest is missing"
        )
    manifest_raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_raw, object_pairs_hook=duplicate_key_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlManifestError("control manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or canonical_json(manifest) != manifest_raw:
        raise ControlManifestError("control manifest is not canonical JSON")
    if set(manifest) != {
        "schema_version",
        "kind",
        "implementation_commit",
        "implementation_tree",
        "files",
    }:
        raise ControlManifestError("control manifest top-level schema differs")
    implementation_commit = manifest.get("implementation_commit")
    implementation_tree = manifest.get("implementation_tree")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or not isinstance(implementation_commit, str)
        or COMMIT_PATTERN.fullmatch(implementation_commit) is None
        or not isinstance(implementation_tree, str)
        or COMMIT_PATTERN.fullmatch(implementation_tree) is None
    ):
        raise ControlManifestError("control manifest implementation identity differs")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(CONTROL_PATHS):
        raise ControlManifestError("control manifest file set is incomplete")
    if str(git(root, "rev-parse", f"{implementation_commit}^{{tree}}")).strip() != (
        implementation_tree
    ):
        raise ControlManifestError("implementation commit/tree join failed")
    current_head = str(git(root, "rev-parse", "HEAD")).strip()
    current_tree = str(git(root, "rev-parse", "HEAD^{tree}")).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, current_head],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ControlManifestError("reviewed implementation is not an ancestor")
    dirty = str(git(root, "status", "--porcelain=v1", "--untracked-files=all"))
    if dirty:
        raise ControlManifestError("root control repository is not clean")
    changed_controls = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            f"{implementation_commit}..{current_head}",
            "--",
            *CONTROL_PATHS,
        ],
        cwd=root,
        check=False,
    )
    if changed_controls.returncode != 0:
        raise ControlManifestError(
            "reviewed control files changed after implementation"
        )

    normalized_files: list[dict[str, object]] = []
    parent_primitives = {
        record["path"]: record for record in parent_identity["primitive_files"]
    }
    for record, expected_path in zip(files, CONTROL_PATHS, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "mode",
        }:
            raise ControlManifestError("control manifest file record is malformed")
        if record.get("path") != expected_path:
            raise ControlManifestError("control manifest file order differs")
        if (
            expected_path in parent_primitives
            and record != parent_primitives[expected_path]
        ):
            raise ControlManifestError(
                f"isolation primitive differs from accepted v4: {expected_path}"
            )
        size = record.get("bytes")
        digest = record.get("sha256")
        mode = record.get("mode")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode not in {0o644, 0o755}
        ):
            raise ControlManifestError("control manifest file identity is invalid")
        path = root / expected_path
        resolved = path.resolve(strict=True)
        identity = resolved.stat()
        if (
            resolved != path
            or path.is_symlink()
            or not stat.S_ISREG(identity.st_mode)
            or identity.st_size != size
            or stat.S_IMODE(identity.st_mode) != mode
            or sha256_file(path) != digest
        ):
            raise ControlManifestError(f"live control file differs: {expected_path}")
        committed = git(
            root, "show", f"{implementation_commit}:{expected_path}", text=False
        )
        assert isinstance(committed, bytes)
        if len(committed) != size or sha256_bytes(committed) != digest:
            raise ControlManifestError(
                f"implementation commit blob differs: {expected_path}"
            )
        normalized_files.append(dict(record))
    return {
        "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_bytes": len(manifest_raw),
        "manifest_sha256": sha256_bytes(manifest_raw),
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "current_head": current_head,
        "current_tree": current_tree,
        "root_clean": True,
        "parent_control_manifest": parent_identity,
        "legacy_control_manifest": legacy_identity,
        "legacy_accepted_report": legacy_report,
        "compile_anchors": compile_anchors,
        "files": normalized_files,
    }
