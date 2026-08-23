#!/usr/bin/env python3
"""Copy frozen Qwen3.5 snapshots out of the protected AMD tree safely.

The source is opened read-only. Files are copied into a temporary directory,
hashed, made read-only, and atomically renamed inside this workspace. Symlinks,
hard links, caches, and AMD-project logs are never imported.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

from preflight import HEAVY_MARKERS, process_table


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "Qwen3.5-0.8B": {
        "source": pathlib.Path("/home/zhaosiying/amdgpu-sim/models/Qwen3.5-0.8B"),
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "revision_file": "manifest.json",
    },
    "Qwen3.5-9B": {
        "source": pathlib.Path("/home/zhaosiying/amdgpu-sim/models/Qwen3.5-9B"),
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "revision_file": ".amdgpu-sim-download.json",
    },
}
EXACT_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
PROTECTED_CHECK_INTERVAL_BYTES = 512 * 1024 * 1024


def ensure_protected_workloads_idle() -> None:
    protected, _workspace = process_table()
    heavy = [
        process
        for process in protected
        if any(marker in process.command for marker in HEAVY_MARKERS)
    ]
    if heavy:
        pids = [process.pid for process in heavy]
        raise RuntimeError(
            f"protected workload started during model import; aborting this copy, PIDs={pids}"
        )


def should_copy(path: pathlib.Path) -> bool:
    return path.name in EXACT_FILES or (
        path.name.startswith("model.safetensors-") and path.suffix == ".safetensors"
    )


def copy_and_hash(source: pathlib.Path, destination: pathlib.Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    bytes_since_check = 0
    with source.open("rb") as source_file, destination.open("xb") as destination_file:
        while chunk := source_file.read(8 * 1024 * 1024):
            destination_file.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            bytes_since_check += len(chunk)
            if bytes_since_check >= PROTECTED_CHECK_INTERVAL_BYTES:
                ensure_protected_workloads_idle()
                bytes_since_check = 0
        destination_file.flush()
        os.fsync(destination_file.fileno())
    destination.chmod(0o444)
    return size, digest.hexdigest()


def verify_revision(spec: dict[str, object]) -> None:
    source = pathlib.Path(spec["source"])
    metadata = json.loads((source / str(spec["revision_file"])).read_text())
    found = metadata.get("revision")
    if found != spec["revision"]:
        raise RuntimeError(
            f"source revision mismatch for {source}: expected {spec['revision']}, got {found}"
        )


def import_model(name: str) -> dict[str, object]:
    spec = MODEL_SPECS[name]
    source = pathlib.Path(spec["source"])
    destination = ROOT / "models" / name
    temporary = ROOT / "models" / f".{name}.import-{os.getpid()}"
    if destination.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite model destination or staging path: {destination}, {temporary}"
        )
    verify_revision(spec)
    ensure_protected_workloads_idle()
    files = sorted(path for path in source.iterdir() if should_copy(path))
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise RuntimeError(f"no safetensors model files found in {source}")
    for path in files:
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise RuntimeError(
                f"source file must be an independent regular file, got {path} "
                f"(symlink={path.is_symlink()}, nlink={path.stat().st_nlink})"
            )

    temporary.mkdir(mode=0o755)
    records: dict[str, dict[str, object]] = {}
    try:
        for source_file in files:
            size, digest = copy_and_hash(source_file, temporary / source_file.name)
            records[source_file.name] = {"bytes": size, "sha256": digest}
        temporary.chmod(0o555)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            for path in temporary.iterdir():
                path.chmod(0o644)
            temporary.chmod(0o755)
            shutil.rmtree(temporary)
        raise

    return {
        "source": str(source),
        "revision": spec["revision"],
        "destination": str(destination.relative_to(ROOT)),
        "files": records,
    }


def write_manifest(imported: dict[str, dict[str, object]]) -> None:
    manifest_path = ROOT / "models" / "MANIFEST.json"
    manifest = {
        "schema": 1,
        "status": "complete",
        "copy_policy": "ordinary independent copy; never symlink or hardlink",
        "imported_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "models": imported,
    }
    temporary = manifest_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    args = parser.parse_args()
    preflight = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "preflight.py"), "--mode", "heavy"],
        cwd=ROOT,
        check=False,
    )
    if preflight.returncode != 0:
        return preflight.returncode
    imported = {name: import_model(name) for name in args.models}
    write_manifest(imported)
    print(json.dumps(imported, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
