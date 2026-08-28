#!/usr/bin/env python3
"""Download or verify the exact Qwen3.5 release model snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/MANIFEST.json"


class ModelReleaseError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != 1
        or value.get("status") != "complete"
        or not isinstance(value.get("models"), dict)
        or "/home/" in path.read_text(encoding="utf-8")
    ):
        raise ModelReleaseError("model manifest is incomplete or non-portable")
    for name, spec in value["models"].items():
        if (
            not isinstance(spec, dict)
            or spec.get("repository_id") != f"Qwen/{name}"
            or not isinstance(spec.get("revision"), str)
            or len(spec["revision"]) != 40
            or not isinstance(spec.get("files"), dict)
            or not spec["files"]
        ):
            raise ModelReleaseError(f"invalid model specification: {name}")
    return value


def verify_model(root: Path, spec: dict[str, object]) -> dict[str, object]:
    destination = (root / str(spec["destination"])).resolve(strict=True)
    models_root = (root / "models").resolve(strict=True)
    if models_root not in destination.parents:
        raise ModelReleaseError(f"model destination escaped models/: {destination}")
    expected = spec["files"]
    records = {}
    for name, metadata in expected.items():
        path = destination / name
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise ModelReleaseError(f"model file is missing or linked: {path}")
        size = path.stat().st_size
        digest = _sha256(path)
        if size != metadata["bytes"] or digest != metadata["sha256"]:
            raise ModelReleaseError(f"model file identity differs: {path}")
        records[name] = {"bytes": size, "sha256": digest}
    extras = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(destination).parts
        and path.relative_to(destination).as_posix() not in expected
    )
    if extras:
        raise ModelReleaseError(f"model snapshot has untracked files: {extras}")
    return {
        "repository_id": spec["repository_id"],
        "revision": spec["revision"],
        "destination": str(destination.relative_to(root)),
        "files": len(records),
        "bytes": sum(item["bytes"] for item in records.values()),
        "status": "verified",
    }


def download_model(root: Path, spec: dict[str, object]) -> None:
    destination = root / str(spec["destination"])
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite model destination: {destination}")
    partial = destination.with_name(f".{destination.name}.partial")
    if os.path.lexists(partial):
        raise FileExistsError(f"refusing to overwrite partial download: {partial}")
    partial.mkdir(parents=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=str(spec["repository_id"]),
        revision=str(spec["revision"]),
        local_dir=partial,
        allow_patterns=sorted(spec["files"]),
    )
    temporary_spec = dict(spec)
    temporary_spec["destination"] = str(partial.relative_to(root))
    verify_model(root, temporary_spec)
    os.replace(partial, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("0.8B", "9B", "all"), default="all"
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    names = (
        tuple(manifest["models"])
        if args.model == "all"
        else (f"Qwen3.5-{args.model}",)
    )
    results = []
    try:
        for name in names:
            spec = manifest["models"][name]
            destination = ROOT / str(spec["destination"])
            if not args.verify_only and not destination.exists():
                download_model(ROOT, spec)
            results.append(verify_model(ROOT, spec))
    except (FileExistsError, ModelReleaseError, OSError) as error:
        print(f"model release failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"schema": 1, "models": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
