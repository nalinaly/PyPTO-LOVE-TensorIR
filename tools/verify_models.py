#!/usr/bin/env python3
"""Verify copied model bytes, permissions, links, and frozen revisions."""

from __future__ import annotations

import hashlib
import json
import pathlib
import stat
import sys

from import_models import (
    PROTECTED_CHECK_INTERVAL_BYTES,
    ROOT,
    ensure_protected_workloads_idle,
)


def hash_file(path: pathlib.Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    bytes_since_check = 0
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            bytes_since_check += len(chunk)
            if bytes_since_check >= PROTECTED_CHECK_INTERVAL_BYTES:
                ensure_protected_workloads_idle()
                bytes_since_check = 0
    return size, digest.hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "models" / "MANIFEST.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"model manifest is not complete: {manifest.get('status')}")
    verified: dict[str, dict[str, object]] = {}
    for model_name, model in sorted(manifest["models"].items()):
        model_root = ROOT / str(model["destination"])
        records: dict[str, object] = {}
        for file_name, expected in sorted(model["files"].items()):
            path = model_root / file_name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                raise RuntimeError(f"model entry is not an independent regular file: {path}")
            if metadata.st_nlink != 1:
                raise RuntimeError(f"model entry has {metadata.st_nlink} hard links: {path}")
            if metadata.st_mode & 0o222:
                raise RuntimeError(f"model entry is writable: {path}")
            size, digest = hash_file(path)
            if size != expected["bytes"] or digest != expected["sha256"]:
                raise RuntimeError(
                    f"model digest mismatch for {path}: expected {expected}, "
                    f"got bytes={size}, sha256={digest}"
                )
            records[file_name] = {"bytes": size, "sha256": digest}
        verified[model_name] = records
    print(json.dumps({"status": "pass", "models": verified}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
