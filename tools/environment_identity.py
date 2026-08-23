#!/usr/bin/env python3
"""Hash the project-local PyTorch runtime and verify ENVIRONMENT.lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = ROOT / "envs" / "pypto-nvidia"


def python_site_packages(prefix: pathlib.Path) -> pathlib.Path:
    candidates = sorted((prefix / "lib").glob("python*/site-packages"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one Python site-packages directory under {prefix}, got {candidates}"
        )
    return candidates[0]


def hash_paths(base: pathlib.Path, roots: tuple[pathlib.Path, ...]) -> dict[str, object]:
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    files = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    for path in files:
        relative = path.relative_to(base).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        if path.is_symlink():
            target = os.readlink(path).encode()
            digest.update(b"L")
            digest.update(len(target).to_bytes(8, "little"))
            digest.update(target)
            continue
        digest.update(b"F")
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        file_count += 1
        byte_count += size
    return {
        "torch_tree_sha256": digest.hexdigest(),
        "torch_tree_files": file_count,
        "torch_tree_bytes": byte_count,
    }


def collect_torch_identity(prefix: pathlib.Path) -> dict[str, object]:
    site_packages = python_site_packages(prefix)
    torch_root = site_packages / "torch"
    distributions = sorted(site_packages.glob("torch-*.dist-info"))
    if not torch_root.is_dir() or len(distributions) != 1:
        raise RuntimeError(
            f"expected one torch package and dist-info under {site_packages}, "
            f"got package={torch_root.is_dir()}, dist-info={distributions}"
        )
    identity = hash_paths(site_packages, (torch_root, distributions[0]))
    identity.update(
        {
            "torch_package_root": str(torch_root),
            "torch_dist_info": str(distributions[0]),
        }
    )
    return identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=pathlib.Path, default=DEFAULT_PREFIX)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    identity = collect_torch_identity(args.prefix.resolve())
    if args.verify:
        locked = json.loads((ROOT / "ENVIRONMENT.lock").read_text())
        mismatches = {
            name: {"locked": locked.get(name), "actual": value}
            for name, value in identity.items()
            if locked.get(name) != value
        }
        if mismatches:
            print(json.dumps(mismatches, indent=2, sort_keys=True), file=sys.stderr)
            return 75
    print(json.dumps(identity, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
