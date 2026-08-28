#!/usr/bin/env python3
"""Materialize the exact shallow-boundary release sources under ``.sources``.

The vendored bundles intentionally start at the audited upstream base commits;
this command records those shallow boundaries before fetching the bundles.
Use this command instead of invoking ``git clone`` on a bundle directly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
from collections.abc import Iterable, Mapping
from typing import Any

import verify_source_release as verifier


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / ".sources"
DEFAULT_LOCK = ROOT / "vendor" / "source-lock.json"


def _git(cwd: pathlib.Path, *arguments: str) -> str:
    return verifier._git(cwd, *arguments)


def clone_bundled_repository(
    root: pathlib.Path,
    destination: pathlib.Path,
    spec: Mapping[str, Any],
) -> None:
    bundle = verifier._relative_path(root, spec["bundle"]["path"], "bundle.path")
    destination.mkdir()
    _git(destination, "init", "--quiet", "--initial-branch=release")
    (destination / ".git" / "shallow").write_text(
        f"{spec['shallow_boundary']}\n", encoding="ascii"
    )
    _git(
        destination,
        "fetch",
        "--quiet",
        str(bundle),
        f"{spec['bundle']['head_ref']}:refs/remotes/release/head",
        f"{spec['bundle']['base_ref']}:refs/tags/release-base",
    )
    _git(destination, "checkout", "--quiet", "--detach", spec["head_commit"])
    # The bundle is the acquisition source; the canonical upstream remains the
    # human-readable remote once materialization is complete.
    _git(destination, "remote", "add", "origin", spec["origin_url"])


def clone_pinned_remote(
    root: pathlib.Path,
    destination: pathlib.Path,
    spec: Mapping[str, Any],
) -> None:
    destination.mkdir()
    _git(destination, "init", "--quiet", "--initial-branch=release")
    _git(destination, "remote", "add", "origin", spec["origin_url"])
    _git(
        destination,
        "fetch",
        "--quiet",
        "--depth=1",
        "origin",
        spec["head_commit"],
    )
    _git(destination, "checkout", "--quiet", "--detach", spec["head_commit"])


def initialize_pypto_submodules(
    pypto_path: pathlib.Path,
    tensor_ir_path: pathlib.Path,
    entries: list[Mapping[str, Any]],
    *,
    jobs: int,
    local_submodule_root: pathlib.Path | None = None,
) -> None:
    local_entries = [entry for entry in entries if entry.get("local_source")]
    if len(local_entries) != 1 or local_entries[0].get("local_source") != "tensor_ir":
        raise verifier.SourceReleaseError(
            "exactly one PyPTO submodule must use the local tensor_ir source"
        )
    tensor_entry = local_entries[0]
    verifier._run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-c",
            f"submodule.{tensor_entry['name']}.url={tensor_ir_path}",
            "submodule",
            "update",
            "--init",
            f"--jobs={jobs}",
            "--",
            str(tensor_entry["path"]),
        ],
        cwd=pypto_path,
    )
    external_paths = [
        str(entry["path"]) for entry in entries if not entry.get("local_source")
    ]
    if external_paths:
        local_overrides: list[str] = []
        if local_submodule_root is not None:
            root = local_submodule_root.resolve()
            local_overrides.extend(("-c", "protocol.file.allow=always"))
            for entry in entries:
                if entry.get("local_source"):
                    continue
                source = root / str(entry["path"])
                if not source.is_dir():
                    raise verifier.SourceReleaseError(
                        f"local submodule acceleration is missing {source}"
                    )
                local_overrides.extend(
                    ("-c", f"submodule.{entry['name']}.url={source}")
                )
        verifier._run(
            [
                "git",
                *local_overrides,
                "submodule",
                "update",
                "--init",
                "--recursive",
                f"--jobs={jobs}",
                "--",
                *external_paths,
            ],
            cwd=pypto_path,
        )


def _bootstrap_manifest(
    destination: pathlib.Path,
    lock: Mapping[str, Any],
    verification: Mapping[str, object],
) -> None:
    payload = {
        "schema": 1,
        "release": lock["release"],
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_lock": "vendor/source-lock.json",
        "repositories": verification,
    }
    (destination / "bootstrap-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def bootstrap(
    root: pathlib.Path,
    destination: pathlib.Path,
    lock: Mapping[str, Any],
    *,
    jobs: int,
    local_submodule_root: pathlib.Path | None = None,
) -> dict[str, object]:
    root = root.resolve()
    destination = destination.resolve()
    if jobs <= 0:
        raise verifier.SourceReleaseError("jobs must be positive")
    if destination == root or destination in root.parents:
        raise verifier.SourceReleaseError(
            f"refusing unsafe source destination: {destination}"
        )
    if os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite source destination: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    incomplete_marker = destination / ".bootstrap-incomplete.json"
    incomplete_marker.write_text(
        json.dumps(
            {
                "schema": 1,
                "release": lock["release"],
                "status": "incomplete-do-not-use",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    materialization = lock["materialization"]
    repositories = lock["repositories"]
    tensor_ir_path = destination / materialization["tensor_ir"]
    pypto_path = destination / materialization["pypto"]
    sglang_path = destination / materialization["sglang"]

    # TensorIR is materialized first so PyPTO can acquire the private gitlink
    # from a local source without changing any global Git URL mapping.
    clone_bundled_repository(root, tensor_ir_path, repositories["tensor_ir"])
    clone_bundled_repository(root, pypto_path, repositories["pypto"])
    initialize_pypto_submodules(
        pypto_path,
        tensor_ir_path,
        lock["pypto_submodules"],
        jobs=jobs,
        local_submodule_root=local_submodule_root,
    )
    clone_pinned_remote(root, sglang_path, repositories["sglang"])

    verification = verifier.verify_materialized_sources(destination, lock)
    _bootstrap_manifest(destination, lock, verification)
    incomplete_marker.unlink()
    return verification


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--lock", type=pathlib.Path)
    parser.add_argument("--destination", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument(
        "--local-submodule-root",
        type=pathlib.Path,
        help=(
            "optional local PyPTO checkout used only to accelerate external "
            "submodule acquisition; locked commits and trees are still verified"
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate an existing materialization without changing it.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    lock_path = (
        args.lock.resolve() if args.lock else root / "vendor" / "source-lock.json"
    )
    destination = (
        args.destination.resolve()
        if args.destination
        else root / DEFAULT_DESTINATION.relative_to(ROOT)
    )
    try:
        lock = verifier.load_lock(lock_path)
        artifact_report = verifier.verify_release_artifacts(root, lock)
        if args.verify_only:
            source_report = verifier.verify_materialized_sources(destination, lock)
        else:
            source_report = bootstrap(
                root,
                destination,
                lock,
                jobs=args.jobs,
                local_submodule_root=args.local_submodule_root,
            )
    except (
        FileExistsError,
        OSError,
        verifier.SourceReleaseError,
    ) as error:
        print(f"source bootstrap failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": 1,
                "release": lock["release"],
                "destination": str(destination),
                "artifacts": artifact_report,
                "sources": source_report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
