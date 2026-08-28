#!/usr/bin/env python3
"""Reject nested Git repositories and gitlinks from the publishable tree."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLISHABLE_ROOTS = ("vendor", "environment", "packages", "tools", "tests")


class NestedGitError(RuntimeError):
    pass


def _git(root: pathlib.Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise NestedGitError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def gitlinks(root: pathlib.Path) -> list[str]:
    entries = _git(root, "ls-files", "--stage", "-z").split("\0")
    found: list[str] = []
    for entry in entries:
        if not entry:
            continue
        metadata, path = entry.split("\t", maxsplit=1)
        mode = metadata.split(maxsplit=1)[0]
        if mode == "160000":
            found.append(path)
    return sorted(found)


def nested_git_entries(
    root: pathlib.Path,
    publishable_roots: Iterable[str] = PUBLISHABLE_ROOTS,
) -> list[str]:
    found: list[str] = []
    for relative_root in publishable_roots:
        base = root / relative_root
        if not base.exists():
            continue
        for directory, directory_names, file_names in os.walk(base):
            current = pathlib.Path(directory)
            if ".git" in directory_names:
                found.append((current / ".git").relative_to(root).as_posix())
                directory_names.remove(".git")
            if ".git" in file_names:
                found.append((current / ".git").relative_to(root).as_posix())
    return sorted(found)


def verify(
    root: pathlib.Path,
    publishable_roots: Iterable[str] = PUBLISHABLE_ROOTS,
) -> dict[str, object]:
    root = root.resolve()
    links = gitlinks(root)
    nested = nested_git_entries(root, publishable_roots)
    if links or nested:
        raise NestedGitError(
            f"nested Git state is forbidden: gitlinks={links}, nested_git={nested}"
        )
    return {
        "schema": 1,
        "status": "clean",
        "gitlinks": 0,
        "nested_git_entries": 0,
        "publishable_roots": list(publishable_roots),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--publishable-root", action="append", dest="roots")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = verify(args.root, args.roots or PUBLISHABLE_ROOTS)
    except NestedGitError as error:
        print(f"nested Git verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
