#!/usr/bin/env python3
"""Create and finalize the two formal release Python environments.

The default path is lock-driven: Conda consumes the explicit SHA-256 lock and
pip installs only wheels admitted by the hash-complete requirements lock.  A
validated ``--base-prefix`` is an acceleration, not the only reproduction path.
Environment identity locks are written only by ``--finalize-only``, after the
release build has installed PyPTO, pypto-kernels, and the framework plugin into
the candidate prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.cupti_overlay import (  # noqa: E402
    materialize_overlay,
    validate_overlay,
)

ENVIRONMENT_DIR = ROOT / "environment"
CONDA_LOCK = ENVIRONMENT_DIR / "conda-linux-64.lock"
PYTHON_LOCK = ENVIRONMENT_DIR / "python-requirements.lock"
ARTIFACT_LOCK = ENVIRONMENT_DIR / "python-artifacts.json"
CANDIDATE_PREFIX = ROOT / "envs" / "pypto-release"
BASELINE_PREFIX = ROOT / "envs" / "sglang-baseline"
SGLANG_SOURCE = ROOT / ".sources" / "sglang"
IDENTITY_LOCK_NAME = ".identity-lock.json"
REQUIRED_CANDIDATE_DISTRIBUTIONS = {
    "pypto",
    "pypto-framework-plugins",
    "pypto-kernels",
}


class EnvironmentBootstrapError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command(
    arguments: list[str],
    *,
    cwd: pathlib.Path = ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise EnvironmentBootstrapError(
            f"command failed ({result.returncode}): {' '.join(arguments)}: {detail}"
        )
    return result


def load_artifact_lock() -> dict[str, object]:
    try:
        lock = json.loads(ARTIFACT_LOCK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EnvironmentBootstrapError(f"cannot read {ARTIFACT_LOCK}: {error}")
    if lock.get("schema") != 1 or lock.get("release") != "qwen35-sm120-v1":
        raise EnvironmentBootstrapError("unexpected Python artifact lock identity")
    for label, path in (
        ("conda_lock", CONDA_LOCK),
        ("python_requirements_lock", PYTHON_LOCK),
    ):
        metadata = lock.get(label)
        if not isinstance(metadata, dict):
            raise EnvironmentBootstrapError(f"missing {label} metadata")
        if path.stat().st_size != metadata.get("bytes"):
            raise EnvironmentBootstrapError(f"{label} size differs")
        if sha256_file(path) != metadata.get("sha256"):
            raise EnvironmentBootstrapError(f"{label} SHA-256 differs")
    return lock


def _conda_executable(value: pathlib.Path | None) -> pathlib.Path:
    candidate = value
    if candidate is None:
        configured = os.environ.get("CONDA_EXE")
        candidate = pathlib.Path(configured) if configured else None
    if candidate is None:
        discovered = shutil.which("conda")
        candidate = pathlib.Path(discovered) if discovered else None
    if candidate is None or not candidate.resolve().is_file():
        raise EnvironmentBootstrapError(
            "Conda is required; pass --conda or set CONDA_EXE"
        )
    return candidate.resolve()


def _require_absent(path: pathlib.Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite environment: {path}")


def create_conda_prefix(
    conda: pathlib.Path,
    destination: pathlib.Path,
    *,
    base_prefix: pathlib.Path | None,
) -> None:
    _require_absent(destination)
    if base_prefix is None:
        arguments = [
            str(conda),
            "create",
            "--yes",
            "--prefix",
            str(destination),
            "--file",
            str(CONDA_LOCK),
        ]
    else:
        source = base_prefix.resolve()
        if not (source / "bin" / "python").is_file():
            raise EnvironmentBootstrapError(f"invalid base prefix: {source}")
        arguments = [
            str(conda),
            "create",
            "--yes",
            "--prefix",
            str(destination),
            "--clone",
            str(source),
        ]
    command(arguments)


def install_locked_python(
    prefix: pathlib.Path,
    artifact_lock: Mapping[str, object],
    *,
    wheelhouse: pathlib.Path | None,
) -> None:
    python = prefix / "bin" / "python"
    arguments = [
        str(python),
        "-m",
        "pip",
        "--isolated",
        "install",
        "--disable-pip-version-check",
        "--require-hashes",
        "--no-deps",
    ]
    if wheelhouse is not None:
        resolved = wheelhouse.resolve()
        if not resolved.is_dir():
            raise EnvironmentBootstrapError(f"wheelhouse is missing: {resolved}")
        arguments.extend(("--no-index", "--find-links", str(resolved)))
    else:
        indexes = artifact_lock.get("indexes")
        if not isinstance(indexes, list) or len(indexes) != 2:
            raise EnvironmentBootstrapError("artifact indexes are not locked")
        arguments.extend(
            ("--index-url", str(indexes[0]), "--extra-index-url", str(indexes[1]))
        )
    arguments.extend(("--requirement", str(PYTHON_LOCK)))
    command(arguments)
    conflict = command([str(python), "-m", "pip", "check"], check=False)
    output = (conflict.stdout + conflict.stderr).strip()
    if conflict.returncode != 0 or output != "No broken requirements found.":
        raise EnvironmentBootstrapError(
            f"formal runtime is not pip-check clean: {output!r}"
        )


def _site_packages(prefix: pathlib.Path) -> pathlib.Path:
    result = command(
        [
            str(prefix / "bin" / "python"),
            "-I",
            "-B",
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ]
    )
    path = pathlib.Path(result.stdout.strip()).resolve()
    if prefix.resolve() not in path.parents:
        raise EnvironmentBootstrapError(f"site-packages escaped prefix: {path}")
    return path


def remove_candidate_distributions(prefix: pathlib.Path) -> None:
    site_packages = _site_packages(prefix)
    patterns = (
        "__editable__.pypto*",
        "_editable_skbc_pypto*",
        "pypto",
        "pypto-*.dist-info",
        "pypto_framework_plugins-*.dist-info",
        "pypto_kernels-*.dist-info",
    )
    for pattern in patterns:
        for path in site_packages.glob(pattern):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def installed_distributions(prefix: pathlib.Path) -> dict[str, str]:
    script = (
        "import importlib.metadata as m,json; "
        "print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version "
        "for d in m.distributions() if d.metadata.get('Name')},sort_keys=True))"
    )
    result = command([str(prefix / "bin" / "python"), "-I", "-B", "-c", script])
    return json.loads(result.stdout)


def create_environments(
    *,
    conda: pathlib.Path,
    base_prefix: pathlib.Path | None,
    wheelhouse: pathlib.Path | None,
) -> dict[str, object]:
    artifact_lock = load_artifact_lock()
    for destination in (CANDIDATE_PREFIX, BASELINE_PREFIX):
        _require_absent(destination)
    for destination in (CANDIDATE_PREFIX, BASELINE_PREFIX):
        create_conda_prefix(conda, destination, base_prefix=base_prefix)
        remove_candidate_distributions(destination)
        install_locked_python(destination, artifact_lock, wheelhouse=wheelhouse)
    overlay = materialize_overlay(wheelhouse, allow_download=wheelhouse is None)
    candidate = installed_distributions(CANDIDATE_PREFIX)
    baseline = installed_distributions(BASELINE_PREFIX)
    if candidate != baseline:
        raise EnvironmentBootstrapError(
            "formal base environments do not have identical distributions"
        )
    return {
        "schema": 1,
        "status": "base-environments-created-candidate-install-pending",
        "candidate": str(CANDIDATE_PREFIX.relative_to(ROOT)),
        "baseline": str(BASELINE_PREFIX.relative_to(ROOT)),
        "distributions": len(candidate),
        "accelerated_from_base_prefix": base_prefix is not None,
        "cupti_overlay": overlay,
    }


def _write_identity_lock(prefix: pathlib.Path) -> dict[str, object]:
    result = command(
        [
            str(prefix / "bin" / "python"),
            str(ROOT / "tools" / "environment_identity.py"),
            "--prefix",
            str(prefix),
        ]
    )
    identity = json.loads(result.stdout)
    identity.update(
        {
            "schema": 2,
            "release": "qwen35-sm120-v1",
            "formal_prefix": str(prefix.relative_to(ROOT)),
        }
    )
    path = prefix / IDENTITY_LOCK_NAME
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return identity


def finalize_environments() -> dict[str, object]:
    for prefix in (CANDIDATE_PREFIX, BASELINE_PREFIX):
        if not (prefix / "bin" / "python").is_file():
            raise EnvironmentBootstrapError(f"formal environment is missing: {prefix}")
    if not SGLANG_SOURCE.is_dir():
        raise EnvironmentBootstrapError(
            "source bootstrap must materialize .sources/sglang before finalization"
        )
    overlay = validate_overlay()
    candidate = installed_distributions(CANDIDATE_PREFIX)
    baseline = installed_distributions(BASELINE_PREFIX)
    candidate_names = set(candidate)
    missing = sorted(REQUIRED_CANDIDATE_DISTRIBUTIONS - candidate_names)
    leaked = sorted(REQUIRED_CANDIDATE_DISTRIBUTIONS & set(baseline))
    if missing or leaked:
        raise EnvironmentBootstrapError(
            f"candidate/baseline package boundary differs: missing={missing}, leaked={leaked}"
        )
    common_candidate = {
        name: version
        for name, version in candidate.items()
        if name not in REQUIRED_CANDIDATE_DISTRIBUTIONS
    }
    if common_candidate != baseline:
        raise EnvironmentBootstrapError(
            "candidate and baseline runtime dependency sets are not identical"
        )
    identities = {
        "pypto-release": _write_identity_lock(CANDIDATE_PREFIX),
        "sglang-baseline": _write_identity_lock(BASELINE_PREFIX),
    }
    return {
        "schema": 1,
        "status": "formal-environments-finalized",
        "identities": {
            name: {
                "torch_tree_sha256": value["torch_tree_sha256"],
                "distributions_sha256": value["distributions_sha256"],
            }
            for name, value in identities.items()
        },
        "cupti_overlay": overlay,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", type=pathlib.Path)
    parser.add_argument(
        "--base-prefix",
        type=pathlib.Path,
        help="optional validated local acceleration; fresh creation is the default",
    )
    parser.add_argument("--wheelhouse", type=pathlib.Path)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.finalize_only:
            if args.base_prefix or args.wheelhouse or args.conda:
                parser.error("--finalize-only accepts no creation options")
            report = finalize_environments()
        else:
            report = create_environments(
                conda=_conda_executable(args.conda),
                base_prefix=args.base_prefix,
                wheelhouse=args.wheelhouse,
            )
    except (EnvironmentBootstrapError, FileExistsError, OSError) as error:
        print(f"release environment bootstrap failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
