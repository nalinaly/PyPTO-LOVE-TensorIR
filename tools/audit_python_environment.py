#!/usr/bin/env python3
"""Reject Python packages or .pth files that escape the project workspace."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import importlib.metadata
import json
import pathlib
import sys
import urllib.parse


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = ROOT / "envs" / "pypto-nvidia"
PROFILE_SOURCE_ROOTS = {
    "pypto": (
        ROOT / ".sources" / "sglang" / "python",
        ROOT / "projects" / "pypto",
        ROOT / "projects" / "pypto-kernels",
        ROOT / "projects" / "pypto-framework-plugins",
        ROOT / "upstream" / "sglang" / "python",
    ),
    "baseline": (
        ROOT / ".sources" / "sglang" / "python",
        ROOT / "upstream" / "sglang" / "python",
    ),
}
FORMAL_PREFIXES = {
    (ROOT / "envs/pypto-release").resolve(),
    (ROOT / "envs/sglang-baseline").resolve(),
}
FORMAL_SOURCE_ROOTS = (ROOT / ".sources" / "sglang" / "python",)
DISTUTILS_PRECEDENCE_PTH = (
    "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; "
    "enabled = os.environ.get(var, 'local') == 'local'; "
    "enabled and __import__('_distutils_hack').add_shim();"
)
CUTLASS_DSL_PACKAGES_PTH = (
    "import sys, os, nvidia_cutlass_dsl; "
    "sys.path.insert(0, os.path.join(nvidia_cutlass_dsl.__path__[0], 'dsl_packages'))"
)


def is_below(path: pathlib.Path, root: pathlib.Path) -> bool:
    resolved = path.resolve()
    root = root.resolve()
    return resolved == root or root in resolved.parents


def is_allowed_source(
    path: pathlib.Path,
    profile: str,
    environment_prefix: pathlib.Path | None = None,
) -> bool:
    roots = (
        FORMAL_SOURCE_ROOTS
        if environment_prefix is not None
        and environment_prefix.resolve() in FORMAL_PREFIXES
        else PROFILE_SOURCE_ROOTS[profile]
    )
    return any(is_below(path, root) for root in roots)


def is_allowed_import_path(
    path: pathlib.Path,
    *,
    environment_prefix: pathlib.Path,
    profile: str,
) -> bool:
    return (
        is_below(path, environment_prefix)
        or is_allowed_source(path, profile, environment_prefix)
        or is_below(path, ROOT / "tools")
    )


def executable_pth_is_allowed(
    path: pathlib.Path, text: str, profile: str, *, formal: bool = False
) -> bool:
    name = path.name
    normalized = text.strip()
    if name == "distutils-precedence.pth":
        return normalized == DISTUTILS_PRECEDENCE_PTH
    if name == "nvidia_cutlass_dsl_packages.pth":
        return normalized == CUTLASS_DSL_PACKAGES_PTH
    if not formal and profile == "pypto" and name == "_editable_skbc_pypto.pth":
        return normalized == "import _editable_skbc_pypto"
    return False


def editable_finder_modules(finder: object) -> tuple[str, ...]:
    """Return every module identity exposed by an instance or class finder."""

    values = {
        value
        for value in (
            getattr(finder, "__module__", None),
            getattr(type(finder), "__module__", None),
        )
        if isinstance(value, str) and value
    }
    return tuple(sorted(values))


def editable_module_is_allowed(
    name: str, profile: str, *, formal: bool = False
) -> bool:
    return not formal and profile == "pypto" and name == "_editable_skbc_pypto"


def external_editable_modules(
    values: Iterable[object], profile: str, *, formal: bool = False
) -> tuple[str, ...]:
    modules = {
        name
        for value in values
        for name in editable_finder_modules(value)
        if name.startswith(("_editable", "__editable"))
        and not editable_module_is_allowed(name, profile, formal=formal)
    }
    return tuple(sorted(modules))


def editable_source_from_direct_url(direct_url: object) -> pathlib.Path:
    if not isinstance(direct_url, str) or not direct_url:
        raise ValueError("editable direct URL must be a non-empty string")
    parsed = urllib.parse.urlparse(direct_url)
    if parsed.scheme != "file" or parsed.netloc:
        raise ValueError("editable direct URL must be a local file URL")
    return pathlib.Path(urllib.parse.unquote(parsed.path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=pathlib.Path, default=DEFAULT_PREFIX)
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_SOURCE_ROOTS), default="pypto"
    )
    args = parser.parse_args()
    environment_prefix = args.prefix.resolve()
    formal = environment_prefix in FORMAL_PREFIXES
    failures: list[dict[str, str]] = []
    if pathlib.Path(sys.prefix).resolve() != environment_prefix:
        failures.append(
            {
                "kind": "python-prefix",
                "value": sys.prefix,
                "expected": str(environment_prefix),
            }
        )

    site_packages = pathlib.Path(
        importlib.metadata.distribution("torch").locate_file("")
    ).resolve()
    if not is_below(site_packages, environment_prefix):
        failures.append(
            {
                "kind": "torch-distribution-path",
                "value": str(site_packages),
                "expected": str(environment_prefix),
            }
        )

    for entry in sys.path:
        if not entry:
            continue
        path = pathlib.Path(entry)
        if not is_allowed_import_path(
            path,
            environment_prefix=environment_prefix,
            profile=args.profile,
        ):
            failures.append(
                {
                    "kind": "sys-path-escape",
                    "value": str(path.resolve()),
                }
            )

    for path in sorted(site_packages.glob("*.pth")):
        for raw_line in path.read_text(errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("import ", "import\t")):
                if not executable_pth_is_allowed(
                    path, line, args.profile, formal=formal
                ):
                    failures.append(
                        {
                            "kind": "pth-executable-code",
                            "file": str(path),
                            "value": line,
                        }
                    )
                continue
            candidate = pathlib.Path(line)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            if not is_allowed_import_path(
                candidate,
                environment_prefix=environment_prefix,
                profile=args.profile,
            ):
                failures.append(
                    {
                        "kind": "pth-path-escape",
                        "file": str(path),
                        "value": str(candidate.resolve()),
                    }
                )

    for distribution in importlib.metadata.distributions():
        direct_url_text = distribution.read_text("direct_url.json")
        if not direct_url_text:
            continue
        try:
            direct_url_record = json.loads(direct_url_text)
        except ValueError:
            failures.append(
                {
                    "kind": "invalid-direct-url",
                    "distribution": distribution.metadata.get("Name", "unknown"),
                }
            )
            continue
        if not direct_url_record.get("dir_info", {}).get("editable", False):
            continue
        direct_url = direct_url_record.get("url", "")
        try:
            source = editable_source_from_direct_url(direct_url)
        except ValueError:
            failures.append(
                {
                    "kind": "invalid-editable-direct-url",
                    "distribution": distribution.metadata.get("Name", "unknown"),
                    "value": direct_url,
                }
            )
            continue
        if not is_allowed_source(source, args.profile, environment_prefix):
            failures.append(
                {
                    "kind": "editable-external-source",
                    "distribution": distribution.metadata.get("Name", "unknown"),
                    "value": str(source),
                }
            )

    importer_carriers = (
        ("editable-meta-path-finder", sys.meta_path),
        ("editable-path-hook", sys.path_hooks),
        ("editable-importer-cache", sys.path_importer_cache.values()),
    )
    for kind, values in importer_carriers:
        external_modules = external_editable_modules(
            values, args.profile, formal=formal
        )
        if external_modules:
            failures.append(
                {
                    "kind": kind,
                    "value": ",".join(external_modules),
                }
            )

    loaded_editable_modules = tuple(
        sorted(
            name
            for name in sys.modules
            if name.startswith(("_editable", "__editable"))
            and not editable_module_is_allowed(name, args.profile, formal=formal)
        )
    )
    if loaded_editable_modules:
        failures.append(
            {
                "kind": "editable-loaded-module",
                "value": ",".join(loaded_editable_modules),
            }
        )

    report = {
        "status": "pass" if not failures else "fail",
        "environment": str(environment_prefix),
        "profile": args.profile,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 75


if __name__ == "__main__":
    sys.exit(main())
