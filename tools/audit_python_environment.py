#!/usr/bin/env python3
"""Reject Python packages or .pth files that escape the project workspace."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import pathlib
import sys
import urllib.parse


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = ROOT / "envs" / "pypto-nvidia"
PROFILE_SOURCE_ROOTS = {
    "pypto": (
        ROOT / "projects" / "pypto",
        ROOT / "projects" / "pypto-kernels",
        ROOT / "projects" / "pypto-framework-plugins",
        ROOT / "upstream" / "sglang" / "python",
    ),
    "baseline": (ROOT / "upstream" / "sglang" / "python",),
}
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


def is_allowed_source(path: pathlib.Path, profile: str) -> bool:
    return any(is_below(path, root) for root in PROFILE_SOURCE_ROOTS[profile])


def is_allowed_import_path(
    path: pathlib.Path,
    *,
    environment_prefix: pathlib.Path,
    profile: str,
) -> bool:
    return (
        is_below(path, environment_prefix)
        or is_allowed_source(path, profile)
        or is_below(path, ROOT / "tools")
    )


def is_strict_editable_install_line(text: str, package_prefix: str) -> bool:
    """Accept only ``import finder; finder.install()`` with no other code."""

    tree = ast.parse(text)
    if len(tree.body) != 2:
        return False
    import_node, call_node = tree.body
    if not isinstance(import_node, ast.Import) or len(import_node.names) != 1:
        return False
    alias = import_node.names[0]
    if alias.asname is not None or not alias.name.startswith(package_prefix):
        return False
    if not isinstance(call_node, ast.Expr) or not isinstance(call_node.value, ast.Call):
        return False
    call = call_node.value
    return (
        not call.args
        and not call.keywords
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "install"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == alias.name
    )


def executable_pth_is_allowed(path: pathlib.Path, text: str, profile: str) -> bool:
    name = path.name
    normalized = text.strip()
    if name == "distutils-precedence.pth":
        return normalized == DISTUTILS_PRECEDENCE_PTH
    if name == "nvidia_cutlass_dsl_packages.pth":
        return normalized == CUTLASS_DSL_PACKAGES_PTH
    if profile == "pypto" and name == "_editable_skbc_pypto.pth":
        return normalized == "import _editable_skbc_pypto"
    if name.startswith("__editable__.sglang-"):
        return is_strict_editable_install_line(
            normalized,
            "__editable___sglang_",
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=pathlib.Path, default=DEFAULT_PREFIX)
    parser.add_argument("--profile", choices=tuple(PROFILE_SOURCE_ROOTS), default="pypto")
    args = parser.parse_args()
    environment_prefix = args.prefix.resolve()
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
                if not executable_pth_is_allowed(path, line, args.profile):
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
        parsed = urllib.parse.urlparse(direct_url)
        if parsed.scheme != "file":
            continue
        source = pathlib.Path(urllib.parse.unquote(parsed.path))
        if not is_allowed_source(source, args.profile):
            failures.append(
                {
                    "kind": "editable-external-source",
                    "distribution": distribution.metadata.get("Name", "unknown"),
                    "value": str(source),
                }
            )

    for finder in sys.meta_path:
        module_name = type(finder).__module__
        if not module_name.startswith(("_editable", "__editable")):
            continue
        allowed = (
            (args.profile == "pypto" and module_name == "_editable_skbc_pypto")
            or module_name.startswith("__editable___sglang_")
        )
        if not allowed:
            failures.append(
                {
                    "kind": "editable-meta-path-finder",
                    "value": module_name,
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
