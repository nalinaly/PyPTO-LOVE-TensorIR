#!/usr/bin/env python3
"""Reject Python packages or .pth files that escape the project workspace."""

from __future__ import annotations

import importlib.metadata
import json
import pathlib
import sys
import urllib.parse


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PREFIX = ROOT / "envs" / "pypto-nvidia"


def is_allowed(path: pathlib.Path) -> bool:
    resolved = path.resolve()
    return resolved == ROOT or ROOT in resolved.parents


def main() -> int:
    failures: list[dict[str, str]] = []
    if pathlib.Path(sys.prefix).resolve() != ENV_PREFIX:
        failures.append(
            {
                "kind": "python-prefix",
                "value": sys.prefix,
                "expected": str(ENV_PREFIX),
            }
        )

    site_packages = pathlib.Path(importlib.metadata.distribution("torch").locate_file(""))
    for path in sorted(site_packages.glob("*.pth")):
        text = path.read_text(errors="replace")
        for token in text.replace("'", " ").replace('"', " ").split():
            if token.startswith("/home/") and not is_allowed(pathlib.Path(token)):
                failures.append(
                    {
                        "kind": "pth-external-path",
                        "file": str(path),
                        "value": token,
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
        if not is_allowed(source):
            failures.append(
                {
                    "kind": "editable-external-source",
                    "distribution": distribution.metadata.get("Name", "unknown"),
                    "value": str(source),
                }
            )

    report = {
        "status": "pass" if not failures else "fail",
        "environment": str(ENV_PREFIX),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 75


if __name__ == "__main__":
    sys.exit(main())
