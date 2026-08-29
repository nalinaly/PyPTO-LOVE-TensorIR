#!/usr/bin/env python3
"""Materialize or verify the hash-locked cupti-python profiler overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.cupti_overlay import (  # noqa: E402
    materialize_overlay,
    validate_overlay,
)
from benchmarks.release.workload import ReleaseContractError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only and (args.wheelhouse or args.allow_download):
        parser.error("--verify-only accepts no materialization options")
    try:
        report = (
            validate_overlay()
            if args.verify_only
            else materialize_overlay(
                args.wheelhouse, allow_download=args.allow_download
            )
        )
    except (OSError, ReleaseContractError, zipfile.BadZipFile) as error:
        print(f"CUPTI overlay materialization failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
