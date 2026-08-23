#!/usr/bin/env python3
"""Clone the approved base environment into this workspace and record it."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from environment_identity import collect_torch_identity


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = pathlib.Path("/home/zhaosiying/miniforge3/envs/triton-dev")
DESTINATION = ROOT / "envs" / "pypto-nvidia"
CONDA = pathlib.Path("/home/zhaosiying/miniforge3/bin/conda")


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, capture_output=True
    ).stdout.strip()


def atomic_json(path: pathlib.Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("PYPTO_SOURCE_CONDA_ENV", DEFAULT_SOURCE)),
    )
    parser.add_argument("--lock-existing", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    preflight = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "preflight.py"), "--mode", "heavy"],
        cwd=ROOT,
        check=False,
    )
    if preflight.returncode != 0:
        return preflight.returncode
    if not source.is_dir() or not (source / "bin" / "python").is_file():
        raise FileNotFoundError(f"source Conda environment is invalid: {source}")
    if DESTINATION.exists() and not args.lock_existing:
        raise FileExistsError(f"refusing to overwrite environment: {DESTINATION}")
    if not CONDA.is_file():
        raise FileNotFoundError(f"Conda executable not found: {CONDA}")

    if not DESTINATION.exists():
        subprocess.run(
            [
                str(CONDA),
                "create",
                "--yes",
                "--prefix",
                str(DESTINATION),
                "--clone",
                str(source),
            ],
            cwd=ROOT,
            check=True,
        )
    python = DESTINATION / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "check"], check=True, cwd=ROOT)
    probe = json.loads(
        command_output(
            [
                str(python),
                "-c",
                "import json,sys,torch; print(json.dumps({"
                "'python':sys.version,'torch':torch.__version__,"
                "'torch_git':torch.version.git_version,'cuda':torch.version.cuda,"
                "'hip':torch.version.hip,'torch_file':torch.__file__}))",
            ]
        )
    )
    probe.update(
        {
            "schema": 1,
            "status": "cloned",
            "source_prefix": str(source),
            "destination_prefix": str(DESTINATION.relative_to(ROOT)),
            "cloned_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "conda_explicit_sha256": hashlib.sha256(
                subprocess.check_output(
                    [
                        str(CONDA),
                        "list",
                        "--prefix",
                        str(DESTINATION),
                        "--explicit",
                    ]
                )
            ).hexdigest(),
        }
    )
    probe.update(collect_torch_identity(DESTINATION))
    atomic_json(ROOT / "ENVIRONMENT.lock", probe)
    print(json.dumps(probe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
