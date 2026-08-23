#!/usr/bin/env python3
"""Terminate only a process group proven to belong to a recorded PyPTO run."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]


def process_start_ticks(pid: int) -> int:
    fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[21])


def process_environment(pid: int) -> dict[str, str]:
    raw = pathlib.Path(f"/proc/{pid}/environ").read_bytes()
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def verify(metadata: dict[str, object]) -> tuple[int, int]:
    run_id = str(metadata["run_id"])
    pid = int(metadata["pid"])
    pgid = int(metadata["pgid"])
    if metadata.get("workspace") != str(ROOT):
        raise RuntimeError(f"run {run_id} belongs to a different workspace")
    if not pathlib.Path(f"/proc/{pid}").exists():
        raise ProcessLookupError(f"run {run_id} PID {pid} no longer exists")
    if process_start_ticks(pid) != int(metadata["start_ticks"]):
        raise RuntimeError(f"PID {pid} was reused; refusing to signal it")
    if os.getpgid(pid) != pgid:
        raise RuntimeError(f"PID {pid} moved process groups; refusing to signal it")
    environment = process_environment(pid)
    if environment.get("PYPTO_RUN_ID") != run_id:
        raise RuntimeError(f"PID {pid} lacks matching PYPTO_RUN_ID; refusing to signal it")
    if environment.get("PYPTO_WORKSPACE_ROOT") != str(ROOT):
        raise RuntimeError(f"PID {pid} lacks matching workspace marker; refusing to signal it")
    return pid, pgid


def signal_verified(
    metadata: dict[str, object],
    requested_signal: signal.Signals = signal.SIGTERM,
) -> tuple[int, int]:
    """Verify ownership immediately before signaling the recorded group."""

    pid, pgid = verify(metadata)
    os.killpg(pgid, requested_signal)
    return pid, pgid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()
    metadata_path = ROOT / "runs" / args.run_id / "process.json"
    metadata = json.loads(metadata_path.read_text())
    pid, pgid = verify(metadata)
    print(f"verified workspace run {args.run_id}: PID={pid} PGID={pgid}")
    if args.check_only:
        return 0
    pid, _ = signal_verified(metadata)
    deadline = time.monotonic() + args.wait_seconds
    while pathlib.Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.25)
    if pathlib.Path(f"/proc/{pid}").exists():
        print(
            f"run {args.run_id} did not exit after SIGTERM; no escalation was performed",
            file=sys.stderr,
        )
        return 75
    return 0


if __name__ == "__main__":
    sys.exit(main())
