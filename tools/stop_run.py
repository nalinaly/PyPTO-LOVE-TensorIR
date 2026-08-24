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


def process_stat(pid: int) -> tuple[int, int]:
    tail = pathlib.Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    if len(tail) <= 19:
        raise RuntimeError(f"malformed /proc/{pid}/stat")
    return int(tail[19]), int(tail[2])


def process_start_ticks(pid: int) -> int:
    return process_stat(pid)[0]


def process_group_members(pgid: int) -> list[int]:
    members: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            _start_ticks, observed_pgid = process_stat(pid)
        except (OSError, RuntimeError, ValueError):
            continue
        if observed_pgid == pgid:
            members.append(pid)
    return sorted(members)


def owned_group_members(metadata: dict[str, object]) -> list[int]:
    run_id = str(metadata["run_id"])
    pgid = int(metadata["pgid"])
    members = process_group_members(pgid)
    for pid in members:
        environment = process_environment(pid)
        if environment.get("PYPTO_RUN_ID") != run_id:
            raise RuntimeError(
                f"PGID {pgid} contains PID {pid} without matching run identity"
            )
        if environment.get("PYPTO_WORKSPACE_ROOT") != str(ROOT):
            raise RuntimeError(
                f"PGID {pgid} contains PID {pid} without matching workspace"
            )
    return members


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

    try:
        pid, pgid = verify(metadata)
    except ProcessLookupError:
        members = owned_group_members(metadata)
        if not members:
            raise
        pid = members[0]
        pgid = int(metadata["pgid"])
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
    try:
        pid, pgid = verify(metadata)
        verification = f"leader PID={pid}"
    except ProcessLookupError:
        members = owned_group_members(metadata)
        if not members:
            raise
        pid = members[0]
        pgid = int(metadata["pgid"])
        verification = f"orphaned owned members={members}"
    print(f"verified workspace run {args.run_id}: {verification} PGID={pgid}")
    if args.check_only:
        return 0
    pid, _ = signal_verified(metadata)
    try:
        signal_verified(metadata, signal.SIGCONT)
    except ProcessLookupError:
        return 0
    deadline = time.monotonic() + args.wait_seconds
    pgid = int(metadata["pgid"])
    while process_group_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if process_group_members(pgid):
        signal_verified(metadata, signal.SIGSTOP)
        print(
            f"run {args.run_id} group did not exit after SIGTERM; "
            "no kill escalation was performed",
            file=sys.stderr,
        )
        return 75
    return 0


if __name__ == "__main__":
    sys.exit(main())
