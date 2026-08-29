#!/usr/bin/env python3
"""Terminate only processes proven to belong to a recorded PyPTO run."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import sys
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]


class GroupRevalidationError(RuntimeError):
    """A follow-up signal could not safely revalidate the recorded group."""

    def __init__(self, message: str, members: list[int] | None = None):
        super().__init__(message)
        self.members = members


class SessionOwnershipError(RuntimeError):
    """A same-session PID did not retain the recorded run identity."""


def process_stat_full(
    pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> tuple[int, int, int, str]:
    tail = (proc_root / str(pid) / "stat").read_text().rpartition(")")[2].split()
    if len(tail) <= 19:
        raise RuntimeError(f"malformed /proc/{pid}/stat")
    return int(tail[19]), int(tail[2]), int(tail[3]), tail[0]


def process_stat(pid: int) -> tuple[int, int]:
    start_ticks, pgid, _sid, _state = process_stat_full(pid)
    return start_ticks, pgid


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


def exact_process_group_members(pgid: int) -> list[int]:
    """Return members, or prove through the kernel that the exact PGID is gone."""

    members = process_group_members(pgid)
    if members:
        return members
    try:
        # Signal zero has no process effect. It distinguishes a genuinely empty
        # group from a /proc scan that could not enumerate an existing member.
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return []
    except OSError as error:
        raise RuntimeError(
            f"could not establish whether recorded PGID {pgid} is empty"
        ) from error
    raise RuntimeError(
        f"recorded PGID {pgid} still exists but its members could not be enumerated"
    )


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


def process_environment(
    pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> dict[str, str]:
    raw = (proc_root / str(pid) / "environ").read_bytes()
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def process_cwd(pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")) -> pathlib.Path:
    return pathlib.Path(os.readlink(proc_root / str(pid) / "cwd")).resolve()


def process_executable(
    pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> pathlib.Path:
    return pathlib.Path(os.readlink(proc_root / str(pid) / "exe")).resolve()


def process_arguments(
    pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> list[str]:
    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    return [item.decode(errors="replace") for item in raw.split(b"\0") if item]


def _session_metadata(metadata: dict[str, object]) -> tuple[int, str, pathlib.Path, str]:
    if metadata.get("workspace") != str(ROOT):
        raise SessionOwnershipError("recorded session belongs to another workspace")
    run_id = metadata.get("run_id")
    pid = metadata.get("pid")
    pgid = metadata.get("pgid")
    sid = metadata.get("sid")
    run_dir = metadata.get("run_dir")
    tmpdir = metadata.get("tmpdir")
    if (
        metadata.get("schema") != 2
        or type(run_id) is not str
        or not run_id
        or type(pid) is not int
        or type(pgid) is not int
        or type(sid) is not int
        or pid <= 0
        or pid != pgid
        or pid != sid
        or type(run_dir) is not str
        or type(tmpdir) is not str
    ):
        raise SessionOwnershipError("recorded session identity is incomplete")
    expected_run_dir = (ROOT / "runs" / run_id).resolve()
    if pathlib.Path(run_dir).resolve() != expected_run_dir:
        raise SessionOwnershipError("recorded run directory differs from run identity")
    expected_tmpdir = (expected_run_dir / "tmp").resolve()
    recorded_tmpdir = pathlib.Path(tmpdir)
    if recorded_tmpdir == expected_tmpdir:
        return sid, run_id, expected_run_dir, tmpdir
    alias = metadata.get("short_tmp_alias")
    if (
        not recorded_tmpdir.is_absolute()
        or recorded_tmpdir.name != "t"
        or recorded_tmpdir.parent.parent != pathlib.Path("/tmp")
        or not recorded_tmpdir.parent.name.startswith("pypto-ipc-")
        or not isinstance(alias, dict)
        or alias.get("path") != tmpdir
        or alias.get("target") != str(expected_tmpdir)
    ):
        raise SessionOwnershipError("recorded TMPDIR is not owned by the run")
    return sid, run_id, expected_run_dir, tmpdir


def _relative_to_owned_tmp(
    raw: str, metadata: dict[str, object]
) -> tuple[pathlib.Path, pathlib.Path]:
    _sid, _run_id, run_dir, tmpdir = _session_metadata(metadata)
    path = pathlib.Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise SessionOwnershipError(f"native child path is not absolute: {raw}")
    for root in (pathlib.Path(tmpdir), run_dir / "tmp"):
        try:
            return path.relative_to(root), root
        except ValueError:
            continue
    raise SessionOwnershipError(f"native child path escaped owned TMPDIR: {raw}")


def _sanitized_tileiras_snapshot(
    pid: int,
    metadata: dict[str, object],
    *,
    start_ticks: int,
    pgid: int,
    sid: int,
    cwd: pathlib.Path,
    environment: dict[str, str] | None,
    proc_root: pathlib.Path,
) -> dict[str, object]:
    executable = process_executable(pid, proc_root)
    arguments = process_arguments(pid, proc_root)
    executable_relative, _root = _relative_to_owned_tmp(str(executable), metadata)
    if (
        len(executable_relative.parts) != 2
        or not executable_relative.parts[0].startswith("tensor-ir-")
        or not executable_relative.name.startswith("tensor-ir-")
        or executable_relative.suffix != ".tileiras"
        or len(arguments) != 4
        or arguments[1] != "--gpu-name=sm_120"
        or not arguments[2].startswith("--output-file=")
    ):
        raise SessionOwnershipError(
            f"PID {pid} is not a canonical sanitized tileiras child"
        )
    compiler_directory = executable_relative.parts[0]
    path_arguments = (
        arguments[0],
        arguments[2].removeprefix("--output-file="),
        arguments[3],
    )
    relatives = [
        _relative_to_owned_tmp(value, metadata)[0] for value in path_arguments
    ]
    if (
        any(len(relative.parts) != 2 for relative in relatives)
        or any(relative.parts[0] != compiler_directory for relative in relatives)
        or relatives[0].name != executable_relative.name
        or relatives[1].suffix != ".cubin"
        or relatives[2].suffix != ".tilebc"
    ):
        raise SessionOwnershipError(
            f"PID {pid} tileiras arguments escaped its compiler directory"
        )
    if environment is not None:
        expected_names = {
            "PATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "LANG",
            "LC_ALL",
            "TZ",
            "HOME",
            "TMPDIR",
            "CUDA_CACHE_DISABLE",
        }
        if (
            set(environment) != expected_names
            or environment.get("LANG") != "C"
            or environment.get("LC_ALL") != "C"
            or environment.get("TZ") != "UTC"
            or environment.get("HOME") != "/nonexistent"
            or environment.get("CUDA_CACHE_DISABLE") != "1"
        ):
            raise SessionOwnershipError(
                f"PID {pid} tileiras environment differs from the sanitized contract"
            )
        tmp_relative, _tmp_root = _relative_to_owned_tmp(
            environment["TMPDIR"], metadata
        )
        if tmp_relative != pathlib.Path(compiler_directory):
            raise SessionOwnershipError(
                f"PID {pid} tileiras TMPDIR differs from its compiler directory"
            )
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "pgid": pgid,
        "sid": sid,
        "cwd": str(cwd),
        "tmpdir": str(pathlib.Path(metadata["tmpdir"]) / compiler_directory),
        "identity_kind": "sanitized-tileiras",
        "executable": str(executable),
        "arguments": arguments,
    }


def session_process_ids(
    sid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> list[int]:
    members = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            _start_ticks, _pgid, observed_sid, state = process_stat_full(pid, proc_root)
        except (OSError, RuntimeError, ValueError):
            continue
        if observed_sid == sid and state != "Z":
            members.append(pid)
    return sorted(members)


def session_member_snapshot(
    pid: int,
    metadata: dict[str, object],
    proc_root: pathlib.Path = pathlib.Path("/proc"),
) -> dict[str, object]:
    sid, run_id, _run_dir, expected_tmpdir = _session_metadata(metadata)
    start_ticks, pgid, observed_sid, state = process_stat_full(pid, proc_root)
    if observed_sid != sid or state == "Z":
        raise SessionOwnershipError(f"PID {pid} left the recorded live session")
    cwd = process_cwd(pid, proc_root)
    workspace = ROOT.resolve()
    if cwd != workspace and workspace not in cwd.parents:
        raise SessionOwnershipError(
            f"PID {pid} cwd escaped the recorded workspace: {cwd}"
        )
    try:
        environment = process_environment(pid, proc_root)
    except PermissionError:
        return _sanitized_tileiras_snapshot(
            pid,
            metadata,
            start_ticks=start_ticks,
            pgid=pgid,
            sid=observed_sid,
            cwd=cwd,
            environment=None,
            proc_root=proc_root,
        )
    expected_mode = metadata.get("mode")
    expected_profile = metadata.get("framework_profile")
    mismatches = {
        name: (environment.get(name), expected)
        for name, expected in (
            ("PYPTO_RUN_ID", run_id),
            ("PYPTO_WORKSPACE_ROOT", str(ROOT)),
            ("PYPTO_RUN_MODE", expected_mode),
            ("PYPTO_FRAMEWORK_PROFILE", expected_profile),
            ("TMPDIR", expected_tmpdir),
        )
        if type(expected) is not str or environment.get(name) != expected
    }
    if mismatches:
        control_names = {
            "PYPTO_RUN_ID",
            "PYPTO_WORKSPACE_ROOT",
            "PYPTO_RUN_MODE",
            "PYPTO_FRAMEWORK_PROFILE",
        }
        if control_names.isdisjoint(environment):
            return _sanitized_tileiras_snapshot(
                pid,
                metadata,
                start_ticks=start_ticks,
                pgid=pgid,
                sid=observed_sid,
                cwd=cwd,
                environment=environment,
                proc_root=proc_root,
            )
        raise SessionOwnershipError(
            f"PID {pid} environment differs from recorded run: {mismatches}"
        )
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "pgid": pgid,
        "sid": observed_sid,
        "cwd": str(cwd),
        "tmpdir": environment["TMPDIR"],
        "identity_kind": "run-environment",
    }


def verified_session_members(
    metadata: dict[str, object],
    proc_root: pathlib.Path = pathlib.Path("/proc"),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sid, _run_id, _run_dir, _tmpdir = _session_metadata(metadata)
    verified = []
    rejected = []
    for pid in session_process_ids(sid, proc_root):
        try:
            verified.append(session_member_snapshot(pid, metadata, proc_root))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, RuntimeError, ValueError) as error:
            rejected.append(
                {"pid": pid, "error": f"{type(error).__name__}: {error}"}
            )
    return verified, rejected


def signal_verified_session_member(
    metadata: dict[str, object],
    snapshot: dict[str, object],
    requested_signal: signal.Signals,
    *,
    proc_root: pathlib.Path = pathlib.Path("/proc"),
    send_signal=os.kill,
) -> bool:
    pid = int(snapshot["pid"])
    try:
        current = session_member_snapshot(pid, metadata, proc_root)
    except (FileNotFoundError, ProcessLookupError):
        return False
    if current != snapshot:
        raise SessionOwnershipError(
            f"PID {pid} identity changed before {requested_signal.name}; refusing signal"
        )
    try:
        send_signal(pid, requested_signal)
    except ProcessLookupError:
        return False
    return True


def terminate_verified_session_residuals(
    metadata: dict[str, object],
    *,
    natural_wait_seconds: float = 0.0,
    term_wait_seconds: float = 5.0,
    kill_wait_seconds: float = 2.0,
    poll_seconds: float = 0.1,
    proc_root: pathlib.Path = pathlib.Path("/proc"),
    send_signal=os.kill,
) -> dict[str, object]:
    if (
        natural_wait_seconds < 0
        or term_wait_seconds < 0
        or kill_wait_seconds < 0
        or poll_seconds <= 0
    ):
        raise ValueError("session cleanup waits must be non-negative and poll positive")
    sid, _run_id, _run_dir, _tmpdir = _session_metadata(metadata)
    term_signaled: dict[int, dict[str, object]] = {}
    kill_signaled: dict[int, dict[str, object]] = {}
    rejected_observations: dict[tuple[int, str], dict[str, object]] = {}

    def scan() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        verified, failures = verified_session_members(metadata, proc_root)
        for failure in failures:
            rejected_observations[
                (int(failure["pid"]), str(failure["error"]))
            ] = failure
        return verified, failures

    natural_deadline = time.monotonic() + natural_wait_seconds
    members, current_rejected = scan()
    while (members or current_rejected) and time.monotonic() < natural_deadline:
        time.sleep(poll_seconds)
        members, current_rejected = scan()

    term_deadline = time.monotonic() + term_wait_seconds
    while True:
        members, _failures = scan()
        for member in members:
            pid = int(member["pid"])
            if pid in term_signaled:
                continue
            try:
                if signal_verified_session_member(
                    metadata,
                    member,
                    signal.SIGTERM,
                    proc_root=proc_root,
                    send_signal=send_signal,
                ):
                    term_signaled[pid] = member
            except (OSError, RuntimeError, ValueError) as error:
                failure = {"pid": pid, "error": f"{type(error).__name__}: {error}"}
                rejected_observations[(pid, failure["error"])] = failure
        if not members or time.monotonic() >= term_deadline:
            break
        time.sleep(poll_seconds)

    remaining, _failures = scan()
    for member in remaining:
        pid = int(member["pid"])
        try:
            if signal_verified_session_member(
                metadata,
                member,
                signal.SIGKILL,
                proc_root=proc_root,
                send_signal=send_signal,
            ):
                kill_signaled[pid] = member
        except (OSError, RuntimeError, ValueError) as error:
            failure = {"pid": pid, "error": f"{type(error).__name__}: {error}"}
            rejected_observations[(pid, failure["error"])] = failure

    kill_deadline = time.monotonic() + kill_wait_seconds
    survivors, current_rejected = scan()
    while survivors and time.monotonic() < kill_deadline:
        time.sleep(poll_seconds)
        survivors, current_rejected = scan()
    return {
        "schema": 1,
        "kind": "pypto-owned-session-cleanup",
        "sid": sid,
        "term_signaled": [term_signaled[pid] for pid in sorted(term_signaled)],
        "kill_signaled": [kill_signaled[pid] for pid in sorted(kill_signaled)],
        "rejected": current_rejected,
        "rejected_observations": [
            rejected_observations[key] for key in sorted(rejected_observations)
        ],
        "survivors": survivors,
        "complete": not current_rejected and not survivors,
    }


def session_cleanup_is_natural(cleanup: object) -> bool:
    """Return true only when the run session drained without any signal."""

    return bool(
        isinstance(cleanup, dict)
        and cleanup.get("schema") == 1
        and cleanup.get("kind") == "pypto-owned-session-cleanup"
        and cleanup.get("complete") is True
        and cleanup.get("term_signaled") == []
        and cleanup.get("kill_signaled") == []
        and cleanup.get("rejected") == []
        and cleanup.get("survivors") == []
    )


def atomic_json(path: pathlib.Path, payload: object) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


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
    if "sid" in metadata and os.getsid(pid) != int(metadata["sid"]):
        raise RuntimeError(f"PID {pid} moved sessions; refusing to signal it")
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
        if "sid" in metadata:
            # A reused PGID is not a safe ownership token once the recorded
            # session leader is gone. Schema-2 callers continue with the
            # per-PID SID cleanup path instead of signaling this group.
            raise
        members = owned_group_members(metadata)
        if not members:
            raise
        pid = members[0]
        pgid = int(metadata["pgid"])
    os.killpg(pgid, requested_signal)
    return pid, pgid


def signal_verified_followup(
    metadata: dict[str, object],
    requested_signal: signal.Signals,
    verified_pgid: int,
) -> bool:
    """Revalidate a follow-up signal after one signal was already verified.

    Return false only when the exact previously verified PGID is now empty.
    A surviving or uninspectable group is left untouched and reported as an
    ownership ambiguity.
    """

    pgid = int(metadata["pgid"])
    if pgid != verified_pgid:
        raise GroupRevalidationError(
            f"recorded PGID changed from {verified_pgid} to {pgid}; "
            f"refusing {requested_signal.name}"
        )
    try:
        signal_verified(metadata, requested_signal)
    except (OSError, RuntimeError) as verification_error:
        try:
            members = exact_process_group_members(pgid)
        except (OSError, RuntimeError) as inspection_error:
            raise GroupRevalidationError(
                f"could not revalidate recorded PGID {pgid} for "
                f"{requested_signal.name}, and group emptiness is ambiguous; "
                "refusing to signal"
            ) from inspection_error
        if not members:
            return False
        raise GroupRevalidationError(
            f"could not revalidate recorded PGID {pgid} for "
            f"{requested_signal.name}; members remain {members}; "
            "refusing to signal",
            members,
        ) from verification_error
    return True


def stop_legacy_group(
    metadata: dict[str, object], *, check_only: bool, wait_seconds: int
) -> int:
    """Preserve schema-1 PGID behavior; it cannot authorize SID residuals."""

    run_id = str(metadata["run_id"])
    try:
        pid, pgid = verify(metadata)
        verification = f"leader PID={pid}"
    except ProcessLookupError:
        members = owned_group_members(metadata)
        if not members:
            raise
        pgid = int(metadata["pgid"])
        verification = f"orphaned owned members={members}"
    print(f"verified legacy run {run_id}: {verification} PGID={pgid}")
    if check_only:
        return 0
    signal_verified(metadata)
    try:
        continued = signal_verified_followup(metadata, signal.SIGCONT, pgid)
    except GroupRevalidationError as error:
        print(f"run {run_id} stop aborted: {error}", file=sys.stderr)
        return 75
    if not continued:
        return 0
    deadline = time.monotonic() + wait_seconds
    while process_group_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if process_group_members(pgid):
        try:
            signal_verified_followup(metadata, signal.SIGSTOP, pgid)
        except GroupRevalidationError as error:
            print(f"run {run_id} stop aborted: {error}", file=sys.stderr)
            return 75
        print(
            f"run {run_id} legacy group did not exit after SIGTERM; "
            "no SID cleanup is authorized by schema 1",
            file=sys.stderr,
        )
        return 75
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--kill-wait-seconds", type=int, default=2)
    args = parser.parse_args()
    if args.wait_seconds < 0 or args.kill_wait_seconds < 0:
        parser.error("cleanup waits must be non-negative")
    metadata_path = ROOT / "runs" / args.run_id / "process.json"
    metadata = json.loads(metadata_path.read_text())
    if "sid" not in metadata:
        return stop_legacy_group(
            metadata,
            check_only=args.check_only,
            wait_seconds=args.wait_seconds,
        )
    session_members, rejected = verified_session_members(metadata)
    try:
        pid, pgid = verify(metadata)
        verification = f"leader PID={pid}"
    except ProcessLookupError:
        members = [] if "sid" in metadata else owned_group_members(metadata)
        pgid = int(metadata["pgid"])
        if members:
            verification = f"orphaned owned PGID members={members}"
        elif session_members:
            verification = "orphaned owned SID members=" + repr(
                [int(item["pid"]) for item in session_members]
            )
        elif rejected:
            raise SessionOwnershipError(
                f"recorded SID has only rejected members: {rejected}"
            )
        else:
            verification = "no live owned members"
    print(
        f"verified workspace run {args.run_id}: {verification} "
        f"PGID={pgid} SID={metadata.get('sid')}",
        flush=True,
    )
    if args.check_only:
        return 75 if rejected else 0

    primary_error = None
    primary_signaled = False
    try:
        signal_verified(metadata, signal.SIGTERM)
        primary_signaled = True
    except ProcessLookupError:
        pass
    except (OSError, RuntimeError) as error:
        primary_error = f"{type(error).__name__}: {error}"
    if primary_signaled:
        try:
            signal_verified_followup(metadata, signal.SIGCONT, pgid)
        except GroupRevalidationError as error:
            primary_error = f"{type(error).__name__}: {error}"
    cleanup = terminate_verified_session_residuals(
        metadata,
        term_wait_seconds=float(args.wait_seconds),
        kill_wait_seconds=float(args.kill_wait_seconds),
        poll_seconds=0.1,
    )
    cleanup["primary_pgid"] = {
        "pgid": pgid,
        "term_signaled": primary_signaled,
        "error": primary_error,
    }
    cleanup["complete"] = bool(cleanup["complete"] and primary_error is None)
    metadata["manual_stop"] = cleanup
    metadata["status"] = "stopped" if cleanup["complete"] else "stop-incomplete"
    atomic_json(metadata_path, metadata)
    if not cleanup["complete"]:
        print(
            f"run {args.run_id} cleanup incomplete: {cleanup}",
            file=sys.stderr,
        )
        return 75
    print(
        f"run {args.run_id} cleanup complete: "
        f"TERM={len(cleanup['term_signaled'])} "
        f"KILL={len(cleanup['kill_signaled'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
