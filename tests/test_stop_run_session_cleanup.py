from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import signal
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def load_stop_run():
    path = REPOSITORY / "tools/stop_run.py"
    spec = importlib.util.spec_from_file_location("test_stop_run_session", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stop_run = load_stop_run()


def _metadata(workspace: Path, sid: int = 900) -> dict[str, object]:
    run_id = "owned-run"
    run_dir = workspace / "runs" / run_id
    tmpdir = run_dir / "tmp"
    tmpdir.mkdir(parents=True)
    return {
        "schema": 2,
        "mode": "gpu-bounded",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "framework_profile": "pypto",
        "pid": sid,
        "pgid": sid,
        "sid": sid,
        "start_ticks": 1,
        "tmpdir": str(tmpdir),
    }


def _write_process(
    proc: Path,
    metadata: dict[str, object],
    *,
    pid: int,
    pgid: int,
    sid: int,
    start_ticks: int,
    cwd: Path,
    environment_changes: dict[str, str] | None = None,
) -> None:
    root = proc / str(pid)
    root.mkdir()
    fields = ["S", "1", str(pgid), str(sid), *(["0"] * 15), str(start_ticks)]
    (root / "stat").write_text(f"{pid} (tileiras {pid}) " + " ".join(fields))
    (root / "cwd").symlink_to(cwd, target_is_directory=True)
    environment = {
        "PYPTO_RUN_ID": str(metadata["run_id"]),
        "PYPTO_WORKSPACE_ROOT": str(metadata["workspace"]),
        "PYPTO_RUN_MODE": str(metadata["mode"]),
        "PYPTO_FRAMEWORK_PROFILE": str(metadata["framework_profile"]),
        "TMPDIR": str(metadata["tmpdir"]),
    }
    environment.update(environment_changes or {})
    (root / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in environment.items())
        + b"\0"
    )


def _write_sanitized_tileiras(
    proc: Path,
    metadata: dict[str, object],
    *,
    pid: int,
    pgid: int,
    sid: int,
    start_ticks: int,
    cwd: Path,
) -> None:
    _write_process(
        proc,
        metadata,
        pid=pid,
        pgid=pgid,
        sid=sid,
        start_ticks=start_ticks,
        cwd=cwd,
    )
    compiler_name = "tensor-ir-a1b2c3"
    compiler_dir = Path(metadata["run_dir"]) / "tmp" / compiler_name
    compiler_dir.mkdir(parents=True, exist_ok=True)
    executable = compiler_dir / "tensor-ir-123456.tileiras"
    executable.write_bytes(b"verified tileiras")
    output = Path(metadata["tmpdir"]) / compiler_name / "tensor-ir-output.cubin"
    bytecode = Path(metadata["tmpdir"]) / compiler_name / "tensor-ir-input.tilebc"
    process_root = proc / str(pid)
    (process_root / "exe").symlink_to(executable)
    (process_root / "cmdline").write_bytes(
        b"\0".join(
            value.encode()
            for value in (
                str(Path(metadata["tmpdir"]) / compiler_name / executable.name),
                "--gpu-name=sm_120",
                f"--output-file={output}",
                str(bytecode),
            )
        )
        + b"\0"
    )
    environment = {
        "PATH": "/producer:/usr/bin:/bin",
        "CUDA_HOME": "/usr/local/cuda-13.3",
        "CUDA_PATH": "/usr/local/cuda-13.3",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "HOME": "/nonexistent",
        "TMPDIR": str(Path(metadata["tmpdir"]) / compiler_name),
        "CUDA_CACHE_DISABLE": "1",
    }
    (process_root / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in environment.items())
        + b"\0"
    )


def test_leader_gone_cleanup_terminates_only_verified_same_sid_residuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_process(
        proc,
        metadata,
        pid=101,
        pgid=501,
        sid=900,
        start_ticks=11,
        cwd=workspace,
    )
    _write_process(
        proc,
        metadata,
        pid=102,
        pgid=502,
        sid=900,
        start_ticks=12,
        cwd=workspace / "runs/owned-run",
    )
    _write_process(
        proc,
        metadata,
        pid=103,
        pgid=503,
        sid=901,
        start_ticks=13,
        cwd=workspace,
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    verified, rejected = stop_run.verified_session_members(metadata, proc)
    assert [item["pid"] for item in verified] == [101, 102]
    assert rejected == []

    signals = []

    def send(pid: int, requested: signal.Signals) -> None:
        signals.append((pid, requested))
        if requested == signal.SIGKILL:
            shutil.rmtree(proc / str(pid))

    result = stop_run.terminate_verified_session_residuals(
        metadata,
        term_wait_seconds=0,
        kill_wait_seconds=0.1,
        poll_seconds=0.001,
        proc_root=proc,
        send_signal=send,
    )
    assert result["complete"] is True
    assert result["rejected"] == []
    assert result["survivors"] == []
    assert signals == [
        (101, signal.SIGTERM),
        (102, signal.SIGTERM),
        (101, signal.SIGKILL),
        (102, signal.SIGKILL),
    ]
    assert (proc / "103").is_dir()


def test_pid_reuse_is_rejected_immediately_before_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_process(
        proc,
        metadata,
        pid=201,
        pgid=601,
        sid=900,
        start_ticks=21,
        cwd=workspace,
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    snapshot = stop_run.session_member_snapshot(201, metadata, proc)
    stat = (proc / "201/stat").read_text()
    (proc / "201/stat").write_text(stat.rsplit(" ", 1)[0] + " 99")
    signals = []
    with pytest.raises(stop_run.SessionOwnershipError, match="identity changed"):
        stop_run.signal_verified_session_member(
            metadata,
            snapshot,
            signal.SIGTERM,
            proc_root=proc,
            send_signal=lambda pid, requested: signals.append((pid, requested)),
        )
    assert signals == []


def test_same_sid_with_wrong_identity_is_rejected_and_different_sid_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_process(
        proc,
        metadata,
        pid=301,
        pgid=701,
        sid=900,
        start_ticks=31,
        cwd=workspace,
        environment_changes={"PYPTO_RUN_ID": "foreign-run"},
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_process(
        proc,
        metadata,
        pid=302,
        pgid=702,
        sid=900,
        start_ticks=32,
        cwd=outside,
    )
    _write_process(
        proc,
        metadata,
        pid=303,
        pgid=703,
        sid=901,
        start_ticks=33,
        cwd=workspace,
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    verified, rejected = stop_run.verified_session_members(metadata, proc)
    assert verified == []
    assert [item["pid"] for item in rejected] == [301, 302]
    signals = []
    result = stop_run.terminate_verified_session_residuals(
        metadata,
        term_wait_seconds=0,
        kill_wait_seconds=0,
        poll_seconds=0.001,
        proc_root=proc,
        send_signal=lambda pid, requested: signals.append((pid, requested)),
    )
    assert result["complete"] is False
    assert signals == []
    assert (proc / "303").is_dir()


def test_session_metadata_rejects_schema_leader_and_tmpdir_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    monkeypatch.setattr(stop_run, "ROOT", workspace)

    assert stop_run._session_metadata(metadata)[0] == metadata["sid"]

    drifted = dict(metadata, schema=1)
    with pytest.raises(stop_run.SessionOwnershipError, match="incomplete"):
        stop_run._session_metadata(drifted)

    drifted = dict(metadata, pgid=int(metadata["pgid"]) + 1)
    with pytest.raises(stop_run.SessionOwnershipError, match="incomplete"):
        stop_run._session_metadata(drifted)

    drifted = dict(metadata, tmpdir=str(tmp_path / "foreign"))
    with pytest.raises(stop_run.SessionOwnershipError, match="not owned"):
        stop_run._session_metadata(drifted)


def test_gpu_short_tmp_alias_metadata_remains_valid_after_alias_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    alias = Path("/tmp/pypto-ipc-test/t")
    metadata["tmpdir"] = str(alias)
    metadata["short_tmp_alias"] = {
        "path": str(alias),
        "target": str(Path(metadata["run_dir"]) / "tmp"),
    }
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    assert stop_run._session_metadata(metadata)[3] == str(alias)


def test_marked_worker_accepts_only_tmpdir_below_owned_run_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    proc = tmp_path / "proc"
    proc.mkdir()
    nested_tmp = Path(metadata["run_dir"]) / "temporary"
    nested_tmp.mkdir()
    _write_process(
        proc,
        metadata,
        pid=351,
        pgid=351,
        sid=int(metadata["sid"]),
        start_ticks=35,
        cwd=workspace,
        environment_changes={"TMPDIR": str(nested_tmp)},
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    snapshot = stop_run.session_member_snapshot(351, metadata, proc)
    assert snapshot["identity_kind"] == "run-environment"
    assert snapshot["tmpdir"] == str(nested_tmp)

    foreign_tmp = tmp_path / "foreign"
    foreign_tmp.mkdir()
    environment = (proc / "351/environ").read_bytes().replace(
        str(nested_tmp).encode(), str(foreign_tmp).encode()
    )
    (proc / "351/environ").write_bytes(environment)
    with pytest.raises(stop_run.SessionOwnershipError, match="TMPDIR"):
        stop_run.session_member_snapshot(351, metadata, proc)


def test_natural_cleanup_requires_zero_signals_and_no_residuals() -> None:
    cleanup = {
        "schema": 1,
        "kind": "pypto-owned-session-cleanup",
        "term_signaled": [],
        "kill_signaled": [],
        "rejected": [],
        "survivors": [],
        "complete": True,
    }
    assert stop_run.session_cleanup_is_natural(cleanup)
    for field in ("term_signaled", "kill_signaled", "rejected", "survivors"):
        drifted = dict(cleanup)
        drifted[field] = [{"pid": 1}]
        assert not stop_run.session_cleanup_is_natural(drifted)
    assert not stop_run.session_cleanup_is_natural(dict(cleanup, complete=False))


def test_natural_wait_accepts_a_residual_that_exits_without_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_process(
        proc,
        metadata,
        pid=401,
        pgid=801,
        sid=int(metadata["sid"]),
        start_ticks=41,
        cwd=workspace,
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    monkeypatch.setattr(
        stop_run.time,
        "sleep",
        lambda _seconds: shutil.rmtree(proc / "401"),
    )
    signals = []
    cleanup = stop_run.terminate_verified_session_residuals(
        metadata,
        natural_wait_seconds=1.0,
        term_wait_seconds=0,
        kill_wait_seconds=0,
        poll_seconds=0.001,
        proc_root=proc,
        send_signal=lambda pid, requested: signals.append((pid, requested)),
    )
    assert signals == []
    assert stop_run.session_cleanup_is_natural(cleanup)


def test_sanitized_tileiras_child_is_verified_by_sid_and_owned_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    alias = Path("/tmp/pypto-ipc-test/t")
    metadata["tmpdir"] = str(alias)
    metadata["short_tmp_alias"] = {
        "path": str(alias),
        "target": str(Path(metadata["run_dir"]) / "tmp"),
    }
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_sanitized_tileiras(
        proc,
        metadata,
        pid=501,
        pgid=501,
        sid=int(metadata["sid"]),
        start_ticks=51,
        cwd=workspace,
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    snapshot = stop_run.session_member_snapshot(501, metadata, proc)
    assert snapshot["identity_kind"] == "sanitized-tileiras"
    assert snapshot["sid"] == metadata["sid"]

    signals = []

    def send(pid: int, requested: signal.Signals) -> None:
        signals.append((pid, requested))
        shutil.rmtree(proc / str(pid))

    cleanup = stop_run.terminate_verified_session_residuals(
        metadata,
        term_wait_seconds=0,
        kill_wait_seconds=0,
        poll_seconds=0.001,
        proc_root=proc,
        send_signal=send,
    )
    assert cleanup["complete"] is True
    assert cleanup["rejected"] == []
    assert signals == [(501, signal.SIGTERM)]


def test_sanitized_tileiras_rejects_foreign_paths_and_marker_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_sanitized_tileiras(
        proc,
        metadata,
        pid=601,
        pgid=601,
        sid=int(metadata["sid"]),
        start_ticks=61,
        cwd=workspace,
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    process_root = proc / "601"
    arguments = (process_root / "cmdline").read_bytes().split(b"\0")
    arguments[2] = b"--output-file=/tmp/foreign.cubin"
    (process_root / "cmdline").write_bytes(b"\0".join(arguments))
    with pytest.raises(stop_run.SessionOwnershipError, match="escaped"):
        stop_run.session_member_snapshot(601, metadata, proc)

    _write_process(
        proc,
        metadata,
        pid=602,
        pgid=602,
        sid=int(metadata["sid"]),
        start_ticks=62,
        cwd=workspace,
        environment_changes={"PYPTO_RUN_ID": "foreign-run"},
    )
    with pytest.raises(stop_run.SessionOwnershipError, match="differs"):
        stop_run.session_member_snapshot(602, metadata, proc)


def test_sanitized_tileiras_allows_unreadable_execve_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_sanitized_tileiras(
        proc,
        metadata,
        pid=701,
        pgid=701,
        sid=int(metadata["sid"]),
        start_ticks=71,
        cwd=workspace,
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    monkeypatch.setattr(
        stop_run,
        "process_environment",
        lambda _pid, _proc_root=proc: (_ for _ in ()).throw(PermissionError()),
    )
    snapshot = stop_run.session_member_snapshot(701, metadata, proc)
    assert snapshot["identity_kind"] == "sanitized-tileiras"


def test_transient_rejection_does_not_poison_a_naturally_empty_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    metadata = _metadata(workspace)
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_process(
        proc,
        metadata,
        pid=801,
        pgid=801,
        sid=int(metadata["sid"]),
        start_ticks=81,
        cwd=workspace,
        environment_changes={"PYPTO_RUN_ID": "foreign-run"},
    )
    monkeypatch.setattr(stop_run, "ROOT", workspace)
    monkeypatch.setattr(
        stop_run.time,
        "sleep",
        lambda _seconds: shutil.rmtree(proc / "801"),
    )
    cleanup = stop_run.terminate_verified_session_residuals(
        metadata,
        natural_wait_seconds=1.0,
        term_wait_seconds=0,
        kill_wait_seconds=0,
        poll_seconds=0.001,
        proc_root=proc,
        send_signal=lambda _pid, _requested: pytest.fail("signal was not expected"),
    )
    assert cleanup["complete"] is True
    assert cleanup["rejected"] == []
    assert cleanup["rejected_observations"]
