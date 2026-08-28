from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/windows/capture_terminal.ps1"


def test_capture_script_has_owned_window_and_output_guards() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Find-WindowByExactTitle" in source
    assert "GetWindowRect" in source
    assert "SetForegroundWindow" in source
    assert "CopyFromScreen" in source
    assert "OutputPath must be an absolute Windows path" in source
    assert "screenshot retained" in source
    assert "PostMessage($window" in source
    assert "Stop-Process" not in source
    assert "/home/" not in source


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="not WSL")
def test_capture_script_parses_in_windows_powershell() -> None:
    windows_path = subprocess.run(
        ["wslpath", "-w", str(SCRIPT)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    escaped_path = windows_path.replace("'", "''")
    command = (
        "$tokens=$null; $errors=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{escaped_path}', [ref]$tokens, [ref]$errors); "
        "if ($errors.Count) { exit 1 }"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert completed.returncode == 0
