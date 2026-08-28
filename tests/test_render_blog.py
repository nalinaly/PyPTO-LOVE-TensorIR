from __future__ import annotations

import base64
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import render_blog


def test_render_embeds_local_image_and_has_no_external_asset(tmp_path: Path) -> None:
    image = tmp_path / "proof.png"
    payload = b"\x89PNG\r\n\x1a\nrelease-proof"
    image.write_bytes(payload)
    report = tmp_path / "report.md"
    report.write_text("# 报告\n\n![proof](proof.png)\n", encoding="utf-8")

    rendered = render_blog.render(report)

    assert "data:image/png;base64," + base64.b64encode(payload).decode() in rendered
    assert "<style>" in rendered
    assert '<html lang="zh-CN">' in rendered
    assert 'src="http' not in rendered


def test_render_rejects_remote_image(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# Report\n\n![remote](https://example.com/x.png)\n")

    with pytest.raises(ValueError, match="remote image"):
        render_blog.render(report)


def test_render_rejects_missing_image(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# Report\n\n![missing](missing.png)\n")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        render_blog.render(report)
