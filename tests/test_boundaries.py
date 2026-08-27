from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_framework_adapter_contains_no_kernel_dsl() -> None:
    forbidden = ("@pl.program", "cuda_tile.Mma", "qwen35_9b_kernel")
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text()
        for marker in forbidden:
            assert marker not in text, f"kernel/model algorithm marker {marker!r} in {path}"


def test_entry_points_are_separate() -> None:
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    entry_points = config["project"]["entry-points"]
    assert entry_points["torch_dynamo_backends"]["pypto"].endswith(":compile_backend")
    assert entry_points["sglang.srt.plugins"]["pypto"].endswith(":register")
    assert config["project"]["dependencies"] == ["pypto-kernels==0.1.0"]
