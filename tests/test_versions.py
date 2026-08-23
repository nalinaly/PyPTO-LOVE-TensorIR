from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pypto_plugins import sglang_plugin, versions
from pypto_plugins.errors import FrameworkCompatibilityError
from pypto_plugins.versions import (
    EXPECTED_SGLANG_COMMIT,
    EXPECTED_TORCH_COMMIT,
    assert_sglang_compatible,
    assert_torch_compatible,
)


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def get_device_capability(self, _device: int) -> tuple[int, int]:
        return (12, 0)


def fake_torch(
    root, *, version="2.13.0+cu130", commit=EXPECTED_TORCH_COMMIT, hip=None
):
    module_path = root / "lib" / "python" / "torch" / "__init__.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("")
    return SimpleNamespace(
        __version__=version,
        __file__=str(module_path),
        version=SimpleNamespace(git_version=commit, cuda="13.0", hip=hip),
        cuda=FakeCuda(),
    )


def write_environment_lock(workspace_root, torch_module) -> None:
    (workspace_root / "ENVIRONMENT.lock").write_text(
        json.dumps(
            {
                "status": "cloned",
                "torch_file": torch_module.__file__,
                "torch_tree_sha256": "a" * 64,
            }
        )
    )


def fake_sglang(source_root):
    module_path = source_root / "python" / "sglang" / "__init__.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("")
    return SimpleNamespace(
        __version__="0.5.18",
        __file__=str(module_path),
    )


def test_torch_exact_cuda_commit_is_accepted(tmp_path) -> None:
    environment = tmp_path / "environment"
    module = fake_torch(environment)
    write_environment_lock(tmp_path, module)
    assert_torch_compatible(
        module, environment_root=environment, workspace_root=tmp_path
    )


def test_torch_rocm_build_fails_closed(tmp_path) -> None:
    with pytest.raises(FrameworkCompatibilityError, match="NVIDIA CUDA build"):
        assert_torch_compatible(
            fake_torch(tmp_path, hip="7.2"),
            environment_root=tmp_path,
            workspace_root=tmp_path,
        )


def test_torch_version_mismatch_fails_closed(tmp_path) -> None:
    with pytest.raises(FrameworkCompatibilityError):
        assert_torch_compatible(
            fake_torch(tmp_path, version="2.13.1"),
            environment_root=tmp_path,
            workspace_root=tmp_path,
        )


def test_sglang_requires_source_identity(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PYPTO_SGLANG_SOURCE_ROOT", raising=False)
    with pytest.raises(FrameworkCompatibilityError, match="PYPTO_SGLANG_SOURCE_ROOT"):
        assert_sglang_compatible(
            installed_version="0.5.18",
            source_root=None,
            sglang_module=fake_sglang(tmp_path),
        )


def test_sglang_import_must_come_from_locked_checkout(monkeypatch, tmp_path) -> None:
    source_root = tmp_path / "source"
    other_root = tmp_path / "other"
    monkeypatch.setattr(
        versions,
        "_git_output",
        lambda _root, command, *args: EXPECTED_SGLANG_COMMIT
        if command == "rev-parse"
        else "",
    )
    with pytest.raises(FrameworkCompatibilityError, match="imported sglang"):
        assert_sglang_compatible(
            source_root=source_root,
            sglang_module=fake_sglang(other_root),
        )


def test_sglang_registration_error_exits_fail_closed(monkeypatch) -> None:
    def fail_registration() -> None:
        raise FrameworkCompatibilityError("wrong SGLang source")

    monkeypatch.setattr(sglang_plugin, "_register_impl", fail_registration)
    with pytest.raises(SystemExit, match="wrong SGLang source"):
        sglang_plugin.register()
