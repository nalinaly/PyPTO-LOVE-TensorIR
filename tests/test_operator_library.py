from __future__ import annotations

import copy
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import pypto_kernels
from pypto_plugins.errors import FrameworkCompatibilityError
from pypto_plugins.operator_library import (
    EXPECTED_OPERATOR_ABI_DIGEST,
    EXPECTED_OPERATOR_ABI_SCHEMA_VERSION,
    EXPECTED_OPERATOR_LIBRARY_VERSION,
    EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST,
    _distribution_source_mode,
    _package_tree_digest,
    assert_operator_library_compatible,
    inspect_operator_library,
)


def fake_module():
    manifest = pypto_kernels.public_abi_manifest()
    values = {
        name: getattr(pypto_kernels, name)
        for name in manifest["abi"]["top_level_bindings"]
    }
    values.update(
        {
            "__version__": EXPECTED_OPERATOR_LIBRARY_VERSION,
            "PYPTO_KERNELS_ABI_SCHEMA_VERSION": EXPECTED_OPERATOR_ABI_SCHEMA_VERSION,
            "PYPTO_KERNELS_ABI_DIGEST": EXPECTED_OPERATOR_ABI_DIGEST,
            "public_abi_manifest": lambda: copy.deepcopy(manifest),
        }
    )
    return SimpleNamespace(**values)


def test_real_standalone_operator_wheel_matches_all_identities() -> None:
    snapshot = assert_operator_library_compatible()
    assert snapshot.version == pypto_kernels.__version__
    assert snapshot.abi_schema_version == EXPECTED_OPERATOR_ABI_SCHEMA_VERSION
    assert snapshot.abi_digest == EXPECTED_OPERATOR_ABI_DIGEST
    assert snapshot.package_tree_digest == EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST
    assert snapshot.distribution_name == "pypto-kernels"
    assert snapshot.resolved_origin == str(Path(pypto_kernels.__file__).resolve())
    with pytest.raises(AttributeError):
        snapshot.version = "changed"


def test_structural_inspection_uses_only_producer_manifest() -> None:
    snapshot = inspect_operator_library(fake_module())
    assert snapshot.version == EXPECTED_OPERATOR_LIBRARY_VERSION
    assert snapshot.abi_digest == EXPECTED_OPERATOR_ABI_DIGEST
    assert snapshot.package_tree_digest is None
    assert snapshot.distribution_name is None
    assert snapshot.resolved_origin is None


def test_version_schema_and_exported_digest_mismatches_fail_closed() -> None:
    wrong_version = fake_module()
    wrong_version.__version__ = "9.9.9"
    with pytest.raises(FrameworkCompatibilityError, match="version mismatch"):
        inspect_operator_library(wrong_version)

    wrong_schema = fake_module()
    wrong_schema.PYPTO_KERNELS_ABI_SCHEMA_VERSION = True
    with pytest.raises(FrameworkCompatibilityError, match="ABI schema mismatch"):
        inspect_operator_library(wrong_schema)

    wrong_digest = fake_module()
    wrong_digest.PYPTO_KERNELS_ABI_DIGEST = "0" * 64
    with pytest.raises(FrameworkCompatibilityError, match="exported ABI digest"):
        inspect_operator_library(wrong_digest)


def test_manifest_envelope_and_payload_are_recomputed_by_consumer() -> None:
    extra_envelope = fake_module()
    manifest = extra_envelope.public_abi_manifest()
    manifest["unexpected"] = True
    extra_envelope.public_abi_manifest = lambda: copy.deepcopy(manifest)
    with pytest.raises(FrameworkCompatibilityError, match="envelope mismatch"):
        inspect_operator_library(extra_envelope)

    wrong_version = fake_module()
    manifest = wrong_version.public_abi_manifest()
    manifest["package_version"] = "different"
    wrong_version.public_abi_manifest = lambda: copy.deepcopy(manifest)
    with pytest.raises(FrameworkCompatibilityError, match="package version disagree"):
        inspect_operator_library(wrong_version)

    modified_payload = fake_module()
    manifest = modified_payload.public_abi_manifest()
    manifest["abi"]["operators"]["gdn"]["constants"]["head_mapping"] = "modified"
    modified_payload.public_abi_manifest = lambda: copy.deepcopy(manifest)
    with pytest.raises(FrameworkCompatibilityError, match="does not match its digest"):
        inspect_operator_library(modified_payload)


def test_every_manifest_binding_must_exist_on_module() -> None:
    module = fake_module()
    binding = next(iter(module.public_abi_manifest()["abi"]["top_level_bindings"]))
    delattr(module, binding)
    with pytest.raises(FrameworkCompatibilityError, match="top-level bindings"):
        inspect_operator_library(module)


@pytest.mark.parametrize("binding_kind", ["symbol", "constant", "catalog", "spec"])
def test_runtime_binding_replacement_fails_closed(binding_kind) -> None:
    module = fake_module()
    if binding_kind == "symbol":
        module.validate_gdn_v1_launch = lambda: None
    elif binding_kind == "constant":
        module.GDN_V1_INACTIVE_OUTPUT_POLICY = "leave-unchanged"
    elif binding_kind == "catalog":
        module.DEFAULT_CATALOG = SimpleNamespace(families=())
    else:
        parameters = list(pypto_kernels.GDN_V1_SPEC.parameters)
        parameters[0] = replace(parameters[0], allowed_dtypes=("fp32",))
        module.GDN_V1_SPEC = replace(
            pypto_kernels.GDN_V1_SPEC,
            parameters=parameters,
        )
    with pytest.raises(FrameworkCompatibilityError, match="runtime binding|inspect"):
        inspect_operator_library(module)


def _copy_real_package(destination: Path) -> Path:
    source = Path(pypto_kernels.__file__).resolve().parent
    return Path(shutil.copytree(source, destination))


def test_package_tree_digest_is_deterministic_and_ignores_bytecode(tmp_path) -> None:
    package = _copy_real_package(tmp_path / "pypto_kernels")
    assert _package_tree_digest(package) == EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST
    cache = package / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "ignored.pyc").write_bytes(b"not production source")
    assert _package_tree_digest(package) == EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST


@pytest.mark.parametrize("mutation", ["modify", "add", "remove"])
def test_package_tree_digest_detects_every_source_set_change(tmp_path, mutation) -> None:
    package = _copy_real_package(tmp_path / "pypto_kernels")
    if mutation == "modify":
        init = package / "__init__.py"
        init.write_bytes(init.read_bytes() + b"\n# tampered\n")
    elif mutation == "add":
        (package / "shadow.py").write_text("VALUE = 1\n")
    else:
        next(path for path in package.rglob("*.py") if path.name != "__init__.py").unlink()
    assert _package_tree_digest(package) != EXPECTED_OPERATOR_PACKAGE_TREE_DIGEST


@pytest.mark.parametrize(
    "kind",
    ["file-link", "dir-link", "dangling-link", "fifo", "native", "sourceless"],
)
def test_package_tree_rejects_links_and_non_regular_sources(tmp_path, kind) -> None:
    package = _copy_real_package(tmp_path / "pypto_kernels")
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE = True\n")
    if kind == "file-link":
        os.symlink(outside, package / "linked.py")
    elif kind == "dir-link":
        external_directory = tmp_path / "external"
        external_directory.mkdir()
        (external_directory / "code.py").write_text("VALUE = 1\n")
        os.symlink(external_directory, package / "linked_directory")
    elif kind == "dangling-link":
        os.symlink(tmp_path / "missing.py", package / "dangling.py")
    elif kind == "fifo":
        os.mkfifo(package / "pipe.py")
    elif kind == "native":
        (package / "shadow.so").write_bytes(b"not an extension")
    else:
        (package / "shadow.pyc").write_bytes(b"not bytecode")
    with pytest.raises(
        FrameworkCompatibilityError,
        match="symbolic link|regular file|import payload",
    ):
        _package_tree_digest(package)


def test_package_tree_rejects_empty_source_tree(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FrameworkCompatibilityError, match="no Python sources"):
        _package_tree_digest(empty)


@pytest.mark.parametrize("link_level", ["root", "ancestor"])
def test_package_tree_rejects_symlinked_root_or_ancestor(tmp_path, link_level) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    package = _copy_real_package(real_parent / "pypto_kernels")
    if link_level == "root":
        linked = tmp_path / "linked-package"
        os.symlink(package, linked)
    else:
        linked_parent = tmp_path / "linked-parent"
        os.symlink(real_parent, linked_parent)
        linked = linked_parent / "pypto_kernels"
    with pytest.raises(FrameworkCompatibilityError, match="symbolic link"):
        _package_tree_digest(linked)


class FakeDistribution:
    def __init__(self, root: Path, files, direct_url: dict | str | None = None):
        self.root = root
        self.files = files
        self.direct_url = direct_url

    def locate_file(self, entry):
        return self.root / str(entry)

    def read_text(self, name: str):
        if name != "direct_url.json" or self.direct_url is None:
            return None
        if type(self.direct_url) is str:
            return self.direct_url
        return json.dumps(self.direct_url)


def _wheel_entries(package: Path) -> list[PurePosixPath]:
    return [
        PurePosixPath("pypto_kernels") / path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
    ]


def test_wheel_record_must_own_the_exact_imported_source_set(tmp_path) -> None:
    install_root = tmp_path / "install"
    package = _copy_real_package(install_root / "pypto_kernels")
    entries = _wheel_entries(package)
    distribution = FakeDistribution(install_root, entries)
    assert (
        _distribution_source_mode(
            distribution,
            package_root=package.resolve(),
            module_path=(package / "__init__.py").resolve(),
        )
        == "wheel"
    )
    incomplete = FakeDistribution(
        install_root,
        entries[1:],
    )
    with pytest.raises(FrameworkCompatibilityError, match="RECORD source set"):
        _distribution_source_mode(
            incomplete,
            package_root=package.resolve(),
            module_path=(package / "__init__.py").resolve(),
        )


@pytest.mark.parametrize("record_error", ["duplicate", "escape", "extra", "native"])
def test_wheel_record_rejects_ambiguous_or_untracked_entries(
    tmp_path,
    record_error,
) -> None:
    install_root = tmp_path / "install"
    package = _copy_real_package(install_root / "pypto_kernels")
    entries = _wheel_entries(package)
    if record_error == "duplicate":
        entries.append(entries[0])
    elif record_error == "escape":
        entries.append(PurePosixPath("pypto_kernels/../escape.py"))
    elif record_error == "extra":
        entries.append(PurePosixPath("pypto_kernels/not-installed.py"))
    else:
        entries.append(PurePosixPath("pypto_kernels/shadow.so"))
    distribution = FakeDistribution(install_root, entries)
    with pytest.raises(FrameworkCompatibilityError):
        _distribution_source_mode(
            distribution,
            package_root=package.resolve(),
            module_path=(package / "__init__.py").resolve(),
        )


@pytest.mark.parametrize("dangling", [False, True])
def test_wheel_record_rejects_symlinked_located_source(tmp_path, dangling) -> None:
    install_root = tmp_path / "install"
    package = _copy_real_package(install_root / "pypto_kernels")
    entries = _wheel_entries(package)
    redirected_entry = entries[0]
    outside = tmp_path / "outside.py"
    if not dangling:
        outside.write_text("OUTSIDE = True\n")
    link = tmp_path / "record-link.py"
    os.symlink(outside, link)

    class RedirectedDistribution(FakeDistribution):
        def locate_file(self, entry):
            if entry == redirected_entry:
                return link
            return super().locate_file(entry)

    distribution = RedirectedDistribution(install_root, entries)
    with pytest.raises(FrameworkCompatibilityError, match="symbolic link"):
        _distribution_source_mode(
            distribution,
            package_root=package.resolve(),
            module_path=(package / "__init__.py").resolve(),
        )


def test_editable_direct_url_binds_exact_src_package(tmp_path) -> None:
    source_root = tmp_path / "source"
    package = _copy_real_package(source_root / "src" / "pypto_kernels")
    distribution = FakeDistribution(
        tmp_path / "metadata",
        [],
        {"url": source_root.as_uri(), "dir_info": {"editable": True}},
    )
    assert (
        _distribution_source_mode(
            distribution,
            package_root=package.resolve(),
            module_path=(package / "__init__.py").resolve(),
        )
        == "editable"
    )
    distribution.direct_url = {
        "url": "https://example.invalid/source",
        "dir_info": {"editable": True},
    }
    with pytest.raises(FrameworkCompatibilityError, match="local file URL"):
        _distribution_source_mode(
            distribution,
            package_root=package.resolve(),
            module_path=(package / "__init__.py").resolve(),
        )


def test_editable_metadata_wins_over_a_full_wheel_like_record(tmp_path) -> None:
    install_root = tmp_path / "install"
    package = _copy_real_package(install_root / "pypto_kernels")
    entries = _wheel_entries(package)
    wrong_source_root = tmp_path / "other-source"
    wrong_source_root.mkdir()
    distribution = FakeDistribution(
        install_root,
        entries,
        {"url": wrong_source_root.as_uri(), "dir_info": {"editable": True}},
    )
    with pytest.raises(FrameworkCompatibilityError, match="editable source root"):
        _distribution_source_mode(
            distribution,
            package_root=package.resolve(),
            module_path=(package / "__init__.py").resolve(),
        )


@pytest.mark.parametrize(
    "direct_url",
    [
        None,
        "{malformed",
        '{"url":"file:///one","url":"file:///two","dir_info":{"editable":true}}',
        {"url": "file:///tmp/source", "dir_info": {"editable": False}},
        {"url": "file:///tmp/source", "dir_info": {"editable": "yes"}},
    ],
)
def test_unowned_or_ambiguous_editable_metadata_fails_closed(
    tmp_path,
    direct_url,
) -> None:
    source_root = tmp_path / "source"
    package = _copy_real_package(source_root / "src" / "pypto_kernels")
    distribution = FakeDistribution(tmp_path / "metadata", [], direct_url)
    with pytest.raises(FrameworkCompatibilityError):
        _distribution_source_mode(
            distribution,
            package_root=package.resolve(),
            module_path=(package / "__init__.py").resolve(),
        )


def test_duplicate_or_missing_distribution_fails_closed(monkeypatch) -> None:
    import pypto_plugins.operator_library as operator_library

    real = importlib.metadata.distribution("pypto-kernels")
    monkeypatch.setattr(
        operator_library.importlib.metadata,
        "distributions",
        lambda: [real, real],
    )
    with pytest.raises(FrameworkCompatibilityError, match="exactly one"):
        assert_operator_library_compatible()
    monkeypatch.setattr(
        operator_library.importlib.metadata,
        "distributions",
        lambda: [],
    )
    with pytest.raises(FrameworkCompatibilityError, match="exactly one"):
        assert_operator_library_compatible()


def test_canonical_equivalent_duplicate_distribution_fails_closed(
    monkeypatch, tmp_path
) -> None:
    import pypto_plugins.operator_library as operator_library

    real = importlib.metadata.distribution("pypto-kernels")
    foreign_root = tmp_path / "not-the-egg-info"
    foreign_root.mkdir()
    equivalent = SimpleNamespace(
        metadata={"Name": "pypto__kernels"},
        read_text=lambda _name: None,
        version=real.version,
        _path=foreign_root,
        locate_file=lambda _name: foreign_root,
    )
    monkeypatch.setattr(
        operator_library.importlib.metadata,
        "distributions",
        lambda: [real, equivalent],
    )
    with pytest.raises(FrameworkCompatibilityError, match="exactly one"):
        assert_operator_library_compatible()


def test_foreign_or_duplicate_distribution_ownership_fails_closed(monkeypatch) -> None:
    import pypto_plugins.operator_library as operator_library

    real = importlib.metadata.distribution("pypto-kernels")
    monkeypatch.setattr(
        operator_library.importlib.metadata,
        "packages_distributions",
        lambda: {"pypto_kernels": ["pypto-kernels", "foreign-provider"]},
    )
    with pytest.raises(FrameworkCompatibilityError, match="exactly one.*owner"):
        assert_operator_library_compatible()
    monkeypatch.setattr(
        operator_library.importlib.metadata,
        "distributions",
        lambda: [real],
    )
    monkeypatch.setattr(
        operator_library.importlib.metadata,
        "packages_distributions",
        lambda: {"pypto_kernels": ["pypto-kernels", "pypto__kernels"]},
    )
    with pytest.raises(FrameworkCompatibilityError, match="exactly one.*owner"):
        assert_operator_library_compatible()


def test_shadow_import_origin_fails_closed(monkeypatch, tmp_path) -> None:
    import pypto_plugins.operator_library as operator_library

    shadow = tmp_path / "pypto_kernels" / "__init__.py"
    shadow.parent.mkdir()
    shadow.write_text("# shadow\n")
    monkeypatch.setattr(
        operator_library.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(shadow)),
    )
    with pytest.raises(FrameworkCompatibilityError, match="import origin"):
        assert_operator_library_compatible()


def test_torch_install_checks_operators_before_any_later_action(monkeypatch) -> None:
    import pypto_plugins.torch_inductor as torch_inductor

    calls: list[str] = []
    monkeypatch.setattr(torch_inductor, "_INSTALLED", False)

    def reject_operator_library():
        calls.append("operators")
        raise FrameworkCompatibilityError("operator mismatch")

    monkeypatch.setattr(
        torch_inductor, "assert_operator_library_compatible", reject_operator_library
    )
    monkeypatch.setattr(
        torch_inductor,
        "assert_backend_executable_ready",
        lambda: calls.append("ready"),
    )
    monkeypatch.setattr(
        torch_inductor, "assert_torch_compatible", lambda: calls.append("torch")
    )
    monkeypatch.setattr(
        torch_inductor, "prepare_process_strict", lambda: calls.append("strict")
    )
    with pytest.raises(FrameworkCompatibilityError, match="operator mismatch"):
        torch_inductor.install()
    assert calls == ["operators"]
    assert torch_inductor._INSTALLED is False


def test_torch_install_readiness_fails_before_framework_actions(monkeypatch) -> None:
    import pypto_plugins.torch_inductor as torch_inductor

    calls: list[str] = []
    monkeypatch.setattr(torch_inductor, "_INSTALLED", False)
    monkeypatch.setattr(
        torch_inductor,
        "assert_operator_library_compatible",
        lambda: calls.append("operators"),
    )

    def reject_readiness():
        calls.append("ready")
        raise RuntimeError("not executable")

    monkeypatch.setattr(
        torch_inductor, "assert_backend_executable_ready", reject_readiness
    )
    monkeypatch.setattr(
        torch_inductor, "assert_torch_compatible", lambda: calls.append("torch")
    )
    monkeypatch.setattr(
        torch_inductor, "prepare_process_strict", lambda: calls.append("strict")
    )
    with pytest.raises(RuntimeError, match="not executable"):
        torch_inductor.install()
    assert calls == ["operators", "ready"]
    assert torch_inductor._INSTALLED is False


def test_torch_install_publishes_only_after_all_preconditions(monkeypatch) -> None:
    import pypto_plugins.torch_inductor as torch_inductor

    calls: list[str] = []
    monkeypatch.setattr(torch_inductor, "_INSTALLED", False)
    monkeypatch.setattr(
        torch_inductor,
        "assert_operator_library_compatible",
        lambda: calls.append("operators"),
    )
    monkeypatch.setattr(
        torch_inductor,
        "assert_backend_executable_ready",
        lambda: calls.append("ready"),
    )
    monkeypatch.setattr(
        torch_inductor, "assert_torch_compatible", lambda: calls.append("torch")
    )
    monkeypatch.setattr(
        torch_inductor, "prepare_process_strict", lambda: calls.append("strict")
    )
    torch_inductor.install()
    assert calls == ["operators", "ready", "torch", "strict"]
    assert torch_inductor._INSTALLED is True


def test_torch_entrypoint_guard_crosses_ordinary_exception_boundary(monkeypatch) -> None:
    import pypto_plugins.torch_backend as torch_backend

    calls: list[str] = []

    def reject_operator_library():
        calls.append("operators")
        raise FrameworkCompatibilityError("operator mismatch")

    monkeypatch.setattr(
        torch_backend, "assert_operator_library_compatible", reject_operator_library
    )
    monkeypatch.setattr(
        torch_backend,
        "_compile_config_patches",
        lambda _options: calls.append("options"),
    )
    with pytest.raises(SystemExit, match="backend preflight failed"):
        torch_backend.compile_backend(None, [], unsupported=True)
    assert calls == ["operators"]


@pytest.mark.parametrize("failure", ["options", "install"])
def test_all_torch_pre_strict_failures_are_non_suppressible(monkeypatch, failure) -> None:
    import pypto_plugins.torch_backend as torch_backend

    calls: list[str] = []
    monkeypatch.setattr(
        torch_backend,
        "assert_operator_library_compatible",
        lambda: calls.append("operators"),
    )
    monkeypatch.setattr(
        torch_backend,
        "mode",
        lambda **_kwargs: calls.append("mode"),
    )
    if failure == "install":
        options = None

        def reject_install():
            calls.append("install")
            raise RuntimeError("readiness or compatibility failure")

        monkeypatch.setattr(torch_backend, "install", reject_install)
    else:
        options = {"cuda_backend": "triton"}
        monkeypatch.setattr(
            torch_backend, "install", lambda: calls.append("install")
        )
    with pytest.raises(SystemExit, match="backend preflight failed") as raised:
        torch_backend.compile_backend(None, [], options=options)
    assert not isinstance(raised.value, Exception)
    if failure == "install":
        assert calls == ["operators", "install"]
    else:
        assert calls == ["operators"]


def test_sglang_registration_checks_operators_before_backend_install(monkeypatch) -> None:
    import pypto_plugins.sglang_plugin as plugin

    calls: list[str] = []
    monkeypatch.setattr(
        plugin, "assert_backend_executable_ready", lambda: calls.append("ready")
    )
    monkeypatch.setattr(plugin, "assert_torch_compatible", lambda: calls.append("torch"))
    monkeypatch.setattr(plugin, "assert_sglang_compatible", lambda: calls.append("sglang"))

    def reject_operator_library():
        calls.append("operators")
        raise FrameworkCompatibilityError("operator mismatch")

    monkeypatch.setattr(plugin, "assert_operator_library_compatible", reject_operator_library)
    monkeypatch.setattr(plugin, "install", lambda: calls.append("install"))
    with pytest.raises(FrameworkCompatibilityError, match="operator mismatch"):
        plugin._register_impl()
    assert calls == ["operators"]


def test_sglang_public_registration_readiness_failure_is_non_suppressible(
    monkeypatch,
) -> None:
    import pypto_plugins.sglang_plugin as plugin

    calls: list[str] = []
    monkeypatch.setattr(
        plugin,
        "assert_operator_library_compatible",
        lambda: calls.append("operators"),
    )

    def reject_readiness():
        calls.append("ready")
        raise RuntimeError("not executable")

    monkeypatch.setattr(plugin, "assert_backend_executable_ready", reject_readiness)
    monkeypatch.setattr(plugin, "assert_torch_compatible", lambda: calls.append("torch"))
    monkeypatch.setattr(plugin, "assert_sglang_compatible", lambda: calls.append("sglang"))
    monkeypatch.setattr(plugin, "install", lambda: calls.append("install"))
    with pytest.raises(SystemExit, match="registration failed") as raised:
        plugin.register()
    assert not isinstance(raised.value, Exception)
    assert calls == ["operators", "ready"]


def test_sglang_registration_orders_preflight_before_registry_mutation(monkeypatch) -> None:
    import pypto_plugins.sglang_plugin as plugin

    calls: list[str] = []
    monkeypatch.setattr(
        plugin,
        "assert_operator_library_compatible",
        lambda: calls.append("operators"),
    )
    monkeypatch.setattr(
        plugin, "assert_backend_executable_ready", lambda: calls.append("ready")
    )
    monkeypatch.setattr(plugin, "assert_torch_compatible", lambda: calls.append("torch"))
    monkeypatch.setattr(plugin, "assert_sglang_compatible", lambda: calls.append("sglang"))
    monkeypatch.setattr(plugin, "install", lambda: calls.append("install"))

    package_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.plugins",
    )
    for name in package_names:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    attention_registry = ModuleType(
        "sglang.srt.layers.attention.attention_registry"
    )

    def register_attention_backend(name):
        calls.append(f"attention-name:{name}")

        def register(_factory):
            calls.append("attention-factory")

        return register

    attention_registry.register_attention_backend = register_attention_backend
    monkeypatch.setitem(sys.modules, attention_registry.__name__, attention_registry)

    server_args = ModuleType("sglang.srt.server_args")
    server_args.add_attention_backend_choices = lambda choices: calls.append(
        f"attention-choices:{choices}"
    )
    server_args.add_linear_attn_kernel_backend_choices = lambda choices: calls.append(
        f"linear-choices:{choices}"
    )
    monkeypatch.setitem(sys.modules, server_args.__name__, server_args)

    hook_registry = ModuleType("sglang.srt.plugins.hook_registry")

    class HookRegistry:
        @staticmethod
        def register(target, _callback, hook_type):
            calls.append(f"hook:{target}:{hook_type}")

    hook_registry.HookRegistry = HookRegistry
    hook_registry.HookType = SimpleNamespace(AROUND="around")
    monkeypatch.setitem(sys.modules, hook_registry.__name__, hook_registry)

    plugin._register_impl()
    assert calls[:5] == ["operators", "ready", "torch", "sglang", "install"]
    assert calls[5:] == [
        "attention-choices:['pypto']",
        "linear-choices:['pypto']",
        "attention-name:pypto",
        "attention-factory",
        f"hook:{plugin.LINEAR_BACKEND_RESOLVER_TARGET}:around",
    ]


def test_plugin_entry_modules_do_not_import_frameworks_at_module_load() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(source_root)!r})
class RejectFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {{'torch', 'sglang'}}:
            raise RuntimeError('framework imported at module load: ' + fullname)
        return None
sys.meta_path.insert(0, RejectFrameworks())
import pypto_plugins
import pypto_plugins.torch_backend
import pypto_plugins.torch_inductor
import pypto_plugins.sglang_plugin
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
