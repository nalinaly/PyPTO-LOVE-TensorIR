from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools" / "environment_identity.py"
    spec = importlib.util.spec_from_file_location(
        "test_environment_identity_tool", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


environment_identity = load_tool()


def write_distribution(site_packages: pathlib.Path, name: str, version: str) -> None:
    metadata = site_packages / f"{name}-{version}.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def test_distribution_identity_is_scoped_to_formal_prefix(tmp_path) -> None:
    formal = tmp_path / "formal"
    foreign = tmp_path / "foreign"
    write_distribution(formal, "release_package", "1.2.3")
    write_distribution(foreign, "ambient_package", "9.9.9")

    assert environment_identity.collect_distribution_records(formal) == [
        ("release-package", "1.2.3")
    ]
