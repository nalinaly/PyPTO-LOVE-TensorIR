from __future__ import annotations

from pathlib import Path

from pypto_plugins.torch import runtime_identity


def test_runtime_identity_uses_loaded_providers_without_local_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PYPTO_PLUGINS_CUDART", raising=False)
    monkeypatch.delenv("PYPTO_PLUGINS_CUDA_DRIVER_LABEL", raising=False)
    monkeypatch.setattr(
        runtime_identity,
        "_symbol_provider_path",
        lambda symbol: f"/runtime/{symbol}.so",
    )
    monkeypatch.setattr(runtime_identity, "_cuda_driver_api_version", lambda: 13030)
    observed = runtime_identity.resolve_live_runtime_expectation()
    assert observed.driver_label == "cuda-driver-api-13030"
    assert observed.cuda_runtime_library_path == "/runtime/cudaRuntimeGetVersion.so"


def test_runtime_identity_allows_explicit_diagnostic_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cudart = tmp_path / "libcudart.so"
    cudart.touch()
    monkeypatch.setenv("PYPTO_PLUGINS_CUDART", str(cudart))
    monkeypatch.setenv("PYPTO_PLUGINS_CUDA_DRIVER_LABEL", "diagnostic-driver")
    monkeypatch.setattr(
        runtime_identity,
        "_symbol_provider_path",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not inspect")),
    )
    observed = runtime_identity.resolve_live_runtime_expectation()
    assert observed.driver_label == "diagnostic-driver"
    assert observed.cuda_runtime_library_path == str(cudart.resolve())


def test_release_runtime_resolution_contains_no_machine_local_default() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "pypto_plugins" / "torch"
    for name in ("pointwise_codegen.py", "runtime_bridge.py", "runtime_identity.py"):
        text = (package_root / name).read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "610.74" not in text
