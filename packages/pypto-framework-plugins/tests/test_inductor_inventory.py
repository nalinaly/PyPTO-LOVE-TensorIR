from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pypto_plugins.torch.inductor_inventory import (
    INDUCTOR_ADAPTER_SCOPE_V1,
    INDUCTOR_SOURCE_SPECS,
    INDUCTOR_SYMBOL_SPECS,
    _derive_inductor_capabilities,
    _validate_callable,
    audit_inductor_inventory,
    validate_inductor_adapter_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
TORCH_ROOT = WORKSPACE_ROOT / "upstream" / "pytorch"


def test_pinned_inductor_inventory_and_capabilities_are_exact() -> None:
    if not TORCH_ROOT.is_dir():
        pytest.skip("workspace PyTorch checkout is not present")
    audit = audit_inductor_inventory(TORCH_ROOT)
    assert len(audit.sources) == 31
    assert len(audit.symbols) == len(INDUCTOR_SYMBOL_SPECS)
    assert dict(audit.capabilities) == {
        "tilekernel_present": False,
        "cutile_present": False,
        "cutedsl_present": True,
        "cuda_backend_values": ("triton", "halide", "pallas"),
        "cuda_backend_default": "triton",
        "cuda_registry_closed_keys": ("triton", "halide", "pallas"),
        "scheduler_requires_has_triton": True,
        "async_compile_has_pypto": False,
        "get_current_backend_occurrences": 12,
        "get_current_backend_call_sites": (
            ("common", 730),
            ("common", 739),
            ("common", 779),
            ("common", 2111),
            ("common", 2675),
            ("mm_common", 155),
            ("scheduler", 200),
            ("utils", 4133),
        ),
        "config_cuda_backend_occurrences": 2,
        "codecache_backend_registry_sequence": (
            "init_backend_registration",
            "custom_backend_passes",
            "custom_backend_codegen_configs",
        ),
        "cseproxy_dtype_shape_backends": ("triton", "cpp", "mps"),
        "scheduler_extern_bypasses_backend": True,
        "scheduler_foreach_backend_types": (
            "SIMDScheduling",
            "CUDACombinedScheduling",
            "XPUCombinedScheduling",
        ),
        "scheduler_multi_template_extern_fallback": True,
        "explicit_extern_choice_counts": (
            ("mm", 8),
            ("bmm", 3),
            ("mm_plus_mm", 1),
        ),
        "autotune_subprocess_present": True,
        "inductor_python_manifest_count": 346,
        "inductor_python_manifest_sha256": (
            "3bef71727acfceb4c1dbc1f433ac26e21c65ff715435feefc80cc32d4cb88cd6"
        ),
    }
    assert audit.scope_digest == (
        "9fa6b8bd2eb912d1fc4510fc032ec09bb64a5bc574544e099f8f2eae8e025856"
    )


def test_inductor_adapter_scope_is_immutable_and_unready() -> None:
    assert INDUCTOR_ADAPTER_SCOPE_V1["registration_ready"] is False
    assert INDUCTOR_ADAPTER_SCOPE_V1["usable_triton_install_required"] is True
    assert INDUCTOR_ADAPTER_SCOPE_V1["exact_triton_source_audit_complete"] is False
    assert INDUCTOR_ADAPTER_SCOPE_V1["triton_compute_allowed"] is False
    assert INDUCTOR_ADAPTER_SCOPE_V1["python_wrapper_in_scope"] is True
    assert INDUCTOR_ADAPTER_SCOPE_V1["python_wrapper_implemented"] is False
    assert INDUCTOR_ADAPTER_SCOPE_V1["python_subgraph_wrapper_implemented"] is False
    assert INDUCTOR_ADAPTER_SCOPE_V1["atomic_registry_install_implemented"] is False
    assert (
        INDUCTOR_ADAPTER_SCOPE_V1["cseproxy_pypto_dtype_shape_implemented"]
        is False
    )
    assert (
        INDUCTOR_ADAPTER_SCOPE_V1["strict_lowering_choice_filter_implemented"]
        is False
    )
    assert INDUCTOR_ADAPTER_SCOPE_V1["cpp_wrapper_supported"] is False
    assert INDUCTOR_ADAPTER_SCOPE_V1["fx_wrapper_supported"] is False
    assert validate_inductor_adapter_scope(INDUCTOR_ADAPTER_SCOPE_V1) == (
        "9fa6b8bd2eb912d1fc4510fc032ec09bb64a5bc574544e099f8f2eae8e025856"
    )
    with pytest.raises(TypeError):
        INDUCTOR_ADAPTER_SCOPE_V1["registration_ready"] = True
    weakened = dict(INDUCTOR_ADAPTER_SCOPE_V1)
    weakened["extern_compute_supported"] = True
    with pytest.raises(RuntimeError, match="scope mismatch"):
        validate_inductor_adapter_scope(weakened)


def test_inductor_source_mismatch_fails_closed(tmp_path) -> None:
    if not TORCH_ROOT.is_dir():
        pytest.skip("workspace PyTorch checkout is not present")
    first = INDUCTOR_SOURCE_SPECS[0]
    target = tmp_path / first.relative_path
    target.parent.mkdir(parents=True)
    target.write_text("changed")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        audit_inductor_inventory(tmp_path)


def test_inventory_imports_no_torch_or_compiler_package() -> None:
    path = PROJECT_ROOT / "src" / "pypto_plugins" / "torch" / "inductor_inventory.py"
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    forbidden_roots = {"pypto", "torch", "triton"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not {alias.name.split(".", 1)[0]
                        for alias in node.names} & forbidden_roots
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".", 1)[0] not in forbidden_roots
        if isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Name)
                        and node.func.id == "__import__")
            assert not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            )

    environment = os.environ.copy()
    source_root = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH", ""))
        if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import pypto_plugins.torch.inductor_inventory; "
                "print('\\n'.join(sorted(sys.modules)))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=PROJECT_ROOT,
    )
    modules = set(result.stdout.splitlines())
    for forbidden in forbidden_roots:
        assert forbidden not in modules
        assert not any(name.startswith(f"{forbidden}.") for name in modules)


def test_capabilities_are_derived_from_synthetic_sources(tmp_path) -> None:
    source_texts = {
        "config": (
            'cuda_backend: Literal["triton", "halide", "pallas"] = "triton"'
        ),
        "common": (
            'cuda_backends = {"triton": 1, "halide": 2, "pallas": 3}\n'
            "class CSEProxy:\n"
            "    def _default(self, name, args, kwargs):\n"
            "        if backend in ('triton', 'cpp', 'mps'):\n"
            "            return None\n"
        ),
        "scheduler": (
            "class Scheduler:\n"
            "    def create_backend(self, device):\n"
            "        return has_triton()\n"
            "    def _codegen(self, nodes):\n"
            "        if node.is_extern():\n"
            "            self.codegen_extern_call(node)\n"
            "        if isinstance(backend_, (SIMDScheduling, "
            "CUDACombinedScheduling, XPUCombinedScheduling)):\n"
            "            pass\n"
            "    def codegen(self):\n"
            "        pass\n"
            "    def finalize_multi_template_buffers(self):\n"
            "        return thing.ExternKernelCaller\n"
        ),
        "async_compile": "class AsyncCompile:\n    pass\n",
        "codecache": (
            "class FxGraphHashDetails:\n"
            "    def __init__(self, gm, example_inputs, fx_kwargs, inputs_to_check):\n"
            "        init_backend_registration()\n"
            "        self.custom_backend_passes = custom_backend_passes\n"
            "        self.custom_backend_codegen_configs = "
            "custom_backend_codegen_configs\n"
        ),
        "mm": "ExternKernelChoice()\n",
        "bmm": "ExternKernelChoice()\n",
        "mm_plus_mm": "ExternKernelChoice()\n",
        "autotune_process": (
            "def run_autotune_in_subprocess(benchmark_request):\n"
            "    pass\n"
        ),
    }
    trees = {name: ast.parse(text) for name, text in source_texts.items()}
    inductor_root = tmp_path / "torch" / "_inductor"
    inductor_root.mkdir(parents=True)
    (inductor_root / "probe.py").write_text("# initially empty\n")

    capabilities = dict(
        _derive_inductor_capabilities(tmp_path, trees, source_texts)
    )
    assert capabilities["scheduler_requires_has_triton"] is True
    assert capabilities["async_compile_has_pypto"] is False
    assert capabilities["tilekernel_present"] is False
    assert capabilities["cutile_present"] is False

    (inductor_root / "probe.py").write_text(
        "class TileKernelScheduling:\n    pass\n# cu_tile\n"
    )
    mutated_texts = dict(source_texts)
    mutated_texts["async_compile"] = "def pypto():\n    pass\n"
    mutated_trees = dict(trees)
    mutated_trees["async_compile"] = ast.parse(mutated_texts["async_compile"])
    mutated_trees["scheduler"] = ast.parse(
        "class Scheduler:\n"
        "    def create_backend(self, device):\n"
        "        return None\n"
        "    def _codegen(self, nodes):\n"
        "        if node.is_extern():\n"
        "            self.codegen_extern_call(node)\n"
        "        if isinstance(backend_, (SIMDScheduling, "
        "CUDACombinedScheduling, XPUCombinedScheduling)):\n"
        "            pass\n"
        "    def codegen(self):\n"
        "        pass\n"
        "    def finalize_multi_template_buffers(self):\n"
        "        return thing.ExternKernelCaller\n"
    )
    capabilities = dict(
        _derive_inductor_capabilities(tmp_path, mutated_trees, mutated_texts)
    )
    assert capabilities["scheduler_requires_has_triton"] is False
    assert capabilities["async_compile_has_pypto"] is True
    assert capabilities["tilekernel_present"] is True
    assert capabilities["cutile_present"] is True

    reordered_trees = dict(trees)
    reordered_trees["codecache"] = ast.parse(
        "class FxGraphHashDetails:\n"
        "    def __init__(self, gm, example_inputs, fx_kwargs, inputs_to_check):\n"
        "        self.custom_backend_passes = custom_backend_passes\n"
        "        init_backend_registration()\n"
        "        self.custom_backend_codegen_configs = "
        "custom_backend_codegen_configs\n"
    )
    with pytest.raises(RuntimeError, match="registry sequence mismatch"):
        _derive_inductor_capabilities(tmp_path, reordered_trees, source_texts)

    dynamic_trees = dict(trees)
    dynamic_trees["common"] = ast.parse(
        "cuda_backends = {**other}\n"
        "class CSEProxy:\n"
        "    def _default(self, name, args, kwargs):\n"
        "        if backend in ('triton', 'cpp', 'mps'):\n"
        "            return None\n"
    )
    with pytest.raises(RuntimeError, match="dynamic or unpacked key"):
        _derive_inductor_capabilities(tmp_path, dynamic_trees, source_texts)


def test_full_callable_signature_contract_detects_category_and_default_drift() -> None:
    function = ast.parse(
        "@staticmethod\n"
        "def sample(value: int, *, flag=True, **kwargs: object):\n"
        "    pass\n"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    _validate_callable(
        function,
        owner="sample",
        arguments=("value", "flag"),
        signature="value: int, *, flag=True, **kwargs: object",
        decorators=("staticmethod",),
        is_async=False,
    )
    with pytest.raises(RuntimeError, match="signature mismatch"):
        _validate_callable(
            function,
            owner="sample",
            arguments=("value", "flag"),
            signature="value: int, flag=True, **kwargs: object",
            decorators=("staticmethod",),
            is_async=False,
        )
    with pytest.raises(RuntimeError, match="decorators mismatch"):
        _validate_callable(
            function,
            owner="sample",
            arguments=("value", "flag"),
            signature="value: int, *, flag=True, **kwargs: object",
            decorators=(),
            is_async=False,
        )
