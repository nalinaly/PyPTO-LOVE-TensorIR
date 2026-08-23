from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
COMMON_PY = (
    WORKSPACE_ROOT
    / "upstream"
    / "pytorch"
    / "torch"
    / "_inductor"
    / "codegen"
    / "common.py"
)


def _pinned_tree() -> ast.Module:
    if not COMMON_PY.is_file():
        pytest.skip("workspace PyTorch checkout is not present")
    return ast.parse(COMMON_PY.read_text(), filename=str(COMMON_PY))


def test_pinned_device_codegen_fields_are_exact() -> None:
    tree = _pinned_tree()
    device_codegen = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeviceCodegen"
    )
    fields = [
        node.target.id
        for node in device_codegen.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert fields == [
        "scheduling",
        "wrapper_codegen",
        "cpp_wrapper_codegen",
        "fx_wrapper_codegen",
    ]


def test_pinned_registration_signature_is_exact() -> None:
    tree = _pinned_tree()
    register = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "register_backend_for_device"
    )
    arguments = [argument.arg for argument in register.args.args]
    assert arguments == [
        "device",
        "device_scheduling",
        "device_wrapper_codegen",
        "device_cpp_wrapper_codegen",
        "device_fx_wrapper_codegen",
        "device_custom_pass",
        "device_custom_config",
    ]


def test_pinned_python_wrapper_create_contract_is_exact() -> None:
    wrapper_py = COMMON_PY.parent / "wrapper.py"
    tree = ast.parse(wrapper_py.read_text(), filename=str(wrapper_py))
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PythonWrapperCodegen"
    )
    create = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "create"
    )
    assert [argument.arg for argument in create.args.args] == [
        "is_subgraph",
        "subgraph_name",
        "parent_wrapper",
        "partition_signatures",
    ]
