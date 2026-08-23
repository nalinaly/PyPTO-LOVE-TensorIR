from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pypto_plugins.sglang_plugin import LINEAR_BACKEND_RESOLVER_TARGET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
SGLANG_UTILS = (
    WORKSPACE_ROOT
    / "upstream"
    / "sglang"
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "linear"
    / "utils.py"
)


def test_pinned_linear_backend_hook_symbol_and_signature() -> None:
    if not SGLANG_UTILS.is_file():
        pytest.skip("workspace SGLang checkout is not present")
    tree = ast.parse(SGLANG_UTILS.read_text(), filename=str(SGLANG_UTILS))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    name = LINEAR_BACKEND_RESOLVER_TARGET.rsplit(".", 1)[1]
    assert name in functions
    arguments = [argument.arg for argument in functions[name].args.args]
    assert arguments == ["prefill_default"]
