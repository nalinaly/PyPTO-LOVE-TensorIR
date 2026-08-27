from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import pypto_kernels
from pypto_plugins.errors import BackendNotReadyError
from pypto_plugins.sglang.attention_backend import create_attention_backend
from pypto_plugins.sglang_plugin import (
    LINEAR_BACKEND_RESOLVER_TARGET,
    _attention_factory,
)


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
ATTENTION_ADAPTER = (
    PROJECT_ROOT / "src" / "pypto_plugins" / "sglang" / "attention_backend.py"
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


def _install_fake_attention_operator(
    monkeypatch, status: str = "native-tile executable"
) -> ModuleType:
    attention = ModuleType("pypto_kernels.attention")
    attention.PAGED_DECODE_STATUS = status
    monkeypatch.setitem(sys.modules, "pypto_kernels.attention", attention)
    monkeypatch.setattr(pypto_kernels, "attention", attention, raising=False)
    return attention


def test_attention_factory_stays_closed_until_operator_run_pass(monkeypatch) -> None:
    _install_fake_attention_operator(monkeypatch, "native-tile source candidate")
    with pytest.raises(BackendNotReadyError, match="source candidate"):
        _attention_factory(None)


def test_attention_adapter_uses_only_pinned_metadata_and_pypto_operators() -> None:
    text = ATTENTION_ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(ATTENTION_ADAPTER))
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert [node.name for node in classes] == ["PyPTOAttentionBackend"]
    methods = {
        node.name
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "init_cuda_graph_state",
        "forward_decode",
        "forward_extend",
        "support_triton",
    }.issubset(methods)
    for field in (
        "req_to_token_pool.req_to_token",
        "req_pool_indices",
        "seq_lens",
        "seq_lens_cpu",
        "extend_prefix_lens",
        "extend_seq_lens_cpu",
        "out_cache_loc",
    ):
        assert field in text
    for operator in (
        "attention.paged_cache_write(",
        "attention.paged_attention_decode(",
        "attention.paged_attention_prefill(",
    ):
        assert operator in text
    assert ".set_kv_buffer(" not in text
    assert ".to(" not in text


def _install_fake_attention_base(monkeypatch) -> None:
    module_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
    )
    for name in module_names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    base = ModuleType("sglang.srt.layers.attention.base_attn_backend")

    class AttentionBackend:
        pass

    base.AttentionBackend = AttentionBackend
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.attention.base_attn_backend", base
    )


def _fake_runner(allocator) -> SimpleNamespace:
    full_pool = SimpleNamespace(
        kv_cache_layout="nhd",
        is_quantized_kv_cache=False,
        dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        device=torch.device("meta"),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32, device="meta")
        ),
        token_to_kv_pool=SimpleNamespace(full_kv_pool=full_pool),
        token_to_kv_pool_allocator=allocator,
    )


def test_attention_adapter_constructs_only_for_direct_bf16_nhd(monkeypatch) -> None:
    _install_fake_attention_base(monkeypatch)
    _install_fake_attention_operator(monkeypatch)
    backend = create_attention_backend(_fake_runner(SimpleNamespace(page_size=1)))
    assert backend.needs_cpu_seq_lens is True
    assert backend.extend_dummy_seqs_capped_by_req_pool is True
    assert backend.support_triton() is False
    assert backend.get_cuda_graph_seq_len_fill_value() == 1
    with pytest.raises(BackendNotReadyError, match="CUDA-graph metadata"):
        backend.init_cuda_graph_state(4, 4)


def test_attention_adapter_rejects_unified_translation_kernel(monkeypatch) -> None:
    _install_fake_attention_base(monkeypatch)
    _install_fake_attention_operator(monkeypatch)
    allocator = SimpleNamespace(translate_kv_loc_dense=lambda value: value)
    with pytest.raises(BackendNotReadyError, match="unified-memory"):
        create_attention_backend(_fake_runner(allocator))
