from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import pypto_kernels
from pypto_plugins.errors import BackendNotReadyError
from pypto_plugins.sglang.attention_backend import create_attention_backend
from pypto_plugins.sglang_plugin import (
    ATTENTION_WRAPPER_TARGET,
    GDN_PROJECTION_TARGET,
    LINEAR_BACKEND_RESOLVER_TARGET,
    TRITON_SUPPORT_TARGETS,
    _attention_factory,
    _gdn_projection_around,
    _support_triton_around,
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
GDN_ADAPTER = PROJECT_ROOT / "src" / "pypto_plugins" / "sglang" / "gdn_backend.py"
SGLANG_PLUGIN = PROJECT_ROOT / "src" / "pypto_plugins" / "sglang_plugin.py"
ATTENTION_REGISTRY = (
    WORKSPACE_ROOT
    / "upstream"
    / "sglang"
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "attention_registry.py"
)
QWEN35_MODEL = (
    WORKSPACE_ROOT
    / "upstream"
    / "sglang"
    / "python"
    / "sglang"
    / "srt"
    / "models"
    / "qwen3_5.py"
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


def test_pinned_attention_wrapper_hook_symbol_and_signature() -> None:
    if not ATTENTION_REGISTRY.is_file():
        pytest.skip("workspace SGLang checkout is not present")
    tree = ast.parse(
        ATTENTION_REGISTRY.read_text(encoding="utf-8"),
        filename=str(ATTENTION_REGISTRY),
    )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    name = ATTENTION_WRAPPER_TARGET.rsplit(".", 1)[1]
    assert name in functions
    arguments = [argument.arg for argument in functions[name].args.args]
    assert arguments == ["runner", "full_attn_backend"]


def test_pinned_qwen35_projection_hook_symbol_is_imported() -> None:
    if not QWEN35_MODEL.is_file():
        pytest.skip("workspace SGLang checkout is not present")
    tree = ast.parse(
        QWEN35_MODEL.read_text(encoding="utf-8"), filename=str(QWEN35_MODEL)
    )
    name = GDN_PROJECTION_TARGET.rsplit(".", 1)[1]
    imported = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert name in imported
    plugin_text = SGLANG_PLUGIN.read_text(encoding="utf-8")
    assert "module = importlib.import_module(module_name)" in plugin_text
    assert "pinned SGLang hook target is not callable" in plugin_text


def test_pypto_scheduler_metadata_never_selects_triton() -> None:
    delegated = []

    def original(backend):
        delegated.append(backend)
        return True

    assert _support_triton_around(original, "pypto") is False
    assert delegated == []
    assert _support_triton_around(original, "flashinfer") is True
    assert delegated == ["flashinfer"]
    assert len(TRITON_SUPPORT_TARGETS) == 4


def test_gdn_projection_hook_routes_only_explicit_pypto_selection(monkeypatch) -> None:
    runtime = ModuleType("sglang.srt.runtime_context")
    selection = SimpleNamespace(
        linear_attn_backend="pypto",
        linear_attn_decode_backend="pypto",
        linear_attn_prefill_backend="pypto",
    )
    runtime.get_exec = lambda: SimpleNamespace(mamba=selection)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    module = ModuleType("pypto_kernels.gdn_projection")
    module.STATUS = "native-tile packed executable"
    calls = []

    def split_projection(qkvz, ba, **kwargs):
        calls.append((qkvz, ba, kwargs))
        return "pypto-result"

    module.split_projection = split_projection
    monkeypatch.setitem(sys.modules, "pypto_kernels.gdn_projection", module)
    monkeypatch.setattr(pypto_kernels, "gdn_projection", module, raising=False)
    delegated = []
    qkvz = SimpleNamespace(device=torch.device("cuda", 0))

    @contextmanager
    def fake_stream(device):
        assert device == qkvz.device
        yield "worker-stream"

    monkeypatch.setattr("pypto_plugins.sglang.stream.pypto_stream", fake_stream)

    def original(*args):
        delegated.append(args)
        return "original-result"

    result = _gdn_projection_around(original, qkvz, "ba", 8, 16, 128, 128)
    assert result == "pypto-result"
    assert calls == [
        (
            qkvz,
            "ba",
            {
                "q_heads": 8,
                "value_heads": 16,
                "key_dim": 128,
                "value_dim": 128,
                "stream": "worker-stream",
            },
        )
    ]
    assert not delegated
    selection.linear_attn_backend = "triton"
    selection.linear_attn_decode_backend = "triton"
    selection.linear_attn_prefill_backend = "triton"
    assert (
        _gdn_projection_around(original, qkvz, "ba", 8, 16, 128, 128)
        == "original-result"
    )
    assert len(delegated) == 1


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


def test_gdn_adapter_uses_only_stateful_pypto_graphs() -> None:
    text = GDN_ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(GDN_ADAPTER))
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert [node.name for node in classes] == ["PyPTOGDNAttnBackend"]
    for operator in (
        "causal_conv1d.causal_conv1d(",
        "gdn.gdn_recurrent(",
        "attach_state_bundle(",
        ".conv_for_layer(",
    ):
        assert operator in text
    for forbidden in (
        "causal_conv1d_update(",
        "fused_gdn_gating(",
        "kernel_dispatcher",
        ".to(",
    ):
        assert forbidden not in text
    plugin_text = SGLANG_PLUGIN.read_text(encoding="utf-8")
    assert 'causal_conv1d.STATUS != "native-tile stateful executable"' in plugin_text
    assert "check_environments" not in plugin_text


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
        size=15,
        page_size=1,
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


def test_attention_adapter_binds_unified_mapping_without_translation_kernel(
    monkeypatch,
) -> None:
    _install_fake_attention_base(monkeypatch)
    _install_fake_attention_operator(monkeypatch)
    mapping = torch.arange(16, dtype=torch.int64, device="meta")
    allocator = SimpleNamespace(
        translate_kv_loc_dense=lambda value: value,
        page_size=1,
        kernel_page_multiplier=1,
        full_v2p_page_table=mapping,
    )
    backend = create_attention_backend(_fake_runner(allocator))
    assert backend.virtual_to_physical is mapping


def test_attention_adapter_rejects_nonunit_unified_pages(monkeypatch) -> None:
    _install_fake_attention_base(monkeypatch)
    _install_fake_attention_operator(monkeypatch)
    allocator = SimpleNamespace(
        translate_kv_loc_dense=lambda value: value,
        page_size=2,
        kernel_page_multiplier=1,
    )
    with pytest.raises(BackendNotReadyError, match="page_size=1"):
        create_attention_backend(_fake_runner(allocator))
