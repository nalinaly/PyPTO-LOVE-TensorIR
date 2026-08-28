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
    EMBEDDING_TARGET,
    FLA_GATED_RMSNORM_TARGET,
    FUSED_SIGMOID_MUL_TARGET,
    GDN_PROJECTION_TARGET,
    GEMMA_RMSNORM_TARGET,
    GEMMA_RMSNORM_WEIGHT_LOADER_TARGET,
    LINEAR_BACKEND_RESOLVER_TARGET,
    LM_HEAD_TARGET,
    PRUNED_STATES_TARGET,
    QK_RMSNORM_ROPE_GATE_TARGET,
    SILU_AND_MUL_TARGET,
    TRITON_SUPPORT_TARGETS,
    UNQUANTIZED_LINEAR_TARGET,
    QWEN_LANGUAGE_MODEL_ONLY_ARCHITECTURES,
    _attention_factory,
    _enable_qwen_language_model_only,
    _embedding_around,
    _fla_gated_rmsnorm_around,
    _fused_sigmoid_mul_around,
    _gdn_projection_around,
    _gemma_rmsnorm_around,
    _gemma_rmsnorm_weight_loader_around,
    _qk_rmsnorm_rope_gate_around,
    _lm_head_around,
    _pruned_states_around,
    _silu_and_mul_around,
    _support_triton_around,
    _unquantized_linear_around,
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


def test_qwen35_language_model_only_compatibility_is_bounded_and_idempotent() -> None:
    class FakeServerArgs:
        LANGUAGE_MODEL_ONLY_ARCHITECTURES = (
            "MuseGlimmerForConditionalGeneration",
        )

    expected = (
        "MuseGlimmerForConditionalGeneration",
        *QWEN_LANGUAGE_MODEL_ONLY_ARCHITECTURES,
    )
    assert _enable_qwen_language_model_only(FakeServerArgs) == expected
    assert _enable_qwen_language_model_only(FakeServerArgs) == expected
    assert FakeServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES == expected

    class IncompatibleServerArgs:
        LANGUAGE_MODEL_ONLY_ARCHITECTURES = ["unexpected-list-contract"]

    with pytest.raises(BackendNotReadyError, match="contract changed"):
        _enable_qwen_language_model_only(IncompatibleServerArgs)


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
    assert "_require_callable_hook_target(target)" in plugin_text
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


def test_gemma_rmsnorm_hook_is_preloaded_and_registered() -> None:
    assert GEMMA_RMSNORM_TARGET.endswith("GemmaRMSNorm._forward_impl")
    text = SGLANG_PLUGIN.read_text(encoding="utf-8")
    assert "rmsnorm.rmsnorm(" in text
    assert "fused_add_rmsnorm.fused_add_rmsnorm(" in text
    assert "post_residual_addition" in text
    assert callable(_gemma_rmsnorm_around)


def test_gemma_rmsnorm_offload_colocates_derived_weight() -> None:
    assert GEMMA_RMSNORM_WEIGHT_LOADER_TARGET.endswith(
        "GemmaRMSNorm._weight_loader"
    )

    class FakeTensor:
        def __init__(self, device: str):
            self.device = torch.device(device)

        def to(self, *, device):
            return FakeTensor(str(device))

    layer = SimpleNamespace(gemma_weight=FakeTensor("cuda"))
    param = FakeTensor("cpu")
    loaded = object()
    calls = []

    def original(observed_layer, observed_param, observed_loaded):
        calls.append((observed_layer, observed_param, observed_loaded))
        return "loaded"

    assert (
        _gemma_rmsnorm_weight_loader_around(original, layer, param, loaded)
        == "loaded"
    )
    assert layer.gemma_weight.device.type == "cpu"
    assert calls == [(layer, param, loaded)]


def test_linear_swiglu_and_lm_head_hooks_are_pinned_and_fail_closed(monkeypatch) -> None:
    assert UNQUANTIZED_LINEAR_TARGET.endswith("UnquantizedLinearMethod.apply")
    assert SILU_AND_MUL_TARGET.endswith("activation.silu_and_mul")
    assert LM_HEAD_TARGET.endswith("LogitsProcessor._compute_lm_head")
    assert EMBEDDING_TARGET.endswith("UnquantizedEmbeddingMethod.embedding")
    assert PRUNED_STATES_TARGET.endswith("LogitsProcessor._get_pruned_states")
    text = SGLANG_PLUGIN.read_text(encoding="utf-8")
    assert "linear.linear(" in text
    assert "silu_and_mul.silu_and_mul(" in text

    monkeypatch.setattr(
        "pypto_plugins.sglang_plugin._pypto_compute_selected", lambda: False
    )
    token = object()
    assert _unquantized_linear_around(
        lambda *args: token, object(), object(), object(), None
    ) is token
    assert _silu_and_mul_around(lambda *args: token, object(), None) is token
    assert _lm_head_around(
        lambda *args: token, object(), object(), object(), None
    ) is token
    assert _embedding_around(lambda *args: token, object(), object(), object()) is token

    monkeypatch.setattr(
        "pypto_plugins.sglang_plugin._pypto_compute_selected", lambda: True
    )
    with pytest.raises(BackendNotReadyError, match="unquantized linear"):
        _unquantized_linear_around(
            lambda *args: None,
            object(),
            SimpleNamespace(weight=torch.nn.Parameter(torch.empty(2, 2))),
            torch.empty(1, 2),
            None,
        )
    with pytest.raises(BackendNotReadyError, match="SiLU-and-mul"):
        _silu_and_mul_around(lambda *args: None, torch.empty(1, 4), None)

    mode = SimpleNamespace(is_extend=lambda: True)
    metadata = SimpleNamespace(
        forward_mode=mode,
        extend_return_logprob=False,
        extend_seq_lens_cpu=[3],
    )
    hidden = torch.arange(12).view(3, 4)
    pruned = _pruned_states_around(
        lambda *args: None,
        object(),
        hidden,
        None,
        None,
        metadata,
    )
    assert torch.equal(pruned[0], hidden[-1:])
    assert pruned[0].data_ptr() == hidden[-1:].data_ptr()


def test_fla_gated_rmsnorm_hook_is_preloaded_and_registered() -> None:
    assert FLA_GATED_RMSNORM_TARGET.endswith("layernorm_gated.layernorm_fn")
    text = SGLANG_PLUGIN.read_text(encoding="utf-8")
    assert "gated_rmsnorm.gated_rmsnorm(" in text
    assert "is_rms_norm" in text and "norm_before_gate" in text
    assert callable(_fla_gated_rmsnorm_around)


def test_fla_gated_rmsnorm_routes_only_exact_pypto_contract(monkeypatch) -> None:
    runtime = ModuleType("sglang.srt.runtime_context")
    selection = SimpleNamespace(
        linear_attn_backend="pypto",
        linear_attn_decode_backend="pypto",
        linear_attn_prefill_backend="pypto",
    )
    runtime.get_exec = lambda: SimpleNamespace(mamba=selection)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    module = ModuleType("pypto_kernels.gated_rmsnorm")
    module.STATUS = "native-tile executable"
    calls = []

    def run(x, gate, weight, **kwargs):
        calls.append((x, gate, weight, kwargs))
        return torch.empty_like(x)

    module.gated_rmsnorm = run
    monkeypatch.setitem(sys.modules, "pypto_kernels.gated_rmsnorm", module)
    monkeypatch.setattr(pypto_kernels, "gated_rmsnorm", module, raising=False)

    @contextmanager
    def fake_stream(device):
        assert device.type == "cpu"
        yield "worker-stream"

    monkeypatch.setattr("pypto_plugins.sglang.stream.pypto_stream", fake_stream)
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    gate = torch.empty_like(x)
    weight = torch.empty(128, dtype=torch.bfloat16)
    delegated = []

    def original(*args):
        delegated.append(args)
        return "original-result"

    output = _fla_gated_rmsnorm_around(
        original,
        x,
        weight,
        None,
        gate,
        1.0e-6,
        None,
        True,
        True,
        "swish",
    )
    assert tuple(output.shape) == tuple(x.shape)
    assert not delegated
    assert len(calls) == 1
    assert tuple(calls[0][0].shape) == (6, 128)
    assert tuple(calls[0][1].shape) == (6, 128)
    assert calls[0][2] is weight
    assert calls[0][3] == {"eps": 1.0e-6, "stream": "worker-stream"}

    with pytest.raises(BackendNotReadyError, match="activation='swish'"):
        _fla_gated_rmsnorm_around(
            original, x, weight, None, gate, activation="silu"
        )
    selection.linear_attn_backend = "triton"
    selection.linear_attn_decode_backend = "triton"
    selection.linear_attn_prefill_backend = "triton"
    assert (
        _fla_gated_rmsnorm_around(original, x, weight, None, gate)
        == "original-result"
    )
    assert len(delegated) == 1


def test_qk_rmsnorm_rope_gate_hook_is_preloaded_and_registered() -> None:
    assert QK_RMSNORM_ROPE_GATE_TARGET.endswith(
        "fused_qk_gemma_rmsnorm_rope_gate"
    )
    text = SGLANG_PLUGIN.read_text(encoding="utf-8")
    assert "qk_rmsnorm_rope.qk_rmsnorm_rope_gate(" in text
    assert callable(_qk_rmsnorm_rope_gate_around)


def test_qk_rmsnorm_rope_gate_routes_exact_pypto_contract(monkeypatch) -> None:
    runtime = ModuleType("sglang.srt.runtime_context")
    selection = SimpleNamespace(
        linear_attn_backend="pypto",
        linear_attn_decode_backend="pypto",
        linear_attn_prefill_backend="pypto",
    )
    runtime.get_exec = lambda: SimpleNamespace(mamba=selection)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    module = ModuleType("pypto_kernels.qk_rmsnorm_rope")
    module.STATUS = "native-tile executable"
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return ("q", "k", "gate")

    module.qk_rmsnorm_rope_gate = run
    monkeypatch.setitem(sys.modules, "pypto_kernels.qk_rmsnorm_rope", module)
    monkeypatch.setattr(pypto_kernels, "qk_rmsnorm_rope", module, raising=False)

    @contextmanager
    def fake_stream(device):
        assert device.type == "cpu"
        yield "worker-stream"

    monkeypatch.setattr("pypto_plugins.sglang.stream.pypto_stream", fake_stream)
    q_gate = torch.empty((2, 512), dtype=torch.bfloat16)
    key = torch.empty((2, 128), dtype=torch.bfloat16)
    q_weight = torch.empty(128, dtype=torch.bfloat16)
    k_weight = torch.empty(128, dtype=torch.bfloat16)
    cache = torch.empty((1024, 64), dtype=torch.bfloat16)
    positions = torch.arange(2, dtype=torch.int64).repeat(3, 1)
    delegated = []

    def original(*args, **kwargs):
        delegated.append((args, kwargs))
        return "original-result"

    result = _qk_rmsnorm_rope_gate_around(
        original,
        q_gate,
        key,
        q_weight,
        k_weight,
        cache,
        positions,
        1.0e-6,
        2,
        1,
        128,
        64,
        True,
    )
    assert result == ("q", "k", "gate")
    assert not delegated
    assert len(calls) == 1
    call_args, call_kwargs = calls[0]
    assert call_args[:5] == (q_gate, key, q_weight, k_weight, cache)
    assert torch.equal(call_args[5], positions[0])
    assert call_kwargs == {
        "q_heads": 2,
        "kv_heads": 1,
        "stream": "worker-stream",
    }
    with pytest.raises(BackendNotReadyError, match="has_gate=True"):
        _qk_rmsnorm_rope_gate_around(
            original,
            q_gate,
            key,
            q_weight,
            k_weight,
            cache,
            positions,
            1.0e-6,
            2,
            1,
            128,
            64,
            False,
        )


def test_fused_sigmoid_mul_routes_strided_gate_and_inplace_alias(monkeypatch) -> None:
    assert FUSED_SIGMOID_MUL_TARGET.endswith("elementwise.fused_sigmoid_mul")
    runtime = ModuleType("sglang.srt.runtime_context")
    selection = SimpleNamespace(
        linear_attn_backend="pypto",
        linear_attn_decode_backend="pypto",
        linear_attn_prefill_backend="pypto",
    )
    runtime.get_exec = lambda: SimpleNamespace(mamba=selection)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime)
    module = ModuleType("pypto_kernels.sigmoid_mul")
    module.STATUS = "native-tile executable"
    calls = []

    def run(value, gate, **kwargs):
        calls.append((value, gate, kwargs))
        return value

    module.sigmoid_mul = run
    monkeypatch.setitem(sys.modules, "pypto_kernels.sigmoid_mul", module)
    monkeypatch.setattr(pypto_kernels, "sigmoid_mul", module, raising=False)

    @contextmanager
    def fake_stream(device):
        assert device.type == "cpu"
        yield "worker-stream"

    monkeypatch.setattr("pypto_plugins.sglang.stream.pypto_stream", fake_stream)
    value = torch.empty((2, 256), dtype=torch.bfloat16)
    gate_storage = torch.empty((2, 4, 128), dtype=torch.bfloat16)
    gate = gate_storage[:, :2, :]
    result = _fused_sigmoid_mul_around(
        lambda *_args, **_kwargs: "original", value, gate, inplace=True
    )
    assert result is value
    assert len(calls) == 1
    assert calls[0][0] is value
    assert tuple(calls[0][1].shape) == (2, 256)
    assert tuple(calls[0][1].stride()) == (512, 1)
    assert calls[0][2] == {"stream": "worker-stream", "inplace": True}


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
