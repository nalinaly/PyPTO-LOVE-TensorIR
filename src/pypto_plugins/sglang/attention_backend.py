"""Fail-closed SGLang adapter for the native PyPTO paged-attention graphs."""

from __future__ import annotations

import math
from typing import Any

from ..errors import BackendNotReadyError


def create_attention_backend(model_runner: Any) -> Any:
    """Create the pinned SGLang ``AttentionBackend`` implementation.

    The first executable boundary intentionally accepts only the cache layout
    already represented by the standalone operator ABI: unquantized BF16 NHD
    storage with physical request-table entries. Unsupported modes fail before
    a model-forward launch instead of delegating to another provider.
    """

    import torch
    from pypto_kernels import attention
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from .stream import pypto_stream

    class PyPTOAttentionBackend(AttentionBackend):
        needs_cpu_seq_lens = True
        extend_dummy_seqs_capped_by_req_pool = True

        def __init__(self, runner: Any):
            super().__init__()
            self.device = runner.device
            self.req_to_token_pool = runner.req_to_token_pool
            self.token_to_kv_pool = runner.token_to_kv_pool
            self.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
            self.forward_metadata = None

            allocator = self.token_to_kv_pool_allocator
            translator = getattr(allocator, "translate_kv_loc_dense", None) or getattr(
                allocator, "translate_kv_loc", None
            )
            full_pool = getattr(
                self.token_to_kv_pool, "full_kv_pool", self.token_to_kv_pool
            )
            if getattr(full_pool, "kv_cache_layout", None) != "nhd":
                raise BackendNotReadyError(
                    "PyPTO attention currently requires the pinned NHD KV-cache layout."
                )
            if bool(getattr(full_pool, "is_quantized_kv_cache", False)):
                raise BackendNotReadyError(
                    "PyPTO attention currently requires an unquantized BF16 KV cache."
                )
            if getattr(full_pool, "dtype", None) is not torch.bfloat16:
                raise BackendNotReadyError(
                    "PyPTO attention currently requires BF16 KV-cache compute dtype."
                )
            if translator is None:
                mapping_rows = int(full_pool.size) + int(full_pool.page_size)
                self.virtual_to_physical = torch.arange(
                    mapping_rows, device=self.device, dtype=torch.int64
                )
            else:
                if (
                    int(getattr(allocator, "page_size", 0)) != 1
                    or int(getattr(allocator, "kernel_page_multiplier", 0)) != 1
                ):
                    raise BackendNotReadyError(
                        "PyPTO fused unified-memory translation currently requires "
                        "page_size=1 and kernel_page_multiplier=1."
                    )
                mapping = getattr(allocator, "full_v2p_page_table", None)
                if (
                    mapping is None
                    or mapping.ndim != 1
                    or mapping.dtype is not torch.int64
                    or not mapping.is_contiguous()
                    or mapping.device != self.device
                ):
                    raise BackendNotReadyError(
                        "PyPTO attention requires the allocator's contiguous INT64 "
                        "full-pool virtual-to-physical table."
                    )
                self.virtual_to_physical = mapping

        @staticmethod
        def _reject_extra_kwargs(kwargs: dict[str, Any]) -> None:
            if kwargs:
                raise BackendNotReadyError(
                    "PyPTO attention does not implement optional attention kwargs: "
                    + ", ".join(sorted(kwargs))
                )

        @staticmethod
        def _bucket_tokens(forward_batch: Any, table_width: int) -> int:
            seq_lens_cpu = forward_batch.seq_lens_cpu
            if seq_lens_cpu is None:
                raise BackendNotReadyError(
                    "PyPTO attention needs SGLang's CPU sequence-length mirror; "
                    "GPU-only metadata would require a forbidden host sync."
                )
            if isinstance(seq_lens_cpu, torch.Tensor):
                if seq_lens_cpu.device.type != "cpu":
                    raise BackendNotReadyError(
                        "PyPTO attention sequence-length mirror is not on CPU."
                    )
                max_tokens = int(seq_lens_cpu.max().item())
            else:
                max_tokens = max(int(value) for value in seq_lens_cpu)
            if max_tokens <= 0:
                raise BackendNotReadyError(
                    "PyPTO attention requires at least one valid KV token."
                )
            bucket = ((max_tokens + 15) // 16) * 16
            if bucket > table_width:
                raise BackendNotReadyError(
                    f"PyPTO attention bucket {bucket} exceeds request-table width "
                    f"{table_width}."
                )
            return bucket

        @staticmethod
        def _validate_layer(layer: Any) -> None:
            if bool(layer.is_cross_attention):
                raise BackendNotReadyError(
                    "PyPTO attention does not implement cross attention."
                )
            if layer.sliding_window_size not in (None, -1):
                raise BackendNotReadyError(
                    "PyPTO attention does not implement sliding-window attention."
                )
            if layer.qk_head_dim != layer.v_head_dim:
                raise BackendNotReadyError(
                    "PyPTO attention requires equal QK and value head dimensions."
                )
            if layer.tp_k_head_num != layer.tp_v_head_num:
                raise BackendNotReadyError(
                    "PyPTO attention requires equal key and value head counts."
                )
            expected_scale = 1.0 / math.sqrt(layer.qk_head_dim)
            if not math.isclose(
                float(layer.scaling), expected_scale, rel_tol=0.0, abs_tol=1e-15
            ):
                raise BackendNotReadyError(
                    "PyPTO attention requires scale=1/sqrt(head_dim)."
                )

        @staticmethod
        def _normalize_qkv(q: Any, k: Any, v: Any) -> tuple[Any, Any, Any]:
            if k is None or v is None:
                raise BackendNotReadyError(
                    "PyPTO attention does not implement cross-layer KV sharing."
                )
            q_valid = (
                q.ndim == 2
                and q.dtype is torch.bfloat16
                and q.stride(1) == 1
                and q.stride(0) >= q.shape[1]
            )
            kv_valid = all(
                tensor.ndim == 3
                and tensor.dtype is torch.bfloat16
                and tensor.stride(2) == 1
                and tensor.stride(1) == tensor.shape[2]
                and tensor.stride(0) >= tensor.shape[1] * tensor.shape[2]
                for tensor in (k, v)
            )
            if not q_valid or not kv_valid:
                details = "; ".join(
                    f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                    f"stride={tuple(tensor.stride())}"
                    for name, tensor in (("q", q), ("k", k), ("v", v))
                )
                raise BackendNotReadyError(
                    "PyPTO attention requires row-pitched rank-2 Q and "
                    "head-dense rank-3 K/V BF16 views; "
                    + details
                )
            if q.shape[0] != k.shape[0] or k.shape != v.shape:
                raise BackendNotReadyError(
                    "PyPTO attention received incompatible Q/K/V token rows."
                )
            rows = int(k.shape[0])
            return q, k.view(rows, -1), v.view(rows, -1)

        def _flat_cache(self, layer: Any) -> tuple[Any, Any]:
            key_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            value_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
            if key_cache.ndim == 4 and value_cache.ndim == 4:
                if (
                    key_cache.shape[1] != 1
                    or value_cache.shape[1] != 1
                    or key_cache.shape[2] != layer.tp_k_head_num
                    or key_cache.shape[3] != layer.qk_head_dim
                    or key_cache.stride(3) != 1
                    or key_cache.stride(2) != layer.qk_head_dim
                ):
                    raise BackendNotReadyError(
                        "PyPTO unified cache requires [pages,1,KV heads,head dim] "
                        "page-size-one views."
                    )
                key_cache = key_cache.view(key_cache.shape[0], -1)
                value_cache = value_cache.view(value_cache.shape[0], -1)
            if (
                key_cache.ndim != 3
                and key_cache.ndim != 2
                or value_cache.ndim != key_cache.ndim
                or key_cache.dtype is not torch.bfloat16
                or value_cache.dtype is not torch.bfloat16
                or key_cache.shape != value_cache.shape
                or key_cache.stride() != value_cache.stride()
            ):
                raise BackendNotReadyError(
                    "PyPTO attention requires matching [slots, KV heads, head dim] "
                    "BF16 caches with contiguous head rows."
                )
            if key_cache.ndim == 3:
                if (
                    key_cache.shape[1] != layer.tp_k_head_num
                    or key_cache.shape[2] != layer.qk_head_dim
                    or key_cache.stride(2) != 1
                    or key_cache.stride(1) != layer.qk_head_dim
                ):
                    raise BackendNotReadyError(
                        "PyPTO attention cache head dimensions are incompatible."
                    )
                rows = int(key_cache.shape[0])
                key_flat = key_cache.view(rows, -1)
                value_flat = value_cache.view(rows, -1)
            else:
                key_flat, value_flat = key_cache, value_cache
            if key_flat.stride(0) < key_flat.shape[1]:
                raise BackendNotReadyError(
                    "PyPTO attention cache rows overlap in physical storage."
                )
            return key_flat, value_flat

        def _write_cache(
            self,
            key_cache: Any,
            value_cache: Any,
            k: Any,
            v: Any,
            forward_batch: Any,
            save_kv_cache: bool,
            stream: Any,
        ) -> None:
            if not save_kv_cache:
                raise BackendNotReadyError(
                    "PyPTO attention requires cache writes on its current model path."
                )
            physical_rows = forward_batch.out_cache_loc
            if (
                physical_rows is None
                or physical_rows.ndim != 1
                or physical_rows.dtype is not torch.int64
                or not physical_rows.is_contiguous()
                or physical_rows.numel() != k.shape[0]
            ):
                raise BackendNotReadyError(
                    "PyPTO attention requires one contiguous INT64 cache row per KV row."
                )
            attention.paged_cache_write(
                key_cache,
                value_cache,
                physical_rows,
                self.virtual_to_physical,
                k,
                v,
                stream=stream,
            )

        def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
            del max_bs, max_num_tokens
            raise BackendNotReadyError(
                "PyPTO attention CUDA-graph metadata is not implemented; "
                "start SGLang with CUDA graphs disabled."
            )

        def get_cuda_graph_seq_len_fill_value(self):
            return 1

        def forward_decode(
            self,
            q: Any,
            k: Any,
            v: Any,
            layer: Any,
            forward_batch: Any,
            save_kv_cache: bool = True,
            **kwargs: Any,
        ) -> Any:
            self._reject_extra_kwargs(kwargs)
            if not forward_batch.forward_mode.is_decode():
                raise BackendNotReadyError(
                    "PyPTO decode received a non-decode SGLang forward mode."
                )
            self._validate_layer(layer)
            q, k, v = self._normalize_qkv(q, k, v)
            key_cache, value_cache = self._flat_cache(layer)
            with pypto_stream(q.device) as stream:
                self._write_cache(
                    key_cache,
                    value_cache,
                    k,
                    v,
                    forward_batch,
                    save_kv_cache,
                    stream,
                )
                request_table = self.req_to_token_pool.req_to_token
                bucket = self._bucket_tokens(forward_batch, int(request_table.shape[1]))
                return attention.paged_attention_decode(
                    q,
                    key_cache,
                    value_cache,
                    request_table,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    self.virtual_to_physical,
                    kv_heads=layer.tp_k_head_num,
                    bucket_tokens=bucket,
                    stream=stream,
                )

        def forward_extend(
            self,
            q: Any,
            k: Any,
            v: Any,
            layer: Any,
            forward_batch: Any,
            save_kv_cache: bool = True,
            **kwargs: Any,
        ) -> Any:
            self._reject_extra_kwargs(kwargs)
            mode = forward_batch.forward_mode
            if (
                not mode.is_extend()
                or mode.is_target_verify()
                or mode.is_draft_extend_v2()
                or mode.is_mixed()
                or mode.is_split_prefill()
                or mode.is_dllm_extend()
            ):
                raise BackendNotReadyError(
                    "PyPTO causal prefill currently accepts only plain EXTEND mode."
                )
            self._validate_layer(layer)
            q, k, v = self._normalize_qkv(q, k, v)
            if forward_batch.req_pool_indices.numel() != 1:
                raise BackendNotReadyError(
                    "PyPTO causal prefill currently supports exactly one request."
                )
            extend_rows = forward_batch.extend_seq_lens_cpu
            if extend_rows is None or len(extend_rows) != 1:
                raise BackendNotReadyError(
                    "PyPTO causal prefill needs one CPU extend-length entry."
                )
            if int(extend_rows[0]) != q.shape[0]:
                raise BackendNotReadyError(
                    "PyPTO causal prefill query rows disagree with extend metadata."
                )
            key_cache, value_cache = self._flat_cache(layer)
            with pypto_stream(q.device) as stream:
                self._write_cache(
                    key_cache,
                    value_cache,
                    k,
                    v,
                    forward_batch,
                    save_kv_cache,
                    stream,
                )
                request_table = self.req_to_token_pool.req_to_token
                bucket = self._bucket_tokens(forward_batch, int(request_table.shape[1]))
                return attention.paged_attention_prefill(
                    q,
                    key_cache,
                    value_cache,
                    request_table,
                    forward_batch.req_pool_indices,
                    forward_batch.extend_prefix_lens,
                    self.virtual_to_physical,
                    kv_heads=layer.tp_k_head_num,
                    bucket_tokens=bucket,
                    stream=stream,
                )

        def support_triton(self):
            return False

    return PyPTOAttentionBackend(model_runner)
