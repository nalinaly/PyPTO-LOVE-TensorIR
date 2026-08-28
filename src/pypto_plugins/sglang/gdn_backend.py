"""Fail-closed SGLang Gated DeltaNet adapter for native PyPTO graphs."""

from __future__ import annotations

from typing import Any

from ..errors import BackendNotReadyError


def create_gdn_backend(model_runner: Any) -> Any:
    """Create the non-speculative, no-radix PyPTO GDN backend."""

    import torch
    from pypto_kernels import causal_conv1d, gdn
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )
    from .state_bundle import attach_state_bundle
    from .stream import pypto_stream

    class PyPTOGDNAttnBackend(MambaAttnBackendBase):
        needs_cpu_seq_lens = True

        def __init__(self, runner: Any):
            super().__init__(runner)
            args = runner.server_args
            if not bool(args.disable_cuda_graph):
                raise BackendNotReadyError(
                    "PyPTO GDN CUDA-graph metadata is not implemented; disable CUDA graphs."
                )
            if not bool(args.disable_radix_cache):
                raise BackendNotReadyError(
                    "PyPTO GDN StateBundle copy/checkpoint is not implemented; "
                    "disable the radix cache."
                )
            if args.speculative_algorithm is not None:
                raise BackendNotReadyError(
                    "PyPTO GDN speculative verify/rollback is not implemented."
                )
            pool = self.req_to_token_pool.mamba_pool
            self._pypto_state_bundle = attach_state_bundle(pool)
            self._pypto_query_start_loc_cache = {}
            if bool(pool.enable_linear_replayssm) or bool(
                pool.enable_linear_replayssm_spec
            ):
                raise BackendNotReadyError(
                    "PyPTO GDN ReplaySSM rings are not implemented."
                )

        def _forward_metadata(self, forward_batch: Any):
            mode = forward_batch.forward_mode
            if (
                mode.is_extend()
                and not mode.is_target_verify()
                and not mode.is_draft_extend_v2()
                and not mode.is_mixed()
                and forward_batch.batch_size == 1
                and forward_batch.mamba_track_indices is None
                and forward_batch.mamba_track_mask is None
            ):
                lengths = forward_batch.extend_seq_lens_cpu
                if lengths is None or len(lengths) != 1:
                    raise BackendNotReadyError(
                        "PyPTO GDN metadata requires one CPU extend length."
                    )
                cache_key = (1, int(lengths[0]))
                query_start_loc = self._pypto_query_start_loc_cache.get(cache_key)
                if query_start_loc is None:
                    from ..activity_trace import trace_window_active

                    if trace_window_active():
                        raise BackendNotReadyError(
                            "PyPTO GDN query metadata was not prepared before "
                            "the model-forward trace window."
                        )
                    query_start_loc = torch.tensor(
                        [0, cache_key[1]],
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self._pypto_query_start_loc_cache[cache_key] = query_start_loc
                mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
                    forward_batch.req_pool_indices
                )
                mamba_cache_indices = self._translate_mamba_indices(
                    mamba_cache_indices
                )
                from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                    ForwardMetadata,
                )

                return ForwardMetadata(
                    query_start_loc=query_start_loc,
                    mamba_cache_indices=mamba_cache_indices,
                    mamba_track_indices=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_parent_token=None,
                    track_conv_indices=None,
                    track_ssm_h_src=None,
                    track_ssm_h_dst=None,
                    track_ssm_final_src=None,
                    track_ssm_final_dst=None,
                    has_mamba_track_mask=False,
                    replayssm_write_pos=None,
                    replayssm_force_flush=None,
                )
            return super()._forward_metadata(forward_batch)

        @staticmethod
        def _validate_wrapper_kwargs(layer: Any, kwargs: dict[str, Any]) -> None:
            optional_tensors = {
                name: kwargs.pop(name, None) for name in ("q", "k", "v")
            }
            if any(value is not None for value in optional_tensors.values()):
                raise BackendNotReadyError(
                    "PyPTO GDN does not accept full-attention Q/K/V inputs."
                )
            save_kv_cache = kwargs.pop("save_kv_cache", True)
            if save_kv_cache is not True:
                raise BackendNotReadyError(
                    "PyPTO GDN requires the stateful cache update path."
                )
            layer_id = kwargs.pop("layer_id", None)
            if layer_id is not None and int(layer_id) != int(layer.layer_id):
                raise BackendNotReadyError(
                    "PyPTO GDN wrapper layer_id disagrees with the layer object."
                )
            if kwargs:
                raise BackendNotReadyError(
                    "PyPTO GDN does not implement optional backend kwargs: "
                    + ", ".join(sorted(kwargs))
                )

        @staticmethod
        def _validate_layer(layer: Any) -> None:
            if layer.bias is not None:
                raise BackendNotReadyError(
                    "PyPTO GDN currently requires bias-free conv1d."
                )
            if layer.activation not in ("silu", "swish"):
                raise BackendNotReadyError(
                    "PyPTO GDN currently requires SiLU/swish convolution activation."
                )
            if layer.head_q_dim != layer.head_k_dim:
                raise BackendNotReadyError(
                    "PyPTO GDN requires equal query and key head dimensions."
                )

        def _run(
            self,
            layer: Any,
            mixed_qkv: Any,
            a: Any,
            b: Any,
            *,
            batch_size: int,
            tokens_per_request: int,
        ) -> Any:
            self._validate_layer(layer)
            if not isinstance(mixed_qkv, torch.Tensor):
                raise BackendNotReadyError(
                    "PyPTO GDN requires packed tensor QKV input."
                )
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
            conv_state = self._pypto_state_bundle.conv_for_layer(
                layer.layer_id, layer_cache.conv[0]
            )
            recurrent_state = layer_cache.temporal
            state_indices = self.forward_metadata.mamba_cache_indices
            self._pypto_state_bundle.prepare_recurrent_clear(
                layer.layer_id,
                mixed_qkv,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                batch_size,
            )
            with pypto_stream(mixed_qkv.device) as stream:
                clear_payload = self._pypto_state_bundle.take_clear_payload(
                    layer.layer_id, batch_size
                )
                if clear_payload is not None:
                    clear_input, recurrent_clear = clear_payload
                    causal_conv1d.causal_conv1d(
                        clear_input,
                        layer.conv_weights,
                        conv_state,
                        state_indices,
                        batch_size=batch_size,
                        tokens_per_request=3,
                        stream=stream,
                    )
                    gdn.gdn_recurrent(
                        *recurrent_clear,
                        recurrent_state,
                        state_indices,
                        batch_size=batch_size,
                        tokens_per_request=1,
                        stream=stream,
                    )
                convolved = causal_conv1d.causal_conv1d(
                    mixed_qkv,
                    layer.conv_weights,
                    conv_state,
                    state_indices,
                    batch_size=batch_size,
                    tokens_per_request=tokens_per_request,
                    stream=stream,
                )
                recurrent = gdn.gdn_recurrent(
                    convolved,
                    a,
                    b,
                    layer.A_log,
                    layer.dt_bias,
                    recurrent_state,
                    state_indices,
                    batch_size=batch_size,
                    tokens_per_request=tokens_per_request,
                    stream=stream,
                )
            return recurrent

        def forward_decode(
            self,
            layer: Any,
            forward_batch: Any,
            mixed_qkv: Any,
            a: Any,
            b: Any,
            **kwargs: Any,
        ) -> Any:
            self._validate_wrapper_kwargs(layer, kwargs)
            if not forward_batch.forward_mode.is_decode():
                raise BackendNotReadyError(
                    "PyPTO GDN decode received a non-decode forward mode."
                )
            output = self._run(
                layer,
                mixed_qkv,
                a,
                b,
                batch_size=forward_batch.batch_size,
                tokens_per_request=1,
            )
            return output.transpose(0, 1)

        def forward_extend(
            self,
            layer: Any,
            forward_batch: Any,
            mixed_qkv: Any,
            a: Any,
            b: Any,
            **kwargs: Any,
        ) -> Any:
            self._validate_wrapper_kwargs(layer, kwargs)
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
                    "PyPTO GDN currently accepts only plain EXTEND mode."
                )
            if forward_batch.batch_size != 1:
                raise BackendNotReadyError(
                    "PyPTO GDN prefill currently supports exactly one request."
                )
            lengths = forward_batch.extend_seq_lens_cpu
            if lengths is None or len(lengths) != 1:
                raise BackendNotReadyError(
                    "PyPTO GDN prefill needs one CPU extend-length entry."
                )
            tokens = int(lengths[0])
            if mixed_qkv.shape[0] != tokens:
                raise BackendNotReadyError(
                    "PyPTO GDN packed rows disagree with extend metadata."
                )
            return self._run(
                layer,
                mixed_qkv,
                a,
                b,
                batch_size=1,
                tokens_per_request=tokens,
            )

        def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
            del max_bs, max_num_tokens
            raise BackendNotReadyError("PyPTO GDN CUDA graphs are disabled.")

        def support_triton(self):
            return False

    return PyPTOGDNAttnBackend(model_runner)
