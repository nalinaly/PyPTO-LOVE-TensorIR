"""Plane-major PyPTO convolution state tied to SGLang slot lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from ..errors import BackendNotReadyError


def _debug_event(kind: str, **fields: object) -> None:
    path = os.environ.get("PYPTO_STATE_DEBUG_REPORT")
    if not path:
        return
    payload = {"kind": kind, "pid": os.getpid(), "time": time.monotonic(), **fields}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()


class PyPTOStateBundle:
    """Own PyPTO-only state layouts while mirroring pool clear/copy events."""

    def __init__(self, pool: Any):
        self._pool = pool
        self._lock = threading.RLock()
        self._conv_by_layer: dict[int, Any] = {}
        self._zero_input_by_layer: dict[int, Any] = {}
        self._recurrent_clear_by_layer: dict[int, tuple[Any, ...]] = {}
        self._recurrent_clear_geometry: dict[int, tuple[object, ...]] = {}
        self._geometry_by_layer: dict[int, tuple[int, int, object, object]] = {}
        layer_ids = getattr(pool, "mamba_layer_ids", ())
        if not isinstance(layer_ids, (list, tuple)) or any(
            type(layer_id) is not int or layer_id < 0 for layer_id in layer_ids
        ):
            raise BackendNotReadyError(
                "PyPTO StateBundle requires static non-negative Mamba layer IDs."
            )
        self._expected_layer_ids = frozenset(layer_ids)
        self._pending_clear_layers: set[int] = set()
        self._pending_clear_count = 0

    def conv_for_layer(self, layer_id: int, upstream_conv: Any) -> Any:
        """Return a stable dense ``[slots,3,channels]`` sidecar for one layer."""

        import torch

        if (
            not isinstance(upstream_conv, torch.Tensor)
            or upstream_conv.ndim != 3
            or int(upstream_conv.shape[-1]) != 3
            or upstream_conv.dtype is not torch.bfloat16
        ):
            raise BackendNotReadyError(
                "PyPTO StateBundle requires BF16 SGLang conv state [slots,D,3]."
            )
        slots = int(upstream_conv.shape[0])
        channels = int(upstream_conv.shape[1])
        geometry = (slots, channels, upstream_conv.dtype, upstream_conv.device)
        with self._lock:
            existing_geometry = self._geometry_by_layer.get(layer_id)
            if existing_geometry is not None and existing_geometry != geometry:
                raise BackendNotReadyError(
                    "PyPTO StateBundle layer geometry changed after allocation."
                )
            state = self._conv_by_layer.get(layer_id)
            if state is None:
                if (
                    self._expected_layer_ids
                    and layer_id not in self._expected_layer_ids
                ):
                    raise BackendNotReadyError(
                        "PyPTO StateBundle received an unknown Mamba layer ID."
                    )
                state = torch.zeros(
                    (slots, 3, channels),
                    dtype=upstream_conv.dtype,
                    device=upstream_conv.device,
                )
                self._conv_by_layer[layer_id] = state
                self._zero_input_by_layer[layer_id] = torch.zeros(
                    (slots * 3, channels),
                    dtype=upstream_conv.dtype,
                    device=upstream_conv.device,
                )
                self._geometry_by_layer[layer_id] = geometry
                if self._pending_clear_count and not self._expected_layer_ids:
                    self._pending_clear_layers.add(layer_id)
            _debug_event(
                "conv_for_layer",
                layer_id=int(layer_id),
                pending_layers=sorted(self._pending_clear_layers),
                pending_count=self._pending_clear_count,
            )
            return state

    @staticmethod
    def _indices(indices: Any):
        import torch

        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 1
            or indices.dtype not in (torch.int32, torch.int64)
        ):
            raise BackendNotReadyError(
                "PyPTO StateBundle slot indices must be rank-1 INT32/INT64."
            )
        return indices

    def clear_slots(self, indices: Any) -> None:
        selected = self._indices(indices)
        with self._lock:
            self._pending_clear_layers = set(
                self._expected_layer_ids or self._conv_by_layer
            )
            self._pending_clear_count = int(selected.numel())
            _debug_event(
                "clear_slots",
                indices=selected.detach().cpu().tolist(),
                pending_layers=sorted(self._pending_clear_layers),
                pending_count=self._pending_clear_count,
            )

    def prepare_recurrent_clear(
        self,
        layer_id: int,
        mixed_qkv: Any,
        a: Any,
        b: Any,
        a_log: Any,
        dt_bias: Any,
        batch_size: int,
    ) -> None:
        """Preallocate one exact zero-state GDN input outside clear handling."""

        import torch

        geometry = (
            tuple(mixed_qkv.shape[1:]),
            tuple(a.shape[1:]),
            tuple(b.shape[1:]),
            tuple(a_log.shape),
            tuple(dt_bias.shape),
            mixed_qkv.dtype,
            a.dtype,
            a_log.dtype,
            dt_bias.dtype,
            mixed_qkv.device,
            batch_size,
        )
        with self._lock:
            existing = self._recurrent_clear_geometry.get(layer_id)
            if existing is not None and existing != geometry:
                raise BackendNotReadyError(
                    "PyPTO StateBundle recurrent clear geometry changed."
                )
            if existing is None:
                self._recurrent_clear_by_layer[layer_id] = (
                    torch.zeros(
                        (batch_size, int(mixed_qkv.shape[1])),
                        dtype=mixed_qkv.dtype,
                        device=mixed_qkv.device,
                    ),
                    torch.zeros(
                        (batch_size, int(a.shape[1])),
                        dtype=a.dtype,
                        device=a.device,
                    ),
                    torch.zeros(
                        (batch_size, int(b.shape[1])),
                        dtype=b.dtype,
                        device=b.device,
                    ),
                    torch.full_like(a_log, float("inf")),
                    torch.zeros_like(dt_bias),
                )
                self._recurrent_clear_geometry[layer_id] = geometry

    def take_clear_payload(
        self, layer_id: int, batch_size: int
    ) -> tuple[Any, tuple[Any, ...]] | None:
        """Consume one pending clear as cached conv and GDN zero inputs."""

        with self._lock:
            if layer_id not in self._pending_clear_layers:
                _debug_event(
                    "take_clear_payload_miss",
                    layer_id=int(layer_id),
                    batch_size=int(batch_size),
                    pending_layers=sorted(self._pending_clear_layers),
                    pending_count=self._pending_clear_count,
                )
                return None
            if batch_size != self._pending_clear_count:
                raise BackendNotReadyError(
                    "PyPTO StateBundle clear count differs from the next batch."
                )
            zero_input = self._zero_input_by_layer.get(layer_id)
            recurrent = self._recurrent_clear_by_layer.get(layer_id)
            if zero_input is None or recurrent is None:
                raise BackendNotReadyError(
                    "PyPTO StateBundle has no preallocated clear payload."
                )
            self._pending_clear_layers.remove(layer_id)
            if not self._pending_clear_layers:
                self._pending_clear_count = 0
            _debug_event(
                "take_clear_payload_hit",
                layer_id=int(layer_id),
                batch_size=int(batch_size),
                remaining_layers=sorted(self._pending_clear_layers),
            )
            return zero_input[: batch_size * 3], recurrent

    def copy_from(self, src_indices: Any, dst_indices: Any) -> None:
        source = self._indices(src_indices)
        destination = self._indices(dst_indices)
        if source.shape != destination.shape:
            raise BackendNotReadyError(
                "PyPTO StateBundle copy source/destination counts differ."
            )
        with self._lock:
            for state in self._conv_by_layer.values():
                snapshot = state.index_select(0, source).clone()
                state.index_copy_(0, destination, snapshot)


def attach_state_bundle(pool: Any) -> PyPTOStateBundle:
    """Attach exactly one lifecycle mirror to an SGLang Mamba pool instance."""

    existing = getattr(pool, "_pypto_state_bundle", None)
    if existing is not None:
        if not isinstance(existing, PyPTOStateBundle):
            raise BackendNotReadyError(
                "SGLang Mamba pool already carries an incompatible StateBundle."
            )
        return existing
    bundle = PyPTOStateBundle(pool)
    pool._pypto_state_bundle = bundle

    def install(name: str) -> None:
        owner = next(
            (base for base in type(pool).__mro__ if name in base.__dict__), None
        )
        if owner is None or not callable(owner.__dict__[name]):
            raise BackendNotReadyError(
                "SGLang Mamba pool lacks the pinned clear/copy lifecycle methods."
            )
        marker = f"_pypto_state_bundle_wrapped_{name}"
        if bool(getattr(owner, marker, False)):
            return
        original = owner.__dict__[name]

        if name == "clear_slots":

            def mirrored(self, indices):
                attached = getattr(self, "_pypto_state_bundle", None)
                if attached is not None:
                    attached.clear_slots(indices)
                    return None
                return original(self, indices)

        else:

            def mirrored(self, source, destination):
                result = original(self, source, destination)
                attached = getattr(self, "_pypto_state_bundle", None)
                if attached is not None:
                    attached.copy_from(source, destination)
                return result

        setattr(owner, name, mirrored)
        setattr(owner, marker, True)

    install("clear_slots")
    install("copy_from")
    return bundle
