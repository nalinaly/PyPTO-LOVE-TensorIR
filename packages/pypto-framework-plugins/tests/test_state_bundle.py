from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pypto_plugins.errors import BackendNotReadyError
from pypto_plugins.sglang.state_bundle import attach_state_bundle


class FakePool:
    def __init__(self, mamba_layer_ids=()):
        self.cleared = []
        self.copied = []
        self.mamba_layer_ids = list(mamba_layer_ids)

    def clear_slots(self, indices):
        self.cleared.append(indices.clone())

    def copy_from(self, source, destination):
        self.copied.append((source.clone(), destination.clone()))


def test_state_bundle_is_stable_plane_major_and_mirrors_lifecycle() -> None:
    pool = FakePool((7,))
    bundle = attach_state_bundle(pool)
    upstream = torch.zeros((6, 8, 3), dtype=torch.bfloat16)
    state = bundle.conv_for_layer(7, upstream)
    assert state.shape == (6, 3, 8)
    assert state.stride() == (24, 8, 1)
    assert bundle.conv_for_layer(7, upstream) is state
    assert attach_state_bundle(pool) is bundle

    state[1] = torch.arange(24, dtype=torch.bfloat16).view(3, 8)
    pool.copy_from(torch.tensor([1], dtype=torch.int32), torch.tensor([4]))
    assert torch.equal(state[4], state[1])
    assert len(pool.copied) == 1
    FakePool.copy_from(pool, torch.tensor([1]), torch.tensor([5]))
    assert torch.equal(state[5], state[1])
    assert len(pool.copied) == 2
    pool.clear_slots(torch.tensor([1, 4], dtype=torch.int64))
    bundle.prepare_recurrent_clear(
        7,
        torch.empty((2, 48), dtype=torch.bfloat16),
        torch.empty((2, 8), dtype=torch.bfloat16),
        torch.empty((2, 8), dtype=torch.bfloat16),
        torch.empty(8, dtype=torch.float32),
        torch.empty(8, dtype=torch.bfloat16),
        2,
    )
    clear_input, recurrent = bundle.take_clear_payload(7, 2)
    assert clear_input.shape == (6, 8)
    assert torch.count_nonzero(clear_input) == 0
    assert [tuple(tensor.shape) for tensor in recurrent] == [
        (2, 48),
        (2, 8),
        (2, 8),
        (8,),
        (8,),
    ]
    assert torch.isinf(recurrent[3]).all()
    assert bundle.take_clear_payload(7, 2) is None
    assert pool.cleared == []


def test_clear_before_first_layer_registration_is_consumed_once_per_layer() -> None:
    pool = FakePool((0, 1))
    bundle = attach_state_bundle(pool)
    bundle.clear_slots(torch.tensor([1], dtype=torch.int64))
    assert bundle._pending_clear_count == 1
    assert bundle._pending_clear_layers == {0, 1}

    for index, layer_id in enumerate((0, 1)):
        upstream = torch.zeros((4, 8, 3), dtype=torch.bfloat16)
        bundle.conv_for_layer(layer_id, upstream)
        bundle.prepare_recurrent_clear(
            layer_id,
            torch.empty((1, 48), dtype=torch.bfloat16),
            torch.empty((1, 8), dtype=torch.bfloat16),
            torch.empty((1, 8), dtype=torch.bfloat16),
            torch.empty(8, dtype=torch.float32),
            torch.empty(8, dtype=torch.bfloat16),
            1,
        )
        assert layer_id in bundle._pending_clear_layers
        assert bundle.take_clear_payload(layer_id, 1) is not None
        assert bundle.take_clear_payload(layer_id, 1) is None
        assert bundle._pending_clear_count == (1 if index == 0 else 0)

    assert bundle._pending_clear_layers == set()
    assert bundle._pending_clear_count == 0


def test_later_clear_marks_every_registered_layer_and_resets_after_consumption() -> None:
    bundle = attach_state_bundle(FakePool((3, 7)))
    for layer_id in (3, 7):
        bundle.conv_for_layer(
            layer_id, torch.zeros((4, 8, 3), dtype=torch.bfloat16)
        )
        bundle.prepare_recurrent_clear(
            layer_id,
            torch.empty((1, 48), dtype=torch.bfloat16),
            torch.empty((1, 8), dtype=torch.bfloat16),
            torch.empty((1, 8), dtype=torch.bfloat16),
            torch.empty(8, dtype=torch.float32),
            torch.empty(8, dtype=torch.bfloat16),
            1,
        )
    bundle.clear_slots(torch.tensor([2], dtype=torch.int32))
    assert bundle._pending_clear_layers == {3, 7}
    assert bundle.take_clear_payload(3, 1) is not None
    assert bundle._pending_clear_count == 1
    assert bundle.take_clear_payload(7, 1) is not None
    assert bundle._pending_clear_count == 0


def test_state_bundle_rejects_unknown_static_layer_id() -> None:
    bundle = attach_state_bundle(FakePool((1, 2)))
    with pytest.raises(BackendNotReadyError, match="unknown Mamba layer ID"):
        bundle.conv_for_layer(
            3, torch.zeros((4, 8, 3), dtype=torch.bfloat16)
        )


def test_state_bundle_fails_closed_on_shape_or_geometry_drift() -> None:
    bundle = attach_state_bundle(FakePool())
    with pytest.raises(BackendNotReadyError, match=r"\[slots,D,3\]"):
        bundle.conv_for_layer(1, torch.zeros((4, 3, 8), dtype=torch.bfloat16))
    bundle.conv_for_layer(1, torch.zeros((4, 8, 3), dtype=torch.bfloat16))
    with pytest.raises(BackendNotReadyError, match="geometry changed"):
        bundle.conv_for_layer(1, torch.zeros((5, 8, 3), dtype=torch.bfloat16))


def test_state_bundle_rejects_bad_indices_and_incompatible_pool() -> None:
    bundle = attach_state_bundle(FakePool())
    bundle.conv_for_layer(0, torch.zeros((4, 8, 3), dtype=torch.bfloat16))
    with pytest.raises(BackendNotReadyError, match="rank-1 INT32/INT64"):
        bundle.clear_slots(torch.tensor([[1]], dtype=torch.int64))
    with pytest.raises(BackendNotReadyError, match="counts differ"):
        bundle.copy_from(torch.tensor([0, 1]), torch.tensor([2]))
    with pytest.raises(BackendNotReadyError, match="lacks the pinned"):
        attach_state_bundle(SimpleNamespace())
