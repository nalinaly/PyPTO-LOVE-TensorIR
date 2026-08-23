from __future__ import annotations

import asyncio
import concurrent.futures

import pytest

from pypto_plugins import torch_inductor
from pypto_plugins.torch.context import (
    activate_mode,
    bind_current_context,
    current_mode,
)


def test_mode_is_nested_and_restored() -> None:
    assert current_mode() is None
    with activate_mode(strict=False) as outer:
        assert current_mode() is outer
        assert outer.strict is False
        with activate_mode(strict=True) as inner:
            assert current_mode() is inner
            assert inner.strict is True
        assert current_mode() is outer
    assert current_mode() is None


def test_nested_mode_cannot_weaken_strict_coverage() -> None:
    with activate_mode(strict=True):
        with pytest.raises(ValueError, match="cannot weaken"):
            with activate_mode(strict=False):
                raise AssertionError("unreachable")


def test_mode_resets_after_an_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with activate_mode(strict=True):
            raise RuntimeError("boom")
    assert current_mode() is None


def test_plain_threads_do_not_inherit_but_bound_callbacks_do() -> None:
    with activate_mode(strict=True):
        bound = bind_current_context(lambda: current_mode())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(current_mode).result() is None
            inherited = pool.submit(bound).result()
    assert inherited is not None
    assert inherited.strict is True


def test_mode_requires_an_exact_boolean() -> None:
    with pytest.raises(TypeError, match="strict"):
        with activate_mode(strict=1):  # type: ignore[arg-type]
            raise AssertionError("unreachable")


def test_async_tasks_keep_independent_mode_values() -> None:
    async def observe(strict: bool) -> bool:
        with activate_mode(strict=strict):
            await asyncio.sleep(0)
            mode = current_mode()
            assert mode is not None
            return mode.strict

    async def gather() -> list[bool]:
        return list(await asyncio.gather(observe(False), observe(True)))

    assert asyncio.run(gather()) == [False, True]
    assert current_mode() is None


def test_public_mode_never_activates_when_install_fails(monkeypatch) -> None:
    def fail_install() -> None:
        raise RuntimeError("not installed")

    monkeypatch.setattr(torch_inductor, "install", fail_install)
    with pytest.raises(RuntimeError, match="not installed"):
        with torch_inductor.mode(strict=True):
            raise AssertionError("unreachable")
    assert current_mode() is None
