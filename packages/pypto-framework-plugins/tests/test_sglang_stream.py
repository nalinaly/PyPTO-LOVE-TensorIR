from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from pypto_plugins.sglang import stream as bridge


class FakeStream:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.waited = []

    def wait_stream(self, other) -> None:
        self.waited.append(other)


def fake_torch(monkeypatch, *, caller_is_default: bool):
    device = SimpleNamespace(type="cuda", index=0)
    default = FakeStream("default", device)
    caller = default if caller_is_default else FakeStream("capture", device)
    created = []
    module = ModuleType("torch")
    module.device = lambda _value: device
    module.cuda = SimpleNamespace(
        current_device=lambda: 0,
        current_stream=lambda _index: caller,
        default_stream=lambda _index: default,
        Stream=lambda device: created.append(FakeStream("worker", device))
        or created[-1],
    )
    monkeypatch.setitem(sys.modules, "torch", module)
    bridge._streams.clear()
    return caller, default, created


def test_default_stream_is_bridged_and_rejoined(monkeypatch) -> None:
    caller, _default, created = fake_torch(monkeypatch, caller_is_default=True)
    with bridge.pypto_stream("cuda:0") as worker:
        assert worker.name == "worker"
        assert worker.waited == [caller]
    assert caller.waited == [worker]
    assert created == [worker]


def test_nondefault_capture_stream_is_used_directly(monkeypatch) -> None:
    caller, _default, created = fake_torch(monkeypatch, caller_is_default=False)
    with bridge.pypto_stream("cuda:0") as observed:
        assert observed is caller
    assert created == []
    assert caller.waited == []
