from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys

import pytest

from pypto_plugins.errors import StrictCoverageError
from pypto_plugins.torch import pointwise_codegen
from pypto_plugins.torch import runtime_bridge
from pypto_plugins.torch.scheduling import REGISTRY


class _Artifact:
    identity_digest = "runtime-identity"
    identities = SimpleNamespace(source_ir_digest="runtime-source")
    producer_identity = SimpleNamespace(
        toolchain_identity=SimpleNamespace(pypto_revision="compiler-revision")
    )
    kernel_abi = SimpleNamespace(
        entry_function_name="pypto_runtime_kernel",
        argument_layout=SimpleNamespace(operand_descriptors=[]),
    )

    def serialize(self) -> bytes:
        return b"runtime-artifact"


def test_pypto_launch_wraps_native_executable_in_artifact_annotation(monkeypatch) -> None:
    name = "pypto_kernel_test"
    artifact = _Artifact()
    request = object()
    REGISTRY.clear()
    REGISTRY.register(name, object())
    pointwise_codegen._RUNTIME_OBJECTS.clear()
    pointwise_codegen._RUNTIME_OBJECTS[name] = (artifact, request)
    pointwise_codegen._RUNTIME_SOURCE_NODES.clear()
    pointwise_codegen._RUNTIME_SOURCE_NODES[name] = "torch-inductor:runtime-test"
    pointwise_codegen._RUNTIME_DEVICE_INDICES.clear()
    pointwise_codegen._RUNTIME_DEVICE_INDICES[name] = 0
    pointwise_codegen._RUNTIME_DSO_SHA256.clear()
    pointwise_codegen._RUNTIME_DSO_SHA256[name] = "d" * 64
    runtime_bridge._executables.clear()
    runtime_bridge._kernel_executable_identities.clear()
    runtime_bridge._execution_streams.clear()
    runtime_bridge._kernel_stream_devices.clear()
    runtime_bridge._artifact_records.clear()
    runtime_bridge._observations.clear()
    monkeypatch.setattr(runtime_bridge, "trace_window_active", lambda: False)
    monkeypatch.setattr(
        runtime_bridge,
        "resolve_live_runtime_expectation",
        lambda: SimpleNamespace(
            driver_label="driver-test",
            cuda_runtime_library_path="/runtime/libcudart.so",
        ),
    )

    launches = []

    class FakeExecutable:
        def __init__(self, observed_artifact, observed_request):
            assert (observed_artifact, observed_request) == (artifact, request)

        def prewarm(self, version):
            assert version == 13030

        def prepare_launch(self, arguments):
            assert arguments == []
            return "packet"

        def launch(self, packet, stream):
            launches.append((packet, stream))

    nvidia = ModuleType("pypto.runtime.nvidia")
    nvidia.NvidiaExecutable = FakeExecutable
    nvidia.NvidiaLaunchArgument = SimpleNamespace(tensor=lambda *_args: None)
    nvidia.observe_current_nvidia_runtime = lambda *_args: SimpleNamespace(
        cuda_runtime_api_version=13030
    )
    runtime = ModuleType("pypto.runtime")
    runtime.nvidia = nvidia
    pypto = ModuleType("pypto")
    pypto.__path__ = []
    pypto.runtime = runtime
    monkeypatch.setitem(sys.modules, "pypto", pypto)
    monkeypatch.setitem(sys.modules, "pypto.runtime", runtime)
    monkeypatch.setitem(sys.modules, "pypto.runtime.nvidia", nvidia)

    class FakeStream:
        device = SimpleNamespace(index=0)

        def __init__(self, pointer):
            self.cuda_stream = pointer
            self.waited = []

        def wait_stream(self, other):
            self.waited.append(other)

    caller = FakeStream(123)
    execution = FakeStream(456)
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        current_stream=lambda: caller,
        current_device=lambda: 0,
        Stream=lambda **_kwargs: execution,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    annotated = []

    @contextmanager
    def annotate(record):
        annotated.append(record)
        yield

    monkeypatch.setattr(runtime_bridge, "annotate_artifact_launch", annotate)
    runtime_bridge.pypto_launch(name, (), 123)

    assert launches == [("packet", 456)]
    assert len(annotated) == 1
    assert annotated[0].artifact_id == "pypto-artifact-v1:runtime-identity"
    assert annotated[0].kernel_name == "pypto_runtime_kernel"
    assert annotated[0].provider == "pypto.generic"
    assert annotated[0].source_node == "torch-inductor:runtime-test"
    assert annotated[0].kernels_revision == "pypto-dso-sha256:" + "d" * 64
    assert caller.waited == [execution]
    assert execution.waited == [caller]


def test_runtime_caches_fail_closed_after_fork(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_bridge,
        "_OWNER_PID",
        runtime_bridge.os.getpid() + 1,
    )
    with pytest.raises(StrictCoverageError, match="inherited across fork"):
        runtime_bridge.kernel_is_prewarmed("kernel")
