from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys

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
    REGISTRY._kernels[name] = object()
    pointwise_codegen._RUNTIME_OBJECTS.clear()
    pointwise_codegen._RUNTIME_OBJECTS[name] = (artifact, request)
    runtime_bridge._executables.clear()
    runtime_bridge._execution_streams.clear()
    runtime_bridge._artifact_records.clear()
    monkeypatch.setattr(runtime_bridge, "_observation", None)

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
    assert caller.waited == [execution]
    assert execution.waited == [caller]
