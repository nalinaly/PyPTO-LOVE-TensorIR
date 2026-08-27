import sys
import time
sys.path.insert(0, "/home/zhaosiying/pypto-love-tensor-ir/projects/pypto-framework-plugins/src")
def log(msg):
    print(f"[{time.monotonic():7.2f}] {msg}", flush=True)

log("probe start")
import torch
log("torch imported")
from pypto_plugins.torch import pointwise_codegen as pc
log("plugin imported")
modules = pc.bootstrap_pypto()
log("pypto bootstrapped")
from pypto.runtime import nvidia as runtime
log("runtime imported")

b = pc.PointwiseProgramBuilder((1024,), "float32")
x = b.add_input("x"); y = b.add_input("y")
b.mark_output(b.emit("tensor.muls", [b.emit("tensor.add", [x, y]), b.scalar(2.0)]))
log("program built")
record = pc.compile_pointwise(b.build(), tile=128, registry_name="probe_kernel_0")
log(f"compiled cubin={record.cubin_sha256[:12]} entry={record.entry_name}")
artifact, request = pc.runtime_objects("probe_kernel_0")
log("runtime objects ok")

lhs = torch.randn(1024, device="cuda")
rhs = torch.randn(1024, device="cuda")
expected = (lhs + rhs) * 2.0
log(f"tensors ready dev={lhs.device}")
out = torch.empty_like(expected)
log("output allocated")

t0 = time.monotonic()
observation = runtime.observe_current_nvidia_runtime(
    "610.74",
    "/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/lib/python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13",
)
log(f"observation done dt={time.monotonic()-t0:.2f} api={observation.cuda_runtime_api_version}")
t0 = time.monotonic()
executable = runtime.NvidiaExecutable(artifact, request)
log(f"executable dt={time.monotonic()-t0:.2f}")
t0 = time.monotonic()
executable.prewarm(observation.cuda_runtime_api_version)
log(f"prewarm dt={time.monotonic()-t0:.2f}")
args = [lhs, rhs, out]
arguments = [runtime.NvidiaLaunchArgument.tensor(int(t.data_ptr()), list(t.shape), list(t.stride())) for t in args]
packet = executable.prepare_launch(arguments)
log(f"packet grid={tuple(packet.grid_dimensions)}")
stream_obj = torch.cuda.Stream()
with torch.cuda.stream(stream_obj):
    stream = int(torch._C._cuda_getCurrentRawStream(0))
    executable.launch(packet, stream)
    log(f"launched stream={stream}")
torch.cuda.synchronize()
log(f"synced correct={torch.equal(out, expected)} maxdiff={(out-expected).abs().max().item():.3e}")
