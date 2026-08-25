#!/usr/bin/env python3
"""Pure constants for the exact PyPTO NvidiaExecutable SM120 smoke."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SMOKE_SCHEMA_VERSION = 1
SMOKE_NAME = "pypto-nvidia-executable-sm120"
GPU_SMOKE_POLICY_VERSION = 1
GPU_SMOKE_AUTHORIZATION = (
    "user-authorized-protected-cpu-lane-with-zero-nvidia-runtime-or-compute"
)
GPU_SMOKE_TIMEOUT_SECONDS = 1_800
GPU_SMOKE_MINIMUM_FREE_DISK_GIB = 64

PYPTO_HEAD = "206447cf8c68b9cff1b86e01f0b40bfd689cd7a7"
PYPTO_TREE = "e0357daaefa74dbf676550015e60701996c400fb"
TENSOR_IR_HEAD = "1dcb38c20e53d07c97d3781cae538e33901bae30"
CUDA_TILE_HEAD = "af2417041cc939b87ef56d92cfdcf61737c5457e"
LLVM_HEAD = "57109befac92811d2253109242ca6fa69c961fb2"
PYPTO_DSO_RELATIVE_PATH = Path(
    "builds/pypto-executable-abi-on-206447c-final/product/"
    "pypto_core.cpython-314-x86_64-linux-gnu.so"
)
PYPTO_DSO_SIZE = 780_535_416
PYPTO_DSO_SHA256 = "15675c471f507b97190b0a770bb16e821c5e99353b65bbbc019988490f59018c"
CUDA_RUNTIME_RELATIVE_PATH = Path(
    "envs/pypto-nvidia/lib/python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13"
)
CUDA_RUNTIME_SIZE = 704_288
CUDA_RUNTIME_SHA256 = "96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0"
EXPECTED_DRIVER_RELEASE = "610.74"
EXPECTED_CUDA_TOOLKIT_VERSION = "13.3.73"
EXPECTED_TORCH_VERSION = "2.13.0+cu130"
EXPECTED_TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
EXPECTED_TORCH_CUDA = "13.0"
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5090 Laptop GPU"
EXPECTED_COMPUTE_CAPABILITY = (12, 0)
EXPECTED_SM_COUNT = 82
PIPELINE_REVISION = "46610e0415598d010981e4bd07d0660c592401ac"
MINIMUM_CUDA_DRIVER_API_VERSION = 13_000
MINIMUM_CUDA_RUNTIME_API_VERSION = 13_000
ENVIRONMENT_LOCK_SHA256 = (
    "29800d50f635e7188e55a6d6f43bfb4b8ac9ab16c4a21687db2960f18941932a"
)
PYTHON_REAL_RELATIVE_PATH = Path("envs/pypto-nvidia/bin/python3.14")
PYTHON_SIZE = 35_989_864
PYTHON_SHA256 = "aa85b78409de29d21c7db9a6ea0479fd73a4e245a733ea325f5ecf21772d030f"
CUDA_RUNTIME_DISTRIBUTION = "nvidia-cuda-runtime"
CUDA_RUNTIME_VERSION = "13.0.96"

RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_nvidia_executable_sm120.py")
RUNNER_SIZE = 43_947
RUNNER_SHA256 = "f22befff45d87097ae42b5725cf33a5e296ed74ff177cd84c2b772be5939abdd"
REPLAY_DIRECTORY_NAME = "pypto-nvidia-executable-sm120"
PROVISIONAL_NAME = "provisional.json"
FINAL_REPORT_DIRECTORY = Path("reports/data")


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Process-independent fixed smoke case identity."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    tile_sizes: tuple[int, ...]
    expected_grid: tuple[int, int, int]
    expected_kernel_arguments: int
    expected_device_code_bytes: int
    expected_device_code_sha256: str
    repetitions: int = 2


CASE_SPECS = (
    CaseSpec(
        name="static",
        dtype="float32",
        shape=(8, 8),
        strides=(8, 1),
        tile_sizes=(16,),
        expected_grid=(4, 1, 1),
        expected_kernel_arguments=3,
        expected_device_code_bytes=13_624,
        expected_device_code_sha256=(
            "6dc121d2574537753229ed537efc5d2558eee26bfac0ad9d21826b5f33632b82"
        ),
    ),
    CaseSpec(
        name="dynamic",
        dtype="float32",
        shape=(17, 9),
        strides=(18, 1),
        tile_sizes=(8, 8),
        expected_grid=(6, 1, 1),
        expected_kernel_arguments=12,
        expected_device_code_bytes=17_408,
        expected_device_code_sha256=(
            "eabdc1377c66f2879a8cf77e43b3f705e4d725b71f6c8b30244521e97d72ed60"
        ),
    ),
    CaseSpec(
        name="scalar",
        dtype="float16",
        shape=(4, 4, 4),
        strides=(16, 4, 1),
        tile_sizes=(16,),
        expected_grid=(4, 1, 1),
        expected_kernel_arguments=3,
        expected_device_code_bytes=13_744,
        expected_device_code_sha256=(
            "fff77b041e032eaae3804105578f49b22fd26cd5d9cb0d483f3170c2bc1a4735"
        ),
    ),
)


def fixed_child_command(workspace: Path) -> list[str]:
    """Return the only direct child accepted by the GPU-smoke lane."""

    root = workspace.resolve()
    return [
        str(root / "envs/pypto-nvidia/bin/python"),
        "-I",
        "-B",
        "-S",
        str(root / RUNNER_RELATIVE_PATH),
    ]


def replay_directory(workspace: Path, run_id: str) -> Path:
    return workspace.resolve() / "runs" / run_id / REPLAY_DIRECTORY_NAME


def provisional_path(workspace: Path, run_id: str) -> Path:
    return replay_directory(workspace, run_id) / PROVISIONAL_NAME


def final_report_path(workspace: Path, run_id: str) -> Path:
    return workspace.resolve() / FINAL_REPORT_DIRECTORY / f"{SMOKE_NAME}-{run_id}.json"
