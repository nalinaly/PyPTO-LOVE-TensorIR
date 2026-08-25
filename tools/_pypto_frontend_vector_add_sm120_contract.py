#!/usr/bin/env python3
"""Pure constants for the v1 PyPTO frontend vector-add SM120 smoke."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SMOKE_SCHEMA_VERSION = 1
SMOKE_NAME = "pypto-frontend-vector-add-sm120"
GPU_SMOKE_POLICY_VERSION = 1
GPU_SMOKE_AUTHORIZATION = (
    "user-authorized-protected-cpu-lane-with-zero-nvidia-runtime-or-compute"
)
GPU_SMOKE_TIMEOUT_SECONDS = 1_800
GPU_SMOKE_MINIMUM_FREE_DISK_GIB = 64

PYPTO_HEAD = "642ff5bd79ee96b9e5a279a2bc945ad7a78362b7"
PYPTO_TREE = "77d8078d8df84dd7cf8544350918e25b8282976d"
TENSOR_IR_HEAD = "1dcb38c20e53d07c97d3781cae538e33901bae30"
CUDA_TILE_HEAD = "af2417041cc939b87ef56d92cfdcf61737c5457e"
LLVM_HEAD = "57109befac92811d2253109242ca6fa69c961fb2"
PYPTO_DSO_RELATIVE_PATH = Path(
    "builds/pypto-structured-build-spec-on-c4cf755/product/"
    "pypto_core.cpython-314-x86_64-linux-gnu.so"
)
PYPTO_DSO_SIZE = 595_300_112
PYPTO_DSO_SHA256 = "4b796b1e1c53386356217f9ea6368468f885e68fb98fa715f90a081031ecc6fb"
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
EXPECTED_SUPPORTED_COMPUTE_DTYPES = ("FP32", "BF16")
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

RUNNER_RELATIVE_PATH = Path("benchmarks/operators/pypto_frontend_vector_add_sm120.py")
RUNNER_SIZE = 42_117
RUNNER_SHA256 = "13504395c8c639bb22aac4d0820e7fe2f953591d0206a8ff357b2eb48a27a73b"
REPLAY_DIRECTORY_NAME = "pypto-frontend-vector-add-sm120"
PROVISIONAL_NAME = "provisional.json"
FINAL_REPORT_DIRECTORY = Path("reports/data")


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Process-independent fixed frontend smoke case identity."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    tile_sizes: tuple[int, ...]
    expected_grid: tuple[int, int, int]
    expected_hir_bytes: int
    expected_hir_sha256: str
    expected_source_ir_digest: str
    expected_static_specialization_digest: str
    expected_symbolic_specialization_digest: str
    expected_argument_abi_digest: str
    expected_result_abi_digest: str
    expected_mutation_abi_digest: str
    expected_callable_abi_digest: str
    expected_device_code_bytes: int
    expected_device_code_sha256: str
    expected_kernel_arguments: int = 3
    repetitions: int = 2


CASE_SPECS = (
    CaseSpec(
        name="fp32_8x8",
        dtype="float32",
        shape=(8, 8),
        strides=(8, 1),
        tile_sizes=(16,),
        expected_grid=(4, 1, 1),
        expected_hir_bytes=2_077,
        expected_hir_sha256=(
            "2a4bf1b74adb2e160ba9a1f6c1596238095d73a1572fd911417f43465b6fced5"
        ),
        expected_source_ir_digest=(
            "0e5fbaf1cd70dffa0c81d43a1d2cad454f97cbf9a57ae5247da1cb27f6a049d3"
        ),
        expected_static_specialization_digest=(
            "0bad0b86c36c86808f12b908692925e6493b4c78027da334d612b57bac5459aa"
        ),
        expected_symbolic_specialization_digest=(
            "1a2d6d32c956e86f41c0dc50dbafe53d33e3f63bd58bd683e78a06595b6ff58c"
        ),
        expected_argument_abi_digest=(
            "2bf59cefa95a2e95ac4e4647a654118b6d8c6fbfaa91a294621a9160d68dbd9b"
        ),
        expected_result_abi_digest=(
            "5a786136731e5c62c80597edbb57551cd010d344bfeab14456a9f6b263e99ea5"
        ),
        expected_mutation_abi_digest=(
            "4adb7fa8fdbdee33582778c543686a6c63953852be40079649e4d2d6f07c766d"
        ),
        expected_callable_abi_digest=(
            "4eee741cbe0c7322f938b83b01b34844c5f12fb1b01da52c2029e6687e24c640"
        ),
        expected_device_code_bytes=13_784,
        expected_device_code_sha256=(
            "dcc529fc856a508642c8b5a98c6fc4e223e10a49cc9f8a200b8984f92b6483ab"
        ),
    ),
    CaseSpec(
        name="bf16_128",
        dtype="bfloat16",
        shape=(128,),
        strides=(1,),
        tile_sizes=(16,),
        expected_grid=(8, 1, 1),
        expected_hir_bytes=1_922,
        expected_hir_sha256=(
            "cc470968297ff4430fdcf43f25bf6cb4b9054e6ba4d67ddd0177040101044baa"
        ),
        expected_source_ir_digest=(
            "c22f2459ad794e89f88de1bbd427f17876c6059b9fd222be706dd5ce300a0a7f"
        ),
        expected_static_specialization_digest=(
            "146fbdd823eb18c77190894a272a39d1298b557acf8d3156226d44d5ce7a6051"
        ),
        expected_symbolic_specialization_digest=(
            "1a2d6d32c956e86f41c0dc50dbafe53d33e3f63bd58bd683e78a06595b6ff58c"
        ),
        expected_argument_abi_digest=(
            "5408cc8d6adb1f52d11d850c745a33dbe6a43c470fa907a238b71371f4bf04c1"
        ),
        expected_result_abi_digest=(
            "e8111628d00263e8f568dc8844a2e0fe5d08576bf77d81682b338dbeb977460a"
        ),
        expected_mutation_abi_digest=(
            "4adb7fa8fdbdee33582778c543686a6c63953852be40079649e4d2d6f07c766d"
        ),
        expected_callable_abi_digest=(
            "e3d31183b4ba5b2f09f01ef777043b9f51818e33d6bdbf18333257211d57c0e8"
        ),
        expected_device_code_bytes=13_784,
        expected_device_code_sha256=(
            "83afb2df234ad90167351d608052d44f86e26a8ca73959369992cd139943bc13"
        ),
    ),
)


def fixed_child_command(workspace: Path) -> list[str]:
    """Return the only direct child accepted by this GPU-smoke lane."""

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
