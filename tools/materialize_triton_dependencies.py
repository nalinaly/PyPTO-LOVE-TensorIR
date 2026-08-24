#!/usr/bin/env python3
"""Materialize exact Triton build inputs under one workspace-owned directory."""

from __future__ import annotations

import argparse
import base64
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import resource
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
TRITON_COMMIT = "5d6048aa0a324e090ada215b609ea76620133845"
TRITON_TREE = "448265acc1eff726c2e528813552865b33546cc9"
TRITON_LLVM_COMMIT = "ac5dc54d509169d387fcfd495d71853d81c46484"
UNREVIEWED = "UNREVIEWED"
REVIEWED_MANIFEST_SHA256: str | None = (
    "29c0736211ba0b286acd562ba097d7f1dea989671003c63a7b988de5afb0fe7d"
)


@dataclass(frozen=True, slots=True)
class PackageSpec:
    name: str
    url: str
    archive_kind: str
    expected_root: str | None = None
    expected_sha256: str | None = None
    expected_bytes: int | None = None


PACKAGE_SPECS = (
    PackageSpec(
        "pybind11",
        "https://files.pythonhosted.org/packages/cd/8a/"
        "37362fc2b949d5f733a8b0f2ff51ba423914cabefe69f1d1b6aab710f5fe/"
        "pybind11-3.0.1-py3-none-any.whl",
        "zip",
        expected_sha256=(
            "aa8f0aa6e0a94d3b64adfc38f560f33f15e589be2175e103c0a33c6bce55ee89"
        ),
        expected_bytes=293_611,
    ),
    PackageSpec(
        "llvm",
        "https://oaitriton.blob.core.windows.net/public/llvm-builds/"
        "llvm-ac5dc54d-ubuntu-x64.tar.gz",
        "tar",
        "llvm-ac5dc54d-ubuntu-x64",
        expected_sha256=(
            "11a11a5a90da7e4b53ef4cf0f259143d14633cae8543a95cb2d99e4af6b902f8"
        ),
        expected_bytes=1_309_519_196,
    ),
    PackageSpec(
        "json",
        "https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip",
        "zip",
        expected_sha256=(
            "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d"
        ),
        expected_bytes=299_825,
    ),
    PackageSpec(
        "ptxas",
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/"
        "linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz",
        "tar",
        "cuda_nvcc-linux-x86_64-12.8.93-archive",
        expected_sha256=(
            "9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3"
        ),
        expected_bytes=79_015_464,
    ),
    PackageSpec(
        "ptxas_blackwell",
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/"
        "linux-x86_64/cuda_nvcc-linux-x86_64-13.1.80-archive.tar.xz",
        "tar",
        "cuda_nvcc-linux-x86_64-13.1.80-archive",
        expected_sha256=(
            "5ed3b7cfe7f12557199773e7769445357ee048958ff51e623e15f36d3393ca8b"
        ),
        expected_bytes=30_014_972,
    ),
    PackageSpec(
        "cuobjdump",
        "https://developer.download.nvidia.com/compute/cuda/redist/"
        "cuda_cuobjdump/linux-x86_64/"
        "cuda_cuobjdump-linux-x86_64-13.1.80-archive.tar.xz",
        "tar",
        "cuda_cuobjdump-linux-x86_64-13.1.80-archive",
        expected_sha256=(
            "3de0169fd8d00e8bdd5ec91a6eb89a78229d82e478cb85554f89748107ba928c"
        ),
        expected_bytes=263_548,
    ),
    PackageSpec(
        "nvdisasm",
        "https://developer.download.nvidia.com/compute/cuda/redist/"
        "cuda_nvdisasm/linux-x86_64/"
        "cuda_nvdisasm-linux-x86_64-13.1.80-archive.tar.xz",
        "tar",
        "cuda_nvdisasm-linux-x86_64-13.1.80-archive",
        expected_sha256=(
            "b169c329bda674e6f9ae5db9845ea09d40f593c96faf11b7f8d4c0a8a2576f17"
        ),
        expected_bytes=4_153_976,
    ),
    PackageSpec(
        "cudacrt",
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_crt/"
        "linux-x86_64/cuda_crt-linux-x86_64-13.1.80-archive.tar.xz",
        "tar",
        "cuda_crt-linux-x86_64-13.1.80-archive",
        expected_sha256=(
            "0e7c365d3301a1b486dbee600b833f6bc771b1a7cc660abca0923269023355ed"
        ),
        expected_bytes=79_624,
    ),
    PackageSpec(
        "cudart",
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/"
        "linux-x86_64/cuda_cudart-linux-x86_64-13.1.80-archive.tar.xz",
        "tar",
        "cuda_cudart-linux-x86_64-13.1.80-archive",
        expected_sha256=(
            "b626f4790f46bc9324a1047f2fbcc9a42bc4a722b053056e61cc00da54ad6f32"
        ),
        expected_bytes=1_549_648,
    ),
    PackageSpec(
        "cupti",
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/"
        "linux-x86_64/cuda_cupti-linux-x86_64-12.8.90-archive.tar.xz",
        "tar",
        "cuda_cupti-linux-x86_64-12.8.90-archive",
        expected_sha256=(
            "7bf5db86cb82f26a6a3cb9e2fa4dc2a131d25885f59fdbc647938929924405db"
        ),
        expected_bytes=15_383_056,
    ),
)


PACKAGE_RESOURCE_LIMITS = {
    "pybind11": (10 << 20, 50 << 20, 5_000),
    "llvm": (3 << 30, 12 << 30, 50_000),
    "json": (50 << 20, 250 << 20, 10_000),
    "ptxas": (500 << 20, 1 << 30, 5_000),
    "ptxas_blackwell": (500 << 20, 1 << 30, 5_000),
    "cuobjdump": (200 << 20, 500 << 20, 5_000),
    "nvdisasm": (500 << 20, 1 << 30, 5_000),
    "cudacrt": (1 << 30, 4 << 30, 50_000),
    "cudart": (1 << 30, 4 << 30, 50_000),
    "cupti": (2 << 30, 6 << 30, 50_000),
}


MATERIALIZATION_HEADROOM_BYTES = 8 << 30
MAX_ARCHIVE_PATH_BYTES = 4096
MAX_ARCHIVE_COMPONENT_BYTES = 255
DOWNLOAD_CHUNK_BYTES = 64 << 10
DOWNLOAD_FSYNC_INTERVAL_BYTES = 64 << 20
DOWNLOAD_MAX_REQUESTS = 4
DOWNLOAD_TIMEOUT_SECONDS = 300
DOWNLOAD_USER_AGENT = "pypto-triton-dependency-materializer/2"
CONTENT_RANGE_PATTERN = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")
STRONG_ETAG_PATTERN = re.compile(r'"[!#-~]*"')


class DownloadContractError(RuntimeError):
    """The peer response cannot be safely interpreted as archive bytes."""


class RetryableDownloadError(RuntimeError):
    """The current response ended before all of its declared bytes arrived."""


LOCK_EXPECTATIONS = {
    "tensor_ir.llvm.commit": "57109befac92811d2253109242ca6fa69c961fb2",
    "pytorch.repo": "https://github.com/pytorch/pytorch.git",
    "pytorch.commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
    "pytorch.tree": "7cda5eae52ace99ca4daa7e623920cc93782cc6c",
    "triton.repo": "https://github.com/triton-lang/triton.git",
    "triton.commit": TRITON_COMMIT,
    "triton.tree": TRITON_TREE,
    "triton.source_archive_sha256": (
        "2ebfd3f7e98dee2e8524b9b210716fbe1f07759b6d89307280a9b10ae359b43e"
    ),
    "triton.version": "3.7.1+git5d6048aa",
    "triton.module_version": "3.7.1",
    "triton.pytorch_pin_blob": "912c4468080ce91efafe5f1fb6364bca0ced2d51",
    "triton.llvm.commit": TRITON_LLVM_COMMIT,
    "triton.libdevice_sha256": (
        "5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee"
    ),
    "triton.dependencies.reviewed_manifest_sha256": REVIEWED_MANIFEST_SHA256,
    "triton.dependencies.archive.pybind11.sha256": (
        "aa8f0aa6e0a94d3b64adfc38f560f33f15e589be2175e103c0a33c6bce55ee89"
    ),
    "triton.dependencies.archive.pybind11.bytes": "293611",
    "triton.dependencies.archive.llvm.sha256": (
        "11a11a5a90da7e4b53ef4cf0f259143d14633cae8543a95cb2d99e4af6b902f8"
    ),
    "triton.dependencies.archive.llvm.bytes": "1309519196",
    "triton.dependencies.archive.json.sha256": (
        "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d"
    ),
    "triton.dependencies.archive.json.bytes": "299825",
    "triton.dependencies.archive.ptxas.sha256": (
        "9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3"
    ),
    "triton.dependencies.archive.ptxas.bytes": "79015464",
    "triton.dependencies.archive.ptxas_blackwell.sha256": (
        "5ed3b7cfe7f12557199773e7769445357ee048958ff51e623e15f36d3393ca8b"
    ),
    "triton.dependencies.archive.ptxas_blackwell.bytes": "30014972",
    "triton.dependencies.archive.cuobjdump.sha256": (
        "3de0169fd8d00e8bdd5ec91a6eb89a78229d82e478cb85554f89748107ba928c"
    ),
    "triton.dependencies.archive.cuobjdump.bytes": "263548",
    "triton.dependencies.archive.nvdisasm.sha256": (
        "b169c329bda674e6f9ae5db9845ea09d40f593c96faf11b7f8d4c0a8a2576f17"
    ),
    "triton.dependencies.archive.nvdisasm.bytes": "4153976",
    "triton.dependencies.archive.cudacrt.sha256": (
        "0e7c365d3301a1b486dbee600b833f6bc771b1a7cc660abca0923269023355ed"
    ),
    "triton.dependencies.archive.cudacrt.bytes": "79624",
    "triton.dependencies.archive.cudart.sha256": (
        "b626f4790f46bc9324a1047f2fbcc9a42bc4a722b053056e61cc00da54ad6f32"
    ),
    "triton.dependencies.archive.cudart.bytes": "1549648",
    "triton.dependencies.archive.cupti.sha256": (
        "7bf5db86cb82f26a6a3cb9e2fa4dc2a131d25885f59fdbc647938929924405db"
    ),
    "triton.dependencies.archive.cupti.bytes": "15383056",
    "triton.toolchain.ptxas": "12.8.93",
    "triton.toolchain.ptxas_blackwell": "13.1.80",
    "triton.toolchain.cuobjdump": "13.1.80",
    "triton.toolchain.nvdisasm": "13.1.80",
    "triton.toolchain.cudacrt": "13.1.80",
    "triton.toolchain.cudart": "13.1.80",
    "triton.toolchain.cupti": "12.8.90",
    "triton.recipe.pybind11": "3.0.1",
    "triton.recipe.pybind11_wheel_url": (
        "https://files.pythonhosted.org/packages/cd/8a/"
        "37362fc2b949d5f733a8b0f2ff51ba423914cabefe69f1d1b6aab710f5fe/"
        "pybind11-3.0.1-py3-none-any.whl"
    ),
    "triton.recipe.pybind11_wheel_sha256": (
        "aa8f0aa6e0a94d3b64adfc38f560f33f15e589be2175e103c0a33c6bce55ee89"
    ),
    "triton.recipe.pybind11_wheel_bytes": "293611",
    "triton.recipe.json": "3.11.3",
    "triton.recipe.ext_enabled": "ON",
    "triton.recipe.clang_lld": "ON",
    "triton.recipe.ccache": "OFF",
    "triton.recipe.offline_build": "ON",
    "triton.recipe.build_proton": "ON",
    "triton.recipe.build_ut": "OFF",
    "triton.recipe.build_type": "TritonRelBuildWithAsserts",
    "triton.recipe.wheel_version_suffix": "+git5d6048aa",
    "triton.recipe.max_jobs": "2",
    "triton.recipe.cmake_build_parallel_level": "2",
    "triton.recipe.parallel_link_jobs": "1",
    "triton.recipe.source_date_epoch": "1781015236",
    "triton.producer.python": "3.14.6",
    "triton.producer.setuptools": "83.0.0",
    "triton.producer.wheel": "0.47.0",
    "triton.producer.build": "1.5.0",
    "triton.producer.cmake": "3.31.10",
    "triton.producer.cmake_wrapper_path": "envs/pypto-nvidia/bin/cmake",
    "triton.producer.cmake_wrapper_sha256": (
        "8e510409ba5512d10ddd4a732c07d95cde22eeb3b6dfa5864124b1ffc70b53c0"
    ),
    "triton.producer.cmake_payload_path": (
        "envs/pypto-nvidia/lib/python3.14/site-packages/cmake/data/bin/cmake"
    ),
    "triton.producer.cmake_payload_sha256": (
        "576c050dab1e1418b6703b5cfb523330567683dad0c60a5ff9cc23128143812e"
    ),
    "triton.producer.ninja": "1.13.0",
    "triton.producer.ninja_sha256": (
        "696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67"
    ),
    "triton.producer.lit": "18.1.8",
    "triton.producer.packaging": "26.2",
    "triton.producer.pyproject_hooks": "1.2.0",
    "triton.producer.clang": "Ubuntu clang 21.1.8 (6ubuntu1)",
    "triton.producer.clang_path": "/usr/lib/llvm-21/bin/clang",
    "triton.producer.clang_sha256": (
        "412bbe8c60571a1eb06f48fde89635033621caeb01a9b4ee76d46711bae8e932"
    ),
    "triton.producer.lld": "Ubuntu LLD 21.1.8",
    "triton.producer.lld_command_path": "/usr/bin/lld",
    "triton.producer.lld_path": "/usr/lib/llvm-21/bin/lld",
    "triton.producer.lld_sha256": (
        "6a65863a9eba1af6b6e8969f8e96a5ad4df0e8b705f98491a28b1790ce35718c"
    ),
    "triton.producer.python_path": "envs/pypto-nvidia/bin/python3.14",
    "triton.producer.python_sha256": (
        "aa85b78409de29d21c7db9a6ea0479fd73a4e245a733ea325f5ecf21772d030f"
    ),
    "triton.producer.bubblewrap": "0.11.1",
    "triton.producer.bubblewrap_sha256": (
        "0abea81db798ebf6b4742ac0664802d97521547a353c2a0dbdc21d76cbbfd2c0"
    ),
    "triton.audit.readelf": "GNU readelf (GNU Binutils for Ubuntu) 2.46",
    "triton.audit.readelf_sha256": (
        "c857339616bbbfa5eba32733e22365048903fbaf6ed2126b897dd138bcb741fc"
    ),
    "triton.audit.ldd": "ldd (Ubuntu GLIBC 2.43-2ubuntu2.3) 2.43",
    "triton.audit.ldd_sha256": (
        "b6c9a28572ea3920442c2f5b1ea11b0999adc407913ffdd7f92e530dfc051894"
    ),
    "triton.producer.binutils": "GNU Binutils for Ubuntu 2.46",
    "triton.producer.ar_sha256": (
        "531473816bf553e863df5aab14c8177c72b732cf80c51dcf0fa990a50125041c"
    ),
    "triton.producer.ranlib_sha256": (
        "369cf0d60a6167b11f39ed9b4bbb3d93903cc364975b244e1d084aaccf48dc92"
    ),
    "triton.producer.nm_sha256": (
        "f04262bf48192a7cbb78a17ca49ae03f8930b0372bd0115576f957b2e2a57a01"
    ),
    "triton.producer.strip_sha256": (
        "4d2ca6ba80677c3b2975e328306a779cd7bd6590a87948b5ebea9c1b41a049c8"
    ),
    "triton.producer.objcopy_sha256": (
        "05f4473d24f7330a9b13f43d007d619a2e792e33de453cd7533a2d97c30da770"
    ),
    "triton.producer.ld_bfd_sha256": (
        "97f48d93b8b076a92d2809ec29dcb17f0f37c8827358f832255e2ed22fef6075"
    ),
    "triton.producer.ld_library_path": (
        "envs/pypto-nvidia/lib:/usr/lib/wsl/lib"
    ),
    "triton.producer.identity_schema": "2",
    "triton.producer.identity_scope": (
        "workspace-prefix-and-selected-host-build-tools"
    ),
    "triton.producer.distribution_set_sha256": (
        "21d700f943d3ca94963d10368bdeb8f17ab1e131dc8f792df68c7055e9ff0896"
    ),
    "triton.producer.record_rewrite_count": "6",
    "triton.producer.record_rewrite_policy_sha256": (
        "cf63e3f6fef06dfd2e4072d3b22382b54b2dd3743215a8c37144f04ad3c42ec6"
    ),
    "triton.producer.site_projection_sha256": (
        "b0ccdda495e52c61a5fc3c05c87677dee95c435cc641c626f6cc343b0dc4a6f0"
    ),
    "triton.producer.site_projection_files": "963",
    "triton.producer.site_projection_directories": "134",
    "triton.producer.site_projection_bytes": "9485023",
    "triton.producer.selected_identity_sha256": (
        "920d856bd68402812f821a437d310b7de41b316c4cfebc95f5b29fe20d64de9c"
    ),
}


PRODUCER_EXECUTABLES = {
    "bubblewrap": Path("/usr/bin/bwrap"),
    "ar": Path("/usr/bin/ar"),
    "ranlib": Path("/usr/bin/ranlib"),
    "nm": Path("/usr/bin/nm"),
    "strip": Path("/usr/bin/strip"),
    "objcopy": Path("/usr/bin/objcopy"),
    "ld.bfd": Path("/usr/bin/ld"),
    "clang": Path("/usr/lib/llvm-21/bin/clang"),
    "lld": Path("/usr/lib/llvm-21/bin/lld"),
    "cmake": (
        ROOT
        / "envs/pypto-nvidia/lib/python3.14/site-packages/cmake/data/bin/cmake"
    ),
    "ninja": ROOT / "envs/pypto-nvidia/bin/ninja",
    "python": ROOT / "envs/pypto-nvidia/bin/python3.14",
}


PRODUCER_PACKAGE_VERSIONS = {
    "build": "1.5.0",
    "cmake": "3.31.10",
    "lit": "18.1.8",
    "ninja": "1.13.0",
    "packaging": "26.2",
    "pyproject-hooks": "1.2.0",
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
}


PRODUCER_RECORD_REWRITES: dict[tuple[str, str], dict[str, object]] = {
    ("cmake", "../../../bin/ccmake"): {
        "path": "bin/ccmake",
        "mode": 0o755,
        "record_size": 194,
        "record_sha256": (
            "f9b9120e3ac45fe1b0dcfab5eae8e981f310196687cf9cf9e2ffd9dfdbd69fd8"
        ),
        "actual_size": 206,
        "actual_sha256": (
            "25a8600d4692b5c5e8b9e26e38eb6e6b2f3e0d505a2e5cf25b9e817464602b90"
        ),
    },
    ("cmake", "../../../bin/cmake"): {
        "path": "bin/cmake",
        "mode": 0o755,
        "record_size": 192,
        "record_sha256": (
            "41cea7d252523ad44069250e7d769e481a6868de1903fa201f949d7e66545397"
        ),
        "actual_size": 204,
        "actual_sha256": (
            "8e510409ba5512d10ddd4a732c07d95cde22eeb3b6dfa5864124b1ffc70b53c0"
        ),
    },
    ("cmake", "../../../bin/cpack"): {
        "path": "bin/cpack",
        "mode": 0o755,
        "record_size": 192,
        "record_sha256": (
            "1cf14769b6360eb7eaa85bf51571788c6150dfb80b3ee9964acf4eb9b794adbc"
        ),
        "actual_size": 204,
        "actual_sha256": (
            "ff412a920de1aca36cd066dc10145be13855ef8e6eda525586cf8ca515036fea"
        ),
    },
    ("cmake", "../../../bin/ctest"): {
        "path": "bin/ctest",
        "mode": 0o755,
        "record_size": 192,
        "record_sha256": (
            "f9f076d43eca2252810c24a38205da0d7aa3fc933bcb1000a64524345d95bfdb"
        ),
        "actual_size": 204,
        "actual_sha256": (
            "fccf0b6f6b32e7d1528b224964340481aa7a3bc3df66621708daf64099940d8e"
        ),
    },
    ("lit", "../../../bin/lit"): {
        "path": "bin/lit",
        "mode": 0o755,
        "record_size": 193,
        "record_sha256": (
            "49edb992450b52ba73530c0a44b4ea5167f8b16fb710501020af3c4089bbff64"
        ),
        "actual_size": 205,
        "actual_sha256": (
            "520448d685c54193d9873d53d89cc69703cc570bd2eeea1c3957ee38347e34ee"
        ),
    },
    ("wheel", "../../../bin/wheel"): {
        "path": "bin/wheel",
        "mode": 0o755,
        "record_size": 200,
        "record_sha256": (
            "5017ad807bbfdc38086415970b465118ee15fc07e674670c559779488e60a99f"
        ),
        "actual_size": 212,
        "actual_sha256": (
            "0f315905e4f65dcc2bf6c5233fc1e1e2c8abbbab28b14455b184b48481df67c9"
        ),
    },
}


PRODUCER_REWRITE_ENTRY_POINTS = {
    ("cmake", "../../../bin/ccmake"): ("ccmake", "cmake:ccmake"),
    ("cmake", "../../../bin/cmake"): ("cmake", "cmake:cmake"),
    ("cmake", "../../../bin/cpack"): ("cpack", "cmake:cpack"),
    ("cmake", "../../../bin/ctest"): ("ctest", "cmake:ctest"),
    ("lit", "../../../bin/lit"): ("lit", "lit.main:main"),
    ("wheel", "../../../bin/wheel"): ("wheel", "wheel._commands:main"),
}


PRODUCER_CONSOLE_SHEBANG = (
    "#!/home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/bin/python"
)


PRODUCER_IDENTITY_SCHEMA = 2
PRODUCER_IDENTITY_SCOPE = "workspace-prefix-and-selected-host-build-tools"


PRODUCER_LD_LIBRARY_PATH = (
    f"{ROOT / 'envs/pypto-nvidia/lib'}:/usr/lib/wsl/lib"
)


OVERLAY_TOOL_VERSIONS = {
    "ptxas": "12.8.93",
    "ptxas-blackwell": "13.1.80",
    "cuobjdump": "13.1.80",
    "nvdisasm": "13.1.80",
}


def git_output(root: Path, *args: str) -> str:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1"})
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    ).stdout.strip()


def validate_source_pins() -> None:
    pytorch = ROOT / "upstream/pytorch"
    triton = ROOT / "upstream/triton"
    if git_output(pytorch, "rev-parse", "HEAD^{commit}") != LOCK_EXPECTATIONS[
        "pytorch.commit"
    ]:
        raise RuntimeError("PyTorch source commit drift")
    if git_output(pytorch, "rev-parse", "HEAD^{tree}") != LOCK_EXPECTATIONS[
        "pytorch.tree"
    ]:
        raise RuntimeError("PyTorch source tree drift")
    if git_output(pytorch, "remote", "get-url", "origin") != LOCK_EXPECTATIONS[
        "pytorch.repo"
    ]:
        raise RuntimeError("PyTorch source origin drift")
    if git_output(pytorch, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("PyTorch source worktree is dirty")
    pin_path = ".ci/docker/ci_commit_pins/triton.txt"
    if git_output(pytorch, "show", f"HEAD:{pin_path}") != TRITON_COMMIT:
        raise RuntimeError("PyTorch Triton pin drift")
    if git_output(pytorch, "rev-parse", f"HEAD:{pin_path}") != LOCK_EXPECTATIONS[
        "triton.pytorch_pin_blob"
    ]:
        raise RuntimeError("PyTorch Triton pin blob drift")
    if git_output(pytorch, "show", "HEAD:.ci/docker/triton_version.txt") != "3.7.1":
        raise RuntimeError("PyTorch Triton version drift")
    if git_output(triton, "rev-parse", "HEAD^{commit}") != TRITON_COMMIT:
        raise RuntimeError("Triton source commit drift")
    if git_output(triton, "rev-parse", "HEAD^{tree}") != TRITON_TREE:
        raise RuntimeError("Triton source tree drift")
    if git_output(triton, "remote", "get-url", "origin") != LOCK_EXPECTATIONS[
        "triton.repo"
    ]:
        raise RuntimeError("Triton source origin drift")
    if git_output(triton, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Triton source worktree is dirty")
    gitlinks = git_output(triton, "ls-tree", "-r", TRITON_COMMIT).splitlines()
    if any(line.startswith("160000 ") for line in gitlinks):
        raise RuntimeError("Triton source unexpectedly contains gitlinks")
    try:
        git_output(triton, "cat-file", "-e", f"{TRITON_COMMIT}:.gitmodules")
    except subprocess.CalledProcessError:
        pass
    else:
        raise RuntimeError("Triton source unexpectedly contains .gitmodules")
    if git_output(triton, "show", f"{TRITON_COMMIT}:cmake/llvm-hash.txt") != (
        TRITON_LLVM_COMMIT
    ):
        raise RuntimeError("Triton LLVM pin drift")
    for field in ("%at", "%ct"):
        if git_output(triton, "show", "-s", f"--format={field}", TRITON_COMMIT) != (
            LOCK_EXPECTATIONS["triton.recipe.source_date_epoch"]
        ):
            raise RuntimeError("Triton commit timestamp drift")
    toolchain = json.loads(
        git_output(
            triton,
            "show",
            f"{TRITON_COMMIT}:cmake/nvidia-toolchain-version.json",
        )
    )
    expected_toolchain = {
        "ptxas-blackwell": "13.1.80",
        "ptxas": "12.8.93",
        "cuobjdump": "13.1.80",
        "nvdisasm": "13.1.80",
        "cudacrt": "13.1.80",
        "cudart": "13.1.80",
        "cupti": "12.8.90",
    }
    if toolchain != expected_toolchain:
        raise RuntimeError("Triton NVIDIA toolchain pin drift")
    libdevice = triton / "third_party/nvidia/backend/lib/libdevice.10.bc"
    if sha256_file(libdevice) != LOCK_EXPECTATIONS["triton.libdevice_sha256"]:
        raise RuntimeError("Triton libdevice source drift")


def load_versions_lock(path: Path = ROOT / "VERSIONS.lock") -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise RuntimeError(f"malformed or duplicate VERSIONS.lock line: {raw_line!r}")
        result[key] = value
    return result


def validate_versions_lock(path: Path = ROOT / "VERSIONS.lock") -> None:
    if set(PACKAGE_RESOURCE_LIMITS) != {spec.name for spec in PACKAGE_SPECS}:
        raise RuntimeError("Triton dependency resource-limit set mismatch")
    observed = load_versions_lock(path)
    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in LOCK_EXPECTATIONS.items()
        if observed.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Triton VERSIONS.lock mismatch: {mismatches}")
    for spec in PACKAGE_SPECS:
        if spec.expected_sha256 is not None and re.fullmatch(
            r"[0-9a-f]{64}", spec.expected_sha256
        ) is None:
            raise RuntimeError(
                f"Triton dependency archive SHA-256 is malformed: {spec.name}"
            )
        if spec.expected_bytes is not None and (
            not _exact_nonnegative_int(spec.expected_bytes)
            or spec.expected_bytes == 0
            or spec.expected_bytes > PACKAGE_RESOURCE_LIMITS[spec.name][0]
        ):
            raise RuntimeError(
                f"Triton dependency archive size is invalid: {spec.name}"
            )
        locked_digest = observed.get(
            f"triton.dependencies.archive.{spec.name}.sha256"
        )
        expected_digest = spec.expected_sha256 or UNREVIEWED
        if locked_digest != expected_digest:
            raise RuntimeError(
                f"Triton dependency archive lock mismatch: {spec.name}"
            )
        if spec.expected_bytes is not None and observed.get(
            f"triton.dependencies.archive.{spec.name}.bytes"
        ) != str(spec.expected_bytes):
            raise RuntimeError(
                f"Triton dependency archive size lock mismatch: {spec.name}"
            )
    locked_manifest = observed.get(
        "triton.dependencies.reviewed_manifest_sha256"
    )
    expected_manifest = REVIEWED_MANIFEST_SHA256 or UNREVIEWED
    if locked_manifest != expected_manifest:
        raise RuntimeError("Triton reviewed manifest source/lock mismatch")


def _ldd_paths(executable: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["ldd", str(executable)],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "LD_LIBRARY_PATH": PRODUCER_LD_LIBRARY_PATH},
    )
    if "not found" in result.stdout.lower():
        raise RuntimeError(f"producer dependency is missing for {executable}")
    paths: set[Path] = set()
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        match = re.search(r"=>\s+(/\S+)", line)
        if match is None:
            match = re.match(r"(/\S+)\s+\(", line)
        if match is None:
            continue
        path = Path(match.group(1)).resolve()
        if not path.is_file():
            raise RuntimeError(f"ldd resolved a non-file dependency: {path}")
        paths.add(path)
    return tuple(sorted(paths))


def _record_sha256(package_path: importlib.metadata.PackagePath) -> str | None:
    if package_path.hash is None:
        return None
    if package_path.hash.mode != "sha256":
        raise RuntimeError(
            f"producer RECORD uses non-SHA256 hash: {package_path}"
        )
    padded = package_path.hash.value + "=" * (-len(package_path.hash.value) % 4)
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    except ValueError as error:
        raise RuntimeError(
            f"producer RECORD has malformed SHA256: {package_path}"
        ) from error
    if len(decoded) != hashlib.sha256().digest_size:
        raise RuntimeError(
            f"producer RECORD SHA256 has wrong length: {package_path}"
        )
    return decoded.hex()


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _producer_distribution(name: str) -> importlib.metadata.Distribution:
    site_packages = (
        ROOT / "envs/pypto-nvidia/lib/python3.14/site-packages"
    ).resolve()
    expected_name = _canonical_distribution_name(name)
    candidates = [
        distribution
        for distribution in importlib.metadata.distributions(
            path=[os.fspath(site_packages)]
        )
        if _canonical_distribution_name(distribution.metadata["Name"] or "")
        == expected_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"producer distribution must have one fixed-prefix candidate: "
            f"{name} ({len(candidates)} found)"
        )
    return candidates[0]


def distribution_record_identity(name: str) -> dict[str, object]:
    distribution = _producer_distribution(name)
    environment = (ROOT / "envs/pypto-nvidia").resolve()
    site_packages = (environment / "lib/python3.14/site-packages").resolve()
    dist_info = Path(getattr(distribution, "_path", ""))
    if not dist_info.is_absolute():
        dist_info = site_packages / dist_info
    dist_info = Path(os.path.abspath(os.fspath(dist_info)))
    if (
        dist_info.is_symlink()
        or not dist_info.is_dir()
        or dist_info.parent != site_packages
        or not dist_info.name.endswith(".dist-info")
    ):
        raise RuntimeError(f"producer dist-info path is not fixed: {dist_info}")
    expected_record_path = f"{dist_info.name}/RECORD"
    entries: list[dict[str, object]] = []
    observed_rewrites: set[tuple[str, str]] = set()
    record_paths: set[str] = set()
    installed_paths: set[str] = set()
    own_record_entries = 0
    package_paths = distribution.files
    if not package_paths:
        raise RuntimeError(f"producer distribution has no RECORD files: {name}")
    all_record_paths = {package_path.as_posix() for package_path in package_paths}
    if len(all_record_paths) != len(package_paths):
        raise RuntimeError(f"producer distribution has duplicate RECORD paths: {name}")
    console_scripts: dict[str, str] = {}
    for entry_point in distribution.entry_points:
        if entry_point.group != "console_scripts":
            continue
        if entry_point.name in console_scripts:
            raise RuntimeError(
                f"producer distribution has duplicate console script: "
                f"{entry_point.name}"
            )
        console_scripts[entry_point.name] = entry_point.value
    record_file_sha256: str | None = None
    for package_path in sorted(package_paths, key=str):
        record_path = package_path.as_posix()
        if record_path in record_paths:
            raise RuntimeError(
                f"producer distribution has duplicate RECORD path: {record_path}"
            )
        record_paths.add(record_path)
        path = Path(os.path.abspath(os.fspath(distribution.locate_file(package_path))))
        try:
            relative = path.relative_to(environment).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"producer distribution file escapes environment: {path}"
            ) from error
        if relative in installed_paths:
            raise RuntimeError(
                f"producer distribution aliases an installed path: {relative}"
            )
        installed_paths.add(relative)
        resolved = path.resolve(strict=False)
        if resolved == environment or environment not in resolved.parents:
            raise RuntimeError(
                f"producer distribution file resolves outside environment: {path}"
            )
        if not path.exists() and not path.is_symlink():
            raise RuntimeError(f"producer distribution file is absent: {path}")
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            raise RuntimeError(
                f"producer distribution contains an unsupported symlink: {path}"
            )
        elif path.is_file():
            digest = sha256_file(path)
            size = path.stat().st_size
            declared_digest = _record_sha256(package_path)
            if (package_path.size is None) != (declared_digest is None):
                raise RuntimeError(
                    f"producer RECORD has partial hash/size metadata: {path}"
                )
            if package_path.size is None:
                is_record = record_path == expected_record_path
                is_generated_pyc = (
                    "/__pycache__/" in f"/{record_path}"
                    and record_path.endswith(".pyc")
                )
                if is_generated_pyc:
                    source_path = re.sub(
                        r"/__pycache__/([^/]+)\.cpython-314(?:\.opt-[12])?\.pyc$",
                        r"/\1.py",
                        record_path,
                    )
                    if source_path == record_path or source_path not in all_record_paths:
                        raise RuntimeError(
                            f"producer RECORD has an unowned generated pyc: {path}"
                        )
                if not (is_record or is_generated_pyc):
                    raise RuntimeError(
                        f"producer RECORD omits hash/size unexpectedly: {path}"
                    )
                if is_record:
                    own_record_entries += 1
                    record_file_sha256 = digest
            entry = {
                "record_path": record_path,
                "path": relative,
                "kind": "file",
                "mode": mode,
                "size": size,
                "sha256": digest,
                "record_size": package_path.size,
                "record_sha256": declared_digest,
            }
            record_matches = (
                (package_path.size is None or package_path.size == size)
                and (declared_digest is None or declared_digest == digest)
            )
            entry["record_matches"] = record_matches
            rewrite_key = (name, record_path)
            if not record_matches:
                observed = {
                    "path": relative,
                    "mode": mode,
                    "record_size": package_path.size,
                    "record_sha256": declared_digest,
                    "actual_size": size,
                    "actual_sha256": digest,
                }
                expected_bytes = PRODUCER_RECORD_REWRITES.get(rewrite_key)
                expected_entry_point = PRODUCER_REWRITE_ENTRY_POINTS.get(rewrite_key)
                if expected_bytes is None or expected_entry_point is None:
                    raise RuntimeError(f"unexpected producer RECORD rewrite: {path}")
                entry_point_name, entry_point_value = expected_entry_point
                observed.update(
                    {
                        "entry_point_name": entry_point_name,
                        "entry_point_value": console_scripts.get(entry_point_name),
                        "shebang": path.read_bytes().splitlines()[0].decode(
                            "utf-8", errors="strict"
                        ),
                    }
                )
                expected = {
                    **expected_bytes,
                    "entry_point_name": entry_point_name,
                    "entry_point_value": entry_point_value,
                    "shebang": PRODUCER_CONSOLE_SHEBANG,
                }
                if observed != expected:
                    raise RuntimeError(f"producer RECORD rewrite drift: {path}")
                observed_rewrites.add(rewrite_key)
        else:
            raise RuntimeError(f"unsupported producer distribution entry: {path}")
        entries.append(entry)
    if own_record_entries != 1:
        raise RuntimeError(
            f"producer distribution must contain exactly one RECORD entry: {name}"
        )
    expected_rewrites = {
        key for key in PRODUCER_RECORD_REWRITES if key[0] == name
    }
    if set(PRODUCER_RECORD_REWRITES) != set(PRODUCER_REWRITE_ENTRY_POINTS):
        raise RuntimeError("producer RECORD rewrite/entry-point policy mismatch")
    if observed_rewrites != expected_rewrites:
        raise RuntimeError(
            f"producer RECORD rewrite set drift for {name}: "
            f"expected {sorted(expected_rewrites)}, got {sorted(observed_rewrites)}"
        )
    payload = {
        "name": name,
        "version": distribution.version,
        "dist_info_path": dist_info.relative_to(environment).as_posix(),
        "entries": entries,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "version": distribution.version,
        "dist_info_path": dist_info.relative_to(environment).as_posix(),
        "record_sha256": record_file_sha256,
        "files": len(entries),
        "record_rewrites": len(observed_rewrites),
        "distribution_identity_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _compact_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def producer_record_rewrite_policy_sha256() -> str:
    if set(PRODUCER_RECORD_REWRITES) != set(PRODUCER_REWRITE_ENTRY_POINTS):
        raise RuntimeError("producer RECORD rewrite/entry-point policy mismatch")
    records = []
    for name, record_path in sorted(PRODUCER_RECORD_REWRITES):
        entry_point_name, entry_point_value = PRODUCER_REWRITE_ENTRY_POINTS[
            (name, record_path)
        ]
        records.append(
            {
                "distribution": name,
                "record_path": record_path,
                **PRODUCER_RECORD_REWRITES[(name, record_path)],
                "entry_point_name": entry_point_name,
                "entry_point_value": entry_point_value,
                "shebang": PRODUCER_CONSOLE_SHEBANG,
            }
        )
    return _compact_json_sha256(records)


def collect_live_producer_identity() -> dict[str, object]:
    executable_records: list[dict[str, str]] = []
    dynamic_paths: set[Path] = set()
    for name, configured_path in PRODUCER_EXECUTABLES.items():
        path = configured_path.resolve()
        if not path.is_file():
            raise RuntimeError(f"producer executable is absent: {path}")
        executable_records.append(
            {"name": name, "path": str(path), "sha256": sha256_file(path)}
        )
        dynamic_paths.update(_ldd_paths(path))
    dynamic_records = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(dynamic_paths)
    ]
    package_records = {
        name: distribution_record_identity(name)
        for name in sorted(PRODUCER_PACKAGE_VERSIONS)
    }
    package_versions = {
        name: record["version"] for name, record in package_records.items()
    }
    distribution_set_sha256 = _compact_json_sha256(package_records)
    identity: dict[str, object] = {
        "identity_schema": PRODUCER_IDENTITY_SCHEMA,
        "identity_scope": PRODUCER_IDENTITY_SCOPE,
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "package_distributions": package_records,
        "distribution_set_sha256": distribution_set_sha256,
        "record_rewrite_count": len(PRODUCER_RECORD_REWRITES),
        "record_rewrite_policy_sha256": producer_record_rewrite_policy_sha256(),
        "executables": executable_records,
        "dynamic_libraries": dynamic_records,
    }
    identity["selected_producer_identity_sha256"] = _compact_json_sha256(identity)
    return identity


def validate_live_producers(lock_path: Path = ROOT / "VERSIONS.lock") -> None:
    observed = collect_live_producer_identity()
    if observed["python_version"] != "3.14.6":
        raise RuntimeError("Triton producer Python version drift")
    if observed["package_versions"] != PRODUCER_PACKAGE_VERSIONS:
        raise RuntimeError("Triton producer Python packages drift")
    locked = load_versions_lock(lock_path)
    scalar_fields = {
        "identity_schema": "triton.producer.identity_schema",
        "identity_scope": "triton.producer.identity_scope",
        "distribution_set_sha256": "triton.producer.distribution_set_sha256",
        "record_rewrite_count": "triton.producer.record_rewrite_count",
        "record_rewrite_policy_sha256": (
            "triton.producer.record_rewrite_policy_sha256"
        ),
    }
    for observed_key, lock_key in scalar_fields.items():
        if str(observed[observed_key]) != locked.get(lock_key):
            raise RuntimeError(f"Triton producer {observed_key} drift")
    distribution_fields = {
        "dist_info_path": "dist_info_path",
        "record_sha256": "record_sha256",
        "files": "files",
        "record_rewrites": "record_rewrites",
        "distribution_identity_sha256": "identity_sha256",
    }
    for name, record in observed["package_distributions"].items():
        lock_name = name.replace("-", "_")
        for observed_key, lock_suffix in distribution_fields.items():
            lock_key = f"triton.producer.dist.{lock_name}.{lock_suffix}"
            if str(record[observed_key]) != locked.get(lock_key):
                raise RuntimeError(
                    f"Triton producer distribution {name} {observed_key} drift"
                )
    cmake_wrapper = ROOT / locked["triton.producer.cmake_wrapper_path"]
    require_regular_file(cmake_wrapper, "Triton producer CMake wrapper")
    if sha256_file(cmake_wrapper) != locked["triton.producer.cmake_wrapper_sha256"]:
        raise RuntimeError("Triton producer CMake wrapper drift")
    if observed["selected_producer_identity_sha256"] != locked.get(
        "triton.producer.selected_identity_sha256"
    ):
        raise RuntimeError("Triton selected producer identity drift")


def assemble_python_producer_site(
    destination: Path,
    distribution_names: tuple[str, ...],
) -> dict[str, object]:
    destination = require_below_workspace(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"producer site already exists: {destination}")
    unknown = set(distribution_names) - set(PRODUCER_PACKAGE_VERSIONS)
    if unknown:
        raise ValueError(f"unknown producer distributions: {sorted(unknown)}")
    site_packages = (
        ROOT / "envs/pypto-nvidia/lib/python3.14/site-packages"
    ).resolve()
    destination.mkdir(parents=True)
    installed: dict[str, tuple[int, int, str]] = {}
    for name in distribution_names:
        distribution_record_identity(name)
        distribution = _producer_distribution(name)
        for package_path in sorted(distribution.files or (), key=str):
            source = Path(
                os.path.abspath(os.fspath(distribution.locate_file(package_path)))
            )
            try:
                relative = source.relative_to(site_packages)
            except ValueError:
                # Console scripts are supplied separately from exact executable
                # inputs; never make the whole environment bin directory visible.
                continue
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(
                    f"producer site source must be a regular file: {source}"
                )
            identity = (
                stat.S_IMODE(source.stat().st_mode),
                source.stat().st_size,
                sha256_file(source),
            )
            relative_name = relative.as_posix()
            if relative_name in installed:
                if installed[relative_name] != identity:
                    raise RuntimeError(
                        f"producer distributions conflict at {relative_name}"
                    )
                continue
            installed[relative_name] = identity
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    result = tree_identity(destination)
    result["distributions"] = list(distribution_names)
    return result


def require_below_workspace(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise ValueError(f"output must be a child of the workspace: {resolved}")
    return resolved


def require_materialization_output(path: Path) -> Path:
    resolved = require_below_workspace(path)
    builds = (ROOT / "builds").resolve()
    if resolved.parent != builds or not resolved.name.startswith(
        "triton-deps-materialize-"
    ):
        raise ValueError(
            "materialization output must be a direct builds/"
            "triton-deps-materialize-* child"
        )
    return resolved


def require_seed_download_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("seed download directory must be an absolute canonical path")
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"seed download directory is absent: {path}") from error
    workspace = ROOT.resolve(strict=True)
    if path != lexical or resolved != lexical:
        raise ValueError("seed download directory must be an absolute canonical path")
    if resolved == workspace or workspace not in resolved.parents:
        raise ValueError("seed download directory must be below the workspace")
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("seed download directory must be a real directory")
    if metadata.st_uid != os.getuid():
        raise ValueError("seed download directory must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError("seed download directory must not be group/other-writable")
    return resolved


def manifest_path(output: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"manifest {field} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"manifest {field} is unsafe: {value!r}")
    resolved_output = output.resolve()
    lexical = Path(
        os.path.abspath(os.fspath(resolved_output / relative.as_posix()))
    )
    if lexical == resolved_output or resolved_output not in lexical.parents:
        raise RuntimeError(f"manifest {field} escapes output: {value!r}")
    resolved = lexical.resolve(strict=False)
    if resolved == resolved_output or resolved_output not in resolved.parents:
        raise RuntimeError(f"manifest {field} escapes output: {value!r}")
    return lexical


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate manifest key: {key!r}")
        result[key] = value
    return result


def require_regular_file(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")


def require_real_directory(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{description} must be a real directory: {path}")


def load_canonical_manifest(path: Path) -> dict[str, object]:
    require_regular_file(path, "canonical JSON")
    raw = path.read_text()
    manifest = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    canonical = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if raw != canonical:
        raise RuntimeError("dependency manifest is not canonical JSON")
    return manifest


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reviewed_archive_size(spec: PackageSpec, actual_bytes: int) -> None:
    if spec.expected_bytes is not None and actual_bytes != spec.expected_bytes:
        raise RuntimeError(
            f"archive differs from reviewed size for {spec.name}: "
            f"expected {spec.expected_bytes}, got {actual_bytes}"
        )


def validate_reviewed_archive_digest(spec: PackageSpec, actual_sha256: str) -> None:
    if spec.expected_sha256 and actual_sha256 != spec.expected_sha256:
        raise RuntimeError(
            f"archive differs from reviewed digest for {spec.name}: "
            f"expected {spec.expected_sha256}, got {actual_sha256}"
        )


def archive_filename(spec: PackageSpec) -> str:
    filename = spec.url.rsplit("/", 1)[-1]
    if (
        not filename
        or PurePosixPath(filename).name != filename
        or filename in {".", ".."}
    ):
        raise RuntimeError(f"dependency URL lacks a safe filename: {spec.name}")
    return filename


def _canonical_workspace_relative(value: object, description: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"{description} must be a canonical workspace-relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise RuntimeError(f"{description} must be a canonical workspace-relative path")
    return relative


def _exact_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_acquisition_provenance(
    spec: PackageSpec,
    acquisition: object,
    *,
    archive_bytes: int,
    archive_sha256: str,
) -> None:
    if not _exact_nonnegative_int(archive_bytes):
        raise RuntimeError(f"acquisition archive size is invalid: {spec.name}")
    if re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None:
        raise RuntimeError(f"acquisition archive digest is invalid: {spec.name}")
    if not isinstance(acquisition, dict):
        raise RuntimeError(f"acquisition provenance is malformed: {spec.name}")
    method = acquisition.get("method")
    if method == "network-download":
        expected_keys = {
            "declared_bytes",
            "etag",
            "effective_url",
            "method",
            "range_request_count",
            "request_count",
            "source_url",
        }
        if set(acquisition) != expected_keys:
            raise RuntimeError(f"acquisition keys mismatch: {spec.name}")
        requests = acquisition["request_count"]
        range_requests = acquisition["range_request_count"]
        if acquisition["source_url"] != spec.url:
            raise RuntimeError(f"acquisition URL mismatch: {spec.name}")
        effective_url = acquisition["effective_url"]
        if not isinstance(effective_url, str):
            raise RuntimeError(f"acquisition effective URL is invalid: {spec.name}")
        parsed_effective_url = urllib.parse.urlsplit(effective_url)
        if (
            parsed_effective_url.scheme != "https"
            or not parsed_effective_url.netloc
            or parsed_effective_url.fragment
        ):
            raise RuntimeError(f"acquisition effective URL is invalid: {spec.name}")
        if not is_strong_etag(acquisition["etag"]):
            raise RuntimeError(f"acquisition ETag is not strong: {spec.name}")
        if (
            not _exact_nonnegative_int(acquisition["declared_bytes"])
            or acquisition["declared_bytes"] != archive_bytes
        ):
            raise RuntimeError(f"acquisition size mismatch: {spec.name}")
        if (
            not _exact_nonnegative_int(requests)
            or requests < 1
            or requests > DOWNLOAD_MAX_REQUESTS
            or not _exact_nonnegative_int(range_requests)
            or range_requests > requests - 1
        ):
            raise RuntimeError(f"acquisition request counts are invalid: {spec.name}")
    elif method == "workspace-seed-copy":
        expected_keys = {
            "copied_bytes",
            "copied_sha256",
            "method",
            "source_bytes",
            "source_path",
            "source_sha256",
        }
        if set(acquisition) != expected_keys:
            raise RuntimeError(f"acquisition keys mismatch: {spec.name}")
        if spec.expected_sha256 is None:
            raise RuntimeError(f"unpinned dependency used a seed: {spec.name}")
        source_path = _canonical_workspace_relative(
            acquisition["source_path"],
            f"seed source for {spec.name}",
        )
        if source_path.name != archive_filename(spec):
            raise RuntimeError(f"seed source filename mismatch: {spec.name}")
        if any(
            not _exact_nonnegative_int(acquisition[key])
            or acquisition[key] != archive_bytes
            for key in ("source_bytes", "copied_bytes")
        ):
            raise RuntimeError(f"seed acquisition size mismatch: {spec.name}")
        if (
            acquisition["source_sha256"] != archive_sha256
            or acquisition["copied_sha256"] != archive_sha256
            or archive_sha256 != spec.expected_sha256
        ):
            raise RuntimeError(f"seed acquisition digest mismatch: {spec.name}")
    else:
        raise RuntimeError(f"unknown acquisition method: {spec.name}")


def dependencies_are_fully_pinned(specs: tuple[PackageSpec, ...]) -> bool:
    return all(spec.expected_sha256 is not None for spec in specs)


def validate_reviewed_manifest_anchor(expected_manifest_sha256: str) -> None:
    if REVIEWED_MANIFEST_SHA256 is None:
        raise RuntimeError(
            "reviewed manifest SHA-256 is not frozen in version-controlled source"
        )
    if expected_manifest_sha256 != REVIEWED_MANIFEST_SHA256:
        raise RuntimeError(
            "expected manifest SHA-256 differs from reviewed source anchor"
        )


def tree_identity(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files = 0
    symlinks = 0
    byte_count = 0
    directories = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        digest.update(mode.to_bytes(4, "little"))
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8")
            digest.update(b"L")
            digest.update(len(target).to_bytes(8, "little"))
            digest.update(target)
            symlinks += 1
            continue
        if path.is_dir():
            digest.update(b"D")
            directories += 1
            continue
        if not path.is_file():
            raise RuntimeError(f"unsupported tree entry: {path}")
        size = path.stat().st_size
        digest.update(b"F")
        digest.update(size.to_bytes(8, "little"))
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        files += 1
        byte_count += size
    return {
        "sha256": digest.hexdigest(),
        "files": files,
        "directories": directories,
        "symlinks": symlinks,
        "bytes": byte_count,
    }


def verify_reference_tree_unchanged(reference: Path, candidate: Path) -> None:
    require_real_directory(reference, "reference source tree")
    require_real_directory(candidate, "candidate source tree")
    for source in sorted(reference.rglob("*")):
        relative = source.relative_to(reference)
        target = candidate / relative
        source_mode = stat.S_IMODE(source.lstat().st_mode)
        if source.is_symlink():
            if (
                not target.is_symlink()
                or stat.S_IMODE(target.lstat().st_mode) != source_mode
                or os.readlink(target) != os.readlink(source)
            ):
                raise RuntimeError(f"source-tree symlink drift: {relative}")
        elif source.is_dir():
            if (
                target.is_symlink()
                or not target.is_dir()
                or stat.S_IMODE(target.lstat().st_mode) != source_mode
            ):
                raise RuntimeError(f"source-tree directory drift: {relative}")
        elif source.is_file():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(f"source-tree file is absent: {relative}")
            if leaf_identity(source) != leaf_identity(target):
                raise RuntimeError(f"source-tree file drift: {relative}")
        else:
            raise RuntimeError(f"unsupported reference source entry: {relative}")


def _safe_member_name(name: str) -> str:
    if not name or name.startswith("./") or "//" in name or "/./" in name:
        raise ValueError(f"unsafe archive member: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    encoded = path.as_posix().encode("utf-8")
    if len(encoded) > MAX_ARCHIVE_PATH_BYTES or any(
        len(part.encode("utf-8")) > MAX_ARCHIVE_COMPONENT_BYTES
        for part in path.parts
    ):
        raise ValueError(f"archive member path is too long: {name!r}")
    return path.as_posix()


def _safe_link_target(member_name: str, link_name: str, *, hardlink: bool) -> None:
    target = PurePosixPath(link_name)
    if target.is_absolute():
        raise ValueError(f"unsafe archive link target: {link_name!r}")
    combined = target if hardlink else PurePosixPath(member_name).parent / target
    stack: list[str] = []
    for part in combined.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not stack:
                raise ValueError(f"unsafe archive link target: {link_name!r}")
            stack.pop()
        else:
            stack.append(part)
    normalized = "/".join(stack)
    if len(normalized.encode("utf-8")) > MAX_ARCHIVE_PATH_BYTES or any(
        len(part.encode("utf-8")) > MAX_ARCHIVE_COMPONENT_BYTES
        for part in stack
    ):
        raise ValueError(f"archive link target is too long: {link_name!r}")


def _validate_extracted_symlinks(root: Path) -> None:
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = (path.parent / os.readlink(path)).resolve(strict=False)
        if target != resolved_root and resolved_root not in target.parents:
            raise ValueError(f"archive symlink escapes extraction root: {path}")


def _canonical_regular_mode(mode: int) -> int:
    return 0o755 if mode & 0o111 else 0o644


def normalize_materialized_tree_modes(
    root: Path,
    *,
    archived_executables: frozenset[str] = frozenset(),
) -> None:
    require_real_directory(root, "materialized tree")
    root.chmod(0o755)
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
            continue
        if not path.is_file():
            raise RuntimeError(f"unsupported materialized tree entry: {path}")
        relative = path.relative_to(root).as_posix()
        extracted_mode = stat.S_IMODE(path.lstat().st_mode)
        executable = relative in archived_executables or bool(
            extracted_mode & 0o111
        )
        path.chmod(0o755 if executable else 0o644)


def inspect_archive(
    archive: Path,
    archive_kind: str,
    *,
    max_expanded_bytes: int | None = None,
    max_members: int | None = None,
) -> dict[str, int]:
    members = 0
    regular_files = 0
    declared_bytes = 0
    if archive_kind == "zip":
        with zipfile.ZipFile(archive) as source:
            seen: set[str] = set()
            for info in source.infolist():
                canonical_name = _safe_member_name(info.filename.rstrip("/"))
                if canonical_name in seen:
                    raise ValueError(f"duplicate zip member: {info.filename}")
                seen.add(canonical_name)
                members += 1
                if max_members is not None and members > max_members:
                    raise ValueError("zip archive exceeds member-count limit")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"zip symlink is not allowed: {info.filename}")
                if not info.is_dir():
                    regular_files += 1
                    declared_bytes += info.file_size
            if max_expanded_bytes is not None and declared_bytes > max_expanded_bytes:
                raise ValueError("zip archive exceeds expanded-byte limit")
    elif archive_kind == "tar":
        with tarfile.open(archive, mode="r:*") as source:
            seen = set()
            for member in source:
                canonical_name = _safe_member_name(member.name.rstrip("/"))
                if canonical_name in seen:
                    raise ValueError(f"duplicate tar member: {member.name}")
                seen.add(canonical_name)
                members += 1
                if max_members is not None and members > max_members:
                    raise ValueError("tar archive exceeds member-count limit")
                if member.issym() or member.islnk():
                    _safe_link_target(
                        member.name,
                        member.linkname,
                        hardlink=member.islnk(),
                    )
                if member.isdev() or member.isfifo():
                    raise ValueError(f"special archive member is not allowed: {member.name}")
                if member.isfile():
                    regular_files += 1
                    declared_bytes += member.size
                elif not (member.isdir() or member.issym() or member.islnk()):
                    raise ValueError(
                        f"unsupported tar archive member: {member.name}"
                    )
            if max_expanded_bytes is not None and declared_bytes > max_expanded_bytes:
                raise ValueError("tar archive exceeds expanded-byte limit")
    else:
        raise ValueError(f"unknown archive kind: {archive_kind}")
    return {
        "members": members,
        "regular_files": regular_files,
        "declared_bytes": declared_bytes,
    }


def extract_archive(
    archive: Path,
    destination: Path,
    archive_kind: str,
    *,
    max_expanded_bytes: int | None = None,
    max_members: int | None = None,
) -> dict[str, int]:
    inventory = inspect_archive(
        archive,
        archive_kind,
        max_expanded_bytes=max_expanded_bytes,
        max_members=max_members,
    )
    destination.mkdir(parents=True)
    archived_executables: frozenset[str] = frozenset()
    if archive_kind == "zip":
        with zipfile.ZipFile(archive) as source:
            archived_executables = frozenset(
                _safe_member_name(info.filename.rstrip("/"))
                for info in source.infolist()
                if not info.is_dir() and (info.external_attr >> 16) & 0o111
            )
            source.extractall(destination)
    elif archive_kind == "tar":
        with tarfile.open(archive, mode="r:*") as source:
            source.extractall(destination, filter="data")
    else:  # guarded by inspect_archive
        raise AssertionError("unreachable archive kind")
    normalize_materialized_tree_modes(
        destination,
        archived_executables=archived_executables,
    )
    _validate_extracted_symlinks(destination)
    return inventory


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_independent_regular_file(
    opened: os.stat_result,
    named: os.stat_result,
    path: Path,
) -> None:
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_dev != named.st_dev
        or opened.st_ino != named.st_ino
        or opened.st_uid != os.getuid()
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) & 0o022
    ):
        raise RuntimeError(
            f"file is not an independent safely-permissioned user-owned "
            f"regular file: {path}"
        )


def _open_private_file(path: Path, flags: int) -> int:
    descriptor = os.open(
        path,
        flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _require_independent_regular_file(
            os.fstat(descriptor),
            path.lstat(),
            path,
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _create_private_partial(destination: Path) -> Path:
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"download destination already exists: {destination}")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(f"download partial already exists: {partial}")
    descriptor = os.open(
        partial,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        opened = os.fstat(descriptor)
        _require_independent_regular_file(opened, partial.lstat(), partial)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(partial.parent)
    return partial


def _rename_no_replace(
    source: Path,
    destination: Path,
    *,
    require_same_parent: bool = True,
) -> None:
    """Atomically rename a file or directory only when destination is absent."""

    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("no-replace publication paths must be absolute")
    source_lexical = Path(os.path.abspath(os.fspath(source)))
    destination_lexical = Path(os.path.abspath(os.fspath(destination)))
    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    workspace = ROOT.resolve(strict=True)
    if (
        source != source_lexical
        or destination != destination_lexical
        or source_resolved != source_lexical
        or destination_resolved != destination_lexical
        or (require_same_parent and source.parent != destination.parent)
        or (source.parent != workspace and workspace not in source.parent.parents)
        or (
            destination.parent != workspace
            and workspace not in destination.parent.parents
        )
        or source.is_symlink()
    ):
        raise ValueError(
            "no-replace publication requires canonical workspace paths "
            "under the same parent"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            "publication destination already exists",
            destination,
        )
    raise OSError(
        error_number,
        f"renameat2(RENAME_NOREPLACE) failed: {os.strerror(error_number)}",
        destination,
    )


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 8 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _verified_file_identity(path: Path) -> tuple[int, str, os.stat_result]:
    descriptor = _open_private_file(path, os.O_RDONLY)
    try:
        metadata = os.fstat(descriptor)
        digest = _sha256_descriptor(descriptor)
        final_metadata = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
            final_metadata.st_ctime_ns,
        ):
            raise RuntimeError(f"file changed while hashing: {path}")
        _require_independent_regular_file(final_metadata, path.lstat(), path)
        return final_metadata.st_size, digest, final_metadata
    finally:
        os.close(descriptor)


def _publish_verified_partial(
    partial: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str | None,
) -> str:
    actual_bytes, actual_sha256, partial_metadata = _verified_file_identity(partial)
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"archive size changed before publication: expected {expected_bytes}, "
            f"got {actual_bytes}"
        )
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "archive digest changed before publication: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"download destination already exists: {destination}")
    _rename_no_replace(partial, destination)
    published = destination.lstat()
    if (
        published.st_dev != partial_metadata.st_dev
        or published.st_ino != partial_metadata.st_ino
        or published.st_size != actual_bytes
        or published.st_nlink != 1
    ):
        raise RuntimeError(f"archive identity changed during publication: {destination}")
    _fsync_directory(destination.parent)
    return actual_sha256


def _decimal_header(response: object, name: str) -> int:
    headers = getattr(response, "headers", None)
    raw_value = None if headers is None else headers.get(name)
    if not isinstance(raw_value, str):
        raise DownloadContractError(f"response lacks required {name}")
    value = raw_value.strip()
    if re.fullmatch(r"[0-9]+", value) is None or str(int(value)) != value:
        raise DownloadContractError(f"response has malformed {name}: {raw_value!r}")
    return int(value)


def is_strong_etag(value: object) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and STRONG_ETAG_PATTERN.fullmatch(value) is not None
    )


def _strong_response_etag(response: object) -> str:
    headers = getattr(response, "headers", None)
    value = None if headers is None else headers.get("ETag")
    if not is_strong_etag(value):
        raise DownloadContractError("response lacks a valid strong ETag")
    return value


def _effective_response_url(response: object) -> str:
    geturl = getattr(response, "geturl", None)
    value = None if geturl is None else geturl()
    if not isinstance(value, str):
        raise DownloadContractError("response lacks an effective URL")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.fragment:
        raise DownloadContractError("response effective URL is not canonical HTTPS")
    return value


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = None if getcode is None else getcode()
    if not isinstance(status, int) or isinstance(status, bool):
        raise DownloadContractError("response lacks an exact HTTP status")
    return status


def _download_response_contract(
    response: object,
    *,
    offset: int,
    expected_total: int | None,
    expected_etag: str | None,
    expected_effective_url: str | None,
    locked_total: int | None,
    max_bytes: int,
) -> tuple[int, int, str, str]:
    headers = getattr(response, "headers", None)
    content_encoding = None if headers is None else headers.get("Content-Encoding")
    if content_encoding is not None and (
        not isinstance(content_encoding, str)
        or content_encoding.strip().lower() != "identity"
    ):
        raise DownloadContractError("download response uses content encoding")
    status = _response_status(response)
    content_length = _decimal_header(response, "Content-Length")
    if offset == 0:
        if status != 200:
            raise DownloadContractError(
                f"initial download requires HTTP 200, got {status}"
            )
        if headers is not None and headers.get("Content-Range") is not None:
            raise DownloadContractError("initial HTTP 200 response has Content-Range")
        total = content_length
        if total <= 0:
            raise DownloadContractError("download declares an empty archive")
        if expected_total is not None and total != expected_total:
            raise DownloadContractError("retried initial response changed its total")
        etag = _strong_response_etag(response)
        if expected_etag is not None and etag != expected_etag:
            raise DownloadContractError("retried initial response changed its ETag")
        effective_url = _effective_response_url(response)
        if (
            expected_effective_url is not None
            and effective_url != expected_effective_url
        ):
            raise DownloadContractError(
                "retried initial response changed its effective URL"
            )
    else:
        if status != 206:
            raise DownloadContractError(
                f"server ignored Range request at offset {offset}: HTTP {status}"
            )
        raw_content_range = None if headers is None else headers.get("Content-Range")
        if not isinstance(raw_content_range, str):
            raise DownloadContractError("range response lacks Content-Range")
        match = CONTENT_RANGE_PATTERN.fullmatch(raw_content_range.strip())
        if match is None:
            raise DownloadContractError(
                f"range response has malformed Content-Range: {raw_content_range!r}"
            )
        start, end, total = (int(value) for value in match.groups())
        if expected_total is None:
            raise DownloadContractError("range response lacks an established total")
        if start != offset or total != expected_total or end != total - 1:
            raise DownloadContractError(
                "range response does not exactly match the requested offset/total"
            )
        if content_length != end - start + 1:
            raise DownloadContractError(
                "range Content-Length disagrees with Content-Range"
            )
        etag = _strong_response_etag(response)
        if expected_etag is None or etag != expected_etag:
            raise DownloadContractError("range response ETag changed or is missing")
        effective_url = _effective_response_url(response)
        if expected_effective_url is None or effective_url != expected_effective_url:
            raise DownloadContractError("range response effective URL changed")
    if total > max_bytes:
        raise ValueError("download exceeds archive-byte limit")
    if locked_total is not None and total != locked_total:
        raise DownloadContractError(
            f"download size differs from source lock: expected {locked_total}, got {total}"
        )
    return total, content_length, etag, effective_url


def _append_download_response(
    response: object,
    partial: Path,
    *,
    declared_bytes: int,
) -> None:
    descriptor = _open_private_file(partial, os.O_WRONLY | os.O_APPEND)
    received = 0
    bytes_since_fsync = 0
    with os.fdopen(descriptor, "ab") as output:
        try:
            while received < declared_bytes:
                try:
                    chunk = response.read(
                        min(DOWNLOAD_CHUNK_BYTES, declared_bytes - received)
                    )
                except (OSError, EOFError, http.client.HTTPException) as error:
                    raise RetryableDownloadError(
                        f"download read failed after {received} of "
                        f"{declared_bytes} declared bytes"
                    ) from error
                if not isinstance(chunk, bytes):
                    raise DownloadContractError("download response returned non-byte data")
                if not chunk:
                    raise RetryableDownloadError(
                        f"premature EOF after {received} of "
                        f"{declared_bytes} declared bytes"
                    )
                if len(chunk) > declared_bytes - received:
                    raise DownloadContractError("download exceeded declared Content-Length")
                written = output.write(chunk)
                if written != len(chunk):
                    raise OSError("short write while storing dependency archive")
                received += written
                bytes_since_fsync += written
                if bytes_since_fsync >= DOWNLOAD_FSYNC_INTERVAL_BYTES:
                    output.flush()
                    os.fsync(output.fileno())
                    bytes_since_fsync = 0
        finally:
            output.flush()
            os.fsync(output.fileno())


def download(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, object]:
    partial = _create_private_partial(destination)
    expected_total: int | None = None
    expected_etag: str | None = None
    expected_effective_url: str | None = None
    request_count = 0
    range_request_count = 0
    last_retryable: BaseException | None = None
    while request_count < DOWNLOAD_MAX_REQUESTS:
        offset = partial.stat().st_size
        if expected_total is not None and offset == expected_total:
            break
        if expected_total is not None and offset > expected_total:
            raise DownloadContractError("download partial exceeds declared total")
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": DOWNLOAD_USER_AGENT,
        }
        if offset:
            if expected_total is None:
                raise DownloadContractError("download partial lacks an established total")
            if expected_etag is None:
                raise DownloadContractError("download partial lacks a strong ETag")
            headers["Range"] = f"bytes={offset}-"
            headers["If-Range"] = expected_etag
            range_request_count += 1
        request = urllib.request.Request(url, headers=headers)
        request_count += 1
        try:
            response = urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
        except (OSError, EOFError, http.client.HTTPException) as error:
            last_retryable = error
            continue
        try:
            with response:
                total, declared_bytes, etag, effective_url = (
                    _download_response_contract(
                        response,
                        offset=offset,
                        expected_total=expected_total,
                        expected_etag=expected_etag,
                        expected_effective_url=expected_effective_url,
                        locked_total=expected_bytes,
                        max_bytes=max_bytes,
                    )
                )
                expected_total = total
                expected_etag = etag
                expected_effective_url = effective_url
                _append_download_response(
                    response,
                    partial,
                    declared_bytes=declared_bytes,
                )
        except RetryableDownloadError as error:
            last_retryable = error
            continue
    if (
        expected_total is None
        or expected_etag is None
        or expected_effective_url is None
        or partial.stat().st_size != expected_total
    ):
        message = (
            f"download incomplete after {request_count} requests: "
            f"received {partial.stat().st_size} of "
            f"{expected_total if expected_total is not None else 'unknown'} bytes"
        )
        raise RuntimeError(message) from last_retryable
    _publish_verified_partial(
        partial,
        destination,
        expected_bytes=expected_total,
        expected_sha256=expected_sha256,
    )
    return {
        "method": "network-download",
        "source_url": url,
        "effective_url": expected_effective_url,
        "etag": expected_etag,
        "declared_bytes": expected_total,
        "request_count": request_count,
        "range_request_count": range_request_count,
    }


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_uid,
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_uid,
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def copy_seed_archive(
    spec: PackageSpec,
    seed_download_dir: Path,
    destination: Path,
    *,
    max_bytes: int,
) -> dict[str, object] | None:
    seed_download_dir = require_seed_download_directory(seed_download_dir)
    filename = archive_filename(spec)
    directory_descriptor = os.open(
        seed_download_dir,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    source_descriptor: int | None = None
    try:
        opened_directory = os.fstat(directory_descriptor)
        named_directory = seed_download_dir.lstat()
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or opened_directory.st_dev != named_directory.st_dev
            or opened_directory.st_ino != named_directory.st_ino
            or opened_directory.st_uid != os.getuid()
            or stat.S_IMODE(opened_directory.st_mode) & 0o022
        ):
            raise RuntimeError("seed download directory identity changed")
        try:
            named_source = os.stat(
                filename,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        source_path = seed_download_dir / filename
        _require_independent_regular_file(
            named_source,
            named_source,
            source_path,
        )
        if spec.expected_sha256 is None:
            raise RuntimeError(
                f"seed archive is forbidden for unpinned dependency: {spec.name}"
            )
        source_descriptor = os.open(
            filename,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        source_before = os.fstat(source_descriptor)
        _require_independent_regular_file(
            source_before,
            named_source,
            source_path,
        )
        if source_before.st_size > max_bytes:
            raise ValueError("seed archive exceeds archive-byte limit")
        validate_reviewed_archive_size(spec, source_before.st_size)
        source_sha256 = _sha256_descriptor(source_descriptor)
        source_after_hash = os.fstat(source_descriptor)
        if not _same_file_state(source_before, source_after_hash):
            raise RuntimeError(f"seed archive changed while hashing: {source_path}")
        validate_reviewed_archive_digest(spec, source_sha256)

        partial = _create_private_partial(destination)
        output_descriptor = _open_private_file(partial, os.O_WRONLY)
        copied_bytes = 0
        copied_digest = hashlib.sha256()
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        with os.fdopen(output_descriptor, "wb") as output:
            while chunk := os.read(source_descriptor, 8 * 1024 * 1024):
                copied_bytes += len(chunk)
                if copied_bytes > source_before.st_size:
                    raise RuntimeError("seed copy exceeds its pre-copy size")
                written = output.write(chunk)
                if written != len(chunk):
                    raise OSError("short write while copying seed archive")
                copied_digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        source_after_copy = os.fstat(source_descriptor)
        named_source_after = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not _same_file_state(source_before, source_after_copy)
            or not _same_file_state(source_before, named_source_after)
        ):
            raise RuntimeError(f"seed archive changed while copying: {source_path}")
        if copied_bytes != source_before.st_size:
            raise RuntimeError("seed copy size differs from its pre-copy size")
        if copied_digest.hexdigest() != source_sha256:
            raise RuntimeError("seed copy digest differs from its pre-copy digest")
        copied_size, copied_sha256, _ = _verified_file_identity(partial)
        if copied_size != source_before.st_size or copied_sha256 != source_sha256:
            raise RuntimeError("seed copy failed post-copy size/digest verification")
        _publish_verified_partial(
            partial,
            destination,
            expected_bytes=source_before.st_size,
            expected_sha256=source_sha256,
        )
        return {
            "method": "workspace-seed-copy",
            "source_path": source_path.relative_to(ROOT.resolve(strict=True)).as_posix(),
            "source_bytes": source_before.st_size,
            "source_sha256": source_sha256,
            "copied_bytes": copied_size,
            "copied_sha256": copied_sha256,
        }
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        os.close(directory_descriptor)


def acquire_archive(
    spec: PackageSpec,
    destination: Path,
    *,
    max_bytes: int,
    seed_download_dir: Path | None,
) -> dict[str, object]:
    if seed_download_dir is not None:
        seeded = copy_seed_archive(
            spec,
            seed_download_dir,
            destination,
            max_bytes=max_bytes,
        )
        if seeded is not None:
            return seeded
    return download(
        spec.url,
        destination,
        max_bytes=max_bytes,
        expected_sha256=spec.expected_sha256,
        expected_bytes=spec.expected_bytes,
    )


def _single_root(expanded: Path, expected_name: str | None) -> Path:
    if expected_name is None:
        return expanded
    expected = expanded / expected_name
    if not expected.is_dir():
        raise RuntimeError(f"archive lacks expected root {expected_name!r}")
    return expected


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"required regular file is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _merge_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"required directory is absent: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)


def assemble_overlay(expanded_roots: dict[str, Path], destination: Path) -> None:
    destination.mkdir()
    _copy_file(
        expanded_roots["ptxas"] / "bin/ptxas",
        destination / "bin/ptxas",
    )
    _copy_file(
        expanded_roots["ptxas_blackwell"] / "bin/ptxas",
        destination / "bin/ptxas-blackwell",
    )
    _copy_file(
        expanded_roots["cuobjdump"] / "bin/cuobjdump",
        destination / "bin/cuobjdump",
    )
    _copy_file(
        expanded_roots["nvdisasm"] / "bin/nvdisasm",
        destination / "bin/nvdisasm",
    )
    for name in ("cudacrt", "cudart", "cupti"):
        _merge_tree(expanded_roots[name] / "include", destination / "include")
    _merge_tree(expanded_roots["cupti"] / "lib", destination / "lib/cupti")
    normalize_materialized_tree_modes(destination)
    _validate_extracted_symlinks(destination)


def leaf_identity(path: Path) -> tuple[object, ...]:
    mode = stat.S_IMODE(path.lstat().st_mode)
    if path.is_symlink():
        return ("symlink", mode, os.readlink(path))
    if not path.is_file():
        raise RuntimeError(f"expected an overlay file or symlink: {path}")
    return ("file", mode, path.stat().st_size, sha256_file(path))


def normalized_overlay_leaf_identity(path: Path) -> tuple[object, ...]:
    identity = leaf_identity(path)
    if identity[0] != "file":
        return identity
    return (identity[0], _canonical_regular_mode(identity[1]), *identity[2:])


def _merge_expected_leaves(
    expected: dict[str, tuple[object, ...]], source: Path, prefix: str
) -> None:
    for path in sorted(source.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative = path.relative_to(source).as_posix()
        expected[f"{prefix}/{relative}"] = normalized_overlay_leaf_identity(path)


def expected_overlay_leaves(expanded_roots: dict[str, Path]) -> dict[str, tuple[object, ...]]:
    expected = {
        "bin/ptxas": normalized_overlay_leaf_identity(
            expanded_roots["ptxas"] / "bin/ptxas"
        ),
        "bin/ptxas-blackwell": normalized_overlay_leaf_identity(
            expanded_roots["ptxas_blackwell"] / "bin/ptxas"
        ),
        "bin/cuobjdump": normalized_overlay_leaf_identity(
            expanded_roots["cuobjdump"] / "bin/cuobjdump"
        ),
        "bin/nvdisasm": normalized_overlay_leaf_identity(
            expanded_roots["nvdisasm"] / "bin/nvdisasm"
        ),
    }
    for name in ("cudacrt", "cudart", "cupti"):
        _merge_expected_leaves(expected, expanded_roots[name] / "include", "include")
    _merge_expected_leaves(expected, expanded_roots["cupti"] / "lib", "lib/cupti")
    return expected


def verify_overlay_derivation(
    expanded_roots: dict[str, Path], overlay: Path
) -> None:
    expected = expected_overlay_leaves(expanded_roots)
    observed = {
        path.relative_to(overlay).as_posix(): leaf_identity(path)
        for path in sorted(overlay.rglob("*"))
        if path.is_file() or path.is_symlink()
    }
    if observed != expected:
        raise RuntimeError("NVIDIA backend overlay is not derived from package roots")


def validate_build_input_layout(llvm: Path, json_root: Path, overlay: Path) -> None:
    required_llvm = (
        llvm / "bin/FileCheck",
        llvm / "lib/cmake/llvm/LLVMConfig.cmake",
        llvm / "lib/cmake/mlir/MLIRConfig.cmake",
        llvm / "lib/cmake/lld/LLDConfig.cmake",
    )
    for path in required_llvm:
        if not path.is_file():
            raise RuntimeError(f"Triton LLVM input is incomplete: {path}")
    if not os.access(llvm / "bin/FileCheck", os.X_OK):
        raise RuntimeError("Triton LLVM FileCheck is not executable")
    if not (json_root / "include/nlohmann/json.hpp").is_file():
        raise RuntimeError("Triton JSON input is incomplete")
    for name in OVERLAY_TOOL_VERSIONS:
        path = overlay / "bin" / name
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"Triton NVIDIA tool is absent/not executable: {path}")
        with path.open("rb") as source:
            magic = source.read(4)
        if magic != b"\x7fELF":
            raise RuntimeError(f"Triton NVIDIA tool is not ELF: {path}")
    for path in (overlay / "include/cuda.h", overlay / "include/cupti.h"):
        if not path.is_file():
            raise RuntimeError(f"Triton NVIDIA headers are incomplete: {path}")
    if not any((overlay / "lib/cupti").rglob("libcupti.so*")):
        raise RuntimeError("Triton CUPTI library input is incomplete")


def validate_overlay_tool_versions_sandboxed(overlay: Path) -> None:
    def probe_limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))

    for name, version in OVERLAY_TOOL_VERSIONS.items():
        path = overlay / "bin" / name
        with tempfile.TemporaryFile(dir=ROOT / "runs") as output:
            result = subprocess.run(
                [
                "/usr/bin/bwrap",
                "--die-with-parent",
                "--new-session",
                "--unshare-net",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
                "--unshare-cgroup",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "--ro-bind",
                "/etc/ld.so.cache",
                "/etc/ld.so.cache",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--ro-bind",
                str(overlay),
                "/input",
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin",
                f"/input/bin/{name}",
                "--version",
                ],
                check=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=30,
                preexec_fn=probe_limits,
            )
            output.seek(0)
            observed_output = output.read().decode("utf-8", errors="replace")
        if result.returncode != 0 or version not in observed_output:
            raise RuntimeError(f"Triton NVIDIA tool version mismatch: {name}")


def _materialize_into(
    output: Path,
    *,
    seed_download_dir: Path | None = None,
) -> dict[str, object]:
    output.mkdir()
    downloads = output / "downloads"
    expanded = output / "expanded"
    downloads.mkdir()
    expanded.mkdir()

    required_free_bytes = MATERIALIZATION_HEADROOM_BYTES + sum(
        archive_bytes + (2 * expanded_bytes)
        for archive_bytes, expanded_bytes, _ in PACKAGE_RESOURCE_LIMITS.values()
    )
    if shutil.disk_usage(output.parent).free < required_free_bytes:
        raise RuntimeError(
            f"insufficient free space for bounded materialization: "
            f"need at least {required_free_bytes} bytes"
        )

    package_records: list[dict[str, object]] = []
    expanded_roots: dict[str, Path] = {}
    for spec in PACKAGE_SPECS:
        filename = archive_filename(spec)
        archive = downloads / filename
        max_archive_bytes, max_expanded_bytes, max_members = PACKAGE_RESOURCE_LIMITS[
            spec.name
        ]
        acquisition = acquire_archive(
            spec,
            archive,
            max_bytes=max_archive_bytes,
            seed_download_dir=seed_download_dir,
        )
        archive_sha256 = sha256_file(archive)
        archive_bytes = archive.stat().st_size
        validate_reviewed_archive_size(spec, archive_bytes)
        validate_reviewed_archive_digest(spec, archive_sha256)
        validate_acquisition_provenance(
            spec,
            acquisition,
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha256,
        )
        destination = expanded / spec.name
        archive_inventory = extract_archive(
            archive,
            destination,
            spec.archive_kind,
            max_expanded_bytes=max_expanded_bytes,
            max_members=max_members,
        )
        package_root = _single_root(destination, spec.expected_root)
        _validate_extracted_symlinks(package_root)
        expanded_identity = tree_identity(package_root)
        if expanded_identity["bytes"] > max_expanded_bytes:
            raise RuntimeError(f"expanded tree exceeds byte limit: {spec.name}")
        expanded_entries = sum(
            expanded_identity[key]
            for key in ("files", "directories", "symlinks")
        )
        if expanded_entries > max_members:
            raise RuntimeError(f"expanded tree exceeds member limit: {spec.name}")
        expanded_roots[spec.name] = package_root
        package_records.append(
            {
                "name": spec.name,
                "url": spec.url,
                "acquisition": acquisition,
                "archive": archive.relative_to(output).as_posix(),
                "archive_sha256": archive_sha256,
                "archive_bytes": archive_bytes,
                "archive_inventory": archive_inventory,
                "expanded_root": package_root.relative_to(output).as_posix(),
                "expanded_tree": expanded_identity,
            }
        )

    json_root = expanded_roots["json"]
    if not (json_root / "include/nlohmann/json.hpp").is_file():
        raise RuntimeError("JSON archive lacks include/nlohmann/json.hpp")

    overlay = output / "nvidia-backend-overlay"
    assemble_overlay(expanded_roots, overlay)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized-unreviewed",
        "triton_commit": TRITON_COMMIT,
        "triton_tree": TRITON_TREE,
        "triton_llvm_commit": TRITON_LLVM_COMMIT,
        "packages": package_records,
        "build_inputs": {
            "llvm_syspath": expanded_roots["llvm"].relative_to(output).as_posix(),
            "json_syspath": json_root.relative_to(output).as_posix(),
            "nvidia_backend_overlay": overlay.relative_to(output).as_posix(),
            "nvidia_backend_overlay_tree": tree_identity(overlay),
            "pybind11_wheel": next(
                record["archive"]
                for record in package_records
                if record["name"] == "pybind11"
            ),
        },
    }
    encoded = canonical_json(manifest)
    temporary = output / f"manifest.{os.getpid()}.tmp"
    with temporary.open("xb") as destination:
        destination.write(encoded.encode("ascii"))
        destination.flush()
        os.fsync(destination.fileno())
    _rename_no_replace(temporary, output / "manifest.json")
    _fsync_directory(output)
    return manifest


def materialize(
    output: Path,
    *,
    seed_download_dir: Path | None = None,
) -> dict[str, object]:
    output = require_materialization_output(output)
    if seed_download_dir is not None:
        seed_download_dir = require_seed_download_directory(seed_download_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    staging = output.parent / f".{output.name}.{os.getpid()}.partial"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"staging output already exists: {staging}")
    manifest = _materialize_into(
        staging,
        seed_download_dir=seed_download_dir,
    )
    if verify(staging) != manifest:
        raise RuntimeError("staged dependency manifest verification changed its value")
    _fsync_directory(staging)
    _rename_no_replace(staging, output)
    _fsync_directory(output.parent)
    return manifest


def verify(
    output: Path,
    *,
    expected_manifest_sha256: str | None = None,
    require_reviewed: bool = False,
) -> dict[str, object]:
    output = require_below_workspace(output)
    if require_reviewed and expected_manifest_sha256 is None:
        raise ValueError(
            "reviewed verification requires an expected manifest SHA-256"
        )
    if require_reviewed:
        validate_reviewed_manifest_anchor(expected_manifest_sha256)
    manifest_path_value = output / "manifest.json"
    require_regular_file(manifest_path_value, "dependency manifest")
    if expected_manifest_sha256 is not None:
        if len(expected_manifest_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in expected_manifest_sha256
        ):
            raise ValueError("expected manifest SHA-256 must be 64 lowercase hex")
        if sha256_file(manifest_path_value) != expected_manifest_sha256:
            raise RuntimeError("dependency manifest digest mismatch")
    manifest = load_canonical_manifest(manifest_path_value)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected dependency manifest schema")
    if manifest.get("status") != "materialized-unreviewed":
        raise RuntimeError("unexpected dependency manifest status")
    if manifest.get("triton_commit") != TRITON_COMMIT:
        raise RuntimeError("dependency manifest Triton commit mismatch")
    if manifest.get("triton_tree") != TRITON_TREE:
        raise RuntimeError("dependency manifest Triton tree mismatch")
    if manifest.get("triton_llvm_commit") != TRITON_LLVM_COMMIT:
        raise RuntimeError("dependency manifest Triton LLVM mismatch")
    records = manifest.get("packages")
    if not isinstance(records, list) or len(records) != len(PACKAGE_SPECS):
        raise RuntimeError("dependency manifest package set mismatch")
    if require_reviewed and not dependencies_are_fully_pinned(PACKAGE_SPECS):
        raise RuntimeError("not every dependency archive has a reviewed SHA-256")
    by_name: dict[str, dict[str, object]] = {}
    expanded_roots: dict[str, Path] = {}
    record_keys = {
        "acquisition",
        "name",
        "url",
        "archive",
        "archive_sha256",
        "archive_bytes",
        "archive_inventory",
        "expanded_root",
        "expanded_tree",
    }
    for spec, record in zip(PACKAGE_SPECS, records, strict=True):
        if not isinstance(record, dict) or record.get("name") != spec.name:
            raise RuntimeError("dependency manifest package order/name mismatch")
        if set(record) != record_keys:
            raise RuntimeError(f"dependency record keys mismatch: {spec.name}")
        if record.get("url") != spec.url:
            raise RuntimeError(f"dependency URL mismatch: {spec.name}")
        archive = manifest_path(output, record.get("archive"), f"{spec.name}.archive")
        expanded_root = manifest_path(
            output,
            record.get("expanded_root"),
            f"{spec.name}.expanded_root",
        )
        expected_archive = (output / "downloads" / archive_filename(spec)).resolve()
        expected_expanded = (output / "expanded" / spec.name).resolve()
        if spec.expected_root is not None:
            expected_expanded /= spec.expected_root
        if archive != expected_archive:
            raise RuntimeError(f"archive path mismatch: {spec.name}")
        if expanded_root != expected_expanded:
            raise RuntimeError(f"expanded path mismatch: {spec.name}")
        require_regular_file(archive, f"dependency archive {spec.name}")
        require_real_directory(expanded_root, f"expanded package {spec.name}")
        actual_archive_sha256 = sha256_file(archive)
        if actual_archive_sha256 != record["archive_sha256"]:
            raise RuntimeError(f"archive digest mismatch: {spec.name}")
        validate_reviewed_archive_digest(spec, record["archive_sha256"])
        if archive.stat().st_size != record["archive_bytes"]:
            raise RuntimeError(f"archive size mismatch: {spec.name}")
        validate_reviewed_archive_size(spec, record["archive_bytes"])
        validate_acquisition_provenance(
            spec,
            record["acquisition"],
            archive_bytes=record["archive_bytes"],
            archive_sha256=actual_archive_sha256,
        )
        max_archive_bytes, max_expanded_bytes, max_members = (
            PACKAGE_RESOURCE_LIMITS[spec.name]
        )
        if archive.stat().st_size > max_archive_bytes:
            raise RuntimeError(f"archive exceeds byte limit: {spec.name}")
        if inspect_archive(
            archive,
            spec.archive_kind,
            max_expanded_bytes=max_expanded_bytes,
            max_members=max_members,
        ) != record["archive_inventory"]:
            raise RuntimeError(f"archive inventory mismatch: {spec.name}")
        expanded_identity = tree_identity(expanded_root)
        if expanded_identity != record["expanded_tree"]:
            raise RuntimeError(f"expanded tree mismatch: {spec.name}")
        if expanded_identity["bytes"] > max_expanded_bytes:
            raise RuntimeError(f"expanded tree exceeds byte limit: {spec.name}")
        if sum(
            expanded_identity[key]
            for key in ("files", "directories", "symlinks")
        ) > max_members:
            raise RuntimeError(f"expanded tree exceeds member limit: {spec.name}")
        _validate_extracted_symlinks(expanded_root)
        by_name[spec.name] = record
        expanded_roots[spec.name] = expanded_root
    build_inputs = manifest.get("build_inputs")
    if not isinstance(build_inputs, dict):
        raise RuntimeError("dependency manifest build_inputs is malformed")
    if set(build_inputs) != {
        "llvm_syspath",
        "json_syspath",
        "nvidia_backend_overlay",
        "nvidia_backend_overlay_tree",
        "pybind11_wheel",
    }:
        raise RuntimeError("dependency manifest build_inputs keys mismatch")
    llvm = manifest_path(output, build_inputs.get("llvm_syspath"), "llvm_syspath")
    json_root = manifest_path(
        output, build_inputs.get("json_syspath"), "json_syspath"
    )
    overlay = manifest_path(
        output,
        build_inputs.get("nvidia_backend_overlay"),
        "nvidia_backend_overlay",
    )
    if overlay != (output / "nvidia-backend-overlay").resolve():
        raise RuntimeError("NVIDIA backend overlay must use the fixed output path")
    pybind11_wheel = manifest_path(
        output,
        build_inputs.get("pybind11_wheel"),
        "pybind11_wheel",
    )
    require_real_directory(llvm, "LLVM build input")
    require_real_directory(json_root, "JSON build input")
    require_real_directory(overlay, "NVIDIA backend overlay")
    require_regular_file(pybind11_wheel, "pybind11 build-input wheel")
    if llvm != manifest_path(
        output, by_name["llvm"]["expanded_root"], "llvm.expanded_root"
    ):
        raise RuntimeError("LLVM build input does not match package record")
    if json_root != manifest_path(
        output, by_name["json"]["expanded_root"], "json.expanded_root"
    ):
        raise RuntimeError("JSON build input does not match package record")
    if pybind11_wheel != manifest_path(
        output, by_name["pybind11"]["archive"], "pybind11.archive"
    ):
        raise RuntimeError("pybind11 build input does not match package record")
    if tree_identity(overlay) != build_inputs["nvidia_backend_overlay_tree"]:
        raise RuntimeError("NVIDIA backend overlay tree mismatch")
    _validate_extracted_symlinks(overlay)
    verify_overlay_derivation(expanded_roots, overlay)
    validate_build_input_layout(llvm, json_root, overlay)
    if require_reviewed:
        review = load_canonical_manifest(output / "review.json")
        manifest_sha256 = sha256_file(manifest_path_value)
        expected_archives = [
            {"name": spec.name, "sha256": spec.expected_sha256}
            for spec in PACKAGE_SPECS
        ]
        expected_review = {
            "schema_version": SCHEMA_VERSION,
            "status": "reviewed",
            "manifest_sha256": manifest_sha256,
            "triton_commit": TRITON_COMMIT,
            "triton_tree": TRITON_TREE,
            "triton_llvm_commit": TRITON_LLVM_COMMIT,
            "archives": expected_archives,
        }
        if review != expected_review:
            raise RuntimeError("dependency review record mismatch")
        cache_parent = (ROOT / "caches/triton-build-deps").resolve()
        if output.parent == cache_parent and output.name != manifest_sha256:
            raise RuntimeError("reviewed dependency cache name must equal manifest SHA-256")
    return manifest


def probe_reviewed_tools(output: Path, expected_manifest_sha256: str) -> None:
    manifest = verify(
        output,
        expected_manifest_sha256=expected_manifest_sha256,
        require_reviewed=True,
    )
    overlay = manifest_path(
        output.resolve(),
        manifest["build_inputs"]["nvidia_backend_overlay"],
        "nvidia_backend_overlay",
    )
    validate_overlay_tool_versions_sandboxed(overlay)


def promote_reviewed(output: Path, expected_manifest_sha256: str) -> dict[str, object]:
    output = require_materialization_output(output)
    validate_reviewed_manifest_anchor(expected_manifest_sha256)
    manifest = verify(
        output,
        expected_manifest_sha256=expected_manifest_sha256,
        require_reviewed=False,
    )
    if not dependencies_are_fully_pinned(PACKAGE_SPECS):
        raise RuntimeError("not every dependency archive has a reviewed SHA-256")
    review = {
        "schema_version": SCHEMA_VERSION,
        "status": "reviewed",
        "manifest_sha256": expected_manifest_sha256,
        "triton_commit": TRITON_COMMIT,
        "triton_tree": TRITON_TREE,
        "triton_llvm_commit": TRITON_LLVM_COMMIT,
        "archives": [
            {"name": spec.name, "sha256": spec.expected_sha256}
            for spec in PACKAGE_SPECS
        ],
    }
    review_path = output / "review.json"
    if review_path.exists() or review_path.is_symlink():
        raise FileExistsError(f"review record already exists: {review_path}")
    temporary = output / f"review.{os.getpid()}.tmp"
    with temporary.open("xb") as destination:
        destination.write(canonical_json(review).encode("ascii"))
        destination.flush()
        os.fsync(destination.fileno())
    _rename_no_replace(temporary, review_path)
    _fsync_directory(output)
    return manifest


def _require_canonical_owned_cache_directory(
    path: Path,
    description: str,
) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"{description} must be absolute")
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{description} is absent") from error
    workspace = ROOT.resolve(strict=True)
    if (
        path != lexical
        or resolved != lexical
        or resolved == workspace
        or workspace not in resolved.parents
        or path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise RuntimeError(f"{description} is not a safe canonical directory")
    return resolved


def prepare_reviewed_cache_parent() -> Path:
    caches_root = _require_canonical_owned_cache_directory(
        ROOT / "caches",
        "workspace cache root",
    )
    cache_parent_path = caches_root / "triton-build-deps"
    try:
        os.mkdir(cache_parent_path, mode=0o700)
    except FileExistsError:
        pass
    cache_parent = _require_canonical_owned_cache_directory(
        cache_parent_path,
        "reviewed dependency cache parent",
    )
    _fsync_directory(cache_parent)
    _fsync_directory(caches_root)
    return cache_parent


def publish_reviewed_cache(
    output: Path,
    expected_manifest_sha256: str,
) -> dict[str, object]:
    output = require_materialization_output(output)
    manifest = verify(
        output,
        expected_manifest_sha256=expected_manifest_sha256,
        require_reviewed=True,
    )
    cache_parent = prepare_reviewed_cache_parent()
    destination = cache_parent / expected_manifest_sha256
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"reviewed dependency cache already exists: {destination}")
    _fsync_directory(output)
    _rename_no_replace(
        output,
        destination,
        require_same_parent=False,
    )
    _fsync_directory(cache_parent)
    _fsync_directory(output.parent)
    if verify(
        destination,
        expected_manifest_sha256=expected_manifest_sha256,
        require_reviewed=True,
    ) != manifest:
        raise RuntimeError("published dependency cache verification changed its value")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-download-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--promote-reviewed", action="store_true")
    parser.add_argument("--publish-reviewed-cache", action="store_true")
    parser.add_argument("--probe-reviewed-tools", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    validate_versions_lock()
    validate_source_pins()
    validate_live_producers()
    modes = sum(
        bool(value)
        for value in (
            args.verify,
            args.promote_reviewed,
            args.publish_reviewed_cache,
            args.probe_reviewed_tools,
        )
    )
    if modes > 1:
        parser.error(
            "verification/promotion/cache-publication/probe modes are "
            "mutually exclusive"
        )
    if modes != 0 and args.seed_download_dir is not None:
        parser.error("--seed-download-dir is valid only during materialization")
    if modes == 0 and (
        args.expected_manifest_sha256 is not None or args.require_reviewed
    ):
        parser.error("review options require --verify")
    if args.promote_reviewed and args.expected_manifest_sha256 is None:
        parser.error("--promote-reviewed requires --expected-manifest-sha256")
    if args.promote_reviewed and args.require_reviewed:
        parser.error("--require-reviewed is invalid during promotion")
    if args.publish_reviewed_cache and args.expected_manifest_sha256 is None:
        parser.error(
            "--publish-reviewed-cache requires --expected-manifest-sha256"
        )
    if args.publish_reviewed_cache and args.require_reviewed:
        parser.error("cache publication already requires reviewed verification")
    if args.probe_reviewed_tools and args.require_reviewed:
        parser.error("tool probe implies reviewed verification")
    if args.probe_reviewed_tools and args.expected_manifest_sha256 is None:
        parser.error("tool probe requires --expected-manifest-sha256")
    if args.verify and args.require_reviewed and args.expected_manifest_sha256 is None:
        parser.error("--require-reviewed requires --expected-manifest-sha256")
    if modes == 0:
        preflight_command = [
            sys.executable,
            str(ROOT / "tools/preflight.py"),
            "--mode",
            "heavy",
        ]
        if os.environ.get(
            "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED"
        ) == "1":
            preflight_command.append(
                "--allow-protected-cpu-only-coexistence"
            )
        preflight = subprocess.run(
            preflight_command,
            cwd=ROOT,
            check=False,
        )
        if preflight.returncode != 0:
            return preflight.returncode
    if args.publish_reviewed_cache:
        manifest = publish_reviewed_cache(
            args.output,
            args.expected_manifest_sha256,
        )
    elif args.probe_reviewed_tools:
        probe_reviewed_tools(args.output, args.expected_manifest_sha256)
        manifest = verify(
            args.output,
            expected_manifest_sha256=args.expected_manifest_sha256,
            require_reviewed=True,
        )
    else:
        manifest = (
            verify(
                args.output,
                expected_manifest_sha256=args.expected_manifest_sha256,
                require_reviewed=args.require_reviewed,
            )
            if args.verify
            else (
                promote_reviewed(args.output, args.expected_manifest_sha256)
                if args.promote_reviewed
                else materialize(
                    args.output,
                    seed_download_dir=args.seed_download_dir,
                )
            )
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
