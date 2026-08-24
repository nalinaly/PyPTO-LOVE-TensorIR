#!/bin/bash

# Build, audit, and probe the exact PyTorch-pinned Triton wheel without network.
# This file must be invoked directly, never embedded in a `bash -c` string.

set -euo pipefail

readonly ws=/home/zhaosiying/pypto-love-tensor-ir
readonly triton_sha=5d6048aa0a324e090ada215b609ea76620133845
readonly triton_tree=448265acc1eff726c2e528813552865b33546cc9
readonly source_tar_sha=2ebfd3f7e98dee2e8524b9b210716fbe1f07759b6d89307280a9b10ae359b43e

die() {
  printf 'offline Triton wheel build failed: %s\n' "$*" >&2
  return 1
}

require_absent_output() {
  local path=$1
  if [[ -e "$path" || -L "$path" ]]; then
    die "refusing to overwrite output: $path"
  fi
}

usage() {
  printf 'usage: %s --reviewed-manifest-sha256 SHA256\n' "$0" >&2
  return 2
}

main() {
  if [[ $# -ne 2 || $1 != --reviewed-manifest-sha256 ]]; then
    usage
  fi
  local reviewed_manifest_sha=$2
  if [[ ! $reviewed_manifest_sha =~ ^[0-9a-f]{64}$ ]]; then
    die "reviewed manifest SHA-256 is malformed"
  fi

  # run_isolated intentionally exports project sources in PYTHONPATH. None of
  # those ambient packages may participate in producer or build-venv probes.
  unset PYTHONPATH PYTHONHOME

  local locked_manifest_sha
  locked_manifest_sha=$(sed -n \
    's/^triton.dependencies.reviewed_manifest_sha256=//p' \
    "$ws/VERSIONS.lock")
  if [[ $locked_manifest_sha != "$reviewed_manifest_sha" ]]; then
    die "reviewed manifest does not match VERSIONS.lock"
  fi

  local repo=$ws/upstream/triton
  local deps=$ws/caches/triton-build-deps/$reviewed_manifest_sha
  local root=$ws/builds/triton-wheel-5d6048aa
  local src=$root/source
  local source_input=$root/source-input
  local build_input=$root/build-input
  local source_tar=$root/source.tar
  local wheel_dir=$root/wheels
  local build_venv=$root/build-venv
  local source_provenance=$root/source-provenance.json
  local producer_provenance=$root/producer-provenance.json
  local log=$ws/logs/triton-wheel-build.log
  local base_py=$ws/envs/pypto-nvidia/bin/python

  require_absent_output "$root"
  require_absent_output "$log"
  local deps_manifest_sha
  deps_manifest_sha=$(basename "$deps")
  "$base_py" -B "$ws/tools/materialize_triton_dependencies.py" \
    --output "$deps" --verify --require-reviewed \
    --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
  "$base_py" -B "$ws/tools/materialize_triton_dependencies.py" \
    --output "$deps" --probe-reviewed-tools \
    --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
  mkdir -p "$src" "$source_input" "$build_input" "$wheel_dir" \
    "$root/cmake" "$root/home" "$root/triton-home"
  local snapshot=$root/deps-snapshot
  mkdir "$snapshot"
  cp -a --reflink=auto "$deps/." "$snapshot/"
  "$base_py" -B "$ws/tools/materialize_triton_dependencies.py" \
    --output "$snapshot" --verify --require-reviewed \
    --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
  git -C "$repo" archive --format=tar --output "$source_tar" "$triton_sha"
  test "$(sha256sum "$source_tar" | cut -d " " -f 1)" = "$source_tar_sha"
  tar -xf "$source_tar" -C "$source_input"
  cp -a --reflink=auto "$source_input/." "$build_input/"
  cp -a "$snapshot/nvidia-backend-overlay/." \
    "$build_input/third_party/nvidia/backend/"
  cp -a --reflink=auto "$build_input/." "$src/"
  local source_overlay=$src/third_party/nvidia/backend
  local snapshot_overlay=$snapshot/nvidia-backend-overlay

  tree_sha() {
    "$base_py" -I -B - "$ws/tools" "$1" <<'PY'
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from materialize_triton_dependencies import tree_identity

print(tree_identity(pathlib.Path(sys.argv[2]))["sha256"])
PY
  }

  local overlay_bin_sha overlay_include_sha overlay_cupti_sha
  local source_input_sha build_input_sha
  overlay_bin_sha=$(tree_sha "$snapshot_overlay/bin")
  overlay_include_sha=$(tree_sha "$snapshot_overlay/include")
  overlay_cupti_sha=$(tree_sha "$snapshot_overlay/lib/cupti")
  source_input_sha=$(tree_sha "$source_input")
  build_input_sha=$(tree_sha "$build_input")

  verify_source_input() {
    "$base_py" -I -B - "$ws/tools" "$source_input" "$src" <<'PY'
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from materialize_triton_dependencies import verify_reference_tree_unchanged

verify_reference_tree_unchanged(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
PY
  }

  verify_source_input
  test "$(tree_sha "$src")" = "$build_input_sha"
  test "$(tree_sha "$source_overlay/bin")" = "$overlay_bin_sha"
  test "$(tree_sha "$source_overlay/include")" = "$overlay_include_sha"
  test "$(tree_sha "$source_overlay/lib/cupti")" = "$overlay_cupti_sha"

  "$base_py" -I -B -m venv --without-pip "$build_venv"
  local build_py=$build_venv/bin/python
  local build_site
  build_site=$("$build_py" -I -B -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  local producer_site=$root/producer-site
  local producer_bin=$root/producer-bin
  local producer_site_identity=$root/producer-site-identity.json
  mkdir "$producer_bin"
  "$base_py" -I -B - "$ws/tools" \
    "$producer_site" "$producer_site_identity" "$producer_provenance" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from materialize_triton_dependencies import (
    assemble_python_producer_site,
    canonical_json,
    collect_live_producer_identity,
    validate_live_producers,
)

validate_live_producers()
pathlib.Path(sys.argv[4]).write_text(
    canonical_json(collect_live_producer_identity())
)
identity = assemble_python_producer_site(
    pathlib.Path(sys.argv[2]),
    ("build", "lit", "packaging", "pyproject-hooks", "setuptools", "wheel"),
)
pathlib.Path(sys.argv[3]).write_text(
    json.dumps(identity, indent=2, sort_keys=True) + "\n"
)
PY
  local pybind_site
  pybind_site=$("$base_py" -I -B - "$snapshot/manifest.json" "$snapshot" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
record = next(item for item in manifest["packages"] if item["name"] == "pybind11")
print(pathlib.Path(sys.argv[2]) / record["expanded_root"])
PY
  )
  test -f "$pybind_site/pybind11/__init__.py"
  printf '%s\n%s\n' "$pybind_site" "$producer_site" > \
    "$build_site/reviewed-build-dependencies.pth"
  local cmake_payload=$ws/envs/pypto-nvidia/lib/python3.14/site-packages/cmake/data/bin/cmake
  printf '#!/bin/sh\nexec %s "$@"\n' "$cmake_payload" > "$producer_bin/cmake"
  cp --reflink=auto "$ws/envs/pypto-nvidia/bin/ninja" "$producer_bin/ninja"
  "$base_py" -I -B - "$producer_bin/lit" "$build_py" <<'PY'
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    f"#!{pathlib.Path(sys.argv[2]).absolute()}\n"
    "import sys\n"
    "from lit.main import main\n"
    "if __name__ == '__main__':\n"
    "    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
    "    sys.exit(main())\n"
)
PY
  chmod 0755 "$producer_bin/cmake" "$producer_bin/ninja" "$producer_bin/lit"
  test "$(sha256sum "$producer_bin/cmake" | cut -d " " -f 1)" = \
    aadd40ffd6b8bc9dac19f6dadc7ee0800cdbb3cf72f5b1f1b8b24e37f61e97da
  test "$("$producer_bin/cmake" --version | sed -n '1p')" = \
    "cmake version 3.31.10"
  local producer_site_sha producer_bin_sha
  producer_site_sha=$(tree_sha "$producer_site")
  producer_bin_sha=$(tree_sha "$producer_bin")
  "$base_py" -I -B - "$source_provenance" "$source_tar" \
    "$source_input_sha" "$build_input_sha" <<'PY'
import hashlib
import json
import pathlib
import sys

archive = pathlib.Path(sys.argv[2])
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
document = {
    "archive_sha256": digest,
    "commit": "5d6048aa0a324e090ada215b609ea76620133845",
    "extracted_tree_sha256": sys.argv[3],
    "build_input_tree_sha256": sys.argv[4],
    "kind": "triton-git-archive",
    "module_version": "3.7.1",
    "repository": "https://github.com/triton-lang/triton.git",
    "schema_version": 1,
    "tree": "448265acc1eff726c2e528813552865b33546cc9",
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n"
)
PY
  test "$("$build_py" -I -B -c 'import pybind11; print(pybind11.__version__)')" = 3.0.1
  case "$("$build_py" -I -B -c \
    'import pathlib, pybind11; print(pathlib.Path(pybind11.__file__).resolve())')" in
    "$pybind_site"/*) ;;
    *) exit 1 ;;
  esac
  test "$(readlink -f /usr/bin/clang)" = /usr/lib/llvm-21/bin/clang
  test "$(readlink -f /usr/bin/clang++)" = /usr/lib/llvm-21/bin/clang
  test "$(readlink -f /usr/bin/ld.lld)" = /usr/lib/llvm-21/bin/lld
  test "$(readlink -f /usr/bin/lld)" = /usr/lib/llvm-21/bin/lld
  test "$(readlink -f "$base_py")" = \
    "$ws/envs/pypto-nvidia/bin/python3.14"
  test "$(readlink -f "$build_py")" = \
    "$ws/envs/pypto-nvidia/bin/python3.14"
  test "$(sha256sum "$(readlink -f "$build_py")" | cut -d " " -f 1)" = \
    aa85b78409de29d21c7db9a6ea0479fd73a4e245a733ea325f5ecf21772d030f
  test "$(sha256sum /usr/lib/llvm-21/bin/clang | cut -d " " -f 1)" = \
    412bbe8c60571a1eb06f48fde89635033621caeb01a9b4ee76d46711bae8e932
  test "$(sha256sum /usr/lib/llvm-21/bin/lld | cut -d " " -f 1)" = \
    6a65863a9eba1af6b6e8969f8e96a5ad4df0e8b705f98491a28b1790ce35718c
  test "$(/usr/bin/clang --version | sed -n '1p')" = \
    "Ubuntu clang version 21.1.8 (6ubuntu1)"
  test "$(/usr/bin/ld.lld --version | sed -n '1p')" = \
    "Ubuntu LLD 21.1.8 (compatible with GNU linkers)"
  test "$(/usr/bin/bwrap --version)" = "bubblewrap 0.11.1"
  test "$(sha256sum /usr/bin/bwrap | cut -d " " -f 1)" = \
    0abea81db798ebf6b4742ac0664802d97521547a353c2a0dbdc21d76cbbfd2c0
  test "$(sha256sum /usr/bin/ar | cut -d " " -f 1)" = \
    531473816bf553e863df5aab14c8177c72b732cf80c51dcf0fa990a50125041c
  test "$(sha256sum /usr/bin/ranlib | cut -d " " -f 1)" = \
    369cf0d60a6167b11f39ed9b4bbb3d93903cc364975b244e1d084aaccf48dc92
  test "$(sha256sum /usr/bin/nm | cut -d " " -f 1)" = \
    f04262bf48192a7cbb78a17ca49ae03f8930b0372bd0115576f957b2e2a57a01
  test "$(sha256sum /usr/bin/strip | cut -d " " -f 1)" = \
    4d2ca6ba80677c3b2975e328306a779cd7bd6590a87948b5ebea9c1b41a049c8
  test "$(sha256sum /usr/bin/objcopy | cut -d " " -f 1)" = \
    05f4473d24f7330a9b13f43d007d619a2e792e33de453cd7533a2d97c30da770
  test "$(sha256sum /usr/bin/ld | cut -d " " -f 1)" = \
    97f48d93b8b076a92d2809ec29dcb17f0f37c8827358f832255e2ed22fef6075
  test "$(sha256sum "$ws/envs/pypto-nvidia/bin/cmake" | cut -d " " -f 1)" = \
    8e510409ba5512d10ddd4a732c07d95cde22eeb3b6dfa5864124b1ffc70b53c0
  test "$(sha256sum "$cmake_payload" | cut -d " " -f 1)" = \
    576c050dab1e1418b6703b5cfb523330567683dad0c60a5ff9cc23128143812e
  test "$(sha256sum "$ws/envs/pypto-nvidia/bin/ninja" | cut -d " " -f 1)" = \
    696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67

  "$build_py" -I -B - <<'PY'
import importlib.metadata as metadata
import platform

expected = {
    "build": "1.5.0",
    "lit": "18.1.8",
    "packaging": "26.2",
    "pybind11": "3.0.1",
    "pyproject-hooks": "1.2.0",
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
}
assert platform.python_version() == "3.14.6"
assert {name: metadata.version(name) for name in expected} == expected
found = {
    distribution.metadata["Name"].lower().replace("_", "-")
    for distribution in metadata.distributions()
}
assert found == set(expected), found
for forbidden in ("pip", "triton", "cmake", "ninja"):
    try:
        metadata.version(forbidden)
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError(forbidden)
PY

  local producer_path=$build_venv/bin:$producer_bin:/usr/bin
  test "$(PATH="$producer_path" command -v clang)" = /usr/bin/clang
  test "$(PATH="$producer_path" command -v clang++)" = /usr/bin/clang++
  test "$(PATH="$producer_path" command -v lld)" = /usr/bin/lld
  test "$(PATH="$producer_path" command -v cmake)" = "$producer_bin/cmake"
  test "$(PATH="$producer_path" command -v ninja)" = "$producer_bin/ninja"
  test "$(PATH="$producer_path" command -v lit)" = "$producer_bin/lit"
  test "$(PATH="$producer_path" command -v ar)" = /usr/bin/ar
  test "$(PATH="$producer_path" command -v ranlib)" = /usr/bin/ranlib
  test "$(PATH="$producer_path" command -v nm)" = /usr/bin/nm
  test "$(PATH="$producer_path" command -v strip)" = /usr/bin/strip
  test "$(PATH="$producer_path" command -v objcopy)" = /usr/bin/objcopy

  local llvm_rel json_rel
  llvm_rel=$("$base_py" -I -B -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["build_inputs"]["llvm_syspath"])' \
    "$snapshot/manifest.json")
  json_rel=$("$base_py" -I -B -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["build_inputs"]["json_syspath"])' \
    "$snapshot/manifest.json")
  local llvm=$snapshot/$llvm_rel
  local json_root=$snapshot/$json_rel
  local pybind_root
  pybind_root=$("$build_py" -I -B -c \
    'import pathlib, pybind11; print(pathlib.Path(pybind11.get_include()).parent)')

  # Reserve the log with noclobber before tee opens it; the initial existence
  # gate and this reservation both fail closed on files and symlinks.
  set -o noclobber
  : > "$log"
  set +o noclobber
  exec > >(tee -a "$log") 2>&1

  (
    ulimit -S -c 0
    ulimit -S -n 4096
    ulimit -S -u 1024
    ulimit -S -v $((24 * 1024 * 1024))
    ulimit -S -f $((20 * 1024 * 1024 * 2))
    local run_tmp_parent
    run_tmp_parent=$(dirname "$TMPDIR")
    exec /usr/bin/bwrap --die-with-parent --new-session \
      --unshare-net --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup \
      --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib \
      --ro-bind /lib64 /lib64 --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
      --ro-bind /etc/alternatives /etc/alternatives --dev /dev --proc /proc \
      --tmpfs /tmp --dir /home --dir /home/zhaosiying \
      --dir "$ws" --dir "$ws/builds" --dir "$ws/envs" --dir "$ws/runs" \
      --dir "$run_tmp_parent" \
      --ro-bind "$ws/envs/pypto-nvidia" "$ws/envs/pypto-nvidia" \
      --ro-bind "$root" "$root" --bind "$src" "$src" \
      --bind "$root/cmake" "$root/cmake" \
      --bind "$wheel_dir" "$wheel_dir" --bind "$root/home" "$root/home" \
      --bind "$root/triton-home" "$root/triton-home" \
      --ro-bind "$snapshot" "$snapshot" \
      --ro-bind "$source_overlay/bin" "$source_overlay/bin" \
      --ro-bind "$source_overlay/include" "$source_overlay/include" \
      --ro-bind "$source_overlay/lib/cupti" "$source_overlay/lib/cupti" \
      --bind "$TMPDIR" "$TMPDIR" \
      /usr/bin/env -i \
        HOME="$root/home" \
        PATH="$producer_path" \
        LD_LIBRARY_PATH="$ws/envs/pypto-nvidia/lib:/usr/lib/wsl/lib" \
        TMPDIR="$TMPDIR" TMP="$TMPDIR" TEMP="$TMPDIR" \
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
        PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_CONFIG_FILE=/dev/null \
        TRITON_OFFLINE_BUILD=1 TRITON_WHEEL_NAME=triton \
        TRITON_WHEEL_VERSION_SUFFIX=+git5d6048aa \
        TRITON_EXT_ENABLED=ON TRITON_BUILD_PROTON=ON \
        TRITON_BUILD_WITH_CCACHE=OFF TRITON_BUILD_WITH_CLANG_LLD=1 \
        TRITON_PARALLEL_LINK_JOBS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 \
        SOURCE_DATE_EPOCH=1781015236 CC=/usr/bin/clang CXX=/usr/bin/clang++ \
        TRITON_BUILD_DIR="$root/cmake" TRITON_HOME="$root/triton-home" \
        LLVM_SYSPATH="$llvm" LLVM_INCLUDE_DIRS="$llvm/include" \
        LLVM_LIBRARY_DIR="$llvm/lib" JSON_SYSPATH="$json_root" \
        PYBIND11_SYSPATH="$pybind_root" \
        TRITON_CUPTI_INCLUDE_PATH="$src/third_party/nvidia/backend/include" \
        TRITON_CUPTI_LIB_PATH="$src/third_party/nvidia/backend/lib/cupti" \
        TRITON_APPEND_CMAKE_ARGS=-DTRITON_OFFLINE_BUILD=ON \
        "$build_py" -I -B -m build --wheel --no-isolation \
          --skip-dependency-check \
          --outdir "$wheel_dir" "$src"
  )

  local generated_source_metadata=$src/python/triton.egg-info
  local retained_source_metadata=$root/generated-source-metadata
  "$base_py" -I -B - "$ws/tools" "$generated_source_metadata" \
    "$retained_source_metadata" "$build_input" <<'PY'
import os
import pathlib
import stat
import sys

sys.path.insert(0, sys.argv[1])
from materialize_triton_dependencies import _rename_no_replace

source = pathlib.Path(os.path.abspath(sys.argv[2]))
destination = pathlib.Path(os.path.abspath(sys.argv[3]))
build_input = pathlib.Path(os.path.abspath(sys.argv[4]))
expected_files = (
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "not-zip-safe",
    "requires.txt",
    "top_level.txt",
)

source_metadata = source.lstat()
if (
    source.resolve(strict=True) != source
    or not stat.S_ISDIR(source_metadata.st_mode)
    or stat.S_IMODE(source_metadata.st_mode) != 0o755
    or source_metadata.st_nlink != 2
):
    raise RuntimeError("generated Triton source metadata is not an independent directory")
if destination.exists() or destination.is_symlink():
    raise FileExistsError(f"retained source metadata already exists: {destination}")
if source_metadata.st_dev != destination.parent.stat().st_dev:
    raise RuntimeError("generated source metadata cannot be atomically retained")
reviewed_metadata = build_input / "python" / "triton.egg-info"
if reviewed_metadata.exists() or reviewed_metadata.is_symlink():
    raise RuntimeError("generated source metadata is not independent of build input")

members = sorted(source.iterdir(), key=lambda path: path.name)
if tuple(path.name for path in members) != expected_files:
    raise RuntimeError("generated Triton source metadata member set drift")
for member in members:
    metadata = member.lstat()
    if (
        member.resolve(strict=True) != member
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o644
        or metadata.st_nlink != 1
    ):
        raise RuntimeError(f"unsafe generated Triton source metadata: {member.name}")

source_identity = (source_metadata.st_dev, source_metadata.st_ino)
_rename_no_replace(source, destination, require_same_parent=False)
retained_metadata = destination.lstat()
if source.exists() or source.is_symlink():
    raise RuntimeError("generated source metadata move left the source behind")
if (
    not stat.S_ISDIR(retained_metadata.st_mode)
    or (retained_metadata.st_dev, retained_metadata.st_ino) != source_identity
):
    raise RuntimeError("generated source metadata was not atomically retained")
PY
  local retained_source_metadata_sha
  retained_source_metadata_sha=$(tree_sha "$retained_source_metadata")
  printf 'GENERATED_SOURCE_METADATA_SHA256=%s\n' "$retained_source_metadata_sha"

  # setuptools also leaves its build staging directory and Triton's generated
  # compile_commands symlink in the source tree. Retain both as immutable
  # evidence before enforcing exact equality with the reviewed build input.
  local generated_source_build=$src/build
  local retained_source_build=$root/generated-source-build
  local generated_compile_commands=$src/compile_commands.json
  local retained_compile_commands=$root/generated-compile-commands.json
  local expected_compile_commands=$root/cmake/compile_commands.json
  "$base_py" -I -B - "$ws/tools" "$generated_source_build" \
    "$retained_source_build" "$generated_compile_commands" \
    "$retained_compile_commands" "$expected_compile_commands" \
    "$build_input" <<'PY'
import ctypes
import errno
import os
import pathlib
import stat
import sys

sys.path.insert(0, sys.argv[1])
from materialize_triton_dependencies import _fsync_directory, _rename_no_replace

source_build = pathlib.Path(os.path.abspath(sys.argv[2]))
retained_build = pathlib.Path(os.path.abspath(sys.argv[3]))
source_commands = pathlib.Path(os.path.abspath(sys.argv[4]))
retained_commands = pathlib.Path(os.path.abspath(sys.argv[5]))
expected_commands = pathlib.Path(os.path.abspath(sys.argv[6]))
build_input = pathlib.Path(os.path.abspath(sys.argv[7]))
if (
    source_commands.parent != source_build.parent
    or retained_commands.parent != retained_build.parent
    or source_commands.parent.resolve(strict=True) != source_commands.parent
    or retained_commands.parent.resolve(strict=True) != retained_commands.parent
):
    raise RuntimeError("generated evidence paths do not have canonical paired parents")

build_metadata = source_build.lstat()
if (
    source_build.resolve(strict=True) != source_build
    or not stat.S_ISDIR(build_metadata.st_mode)
    or stat.S_IMODE(build_metadata.st_mode) != 0o755
    or build_metadata.st_nlink < 2
):
    raise RuntimeError("generated Triton source build is not an independent directory")
if retained_build.exists() or retained_build.is_symlink():
    raise FileExistsError(f"retained source build already exists: {retained_build}")
if build_metadata.st_dev != retained_build.parent.stat().st_dev:
    raise RuntimeError("generated source build cannot be atomically retained")
reviewed_build = build_input / "build"
if reviewed_build.exists() or reviewed_build.is_symlink():
    raise RuntimeError("generated source build is not independent of build input")

commands_metadata = source_commands.lstat()
if (
    not stat.S_ISLNK(commands_metadata.st_mode)
    or stat.S_IMODE(commands_metadata.st_mode) != 0o777
    or commands_metadata.st_nlink != 1
):
    raise RuntimeError("generated compile_commands path is not an independent symlink")
if os.readlink(source_commands) != str(expected_commands):
    raise RuntimeError("generated compile_commands target drift")
expected_metadata = expected_commands.lstat()
if (
    expected_commands.resolve(strict=True) != expected_commands
    or not stat.S_ISREG(expected_metadata.st_mode)
    or stat.S_IMODE(expected_metadata.st_mode) != 0o644
    or expected_metadata.st_nlink != 1
):
    raise RuntimeError("generated compile_commands target is unsafe")
if retained_commands.exists() or retained_commands.is_symlink():
    raise FileExistsError(
        f"retained compile_commands evidence already exists: {retained_commands}"
    )
reviewed_commands = build_input / "compile_commands.json"
if reviewed_commands.exists() or reviewed_commands.is_symlink():
    raise RuntimeError("generated compile_commands is not independent of build input")

build_identity = (build_metadata.st_dev, build_metadata.st_ino)
commands_identity = (commands_metadata.st_dev, commands_metadata.st_ino)
_rename_no_replace(source_build, retained_build, require_same_parent=False)
_fsync_directory(source_build.parent)
if retained_build.parent != source_build.parent:
    _fsync_directory(retained_build.parent)

# _rename_no_replace deliberately rejects symlink sources. Use the same Linux
# no-replace primitive only after validating the exact symlink and both
# canonical parents above; never follow the link while moving it.
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
    -100,
    os.fsencode(source_commands),
    -100,
    os.fsencode(retained_commands),
    1,
)
if result != 0:
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            "retained compile_commands evidence already exists",
            retained_commands,
        )
    raise OSError(
        error_number,
        f"renameat2(RENAME_NOREPLACE) failed: {os.strerror(error_number)}",
        retained_commands,
    )
_fsync_directory(source_commands.parent)
if retained_commands.parent != source_commands.parent:
    _fsync_directory(retained_commands.parent)

retained_build_metadata = retained_build.lstat()
retained_commands_metadata = retained_commands.lstat()
if source_build.exists() or source_build.is_symlink():
    raise RuntimeError("generated source build move left the source behind")
if source_commands.exists() or source_commands.is_symlink():
    raise RuntimeError("generated compile_commands move left the source behind")
if (
    not stat.S_ISDIR(retained_build_metadata.st_mode)
    or (retained_build_metadata.st_dev, retained_build_metadata.st_ino)
    != build_identity
):
    raise RuntimeError("generated source build was not atomically retained")
if (
    not stat.S_ISLNK(retained_commands_metadata.st_mode)
    or (retained_commands_metadata.st_dev, retained_commands_metadata.st_ino)
    != commands_identity
    or os.readlink(retained_commands) != str(expected_commands)
):
    raise RuntimeError("generated compile_commands was not atomically retained")
PY
  local retained_source_build_sha
  retained_source_build_sha=$(tree_sha "$retained_source_build")
  printf 'GENERATED_SOURCE_BUILD_SHA256=%s\n' "$retained_source_build_sha"
  printf 'GENERATED_COMPILE_COMMANDS_TARGET=%s\n' \
    "$(readlink "$retained_compile_commands")"

  "$base_py" -I -B - "$ws/tools" "$build_input" "$src" <<'PY'
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from audit_triton_wheel import verify_reference_tree_exact

verify_reference_tree_exact(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
PY

  "$base_py" -B "$ws/tools/materialize_triton_dependencies.py" \
    --output "$snapshot" --verify --require-reviewed \
    --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
  "$base_py" -B "$ws/tools/materialize_triton_dependencies.py" \
    --output "$deps" --verify --require-reviewed \
    --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
  test "$(tree_sha "$source_overlay/bin")" = "$overlay_bin_sha"
  test "$(tree_sha "$source_overlay/include")" = "$overlay_include_sha"
  test "$(tree_sha "$source_overlay/lib/cupti")" = "$overlay_cupti_sha"
  test "$(tree_sha "$producer_site")" = "$producer_site_sha"
  test "$(tree_sha "$producer_bin")" = "$producer_bin_sha"
  test "$(tree_sha "$retained_source_metadata")" = \
    "$retained_source_metadata_sha"
  test "$(tree_sha "$retained_source_build")" = \
    "$retained_source_build_sha"
  test "$(readlink "$retained_compile_commands")" = \
    "$expected_compile_commands"
  test "$(tree_sha "$source_input")" = "$source_input_sha"
  test "$(tree_sha "$build_input")" = "$build_input_sha"
  verify_source_input

  local -a built_wheels
  mapfile -t built_wheels < <(find "$wheel_dir" -maxdepth 1 -type f \
    -name 'triton-3.7.1+git5d6048aa-cp314-cp314-*.whl')
  test "${#built_wheels[@]}" -eq 1
  local producer_sha
  producer_sha=$(sed -n \
    's/^triton.producer.selected_identity_sha256=//p' "$ws/VERSIONS.lock")
  local audit_temp=$root/wheel-audit-temp
  local audit_evidence=$root/triton-wheel-audit.json
  mkdir "$audit_temp"
  "$base_py" -B "$ws/tools/audit_triton_wheel.py" \
    --workspace "$ws" \
    --wheel "${built_wheels[0]}" \
    --dependency-manifest "$deps/manifest.json" \
    --reviewed-dependency-manifest-sha256 "$deps_manifest_sha" \
    --source-provenance "$source_provenance" \
    --source-archive "$source_tar" \
    --source-input "$source_input" \
    --build-input "$build_input" \
    --built-source "$src" \
    --producer-provenance "$producer_provenance" \
    --producer-site-identity "$producer_site_identity" \
    --producer-site "$producer_site" \
    --producer-bin "$producer_bin" \
    --expected-producer-identity-sha256 "$producer_sha" \
    --evidence "$audit_evidence" \
    --temp-root "$audit_temp" \
    --readelf /usr/bin/readelf --ldd /usr/bin/ldd \
    --bwrap /usr/bin/bwrap --timeout-seconds 30
  sha256sum "${built_wheels[0]}" "$audit_evidence"
  local audit_sha
  audit_sha=$(sha256sum "$audit_evidence" | cut -d " " -f 1)
  local torch_site
  torch_site=$("$base_py" -I -B -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  local probe_prefix=$root/triton-probe
  local probe_evidence=$root/triton-wheel-probe.json
  "$base_py" -B "$ws/tools/probe_triton_wheel.py" \
    --workspace "$ws" \
    --wheel "${built_wheels[0]}" \
    --wheel-audit-evidence "$audit_evidence" \
    --expected-wheel-audit-evidence-sha256 "$audit_sha" \
    --base-python "$base_py" \
    --torch-site-packages "$torch_site" \
    --environment-lock "$ws/ENVIRONMENT.lock" \
    --probe-prefix "$probe_prefix" \
    --evidence "$probe_evidence" \
    --timeout-seconds 120
  sha256sum "$probe_evidence"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
