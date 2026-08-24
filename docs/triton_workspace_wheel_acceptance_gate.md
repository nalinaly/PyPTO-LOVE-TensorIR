# Exact PyTorch-pinned Triton workspace wheel gate

This runbook replaces the inherited external editable Triton in
`envs/pypto-nvidia` with a workspace-built wheel from PyTorch's exact source
pin. It is compatibility infrastructure only: a successful Triton kernel is a
reference smoke, never PyPTO strict-coverage evidence.

## Current source and runtime truth

- PyTorch `cf30153c...` pins Triton
  `5d6048aa0a324e090ada215b609ea76620133845`, version `3.7.1`.
- `upstream/triton` is clean at that exact commit and tree
  `448265acc1eff726c2e528813552865b33546cc9`; it has no gitlinks.
- Triton's LLVM pin is `ac5dc54d...`, distinct from TensorIR's `57109bef...`.
- The current distribution metadata says `3.7.1+git5d6048aa`, but its editable
  finder points outside the workspace to `/home/zhaosiying/codebase/triton`.
  That source has advanced to `8840a2d7...`, imports as `3.8.0`, and loads an
  external `libtriton.so`. The environment must continue to fail closed.
- The single-DSO prerequisite is accepted. Execute this runbook only after
  TargetInfo acceptance and a fresh explicit CPU-only coexistence heavy
  preflight. This mode uses the user-authorized 24 GiB launch floor, a 16 GiB
  owned-run pause floor, and retains the NVIDIA-isolation gates. It is never
  valid for benchmark evidence.

Never build inside `upstream/triton`. Never reuse `~/.triton`, the original
`triton-dev` environment, or TensorIR's LLVM.

## 1. Source-pin gate

```bash
set -euo pipefail
export GIT_OPTIONAL_LOCKS=0
export GIT_NO_LAZY_FETCH=1
ws=/home/zhaosiying/pypto-love-tensor-ir
pytorch=$ws/upstream/pytorch
triton=$ws/upstream/triton
pytorch_sha=cf30153c4c131c8164ee7798e5022d810682e2cb
triton_sha=5d6048aa0a324e090ada215b609ea76620133845

test "$(git -C "$pytorch" rev-parse HEAD^{commit})" = "$pytorch_sha"
test "$(git -C "$pytorch" rev-parse HEAD^{tree})" = \
  7cda5eae52ace99ca4daa7e623920cc93782cc6c
test "$(git -C "$pytorch" show \
  "$pytorch_sha:.ci/docker/ci_commit_pins/triton.txt")" = "$triton_sha"
test "$(git -C "$pytorch" show \
  "$pytorch_sha:.ci/docker/triton_version.txt")" = 3.7.1
test "$(git -C "$triton" rev-parse HEAD^{commit})" = "$triton_sha"
test "$(git -C "$triton" rev-parse HEAD^{tree})" = \
  448265acc1eff726c2e528813552865b33546cc9
test "$(git -C "$triton" remote get-url origin)" = \
  https://github.com/triton-lang/triton.git
test -z "$(git -C "$triton" status --porcelain=v1 --untracked-files=all)"
test "$(git -C "$triton" ls-tree -r "$triton_sha" | \
  awk '$1=="160000" {n++} END {print n+0}')" = 0
if git -C "$triton" cat-file -e "$triton_sha:.gitmodules" 2>/dev/null; then
  exit 1
fi
test "$(git -C "$triton" show "$triton_sha:cmake/llvm-hash.txt")" = \
  ac5dc54d509169d387fcfd495d71853d81c46484
test "$(sha256sum "$triton/third_party/nvidia/backend/lib/libdevice.10.bc" | \
  cut -d " " -f 1)" = \
  5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee
```

The shallow checkout proves the local object/tree identity, not a complete
history or trusted GitHub signing chain.

## 2. Online dependency materialization

The upstream downloader checks URLs/versions but not content hashes. This
workspace materializer requires an exact `Content-Length` on the initial HTTP
200 response. A premature EOF is retried a bounded number of times with a
`Range` request; a resumed response is accepted only when HTTP 206,
`Content-Range`, `Content-Length`, the resume offset and the original total all
agree exactly. The partial file is size/hash verified and fsynced before it is
published inside the private materialization staging tree.

An optional seed directory can avoid downloading an already source-locked
archive. It must be an absolute canonical, user-owned directory below this
workspace. A matching URL-basename file is accepted only when it is a regular
non-symlink file with one hard link and its digest (and source-locked size, when
present) matches `PACKAGE_SPECS`. The materializer makes an independent private
copy, checks the source and copy size/digest, and records exact acquisition
provenance. Missing seed files use the normal network path; an existing seed for
an archive without a source-locked SHA-256 fails closed.

The LLVM archive is source-locked at 1,309,519,196 bytes and SHA-256
`11a11a5a90da7e4b53ef4cf0f259143d14633cae8543a95cb2d99e4af6b902f8`.
That identity is backed by an independent complete-archive inspection. Four
live official-URL range samples (start, two interior offsets and near-end)
matched the inspected bytes; that is spot evidence, not a complete live
official-URL re-download. The materialized manifest and expanded contents still
require the independent review below.

Download/copy and expand each dependency first, without compiling anything:

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --allow-protected-cpu-only-coexistence \
  --timeout-seconds 7200 --minimum-free-disk-gib 128 \
  --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
out=$ws/builds/triton-deps-materialize-5d6048aa
log=$ws/logs/triton-dependency-materialization.log
py=$ws/envs/pypto-nvidia/bin/python
seed=$ws/caches/triton-download-seeds
test ! -e "$out"
test ! -e "$log"
test -d "$seed"
exec > >(tee "$log") 2>&1
"$py" "$ws/tools/materialize_triton_dependencies.py" \
  --output "$out" --seed-download-dir "$seed"
"$py" "$ws/tools/materialize_triton_dependencies.py" \
  --output "$out" --verify >/dev/null
sha256sum "$out/manifest.json"
'
```

This creates an explicitly `materialized-unreviewed` manifest. Stop here.
Independently review every URL, acquisition record, archive SHA/size, expanded
tree digest, overlay contents/symlinks and tool version. Then add every newly
approved archive SHA-256 to the version-controlled `PACKAGE_SPECS` in
`tools/materialize_triton_dependencies.py` and the corresponding
`triton.dependencies.archive.*` fields in `VERSIONS.lock`. Freeze the reviewed
candidate manifest SHA-256 in both `REVIEWED_MANIFEST_SHA256` and
`triton.dependencies.reviewed_manifest_sha256`; update `LOCK_EXPECTATIONS`, run
the unit tests, obtain independent review, and commit that lock update. Do not
build from an unreviewed manifest, from hashes stored only inside the generated
manifest, or from a manifest digest calculated at promotion time.

After review, promote without changing bytes:

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --allow-protected-cpu-only-coexistence \
  --timeout-seconds 7200 --minimum-free-disk-gib 128 \
  --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/builds/triton-deps-materialize-5d6048aa
manifest_sha=$(sed -n \
  "s/^triton.dependencies.reviewed_manifest_sha256=//p" \
  "$ws/VERSIONS.lock")
test "$manifest_sha" != UNREVIEWED
test "${#manifest_sha}" -eq 64
test "$(sha256sum "$src/manifest.json" | cut -d " " -f 1)" = \
  "$manifest_sha"
dst=$ws/caches/triton-build-deps/$manifest_sha
test ! -e "$dst"
envs/pypto-nvidia/bin/python tools/materialize_triton_dependencies.py \
  --output "$src" --promote-reviewed \
  --expected-manifest-sha256 "$manifest_sha" >/dev/null
envs/pypto-nvidia/bin/python tools/materialize_triton_dependencies.py \
  --output "$src" --verify --require-reviewed \
  --expected-manifest-sha256 "$manifest_sha" >/dev/null
envs/pypto-nvidia/bin/python tools/materialize_triton_dependencies.py \
  --output "$src" --probe-reviewed-tools \
  --expected-manifest-sha256 "$manifest_sha" >/dev/null
mkdir -p "$ws/caches/triton-build-deps"
mv "$src" "$dst"
envs/pypto-nvidia/bin/python tools/materialize_triton_dependencies.py \
  --output "$dst" --verify --require-reviewed \
  --expected-manifest-sha256 "$manifest_sha" >/dev/null
printf "TRITON_DEPS=%s\n" "$dst"
'
```

Record the manifest digest in evidence before the offline build.

## 3. Frozen wheel recipe

This project chooses the official PyTorch wheel-builder ABI option
`TRITON_EXT_ENABLED=ON`, keeps default Proton enabled, uses PyTorch CI's
`pybind11==3.0.1`, and selects the official builder's clang/lld mode. The
complete recipe is frozen in `VERSIONS.lock`.

The producer snapshot validates the selected Python/package versions and the
actual Python/CMake/Ninja/Clang/LLD executables plus their dynamic runtime
closure. It prevents silent local drift but is not a claim that arbitrary host
headers or two independent machines produce bit-identical wheels. The built
wheel SHA/native manifest is the accepted binary identity; a later rebuild is a
new artifact requiring review.

Build from `git archive` in a bubblewrap sandbox. Replace
`REPLACE_WITH_REVIEWED_MANIFEST_SHA256` only with the reviewed value from
section 2.

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --allow-protected-cpu-only-coexistence \
  --timeout-seconds 14400 --minimum-free-disk-gib 128 \
  --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
repo=$ws/upstream/triton
deps=$ws/caches/triton-build-deps/REPLACE_WITH_REVIEWED_MANIFEST_SHA256
root=$ws/builds/triton-wheel-5d6048aa
src=$root/source
source_input=$root/source-input
build_input=$root/build-input
source_tar=$root/source.tar
wheel_dir=$root/wheels
build_venv=$root/build-venv
source_provenance=$root/source-provenance.json
producer_provenance=$root/producer-provenance.json
log=$ws/logs/triton-wheel-build.log
base_py=$ws/envs/pypto-nvidia/bin/python

test ! -e "$root"
test ! -e "$log"
deps_manifest_sha=$(basename "$deps")
"$base_py" "$ws/tools/materialize_triton_dependencies.py" \
  --output "$deps" --verify --require-reviewed \
  --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
"$base_py" "$ws/tools/materialize_triton_dependencies.py" \
  --output "$deps" --probe-reviewed-tools \
  --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
mkdir -p "$src" "$source_input" "$build_input" "$wheel_dir" \
  "$root/cmake" "$root/home" \
  "$root/triton-home"
snapshot=$root/deps-snapshot
mkdir "$snapshot"
cp -a --reflink=auto "$deps/." "$snapshot/"
"$base_py" "$ws/tools/materialize_triton_dependencies.py" \
  --output "$snapshot" --verify --require-reviewed \
  --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
git -C "$repo" archive --format=tar --output "$source_tar" \
  5d6048aa0a324e090ada215b609ea76620133845
test "$(sha256sum "$source_tar" | cut -d " " -f 1)" = \
  2ebfd3f7e98dee2e8524b9b210716fbe1f07759b6d89307280a9b10ae359b43e
tar -xf "$source_tar" -C "$source_input"
cp -a --reflink=auto "$source_input/." "$build_input/"
cp -a "$snapshot/nvidia-backend-overlay/." \
  "$build_input/third_party/nvidia/backend/"
cp -a --reflink=auto "$build_input/." "$src/"
source_overlay=$src/third_party/nvidia/backend
snapshot_overlay=$snapshot/nvidia-backend-overlay
tree_sha() {
  PYTHONPATH="$ws/tools" "$base_py" -B - "$1" <<'"'"'PY'"'"'
import pathlib, sys
from materialize_triton_dependencies import tree_identity
print(tree_identity(pathlib.Path(sys.argv[1]))["sha256"])
PY
}
overlay_bin_sha=$(tree_sha "$snapshot_overlay/bin")
overlay_include_sha=$(tree_sha "$snapshot_overlay/include")
overlay_cupti_sha=$(tree_sha "$snapshot_overlay/lib/cupti")
source_input_sha=$(tree_sha "$source_input")
build_input_sha=$(tree_sha "$build_input")
verify_source_input() {
  PYTHONPATH="$ws/tools" "$base_py" -B - "$source_input" "$src" <<'"'"'PY'"'"'
import pathlib, sys
from materialize_triton_dependencies import verify_reference_tree_unchanged
verify_reference_tree_unchanged(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
PY
}
verify_source_input
test "$(tree_sha "$src")" = "$build_input_sha"
test "$(tree_sha "$source_overlay/bin")" = "$overlay_bin_sha"
test "$(tree_sha "$source_overlay/include")" = "$overlay_include_sha"
test "$(tree_sha "$source_overlay/lib/cupti")" = "$overlay_cupti_sha"

"$base_py" -m venv --without-pip "$build_venv"
build_py=$build_venv/bin/python
build_site=$("$build_py" -c \
  "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
producer_site=$root/producer-site
producer_bin=$root/producer-bin
producer_site_identity=$root/producer-site-identity.json
mkdir "$producer_bin"
PYTHONPATH="$ws/tools" "$base_py" -B - \
  "$producer_site" "$producer_site_identity" "$producer_provenance" <<'"'"'PY'"'"'
import json, pathlib, sys
from materialize_triton_dependencies import (
    assemble_python_producer_site,
    canonical_json,
    collect_live_producer_identity,
    validate_live_producers,
)

validate_live_producers()
pathlib.Path(sys.argv[3]).write_text(
    canonical_json(collect_live_producer_identity())
)
identity = assemble_python_producer_site(
    pathlib.Path(sys.argv[1]),
    ("build", "lit", "packaging", "pyproject-hooks", "setuptools", "wheel"),
)
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(identity, indent=2, sort_keys=True) + "\n"
)
PY
pybind_site=$("$base_py" - "$snapshot/manifest.json" "$snapshot" <<'"'"'PY'"'"'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
record = next(item for item in manifest["packages"] if item["name"] == "pybind11")
print(pathlib.Path(sys.argv[2]) / record["expanded_root"])
PY
)
test -f "$pybind_site/pybind11/__init__.py"
printf "%s\n%s\n" "$pybind_site" "$producer_site" > \
  "$build_site/reviewed-build-dependencies.pth"
cmake_payload=$ws/envs/pypto-nvidia/lib/python3.14/site-packages/cmake/data/bin/cmake
cp --reflink=auto "$cmake_payload" "$producer_bin/cmake"
cp --reflink=auto "$ws/envs/pypto-nvidia/bin/ninja" "$producer_bin/ninja"
"$base_py" -B - "$producer_bin/lit" "$build_py" <<'"'"'PY'"'"'
import pathlib, sys

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
producer_site_sha=$(tree_sha "$producer_site")
producer_bin_sha=$(tree_sha "$producer_bin")
"$base_py" -B - "$source_provenance" "$source_tar" \
  "$source_input_sha" "$build_input_sha" <<'"'"'PY'"'"'
import hashlib, json, pathlib, sys

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
test "$("$build_py" -B -c "import pybind11; print(pybind11.__version__)")" = 3.0.1
case "$("$build_py" -B -c "import pathlib,pybind11; print(pathlib.Path(pybind11.__file__).resolve())")" in
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
"$build_py" - <<'"'"'PY'"'"'
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
found = {distribution.metadata["Name"].lower() for distribution in metadata.distributions()}
assert found == set(expected), found
for forbidden in ("pip", "triton", "cmake", "ninja"):
    try:
        metadata.version(forbidden)
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError(forbidden)
PY
producer_path=$build_venv/bin:$producer_bin:/usr/bin
test "$(PATH="$producer_path" command -v clang)" = /usr/bin/clang
test "$(PATH="$producer_path" command -v clang++)" = /usr/bin/clang++
test "$(PATH="$producer_path" command -v lld)" = /usr/bin/lld
test "$(PATH="$producer_path" command -v cmake)" = \
  "$producer_bin/cmake"
test "$(PATH="$producer_path" command -v ninja)" = \
  "$producer_bin/ninja"
test "$(PATH="$producer_path" command -v lit)" = "$producer_bin/lit"
test "$(PATH="$producer_path" command -v ar)" = /usr/bin/ar
test "$(PATH="$producer_path" command -v ranlib)" = /usr/bin/ranlib
test "$(PATH="$producer_path" command -v nm)" = /usr/bin/nm
test "$(PATH="$producer_path" command -v strip)" = /usr/bin/strip
test "$(PATH="$producer_path" command -v objcopy)" = /usr/bin/objcopy

llvm_rel=$("$base_py" -c \
  "import json; print(json.load(open(\"$snapshot/manifest.json\"))[\"build_inputs\"][\"llvm_syspath\"])")
json_rel=$("$base_py" -c \
  "import json; print(json.load(open(\"$snapshot/manifest.json\"))[\"build_inputs\"][\"json_syspath\"])")
llvm=$snapshot/$llvm_rel
json_root=$snapshot/$json_rel
pybind_root=$("$build_py" -c \
  "import pathlib,pybind11; print(pathlib.Path(pybind11.get_include()).parent)")
exec > >(tee "$log") 2>&1

(
ulimit -S -c 0
ulimit -S -n 4096
ulimit -S -u 1024
ulimit -S -v $((24 * 1024 * 1024))
ulimit -S -f $((20 * 1024 * 1024 * 2))
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
    "$build_py" -m build --wheel --no-isolation \
      --outdir "$wheel_dir" "$src"
)

"$base_py" "$ws/tools/materialize_triton_dependencies.py" \
  --output "$snapshot" --verify --require-reviewed \
  --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
"$base_py" "$ws/tools/materialize_triton_dependencies.py" \
  --output "$deps" --verify --require-reviewed \
  --expected-manifest-sha256 "$deps_manifest_sha" >/dev/null
test "$(tree_sha "$source_overlay/bin")" = "$overlay_bin_sha"
test "$(tree_sha "$source_overlay/include")" = "$overlay_include_sha"
test "$(tree_sha "$source_overlay/lib/cupti")" = "$overlay_cupti_sha"
test "$(tree_sha "$producer_site")" = "$producer_site_sha"
test "$(tree_sha "$producer_bin")" = "$producer_bin_sha"
test "$(tree_sha "$source_input")" = "$source_input_sha"
test "$(tree_sha "$build_input")" = "$build_input_sha"
PYTHONPATH="$ws/tools" "$base_py" -B - "$build_input" "$src" <<'"'"'PY'"'"'
import pathlib, sys
from materialize_triton_dependencies import verify_reference_tree_unchanged
verify_reference_tree_unchanged(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
PY
mapfile -t built_wheels < <(find "$wheel_dir" -maxdepth 1 -type f \
  -name "triton-3.7.1+git5d6048aa-cp314-cp314-*.whl")
test "${#built_wheels[@]}" -eq 1
producer_sha=$(sed -n \
  "s/^triton.producer.selected_identity_sha256=//p" "$ws/VERSIONS.lock")
audit_temp=$root/wheel-audit-temp
audit_evidence=$root/triton-wheel-audit.json
mkdir "$audit_temp"
"$base_py" "$ws/tools/audit_triton_wheel.py" \
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
audit_sha=$(sha256sum "$audit_evidence" | cut -d " " -f 1)
torch_site=$("$base_py" -B -c \
  "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
probe_prefix=$root/triton-probe
probe_evidence=$root/triton-wheel-probe.json
"$base_py" "$ws/tools/probe_triton_wheel.py" \
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
'
```

The source checkout and original `triton-dev` environment are absent. The
project prefix, build venv, dependency snapshot and build-root remainder are
read-only; only the disposable source, CMake output, wheel output, build home,
Triton home and this run's temp directory are writable.

## 4. Wheel and dependency audit

Before installation, require:

- one cp314 Linux wheel, distribution `3.7.1+git5d6048aa`, module version
  `3.7.1`;
- unique safe ZIP members, no archive symlink, editable `.pth` or finder;
- every RECORD entry except RECORD itself has matching SHA-256 and size;
- an ELF-magic-derived native manifest with SHA, Build ID, architecture,
  RPATH/RUNPATH, `DT_NEEDED` and successful `ldd`;
- no native dependency on HIP/HSA/ROCm/GemSim. AMD backend source in the exact
  official wheel is allowed but must not load an AMD runtime;
- wheel-owned `FileCheck`, `ptxas`, `ptxas-blackwell`, `cuobjdump`, `nvdisasm`,
  CUDA headers, libdevice and CUPTI;
- exact tool versions 12.8.93/13.1.80/13.1.80/13.1.80 and per-file SHA;
- libdevice SHA-256
  `5c2fae37c86e68c3a38605a95f512d7d12d5f3db986310be47f57304aa72a5ee`;
- dependency manifest/archive/tree hashes and producer versions in a
  machine-readable wheel evidence file.

Do not apply PyPTO's one-DSO rule to Triton: Proton/instrumentation may add
legitimate native objects. Freeze and review the complete native manifest.

## 5. Fresh probe and reference-only SM120 smoke

`tools/probe_triton_wheel.py` creates a fresh `venv --without-pip`, installs
only the audit-bound wheel with its narrow stdlib Wheel/RECORD installer, and
builds a Torch runtime view that excludes ambient `.pth`, editable and Triton
carriers. Two new `-I -B -S` processes prove:

- exactly one non-editable distribution owned by the probe;
- `direct_url.json` is a workspace archive whose SHA equals the wheel;
- every imported `triton.*` module and mapped `libtriton` is below the probe;
- installed RECORD/package/native bytes equal the audited wheel;
- no editable finder exists in meta path, path hooks, modules or importer
  cache;
- default SM120 backend selects wheel-owned `ptxas-blackwell` 13.1.80;
- `has_triton_package()` is true, `get_triton_version()==(3, 7)`, core
  `triton_compat` imports succeed, and `triton_key` is stable in two processes.

After an exclusive `gpu-benchmark` preflight, run a minimal Triton vector add
on SM120 using a fresh workspace cache and compare with Torch. Label it
`reference-only-triton-sm120`; it is not a PyPTO kernel, coverage or performance
result.

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode gpu-benchmark --timeout-seconds 600 --minimum-free-disk-gib 64 \
  --run-id-file /home/zhaosiying/pypto-love-tensor-ir/builds/triton-wheel-5d6048aa/reference-smoke-run-id.json \
  --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
root=$ws/builds/triton-wheel-5d6048aa
probe_evidence=$root/triton-wheel-probe.json
probe_prefix=$root/triton-probe
probe_site=$probe_prefix/lib/python3.14/site-packages
torch_view=$probe_prefix/.torch-runtime-view
cache=$root/reference-smoke-cache
evidence=$root/reference-sm120-smoke.provisional.json
test -f "$probe_evidence"
test -d "$probe_site"
test -d "$torch_view"
test ! -e "$cache"
test ! -e "$evidence"
probe_sha=$(sha256sum "$probe_evidence" | cut -d " " -f 1)
"$probe_prefix/bin/python" -I -B -S \
  "$ws/benchmarks/operators/triton_reference_sm120.py" \
  --workspace "$ws" \
  --probe-evidence "$probe_evidence" \
  --expected-probe-evidence-sha256 "$probe_sha" \
  --probe-prefix "$probe_prefix" \
  --probe-site "$probe_site" \
  --torch-runtime-view "$torch_view" \
  --cache-dir "$cache" \
  --evidence "$evidence"
sha256sum "$evidence"
'
```

After the owned GPU run exits and `process.json` is final, bind the provisional
smoke to the exclusive run:

```bash
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
root=$ws/builds/triton-wheel-5d6048aa
run_id_file=$root/reference-smoke-run-id.json
provisional=$root/reference-sm120-smoke.provisional.json
final=$root/reference-sm120-smoke.json
base_py=$ws/envs/pypto-nvidia/bin/python
test -f "$run_id_file"
test -f "$provisional"
test ! -e "$final"
run_id=$("$base_py" -B -c \
  "import json; print(json.load(open(\"$run_id_file\"))[\"run_id\"])")
provisional_sha=$(sha256sum "$provisional" | cut -d " " -f 1)
"$base_py" "$ws/tools/finalize_triton_reference_smoke.py" \
  --workspace "$ws" \
  --provisional-evidence "$provisional" \
  --expected-provisional-evidence-sha256 "$provisional_sha" \
  --run-id "$run_id" \
  --final-evidence "$final"
sha256sum "$final"
```

## 6. Project-environment replacement transaction

Only after the audit, fresh probe, and finalized exclusive SM120 smoke pass may
`tools/replace_triton_environment.py` mutate the project prefix. It is the only
installer for this gate: it uses the stdlib Wheel/RECORD installer, not pip;
seals the complete old editable/source/native/Torch identity; publishes a
durable phase journal before mutation; backs up every removal target; audits in
fresh subprocesses; and rolls back on `INT`, `TERM`, `HUP`, or any post-audit
failure.

First freeze every input digest and run the read-only plan under the ordinary
shared environment-consumer lock:

```bash
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
root=$ws/builds/triton-wheel-5d6048aa
base_py=$ws/envs/pypto-nvidia/bin/python
tool=$ws/tools/replace_triton_environment.py
mapfile -t wheels < <(find "$root/wheels" -maxdepth 1 -type f \
  -name 'triton-3.7.1+git5d6048aa-cp314-cp314-linux_x86_64.whl')
test "${#wheels[@]}" -eq 1
wheel=${wheels[0]}
audit=$root/triton-wheel-audit.json
probe=$root/triton-wheel-probe.json
smoke=$root/reference-sm120-smoke.json
backup=$root/environment-replacement-backup
terminal=$root/environment-replacement.json
test -f "$audit" -a -f "$probe" -a -f "$smoke"
test ! -e "$backup" -a ! -e "$terminal"
audit_sha=$(sha256sum "$audit" | cut -d ' ' -f 1)
probe_sha=$(sha256sum "$probe" | cut -d ' ' -f 1)
smoke_sha=$(sha256sum "$smoke" | cut -d ' ' -f 1)
environment_sha=$(sha256sum "$ws/ENVIRONMENT.lock" | cut -d ' ' -f 1)
common=(
  --workspace "$ws"
  --prefix "$ws/envs/pypto-nvidia"
  --wheel "$wheel"
  --wheel-audit-evidence "$audit"
  --expected-wheel-audit-evidence-sha256 "$audit_sha"
  --wheel-probe-evidence "$probe"
  --expected-wheel-probe-evidence-sha256 "$probe_sha"
  --gpu-smoke-evidence "$smoke"
  --expected-gpu-smoke-evidence-sha256 "$smoke_sha"
  --environment-lock "$ws/ENVIRONMENT.lock"
  --expected-environment-lock-sha256 "$environment_sha"
  --backup-root "$backup"
  --evidence "$terminal"
  --timeout-seconds 180
)
"$base_py" -B "$ws/tools/run_isolated.py" \
  --mode heavy --allow-protected-cpu-only-coexistence \
  --timeout-seconds 1800 --minimum-free-disk-gib 64 \
  --environment pypto-nvidia --framework-profile pypto -- \
  "$base_py" -B "$tool" "${common[@]}" --plan
```

Independently review the printed plan. Then repeat the variable/digest setup in
the same shell and execute the mutating action with the exclusive environment
lock. The replacement interpreter must be the direct child of
`run_isolated.py`; do not insert a shell, `python -c`, or `python -m` between
them.

```bash
"$base_py" -B "$ws/tools/run_isolated.py" \
  --mode heavy --allow-protected-cpu-only-coexistence \
  --timeout-seconds 1800 --minimum-free-disk-gib 64 \
  --environment pypto-nvidia --framework-profile pypto \
  --environment-lock-mode exclusive -- \
  "$base_py" -B "$tool" "${common[@]}" --apply
test -f "$terminal"
sha256sum "$terminal" "$backup/journal.json" "$backup/manifest.json"
```

All ordinary `run_isolated.py --environment pypto-nvidia` consumers hold the
same lock in shared mode for their full child lifecycle. The transaction holds
it exclusively, passes the descriptor plus device/inode and exact controller
PID/start-ticks to the direct child, and still performs `/proc` prefix-user
audits at every mutation boundary. No new framework consumer can enter a
partially replaced prefix.

After `SIGKILL`, WSL interruption, or any ambiguous terminal state, do not
rerun `--apply`. Use the durable journal under `backup` with one of these direct
exclusive actions. `--recover` verifies an already committed install or rolls
back an incomplete transaction; `--rollback` explicitly restores the sealed
old editable installation. Recovery evidence must not overwrite the original
terminal evidence.

```bash
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
root=$ws/builds/triton-wheel-5d6048aa
base_py=$ws/envs/pypto-nvidia/bin/python
tool=$ws/tools/replace_triton_environment.py
backup=$root/environment-replacement-backup
test -d "$backup"
test -f "$backup/journal.json"
recovery=$root/environment-replacement-recovery.json
test ! -e "$recovery"
"$base_py" -B "$ws/tools/run_isolated.py" \
  --mode heavy --allow-protected-cpu-only-coexistence \
  --timeout-seconds 1800 --minimum-free-disk-gib 64 \
  --environment pypto-nvidia --framework-profile pypto \
  --environment-lock-mode exclusive -- \
  "$base_py" -B "$tool" --workspace "$ws" \
    --backup-root "$backup" --evidence "$recovery" \
    --timeout-seconds 180 --recover

# Use only when an explicit restoration of the old environment is intended.
rollback=$root/environment-replacement-rollback.json
test -f "$backup/manifest.json"
test ! -e "$rollback"
"$base_py" -B "$ws/tools/run_isolated.py" \
  --mode heavy --allow-protected-cpu-only-coexistence \
  --timeout-seconds 1800 --minimum-free-disk-gib 64 \
  --environment pypto-nvidia --framework-profile pypto \
  --environment-lock-mode exclusive -- \
  "$base_py" -B "$tool" --workspace "$ws" \
    --backup-root "$backup" --evidence "$rollback" \
    --timeout-seconds 180 --rollback
```

On accepted apply, stop before a framework launch: the installed distribution
identity has changed, so the old `ENVIRONMENT.lock` must fail verification.
Generate and independently review the new complete environment/Triton identity,
then land the lock update and its `environment_identity.py` /
`runtime_identity.py` enforcement as a separate commit. Retain the backup and
journal until that promotion is accepted. Never write to the original
`triton-dev`, `upstream/triton`, external editable source, user cache, or either
protected AMD/zcode tree.
