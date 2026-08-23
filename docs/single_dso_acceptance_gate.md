# PyPTO single-DSO acceptance gate

This is the zero-context runbook for the staged two-file transaction in
`projects/pypto/CMakeLists.txt` and
`projects/pypto/python/bindings/CMakeLists.txt`.

## Current status

The transaction is **not accepted**. Historical support is an editable build,
486 focused tests, and one old full run with 10,173 passed, 58 skipped, and 3
failed. Targeted fixes exist, but there is no post-fix full rerun or fresh
wheel. `builds/pypto-wheel.dfB4Xk/wheels` is empty.

The current heavy preflight is red for protected zcode TP=2 vLLM/gem5 work.
Do not run any command below until that lane exits naturally and a fresh
`./tools/preflight.py --mode heavy` returns zero. Never signal or clean up the
protected lane.

## Frozen invariants

- Git tracked dirt is exactly the two CMake files above. Ignore, but do not
  clean or commit, untracked content inside pinned NVIDIA submodules.
- Compiler sources compile once into `pypto_compiler_objects`.
- The installed product contains exactly one native DSO,
  `pypto/pypto_core*.so`.
- The binding target contains only binding sources and embeds the compiler
  object target. It must not restore `${PYPTO_SOURCES}` directly.
- `include/` and `runtime/src/common` are PUBLIC object-target usage
  requirements; libbacktrace/msgpack are object-target dependencies.
- Jobs remain `PYPTO_BUILD_JOBS=2` and `PYPTO_TEST_JOBS=2` unless a later
  measured resource decision changes them.

Every block below starts through `tools/run_isolated.py`, which performs a new
heavy preflight and assigns workspace-local temp/cache/run ownership.

## 1. Fresh native object and C++ test

Run from the workspace root:

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/projects/pypto
bld=$ws/builds/pypto-single-dso-native
log=$ws/logs/pypto-single-dso-native.log
py=$ws/envs/pypto-nvidia/bin/python

test ! -e "$bld"
test ! -e "$log"
mkdir -p "$bld"
source "$src/.claude/skills/testing/load-env.sh"
exec > >(tee "$log") 2>&1

nanobind_dir=$("$py" -m nanobind --cmake_dir)
cmake -S "$src" -B "$bld" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_TESTING=ON \
  -DPython_EXECUTABLE="$py" \
  -DPython3_EXECUTABLE="$py" \
  -Dnanobind_DIR="$nanobind_dir"
cmake --build "$bld" \
  --target pypto_compiler_objects pypto_dsa_reuse_penalty_solver_test \
  --parallel "$PYPTO_BUILD_JOBS"
ctest --test-dir "$bld" --output-on-failure -j "$PYPTO_TEST_JOBS"

"$py" - "$bld/compile_commands.json" <<'"'"'PY'"'"'
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text())
binding = next(x for x in rows if x["file"].endswith("python/bindings/modules/ir.cpp"))
native = next(x for x in rows if x["file"].endswith("src/ir/core.cpp"))
assert "/runtime/src/common" in binding["command"]
assert "pypto_compiler_objects.dir" in native["command"]
PY
'
```

Expected: native CTest 1/1 passes. This build directory must not preexist.

## 2. Post-fix editable full rerun

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --environment pypto-nvidia --framework-profile pypto -- \
  /usr/bin/env \
    PYPTO_BUILD_JOBS=2 \
    PYPTO_TEST_JOBS=2 \
    PYTEST_XDIST_AUTO_NUM_WORKERS=2 \
    PYTHONPATH=/home/zhaosiying/pypto-love-tensor-ir/projects/pypto/python \
    /home/zhaosiying/pypto-love-tensor-ir/envs/pypto-nvidia/bin/python \
    -m pytest \
    /home/zhaosiying/pypto-love-tensor-ir/projects/pypto/tests/ut \
    -n 2 -q
```

Expected editable result is 10,176 passed and 59 skipped; the symlink caller
probe is the expected editable-only skip. This rerun closes the historical
three failures but does not replace the wheel gate.

## 3. Fresh wheel and DSO audit

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/projects/pypto
root=$ws/builds/pypto-wheel.dfB4Xk
out=$root/wheels
skbuild=$root/skbuild
log=$ws/logs/pypto-single-dso-wheel.log
py=$ws/envs/pypto-nvidia/bin/python

test ! -e "$skbuild"
test ! -e "$log"
test -z "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)"
source "$src/.claude/skills/testing/load-env.sh"
export PIP_NO_INDEX=1
export SKBUILD_BUILD_DIR="$skbuild"
exec > >(tee "$log") 2>&1

"$py" -m build --wheel --no-isolation \
  --outdir "$out" \
  --config-setting="build-dir=$skbuild" \
  "$src"
mapfile -t wheels < <(find "$out" -maxdepth 1 -type f -name "pypto-0.1.0-*.whl")
test "${#wheels[@]}" -eq 1
wheel=${wheels[0]}
sha256sum "$wheel"

"$py" - "$wheel" <<'"'"'PY'"'"'
import pathlib, stat, sys, zipfile
wheel = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(wheel) as zf:
    infos = zf.infolist()
    names = [x.filename for x in infos]
    assert all(not pathlib.PurePosixPath(n).is_absolute() and ".." not in pathlib.PurePosixPath(n).parts for n in names)
    assert not any(stat.S_ISLNK(x.external_attr >> 16) for x in infos)
    native = [n for n in names if n.startswith("pypto/pypto_core") and n.endswith(".so")]
    assert len(native) == 1, native
PY

mapfile -t native_so < <(find "$skbuild" -type f -name "pypto_core*.so")
test "${#native_so[@]}" -eq 1
if ldd "${native_so[0]}" | grep -Eiq "libpypto|amdhip|hsa-runtime|gemsim"; then
  echo "unexpected external compiler/AMD dependency" >&2
  exit 1
fi
'
```

The scikit build directory is outside `projects/pypto/build`; a plain wheel
command using the pyproject persistent build directory is not fresh evidence.

## 4. Clean wheel install, full suite and symlink probe

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/projects/pypto
root=$ws/builds/pypto-wheel.dfB4Xk
venv=$root/probe-venv
log=$ws/logs/pypto-single-dso-wheel-tests.log
py=$ws/envs/pypto-nvidia/bin/python

test ! -e "$venv"
test ! -e "$log"
source "$src/.claude/skills/testing/load-env.sh"
exec > >(tee "$log") 2>&1
mapfile -t wheels < <(find "$root/wheels" -maxdepth 1 -type f -name "pypto-0.1.0-*.whl")
test "${#wheels[@]}" -eq 1
wheel=${wheels[0]}

"$py" -m venv --without-pip "$venv"
probe_py=$venv/bin/python
base_site=$("$py" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
probe_site=$("$probe_py" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
printf "%s\n" "$base_site" > "$probe_site/base-environment-dependencies.pth"
"$py" -m pip install --no-deps --no-compile --target "$probe_site" "$wheel"

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$probe_py" -I - "$probe_site" <<'"'"'PY'"'"'
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
pypto = importlib.import_module("pypto")
core = importlib.import_module("pypto.pypto_core")
for module in (pypto, core):
    path = pathlib.Path(module.__file__).resolve()
    assert path.is_relative_to(root), path
assert not type(pypto.__loader__).__module__.startswith("_editable_skbc_")
assert len(list(root.joinpath("pypto").glob("pypto_core*.so"))) == 1
print(pypto.__file__)
print(core.__file__)
PY

cd "$src"
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$probe_py" -m pytest tests/ut -n "$PYPTO_TEST_JOBS" -q
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$probe_py" -m pytest \
  tests/ut/language/test_unified_ops.py::TestUnifiedSlicePadValue::test_symlinked_import_path_still_names_the_caller \
  -q
'
```

Expected wheel-backed full result: 10,177 passed, 58 skipped, zero failed.
The independent symlink probe must pass, not skip. Both `pypto.__file__` and
`pypto.pypto_core.__file__` must resolve beneath `probe-venv`; the loader must
not be `_editable_skbc_*`.

## Commit boundary

After all four gates pass, commit only the two CMake files. Do not include
submodule cache dirt, build outputs, logs, or the later TargetInfo candidate.
Then update evidence/checkpoint state before cherry-picking `9939b88`.
