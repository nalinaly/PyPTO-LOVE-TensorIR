# PyPTO single-DSO acceptance gate

This is the zero-context runbook for the staged two-file transaction in
`projects/pypto/CMakeLists.txt` and
`projects/pypto/python/bindings/CMakeLists.txt`.

## Current status

The two-file source transaction passed all four gates and was committed as
PyPTO `e463bce7849b2239d0457dcae78ccf41c65ffa55`. Evidence is:

- native object build plus CTest 1/1, with corrected compile-database audit in
  run `pypto-20260823T233300Z-2314004-d79f10`;
- editable full suite: 10,176 passed and 59 skipped in
  `pypto-20260823T235818Z-2325491-f8dd3c`;
- fresh wheel SHA-256
  `bd6d24c9857a409df9d48c604bd329d10808cde354803ee10765680d252f1da1`,
  with corrected archive/DT_NEEDED audit in
  `pypto-20260823T234623Z-2320208-c9dba2`;
- clean install/import plus first full suite and symlink pass in
  `pypto-20260823T234644Z-2320294-5cf53c`; installed-DSO/JUnit
  `(10235, 0 failures, 0 errors, 57 skipped)` re-audit in
  `pypto-20260823T235200Z-2322804-e42ab7`.

The original native and wheel blocks completed their fresh build products but
returned nonzero because the old audit scripts respectively treated one
intentional C++ test compile as a duplicate product source and matched the
workspace directory name `tensor-ir` in an `ldd` path. The corrected audits
reuse those unchanged fresh artifacts and return zero; this recovery lineage is
intentional and must remain visible in evidence.

Every future invocation still requires a fresh heavy preflight. Never signal or
clean up a protected external lane.

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
export PYPTO_BUILD_JOBS=2
export PYPTO_TEST_JOBS=2
cd "$src"
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

"$py" - "$bld/compile_commands.json" "$src" <<'"'"'PY'"'"'
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2]).resolve()
native_rows = [
    row for row in rows
    if pathlib.Path(row["file"]).resolve().is_relative_to(root / "src")
]
native = [
    row for row in native_rows
    if "pypto_compiler_objects.dir" in row["command"]
]
native_test_rows = [
    row for row in native_rows
    if "pypto_compiler_objects.dir" not in row["command"]
]
bindings = [
    row for row in rows
    if pathlib.Path(row["file"]).resolve().is_relative_to(root / "python/bindings")
]
assert native and bindings
assert len({pathlib.Path(row["file"]).resolve() for row in native}) == len(native)
assert {pathlib.Path(row["file"]).resolve() for row in native_rows} == {
    pathlib.Path(row["file"]).resolve() for row in native
}
assert all("tests/ut/cpp/CMakeFiles/" in row["command"] for row in native_test_rows)
assert len({pathlib.Path(row["file"]).resolve() for row in bindings}) == len(bindings)
assert all("pypto_compiler_objects.dir" in row["command"] for row in native)
assert all("pypto_core.dir" in row["command"] for row in bindings)
assert all("/runtime/src/common" in row["command"] for row in bindings)
assert any(row["file"].endswith("src/ir/core.cpp") for row in native)
assert any(row["file"].endswith("python/bindings/modules/ir.cpp") for row in bindings)
PY
'
```

Expected: native CTest 1/1 passes. This build directory must not preexist.

## 2. Post-fix editable full rerun

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/projects/pypto
xml=$ws/builds/pypto-single-dso-native/editable-full-suite.junit.xml
log=$ws/logs/pypto-single-dso-editable-junit.log
py=$ws/envs/pypto-nvidia/bin/python
test ! -e "$xml"
test ! -e "$log"
exec > >(tee "$log") 2>&1
/usr/bin/env \
  PYPTO_BUILD_JOBS=2 \
  PYPTO_TEST_JOBS=2 \
  PYTEST_XDIST_AUTO_NUM_WORKERS=2 \
  PYTHONPATH="$src/python" \
  "$py" -m pytest "$src/tests/ut" -n 2 --junitxml="$xml" -q
"$py" - "$xml" <<'"'"'PY'"'"'
import pathlib, sys, xml.etree.ElementTree as ET

root = ET.parse(pathlib.Path(sys.argv[1])).getroot()
if root.tag == "testsuite":
    suite = root
else:
    assert root.tag == "testsuites", root.tag
    suites = root.findall("testsuite")
    assert len(suites) == 1, len(suites)
    suite = suites[0]
counts = tuple(int(suite.attrib[name]) for name in ("tests", "failures", "errors", "skipped"))
assert counts == (10235, 0, 0, 59), counts
print("editable_junit_counts", counts)
PY
'
```

Expected editable result is 10,176 passed and 59 skipped. The installed
console-script and symlink caller cases are the two expected editable-only
skips that become passes in the wheel gate. This rerun closes the historical
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
export PYPTO_BUILD_JOBS=2
export PYPTO_TEST_JOBS=2
cd "$src"
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
    assert len(names) == len(set(names)), "duplicate wheel members"
    assert all(not pathlib.PurePosixPath(n).is_absolute() and ".." not in pathlib.PurePosixPath(n).parts for n in names)
    assert not any(stat.S_ISLNK(x.external_attr >> 16) for x in infos)
    core = [n for n in names if n.startswith("pypto/pypto_core") and n.endswith(".so")]
    dsos = [
        n for n in names
        if pathlib.PurePosixPath(n).name.endswith(".so")
        or ".so." in pathlib.PurePosixPath(n).name
    ]
    assert len(core) == 1, core
    assert dsos == core, dsos
PY

mapfile -t native_so < <(find "$skbuild" -type f -name "pypto_core*.so")
test "${#native_so[@]}" -eq 1
if ! ldd_output=$(ldd "${native_so[0]}" 2>&1); then
  printf "%s\n" "$ldd_output" >&2
  exit 1
fi
if grep -Eiq "not found" <<<"$ldd_output"; then
  printf "%s\n" "$ldd_output" >&2
  exit 1
fi
if ! dynamic_output=$(readelf -d "${native_so[0]}" 2>&1); then
  printf "%s\n" "$dynamic_output" >&2
  exit 1
fi
needed=$(sed -n "s/.*Shared library: \[\([^]]*\)\].*/\1/p" <<<"$dynamic_output")
if grep -Eiq "libpypto|tensor.?ir|cuda.?tile|amdhip|hsa-runtime|gemsim" <<<"$needed"; then
  printf "forbidden DT_NEEDED entry:\n%s\n" "$needed" >&2
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
export PYPTO_BUILD_JOBS=2
export PYPTO_TEST_JOBS=2
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd "$src"
source "$src/.claude/skills/testing/load-env.sh"
exec > >(tee "$log") 2>&1
mapfile -t wheels < <(find "$root/wheels" -maxdepth 1 -type f -name "pypto-0.1.0-*.whl")
test "${#wheels[@]}" -eq 1
wheel=${wheels[0]}

"$py" -m venv --without-pip "$venv"
probe_py=$venv/bin/python
probe_site=$("$probe_py" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
"$py" -m pip --python "$probe_py" install \
  --no-deps --no-compile "$wheel"
test -x "$venv/bin/pypto-ir-trace"
mapfile -t installed_so < <(find "$probe_site/pypto" -maxdepth 1 -type f -name "pypto_core*.so")
test "${#installed_so[@]}" -eq 1
if ! ldd_output=$(ldd "${installed_so[0]}" 2>&1); then
  printf "%s\n" "$ldd_output" >&2
  exit 1
fi
if grep -Eiq "not found" <<<"$ldd_output"; then
  printf "%s\n" "$ldd_output" >&2
  exit 1
fi
if ! dynamic_output=$(readelf -d "${installed_so[0]}" 2>&1); then
  printf "%s\n" "$dynamic_output" >&2
  exit 1
fi
needed=$(sed -n "s/.*Shared library: \[\([^]]*\)\].*/\1/p" <<<"$dynamic_output")
if grep -Eiq "libpypto|tensor.?ir|cuda.?tile|amdhip|hsa-runtime|gemsim" <<<"$needed"; then
  printf "forbidden DT_NEEDED entry:\n%s\n" "$needed" >&2
  exit 1
fi

base_site=$("$py" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
printf "%s\n" "$base_site" > "$probe_site/base-environment-dependencies.pth"

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
full_xml=$root/full-suite.junit.xml
test ! -e "$full_xml"
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$probe_py" -m pytest tests/ut -n "$PYPTO_TEST_JOBS" \
    --junitxml="$full_xml" -q
"$probe_py" - "$full_xml" <<'"'"'PY'"'"'
import pathlib, sys, xml.etree.ElementTree as ET

root = ET.parse(pathlib.Path(sys.argv[1])).getroot()
if root.tag == "testsuite":
    suite = root
else:
    assert root.tag == "testsuites", root.tag
    suites = root.findall("testsuite")
    assert len(suites) == 1, len(suites)
    suite = suites[0]
counts = tuple(int(suite.attrib[name]) for name in ("tests", "failures", "errors", "skipped"))
assert counts == (10235, 0, 0, 57), counts
PY
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$probe_py" -m pytest \
  tests/ut/language/test_unified_ops.py::TestUnifiedSlicePadValue::test_symlinked_import_path_still_names_the_caller \
  -q
'
```

Expected wheel-backed full result: 10,178 passed, 57 skipped, zero failed.
Compared with editable mode, the installed console-script and symlink caller
cases both change from intentional skip to pass. The independent symlink probe
must pass, not skip. Both `pypto.__file__` and
`pypto.pypto_core.__file__` must resolve beneath `probe-venv`; the loader must
not be `_editable_skbc_*`.

## Commit boundary

After all four gates pass, commit only the two CMake files. Do not include
submodule cache dirt, build outputs, logs, or the later TargetInfo candidate.
Then update evidence/checkpoint state before cherry-picking `9939b88`.

Before committing, stage and verify the exact transaction:

```bash
cd /home/zhaosiying/pypto-love-tensor-ir/projects/pypto
git add -- CMakeLists.txt python/bindings/CMakeLists.txt
test "$(git diff --cached --name-only)" = \
  $'CMakeLists.txt\npython/bindings/CMakeLists.txt'
git diff --cached --check
```
