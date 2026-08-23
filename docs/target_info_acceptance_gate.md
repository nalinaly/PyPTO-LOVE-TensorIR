# NVIDIA SM120 TargetInfo acceptance gate

This runbook applies only after every gate in
`docs/single_dso_acceptance_gate.md` passes, the two-file DSO transaction is
committed, and its checkpoint/evidence is durable. Candidate
`9939b885041b931284f7ce56ecbf888601b60b6e` remains source-reviewed and unbuilt
until then.

Every build/test block starts through `tools/run_isolated.py --mode heavy` and
therefore requires a fresh green protected-workload preflight. Never signal or
clean an external zcode/gem5 process.

## 1. Ordered integration transaction

The candidate and the pre-DSO branch share parent
`f550d4c33e8fef03e7dabcbf60c3db38b0f0a215`. A read-only 34-path patch check
against the staged DSO worktree passed. If the following invariants fail, stop
for state-drift review rather than resolving unrelated conflicts.

```bash
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/projects/pypto
candidate=9939b885041b931284f7ce56ecbf888601b60b6e
base=f550d4c33e8fef03e7dabcbf60c3db38b0f0a215

cd "$src"
test "$(git rev-parse "$candidate^")" = "$base"
test "$(git merge-base HEAD "$candidate")" = "$base"
git diff --quiet
git diff --cached --quiet

git cherry-pick "$candidate"
git diff --check HEAD^ HEAD
test -z "$(git diff --name-only --diff-filter=U)"

python - <<'PY'
from pathlib import Path

root = Path.cwd()
top = (root / "CMakeLists.txt").read_text()
bindings = (root / "python/bindings/CMakeLists.txt").read_text()
assert 'set(COMPILER_OBJECT_LIBRARY_NAME "${PROJECT_NAME}_compiler_objects")' in top
assert "src/compiler/target_info.cpp" in top
assert "add_library(${COMPILER_OBJECT_LIBRARY_NAME} OBJECT ${PYPTO_SOURCES})" in top
assert "modules/compiler.cpp" in bindings
module_call = bindings.split("nanobind_add_module(${LIBRARY_NAME}", 1)[1].split(")", 1)[0]
assert "${PYPTO_BINDING_SOURCES}" in module_call
assert "${PYPTO_SOURCES}" not in module_call
assert "${COMPILER_OBJECT_LIBRARY_NAME}" in bindings
PY
```

The expected result is one clean cherry-pick commit containing the complete
34-path candidate. Do not hand-copy only the CMake changes.

## 2. Fresh native build and C++ contracts

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/projects/pypto
bld=$ws/builds/pypto-target-info-native
log=$ws/logs/pypto-target-info-native.log
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
  --target pypto_compiler_objects \
    pypto_dsa_reuse_penalty_solver_test pypto_nvidia_target_info_test \
  --parallel "$PYPTO_BUILD_JOBS"
ctest --test-dir "$bld" --output-on-failure -j "$PYPTO_TEST_JOBS"

"$py" - "$bld/compile_commands.json" "$src" <<'"'"'PY'"'"'
import json, pathlib, sys

rows = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2]).resolve()
target_source = (root / "src/compiler/target_info.cpp").resolve()
binding_source = (root / "python/bindings/modules/compiler.cpp").resolve()
target_rows = [row for row in rows if pathlib.Path(row["file"]).resolve() == target_source]
binding_rows = [row for row in rows if pathlib.Path(row["file"]).resolve() == binding_source]
assert len(target_rows) == 2, target_rows
assert len(binding_rows) == 1, binding_rows
assert sum("pypto_compiler_objects.dir" in row["command"] for row in target_rows) == 1
assert sum("pypto_nvidia_target_info_test.dir" in row["command"] for row in target_rows) == 1
assert not any("pypto_core.dir" in row["command"] for row in target_rows)
assert "pypto_core.dir" in binding_rows[0]["command"]
assert "/runtime/src/common" in binding_rows[0]["command"]
PY
'
```

Expected CTest result is 2/2. This is native build evidence, not Python package
or CUDA runtime evidence.

## 3. Fresh wheel, one-DSO and dependency audit

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/projects/pypto
root=$ws/builds/pypto-target-info-wheel
out=$root/wheels
skbuild=$root/skbuild
log=$ws/logs/pypto-target-info-wheel.log
py=$ws/envs/pypto-nvidia/bin/python

test ! -e "$root"
test ! -e "$log"
mkdir -p "$out"
export PYPTO_BUILD_JOBS=2
export PYPTO_TEST_JOBS=2
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export SKBUILD_BUILD_DIR="$skbuild"
cd "$src"
source "$src/.claude/skills/testing/load-env.sh"
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

with zipfile.ZipFile(pathlib.Path(sys.argv[1])) as zf:
    infos = zf.infolist()
    names = [info.filename for info in infos]
    assert len(names) == len(set(names)), "duplicate wheel members"
    assert all(not pathlib.PurePosixPath(name).is_absolute() for name in names)
    assert all(".." not in pathlib.PurePosixPath(name).parts for name in names)
    assert not any(stat.S_ISLNK(info.external_attr >> 16) for info in infos)
    core = [name for name in names if name.startswith("pypto/pypto_core") and name.endswith(".so")]
    dsos = [
        name for name in names
        if pathlib.PurePosixPath(name).name.endswith(".so")
        or ".so." in pathlib.PurePosixPath(name).name
    ]
    assert len(core) == 1, core
    assert dsos == core, dsos
    assert "pypto/pypto_core/compiler.pyi" in names
    assert "pypto/compiler/__init__.py" in names
PY

mapfile -t native_so < <(find "$skbuild" -type f -name "pypto_core*.so")
test "${#native_so[@]}" -eq 1
if ! ldd_output=$(ldd "${native_so[0]}" 2>&1); then
  printf "%s\n" "$ldd_output" >&2
  exit 1
fi
if grep -Eiq \
  "not found|libpypto|tensor.?ir|cuda.?tile|amdhip|hsa-runtime|gemsim" \
  <<<"$ldd_output"; then
  printf "%s\n" "$ldd_output" >&2
  exit 1
fi
'
```

This wheel is new TargetInfo evidence. The earlier single-DSO wheel must not be
reused.

## 4. Clean install, targeted API contracts and full regression

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash -c '
set -euo pipefail
ws=/home/zhaosiying/pypto-love-tensor-ir
src=$ws/projects/pypto
root=$ws/builds/pypto-target-info-wheel
venv=$root/probe-venv
log=$ws/logs/pypto-target-info-wheel-tests.log
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
"$py" -m pip --python "$probe_py" install --no-deps --no-compile "$wheel"
test -x "$venv/bin/pypto-ir-trace"
mapfile -t installed_so < <(find "$probe_site/pypto" -maxdepth 1 -type f -name "pypto_core*.so")
test "${#installed_so[@]}" -eq 1
if ! ldd_output=$(ldd "${installed_so[0]}" 2>&1); then
  printf "%s\n" "$ldd_output" >&2
  exit 1
fi
if grep -Eiq \
  "not found|libpypto|tensor.?ir|cuda.?tile|amdhip|hsa-runtime|gemsim" \
  <<<"$ldd_output"; then
  printf "%s\n" "$ldd_output" >&2
  exit 1
fi
base_site=$("$py" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")
printf "%s\n" "$base_site" > "$probe_site/base-environment-dependencies.pth"

env -u PYTHONPATH PYTHONNOUSERSITE=1 "$probe_py" -I - "$probe_site" <<'"'"'PY'"'"'
import importlib, pathlib, sys

root = pathlib.Path(sys.argv[1]).resolve()
pypto = importlib.import_module("pypto")
compiler = importlib.import_module("pypto.compiler")
core = importlib.import_module("pypto.pypto_core")
for module in (pypto, compiler, core):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root)
assert not type(pypto.__loader__).__module__.startswith("_editable_skbc_")
assert compiler.is_supported_nvidia_compute_capability(120)
assert not compiler.is_supported_nvidia_compute_capability(100)
assert len(list(root.joinpath("pypto").glob("pypto_core*.so"))) == 1
PY

target_xml=$root/target-info.junit.xml
full_xml=$root/full-suite.junit.xml
test ! -e "$target_xml"
test ! -e "$full_xml"
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$probe_py" -m pytest tests/ut/compiler/test_target_info.py \
    --junitxml="$target_xml" -q
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$probe_py" -m pytest \
    tests/ut/ir/test_compiled_program.py \
    tests/ut/tools/test_memory_map.py -q
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$probe_py" -m pytest tests/ut -n "$PYPTO_TEST_JOBS" \
    --junitxml="$full_xml" -q
"$probe_py" - "$target_xml" "$full_xml" <<'"'"'PY'"'"'
import pathlib, sys, xml.etree.ElementTree as ET

def counts(path):
    root = ET.parse(pathlib.Path(path)).getroot()
    if root.tag == "testsuite":
        suite = root
    else:
        assert root.tag == "testsuites", root.tag
        suites = root.findall("testsuite")
        assert len(suites) == 1, len(suites)
        suite = suites[0]
    return tuple(int(suite.attrib[name]) for name in ("tests", "failures", "errors", "skipped"))

assert counts(sys.argv[1]) == (31, 0, 0, 0), counts(sys.argv[1])
assert counts(sys.argv[2]) == (10266, 0, 0, 58), counts(sys.argv[2])
PY
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$probe_py" -m pytest \
    tests/ut/language/test_unified_ops.py::TestUnifiedSlicePadValue::test_symlinked_import_path_still_names_the_caller \
    -q
'
```

Expected TargetInfo unit result is 31 passed. Based on the accepted DSO wheel
count plus those 31 new cases, the expected full wheel result is 10,208 passed
and 58 skipped; any collection/count drift must be explained before acceptance.
The independent symlink case passes, not skips.

## Acceptance boundary

Record the cherry-pick result SHA, native and wheel logs, CTest/Python counts,
wheel digest, DSO/dependency audit and clean import paths in evidence. TargetInfo
acceptance still does not prove TensorIR build composition, CompileRequest,
CUDA compilation/launch or performance. The next code transaction is the
data-only CompileRequest contract described in
`docs/compile_request_artifact_design.md`.
