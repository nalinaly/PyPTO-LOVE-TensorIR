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
`Range` plus `If-Range`; a resumed response is accepted only when HTTP 206,
`Content-Range`, `Content-Length`, the resume offset, original total, strong
ETag and effective URL all agree exactly. The strong ETag and effective URL are
recorded in acquisition provenance. The partial file is size/hash verified and
fsynced before it is published inside the private materialization staging tree.

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
envs/pypto-nvidia/bin/python tools/materialize_triton_dependencies.py \
  --output "$src" --publish-reviewed-cache \
  --expected-manifest-sha256 "$manifest_sha" >/dev/null
test ! -e "$src"
test -d "$dst"
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

`tools/build_triton_wheel_offline.sh` is the source-controlled recipe. It
accepts only the manifest SHA-256 locked in `VERSIONS.lock`, rejects existing
build/log outputs, clears the framework profile's ambient `PYTHONPATH`, and
uses isolated build-venv probes that still load the reviewed
`reviewed-build-dependencies.pth`. Its build subprocess retains the
networkless minimal bubblewrap filesystem, producer hashes, resource limits,
post-build immutability checks, wheel audit, and fresh probe from this gate.
The PEP 517 frontend skips its distribution dependency check because reviewed
CMake and Ninja are intentionally supplied as exact executables, not Python
distributions. The exact seven reviewed Python distributions remain enforced.
The producer CMake command is an exact two-line exec-wrapper for the hash-pinned
payload in the read-only environment, so CMake resolves its adjacent reviewed
`share/cmake-3.31` data closure without loading the base Python environment;
copying the ELF payload alone is forbidden.
The build sets `CMAKE_SKIP_RPATH=ON`. The reviewed LLVM `FileCheck` input has
an otherwise unnecessary `$ORIGIN/../lib` RUNPATH, so the runner copies it to
`derived-llvm-tools/FileCheck` and removes that tag with the same hash-pinned
CMake payload plus `tools/remove_elf_rpath.cmake`. The original reviewed LLVM
tree remains unchanged. Source provenance binds the input, transform, tool,
and derived-output hashes, and the build sandbox overlays only that exact
derived file. The wheel auditor still rejects every RPATH/RUNPATH entry.
The three exact upstream example-plugin ELFs have `DT_NEEDED=libtriton.so`.
For only those paths, the isolated `ldd` audit sets
`LD_LIBRARY_PATH=/wheel/triton/_C` so resolution is proven against the wheel's
own ELF without reintroducing a RUNPATH or preloading libtriton into the audit
shell. Any other libtriton-dependent ELF, unresolved dependency, or external
resolution path is rejected.
After a successful build, the runner accepts only the seven expected regular
files in generated `python/triton.egg-info`, atomically retains that whole
directory as `generated-source-metadata`, retains the setuptools staging tree
as `generated-source-build`, and retains the exact generated compilation-
database symlink as `generated-compile-commands.json`. It then requires the
remaining built source to equal the reviewed build input exactly before wheel
audit. These generated paths remain build evidence; they are never silently
deleted to satisfy the source-tree gate.

Invoke the file directly through `run_isolated.py`; do not copy its body into
`bash -c` or another quoting layer:

```bash
envs/pypto-nvidia/bin/python tools/run_isolated.py \
  --mode heavy --allow-protected-cpu-only-coexistence \
  --timeout-seconds 14400 --minimum-free-disk-gib 128 \
  --run-id-file builds/triton-wheel-build-run-id.json \
  --environment pypto-nvidia --framework-profile pypto -- \
  /bin/bash /home/zhaosiying/pypto-love-tensor-ir/tools/build_triton_wheel_offline.sh \
    --reviewed-manifest-sha256 \
    29c0736211ba0b286acd562ba097d7f1dea989671003c63a7b988de5afb0fe7d
```

Both the run-id file and the canonical build/log outputs are no-replace. Use
new evidence names to preserve any failed attempt rather than moving or
overwriting them implicitly.

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
