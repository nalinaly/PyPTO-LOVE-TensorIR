# PyPTO CPU-only coexistence policy v2

This additive policy implements D-0016 without changing the accepted
`tools/preflight.py`, `tools/run_isolated.py`, or `tools/stop_run.py` bytes.
Those three sources, the original NVIDIA executable contract/control, and the
accepted v4 manifest are loaded or checked by exact path, size, and SHA-256.
They remain the owners of observation helpers, environment isolation, and
verified process-group signalling; the v2 controller never replaces their
module globals.

The v2 thresholds are exact:

- admission and resume: 22 GiB MemAvailable;
- pause: below 16 GiB MemAvailable;
- a running child between 16 and 22 GiB continues;
- a paused child remains paused until at least 22 GiB is available.

The controller is only for bounded, non-framework CPU-heavy commands. Its child
has `CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=void`, an empty
`PYTHONPATH`, and no SGLang plugin. GPU smoke, GPU benchmarks, model servers,
and framework launch commands are rejected. Shell/environment/session wrappers,
signal utilities, and commands naming protected amdgpu-sim/zcode paths are also
rejected. The argument vector is passed directly to `Popen` without a shell.

Command rejection is token-aware, not a plain substring match. The executable
is resolved through symlinks and `PATH` before its basename is checked and
before its resolved path joins the marker scan, so a renamed link to a
forbidden wrapper or GPU tool is still rejected; the deny set covers shells,
session/nice/taskset wrappers, signal utilities, common interpreter
mediators (`xargs`, `perl`, `node`, `busybox`, `find`), and the bare
`vllm`/`sglang`/`sglang_router`/`ray`/`deepspeed` CLI entry points. Any
`python*`/`pypy*` interpreter form rejects every argv inline-code spelling —
the separate `-c` token, the glued `-cCODE` form, bundled short options such
as `-Bc`/`-Sc` including glued code (`-Bcprint(...)`), and every stdin form
(bare `-` both before and after `--`, and clusters ending in `-` such as
`-u-`/`-B-`). Short-option clusters are decoded
character-wise with CPython's option-argument grammar: `-W`/`-X` and
`--check-hash-based-pycs` consume a glued or following value token that is
never counted as a file target, so `-W ignore`/`-X utf8` cannot smuggle a
stdin interpreter. Bundled `-m` works for every glued spelling
(`-Bm module`, `-mMODULE`, `-mray.scripts`), `-m` operands must be plain
dotted identifier module names, any module whose lowercased name starts with
`sglang`, `vllm`, `deepspeed`, `ray`, or `torch.distributed` is refused, a
trailing `-m` without an operand is refused, and an interpreter argv with
neither a positional file target nor a module operand is refused outright
(no bare interactive/stdin interpreter). Every marker is matched
case-insensitively against the space-joined command, the fully concatenated
argument text, and an alphanumeric-only flattening of both (including the
resolved executable path), so markers split across two arguments
(`run_engine` + `_lane.sh`, `gpu` + `benchmark`, `torch` + `distributed` +
`run`), whitespace- or dot-separated spellings, underscore or hyphen
spellings, renamed links, or different interpreter suffixes
(`python3 -m vllm`) cannot bypass the policy. The child environment passes
through no `PYTHON*` variables at all, so `PYTHONSTARTUP`/`PYTHONINSPECT`
cannot reopen an interactive code channel.
Argument tokens have leading dashes and `--opt=` prefixes stripped before
resolution, bare tokens are resolved against the workspace root, and any
token or resolved path resolving into one of the three protected roots
(`amdgpu-sim`, `amdgpu-sim-agentenv`, `zcode-lane`) or naming `amdgpu-sim` or
`zcode-lane` is refused.

The deny sets are an operator guardrail against accidental policy violations,
not a sandbox against a hostile operator: the command source is this
workspace's own reviewed tooling, a renamed copy of an interpreter or
executable cannot be caught by name, and position-blind option scanning
fail-closed by over-rejecting (for example a pytest `-c config.ini` flag, or
a `-m marker` selection whose marker text happens to start with a denied
prefix) rather than under-rejecting. Over-rejection is always the intended
direction.

Exit-code conventions: admission, watchdog policy, audit, ambiguity, and
unexpected-controller-failure refusals return 75; an interrupting parent signal
returns 130 (SIGINT) or 128+signum after verified cleanup; argparse usage
errors exit 2. Early refusals (after the run-id file exists) deliberately
retain the run directory and both preflight reports as diagnostic evidence;
they are never cleaned up by removal. An unexpected controller exception
prints its traceback and also exits 75. Post-manifest tests must run without
bytecode writes (`-B`/`PYTHONDONTWRITEBYTECODE=1`), because the control
validator rejects bytecode caches for the v2 sources; every documented
invocation form already does this.

Documented residual risks that this policy accepts without code changes:
argument paths are resolved once at admission, so a file swapped between
validation and child `execvpe` is bound only through the argv digest and the
CUDA-hidden environment; an owned descendant that calls `setsid()` leaves the
recorded process group and can then only be classified, never signalled; the
underlying accepted v1 `verify`-to-signal window and zombie-member ambiguity
behavior are inherited unchanged from the exact-hashed base primitives.

The focused tests are phase-aware: the missing-manifest and additive-source
tests hold before the canonical manifest is published, and after publication
the same tests validate the real manifest and the clean-root contract instead.
The suite therefore passes unchanged on both sides of the manifest-only
commit.

Before `Popen`, the controller validates a separate reviewed v2 manifest,
acquires the shared environment lease, runs an initial preflight and a second
action-boundary preflight, and records both canonical reports. Termination
signals are blocked across `Popen`, ownership capture, durable `process.json`
publication, and immediate `stop_run.verify`; the parent unmasks only after
that transaction completes. If initial metadata capture fails, capture,
publication, and verification are retried while signals remain blocked before
verified cleanup. Ownership that cannot be re-established is recorded as
ambiguous and is never signalled.

The living watchdog periodically rechecks host memory, disk, NVIDIA compute
processes, and protected-process NVIDIA mappings. Resource pressure or
protected NVIDIA activity pauses only the verified owned PGID. Owned NVIDIA
compute or timeout is terminal and terminates through a v2-owned staircase
(SIGTERM, then SIGCONT, a bounded wait with exact member re-audits, and
SIGSTOP of any survivor) built exclusively on the exact base signalling and
re-verification primitives. Resume sends `SIGCONT` only through
`stop_run.signal_verified`. No PID from an external or protected workload is
ever a signal target, and there is no kill escalation.
After child exit, final NVIDIA/protected-mapping and survivor audits are
recorded. Audit failure, protected/owned activity, survivors, or ambiguous
ownership returns 75.

The final manifest is deliberately absent during implementation. Until a clean
implementation commit is independently reviewed and a separate canonical
manifest is published, `validate_control_manifest` fails before lease
acquisition or child creation. This policy does not reinterpret EV-0035, alter
the 24 GiB v1 family, authorize GPU work, or establish compiler/model/
performance evidence.

After manifest publication, the fixed invocation form is:

```text
env -i PATH=/usr/bin:/bin \
  envs/pypto-nvidia/bin/python -E -B -S \
  tools/run_pypto_cpu_coexistence_v2_isolated.py \
  --run-id-file runs/cpu-v2-run-id.json \
  --timeout-seconds 3600 --minimum-free-disk-gib 64 -- COMMAND ARG...
```

Run focused CPU-only tests with no bytecode writes:

```text
PYTHONDONTWRITEBYTECODE=1 envs/pypto-nvidia/bin/python -B \
  tests/test_pypto_cpu_coexistence_v2.py
```
