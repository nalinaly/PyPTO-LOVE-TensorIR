from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import pathlib
import secrets
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool(name: str, path: pathlib.Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


contract = load_tool(
    "_pypto_nvidia_executable_sm120_contract",
    ROOT / "tools/_pypto_nvidia_executable_sm120_contract.py",
)
control_manifest = load_tool(
    "_pypto_nvidia_sm120_control_manifest",
    ROOT / "tools/_pypto_nvidia_sm120_control_manifest.py",
)
preflight = load_tool("smoke_test_preflight", ROOT / "tools/preflight.py")
sys.modules["preflight"] = preflight
stop_run = load_tool("smoke_test_stop_run", ROOT / "tools/stop_run.py")
sys.modules["stop_run"] = stop_run
run_isolated = load_tool("smoke_test_run_isolated", ROOT / "tools/run_isolated.py")
runner = load_tool(
    "smoke_test_runner",
    ROOT / "benchmarks/operators/pypto_nvidia_executable_sm120.py",
)
finalizer = load_tool(
    "smoke_test_finalizer",
    ROOT / "tools/finalize_pypto_nvidia_executable_sm120.py",
)


class SmokePayloadContractTest(unittest.TestCase):
    def test_control_manifest_is_required_before_gpu_launch(self) -> None:
        manifest = ROOT / control_manifest.MANIFEST_RELATIVE_PATH
        if manifest.is_file():
            identity = control_manifest.validate_control_manifest(ROOT)
            self.assertTrue(identity["root_clean"])
            self.assertEqual(
                [record["path"] for record in identity["files"]],
                list(control_manifest.CONTROL_PATHS),
            )
        else:
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "manifest is missing"
            ):
                control_manifest.validate_control_manifest(ROOT)

    def test_control_manifest_validates_real_git_ancestry_and_blob_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            repository = pathlib.Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Smoke Fixture"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "smoke@example.invalid"],
                cwd=repository,
                check=True,
            )
            original: dict[str, bytes] = {}
            for relative in control_manifest.CONTROL_PATHS:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"reviewed:{relative}\n".encode()
                original[relative] = payload
                path.write_bytes(payload)
                path.chmod(0o644)
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "implementation"],
                cwd=repository,
                check=True,
            )
            implementation = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            tree = subprocess.run(
                ["git", "rev-parse", "HEAD^{tree}"],
                cwd=repository,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            records = []
            for relative in control_manifest.CONTROL_PATHS:
                path = repository / relative
                records.append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": control_manifest.sha256_file(path),
                        "mode": path.stat().st_mode & 0o777,
                    }
                )
            manifest = {
                "schema_version": control_manifest.MANIFEST_SCHEMA_VERSION,
                "kind": control_manifest.MANIFEST_KIND,
                "implementation_commit": implementation,
                "implementation_tree": tree,
                "files": records,
            }
            manifest_path = repository / control_manifest.MANIFEST_RELATIVE_PATH
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_bytes(control_manifest.canonical_json(manifest))
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "manifest"],
                cwd=repository,
                check=True,
            )
            identity = control_manifest.validate_control_manifest(repository)
            self.assertEqual(identity["implementation_commit"], implementation)
            changed = repository / control_manifest.CONTROL_PATHS[0]
            changed.write_bytes(b"dirty")
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "not clean"
            ):
                control_manifest.validate_control_manifest(repository)
            changed.write_bytes(original[control_manifest.CONTROL_PATHS[0]])
            manifest["files"][0]["sha256"] = "0" * 64
            manifest_path.write_bytes(control_manifest.canonical_json(manifest))
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "tamper manifest"],
                cwd=repository,
                check=True,
            )
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "live control file differs"
            ):
                control_manifest.validate_control_manifest(repository)

    def test_importing_payload_adds_no_gpu_or_framework_module(self) -> None:
        path = ROOT / "benchmarks/operators/pypto_nvidia_executable_sm120.py"
        before = set(sys.modules)
        load_tool("smoke_test_runner_second_import", path)
        added = set(sys.modules) - before
        self.assertTrue(
            {"torch", "pypto", "triton", "sglang", "flashinfer"}.isdisjoint(added)
        )

    def test_fixed_command_and_runner_bytes_are_exact(self) -> None:
        command = contract.fixed_child_command(ROOT)
        self.assertEqual(command[1:4], ["-I", "-B", "-S"])
        self.assertEqual(command[-1], str(ROOT / contract.RUNNER_RELATIVE_PATH))
        path = ROOT / contract.RUNNER_RELATIVE_PATH
        self.assertEqual(path.stat().st_size, contract.RUNNER_SIZE)
        self.assertEqual(runner.sha256_file(path), contract.RUNNER_SHA256)

    def test_start_barrier_joins_gate_and_live_process(self) -> None:
        run_id = "pypto-20990101T000000Z-123456-abcdef"
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            workspace = pathlib.Path(directory)
            run_dir = workspace / "runs" / run_id
            run_dir.mkdir(parents=True)
            gate_path = run_dir / "gpu-smoke-gate.json"
            gate = {
                "pgid": os.getpgrp(),
                "pid": os.getpid(),
                "run_id": run_id,
                "schema": 1,
                "start_ticks": runner._process_start_ticks(os.getpid()),
            }
            gate_raw = runner.canonical_json(gate)
            gate_path.write_bytes(gate_raw)
            barrier_path = run_dir / "gpu-smoke-start-barrier.json"
            barrier = {
                "gate_path": str(gate_path),
                "gate_sha256": runner.sha256_bytes(gate_raw),
                "pgid": os.getpgrp(),
                "pid": os.getpid(),
                "run_id": run_id,
                "schema": 1,
                "start_ticks": runner._process_start_ticks(os.getpid()),
            }
            barrier_path.write_bytes(runner.canonical_json(barrier))
            with mock.patch.dict(
                os.environ,
                {"PYPTO_GPU_SMOKE_START_BARRIER": str(barrier_path)},
                clear=False,
            ):
                evidence = runner.wait_for_start_barrier(workspace, run_id)
            self.assertEqual(evidence, {"barrier": barrier, "gate": gate})

    def test_publication_is_read_only_and_no_replace(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            path = pathlib.Path(directory) / "evidence.bin"
            digest = runner.publish_no_replace(path, b"evidence")
            self.assertEqual(digest, runner.sha256_bytes(b"evidence"))
            self.assertEqual(path.stat().st_mode & 0o777, 0o444)
            with self.assertRaisesRegex(runner.SmokeError, "replace"):
                runner.publish_no_replace(path, b"other")


class GpuSmokeAdmissionTest(unittest.TestCase):
    def test_controller_and_finalizer_ignore_pythonpath_shadow_modules(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            shadow_root = pathlib.Path(directory)
            marker = shadow_root / "imported.txt"
            (shadow_root / "json.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
            )
            environment = {
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(shadow_root),
            }
            python = ROOT / "envs/pypto-nvidia/bin/python"
            for tool in (
                ROOT / "tools/run_isolated.py",
                ROOT / "tools/finalize_pypto_nvidia_executable_sm120.py",
            ):
                result = subprocess.run(
                    [str(python), "-E", "-B", "-S", str(tool), "--help"],
                    cwd=ROOT,
                    env=environment,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(marker.exists(), tool)

    def test_exact_environment_discards_ambient_compiler_and_plugin_controls(
        self,
    ) -> None:
        ambient = {
            "PYPTO_ALLOW_FALLBACK": "1",
            "PTOAS_ROOT": "/tmp/ambient",
            "PYPTO_INDUCTOR_CUDA_BACKEND": "triton",
            "PYTHONPATH": "/tmp/ambient",
            "SGLANG_PLUGINS": "ambient",
        }
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            with mock.patch.dict(run_isolated.os.environ, ambient, clear=False):
                environment = run_isolated.isolated_environment(
                    "fixture",
                    pathlib.Path(directory),
                    environment_prefix=run_isolated.ENVIRONMENTS["pypto-nvidia"],
                    framework_profile="pypto",
                    protected_zero_nvidia_gpu_smoke_requested=True,
                    exact_nvidia_smoke=True,
                )
        self.assertEqual(environment["PYPTO_ALLOW_FALLBACK"], "0")
        self.assertEqual(environment["PYPTO_STRICT_COVERAGE"], "1")
        self.assertEqual(environment["PYTHONPATH"], "")
        self.assertEqual(
            environment["SGLANG_PLUGINS"],
            "__pypto_exact_nvidia_smoke_no_plugins__",
        )
        self.assertNotIn("PTOAS_ROOT", environment)
        self.assertNotIn("PYPTO_INDUCTOR_CUDA_BACKEND", environment)

    def _main_report(
        self,
        *,
        compute_side_effect: object,
        protected: list[object] | None = None,
        runtime_mapping: tuple[list[int], list[int]] = ([], []),
    ) -> tuple[int, dict[str, object]]:
        protected = protected or []
        output = io.StringIO()
        argv = [
            "preflight.py",
            "--mode",
            "gpu-smoke",
            "--json",
            "--allow-protected-zero-nvidia-gpu-smoke",
        ]
        static_identity = {
            "version": contract.EXPECTED_TORCH_VERSION,
            "git_version": contract.EXPECTED_TORCH_GIT,
            "cuda": contract.EXPECTED_TORCH_CUDA,
            "hip": None,
            "forbidden_dsos": [],
        }
        compute_patch = (
            mock.patch.object(
                preflight, "nvidia_compute_pids", side_effect=compute_side_effect
            )
            if isinstance(compute_side_effect, BaseException)
            else mock.patch.object(
                preflight, "nvidia_compute_pids", return_value=compute_side_effect
            )
        )
        with (
            mock.patch.object(preflight.sys, "argv", argv),
            mock.patch.object(preflight.sys, "stdout", output),
            mock.patch.object(
                preflight,
                "nvidia_identity",
                return_value={
                    "compute_capability": "12.0",
                    "memory_mib": "24463",
                    "used_mib": "1024",
                    "driver": contract.EXPECTED_DRIVER_RELEASE,
                },
            ),
            mock.patch.object(
                preflight, "static_torch_identity", return_value=static_identity
            ),
            mock.patch.object(
                preflight,
                "process_table",
                return_value=(protected, protected, []),
            ),
            mock.patch.object(
                preflight,
                "protected_nvidia_runtime_mappings",
                return_value=runtime_mapping,
            ),
            mock.patch.object(
                preflight, "mem_available_kib", return_value=40 * 1024 * 1024
            ),
            compute_patch,
        ):
            result = preflight.main()
        return result, json.loads(output.getvalue())

    def test_compute_query_exception_fails_closed(self) -> None:
        result, report = self._main_report(
            compute_side_effect=RuntimeError("indeterminate")
        )
        self.assertEqual(result, 75)
        self.assertFalse(report["ok"])
        self.assertFalse(report["nvidia_compute_audit_ok"])
        self.assertTrue(any("cannot audit" in value for value in report["failures"]))

    def test_nonheavy_protected_nvidia_mapping_is_rejected(self) -> None:
        process = preflight.ProcessInfo(
            pid=10,
            ppid=1,
            start_ticks=100,
            rss_kib=1,
            command="python helper.py",
            cwd="/home/zhaosiying/zcode-lane",
        )
        result, report = self._main_report(
            compute_side_effect=set(),
            protected=[process],
            runtime_mapping=([10], []),
        )
        self.assertEqual(result, 75)
        self.assertFalse(report["ok"])
        self.assertEqual(report["protected_nvidia_runtime_mapping_pids"], [10])

    def test_marker_only_or_wrong_pgid_compute_is_external(self) -> None:
        root = preflight.ProcessInfo(
            pid=99,
            ppid=1,
            start_ticks=990,
            rss_kib=1,
            command="owned",
            cwd=str(ROOT),
        )
        foreign = preflight.ProcessInfo(
            pid=10,
            ppid=1,
            start_ticks=100,
            rss_kib=1,
            command="foreign",
            cwd=str(ROOT),
        )
        metadata = {"pid": 99, "pgid": 999, "start_ticks": 990}
        with mock.patch.object(
            run_isolated.os,
            "getpgid",
            side_effect=lambda pid: 999 if pid == 99 else 123,
        ):
            owned, external = run_isolated.partition_compute_pids(
                {10}, metadata, [root, foreign]
            )
        self.assertEqual(owned, [])
        self.assertEqual(external, [10])

    def test_correctness_runner_is_rejected_in_benchmark_mode(self) -> None:
        argv = [
            "run_isolated.py",
            "--mode",
            "gpu-benchmark",
            "--timeout-seconds",
            "1",
            "--",
            *contract.fixed_child_command(ROOT),
        ]
        with mock.patch.object(run_isolated.sys, "argv", argv):
            with self.assertRaises(SystemExit) as error:
                run_isolated.main()
        self.assertEqual(error.exception.code, 2)

    def test_exact_controller_rejects_site_enabled_python(self) -> None:
        argv = [
            "run_isolated.py",
            "--mode",
            "gpu-smoke",
            "--exact-pypto-nvidia-smoke",
            "--timeout-seconds",
            str(contract.GPU_SMOKE_TIMEOUT_SECONDS),
            "--minimum-free-disk-gib",
            str(contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB),
            "--environment",
            "pypto-nvidia",
            "--framework-profile",
            "pypto",
            "--environment-lock-mode",
            "shared",
            "--",
            *contract.fixed_child_command(ROOT),
        ]
        with mock.patch.object(run_isolated.sys, "argv", argv):
            with self.assertRaises(SystemExit) as error:
                run_isolated.main()
        self.assertEqual(error.exception.code, 2)


class SmokeFinalizerUnitTest(unittest.TestCase):
    def test_scope_rejects_performance_or_benchmark_fields(self) -> None:
        provisional = {
            "scope": {
                "provider": "pypto.tensorir",
                "runtime_object": "NvidiaExecutable",
                "operator_correctness": True,
                "model_forward": False,
                "strict_coverage_result": False,
                "performance_result": False,
                "cuda_graph_result": False,
            }
        }
        finalizer.validate_scope(provisional)
        provisional["runtime"] = {"latency_us": 1}
        with self.assertRaisesRegex(finalizer.FinalizeError, "performance-like"):
            finalizer.validate_scope(provisional)

    def test_duplicate_or_noncanonical_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            path = pathlib.Path(directory) / "value.json"
            path.write_text('{"a":1,"a":2}\n')
            with self.assertRaisesRegex(finalizer.FinalizeError, "duplicate"):
                finalizer.load_canonical(path, ROOT, "fixture")
            path.write_text('{"a": 1}\n')
            with self.assertRaisesRegex(finalizer.FinalizeError, "canonical"):
                finalizer.load_canonical(path, ROOT, "fixture")


class SmokeFinalizerFullFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"pypto-20990101T000000Z-{os.getpid()}-{secrets.token_hex(3)}"
        self.run_dir = ROOT / "runs" / self.run_id
        self.run_dir.mkdir()
        self.addCleanup(shutil.rmtree, self.run_dir, True)
        self.replay = contract.replay_directory(ROOT, self.run_id)
        self.replay.mkdir(mode=0o700)
        self.process_path = self.run_dir / "process.json"
        self.preflight_path = self.run_dir / "preflight.json"
        self.gate_path = self.run_dir / "gpu-smoke-gate.json"
        self.barrier_path = self.run_dir / "gpu-smoke-start-barrier.json"
        self.provisional_path = contract.provisional_path(ROOT, self.run_id)
        self.control_identity = {
            "manifest_path": "state/contracts/fixture.json",
            "manifest_bytes": 1,
            "manifest_sha256": "f" * 64,
            "implementation_commit": "1" * 40,
            "implementation_tree": "2" * 40,
            "current_head": "3" * 40,
            "current_tree": "4" * 40,
            "root_clean": True,
            "files": [],
        }
        self.protected_process = {
            "pid": 10,
            "ppid": 1,
            "start_ticks": 100,
            "rss_kib": 1,
            "command": "gem5.opt",
            "cwd": "/home/zhaosiying/zcode-lane",
        }
        self.gpu = {
            "name": contract.EXPECTED_DEVICE_NAME,
            "compute_capability": "12.0",
            "memory_mib": "24463",
            "used_mib": "1024",
            "driver": contract.EXPECTED_DRIVER_RELEASE,
        }
        self.static_identity = {
            "source": "static ENVIRONMENT.lock and selected-prefix file audit",
            "environment_lock_sha256": contract.ENVIRONMENT_LOCK_SHA256,
            "version": contract.EXPECTED_TORCH_VERSION,
            "git_version": contract.EXPECTED_TORCH_GIT,
            "cuda": contract.EXPECTED_TORCH_CUDA,
            "hip": None,
            "python_executable": str(
                (ROOT / contract.PYTHON_REAL_RELATIVE_PATH).resolve()
            ),
            "libcudart_path": str(
                (ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve()
            ),
            "libcudart_size": contract.CUDA_RUNTIME_SIZE,
            "libcudart_sha256": contract.CUDA_RUNTIME_SHA256,
            "libcudart_record_owned": True,
            "nvidia_runtime_mappings": [],
            "cuda_initialized": False,
            "forbidden_dsos": [],
        }

    def write_json(self, path: pathlib.Path, value: object, mode: int = 0o600) -> str:
        raw = finalizer.canonical_json(value)
        path.write_bytes(raw)
        path.chmod(mode)
        return finalizer.sha256_bytes(raw)

    def audit(self, owned: list[int], *, authorized: bool = True) -> dict[str, object]:
        return {
            "owned_nvidia_compute_pids": owned,
            "external_nvidia_compute_pids": [],
            "protected_nvidia_compute_pids": [],
            "protected_nvidia_runtime_mapping_pids": [],
            "unreadable_protected_maps": [],
            "protected_heavy_pids": [10] if authorized else [],
            "protected_cpu_lane_authorized": authorized,
            "free_memory_mib": 20_000,
            "gpu": self.gpu,
        }

    def build_fixture(
        self, *, authorized: bool = True
    ) -> tuple[dict[str, object], str, dict[str, object]]:
        protected = [self.protected_process] if authorized else []
        memory_floor = (24 if authorized else 32) * 1024 * 1024
        preflight = {
            "coexistence_policy_version": 1,
            "cwd": str(ROOT),
            "failures": [],
            "gpu": self.gpu,
            "gpu_smoke_free_memory_floor_mib": 4096,
            "gpu_smoke_policy_version": 1,
            "mem_available_kib": 40 * 1024 * 1024,
            "memory_floor_kib": memory_floor,
            "mode": "gpu-smoke",
            "nvidia_compute_audit_ok": True,
            "nvidia_compute_pids": [],
            "ok": True,
            "policy": "observation-only; no external process is ever signalled",
            "policy_version": 3,
            "protected_activity_waiver_applied": False,
            "protected_cpu_only_coexistence_requested": False,
            "protected_gpu_smoke_waiver_applied": authorized,
            "protected_heavy_processes": protected,
            "protected_nvidia_compute_pids": [],
            "protected_nvidia_runtime_mapping_pids": [],
            "protected_processes": protected,
            "protected_zero_nvidia_gpu_smoke_requested": authorized,
            "torch": self.static_identity,
            "unreadable_protected_maps": [],
            "workspace": str(ROOT),
            "workspace_processes": [],
        }
        preflight_sha = self.write_json(self.preflight_path, preflight)
        preflight_anchor = {
            "path": str(self.preflight_path),
            "sha256": preflight_sha,
        }
        pre_release = self.audit([], authorized=authorized)
        gate = {
            "schema": 1,
            "run_id": self.run_id,
            "pid": 99,
            "pgid": 99,
            "start_ticks": 990,
            "command": contract.fixed_child_command(ROOT),
            "preflight": preflight_anchor,
            "static_identity": self.static_identity,
            "control_manifest": self.control_identity,
            "runtime_isolation": pre_release,
        }
        gate_sha = self.write_json(self.gate_path, gate)
        barrier = {
            "schema": 1,
            "run_id": self.run_id,
            "pid": 99,
            "pgid": 99,
            "start_ticks": 990,
            "gate_path": str(self.gate_path),
            "gate_sha256": gate_sha,
        }
        barrier_sha = self.write_json(self.barrier_path, barrier)
        process = {
            "schema": 3,
            "run_id": self.run_id,
            "workspace": str(ROOT),
            "environment": str(ROOT / "envs/pypto-nvidia"),
            "environment_access_lock": {
                "path": str(ROOT / "runs/environment-pypto-nvidia.lock"),
                "mode": "shared",
                "device": 1,
                "inode": 2,
            },
            "framework_profile": "pypto",
            "framework_launch": False,
            "mode": "gpu-smoke",
            "coexistence": {
                "policy_version": 1,
                "requested": False,
                "waiver_applied": False,
                "memory_floor_kib": memory_floor,
                "protected_heavy_processes": protected,
                "protected_nvidia_compute_pids": [],
            },
            "gpu_smoke": {
                "policy_version": 1,
                "requested": authorized,
                "waiver_applied": authorized,
                "authorization": (
                    contract.GPU_SMOKE_AUTHORIZATION if authorized else None
                ),
                "start_barrier_path": str(self.barrier_path),
                "gate_path": str(self.gate_path),
                "memory_floor_kib": memory_floor,
                "gpu_free_memory_floor_mib": 4096,
                "protected_heavy_processes": protected,
                "protected_nvidia_compute_pids": [],
                "protected_nvidia_runtime_mapping_pids": [],
                "unreadable_protected_maps": [],
                "gate_sha256": gate_sha,
                "start_barrier_sha256": barrier_sha,
                "release_authorized_at": "20990101T000000Z",
            },
            "preflight": preflight_anchor,
            "resource_policy": {
                "timeout_seconds": 1800,
                "minimum_free_disk_bytes": 64 << 30,
                "owned_run_pause_memory_kib": 16 * 1024 * 1024,
            },
            "command": contract.fixed_child_command(ROOT),
            "pid": 99,
            "pgid": 99,
            "start_ticks": 990,
            "started_at": "20990101T000000Z",
            "status": "exited",
            "gpu_smoke_pre_release_audit": pre_release,
            "gpu_smoke_last_audit": self.audit([99], authorized=authorized),
            "gpu_smoke_post_exit_audit": self.audit([], authorized=authorized),
            "return_code": 0,
            "finished_at": "20990101T000100Z",
        }
        self.write_json(self.process_path, process)
        replay_files = []
        names = [
            "compile-request.msgpack",
            "static.build-spec.msgpack",
            "static.artifact.msgpack",
            "dynamic.build-spec.msgpack",
            "dynamic.artifact.msgpack",
            "scalar.build-spec.msgpack",
            "scalar.artifact.msgpack",
        ]
        for index, name in enumerate(names):
            path = self.replay / name
            payload = f"replay-{index}".encode()
            path.write_bytes(payload)
            path.chmod(0o444)
            replay_files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": len(payload),
                    "sha256": finalizer.sha256_bytes(payload),
                }
            )
        integrity_paths = {
            "contract": ROOT / "tools/_pypto_nvidia_executable_sm120_contract.py",
            "runner": ROOT / contract.RUNNER_RELATIVE_PATH,
            "environment_lock": ROOT / "ENVIRONMENT.lock",
            "versions_lock": ROOT / "VERSIONS.lock",
            "workspace_lock": ROOT / "WORKSPACE.lock",
            "pypto_dso": ROOT / contract.PYPTO_DSO_RELATIVE_PATH,
            "cuda_runtime": ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH,
        }
        integrity = {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": finalizer.sha256_file(path),
            }
            for name, path in integrity_paths.items()
        }
        traits = {
            "compute_capability": 120,
            "multiprocessor_count": 82,
            "warp_size": 32,
            "max_threads_per_block": 1024,
            "max_threads_per_multiprocessor": 1536,
            "max_blocks_per_multiprocessor": 24,
            "max_block_dim_x": 1024,
            "max_block_dim_y": 1024,
            "max_block_dim_z": 64,
            "max_grid_dim_x": (1 << 31) - 1,
            "max_grid_dim_y": 65535,
            "max_grid_dim_z": 65535,
            "l1_cache_line_bytes": 128,
            "default_shared_memory_per_cta_bytes": 48 * 1024,
            "max_shared_memory_per_cta_bytes": 101376,
            "shared_memory_per_multiprocessor_bytes": 131072,
            "registers_per_cta": 65536,
            "max_registers_per_thread": 255,
            "registers_per_multiprocessor": 65536,
            "l2_cache_size_bytes": 96 * 1024 * 1024,
            "total_global_memory_bytes": 24 * 1024 * 1024 * 1024,
        }
        artifacts = []
        executions = []
        entry_names = {
            "static": "add_op_simple",
            "dynamic": "add_dynamic_shape",
            "scalar": "add_tensor_scalar",
        }
        for index, case in enumerate(contract.CASE_SPECS):
            artifact_identity = f"{index + 1:x}" * 64
            artifacts.append(
                {
                    "case": case.name,
                    "source_sha256": "a" * 64,
                    "build_spec_identity_digest": "b" * 64,
                    "artifact_identity_digest": artifact_identity,
                    "cache_key_digest": "c" * 64,
                    "loader_compatibility_digest": "d" * 64,
                    "device_code_bytes": case.expected_device_code_bytes,
                    "device_code_sha256": case.expected_device_code_sha256,
                    "kernel_abi_identity_digest": "e" * 64,
                    "entry_function_name": entry_names[case.name],
                    "fallback_used": False,
                    "expected_grid": list(case.expected_grid),
                    "expected_kernel_arguments": case.expected_kernel_arguments,
                    "expected_device_code_bytes": case.expected_device_code_bytes,
                    "expected_device_code_sha256": case.expected_device_code_sha256,
                }
            )
            for repetition in range(2):
                executions.append(
                    {
                        "case": case.name,
                        "repetition": repetition,
                        "artifact_identity_digest": artifact_identity,
                        "dtype": case.dtype,
                        "shape": list(case.shape),
                        "strides": list(case.strides),
                        "grid": list(case.expected_grid),
                        "kernel_argument_count": case.expected_kernel_arguments,
                        "raw_current_stream": 100 + index,
                        "non_default_stream": True,
                        "external_stream_synchronized": True,
                        "expected_logical_bytes_sha256": "f" * 64,
                        "actual_logical_bytes_sha256": "f" * 64,
                        "input_bytes_sha256": "9" * 64,
                        "input_unchanged": True,
                        "torch_equal": True,
                        "padding_unchanged": True,
                        "packet_released_after_synchronization": True,
                        "explicit_unload": True,
                        "terminal_state": "Unloaded",
                        "bound_context_before_unload": 123,
                        "bound_context_id_before_unload": 456,
                        "bound_context_after_unload": 0,
                    }
                )
        compile_request = {
            "byte_identity_digest": "1" * 64,
            "loader_compatibility_input_digest": "2" * 64,
            "device_autotune_identity_digest": "3" * 64,
        }
        child_gate = {
            "static_identity": self.static_identity,
            "gpu": self.gpu,
            "free_memory_mib": 20_000,
            "protected_heavy_pids": [10] if authorized else [],
            "protected_runtime_pids": [],
            "unreadable_protected_maps": [],
            "nvidia_compute_pids": [],
            "control_manifest": self.control_identity,
        }
        provisional = {
            "schema_version": 1,
            "smoke": contract.SMOKE_NAME,
            "acceptance": "gpu-execution-complete-awaiting-run-finalization",
            "scope": {
                "provider": "pypto.tensorir",
                "runtime_object": "NvidiaExecutable",
                "operator_correctness": True,
                "model_forward": False,
                "strict_coverage_result": False,
                "performance_result": False,
                "cuda_graph_result": False,
            },
            "inputs": {
                "integrity": integrity,
                "pypto": {
                    "head": contract.PYPTO_HEAD,
                    "tree": contract.PYPTO_TREE,
                    "clean": True,
                },
                "tensor_ir_head": contract.TENSOR_IR_HEAD,
                "cuda_tile_head": contract.CUDA_TILE_HEAD,
                "llvm_head": contract.LLVM_HEAD,
                "replay_files": replay_files,
                "control_manifest": self.control_identity,
            },
            "run_context": {
                "run_id": self.run_id,
                "mode": "gpu-smoke",
                "pid": 99,
                "pgid": 99,
                "start_ticks": 990,
                "preflight": {
                    "path": self.preflight_path.relative_to(ROOT).as_posix(),
                    "sha256": preflight_sha,
                },
                "gate": {
                    "path": str(self.gate_path),
                    "sha256": gate_sha,
                    "document": gate,
                },
                "start_barrier_sha256": barrier_sha,
                "protected_zero_nvidia_policy": authorized,
            },
            "runtime": {
                "torch": {
                    "version": contract.EXPECTED_TORCH_VERSION,
                    "git_version": contract.EXPECTED_TORCH_GIT,
                    "cuda": contract.EXPECTED_TORCH_CUDA,
                    "hip": None,
                    "module_path": str(
                        (
                            ROOT / "envs/pypto-nvidia/lib/python3.14/"
                            "site-packages/torch/__init__.py"
                        ).resolve()
                    ),
                },
                "child_pre_cuda_gate": child_gate,
                "libcudart_paths": [
                    str((ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve())
                ],
                "observation": {
                    "device_ordinal": 0,
                    "device_name": contract.EXPECTED_DEVICE_NAME,
                    "device_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                    "pci_device_id": "0000:01:00.0",
                    "traits": traits,
                    "cuda_toolkit_version": contract.EXPECTED_CUDA_TOOLKIT_VERSION,
                    "cuda_driver_version": contract.EXPECTED_DRIVER_RELEASE,
                    "tensor_ir_revision": contract.TENSOR_IR_HEAD,
                    "cuda_tile_revision": contract.CUDA_TILE_HEAD,
                    "supported_compute_dtypes": list(
                        contract.EXPECTED_SUPPORTED_COMPUTE_DTYPES
                    ),
                    "cuda_driver_release_provenance": contract.EXPECTED_DRIVER_RELEASE,
                    "cuda_driver_api_version": 13000,
                    "cuda_runtime_api_version": 13000,
                    "cuda_runtime_library_path": str(
                        (ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve()
                    ),
                    "context_address": 123,
                    "context_id": 456,
                },
                "compile_request": compile_request,
                "artifacts": artifacts,
                "executions": executions,
                "case_order": [case.name for case in contract.CASE_SPECS],
                "repetitions_per_case": 2,
                "module_lifetimes": 6,
                "explicit_unloads": 6,
                "non_default_current_stream": True,
                "external_synchronization": True,
                "fallback_used": False,
                "forbidden_provider_imports": [],
            },
        }
        provisional_sha = self.write_json(
            self.provisional_path, provisional, mode=0o444
        )
        semantic_audit = {
            "command_sha256": "1" * 64,
            "stdout_sha256": "2" * 64,
            "compile_request": compile_request,
            "cases": [],
        }
        return provisional, provisional_sha, semantic_audit

    def test_complete_synthetic_finalization_and_unknown_claim_rejection(self) -> None:
        provisional, provisional_sha, semantic_audit = self.build_fixture()
        final_directory = pathlib.Path("runs") / self.run_id / "final"
        with (
            mock.patch.object(finalizer, "require_no_site_finalizer"),
            mock.patch.object(
                finalizer.control_manifest,
                "validate_control_manifest",
                return_value=self.control_identity,
            ),
            mock.patch.object(
                finalizer,
                "audit_replay_semantics",
                return_value=semantic_audit,
            ),
            mock.patch.object(
                finalizer.contract,
                "FINAL_REPORT_DIRECTORY",
                final_directory,
            ),
        ):
            report, output, digest = finalizer.finalize(
                workspace=ROOT,
                run_id=self.run_id,
                expected_provisional_sha256=provisional_sha,
            )
            self.assertEqual(report["status"], "accepted-real-sm120-correctness-smoke")
            self.assertEqual(digest, finalizer.sha256_file(output))
            provisional["runtime"]["qwen_correct"] = True
            self.provisional_path.chmod(0o600)
            tampered_sha = self.write_json(
                self.provisional_path, provisional, mode=0o444
            )
            with self.assertRaisesRegex(finalizer.FinalizeError, "key set differs"):
                finalizer.finalize(
                    workspace=ROOT,
                    run_id=self.run_id,
                    expected_provisional_sha256=tampered_sha,
                )

    def test_runtime_identity_requires_canonical_compute_dtype_order(self) -> None:
        provisional, _sha, _semantic_audit = self.build_fixture()
        preflight, _preflight_raw = finalizer.load_canonical(
            self.preflight_path, ROOT, "synthetic preflight"
        )
        gate, _gate_raw = finalizer.load_canonical(
            self.gate_path, ROOT, "synthetic gate"
        )
        finalizer.validate_runtime_identity(
            provisional, ROOT, preflight, gate, self.control_identity
        )
        invalid_values = (
            ["BF16", "FP32"],
            ["FP32"],
            ["FP32", "FP32"],
            ["FP32", "BF16", "FP16"],
            [52, 64],
            "FP32,BF16",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                candidate = copy.deepcopy(provisional)
                candidate["runtime"]["observation"]["supported_compute_dtypes"] = value
                with self.assertRaisesRegex(
                    finalizer.FinalizeError,
                    "live PyPTO runtime observation differs",
                ):
                    finalizer.validate_runtime_identity(
                        candidate, ROOT, preflight, gate, self.control_identity
                    )

    def test_execution_artifact_stream_and_context_mutations_are_rejected(self) -> None:
        provisional, _sha, _semantic = self.build_fixture()
        mutations = (
            ("artifact_identity_digest", "8" * 64, "Artifact join"),
            ("raw_current_stream", 1, "default stream"),
            ("bound_context_before_unload", 999, "context join"),
            ("bound_context_id_before_unload", 999, "context join"),
            ("input_unchanged", False, "prove correctness"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                candidate = copy.deepcopy(provisional)
                candidate["runtime"]["executions"][0][field] = value
                with self.assertRaisesRegex(finalizer.FinalizeError, message):
                    finalizer.validate_executions(candidate)

    def test_complete_exclusive_finalization_without_protected_lane(self) -> None:
        _provisional, provisional_sha, semantic_audit = self.build_fixture(
            authorized=False
        )
        final_directory = pathlib.Path("runs") / self.run_id / "exclusive-final"
        with (
            mock.patch.object(finalizer, "require_no_site_finalizer"),
            mock.patch.object(
                finalizer.control_manifest,
                "validate_control_manifest",
                return_value=self.control_identity,
            ),
            mock.patch.object(
                finalizer,
                "audit_replay_semantics",
                return_value=semantic_audit,
            ),
            mock.patch.object(
                finalizer.contract,
                "FINAL_REPORT_DIRECTORY",
                final_directory,
            ),
        ):
            report, _output, _digest = finalizer.finalize(
                workspace=ROOT,
                run_id=self.run_id,
                expected_provisional_sha256=provisional_sha,
            )
        self.assertTrue(report["run"]["zero_nvidia_interference"])

    def test_semantic_replay_rejects_target_info_drift(self) -> None:
        provisional, _sha, _semantic = self.build_fixture()
        runtime = provisional["runtime"]
        observation = runtime["observation"]
        target_fields = (
            "device_ordinal",
            "device_name",
            "device_uuid",
            "pci_device_id",
            "traits",
            "cuda_toolkit_version",
            "cuda_driver_version",
            "tensor_ir_revision",
            "cuda_tile_revision",
            "supported_compute_dtypes",
        )
        target = {name: copy.deepcopy(observation[name]) for name in target_fields}
        cases = []
        for artifact in runtime["artifacts"]:
            cases.append(
                {
                    name: artifact[name]
                    for name in (
                        "case",
                        "source_sha256",
                        "build_spec_identity_digest",
                        "artifact_identity_digest",
                        "cache_key_digest",
                        "loader_compatibility_digest",
                        "device_code_bytes",
                        "device_code_sha256",
                        "kernel_abi_identity_digest",
                        "entry_function_name",
                        "fallback_used",
                    )
                }
            )
        audited = {
            "compile_request": runtime["compile_request"],
            "target_info": target,
            "cases": cases,
        }
        completed = types.SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(audited, sort_keys=True, separators=(",", ":")) + "\n",
        )
        with mock.patch.object(finalizer.subprocess, "run", return_value=completed):
            finalizer.audit_replay_semantics(provisional, ROOT, self.run_id)
            for mutation in (
                ("device_uuid", "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"),
                ("pci_device_id", "0000:02:00.0"),
                ("traits", {**target["traits"], "max_threads_per_block": 512}),
                ("supported_compute_dtypes", ["BF16", "FP32"]),
            ):
                with self.subTest(field=mutation[0]):
                    candidate = copy.deepcopy(provisional)
                    candidate["runtime"]["observation"][mutation[0]] = mutation[1]
                    with self.assertRaisesRegex(
                        finalizer.FinalizeError, "TargetInfo differs"
                    ):
                        finalizer.audit_replay_semantics(candidate, ROOT, self.run_id)


if __name__ == "__main__":
    unittest.main()
