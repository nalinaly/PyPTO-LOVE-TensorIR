from __future__ import annotations

import base64
import hashlib
import io
import importlib.util
import json
import pathlib
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stop_run = load_tool("stop_run")
sys.modules["stop_run"] = stop_run
preflight = load_tool("preflight")
sys.modules["preflight"] = preflight
run_isolated = load_tool("run_isolated")
audit_environment = load_tool("audit_python_environment")
triton_dependencies = load_tool("materialize_triton_dependencies")
import_models = load_tool("import_models")


class IsolationEnvironmentTest(unittest.TestCase):
    def make_environment(self, profile: str, environment: str) -> dict[str, str]:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            return run_isolated.isolated_environment(
                "test-run",
                pathlib.Path(directory),
                environment_prefix=run_isolated.ENVIRONMENTS[environment],
                framework_profile=profile,
            )

    def test_pypto_profile_contains_only_workspace_sources(self) -> None:
        environment = self.make_environment("pypto", "pypto-nvidia")
        paths = environment["PYTHONPATH"].split(":")
        self.assertIn(str(ROOT / "projects" / "pypto-kernels" / "src"), paths)
        self.assertIn(str(ROOT / "upstream" / "sglang" / "python"), paths)
        self.assertEqual(environment["SGLANG_PLUGINS"], "pypto")
        self.assertEqual(environment["PYPTO_FRAMEWORK_PROFILE"], "pypto")

    def test_baseline_profile_excludes_every_pypto_project(self) -> None:
        environment = self.make_environment(
            "baseline", "sglang-baseline-py312"
        )
        self.assertEqual(
            environment["PYTHONPATH"],
            str(ROOT / "upstream" / "sglang" / "python"),
        )
        self.assertEqual(
            environment["SGLANG_PLUGINS"], "__pypto_baseline_no_plugins__"
        )
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(
            environment["PYPTO_ENV_PREFIX"],
            str(ROOT / "envs" / "sglang-baseline-py312"),
        )

    def test_amd_and_simulator_environment_is_removed(self) -> None:
        protected = {
            "ROCM_PATH": "/bad",
            "HIP_VISIBLE_DEVICES": "0",
            "HSA_TEST": "bad",
            "ROCR_TEST": "bad",
            "GEMSIM_TEST": "bad",
            "LLVM_SYSPATH": "/home/zhaosiying/.triton/external-llvm",
            "JSON_SYSPATH": "/external/json",
            "TRITON_OFFLINE_BUILD": "1",
            "TRITON_BUILD_WITH_CLANG_LLD": "1",
        }
        original = {name: run_isolated.os.environ.get(name) for name in protected}
        try:
            run_isolated.os.environ.update(protected)
            environment = self.make_environment("pypto", "pypto-nvidia")
        finally:
            for name, value in original.items():
                if value is None:
                    run_isolated.os.environ.pop(name, None)
                else:
                    run_isolated.os.environ[name] = value
        self.assertTrue(set(protected).isdisjoint(environment))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("triton-dev", environment["PATH"])
        self.assertEqual(
            environment["CONDA_PREFIX"],
            str(ROOT / "envs" / "pypto-nvidia"),
        )

    def test_all_mutable_build_and_runtime_caches_are_workspace_owned(self) -> None:
        environment = self.make_environment("pypto", "pypto-nvidia")
        for name in (
            "CCACHE_DIR",
            "CONDA_PKGS_DIRS",
            "CUDA_CACHE_PATH",
            "HF_HOME",
            "PIP_CACHE_DIR",
            "PYPTO_CACHE_DIR",
            "PYPTO_PROG_BUILD_DIR",
            "SGLANG_CACHE_DIR",
            "TMPDIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "TRITON_HOME",
            "UV_CACHE_DIR",
            "XDG_CACHE_HOME",
        ):
            path = pathlib.Path(environment[name]).resolve()
            self.assertTrue(
                path == ROOT or ROOT in path.parents,
                f"{name} escaped the workspace: {path}",
            )

    def test_cpu_only_coexistence_marker_is_explicit_not_ambient(self) -> None:
        original = run_isolated.os.environ.get(
            "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED"
        )
        try:
            run_isolated.os.environ[
                "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED"
            ] = "ambient-must-not-pass"
            default = self.make_environment("pypto", "pypto-nvidia")
            with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
                authorized = run_isolated.isolated_environment(
                    "test-run",
                    pathlib.Path(directory),
                    environment_prefix=run_isolated.ENVIRONMENTS[
                        "pypto-nvidia"
                    ],
                    framework_profile="pypto",
                    protected_cpu_only_coexistence_requested=True,
                )
        finally:
            if original is None:
                run_isolated.os.environ.pop(
                    "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED", None
                )
            else:
                run_isolated.os.environ[
                    "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED"
                ] = original
        self.assertNotIn(
            "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED", default
        )
        self.assertEqual(
            authorized["PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED"], "1"
        )
        self.assertEqual(authorized["CUDA_VISIBLE_DEVICES"], "")

    def test_candidate_controls_are_exact_and_baseline_discards_them(self) -> None:
        controls = {
            "PTO_BACKTRACE": "1",
            "PYPTO_ALLOW_FALLBACK": "0",
            "PYPTO_INDUCTOR_CUDA_BACKEND": "pypto",
            "PYPTO_STRICT_COVERAGE": "1",
            "PYPTO_VERIFY_LEVEL": "roundtrip",
        }
        original = {name: run_isolated.os.environ.get(name) for name in controls}
        try:
            run_isolated.os.environ.update(controls)
            candidate = self.make_environment("pypto", "pypto-nvidia")
            baseline = self.make_environment(
                "baseline", "sglang-baseline-py312"
            )
        finally:
            for name, value in original.items():
                if value is None:
                    run_isolated.os.environ.pop(name, None)
                else:
                    run_isolated.os.environ[name] = value
        for name, value in controls.items():
            self.assertEqual(candidate[name], value)
            self.assertNotIn(name, baseline)

    def test_external_path_controls_fail_closed(self) -> None:
        original = run_isolated.os.environ.get("PTOAS_ROOT")
        try:
            run_isolated.os.environ["PTOAS_ROOT"] = "/opt/external-ptoas"
            with self.assertRaisesRegex(ValueError, "below the workspace"):
                self.make_environment("pypto", "pypto-nvidia")
        finally:
            if original is None:
                run_isolated.os.environ.pop("PTOAS_ROOT", None)
            else:
                run_isolated.os.environ["PTOAS_ROOT"] = original

    def test_unknown_direct_profile_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            with self.assertRaisesRegex(ValueError, "unknown framework profile"):
                run_isolated.isolated_environment(
                    "test-run",
                    pathlib.Path(directory),
                    environment_prefix=run_isolated.ENVIRONMENTS["pypto-nvidia"],
                    framework_profile="unknown",
                )

    def test_direct_profile_environment_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            with self.assertRaisesRegex(ValueError, "requires environment"):
                run_isolated.isolated_environment(
                    "test-run",
                    pathlib.Path(directory),
                    environment_prefix=run_isolated.ENVIRONMENTS[
                        "sglang-baseline-py312"
                    ],
                    framework_profile="pypto",
                )


class EnvironmentTransactionLockTest(unittest.TestCase):
    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile(
            dir=ROOT / "runs",
            prefix="test-environment-lock-",
            delete=False,
        )
        handle.close()
        self.lock_path = pathlib.Path(handle.name)
        self.lock_path.unlink()
        self.mapping = mock.patch.dict(
            run_isolated.ENVIRONMENT_TRANSACTION_LOCKS,
            {"pypto-nvidia": self.lock_path},
        )
        self.mapping.start()

    def tearDown(self) -> None:
        self.mapping.stop()
        self.lock_path.unlink(missing_ok=True)

    def test_shared_consumers_coexist_and_block_exclusive_transaction(self) -> None:
        first = run_isolated.acquire_environment_lock(
            "pypto-nvidia", "shared"
        )
        second = run_isolated.acquire_environment_lock(
            "pypto-nvidia", "shared"
        )
        try:
            with self.assertRaises(run_isolated.EnvironmentLockBusy):
                run_isolated.acquire_environment_lock(
                    "pypto-nvidia", "exclusive"
                )
        finally:
            second.close()
            first.close()

    def test_exclusive_transaction_blocks_new_consumer_then_releases(self) -> None:
        exclusive = run_isolated.acquire_environment_lock(
            "pypto-nvidia", "exclusive"
        )
        try:
            with self.assertRaises(run_isolated.EnvironmentLockBusy):
                run_isolated.acquire_environment_lock(
                    "pypto-nvidia", "shared"
                )
        finally:
            exclusive.close()
        shared = run_isolated.acquire_environment_lock(
            "pypto-nvidia", "shared"
        )
        shared.close()

    def test_parent_close_keeps_lock_until_inherited_child_fd_closes(self) -> None:
        lease = run_isolated.acquire_environment_lock(
            "pypto-nvidia", "exclusive"
        )
        ready_read, ready_write = run_isolated.os.pipe()
        release_read, release_write = run_isolated.os.pipe()
        child = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import os,sys; "
                    "os.write(int(sys.argv[1]), b'1'); "
                    "os.read(int(sys.argv[2]), 1)"
                ),
                str(ready_write),
                str(release_read),
            ],
            pass_fds=(lease.descriptor, ready_write, release_read),
        )
        run_isolated.os.close(ready_write)
        run_isolated.os.close(release_read)
        try:
            self.assertEqual(run_isolated.os.read(ready_read, 1), b"1")
            lease.close()
            with self.assertRaises(run_isolated.EnvironmentLockBusy):
                run_isolated.acquire_environment_lock(
                    "pypto-nvidia", "shared"
                )
            run_isolated.os.write(release_write, b"1")
            self.assertEqual(child.wait(timeout=5), 0)
            shared = run_isolated.acquire_environment_lock(
                "pypto-nvidia", "shared"
            )
            shared.close()
        finally:
            lease.close()
            run_isolated.os.close(ready_read)
            run_isolated.os.close(release_write)
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

    def test_registered_guard_releases_lock_while_exception_is_retained(self) -> None:
        lease = run_isolated.acquire_environment_lock(
            "pypto-nvidia", "exclusive"
        )

        @run_isolated.close_registered_environment_lock
        def fail_after_registration() -> None:
            run_isolated._ACTIVE_ENVIRONMENT_LOCK = lease
            raise RuntimeError("retained fixture")

        retained: BaseException | None = None
        try:
            fail_after_registration()
        except RuntimeError as error:
            retained = error
        self.assertIsNotNone(retained)
        replacement = run_isolated.acquire_environment_lock(
            "pypto-nvidia", "exclusive"
        )
        replacement.close()

    def test_markers_bind_fd_inode_mode_and_controller_identity(self) -> None:
        lease = run_isolated.acquire_environment_lock(
            "pypto-nvidia", "exclusive"
        )
        try:
            markers = run_isolated.environment_lock_markers(lease)
            self.assertEqual(
                markers["PYPTO_ENVIRONMENT_LOCK_FD"],
                str(lease.descriptor),
            )
            self.assertEqual(
                markers["PYPTO_ENVIRONMENT_LOCK_MODE"], "exclusive"
            )
            self.assertEqual(
                markers["PYPTO_ENVIRONMENT_LOCK_PATH"], str(self.lock_path)
            )
            self.assertEqual(
                markers["PYPTO_ENVIRONMENT_LOCK_DEV"], str(lease.device)
            )
            self.assertEqual(
                markers["PYPTO_ENVIRONMENT_LOCK_INO"], str(lease.inode)
            )
            self.assertEqual(
                markers["PYPTO_ENVIRONMENT_LOCK_CONTROLLER_PID"],
                str(run_isolated.os.getpid()),
            )
            self.assertEqual(
                markers["PYPTO_ENVIRONMENT_LOCK_CONTROLLER_START_TICKS"],
                str(run_isolated.process_start_ticks(run_isolated.os.getpid())),
            )
        finally:
            lease.close()

    def test_exclusive_command_is_restricted_to_mutating_replacement(self) -> None:
        python = run_isolated.ENVIRONMENTS["pypto-nvidia"] / "bin/python"
        replacement = ROOT / "tools/replace_triton_environment.py"
        run_isolated.validate_exclusive_environment_command(
            [str(python), "-B", str(replacement), "--apply"],
            run_isolated.ENVIRONMENTS["pypto-nvidia"],
        )
        for command in (
            [str(python), "-B", str(replacement), "--plan"],
            [str(python), "-B", str(replacement), "--apply", "--recover"],
            [str(python), "-B", str(ROOT / "tools/preflight.py"), "--apply"],
            ["/usr/bin/python3", "-B", str(replacement), "--apply"],
            [
                str(python),
                "-c",
                "raise SystemExit('bypass')",
                str(replacement),
                "--apply",
            ],
            [str(python), "-m", "arbitrary", str(replacement), "--apply"],
            [
                str(python),
                "-B",
                str(replacement),
                str(ROOT / "tools/preflight.py"),
                "--apply",
            ],
        ):
            with self.assertRaises(ValueError):
                run_isolated.validate_exclusive_environment_command(
                    command,
                    run_isolated.ENVIRONMENTS["pypto-nvidia"],
                )


class ProtectedProcessClassificationTest(unittest.TestCase):
    def test_cwd_binds_a_process_to_protected_scope(self) -> None:
        self.assertTrue(
            preflight.belongs_to_roots(
                "python worker.py",
                "/home/zhaosiying/zcode-lane/run",
                preflight.PROTECTED_ROOTS,
            )
        )

    def test_framework_module_launches_are_heavy(self) -> None:
        for command in (
            "python -m sglang.launch_server --model x",
            "python -m vllm.entrypoints.openai.api_server --model x",
            "python examples/vllm/qwen35_inference.py",
        ):
            self.assertTrue(preflight.is_heavy_command(command), command)

    def test_unrelated_sleep_is_not_heavy(self) -> None:
        self.assertFalse(preflight.is_heavy_command("/usr/bin/sleep 30"))

    @staticmethod
    def process(
        pid: int,
        ppid: int,
        command: str,
        cwd: str = "/home/zhaosiying/zcode-lane",
    ) -> object:
        return preflight.ProcessInfo(
            pid=pid,
            ppid=ppid,
            start_ticks=pid * 10,
            rss_kib=1,
            command=command,
            cwd=cwd,
        )

    def test_cpu_only_coexistence_waives_only_protected_activity(self) -> None:
        protected = [self.process(10, 1, "gem5.opt")]
        default = preflight.heavy_policy_failures(
            mode="heavy",
            coexistence_authorized=False,
            protected_heavy=protected,
            available_kib=40 * 1024 * 1024,
            protected_nvidia_compute_pids=[],
        )
        self.assertTrue(any("protected zcode" in value for value in default))
        coexistence = preflight.heavy_policy_failures(
            mode="heavy",
            coexistence_authorized=True,
            protected_heavy=protected,
            available_kib=40 * 1024 * 1024,
            protected_nvidia_compute_pids=[],
        )
        self.assertEqual(coexistence, [])

    def test_cpu_only_coexistence_keeps_memory_and_nvidia_failures(self) -> None:
        protected = [self.process(10, 1, "gem5.opt")]
        failures = preflight.heavy_policy_failures(
            mode="heavy",
            coexistence_authorized=True,
            protected_heavy=protected,
            available_kib=23 * 1024 * 1024,
            protected_nvidia_compute_pids=[10],
        )
        self.assertTrue(any("NVIDIA" in value for value in failures))
        self.assertTrue(any("24 GiB safety floor" in value for value in failures))

    def test_protected_lane_closure_includes_descendants_not_unrelated(self) -> None:
        supervisor = self.process(10, 1, "run_model_lane.sh")
        child = self.process(11, 10, "python worker.py")
        grandchild = self.process(12, 11, "gem5.opt")
        unrelated = self.process(20, 1, "agentenv server")
        selected = preflight.protected_lane_processes(
            [supervisor, child, grandchild, unrelated],
            [supervisor, grandchild],
        )
        self.assertEqual({process.pid for process in selected}, {10, 11, 12})

    def test_protected_root_seed_captures_heavy_child_after_cwd_change(self) -> None:
        supervisor = self.process(10, 1, "python lane.py")
        child = self.process(11, 10, "gem5.opt", cwd="/tmp")
        unrelated = self.process(20, 1, "gem5.opt", cwd="/tmp")
        protected = preflight.process_descendant_closure(
            [supervisor, child, unrelated], [supervisor]
        )
        self.assertEqual({process.pid for process in protected}, {10, 11})
        self.assertEqual(
            [process.pid for process in protected if preflight.is_heavy_command(process.command)],
            [11],
        )


class ModelImportCoexistenceTest(unittest.TestCase):
    @staticmethod
    def protected_process() -> object:
        return preflight.ProcessInfo(
            pid=4242,
            ppid=1,
            start_ticks=100,
            rss_kib=1,
            command="gem5.opt --cpu-only",
            cwd="/home/zhaosiying/amdgpu-sim",
        )

    def test_default_model_copy_boundary_rejects_protected_heavy(self) -> None:
        protected = self.protected_process()
        with mock.patch.object(
            import_models,
            "process_table",
            return_value=([protected], [protected], []),
        ):
            with self.assertRaisesRegex(RuntimeError, "protected workload"):
                import_models.ensure_model_copy_boundary_safe()

    def test_authorized_cpu_only_model_copy_boundary_accepts_lane(self) -> None:
        protected = self.protected_process()
        with mock.patch.object(
            import_models,
            "process_table",
            return_value=([protected], [protected], []),
        ), mock.patch.object(
            import_models,
            "mem_available_kib",
            return_value=40 * 1024 * 1024,
        ), mock.patch.object(
            import_models,
            "nvidia_compute_pids",
            return_value=set(),
        ):
            import_models.ensure_model_copy_boundary_safe(
                allow_protected_cpu_only_coexistence=True
            )

    def test_authorized_model_copy_keeps_memory_and_gpu_gates(self) -> None:
        protected = self.protected_process()
        process_table = ([protected], [protected], [])
        with mock.patch.object(
            import_models,
            "process_table",
            return_value=process_table,
        ), mock.patch.object(
            import_models,
            "mem_available_kib",
            return_value=23 * 1024 * 1024,
        ):
            with self.assertRaisesRegex(RuntimeError, "memory floor"):
                import_models.ensure_model_copy_boundary_safe(
                    allow_protected_cpu_only_coexistence=True
                )
        with mock.patch.object(
            import_models,
            "process_table",
            return_value=process_table,
        ), mock.patch.object(
            import_models,
            "mem_available_kib",
            return_value=40 * 1024 * 1024,
        ), mock.patch.object(
            import_models,
            "nvidia_compute_pids",
            return_value={4242},
        ):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA compute"):
                import_models.ensure_model_copy_boundary_safe(
                    allow_protected_cpu_only_coexistence=True
                )

    def test_short_file_checks_the_boundary_at_eof(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.write_bytes(b"short")
            with mock.patch.object(
                import_models,
                "ensure_model_copy_boundary_safe",
                side_effect=RuntimeError("late protected compute"),
            ) as boundary:
                with self.assertRaisesRegex(RuntimeError, "late protected"):
                    import_models.copy_and_hash(source, destination)
            boundary.assert_called_once_with(
                allow_protected_cpu_only_coexistence=False
            )

    def test_final_publication_boundary_failure_removes_staging(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            fake_root = pathlib.Path(directory)
            source = fake_root / "protected-source"
            models = fake_root / "models"
            source.mkdir()
            models.mkdir()
            revision = "fixture-revision"
            (source / "manifest.json").write_text(
                json.dumps({"revision": revision})
            )
            (source / "config.json").write_text("{}")
            (source / "model.safetensors-00001-of-00001.safetensors").write_bytes(
                b"weights"
            )
            spec = {
                "source": source,
                "revision": revision,
                "revision_file": "manifest.json",
            }
            with mock.patch.object(
                import_models, "ROOT", fake_root
            ), mock.patch.dict(
                import_models.MODEL_SPECS,
                {"Fixture": spec},
                clear=True,
            ), mock.patch.object(
                import_models,
                "ensure_model_copy_boundary_safe",
                side_effect=(None, None, None, RuntimeError("publication gate")),
            ):
                with self.assertRaisesRegex(RuntimeError, "publication gate"):
                    import_models.import_model("Fixture")
            self.assertFalse((models / "Fixture").exists())
            self.assertEqual(list(models.iterdir()), [])


class CoexistenceWatchdogTest(unittest.TestCase):
    def test_low_memory_pauses_only_verified_owned_run(self) -> None:
        process = mock.Mock()
        process.wait.side_effect = (
            subprocess.TimeoutExpired("fixture", 5),
            subprocess.TimeoutExpired("fixture", 5),
            0,
        )
        metadata: dict[str, object] = {
            "run_id": "fixture",
            "pid": 99,
            "pgid": 999,
        }
        with mock.patch.object(
            run_isolated.preflight_tool,
            "mem_available_kib",
            side_effect=(
                run_isolated.COEXISTENCE_ABORT_MEMORY_KIB - 1,
                run_isolated.COEXISTENCE_RESUME_MEMORY_KIB,
            ),
        ), mock.patch.object(
            run_isolated.preflight_tool,
            "nvidia_compute_pids",
            return_value=set(),
        ), mock.patch.object(
            run_isolated.preflight_tool,
            "process_table",
            return_value=([], [], []),
        ), mock.patch.object(
            run_isolated.stop_run,
            "signal_verified",
        ) as signal_verified:
            result = run_isolated.wait_with_coexistence_watchdog(
                process, metadata
            )
        self.assertEqual(result, (0, False))
        self.assertEqual(
            signal_verified.call_args_list,
            [mock.call(metadata, signal.SIGSTOP), mock.call(metadata, signal.SIGCONT)],
        )
        self.assertEqual(
            metadata["coexistence_pauses"][0]["reason"], "host-memory-floor"
        )
        self.assertEqual(
            metadata["coexistence_pauses"][0]["owned_run_action"],
            "verified-sigstop",
        )
        self.assertIn("resumed_at", metadata["coexistence_pauses"][0])

    def test_new_protected_nvidia_compute_pauses_owned_run(self) -> None:
        process = mock.Mock()
        process.wait.side_effect = (
            subprocess.TimeoutExpired("fixture", 5),
            subprocess.TimeoutExpired("fixture", 5),
            0,
        )
        metadata: dict[str, object] = {
            "run_id": "fixture",
            "pid": 99,
            "pgid": 999,
        }
        protected = preflight.ProcessInfo(
            pid=10,
            ppid=1,
            start_ticks=100,
            rss_kib=1,
            command="gem5.opt",
            cwd="/home/zhaosiying/zcode-lane",
        )
        owned_root = preflight.ProcessInfo(
            pid=99,
            ppid=1,
            start_ticks=990,
            rss_kib=1,
            command="owned build",
            cwd=str(ROOT),
        )
        with mock.patch.object(
            run_isolated.preflight_tool,
            "mem_available_kib",
            return_value=40 * 1024 * 1024,
        ), mock.patch.object(
            run_isolated.preflight_tool,
            "nvidia_compute_pids",
            side_effect=({10}, set()),
        ), mock.patch.object(
            run_isolated.preflight_tool,
            "process_table",
            return_value=([owned_root, protected], [protected], [owned_root]),
        ), mock.patch.object(
            run_isolated.stop_run,
            "signal_verified",
        ) as signal_verified:
            result = run_isolated.wait_with_coexistence_watchdog(
                process, metadata
            )
        self.assertEqual(result, (0, False))
        self.assertEqual(
            signal_verified.call_args_list,
            [mock.call(metadata, signal.SIGSTOP), mock.call(metadata, signal.SIGCONT)],
        )
        self.assertEqual(
            metadata["coexistence_pauses"][0]["reason"],
            "protected-nvidia-compute-became-active",
        )

    def test_owned_run_timeout_pauses_verified_group(self) -> None:
        process = mock.Mock()
        process.poll.return_value = 0
        process.wait.side_effect = (
            subprocess.TimeoutExpired("fixture", 5),
            subprocess.TimeoutExpired("fixture", 5),
        )
        metadata: dict[str, object] = {
            "run_id": "fixture",
            "pid": 99,
            "pgid": 999,
        }
        with mock.patch.object(
            run_isolated.time,
            "monotonic",
            side_effect=(0.0, 11.0, 12.0),
        ), mock.patch.object(
            run_isolated.preflight_tool,
            "mem_available_kib",
            return_value=40 * 1024 * 1024,
        ), mock.patch.object(
            run_isolated.stop_run,
            "signal_verified",
        ) as signal_verified, mock.patch.object(
            run_isolated,
            "terminate_owned_process",
            return_value=-15,
        ) as terminate:
            result = run_isolated.wait_with_coexistence_watchdog(
                process,
                metadata,
                timeout_seconds=10,
            )
        self.assertEqual(result, (-15, True))
        signal_verified.assert_called_once_with(metadata, signal.SIGSTOP)
        terminate.assert_called_once_with(process, metadata, wait_seconds=5)
        self.assertEqual(
            metadata["coexistence_abort"]["reason"],
            "owned-run-timeout",
        )

    def test_gpu_benchmark_stops_for_external_compute(self) -> None:
        process = mock.Mock()
        process.wait.side_effect = subprocess.TimeoutExpired("fixture", 1)
        process.poll.return_value = 0
        metadata: dict[str, object] = {
            "run_id": "fixture",
            "pid": 99,
            "pgid": 999,
        }
        owned_root = preflight.ProcessInfo(
            pid=99,
            ppid=1,
            start_ticks=990,
            rss_kib=1,
            command="owned benchmark",
            cwd=str(ROOT),
        )
        external = preflight.ProcessInfo(
            pid=10,
            ppid=1,
            start_ticks=100,
            rss_kib=1,
            command="external cuda",
            cwd="/tmp",
        )
        with mock.patch.object(
            run_isolated.preflight_tool,
            "mem_available_kib",
            return_value=40 * 1024 * 1024,
        ), mock.patch.object(
            run_isolated.preflight_tool,
            "nvidia_compute_pids",
            return_value={10},
        ), mock.patch.object(
            run_isolated.preflight_tool,
            "process_table",
            return_value=([owned_root, external], [], [owned_root]),
        ), mock.patch.object(
            run_isolated.stop_run,
            "process_environment",
            side_effect=ProcessLookupError,
        ), mock.patch.object(
            run_isolated,
            "terminate_owned_process",
            return_value=-15,
        ) as terminate:
            result = run_isolated.wait_with_gpu_benchmark_watchdog(
                process, metadata
            )
        self.assertEqual(result, (-15, True))
        terminate.assert_called_once_with(process, metadata, wait_seconds=5)
        self.assertEqual(
            metadata["gpu_benchmark_abort"]["reason"],
            "external-nvidia-compute-became-active",
        )

    def test_terminating_paused_owned_run_continues_pending_sigterm(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = -15
        process.returncode = -15
        metadata: dict[str, object] = {"status": "paused", "pgid": 999}
        with mock.patch.object(
            run_isolated.stop_run,
            "signal_verified",
        ) as signal_verified:
            result = run_isolated.terminate_owned_process(
                process, metadata, wait_seconds=5
            )
        self.assertEqual(result, -15)
        self.assertEqual(
            signal_verified.call_args_list,
            [mock.call(metadata, signal.SIGTERM), mock.call(metadata, signal.SIGCONT)],
        )
        process.wait.assert_called_once()
        self.assertGreater(process.wait.call_args.kwargs["timeout"], 4.9)
        self.assertLessEqual(process.wait.call_args.kwargs["timeout"], 5)

    def test_ignored_sigterm_restores_paused_owned_group(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("fixture", 5)
        metadata: dict[str, object] = {"status": "paused", "pgid": 999}
        with mock.patch.object(
            run_isolated.stop_run,
            "signal_verified",
        ) as signal_verified, mock.patch.object(
            run_isolated.stop_run,
            "process_group_members",
            side_effect=([123], [123]),
        ), mock.patch.object(
            run_isolated.time,
            "monotonic",
            side_effect=(0.0, 0.0, 6.0),
        ):
            result = run_isolated.terminate_owned_process(
                process, metadata, wait_seconds=5
            )
        self.assertEqual(result, 75)
        self.assertEqual(metadata["status"], "paused")
        self.assertEqual(
            signal_verified.call_args_list,
            [
                mock.call(metadata, signal.SIGTERM),
                mock.call(metadata, signal.SIGCONT),
                mock.call(metadata, signal.SIGSTOP),
            ],
        )


class StopRunVerificationTest(unittest.TestCase):
    def test_signal_verified_rechecks_before_killpg(self) -> None:
        metadata = {"run_id": "test"}
        with mock.patch.object(
            stop_run, "verify", return_value=(123, 456)
        ) as verify, mock.patch.object(stop_run.os, "killpg") as killpg:
            self.assertEqual(
                stop_run.signal_verified(metadata, stop_run.signal.SIGTERM),
                (123, 456),
            )
        verify.assert_called_once_with(metadata)
        killpg.assert_called_once_with(456, stop_run.signal.SIGTERM)


class PythonImportAuditTest(unittest.TestCase):
    def test_editable_finder_modules_include_class_object_module(self) -> None:
        class ExternalEditableFinder:
            pass

        ExternalEditableFinder.__module__ = "__editable___triton_external_finder"
        self.assertEqual(
            audit_environment.editable_finder_modules(ExternalEditableFinder),
            ("__editable___triton_external_finder", "builtins"),
        )

    def test_editable_finder_modules_include_instance_type_module(self) -> None:
        class AllowedFinder:
            pass

        AllowedFinder.__module__ = "_editable_skbc_pypto"
        finder = AllowedFinder()
        self.assertEqual(
            audit_environment.editable_finder_modules(finder),
            ("_editable_skbc_pypto",),
        )

    def test_external_editable_modules_reject_every_importer_carrier(self) -> None:
        class TritonHook:
            pass

        TritonHook.__module__ = "__editable___triton_external_finder"
        self.assertEqual(
            audit_environment.external_editable_modules(
                [TritonHook],
                "pypto",
            ),
            ("__editable___triton_external_finder",),
        )
        self.assertEqual(
            audit_environment.external_editable_modules(
                [TritonHook()],
                "baseline",
            ),
            ("__editable___triton_external_finder",),
        )

    def test_allowed_editable_module_policy_is_profile_specific(self) -> None:
        self.assertTrue(
            audit_environment.editable_module_is_allowed(
                "_editable_skbc_pypto", "pypto"
            )
        )
        self.assertFalse(
            audit_environment.editable_module_is_allowed(
                "_editable_skbc_pypto", "baseline"
            )
        )
        self.assertFalse(
            audit_environment.editable_module_is_allowed(
                "__editable___sglang_0_5_18_finder", "baseline"
            )
        )

    def test_editable_direct_url_must_be_local_file_without_netloc(self) -> None:
        self.assertEqual(
            audit_environment.editable_source_from_direct_url(
                "file:///workspace/source"
            ),
            pathlib.Path("/workspace/source"),
        )
        for value in (
            "https://example.invalid/source",
            "file://remote-host/source",
            "",
            None,
        ):
            with self.assertRaisesRegex(ValueError, "editable direct URL"):
                audit_environment.editable_source_from_direct_url(value)

    def test_baseline_allows_only_the_official_sglang_source(self) -> None:
        self.assertTrue(
            audit_environment.is_allowed_source(
                ROOT / "upstream" / "sglang" / "python" / "sglang",
                "baseline",
            )
        )
        for candidate in (
            ROOT / "projects" / "pypto" / "python",
            ROOT / "projects" / "pypto-kernels" / "src",
            ROOT / "projects" / "pypto-framework-plugins" / "src",
        ):
            self.assertFalse(
                audit_environment.is_allowed_source(candidate, "baseline")
            )

    def test_executable_pth_allowlist_rejects_editable_triton(self) -> None:
        triton = pathlib.Path("__editable__.triton-3.7.1.pth")
        line = "import __editable___triton_3_7_1_finder; __editable___triton_3_7_1_finder.install()"
        self.assertFalse(
            audit_environment.executable_pth_is_allowed(triton, line, "pypto")
        )
        self.assertTrue(
            audit_environment.executable_pth_is_allowed(
                pathlib.Path("_editable_skbc_pypto.pth"),
                "import _editable_skbc_pypto",
                "pypto",
            )
        )
        self.assertFalse(
            audit_environment.executable_pth_is_allowed(
                pathlib.Path("_editable_skbc_pypto.pth"),
                "import _editable_skbc_pypto",
                "baseline",
            )
        )
        self.assertFalse(
            audit_environment.executable_pth_is_allowed(
                pathlib.Path("distutils-precedence.pth"),
                "import os; os.system('unexpected')",
                "pypto",
            )
        )
        self.assertTrue(
            audit_environment.executable_pth_is_allowed(
                pathlib.Path("distutils-precedence.pth"),
                audit_environment.DISTUTILS_PRECEDENCE_PTH,
                "baseline",
            )
        )


class TritonDependencyMaterializerTest(unittest.TestCase):
    class FakeHTTPResponse:
        def __init__(
            self,
            status: int,
            headers: dict[str, str],
            payload: bytes,
        ) -> None:
            self.status = status
            self.headers = headers
            self._payload = io.BytesIO(payload)
            self.bytes_read = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            value = self._payload.read(size)
            self.bytes_read += len(value)
            return value

    @staticmethod
    def make_tar_archive(
        path: pathlib.Path,
        files: dict[str, tuple[bytes, int]],
    ) -> None:
        with tarfile.open(path, "w") as archive:
            for name, (payload, mode) in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.mode = mode
                info.size = len(payload)
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))

    @staticmethod
    def fake_package_path(
        value: str,
        *,
        size: int | None,
        sha256: str | None,
    ):
        class FakePackagePath:
            def __init__(self) -> None:
                self.size = size
                if sha256 is None:
                    self.hash = None
                else:
                    encoded = base64.urlsafe_b64encode(
                        bytes.fromhex(sha256)
                    ).decode("ascii").rstrip("=")
                    self.hash = type(
                        "FakeFileHash",
                        (),
                        {"mode": "sha256", "value": encoded},
                    )()

            def __str__(self) -> str:
                return value

            def as_posix(self) -> str:
                return value

        return FakePackagePath()

    def fake_distribution(
        self,
        site: pathlib.Path,
        package_paths: tuple[object, ...],
        fake_entry_points: tuple[object, ...] = (),
    ):
        class FakeDistribution:
            version = "1.0"
            files = package_paths
            metadata = {"Name": "producer"}
            _path = site / "producer-1.0.dist-info"
            entry_points = fake_entry_points

            @staticmethod
            def locate_file(package_path):
                return site / str(package_path)

        return FakeDistribution()

    def test_download_rejects_premature_eof_before_archive_publish(self) -> None:
        response = self.FakeHTTPResponse(
            200,
            {"Content-Length": "6"},
            b"abc",
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            destination = pathlib.Path(directory) / "archive.tar"
            with mock.patch.object(
                triton_dependencies,
                "DOWNLOAD_MAX_REQUESTS",
                1,
            ), mock.patch.object(
                triton_dependencies.urllib.request,
                "urlopen",
                return_value=response,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "download incomplete after 1 requests: received 3 of 6 bytes",
                ):
                    triton_dependencies.download(
                        "https://example.invalid/archive.tar",
                        destination,
                        max_bytes=64,
                    )
            self.assertFalse(destination.exists())
            partial = destination.with_suffix(".tar.partial")
            self.assertEqual(partial.read_bytes(), b"abc")

    def test_download_resumes_only_an_exact_http_range(self) -> None:
        responses = (
            self.FakeHTTPResponse(
                200,
                {"Content-Length": "6"},
                b"abc",
            ),
            self.FakeHTTPResponse(
                206,
                {
                    "Content-Length": "3",
                    "Content-Range": "bytes 3-5/6",
                },
                b"def",
            ),
        )
        expected_sha256 = hashlib.sha256(b"abcdef").hexdigest()
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            destination = pathlib.Path(directory) / "archive.tar"
            with mock.patch.object(
                triton_dependencies.urllib.request,
                "urlopen",
                side_effect=responses,
            ) as urlopen:
                acquisition = triton_dependencies.download(
                    "https://example.invalid/archive.tar",
                    destination,
                    max_bytes=64,
                    expected_sha256=expected_sha256,
                    expected_bytes=6,
                )
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertFalse(destination.with_suffix(".tar.partial").exists())
            self.assertEqual(
                urlopen.call_args_list[1].args[0].get_header("Range"),
                "bytes=3-",
            )
            self.assertEqual(
                acquisition,
                {
                    "method": "network-download",
                    "source_url": "https://example.invalid/archive.tar",
                    "declared_bytes": 6,
                    "request_count": 2,
                    "range_request_count": 1,
                },
            )

    def test_download_requires_an_exact_declared_content_length(self) -> None:
        invalid_headers = ({}, {"Content-Length": "06"})
        for headers in invalid_headers:
            with self.subTest(headers=headers), tempfile.TemporaryDirectory(
                dir=ROOT / "builds"
            ) as directory:
                response = self.FakeHTTPResponse(200, headers, b"abcdef")
                destination = pathlib.Path(directory) / "archive.tar"
                with mock.patch.object(
                    triton_dependencies.urllib.request,
                    "urlopen",
                    return_value=response,
                ):
                    with self.assertRaisesRegex(
                        triton_dependencies.DownloadContractError,
                        "Content-Length",
                    ):
                        triton_dependencies.download(
                            "https://example.invalid/archive.tar",
                            destination,
                            max_bytes=64,
                        )
                self.assertFalse(destination.exists())
                self.assertEqual(
                    destination.with_suffix(".tar.partial").read_bytes(),
                    b"",
                )
                self.assertEqual(response.bytes_read, 0)

    def test_download_never_appends_ignored_or_malformed_ranges(self) -> None:
        invalid_responses = (
            (
                self.FakeHTTPResponse(
                    200,
                    {"Content-Length": "3"},
                    b"def",
                ),
                "ignored Range",
            ),
            (
                self.FakeHTTPResponse(
                    206,
                    {
                        "Content-Length": "3",
                        "Content-Range": "bytes malformed",
                    },
                    b"def",
                ),
                "malformed Content-Range",
            ),
            (
                self.FakeHTTPResponse(
                    206,
                    {
                        "Content-Length": "3",
                        "Content-Range": "bytes 2-4/6",
                    },
                    b"def",
                ),
                "requested offset/total",
            ),
            (
                self.FakeHTTPResponse(
                    206,
                    {
                        "Content-Length": "2",
                        "Content-Range": "bytes 3-5/6",
                    },
                    b"def",
                ),
                "disagrees with Content-Range",
            ),
        )
        for invalid_response, message in invalid_responses:
            with self.subTest(message=message), tempfile.TemporaryDirectory(
                dir=ROOT / "builds"
            ) as directory:
                initial = self.FakeHTTPResponse(
                    200,
                    {"Content-Length": "6"},
                    b"abc",
                )
                destination = pathlib.Path(directory) / "archive.tar"
                with mock.patch.object(
                    triton_dependencies.urllib.request,
                    "urlopen",
                    side_effect=(initial, invalid_response),
                ):
                    with self.assertRaisesRegex(
                        triton_dependencies.DownloadContractError,
                        message,
                    ):
                        triton_dependencies.download(
                            "https://example.invalid/archive.tar",
                            destination,
                            max_bytes=64,
                        )
                self.assertFalse(destination.exists())
                self.assertEqual(
                    destination.with_suffix(".tar.partial").read_bytes(),
                    b"abc",
                )
                self.assertEqual(invalid_response.bytes_read, 0)

    def test_seed_copy_requires_exact_source_locked_identity(self) -> None:
        payload = b"reviewed seed archive"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            seed_dir = root / "seeds"
            seed_dir.mkdir()
            seed = seed_dir / "archive.tar"
            seed.write_bytes(payload)
            seed_dir = triton_dependencies.require_seed_download_directory(seed_dir)

            mismatched = triton_dependencies.PackageSpec(
                "probe",
                "https://example.invalid/archive.tar",
                "tar",
                expected_sha256="0" * 64,
                expected_bytes=len(payload),
            )
            with self.assertRaisesRegex(RuntimeError, "reviewed digest"):
                triton_dependencies.copy_seed_archive(
                    mismatched,
                    seed_dir,
                    root / "mismatched.tar",
                    max_bytes=64,
                )
            self.assertFalse((root / "mismatched.tar").exists())

            unpinned = triton_dependencies.PackageSpec(
                "probe",
                "https://example.invalid/archive.tar",
                "tar",
            )
            with self.assertRaisesRegex(RuntimeError, "unpinned dependency"):
                triton_dependencies.copy_seed_archive(
                    unpinned,
                    seed_dir,
                    root / "unpinned.tar",
                    max_bytes=64,
                )
            self.assertFalse((root / "unpinned.tar").exists())

            pinned = triton_dependencies.PackageSpec(
                "probe",
                "https://example.invalid/archive.tar",
                "tar",
                expected_sha256=digest,
                expected_bytes=len(payload),
            )
            destination = root / "copied.tar"
            acquisition = triton_dependencies.copy_seed_archive(
                pinned,
                seed_dir,
                destination,
                max_bytes=64,
            )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(destination.stat().st_nlink, 1)
            self.assertEqual(
                acquisition,
                {
                    "method": "workspace-seed-copy",
                    "source_path": seed.relative_to(ROOT).as_posix(),
                    "source_bytes": len(payload),
                    "source_sha256": digest,
                    "copied_bytes": len(payload),
                    "copied_sha256": digest,
                },
            )

    def test_seed_directory_and_file_must_be_canonical_and_independent(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            seed_dir = root / "seeds"
            seed_dir.mkdir()
            linked_dir = root / "linked-seeds"
            linked_dir.symlink_to(seed_dir, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "canonical path"):
                triton_dependencies.require_seed_download_directory(linked_dir)
            with self.assertRaisesRegex(ValueError, "absolute canonical path"):
                triton_dependencies.require_seed_download_directory(
                    pathlib.Path("caches/triton-download-seeds")
                )
            seed_dir.chmod(0o777)
            with self.assertRaisesRegex(ValueError, "group/other-writable"):
                triton_dependencies.require_seed_download_directory(seed_dir)
            seed_dir.chmod(0o755)

            pinned = triton_dependencies.PackageSpec(
                "probe",
                "https://example.invalid/archive.tar",
                "tar",
                expected_sha256=hashlib.sha256(b"payload").hexdigest(),
            )
            original = root / "original.tar"
            original.write_bytes(b"payload")
            hardlink = seed_dir / "archive.tar"
            hardlink.hardlink_to(original)
            with self.assertRaisesRegex(RuntimeError, "independent"):
                triton_dependencies.copy_seed_archive(
                    pinned,
                    triton_dependencies.require_seed_download_directory(seed_dir),
                    root / "hardlink-copy.tar",
                    max_bytes=64,
                )
            hardlink.unlink()
            hardlink.symlink_to(original)
            with self.assertRaisesRegex(RuntimeError, "independent"):
                triton_dependencies.copy_seed_archive(
                    pinned,
                    triton_dependencies.require_seed_download_directory(seed_dir),
                    root / "symlink-copy.tar",
                    max_bytes=64,
                )
            hardlink.unlink()
            hardlink.write_bytes(b"payload")
            hardlink.chmod(0o666)
            with self.assertRaisesRegex(RuntimeError, "safely-permissioned"):
                triton_dependencies.copy_seed_archive(
                    pinned,
                    triton_dependencies.require_seed_download_directory(seed_dir),
                    root / "writable-copy.tar",
                    max_bytes=64,
                )

    def test_archive_publication_cannot_replace_a_competing_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            destination = root / "archive.tar"
            partial = triton_dependencies._create_private_partial(destination)
            partial.write_bytes(b"verified archive")
            digest = hashlib.sha256(b"verified archive").hexdigest()
            real_rename = triton_dependencies._rename_no_replace

            def create_competitor_then_rename(source, target) -> None:
                pathlib.Path(target).write_bytes(b"competing archive")
                real_rename(pathlib.Path(source), pathlib.Path(target))

            with mock.patch.object(
                triton_dependencies,
                "_rename_no_replace",
                side_effect=create_competitor_then_rename,
            ):
                with self.assertRaises(FileExistsError):
                    triton_dependencies._publish_verified_partial(
                        partial,
                        destination,
                        expected_bytes=len(b"verified archive"),
                        expected_sha256=digest,
                    )
            self.assertEqual(destination.read_bytes(), b"competing archive")
            self.assertEqual(partial.read_bytes(), b"verified archive")

    def test_producer_record_requires_exact_declared_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            fake_root = pathlib.Path(directory)
            site = (
                fake_root
                / "envs/pypto-nvidia/lib/python3.14/site-packages"
            )
            site.mkdir(parents=True)
            payload = site / "producer.py"
            payload.write_bytes(b"reviewed")
            record = site / "producer-1.0.dist-info/RECORD"
            record.parent.mkdir()
            record.write_text("producer.py,...\n")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            package_path = self.fake_package_path(
                "producer.py",
                size=payload.stat().st_size,
                sha256=digest,
            )
            record_path = self.fake_package_path(
                "producer-1.0.dist-info/RECORD",
                size=None,
                sha256=None,
            )
            distribution = self.fake_distribution(
                site, (package_path, record_path)
            )
            with mock.patch.object(
                triton_dependencies, "ROOT", fake_root
            ), mock.patch.object(
                triton_dependencies.importlib.metadata,
                "distributions",
                return_value=(distribution,),
            ), mock.patch.object(
                triton_dependencies, "PRODUCER_RECORD_REWRITES", {}
            ), mock.patch.object(
                triton_dependencies, "PRODUCER_REWRITE_ENTRY_POINTS", {}
            ):
                identity = triton_dependencies.distribution_record_identity(
                    "producer"
                )
                self.assertEqual(identity["record_rewrites"], 0)
                payload.write_bytes(b"tampered")
                with self.assertRaisesRegex(
                    RuntimeError, "unexpected producer RECORD rewrite"
                ):
                    triton_dependencies.distribution_record_identity("producer")

    def test_producer_rewrite_policy_is_exactly_six_console_scripts(self) -> None:
        expected = {
            ("cmake", "../../../bin/ccmake"),
            ("cmake", "../../../bin/cmake"),
            ("cmake", "../../../bin/cpack"),
            ("cmake", "../../../bin/ctest"),
            ("lit", "../../../bin/lit"),
            ("wheel", "../../../bin/wheel"),
        }
        self.assertEqual(
            set(triton_dependencies.PRODUCER_RECORD_REWRITES), expected
        )
        self.assertEqual(
            set(triton_dependencies.PRODUCER_REWRITE_ENTRY_POINTS), expected
        )
        counts = {
            name: sum(key[0] == name for key in expected)
            for name in ("cmake", "lit", "wheel")
        }
        self.assertEqual(counts, {"cmake": 4, "lit": 1, "wheel": 1})
        locked = triton_dependencies.load_versions_lock()
        self.assertEqual(
            triton_dependencies.producer_record_rewrite_policy_sha256(),
            locked["triton.producer.record_rewrite_policy_sha256"],
        )

    def test_producer_record_allows_only_an_exact_frozen_rewrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            fake_root = pathlib.Path(directory)
            environment = fake_root / "envs/pypto-nvidia"
            site = environment / "lib/python3.14/site-packages"
            site.mkdir(parents=True)
            wrapper = environment / "bin/producer"
            wrapper.parent.mkdir()
            wrapper.write_bytes(b"installed-wrapper")
            wrapper.chmod(0o755)
            record = site / "producer-1.0.dist-info/RECORD"
            record.parent.mkdir()
            record.write_text("../../../bin/producer,...\n")
            declared = hashlib.sha256(b"wheel-wrapper").hexdigest()
            package_path = self.fake_package_path(
                "../../../bin/producer",
                size=len(b"wheel-wrapper"),
                sha256=declared,
            )
            record_path = self.fake_package_path(
                "producer-1.0.dist-info/RECORD",
                size=None,
                sha256=None,
            )
            entry_point = type(
                "FakeEntryPoint",
                (),
                {
                    "group": "console_scripts",
                    "name": "producer",
                    "value": "producer:main",
                },
            )()
            distribution = self.fake_distribution(
                site, (package_path, record_path), (entry_point,)
            )
            rewrite = {
                ("producer", "../../../bin/producer"): {
                    "path": "bin/producer",
                    "mode": 0o755,
                    "record_size": len(b"wheel-wrapper"),
                    "record_sha256": declared,
                    "actual_size": wrapper.stat().st_size,
                    "actual_sha256": hashlib.sha256(
                        wrapper.read_bytes()
                    ).hexdigest(),
                }
            }
            with mock.patch.object(
                triton_dependencies, "ROOT", fake_root
            ), mock.patch.object(
                triton_dependencies.importlib.metadata,
                "distributions",
                return_value=(distribution,),
            ), mock.patch.object(
                triton_dependencies, "PRODUCER_RECORD_REWRITES", rewrite
            ), mock.patch.object(
                triton_dependencies,
                "PRODUCER_REWRITE_ENTRY_POINTS",
                {
                    ("producer", "../../../bin/producer"): (
                        "producer",
                        "producer:main",
                    )
                },
            ), mock.patch.object(
                triton_dependencies,
                "PRODUCER_CONSOLE_SHEBANG",
                "installed-wrapper",
            ):
                identity = triton_dependencies.distribution_record_identity(
                    "producer"
                )
                self.assertEqual(identity["record_rewrites"], 1)
                wrapper.write_bytes(b"drifted-wrapper")
                with self.assertRaisesRegex(
                    RuntimeError, "producer RECORD rewrite drift"
                ):
                    triton_dependencies.distribution_record_identity("producer")
                wrapper.write_bytes(b"wheel-wrapper")
                wrapper.chmod(0o755)
                with self.assertRaisesRegex(
                    RuntimeError, "rewrite set drift"
                ):
                    triton_dependencies.distribution_record_identity("producer")

    def test_producer_record_rejects_partial_metadata_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            fake_root = pathlib.Path(directory)
            environment = fake_root / "envs/pypto-nvidia"
            site = environment / "lib/python3.14/site-packages"
            site.mkdir(parents=True)
            payload = site / "producer.py"
            payload.write_bytes(b"producer")
            record = site / "producer-1.0.dist-info/RECORD"
            record.parent.mkdir()
            record.write_text("producer.py,...\n")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            record_path = self.fake_package_path(
                "producer-1.0.dist-info/RECORD", size=None, sha256=None
            )
            partial = self.fake_package_path(
                "producer.py", size=None, sha256=digest
            )
            distribution = self.fake_distribution(site, (partial, record_path))
            common_patches = (
                mock.patch.object(triton_dependencies, "ROOT", fake_root),
                mock.patch.object(
                    triton_dependencies.importlib.metadata,
                    "distributions",
                    return_value=(distribution,),
                ),
                mock.patch.object(
                    triton_dependencies, "PRODUCER_RECORD_REWRITES", {}
                ),
                mock.patch.object(
                    triton_dependencies, "PRODUCER_REWRITE_ENTRY_POINTS", {}
                ),
            )
            with common_patches[0], common_patches[1], common_patches[
                2
            ], common_patches[3]:
                with self.assertRaisesRegex(RuntimeError, "partial hash/size"):
                    triton_dependencies.distribution_record_identity("producer")
            payload.unlink()
            target = site / "target.py"
            target.write_bytes(b"target")
            payload.symlink_to(target.name)
            symlink_path = self.fake_package_path(
                "producer.py", size=None, sha256=None
            )
            distribution.files = (symlink_path, record_path)
            with common_patches[0], common_patches[1], common_patches[
                2
            ], common_patches[3]:
                with self.assertRaisesRegex(RuntimeError, "unsupported symlink"):
                    triton_dependencies.distribution_record_identity("producer")

    def test_archive_member_rejects_absolute_and_parent_paths(self) -> None:
        for value in ("/absolute", "../escape", "root/../../escape"):
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                triton_dependencies._safe_member_name(value)
        triton_dependencies._safe_member_name("root/include/header.h")

    def test_archive_links_cannot_escape_after_relocation(self) -> None:
        for hardlink in (False, True):
            with self.assertRaisesRegex(ValueError, "unsafe archive link target"):
                triton_dependencies._safe_link_target(
                    "root/lib/link",
                    "../../../escape",
                    hardlink=hardlink,
                )
        triton_dependencies._safe_link_target(
            "root/lib/link",
            "../include/header",
            hardlink=False,
        )

    def test_extract_rejects_duplicate_zip_members(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            archive = root / "duplicate.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("same", b"first")
                with self.assertWarns(UserWarning):
                    output.writestr("same", b"second")
            with self.assertRaisesRegex(ValueError, "duplicate zip member"):
                triton_dependencies.extract_archive(
                    archive,
                    root / "out",
                    "zip",
                )

    def test_inspect_tar_rejects_duplicate_and_excess_directory_members(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            duplicate = root / "duplicate.tar"
            with tarfile.open(duplicate, "w") as output:
                for _ in range(2):
                    info = tarfile.TarInfo("same")
                    info.size = 1
                    output.addfile(info, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "duplicate tar member"):
                triton_dependencies.inspect_archive(duplicate, "tar")

            directories = root / "directories.tar"
            with tarfile.open(directories, "w") as output:
                for name in ("one", "two"):
                    info = tarfile.TarInfo(name)
                    info.type = tarfile.DIRTYPE
                    output.addfile(info)
            with self.assertRaisesRegex(ValueError, "member-count limit"):
                triton_dependencies.inspect_archive(
                    directories,
                    "tar",
                    max_members=1,
                )

    def test_archive_member_rejects_oversized_path_component(self) -> None:
        with self.assertRaisesRegex(ValueError, "path is too long"):
            triton_dependencies._safe_member_name("x" * 256)

    def test_extract_rejects_declared_resource_overflow(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            archive = root / "large.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("payload", b"12345")
            with self.assertRaisesRegex(ValueError, "expanded-byte limit"):
                triton_dependencies.extract_archive(
                    archive,
                    root / "out",
                    "zip",
                    max_expanded_bytes=4,
                    max_members=1,
                )

    def test_extract_rejects_escaping_tar_symlink_and_hardlink(self) -> None:
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
                root = pathlib.Path(directory)
                archive = root / "escape.tar"
                with tarfile.open(archive, "w") as output:
                    info = tarfile.TarInfo("root/link")
                    info.type = kind
                    info.linkname = "../../escape"
                    output.addfile(info)
                with self.assertRaisesRegex(
                    ValueError, "unsafe archive link target"
                ):
                    triton_dependencies.extract_archive(
                        archive,
                        root / "out",
                        "tar",
                    )

    def test_tree_identity_binds_file_bytes_and_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            (root / "value").write_bytes(b"first")
            (root / "link").symlink_to("value")
            first = triton_dependencies.tree_identity(root)
            (root / "value").write_bytes(b"second")
            second = triton_dependencies.tree_identity(root)
            self.assertNotEqual(first["sha256"], second["sha256"])
            (root / "link").unlink()
            (root / "link").symlink_to("missing")
            third = triton_dependencies.tree_identity(root)
            self.assertNotEqual(second["sha256"], third["sha256"])

    def test_tree_identity_binds_executable_mode_and_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            value = root / "tool"
            value.write_bytes(b"same")
            first = triton_dependencies.tree_identity(root)
            value.chmod(0o755)
            second = triton_dependencies.tree_identity(root)
            self.assertNotEqual(first["sha256"], second["sha256"])
            (root / "empty").mkdir()
            third = triton_dependencies.tree_identity(root)
            self.assertNotEqual(second["sha256"], third["sha256"])

    def test_reference_source_tree_allows_extras_but_rejects_input_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            (reference / "source.py").write_text("reviewed\n")
            shutil.copy2(reference / "source.py", candidate / "source.py")
            (candidate / "generated.txt").write_text("allowed extra\n")
            triton_dependencies.verify_reference_tree_unchanged(
                reference, candidate
            )
            (candidate / "source.py").write_text("drifted\n")
            with self.assertRaisesRegex(RuntimeError, "source-tree file drift"):
                triton_dependencies.verify_reference_tree_unchanged(
                    reference, candidate
                )

    def test_overlay_derivation_rejects_extra_or_mutated_leaf(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            names = (
                "ptxas",
                "ptxas_blackwell",
                "cuobjdump",
                "nvdisasm",
                "cudacrt",
                "cudart",
                "cupti",
            )
            expanded = {name: root / "expanded" / name for name in names}
            for name, path in expanded.items():
                path.mkdir(parents=True)
                if name in {"ptxas", "ptxas_blackwell"}:
                    tool = path / "bin/ptxas"
                elif name in {"cuobjdump", "nvdisasm"}:
                    tool = path / f"bin/{name}"
                else:
                    (path / "include").mkdir()
                    (path / "include" / f"{name}.h").write_text(name)
                    continue
                tool.parent.mkdir()
                tool.write_text(name)
                tool.chmod(0o755)
            (expanded["cupti"] / "lib").mkdir()
            (expanded["cupti"] / "lib/libcupti.so").write_text("cupti")
            overlay = root / "overlay"
            triton_dependencies.assemble_overlay(expanded, overlay)
            triton_dependencies.verify_overlay_derivation(expanded, overlay)
            (overlay / "unexpected").write_text("extra")
            with self.assertRaisesRegex(RuntimeError, "not derived"):
                triton_dependencies.verify_overlay_derivation(expanded, overlay)

    def test_unreviewed_layout_validation_never_executes_downloaded_tools(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            llvm = root / "llvm"
            json_root = root / "json"
            overlay = root / "overlay"
            for relative in (
                "bin/FileCheck",
                "lib/cmake/llvm/LLVMConfig.cmake",
                "lib/cmake/mlir/MLIRConfig.cmake",
                "lib/cmake/lld/LLDConfig.cmake",
            ):
                path = llvm / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("input")
            (llvm / "bin/FileCheck").chmod(0o755)
            json_header = json_root / "include/nlohmann/json.hpp"
            json_header.parent.mkdir(parents=True)
            json_header.write_text("json")
            for name in triton_dependencies.OVERLAY_TOOL_VERSIONS:
                path = overlay / "bin" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\x7fELFprobe")
                path.chmod(0o755)
            for relative in ("include/cuda.h", "include/cupti.h"):
                path = overlay / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("header")
            cupti = overlay / "lib/cupti/libcupti.so"
            cupti.parent.mkdir(parents=True)
            cupti.write_text("library")
            with mock.patch.object(
                triton_dependencies.subprocess,
                "run",
                side_effect=AssertionError("must not execute"),
            ):
                triton_dependencies.validate_build_input_layout(
                    llvm,
                    json_root,
                    overlay,
                )

    def test_reviewed_tool_probe_uses_read_only_networkless_bwrap(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            overlay = pathlib.Path(directory)
            for name, version in triton_dependencies.OVERLAY_TOOL_VERSIONS.items():
                path = overlay / "bin" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("tool")
            versions = iter(triton_dependencies.OVERLAY_TOOL_VERSIONS.values())

            def fake_run(command, **kwargs):
                kwargs["stdout"].write(next(versions).encode("utf-8"))
                return triton_dependencies.subprocess.CompletedProcess(
                    command, 0
                )

            with mock.patch.object(
                triton_dependencies.subprocess,
                "run",
                side_effect=fake_run,
            ) as run:
                triton_dependencies.validate_overlay_tool_versions_sandboxed(
                    overlay
                )
            self.assertEqual(run.call_count, 4)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[0], "/usr/bin/bwrap")
                self.assertIn("--unshare-net", command)
                self.assertIn("--ro-bind", command)
                self.assertEqual(call.kwargs["timeout"], 30)
                self.assertIsNotNone(call.kwargs["preexec_fn"])

    def test_output_must_be_a_workspace_child(self) -> None:
        with self.assertRaisesRegex(ValueError, "child of the workspace"):
            triton_dependencies.require_below_workspace(ROOT)
        with self.assertRaisesRegex(ValueError, "child of the workspace"):
            triton_dependencies.require_below_workspace(pathlib.Path("/tmp"))

    def test_materialization_output_cannot_target_project_or_environment(self) -> None:
        for path in (
            ROOT / "projects" / "unexpected",
            ROOT / "upstream" / "unexpected",
            ROOT / "envs" / "unexpected",
            ROOT / "builds" / "unexpected",
        ):
            with self.assertRaisesRegex(ValueError, "materialization output"):
                triton_dependencies.require_materialization_output(path)
        accepted = ROOT / "builds" / "triton-deps-materialize-test"
        self.assertEqual(
            triton_dependencies.require_materialization_output(accepted),
            accepted.resolve(),
        )

    def test_manifest_path_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            for value in ("/absolute", "../escape", "inside/../../escape"):
                with self.assertRaisesRegex(RuntimeError, "unsafe|escapes"):
                    triton_dependencies.manifest_path(root, value, "probe")
            expected = root / "inside" / "file"
            self.assertEqual(
                triton_dependencies.manifest_path(root, "inside/file", "probe"),
                expected.resolve(),
            )
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                (root / "linked").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError, "escapes output"):
                    triton_dependencies.manifest_path(
                        root, "linked/file", "probe"
                    )
            finally:
                outside.rmdir()

    def test_reviewed_mode_requires_every_archive_sha(self) -> None:
        unpinned = triton_dependencies.PackageSpec(
            "probe",
            "https://example.invalid/probe.tar",
            "tar",
        )
        pinned = triton_dependencies.PackageSpec(
            "probe",
            "https://example.invalid/probe.tar",
            "tar",
            expected_sha256="0" * 64,
        )
        self.assertFalse(
            triton_dependencies.dependencies_are_fully_pinned((unpinned,))
        )
        self.assertTrue(
            triton_dependencies.dependencies_are_fully_pinned((pinned,))
        )

    def test_reviewed_manifest_requires_version_controlled_source_anchor(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            triton_dependencies.validate_reviewed_manifest_anchor("0" * 64)
        with mock.patch.object(
            triton_dependencies,
            "REVIEWED_MANIFEST_SHA256",
            "1" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "source anchor"):
                triton_dependencies.validate_reviewed_manifest_anchor("0" * 64)
            triton_dependencies.validate_reviewed_manifest_anchor("1" * 64)

    def test_materializer_constants_match_versions_lock(self) -> None:
        triton_dependencies.validate_versions_lock()

    def test_materializer_constants_match_pinned_source_objects(self) -> None:
        triton_dependencies.validate_source_pins()

    def test_live_triton_producer_identity_matches_lock(self) -> None:
        triton_dependencies.validate_live_producers()

    def test_python_producer_site_contains_only_selected_record_files(self) -> None:
        selected = ("build", "packaging", "pyproject-hooks", "wheel")
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            destination = pathlib.Path(directory) / "producer-site"
            identity = triton_dependencies.assemble_python_producer_site(
                destination, selected
            )
            self.assertEqual(identity["distributions"], list(selected))
            self.assertTrue((destination / "build/__init__.py").is_file())
            self.assertTrue((destination / "packaging/__init__.py").is_file())
            self.assertTrue((destination / "pyproject_hooks/__init__.py").is_file())
            self.assertTrue((destination / "wheel/__init__.py").is_file())
            self.assertFalse((destination / "pip").exists())
            self.assertFalse((destination / "triton").exists())

    def test_review_transition_keeps_actual_and_expected_digest_separate(self) -> None:
        pinned = triton_dependencies.PACKAGE_SPECS[0]
        self.assertIsNotNone(pinned.expected_sha256)
        triton_dependencies.validate_reviewed_archive_digest(
            pinned,
            pinned.expected_sha256,
        )
        with self.assertRaisesRegex(RuntimeError, "reviewed digest"):
            triton_dependencies.validate_reviewed_archive_digest(
                pinned,
                "0" * 64,
            )
        unpinned = triton_dependencies.PackageSpec(
            "probe",
            "https://example.invalid/probe.tar",
            "tar",
        )
        triton_dependencies.validate_reviewed_archive_digest(
            unpinned,
            "0" * 64,
        )

    def test_canonical_manifest_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            path = pathlib.Path(directory) / "manifest.json"
            path.write_text('{"schema_version":1,"schema_version":1}\n')
            with self.assertRaisesRegex(RuntimeError, "duplicate manifest key"):
                triton_dependencies.load_canonical_manifest(path)

    def test_canonical_manifest_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            root = pathlib.Path(directory)
            target = root / "target.json"
            target.write_text('{"schema_version":1}\n')
            path = root / "manifest.json"
            path.symlink_to(target.name)
            with self.assertRaisesRegex(RuntimeError, "non-symlink"):
                triton_dependencies.load_canonical_manifest(path)

    def test_reviewed_verify_requires_external_manifest_anchor(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            with self.assertRaisesRegex(ValueError, "expected manifest SHA"):
                triton_dependencies.verify(
                    pathlib.Path(directory),
                    require_reviewed=True,
                )

    def test_local_materialize_review_cache_lifecycle_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            fake_root = pathlib.Path(directory) / "workspace"
            fixtures = fake_root / "fixtures"
            builds = fake_root / "builds"
            cache_parent = fake_root / "caches/triton-build-deps"
            seed_downloads = fake_root / "caches/triton-download-seeds"
            fixtures.mkdir(parents=True)
            builds.mkdir()
            cache_parent.mkdir(parents=True)
            seed_downloads.mkdir()
            package_files = {
                "pybind11": {"marker": (b"pybind11", 0o644)},
                "llvm": {
                    "bin/FileCheck": (b"filecheck", 0o755),
                    "lib/cmake/llvm/LLVMConfig.cmake": (b"llvm", 0o644),
                    "lib/cmake/mlir/MLIRConfig.cmake": (b"mlir", 0o644),
                    "lib/cmake/lld/LLDConfig.cmake": (b"lld", 0o644),
                },
                "json": {
                    "include/nlohmann/json.hpp": (b"json", 0o644),
                },
                "ptxas": {"bin/ptxas": (b"\x7fELFptxas", 0o755)},
                "ptxas_blackwell": {
                    "bin/ptxas": (b"\x7fELFblackwell", 0o755),
                },
                "cuobjdump": {
                    "bin/cuobjdump": (b"\x7fELFcuobjdump", 0o755),
                },
                "nvdisasm": {
                    "bin/nvdisasm": (b"\x7fELFnvdisasm", 0o755),
                },
                "cudacrt": {"include/cuda.h": (b"cuda", 0o644)},
                "cudart": {
                    "include/cuda_runtime.h": (b"cudart", 0o644),
                },
                "cupti": {
                    "include/cupti.h": (b"cupti", 0o644),
                    "lib/libcupti.so": (b"cupti-so", 0o644),
                },
            }
            archives: dict[str, pathlib.Path] = {}
            specs = []
            limits = {}
            for name, files in package_files.items():
                archive = fixtures / f"{name}.tar"
                self.make_tar_archive(archive, files)
                url = f"fixture:///{archive.name}"
                archives[url] = archive
                specs.append(
                    triton_dependencies.PackageSpec(
                        name,
                        url,
                        "tar",
                        expected_sha256=triton_dependencies.sha256_file(archive),
                    )
                )
                limits[name] = (
                    archive.stat().st_size + 1,
                    1 << 20,
                    32,
                )
            shutil.copyfile(
                archives[specs[0].url],
                seed_downloads / pathlib.PurePosixPath(specs[0].url).name,
            )

            def local_download(
                url: str,
                destination: pathlib.Path,
                *,
                max_bytes: int,
                expected_sha256: str | None,
                expected_bytes: int | None,
            ) -> dict[str, object]:
                payload = archives[url].read_bytes()
                self.assertLessEqual(len(payload), max_bytes)
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    expected_sha256,
                )
                self.assertIsNone(expected_bytes)
                self.assertTrue(destination.parent.parent.name.startswith("."))
                shutil.copyfile(archives[url], destination)
                return {
                    "method": "network-download",
                    "source_url": url,
                    "declared_bytes": len(payload),
                    "request_count": 1,
                    "range_request_count": 0,
                }

            output = builds / "triton-deps-materialize-fixture"
            real_publish = triton_dependencies._rename_no_replace
            atomic_publications = []

            def recorded_publish(source, destination) -> None:
                source_path = pathlib.Path(source)
                destination_path = pathlib.Path(destination)
                if destination_path == output:
                    self.assertTrue(source_path.name.startswith(f".{output.name}."))
                    self.assertTrue(source_path.is_dir())
                    self.assertFalse(output.exists())
                    atomic_publications.append((source_path, destination_path))
                real_publish(source_path, destination_path)

            patches = (
                mock.patch.object(triton_dependencies, "ROOT", fake_root),
                mock.patch.object(
                    triton_dependencies, "PACKAGE_SPECS", tuple(specs)
                ),
                mock.patch.object(
                    triton_dependencies, "PACKAGE_RESOURCE_LIMITS", limits
                ),
                mock.patch.object(
                    triton_dependencies, "MATERIALIZATION_HEADROOM_BYTES", 0
                ),
                mock.patch.object(
                    triton_dependencies, "download", side_effect=local_download
                ),
                mock.patch.object(
                    triton_dependencies,
                    "_rename_no_replace",
                    side_effect=recorded_publish,
                ),
                mock.patch.object(
                    triton_dependencies.urllib.request,
                    "urlopen",
                    side_effect=AssertionError("network forbidden"),
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
                5
            ], patches[6] as urlopen:
                manifest = triton_dependencies.materialize(
                    output,
                    seed_download_dir=seed_downloads,
                )
                self.assertEqual(len(atomic_publications), 1)
                self.assertFalse(atomic_publications[0][0].exists())
                self.assertEqual(
                    triton_dependencies.verify(output),
                    manifest,
                )
                by_name = {
                    record["name"]: record for record in manifest["packages"]
                }
                self.assertEqual(
                    by_name["pybind11"]["acquisition"]["method"],
                    "workspace-seed-copy",
                )
                self.assertEqual(
                    by_name["pybind11"]["acquisition"]["source_path"],
                    "caches/triton-download-seeds/pybind11.tar",
                )
                self.assertTrue(
                    all(
                        record["acquisition"]["method"] == "network-download"
                        for record in manifest["packages"][1:]
                    )
                )
                tampered_manifest = json.loads(json.dumps(manifest))
                tampered_manifest["packages"][0]["acquisition"]["unexpected"] = (
                    True
                )
                (output / "manifest.json").write_text(
                    triton_dependencies.canonical_json(tampered_manifest)
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "acquisition keys mismatch: pybind11",
                ):
                    triton_dependencies.verify(output)
                (output / "manifest.json").write_text(
                    triton_dependencies.canonical_json(manifest)
                )
                manifest_sha = triton_dependencies.sha256_file(
                    output / "manifest.json"
                )
                with mock.patch.object(
                    triton_dependencies,
                    "REVIEWED_MANIFEST_SHA256",
                    manifest_sha,
                ):
                    self.assertEqual(
                        triton_dependencies.promote_reviewed(
                            output, manifest_sha
                        ),
                        manifest,
                    )
                    self.assertEqual(
                        triton_dependencies.verify(
                            output,
                            expected_manifest_sha256=manifest_sha,
                            require_reviewed=True,
                        ),
                        manifest,
                    )
                    cache = cache_parent / manifest_sha
                    output.rename(cache)
                    self.assertEqual(
                        triton_dependencies.verify(
                            cache,
                            expected_manifest_sha256=manifest_sha,
                            require_reviewed=True,
                        ),
                        manifest,
                    )
                    header = cache / "expanded/json/include/nlohmann/json.hpp"
                    header.write_bytes(b"tampered")
                    self.assertEqual(
                        triton_dependencies.sha256_file(
                            cache / "manifest.json"
                        ),
                        manifest_sha,
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "expanded tree mismatch: json"
                    ):
                        triton_dependencies.verify(
                            cache,
                            expected_manifest_sha256=manifest_sha,
                            require_reviewed=True,
                        )
                urlopen.assert_not_called()

    def test_failed_materialization_never_publishes_final_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            fake_root = pathlib.Path(directory) / "workspace"
            builds = fake_root / "builds"
            builds.mkdir(parents=True)
            output = builds / "triton-deps-materialize-failure"
            partials: list[pathlib.Path] = []

            def fail_after_staging(
                staging: pathlib.Path,
                *,
                seed_download_dir: pathlib.Path | None,
            ):
                self.assertIsNone(seed_download_dir)
                staging.mkdir()
                partials.append(staging)
                raise RuntimeError("fixture failure")

            with mock.patch.object(
                triton_dependencies, "ROOT", fake_root
            ), mock.patch.object(
                triton_dependencies,
                "_materialize_into",
                side_effect=fail_after_staging,
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    triton_dependencies.materialize(output)
            self.assertFalse(output.exists())
            self.assertEqual(len(partials), 1)
            self.assertTrue(partials[0].is_dir())
            self.assertTrue(partials[0].name.startswith(f".{output.name}."))

    def test_materialization_cannot_replace_a_competing_formal_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "builds") as directory:
            fake_root = pathlib.Path(directory) / "workspace"
            builds = fake_root / "builds"
            builds.mkdir(parents=True)
            output = builds / "triton-deps-materialize-race"
            staged: list[pathlib.Path] = []
            manifest = {"fixture": "verified"}

            def make_staging(
                staging: pathlib.Path,
                *,
                seed_download_dir: pathlib.Path | None,
            ) -> dict[str, str]:
                self.assertIsNone(seed_download_dir)
                staging.mkdir()
                (staging / "payload").write_bytes(b"staged")
                staged.append(staging)
                return manifest

            real_rename = triton_dependencies._rename_no_replace

            def create_competitor_then_rename(source, destination) -> None:
                destination = pathlib.Path(destination)
                destination.mkdir()
                (destination / "payload").write_bytes(b"competing")
                real_rename(pathlib.Path(source), destination)

            with mock.patch.object(
                triton_dependencies,
                "ROOT",
                fake_root,
            ), mock.patch.object(
                triton_dependencies,
                "_materialize_into",
                side_effect=make_staging,
            ), mock.patch.object(
                triton_dependencies,
                "verify",
                return_value=manifest,
            ), mock.patch.object(
                triton_dependencies,
                "_rename_no_replace",
                side_effect=create_competitor_then_rename,
            ):
                with self.assertRaises(FileExistsError):
                    triton_dependencies.materialize(output)
            self.assertEqual((output / "payload").read_bytes(), b"competing")
            self.assertEqual(len(staged), 1)
            self.assertEqual((staged[0] / "payload").read_bytes(), b"staged")

if __name__ == "__main__":
    unittest.main()
