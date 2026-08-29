from __future__ import annotations

import importlib.util
import pathlib
import signal
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def load_tool(name: str, source: str) -> ModuleType:
    path = TOOLS / source
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(TOOLS))
try:
    cpu = load_tool("test_run_pypto_cpu_bounded", "run_pypto_cpu_bounded.py")
    gpu = load_tool("test_run_pypto_gpu_bounded", "run_pypto_gpu_bounded.py")
finally:
    sys.path.remove(str(TOOLS))


class BoundedCpuCommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cmake = str((cpu.ENVIRONMENT / "bin/cmake").resolve(strict=True))
        cls.ctest = str((cpu.ENVIRONMENT / "bin/ctest").resolve(strict=True))
        cls.build = str((ROOT / "builds/pypto-paged-f34c3f5").resolve(strict=True))

    def test_policy_retires_22_gib_and_pins_parallel_24(self) -> None:
        self.assertEqual(cpu.PROCESS_SCHEMA_VERSION, 2)
        self.assertEqual(cpu.POLICY_SCHEMA_VERSION, 2)
        self.assertEqual(cpu.PAUSE_MEMORY_KIB, 12 * 1024 * 1024)
        self.assertEqual(cpu.RESUME_MEMORY_KIB, 13 * 1024 * 1024)
        self.assertEqual(cpu.PYTEST_LOW_MEMORY_RECORD_KIB, 12 * 1024 * 1024)
        self.assertEqual(cpu.PYTEST_EMERGENCY_MEMORY_KIB, 6 * 1024 * 1024)
        self.assertEqual(cpu.PYTEST_EMERGENCY_CONSECUTIVE_SAMPLES, 3)
        self.assertEqual(cpu.POLL_SECONDS, 0.2)
        self.assertFalse(hasattr(cpu, "LAUNCH_MEMORY_KIB"))
        self.assertIsNone(
            cpu.memory_policy_record(cpu.PYTEST_RESIDENT_MODE)[
                "launch_admission_floor_kib"
            ]
        )
        pytest_policy = cpu.memory_policy_record(cpu.PYTEST_RESIDENT_MODE)
        self.assertFalse(pytest_policy["pause_enabled"])
        self.assertIsNone(pytest_policy["pause_memory_floor_kib"])
        self.assertEqual(
            pytest_policy["emergency_abort_memory_floor_kib"], 6 * 1024 * 1024
        )
        self.assertEqual(pytest_policy["emergency_abort_consecutive_samples"], 3)
        self.assertFalse(pytest_policy["external_process_signals"])
        self.assertEqual(
            pytest_policy["pause_signal_scope"], "verified-owned-pgid-only"
        )
        self.assertEqual(
            pytest_policy["termination_signal_scope"],
            "verified-pgid-then-verified-session-residuals",
        )
        self.assertEqual(pytest_policy["rss_accounting_scope"], "owned-session-id")
        build_policy = cpu.memory_policy_record(cpu.PAUSE_DRAIN_MODE)
        self.assertTrue(build_policy["pause_enabled"])
        self.assertEqual(build_policy["pause_memory_floor_kib"], 12 * 1024 * 1024)
        self.assertEqual(build_policy["resume_memory_floor_kib"], 13 * 1024 * 1024)
        self.assertIsNone(build_policy["emergency_abort_memory_floor_kib"])

    def test_accepts_only_exact_workspace_build_and_ctest_commands(self) -> None:
        for command in (
            ["--", self.cmake, "--build", self.build, "--parallel", "24"],
            ["--", self.cmake, "--build", self.build, "--parallel=24"],
            ["--", self.ctest, "--test-dir", self.build, "-j24"],
            ["--", self.ctest, f"--test-dir={self.build}", "-j", "24"],
        ):
            with self.subTest(command=command):
                self.assertEqual(cpu.validate_command(command), command[1:])

    def test_rejects_missing_or_conflicting_parallelism(self) -> None:
        for command in (
            ["--", self.cmake, "--build", self.build],
            ["--", self.cmake, "--build", self.build, "--parallel", "2"],
            [
                "--",
                self.cmake,
                "--build",
                self.build,
                "--parallel",
                "24",
                "-j2",
            ],
            [
                "--",
                self.cmake,
                "--build",
                self.build,
                "--parallel=24",
                "--parallel=2",
            ],
            ["--", self.ctest, "--test-dir", self.build, "-j2"],
            ["--", self.ctest, "--test-dir", self.build, "-j24", "-j", "2"],
            ["--", self.ctest, "--test-dir", self.build, "--parallel", "24"],
        ):
            with self.subTest(command=command), self.assertRaises(cpu.BoundedCpuError):
                cpu.validate_command(command)

    def test_rejects_missing_or_escaped_build_directory(self) -> None:
        for command in (
            ["--", self.cmake, "--build", "/tmp", "--parallel", "24"],
            ["--", self.ctest, "-j24"],
            ["--", self.ctest, "--test-dir", "/tmp", "-j24"],
            [
                "--",
                self.ctest,
                "--test-dir",
                self.build,
                f"--test-dir={self.build}",
                "-j24",
            ],
        ):
            with self.subTest(command=command), self.assertRaises(cpu.BoundedCpuError):
                cpu.validate_command(command)

    def test_formal_cpu_environment_accepts_workspace_script_and_pytest_24(
        self,
    ) -> None:
        environment = cpu.validate_environment_profile("pypto-release", "pypto")
        python = str((environment / "bin/python").resolve(strict=True))
        script = str((ROOT / "tools/run_operator_regression.py").resolve(strict=True))
        self.assertEqual(
            cpu.validate_command(["--", python, "-B", script], environment),
            [python, "-B", script],
        )
        self.assertEqual(
            cpu.validate_command(
                ["--", python, "-B", "-m", "pytest", "-n24", "tests"],
                environment,
            ),
            [python, "-B", "-m", "pytest", "-n24", "tests"],
        )
        with self.assertRaises(cpu.BoundedCpuError):
            cpu.validate_command(
                ["--", python, "-B", "-m", "pytest", "-n2", "tests"],
                environment,
            )

    def test_workload_policy_classifies_build_and_resident_pytest(self) -> None:
        release = cpu.validate_environment_profile("pypto-release", "pypto")
        python = str((release / "bin/python").resolve(strict=True))
        direct_pytest = cpu.validate_command(
            ["--", python, "-B", "-m", "pytest", "-n24", "tests"], release
        )
        structure_worker = cpu.validate_command(
            [
                "--",
                python,
                "-B",
                str((ROOT / "tools/run_operator_regression.py").resolve(strict=True)),
                "--_worker-structure",
                "--_jobs",
                "24",
            ],
            release,
        )
        build_worker = cpu.validate_command(
            [
                "--",
                python,
                "-B",
                str((ROOT / "tools/build_release.py").resolve(strict=True)),
                "--_worker",
                "native",
                "--_jobs",
                "24",
            ],
            release,
        )
        self.assertEqual(
            cpu.workload_mode(direct_pytest, release), cpu.PYTEST_RESIDENT_MODE
        )
        self.assertEqual(
            cpu.workload_mode(structure_worker, release), cpu.PYTEST_RESIDENT_MODE
        )
        self.assertEqual(cpu.workload_mode(build_worker, release), cpu.PAUSE_DRAIN_MODE)
        changed_jobs = list(structure_worker)
        changed_jobs[changed_jobs.index("--_jobs") + 1] = "2"
        with self.assertRaises(cpu.BoundedCpuError):
            cpu.workload_mode(changed_jobs, release)
        changed_stage = list(build_worker)
        changed_stage[changed_stage.index("--_worker") + 1] = "unknown"
        with self.assertRaises(cpu.BoundedCpuError):
            cpu.workload_mode(changed_stage, release)
        for command in (
            [self.cmake, "--build", self.build, "--parallel", "24"],
            [self.ctest, "--test-dir", self.build, "-j24"],
        ):
            self.assertEqual(cpu.workload_mode(command), cpu.PAUSE_DRAIN_MODE)

    def test_pytest_low_memory_never_pauses_and_three_emergency_samples_abort(
        self,
    ) -> None:
        count = 0
        for available in (
            cpu.PYTEST_LOW_MEMORY_RECORD_KIB - 1,
            cpu.PYTEST_EMERGENCY_MEMORY_KIB,
        ):
            action, paused, count = cpu.memory_policy_update(
                cpu.PYTEST_RESIDENT_MODE,
                available,
                paused=False,
                consecutive_emergency_samples=count,
            )
            self.assertIsNone(action)
            self.assertFalse(paused)
            self.assertEqual(count, 0)
        for expected_count in (1, 2):
            action, paused, count = cpu.memory_policy_update(
                cpu.PYTEST_RESIDENT_MODE,
                cpu.PYTEST_EMERGENCY_MEMORY_KIB - 1,
                paused=False,
                consecutive_emergency_samples=count,
            )
            self.assertIsNone(action)
            self.assertFalse(paused)
            self.assertEqual(count, expected_count)
        action, paused, count = cpu.memory_policy_update(
            cpu.PYTEST_RESIDENT_MODE,
            cpu.PYTEST_EMERGENCY_MEMORY_KIB - 1,
            paused=False,
            consecutive_emergency_samples=count,
        )
        self.assertEqual(action, "abort")
        self.assertFalse(paused)
        self.assertEqual(count, 3)
        self.assertEqual(
            cpu.low_memory_sample_record(
                cpu.PYTEST_RESIDENT_MODE,
                sample_index=7,
                available_kib=cpu.PYTEST_EMERGENCY_MEMORY_KIB - 1,
                owned_sid_rss_kib=14 * 1024 * 1024,
                consecutive_emergency_samples=count,
            ),
            {
                "sample_index": 7,
                "mem_available_kib": cpu.PYTEST_EMERGENCY_MEMORY_KIB - 1,
                "owned_sid_rss_kib": 14 * 1024 * 1024,
                "consecutive_emergency_samples": 3,
            },
        )
        self.assertIsNone(
            cpu.low_memory_sample_record(
                cpu.PAUSE_DRAIN_MODE,
                sample_index=7,
                available_kib=0,
                owned_sid_rss_kib=0,
                consecutive_emergency_samples=0,
            )
        )
        with self.assertRaises(cpu.BoundedCpuError):
            cpu.memory_policy_update(
                cpu.PYTEST_RESIDENT_MODE,
                cpu.PYTEST_EMERGENCY_MEMORY_KIB - 1,
                paused=True,
                consecutive_emergency_samples=0,
            )

    def test_cmake_ctest_policy_still_pauses_and_resumes_with_hysteresis(self) -> None:
        action, paused, count = cpu.memory_policy_update(
            cpu.PAUSE_DRAIN_MODE,
            cpu.PAUSE_MEMORY_KIB - 1,
            paused=False,
            consecutive_emergency_samples=0,
        )
        self.assertEqual((action, paused, count), ("pause", True, 0))
        action, paused, count = cpu.memory_policy_update(
            cpu.PAUSE_DRAIN_MODE,
            cpu.RESUME_MEMORY_KIB - 1,
            paused=paused,
            consecutive_emergency_samples=count,
        )
        self.assertEqual((action, paused, count), (None, True, 0))
        action, paused, count = cpu.memory_policy_update(
            cpu.PAUSE_DRAIN_MODE,
            cpu.RESUME_MEMORY_KIB,
            paused=paused,
            consecutive_emergency_samples=count,
        )
        self.assertEqual((action, paused, count), ("resume", False, 0))

    def test_owned_session_rss_includes_workers_with_distinct_pgids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = pathlib.Path(directory)

            def process(pid: int, pgid: int, sid: int, rss_kib: int) -> None:
                root = proc / str(pid)
                root.mkdir()
                fields = ["S", "1", str(pgid), str(sid), *(["0"] * 15), "123"]
                (root / "stat").write_text(
                    f"{pid} (pytest worker {pid}) " + " ".join(fields)
                )
                (root / "status").write_text(f"Name:\tworker\nVmRSS:\t{rss_kib} kB\n")

            process(101, 501, 900, 100)
            process(102, 502, 900, 200)
            process(103, 503, 901, 400)
            self.assertEqual(cpu.process_stat(101, proc), (501, 900, 123))
            self.assertEqual(cpu.owned_sid_rss_kib(900, proc), 300)

    def test_cpu_termination_signals_only_the_verified_target_group(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        metadata = {"pid": 101, "pgid": 101, "sid": 101, "start_ticks": 7}
        with (
            mock.patch.object(cpu.stop_run, "signal_verified") as verified,
            mock.patch.object(
                cpu.stop_run,
                "terminate_verified_session_residuals",
                return_value={"complete": True},
            ),
        ):
            cpu.terminate_owned(metadata, process)
        self.assertEqual(
            verified.call_args_list,
            [
                mock.call(metadata, signal.SIGCONT),
                mock.call(metadata, signal.SIGTERM),
            ],
        )

    def test_cpu_environment_profile_pairs_are_allowlisted(self) -> None:
        self.assertEqual(
            cpu.validate_environment_profile("sglang-baseline", "baseline"),
            (ROOT / "envs/sglang-baseline").resolve(),
        )
        with self.assertRaises(cpu.BoundedCpuError):
            cpu.validate_environment_profile("sglang-baseline", "pypto")


class BoundedGpuPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.python = str((gpu.ENVIRONMENT / "bin/python").resolve(strict=True))
        cls.script = str((ROOT / "tools/run_pypto_gpu_bounded.py").resolve(strict=True))

    def test_policy_has_runtime_safety_floors_not_22_gib_admission(self) -> None:
        self.assertEqual(gpu.PROCESS_SCHEMA_VERSION, 2)
        self.assertEqual(gpu.HOST_ABORT_KIB, 12 * 1024 * 1024)
        self.assertEqual(gpu.HOST_EMERGENCY_ABORT_KIB, 11 * 1024 * 1024)
        self.assertEqual(gpu.HOST_FLOOR_CONSECUTIVE_SAMPLES, 3)
        self.assertEqual(gpu.GPU_FREE_FLOOR_MIB, 4 * 1024)
        self.assertEqual(gpu.POLL_SECONDS, 1)
        self.assertFalse(hasattr(gpu, "LAUNCH_MEMORY_KIB"))

    def test_short_tmp_alias_preserves_owned_run_storage(self) -> None:
        with (
            tempfile.TemporaryDirectory(dir=ROOT / "runs") as run,
            tempfile.TemporaryDirectory() as parent_root,
        ):
            target = pathlib.Path(run) / "tmp"
            target.mkdir()
            parent, alias, observed_target = gpu.create_short_tmp_alias(
                pathlib.Path(run), parent_root=pathlib.Path(parent_root)
            )
            self.assertTrue(alias.is_symlink())
            self.assertEqual(observed_target, target.resolve())
            self.assertEqual(alias.resolve(), target.resolve())
            gpu.remove_short_tmp_alias(parent, alias, observed_target)
            self.assertFalse(parent.exists())

    def test_gpu_controller_rss_counts_the_owned_session_not_only_pgid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = pathlib.Path(directory)

            def process(pid: int, pgid: int, sid: int, rss_kib: int) -> None:
                root = proc / str(pid)
                root.mkdir()
                fields = ["S", "1", str(pgid), str(sid), *(["0"] * 15), "123"]
                (root / "stat").write_text(
                    f"{pid} (tileiras worker {pid}) " + " ".join(fields)
                )
                (root / "status").write_text(f"VmRSS:\t{rss_kib} kB\n")

            process(201, 701, 990, 300)
            process(202, 702, 990, 400)
            process(203, 703, 991, 800)
            self.assertEqual(gpu.process_stat_full(201, proc), (701, 990, 123))
            self.assertEqual(gpu.owned_sid_rss_kib(990, proc), 700)

    def test_host_floor_debounces_noise_but_emergency_aborts_immediately(self) -> None:
        just_below = gpu.HOST_ABORT_KIB - 1
        reason, count = gpu.host_floor_update(just_below, 0)
        self.assertIsNone(reason)
        self.assertEqual(count, 1)
        reason, count = gpu.host_floor_update(gpu.HOST_ABORT_KIB, count)
        self.assertIsNone(reason)
        self.assertEqual(count, 0)
        for expected_count in (1, 2):
            reason, count = gpu.host_floor_update(just_below, count)
            self.assertIsNone(reason)
            self.assertEqual(count, expected_count)
        reason, count = gpu.host_floor_update(just_below, count)
        self.assertEqual(reason, "host-memory-floor")
        self.assertEqual(count, 3)
        reason, _count = gpu.host_floor_update(gpu.HOST_EMERGENCY_ABORT_KIB - 1, 0)
        self.assertEqual(reason, "host-memory-emergency-floor")

    def test_nvidia_telemetry_allows_one_failure_then_fails_closed(self) -> None:
        self.assertEqual(gpu.NVIDIA_AUDIT_FAILURE_CONSECUTIVE_SAMPLES, 2)
        reason, count = gpu.nvidia_audit_failure_update(0)
        self.assertIsNone(reason)
        self.assertEqual(count, 1)
        reason, count = gpu.nvidia_audit_failure_update(count)
        self.assertEqual(reason, "nvidia-telemetry-unavailable")
        self.assertEqual(count, 2)

    def test_nvidia_telemetry_failure_record_preserves_timeout_boundary(self) -> None:
        error = gpu.subprocess.TimeoutExpired(["nvidia-smi", "--query-gpu=x"], 10)
        self.assertEqual(
            gpu.nvidia_audit_failure_record(
                error, phase="runtime", sample_index=17
            ),
            {
                "phase": "runtime",
                "sample_index": 17,
                "error_type": "TimeoutExpired",
                "error": (
                    "Command '['nvidia-smi', '--query-gpu=x']' timed out "
                    "after 10 seconds"
                ),
                "command": ["nvidia-smi", "--query-gpu=x"],
                "timeout_seconds": 10,
            },
        )

    def test_child_must_be_selected_python_and_workspace_script(self) -> None:
        command = ["--", self.python, "-B", self.script, "--example"]
        self.assertEqual(gpu.validate_child(command), command[1:])
        for changed in (
            command[1:],
            ["--", self.python, "-I", self.script],
            ["--", "/usr/bin/python3", "-B", self.script],
        ):
            with self.subTest(command=changed), self.assertRaises(gpu.BoundedGpuError):
                gpu.validate_child(changed)
        with tempfile.TemporaryDirectory() as directory:
            outside = pathlib.Path(directory) / "outside.py"
            outside.write_text("raise SystemExit(0)\n")
            with self.assertRaises(gpu.BoundedGpuError):
                gpu.validate_child(["--", self.python, "-B", str(outside)])

    def test_release_gpu_environment_profile_pairs_are_allowlisted(self) -> None:
        for environment_name, profile in (
            ("pypto-release", "pypto"),
            ("sglang-baseline", "baseline"),
        ):
            with self.subTest(environment=environment_name):
                environment = gpu.validate_environment_profile(
                    environment_name, profile
                )
                python = str((environment / "bin/python").resolve(strict=True))
                command = ["--", python, "-B", self.script]
                self.assertEqual(gpu.validate_child(command, environment), command[1:])
        with self.assertRaises(gpu.BoundedGpuError):
            gpu.validate_environment_profile("sglang-baseline", "pypto")

    @staticmethod
    def base_audit() -> dict[str, object]:
        return {
            "gpu_free_mib": 20 * 1024,
            "external_compute_pids": [],
            "protected_compute_pids": [],
            "protected_runtime_mapping_pids": [],
            "unreadable_protected_maps": [],
            "owned_compute_pids": [],
        }

    def test_audit_ok_is_fail_closed_for_external_or_post_exit_gpu_use(self) -> None:
        report = self.base_audit()
        self.assertTrue(gpu.audit_ok(report, child_running=False))
        for key, value in (
            ("gpu_free_mib", gpu.GPU_FREE_FLOOR_MIB - 1),
            ("external_compute_pids", [71]),
            ("protected_compute_pids", [72]),
            ("protected_runtime_mapping_pids", [73]),
            ("unreadable_protected_maps", [74]),
        ):
            changed = {**report, key: value}
            with self.subTest(key=key):
                self.assertFalse(gpu.audit_ok(changed, child_running=False))
        owned = {**report, "owned_compute_pids": [75]}
        self.assertTrue(gpu.audit_ok(owned, child_running=True))
        self.assertFalse(gpu.audit_ok(owned, child_running=False))

    def test_audit_distinguishes_owned_external_and_protected_compute(self) -> None:
        protected = SimpleNamespace(pid=22)
        with (
            mock.patch.object(
                gpu.preflight,
                "nvidia_identity",
                return_value={"memory_mib": "24576", "used_mib": "1024"},
            ),
            mock.patch.object(
                gpu.preflight,
                "process_table",
                return_value=([protected], [protected], []),
            ),
            mock.patch.object(
                gpu.preflight, "nvidia_compute_pids", return_value={11, 22}
            ),
            mock.patch.object(
                gpu.preflight,
                "protected_nvidia_runtime_mappings",
                return_value=([], []),
            ),
            mock.patch.object(
                gpu,
                "process_environment",
                side_effect=lambda pid: {"PYPTO_RUN_ID": "owned"} if pid == 11 else {},
            ),
        ):
            report = gpu.audit("owned")
        self.assertEqual(report["owned_compute_pids"], [11])
        self.assertEqual(report["external_compute_pids"], [22])
        self.assertEqual(report["protected_compute_pids"], [22])

    def test_audit_retains_owned_pgid_when_exit_environment_disappears(self) -> None:
        with (
            mock.patch.object(
                gpu.preflight,
                "nvidia_identity",
                return_value={"memory_mib": "24576", "used_mib": "1024"},
            ),
            mock.patch.object(
                gpu.preflight, "process_table", return_value=([], [], [])
            ),
            mock.patch.object(
                gpu.preflight, "nvidia_compute_pids", return_value={91, 92}
            ),
            mock.patch.object(
                gpu.preflight,
                "protected_nvidia_runtime_mappings",
                return_value=([], []),
            ),
            mock.patch.object(
                gpu,
                "process_stat_full",
                side_effect=lambda pid: (
                    700 if pid == 91 else 800,
                    900 if pid == 91 else 901,
                    1,
                ),
            ),
            mock.patch.object(
                gpu, "process_environment", side_effect=FileNotFoundError
            ),
        ):
            report = gpu.audit("owned", owned_pgid=700)
        self.assertEqual(report["owned_compute_pids"], [91])
        self.assertEqual(report["external_compute_pids"], [92])

    def test_termination_signals_only_verified_owned_metadata(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        metadata = {"pid": 101, "pgid": 101, "sid": 101, "start_ticks": 7}
        with (
            mock.patch.object(gpu.stop_run, "signal_verified") as verified,
            mock.patch.object(
                gpu.stop_run,
                "terminate_verified_session_residuals",
                return_value={"complete": True},
            ),
        ):
            gpu.terminate_owned(metadata, process)
        verified.assert_called_once_with(metadata, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
