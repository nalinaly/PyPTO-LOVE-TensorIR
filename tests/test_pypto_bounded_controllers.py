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
        self.assertEqual(cpu.PAUSE_MEMORY_KIB, 16 * 1024 * 1024)
        self.assertEqual(cpu.RESUME_MEMORY_KIB, 17 * 1024 * 1024)
        self.assertEqual(cpu.POLL_SECONDS, 0.2)
        self.assertFalse(hasattr(cpu, "LAUNCH_MEMORY_KIB"))

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
        self.assertEqual(gpu.HOST_ABORT_KIB, 16 * 1024 * 1024)
        self.assertEqual(gpu.HOST_EMERGENCY_ABORT_KIB, 15 * 1024 * 1024)
        self.assertEqual(gpu.HOST_FLOOR_CONSECUTIVE_SAMPLES, 3)
        self.assertEqual(gpu.GPU_FREE_FLOOR_MIB, 4 * 1024)
        self.assertEqual(gpu.POLL_SECONDS, 1)
        self.assertFalse(hasattr(gpu, "LAUNCH_MEMORY_KIB"))

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
                "process_stat",
                side_effect=lambda pid: (700 if pid == 91 else 800, 1),
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
        metadata = {"pid": 101, "pgid": 101, "start_ticks": 7}
        with mock.patch.object(gpu.stop_run, "signal_verified") as verified:
            gpu.terminate_owned(metadata, process)
        verified.assert_called_once_with(metadata, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
