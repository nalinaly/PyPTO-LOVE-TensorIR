from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import pathlib
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_source(name: str, relative: str) -> ModuleType:
    path = (ROOT / relative).resolve(strict=True)
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


_CANONICAL_BASE_NAMES = (
    "_pypto_nvidia_executable_sm120_contract",
    "_pypto_nvidia_sm120_control_manifest",
    "preflight",
    "stop_run",
)
_saved_base_modules = {
    name: sys.modules.pop(name)
    for name in _CANONICAL_BASE_NAMES
    if name in sys.modules
}
try:
    contract = load_source(
        "test_cpu_v2_contract", "tools/_pypto_cpu_coexistence_v2_contract.py"
    )
    preflight = load_source(
        "test_cpu_v2_preflight", "tools/preflight_cpu_coexistence_v2.py"
    )
    control = load_source(
        "test_cpu_v2_control", "tools/_pypto_cpu_coexistence_v2_control_manifest.py"
    )
    controller = load_source(
        "test_cpu_v2_controller", "tools/run_pypto_cpu_coexistence_v2_isolated.py"
    )
finally:
    for name in _CANONICAL_BASE_NAMES:
        sys.modules.pop(name, None)
    sys.modules.update(_saved_base_modules)


class ContractAndPreflightTest(unittest.TestCase):
    def test_exact_base_dependencies_and_thresholds(self) -> None:
        for path, size, digest in (
            (
                ROOT / contract.BASE_PREFLIGHT_RELATIVE_PATH,
                contract.BASE_PREFLIGHT_SIZE,
                contract.BASE_PREFLIGHT_SHA256,
            ),
            (
                ROOT / contract.BASE_ISOLATION_RELATIVE_PATH,
                contract.BASE_ISOLATION_SIZE,
                contract.BASE_ISOLATION_SHA256,
            ),
            (
                ROOT / contract.BASE_STOP_RELATIVE_PATH,
                contract.BASE_STOP_SIZE,
                contract.BASE_STOP_SHA256,
            ),
            (
                ROOT / contract.BASE_NVIDIA_CONTRACT_RELATIVE_PATH,
                contract.BASE_NVIDIA_CONTRACT_SIZE,
                contract.BASE_NVIDIA_CONTRACT_SHA256,
            ),
            (
                ROOT / contract.BASE_NVIDIA_CONTROL_RELATIVE_PATH,
                contract.BASE_NVIDIA_CONTROL_SIZE,
                contract.BASE_NVIDIA_CONTROL_SHA256,
            ),
            (
                ROOT / contract.BASE_NVIDIA_MANIFEST_RELATIVE_PATH,
                contract.BASE_NVIDIA_MANIFEST_SIZE,
                contract.BASE_NVIDIA_MANIFEST_SHA256,
            ),
        ):
            raw = path.read_bytes()
            self.assertEqual(len(raw), size)
            self.assertEqual(contract.sha256_bytes(raw), digest)
        self.assertEqual(contract.LAUNCH_MEMORY_FLOOR_KIB, 22 * 1024 * 1024)
        self.assertEqual(contract.RESUME_MEMORY_FLOOR_KIB, 22 * 1024 * 1024)
        self.assertEqual(contract.PAUSE_MEMORY_FLOOR_KIB, 16 * 1024 * 1024)
        self.assertEqual(controller.base.COEXISTENCE_RESUME_MEMORY_KIB, 24 * 1024 * 1024)
        self.assertEqual(controller.base.COEXISTENCE_ABORT_MEMORY_KIB, 16 * 1024 * 1024)
        self.assertIs(controller.base.preflight_tool, controller.base_preflight)
        self.assertIs(controller.base.stop_run, controller.stop_run)
        self.assertIs(
            controller.base.nvidia_smoke_contract,
            controller.base_nvidia_contract,
        )
        self.assertIs(
            controller.base.nvidia_smoke_control,
            controller.base_nvidia_control,
        )

    def _report(self, *, memory_kib: int, compute: set[int] | None = None):
        protected = [
            preflight.ProcessInfo(
                pid=77,
                ppid=1,
                start_ticks=10,
                rss_kib=1,
                command="gem5.opt",
                cwd="/home/zhaosiying/amdgpu-sim",
            )
        ]
        static = {"hip": None, "forbidden_dsos": [], "cuda_initialized": False}
        gpu = {
            "name": contract.EXPECTED_DEVICE_NAME,
            "compute_capability": "12.0",
            "memory_mib": "24576",
            "used_mib": "0",
            "driver": contract.EXPECTED_DRIVER_RELEASE,
        }
        with (
            mock.patch.dict(preflight.os.environ, {}, clear=True),
            mock.patch.object(
                preflight, "process_table", return_value=(protected, protected, [])
            ),
            mock.patch.object(preflight, "mem_available_kib", return_value=memory_kib),
            mock.patch.object(preflight, "nvidia_identity", return_value=gpu),
            mock.patch.object(preflight, "static_torch_identity", return_value=static),
            mock.patch.object(
                preflight, "nvidia_compute_pids", return_value=(compute or set())
            ),
            mock.patch.object(
                preflight,
                "protected_nvidia_runtime_mappings",
                return_value=([], []),
            ),
        ):
            return preflight.build_report()

    def test_preflight_uses_22_gib_and_is_observation_only(self) -> None:
        accepted = self._report(memory_kib=contract.LAUNCH_MEMORY_FLOOR_KIB)
        self.assertIs(accepted["ok"], True)
        self.assertIs(accepted["protected_activity_waiver_applied"], True)
        below = self._report(memory_kib=contract.LAUNCH_MEMORY_FLOOR_KIB - 1)
        self.assertIs(below["ok"], False)
        self.assertTrue(any("below 22 GiB" in item for item in below["failures"]))
        protected_compute = self._report(
            memory_kib=contract.LAUNCH_MEMORY_FLOOR_KIB, compute={77}
        )
        self.assertIs(protected_compute["ok"], False)
        self.assertIn(77, protected_compute["protected_nvidia_compute_pids"])
        self.assertEqual(
            accepted["policy"], "observation-only; no external process is ever signalled"
        )

    def test_report_schema_and_command_policy_are_fail_closed(self) -> None:
        report = self._report(memory_kib=contract.LAUNCH_MEMORY_FLOOR_KIB)
        preflight.validate_report(report, description="fixture")
        candidate = copy.deepcopy(report)
        candidate["resume_memory_floor_kib"] += 1
        with self.assertRaises(preflight.PreflightError):
            preflight.validate_report(candidate, description="fixture")
        candidate = copy.deepcopy(report)
        candidate["ok"] = True
        candidate["failures"] = []
        candidate["mem_available_kib"] = contract.LAUNCH_MEMORY_FLOOR_KIB - 1
        candidate["protected_activity_waiver_applied"] = True
        with self.assertRaises(preflight.PreflightError):
            preflight.validate_report(candidate, description="fixture")
        self.assertEqual(contract.validate_command(["/usr/bin/true"]), ["/usr/bin/true"])
        for command in (
            ["/bin/bash", "-c", "true"],
            ["/usr/bin/env", "python", "worker.py"],
            ["pkill", "python"],
            ["python", "/home/zhaosiying/amdgpu-sim/worker.py"],
            ["python", "-m", "sglang.launch_server"],
            ["torchrun", "model.py"],
            ["nvidia-smi"],
            ["runner", "--mode", "gpu-benchmark"],
            ["python3", "-m", "vllm"],
            ["python3.12", "-m", "vllm", "serve", "model"],
            ["pypy3", "-m", "sglang", "serve"],
            ["python3", "-m", "torch.distributed"],
            ["python3", "-m", "deepspeed"],
            ["python3", "-m", "ray", "start"],
            ["python3", "-cprint(1)"],
            ["python3", "-c", "print(1)"],
            ["python3", "-Bc", "print(1)"],
            ["python3", "-Sc", "import os"],
            ["python3", "-Bcprint(1)"],
            ["python3", "-Scimport os"],
            ["python3", "-Bmtorch.distributed.launch", "--nproc_per_node", "2"],
            ["python3", "-mray.scripts", "start"],
            ["python3", "-u-"],
            ["python3", "-B-"],
            ["python3"],
            ["python3", "-i"],
            ["python3", "--"],
            ["python3", "-W", "ignore"],
            ["python3", "-X", "utf8"],
            ["python3", "--check-hash-based-pycs", "never"],
            ["python3", "-uW", "ignore"],
            ["python3", "-q", "-X", "utf8"],
            ["python3", "--", "-"],
            ["python3", "-u", "--", "-"],
            ["python3", "-W", "ignore", "--", "-"],
            ["python3", "-m", ".ray"],
            ["python3", "-m", "123"],
            ["pypy3", "-c", "print(1)"],
            ["python3", "-Bm", "ray"],
            ["python3", "-Bm", "torch.distributed.launch", "--nproc_per_node", "2"],
            ["python3", "-mray"],
            ["python3", "-m"],
            ["python", "x.py", "gpu", "benchmark"],
            ["python", "x.py", "nvidia", "smi"],
            ["python", "x.py", "amdgpu", "sim"],
            ["python", "x.py", "gpu\tbenchmark"],
            ["python", "t.py", "torch", "distributed", "run"],
            ["python", "t.py", "ra", "y start"],
            ["python3", "-m", " ray"],
            ["python3", "-m", " torch.distributed"],
            ["python3", "-"],
            ["xargs", "kill", "-9", "1"],
            ["perl", "-e", "kill 9, 1"],
            ["nice", "-n", "5", "python", "worker.py"],
            ["python3", "run_engine", "_lane.sh"],
            ["python3", "run_model", "_lane.sh"],
            ["vllm", "serve", "model"],
            ["./vllm", "serve", "model"],
            ["sglang", "launch_server"],
            ["sglang_router", "--host", "0.0.0.0"],
            ["python3", "-m", "Vllm"],
            ["python3", "-m", "vllm_x"],
            ["python3", "-m", "sglang_router"],
            ["/usr/bin/find", ".", "-exec", "kill", "{}", ";"],
            ["node", "-e", "process.kill(1)"],
            ["busybox", "sh"],
            ["python", "tool.py", "-f../zcode-lane/CHECKPOINT.md"],
            ["python", "tool.py", "--out=../amdgpu-sim/report.txt"],
            ["python", "tools/preflight_gpu_smoke_v2.py"],
            ["python", "benchmarks/gpu_benchmark.py", "--case", "9b"],
            ["python", "tools/nvidia_smi_probe.py"],
        ):
            with self.subTest(command=command), self.assertRaises(contract.ContractError):
                contract.validate_command(command)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            protected_link = pathlib.Path(directory) / "protected-link"
            protected_link.symlink_to(
                pathlib.Path("/home/zhaosiying/amdgpu-sim"),
                target_is_directory=True,
            )
            with self.assertRaises(contract.ContractError):
                contract.validate_command(["python", str(protected_link)])
            shadow = pathlib.Path(directory) / "zz-worker"
            shadow.symlink_to("/bin/sh")
            with self.assertRaises(contract.ContractError):
                contract.validate_command([str(shadow), "-c", "true"])
            if pathlib.Path("/usr/bin/nvidia-smi").exists():
                gpu_tool = pathlib.Path(directory) / "nvsmi"
                gpu_tool.symlink_to("/usr/bin/nvidia-smi")
                with self.assertRaises(contract.ContractError):
                    contract.validate_command([str(gpu_tool)])
            with (
                mock.patch.object(contract.shutil, "which", return_value="/bin/bash"),
                self.assertRaises(contract.ContractError),
            ):
                contract.validate_command(["zz-helper", "-c", "true"])
            interpreter = ROOT / "envs/pypto-nvidia/bin/python3.14"
            with mock.patch.object(contract.shutil, "which", return_value=str(interpreter)):
                self.assertEqual(
                    contract.validate_command(["zz-runner", "-B", "-m", "pytest", "-q"]),
                    ["zz-runner", "-B", "-m", "pytest", "-q"],
                )
            self.assertEqual(
                contract.validate_command(
                    [
                        str(interpreter),
                        "-B",
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "tests",
                    ]
                ),
                [
                    str(interpreter),
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "tests",
                ],
            )
            bare_link = pathlib.Path(directory) / "protected-bare-link"
            bare_link.symlink_to(
                pathlib.Path("/home/zhaosiying/amdgpu-sim"),
                target_is_directory=True,
            )
            with (
                mock.patch.object(contract, "ROOT", pathlib.Path(directory).resolve()),
                self.assertRaises(contract.ContractError),
            ):
                contract.validate_command(["python", "protected-bare-link"])


class ControllerTransactionTest(unittest.TestCase):
    @staticmethod
    def _report() -> dict[str, object]:
        protected = {
            "pid": 77,
            "ppid": 1,
            "start_ticks": 10,
            "rss_kib": 1,
            "command": "gem5.opt",
            "cwd": "/home/zhaosiying/amdgpu-sim",
        }
        return {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": contract.POLICY_KIND,
            "mode": contract.MODE,
            "workspace": str(ROOT),
            "cwd": str(ROOT),
            "ok": True,
            "failures": [],
            "mem_available_kib": contract.LAUNCH_MEMORY_FLOOR_KIB,
            "launch_memory_floor_kib": contract.LAUNCH_MEMORY_FLOOR_KIB,
            "resume_memory_floor_kib": contract.RESUME_MEMORY_FLOOR_KIB,
            "pause_memory_floor_kib": contract.PAUSE_MEMORY_FLOOR_KIB,
            "protected_cpu_only_coexistence_requested": True,
            "protected_activity_waiver_applied": True,
            "gpu": {
                "name": contract.EXPECTED_DEVICE_NAME,
                "compute_capability": contract.EXPECTED_COMPUTE_CAPABILITY,
                "memory_mib": "24576",
                "used_mib": "0",
                "driver": contract.EXPECTED_DRIVER_RELEASE,
            },
            "torch": {"hip": None, "forbidden_dsos": []},
            "protected_processes": [protected],
            "protected_heavy_processes": [protected],
            "workspace_processes": [],
            "nvidia_compute_audit_ok": True,
            "nvidia_compute_pids": [],
            "protected_nvidia_compute_pids": [],
            "protected_nvidia_runtime_mapping_pids": [],
            "unreadable_protected_maps": [],
            "policy": "observation-only; no external process is ever signalled",
            "admission_policy": preflight.policy_document(),
        }

    def _run_main(
        self,
        post_exit: dict[str, object] | BaseException | None = None,
        *,
        disk_free_bytes: int = 1 << 50,
        getpgid_error: bool = False,
        lease_busy: bool = False,
        leader_exited_at_start: bool = False,
        metadata_fail_always: bool = False,
        metadata_fail_once: bool = False,
        popen_error: bool = False,
        survivor_members: list[int] | None = None,
        survivor_error: bool = False,
        reject_preflight: str | None = None,
        watchdog_interrupt_after_leader_exit: bool = False,
        watchdog_returns_paused: bool = False,
    ) -> tuple[int, list[str], dict[str, object]]:
        events: list[str] = []
        writes: dict[str, list[dict[str, object]]] = {}
        transaction: dict[str, object] = {"writes": writes}
        if post_exit is None:
            post_exit = {
                "nvidia_compute_pids": [],
                "owned_nvidia_compute_pids": [],
                "external_nvidia_compute_pids": [],
                "protected_nvidia_compute_pids": [],
                "workspace_nvidia_compute_pids": [],
                "protected_nvidia_runtime_mapping_pids": [],
                "unreadable_protected_maps": [],
            }

        class Lease:
            descriptor = 9
            path = ROOT / "runs/environment-pypto-nvidia.lock"
            mode = "shared"
            device = 1
            inode = 2

            def close(self) -> None:
                events.append("lease-close")

        class Process:
            pid = 1234
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, timeout=None) -> int:
                if self.returncode is None:
                    raise subprocess.TimeoutExpired(["fixture"], timeout)
                return self.returncode

        process = Process()
        checks = 0
        metadata_calls = 0
        survivor_calls = 0
        real_cpu_environment = controller.cpu_environment
        real_build_metadata = controller.build_metadata

        def preflight_side_effect(*, description: str):
            nonlocal checks
            checks += 1
            events.append("initial-preflight" if checks == 1 else "action-preflight")
            stage = "initial" if checks == 1 else "action"
            if reject_preflight == stage:
                return 75, {"ok": False}
            report = copy.deepcopy(self._report())
            preflight.validate_report(report, description=f"{stage} fixture")
            return 0, report

        def atomic_side_effect(path, value):
            events.append(f"write:{path.name}")
            writes.setdefault(path.name, []).append(copy.deepcopy(value))

        def popen_side_effect(*args, **kwargs):
            if popen_error:
                events.append("popen-failed")
                raise RuntimeError("popen failed")
            events.append("popen")
            transaction["popen_args"] = args
            transaction["popen_kwargs"] = kwargs
            if leader_exited_at_start:
                process.returncode = 0
            return process

        def metadata_side_effect(*_args, **kwargs):
            nonlocal metadata_calls
            metadata_calls += 1
            events.append("build-metadata")
            if metadata_fail_always or (metadata_fail_once and metadata_calls == 1):
                raise RuntimeError("metadata capture failed once")
            return real_build_metadata(*_args, **kwargs)

        def environment_side_effect(**kwargs):
            events.append("environment")
            return real_cpu_environment(**kwargs)

        def release_side_effect(*, path, metadata_path, metadata):
            events.append("release-start-gate")
            self.assertIs(metadata["metadata_complete"], True)
            self.assertEqual(metadata["start_gate"]["path"], str(path))
            metadata["start_gate"].update(
                {"released": True, "sha256": "b" * 64, "released_at": "fixture"}
            )
            controller.base.atomic_json(metadata_path, metadata)
            return {"released": True}

        def watchdog_side_effect(_process, metadata, **_kwargs):
            events.append("watchdog")
            process.returncode = 0
            if watchdog_interrupt_after_leader_exit:
                raise controller.base.RunInterrupted(signal.SIGTERM)
            if watchdog_returns_paused:
                metadata["status"] = "paused"
            return 0, False

        def post_exit_side_effect(_metadata):
            if isinstance(post_exit, BaseException):
                raise post_exit
            return copy.deepcopy(post_exit)

        def survivor_side_effect(_metadata):
            nonlocal survivor_calls
            survivor_calls += 1
            if survivor_error:
                raise RuntimeError("survivor ownership ambiguous")
            if survivor_members is not None and survivor_calls == 1:
                return list(survivor_members)
            return []

        def mask_side_effect(how, _values):
            events.append("mask" if how == signal.SIG_BLOCK else "unmask")
            return set()

        def terminate_side_effect(_process, _metadata, _metadata_path=None):
            events.append("terminate-owned")
            process.returncode = 75
            return 75

        fake_flags = SimpleNamespace(
            ignore_environment=1, no_site=1, dont_write_bytecode=1
        )
        argv = [
            "run_pypto_cpu_coexistence_v2_isolated.py",
            "--run-id-file",
            str(ROOT / "runs/cpu-v2-run-id.json"),
            "--",
            "/usr/bin/true",
        ]
        getpgid_patch = (
            mock.patch.object(controller.os, "getpgid", side_effect=ProcessLookupError)
            if getpgid_error
            else mock.patch.object(controller.os, "getpgid", return_value=process.pid)
        )
        patches = (
            mock.patch.object(
                controller.control,
                "validate_control_manifest",
                side_effect=lambda _root: events.append("manifest") or {"v2": True},
            ),
            mock.patch.object(
                controller,
                "acquire_shared_environment_lease",
                side_effect=lambda: events.append("lease")
                or (None if lease_busy else Lease()),
            ),
            mock.patch.object(controller, "publish_run_id", side_effect=lambda *_: events.append("run-id")),
            mock.patch.object(controller.pathlib.Path, "mkdir"),
            mock.patch.object(controller, "run_preflight", side_effect=preflight_side_effect),
            mock.patch.object(
                controller,
                "cpu_environment",
                side_effect=environment_side_effect,
            ),
            mock.patch.object(controller.base, "atomic_json", side_effect=atomic_side_effect),
            mock.patch.object(
                controller,
                "sha256_file",
                side_effect=lambda path: (
                    controller.base_nvidia_contract.PYTHON_SHA256
                    if pathlib.Path(path)
                    == ROOT / controller.base_nvidia_contract.PYTHON_REAL_RELATIVE_PATH
                    else "a" * 64
                ),
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=disk_free_bytes),
            ),
            mock.patch.object(controller.subprocess, "Popen", side_effect=popen_side_effect),
            mock.patch.object(controller, "build_metadata", side_effect=metadata_side_effect),
            mock.patch.object(
                controller, "release_start_gate", side_effect=release_side_effect
            ),
            getpgid_patch,
            mock.patch.object(controller.base, "process_start_ticks", return_value=5),
            mock.patch.object(controller.stop_run, "process_start_ticks", return_value=5),
            mock.patch.object(
                controller.stop_run,
                "verify",
                side_effect=lambda _metadata: events.append("verify"),
            ),
            mock.patch.object(controller, "wait_with_watchdog", side_effect=watchdog_side_effect),
            mock.patch.object(
                controller, "terminate_owned", side_effect=terminate_side_effect
            ),
            mock.patch.object(
                controller,
                "audit_runtime_state",
                side_effect=post_exit_side_effect,
            ),
            mock.patch.object(
                controller,
                "exact_owned_group_members",
                side_effect=survivor_side_effect,
            ),
            mock.patch.object(controller.signal, "getsignal", return_value=None),
            mock.patch.object(controller.signal, "signal"),
            mock.patch.object(
                controller.signal, "pthread_sigmask", side_effect=mask_side_effect
            ),
            mock.patch.object(controller, "print", create=True),
            mock.patch.object(controller.sys, "flags", fake_flags),
            mock.patch.object(controller.sys, "argv", argv),
        )
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            try:
                code = controller.main()
            except Exception as error:
                transaction["exception"] = error
                code = 75
            return code, events, transaction

    def test_full_transaction_publishes_ownership_before_unmask(self) -> None:
        code, events, transaction = self._run_main()
        self.assertEqual(code, 0)
        ordered = (
            "manifest",
            "lease",
            "run-id",
            "initial-preflight",
            "write:initial-preflight.json",
            "environment",
            "action-preflight",
            "write:preflight.json",
            "mask",
            "popen",
            "build-metadata",
            "write:process.json",
            "verify",
            "release-start-gate",
            "write:process.json",
            "unmask",
            "watchdog",
            "write:process.json",
            "lease-close",
        )
        self.assertEqual(tuple(events), ordered)
        launch = transaction["popen_args"][0]
        self.assertEqual(
            launch[:6],
            [
                str(ROOT / controller.base_nvidia_contract.PYTHON_REAL_RELATIVE_PATH),
                "-I",
                "-B",
                "-S",
                "-c",
                controller.START_GATE_PROGRAM,
            ],
        )
        self.assertEqual(launch[-1], "/usr/bin/true")
        kwargs = transaction["popen_kwargs"]
        self.assertEqual(kwargs["cwd"], ROOT)
        self.assertIs(kwargs["start_new_session"], True)
        self.assertTrue(callable(kwargs["preexec_fn"]))
        self.assertEqual(kwargs["pass_fds"], (9,))
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(kwargs["env"]["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertEqual(kwargs["env"]["PYTHONPATH"], "")
        self.assertEqual(
            kwargs["env"]["SGLANG_PLUGINS"], "__pypto_cpu_v2_no_plugins__"
        )
        self.assertEqual(kwargs["env"]["PYPTO_ENVIRONMENT_LOCK_FD"], "9")
        self.assertEqual(kwargs["env"]["PYPTO_WORKSPACE_ROOT"], str(ROOT))
        self.assertEqual(kwargs["env"]["PYPTO_CPU_COEXISTENCE_POLICY_VERSION"], "2")
        self.assertEqual(kwargs["env"]["PYPTO_ALLOW_FALLBACK"], "0")
        self.assertEqual(kwargs["env"]["PYPTO_STRICT_COVERAGE"], "1")
        self.assertFalse(
            any(
                name.startswith(("HSA_", "ROCR_", "GEMSIM_", "AMDGPU_SIM_"))
                for name in kwargs["env"]
            )
        )
        self.assertLess(events.index("write:process.json"), events.index("unmask"))
        self.assertLess(events.index("verify"), events.index("unmask"))
        first_metadata = transaction["writes"]["process.json"][0]
        self.assertEqual(first_metadata["command"], ["/usr/bin/true"])
        self.assertEqual(first_metadata["launch_command"], launch)
        self.assertIs(first_metadata["metadata_complete"], True)
        self.assertEqual(first_metadata["environment_access_lock"]["mode"], "shared")
        self.assertEqual(first_metadata["resource_policy"], {
            "launch_memory_floor_kib": 22 * 1024 * 1024,
            "resume_memory_floor_kib": 22 * 1024 * 1024,
            "pause_memory_floor_kib": 16 * 1024 * 1024,
            "timeout_seconds": contract.DEFAULT_TIMEOUT_SECONDS,
            "minimum_free_disk_bytes": contract.DEFAULT_MINIMUM_FREE_DISK_GIB << 30,
        })
        self.assertEqual(first_metadata["control_manifest"], {"v2": True})
        self.assertEqual(
            first_metadata["coexistence"]["policy"], preflight.policy_document()
        )
        self.assertIs(first_metadata["coexistence"]["waiver_applied"], True)
        self.assertEqual(
            first_metadata["initial_preflight"]["sha256"], "a" * 64
        )
        self.assertEqual(first_metadata["preflight"]["sha256"], "a" * 64)
        self.assertEqual(
            transaction["writes"]["process.json"][-1][
                "coexistence_post_exit_audit"
            ],
            WatchdogTest.safe_audit(),
        )

    def test_missing_manifest_blocks_before_popen(self) -> None:
        fake_flags = SimpleNamespace(
            ignore_environment=1, no_site=1, dont_write_bytecode=1
        )
        argv = [
            "run_pypto_cpu_coexistence_v2_isolated.py",
            "--run-id-file",
            str(ROOT / "runs/cpu-v2-run-id.json"),
            "--",
            "/usr/bin/true",
        ]
        with (
            mock.patch.object(controller.sys, "flags", fake_flags),
            mock.patch.object(controller.sys, "argv", argv),
            mock.patch.object(
                controller.control.contract,
                "MANIFEST_RELATIVE_PATH",
                pathlib.Path("state/contracts/pypto_cpu_coexistence_v2-absent.json"),
            ),
            mock.patch.object(controller, "print", create=True) as printed,
            mock.patch.object(controller.subprocess, "Popen") as popen,
        ):
            self.assertEqual(controller.main(), 75)
        popen.assert_not_called()
        self.assertTrue(
            any(
                "control manifest refusal" in str(call.args)
                for call in printed.call_args_list
            )
        )

    def test_lease_preflight_and_disk_rejections_never_create_child(self) -> None:
        cases = (
            {"lease_busy": True},
            {"reject_preflight": "initial"},
            {"reject_preflight": "action"},
            {"disk_free_bytes": 0},
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                code, events, transaction = self._run_main(**kwargs)
                self.assertEqual(code, 75)
                self.assertNotIn("popen", events)
                self.assertNotIn("popen_args", transaction)

    def test_post_exit_audit_failure_or_protected_activity_returns_75(self) -> None:
        code, _events, transaction = self._run_main(RuntimeError("audit failed"))
        self.assertEqual(code, 75)
        self.assertIn(
            "coexistence_post_exit_audit_error",
            transaction["writes"]["process.json"][-1],
        )
        protected = WatchdogTest.safe_audit()
        protected["protected_nvidia_compute_pids"] = [77]
        code, _events, transaction = self._run_main(protected)
        self.assertEqual(code, 75)
        self.assertEqual(
            transaction["writes"]["process.json"][-1][
                "coexistence_post_exit_audit"
            ]["protected_nvidia_compute_pids"],
            [77],
        )

    def test_survivor_ownership_ambiguity_returns_75_without_cleanup_signal(self) -> None:
        code, _events, transaction = self._run_main(survivor_error=True)
        self.assertEqual(code, 75)
        final = transaction["writes"]["process.json"][-1]
        self.assertEqual(final["status"], "group-ownership-ambiguous")
        self.assertIn("survivor ownership ambiguous", final["group_exit_error"])

    def test_verified_nonempty_survivors_use_owned_cleanup_and_exact_reaudit(self) -> None:
        code, events, transaction = self._run_main(survivor_members=[222])
        self.assertEqual(code, 75)
        self.assertIn("terminate-owned", events)
        final = transaction["writes"]["process.json"][-1]
        self.assertEqual(final["post_exit_group_members"], [222])
        self.assertEqual(final["post_cleanup_group_members"], [])
        self.assertEqual(final["surviving_group_cleanup_code"], 75)

    def test_paused_leader_exit_with_survivors_still_uses_verified_cleanup(self) -> None:
        code, events, transaction = self._run_main(
            survivor_members=[222], watchdog_returns_paused=True
        )
        self.assertEqual(code, 75)
        self.assertIn("terminate-owned", events)
        final = transaction["writes"]["process.json"][-1]
        self.assertEqual(final["post_exit_group_members"], [222])
        self.assertEqual(final["post_cleanup_group_members"], [])

    def test_interrupt_after_leader_exit_still_audits_and_cleans_recorded_group(
        self,
    ) -> None:
        code, events, transaction = self._run_main(
            survivor_members=[222],
            watchdog_interrupt_after_leader_exit=True,
        )
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertIn("terminate-owned", events)
        self.assertNotIn("exception", transaction)
        last_mask = len(events) - 1 - events[::-1].index("mask")
        last_unmask = len(events) - 1 - events[::-1].index("unmask")
        self.assertLess(events.index("watchdog"), last_mask)
        self.assertLess(last_mask, events.index("terminate-owned"))
        self.assertLess(events.index("terminate-owned"), last_unmask)

    def test_metadata_capture_failure_retries_durable_identity_before_termination(
        self,
    ) -> None:
        code, events, transaction = self._run_main(metadata_fail_once=True)
        self.assertEqual(code, 75)
        self.assertNotIn("exception", transaction)
        self.assertEqual(events.count("build-metadata"), 2)
        self.assertLess(events.index("write:process.json"), events.index("terminate-owned"))
        self.assertLess(events.index("verify"), events.index("unmask"))
        self.assertLess(events.index("terminate-owned"), events.index("unmask"))

    def test_persistent_metadata_failure_uses_verified_emergency_record(self) -> None:
        code, events, transaction = self._run_main(metadata_fail_always=True)
        self.assertEqual(code, 75)
        self.assertNotIn("exception", transaction)
        self.assertEqual(events.count("build-metadata"), 2)
        self.assertLess(events.index("write:process.json"), events.index("verify"))
        self.assertLess(events.index("verify"), events.index("terminate-owned"))
        self.assertLess(events.index("terminate-owned"), events.index("unmask"))
        emergency = transaction["writes"]["process.json"][0]
        self.assertIs(emergency["metadata_complete"], False)
        self.assertEqual(emergency["status"], "startup-ownership-recovery")

    def test_exited_start_gate_leader_still_gets_ownership_audit(self) -> None:
        code, events, transaction = self._run_main(
            leader_exited_at_start=True,
            metadata_fail_once=True,
        )
        self.assertEqual(code, 75)
        self.assertNotIn("watchdog", events)
        self.assertIn("verify", events)
        self.assertIn("terminate-owned", events)
        self.assertNotIn("exception", transaction)

    def test_popen_failure_records_startup_error_without_child(self) -> None:
        code, events, transaction = self._run_main(popen_error=True)
        self.assertEqual(code, 75)
        self.assertNotIn("exception", transaction)
        self.assertNotIn("watchdog", events)
        self.assertNotIn("terminate-owned", events)
        startup_error = transaction["writes"]["startup-error.json"][0]
        self.assertIs(startup_error["child_created"], False)
        self.assertIn("popen failed", startup_error["error"])

    def test_unrecoverable_startup_ownership_never_signals(self) -> None:
        code, events, transaction = self._run_main(
            metadata_fail_always=True,
            getpgid_error=True,
        )
        self.assertEqual(code, 75)
        self.assertNotIn("exception", transaction)
        self.assertNotIn("verify", events)
        self.assertNotIn("terminate-owned", events)
        ownership_error = transaction["writes"]["ownership-error.json"][0]
        self.assertIs(ownership_error["signal_sent"], False)
        gate_timeout = transaction["writes"]["start-gate-timeout.json"][0]
        self.assertIs(gate_timeout["signal_sent"], False)
        self.assertIn("did not self-terminate", gate_timeout["error"])

    def test_cpu_environment_forces_cuda_hidden_nonframework_profile(self) -> None:
        lease = SimpleNamespace(descriptor=9)
        base_environment = {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": "/ambient",
            "SGLANG_PLUGINS": "pypto",
        }
        with (
            mock.patch.object(
                controller.base,
                "isolated_environment",
                return_value=base_environment,
            ),
            mock.patch.object(
                controller.base,
                "environment_lock_markers",
                return_value={"PYPTO_ENVIRONMENT_LOCK_FD": "9"},
            ),
        ):
            environment = controller.cpu_environment(
                run_id="fixture", run_dir=ROOT / "runs/fixture", lease=lease
            )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertEqual(environment["PYTHONPATH"], "")
        self.assertEqual(environment["SGLANG_PLUGINS"], "__pypto_cpu_v2_no_plugins__")
        self.assertEqual(environment["PYPTO_FRAMEWORK_PROFILE"], "cpu-only-v2")
        self.assertEqual(environment["PYPTO_CPU_COEXISTENCE_V2"], "1")
        self.assertEqual(environment["PYPTO_CPU_COEXISTENCE_POLICY_VERSION"], "2")
        self.assertEqual(
            environment["PYPTO_CPU_LAUNCH_MEMORY_FLOOR_KIB"],
            str(22 * 1024 * 1024),
        )
        self.assertEqual(
            environment["PYPTO_CPU_RESUME_MEMORY_FLOOR_KIB"],
            str(22 * 1024 * 1024),
        )
        self.assertEqual(
            environment["PYPTO_CPU_PAUSE_MEMORY_FLOOR_KIB"],
            str(16 * 1024 * 1024),
        )

    def test_start_gate_publication_is_exact_and_no_replace(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            root = pathlib.Path(directory).resolve()
            gate = root / "start-gate.json"
            metadata_path = root / "process.json"
            metadata = {
                "metadata_complete": True,
                "run_id": "pypto-cpu-v2-20990101T000000Z-123-abcdef",
                "pid": 123,
                "pgid": 123,
                "start_ticks": 456,
                "requested_command_sha256": controller.command_sha256(
                    ["/usr/bin/true"]
                ),
                "start_gate": {
                    "schema_version": contract.START_GATE_SCHEMA_VERSION,
                    "path": str(gate),
                    "timeout_seconds": contract.START_GATE_TIMEOUT_SECONDS,
                    "released": False,
                    "sha256": None,
                    "released_at": None,
                },
            }
            document = controller.release_start_gate(
                path=gate,
                metadata_path=metadata_path,
                metadata=metadata,
            )
            self.assertEqual(stat.S_IMODE(gate.stat().st_mode), 0o600)
            self.assertEqual(json.loads(gate.read_text()), document)
            self.assertIs(metadata["start_gate"]["released"], True)
            self.assertEqual(
                metadata["start_gate"]["sha256"],
                hashlib.sha256(gate.read_bytes()).hexdigest(),
            )
            with self.assertRaises(controller.ControllerError):
                controller.release_start_gate(
                    path=gate,
                    metadata_path=metadata_path,
                    metadata=metadata,
                )

    def test_fixed_start_gate_executes_only_after_exact_release(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            root = pathlib.Path(directory).resolve()
            run_id = "pypto-cpu-v2-20990101T000000Z-123-abcdef"
            gate = root / "start-gate.json"
            metadata_path = root / "process.json"
            command = ["/usr/bin/true"]
            launch = controller.gated_launch_command(
                command, run_id=run_id, gate_path=gate
            )
            environment = {
                "PATH": "/usr/bin:/bin",
                "PYPTO_RUN_ID": run_id,
                "PYPTO_WORKSPACE_ROOT": str(ROOT),
            }
            process = subprocess.Popen(
                launch,
                cwd=ROOT,
                env=environment,
                start_new_session=True,
            )
            metadata = {
                "metadata_complete": True,
                "run_id": run_id,
                "pid": process.pid,
                "pgid": os.getpgid(process.pid),
                "start_ticks": controller.stop_run.process_start_ticks(process.pid),
                "requested_command_sha256": controller.command_sha256(command),
                "start_gate": {
                    "schema_version": contract.START_GATE_SCHEMA_VERSION,
                    "path": str(gate),
                    "timeout_seconds": contract.START_GATE_TIMEOUT_SECONDS,
                    "released": False,
                    "sha256": None,
                    "released_at": None,
                },
            }
            try:
                controller.base.atomic_json(metadata_path, metadata)
                controller.release_start_gate(
                    path=gate,
                    metadata_path=metadata_path,
                    metadata=metadata,
                )
                self.assertEqual(process.wait(timeout=10), 0)
            finally:
                if process.poll() is None:
                    process.wait(timeout=contract.START_GATE_TIMEOUT_SECONDS + 5)

    def test_start_gate_rejects_malformed_digest_mode_and_self_timeout(self) -> None:
        cases = (
            "malformed",
            "digest",
            "mode",
            "timeout",
            "pid",
            "pgid",
            "ticks",
            "run_id",
            "kind",
            "extra",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                dir=ROOT / ".cache"
            ) as directory:
                root = pathlib.Path(directory).resolve()
                run_id = "pypto-cpu-v2-20990101T000000Z-123-abcdef"
                gate = root / "start-gate.json"
                command = ["/usr/bin/true"]
                launch = controller.gated_launch_command(
                    command,
                    run_id=run_id,
                    gate_path=gate,
                    timeout_seconds=0.2 if case == "timeout" else 2.0,
                )
                process = subprocess.Popen(
                    launch,
                    cwd=ROOT,
                    env={"PATH": "/usr/bin:/bin"},
                    start_new_session=True,
                )
                try:
                    if case != "timeout":
                        document = {
                            "schema_version": contract.START_GATE_SCHEMA_VERSION,
                            "kind": contract.POLICY_KIND,
                            "run_id": run_id,
                            "pid": process.pid,
                            "pgid": os.getpgid(process.pid),
                            "start_ticks": controller.stop_run.process_start_ticks(
                                process.pid
                            ),
                            "command_sha256": controller.command_sha256(command),
                        }
                        if case == "digest":
                            document["command_sha256"] = "0" * 64
                        elif case == "pid":
                            document["pid"] = process.pid + 1
                        elif case == "pgid":
                            document["pgid"] = os.getpgid(process.pid) + 1
                        elif case == "ticks":
                            document["start_ticks"] += 1
                        elif case == "run_id":
                            document["run_id"] = "pypto-cpu-v2-other-run"
                        elif case == "kind":
                            document["kind"] = "other-policy"
                        elif case == "extra":
                            document["unexpected"] = 1
                        mode = 0o644 if case == "mode" else 0o600
                        payload = (
                            b"not-json\n"
                            if case == "malformed"
                            else (
                                json.dumps(
                                    document, ensure_ascii=True, indent=2, sort_keys=True
                                )
                                + "\n"
                            ).encode("ascii")
                        )
                        descriptor = os.open(
                            gate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode
                        )
                        with os.fdopen(descriptor, "wb") as handle:
                            handle.write(payload)
                    self.assertEqual(process.wait(timeout=5), 75)
                finally:
                    if process.poll() is None:
                        process.wait(timeout=5)


class WatchdogTest(unittest.TestCase):
    class Process:
        def __init__(self, timeouts: int):
            self.timeouts = timeouts
            self.returncode: int | None = None

        def wait(self, timeout=None):
            if self.timeouts:
                self.timeouts -= 1
                raise subprocess.TimeoutExpired(["fixture"], timeout)
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    @staticmethod
    def metadata() -> dict[str, object]:
        return {
            "run_id": "fixture",
            "workspace": str(ROOT),
            "pid": 100,
            "pgid": 100,
            "start_ticks": 1,
            "status": "running",
            "coexistence_pauses": [],
        }

    @staticmethod
    def safe_audit() -> dict[str, object]:
        return {
            "nvidia_compute_pids": [],
            "owned_nvidia_compute_pids": [],
            "external_nvidia_compute_pids": [],
            "protected_nvidia_compute_pids": [],
            "workspace_nvidia_compute_pids": [],
            "protected_nvidia_runtime_mapping_pids": [],
            "unreadable_protected_maps": [],
        }

    def test_pause_at_16_resume_at_22_signals_only_verified_metadata(self) -> None:
        process = self.Process(2)
        metadata = self.metadata()
        signals: list[tuple[dict[str, object], signal.Signals]] = []

        def signal_side_effect(value, requested):
            signals.append((value, requested))
            return 100, 100

        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                side_effect=[
                    contract.PAUSE_MEMORY_FLOOR_KIB - 1,
                    contract.RESUME_MEMORY_FLOOR_KIB,
                ],
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(controller, "audit_runtime_state", return_value=self.safe_audit()),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (0, False))
        self.assertEqual([item[1] for item in signals], [signal.SIGSTOP, signal.SIGCONT])
        self.assertTrue(all(item[0] is metadata for item in signals))
        self.assertEqual(metadata["coexistence_pauses"][0]["floor_kib"], 16 * 1024 * 1024)

    def test_running_between_16_and_22_continues_without_signal(self) -> None:
        process = self.Process(1)
        metadata = self.metadata()
        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.RESUME_MEMORY_FLOOR_KIB - 1,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(controller, "audit_runtime_state", return_value=self.safe_audit()),
            mock.patch.object(controller.stop_run, "signal_verified") as signal_verified,
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (0, False))
        signal_verified.assert_not_called()

    def test_exactly_16_gib_does_not_pause(self) -> None:
        process = self.Process(1)
        metadata = self.metadata()
        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.PAUSE_MEMORY_FLOOR_KIB,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller, "audit_runtime_state", return_value=self.safe_audit()
            ),
            mock.patch.object(controller.stop_run, "signal_verified") as signal_verified,
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (0, False))
        signal_verified.assert_not_called()

    def test_paused_child_does_not_resume_at_22_minus_one(self) -> None:
        process = self.Process(3)
        metadata = self.metadata()
        signals: list[signal.Signals] = []

        def signal_side_effect(_metadata, requested):
            signals.append(requested)
            return 100, 100

        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                side_effect=[
                    contract.PAUSE_MEMORY_FLOOR_KIB - 1,
                    contract.RESUME_MEMORY_FLOOR_KIB - 1,
                    contract.RESUME_MEMORY_FLOOR_KIB,
                ],
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller, "audit_runtime_state", return_value=self.safe_audit()
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (0, False))
        self.assertEqual(signals, [signal.SIGSTOP, signal.SIGCONT])

    def test_audit_failure_pauses_and_requires_clean_recovery(self) -> None:
        process = self.Process(2)
        metadata = self.metadata()
        signals: list[signal.Signals] = []

        def signal_side_effect(_metadata, requested):
            signals.append(requested)
            return 100, 100

        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.RESUME_MEMORY_FLOOR_KIB,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller,
                "audit_runtime_state",
                side_effect=[RuntimeError("audit unavailable"), self.safe_audit()],
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (0, False))
        self.assertEqual(signals, [signal.SIGSTOP, signal.SIGCONT])
        self.assertEqual(
            metadata["coexistence_pauses"][0]["reason"],
            "resource-observation-failed",
        )

    def test_disk_floor_pauses_then_resumes(self) -> None:
        process = self.Process(2)
        metadata = self.metadata()
        signals: list[signal.Signals] = []

        def signal_side_effect(_metadata, requested):
            signals.append(requested)
            return 100, 100

        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.RESUME_MEMORY_FLOOR_KIB,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                side_effect=[SimpleNamespace(free=0), SimpleNamespace(free=2)],
            ),
            mock.patch.object(
                controller, "audit_runtime_state", return_value=self.safe_audit()
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (0, False))
        self.assertEqual(signals, [signal.SIGSTOP, signal.SIGCONT])
        self.assertEqual(
            metadata["coexistence_pauses"][0]["reason"], "workspace-disk-floor"
        )

    def test_timeout_is_terminal_for_verified_owned_group(self) -> None:
        process = self.Process(1)
        metadata = self.metadata()
        with (
            mock.patch.object(controller.time, "monotonic", side_effect=[0.0, 61.0]),
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.RESUME_MEMORY_FLOOR_KIB,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller, "audit_runtime_state", return_value=self.safe_audit()
            ),
            mock.patch.object(controller, "terminate_owned", return_value=75) as terminate,
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (75, True))
        terminate.assert_called_once_with(
            process, metadata, ROOT / "runs/fixture.json"
        )

    def test_pause_ownership_ambiguity_returns_75_without_fallback_signal(self) -> None:
        process = self.Process(1)
        metadata = self.metadata()
        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.PAUSE_MEMORY_FLOOR_KIB - 1,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller, "audit_runtime_state", return_value=self.safe_audit()
            ),
            mock.patch.object(
                controller.stop_run,
                "signal_verified",
                side_effect=RuntimeError("ownership ambiguous"),
            ) as signal_verified,
            mock.patch.object(controller, "terminate_owned") as terminate,
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (75, True))
        signal_verified.assert_called_once_with(metadata, signal.SIGSTOP)
        terminate.assert_not_called()
        self.assertEqual(metadata["status"], "group-ownership-ambiguous")

    def test_resume_ownership_ambiguity_returns_75_without_termination(self) -> None:
        process = self.Process(2)
        metadata = self.metadata()
        calls = 0

        def signal_side_effect(_metadata, requested):
            nonlocal calls
            calls += 1
            if requested == signal.SIGCONT:
                raise RuntimeError("resume ownership ambiguous")
            return 100, 100

        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                side_effect=[
                    contract.PAUSE_MEMORY_FLOOR_KIB - 1,
                    contract.RESUME_MEMORY_FLOOR_KIB,
                ],
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller, "audit_runtime_state", return_value=self.safe_audit()
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(controller, "terminate_owned") as terminate,
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (75, True))
        self.assertEqual(calls, 2)
        terminate.assert_not_called()
        self.assertEqual(metadata["status"], "group-ownership-ambiguous")

    def test_pause_and_resume_process_lookup_races_are_reaped_without_signal(self) -> None:
        pause_process = self.Process(1)
        pause_metadata = self.metadata()
        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.PAUSE_MEMORY_FLOOR_KIB - 1,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller, "audit_runtime_state", return_value=self.safe_audit()
            ),
            mock.patch.object(
                controller.stop_run,
                "signal_verified",
                side_effect=ProcessLookupError,
            ),
        ):
            self.assertEqual(
                controller.wait_with_watchdog(
                    pause_process,
                    pause_metadata,
                    timeout_seconds=60,
                    minimum_free_disk_bytes=1,
                    metadata_path=ROOT / "runs/fixture.json",
                ),
                (0, False),
            )

        resume_process = self.Process(2)
        resume_metadata = self.metadata()
        calls = 0

        def resume_signal(_metadata, requested):
            nonlocal calls
            calls += 1
            if requested == signal.SIGCONT:
                raise ProcessLookupError
            return 100, 100

        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                side_effect=[
                    contract.PAUSE_MEMORY_FLOOR_KIB - 1,
                    contract.RESUME_MEMORY_FLOOR_KIB,
                ],
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller, "audit_runtime_state", return_value=self.safe_audit()
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=resume_signal
            ),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            self.assertEqual(
                controller.wait_with_watchdog(
                    resume_process,
                    resume_metadata,
                    timeout_seconds=60,
                    minimum_free_disk_bytes=1,
                    metadata_path=ROOT / "runs/fixture.json",
                ),
                (0, False),
            )
        self.assertEqual(calls, 2)

    def test_protected_activation_pauses_owned_group_not_protected_pid(self) -> None:
        process = self.Process(2)
        metadata = self.metadata()
        protected = self.safe_audit()
        protected["protected_nvidia_compute_pids"] = [777]
        calls: list[tuple[object, object]] = []

        def signal_side_effect(value, requested):
            calls.append((value, requested))
            return 100, 100

        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                side_effect=[
                    contract.RESUME_MEMORY_FLOOR_KIB,
                    contract.RESUME_MEMORY_FLOOR_KIB,
                ],
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(
                controller,
                "audit_runtime_state",
                side_effect=[protected, self.safe_audit()],
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (0, False))
        self.assertEqual([call[1] for call in calls], [signal.SIGSTOP, signal.SIGCONT])
        self.assertTrue(all(call[0] is metadata for call in calls))
        self.assertNotIn(777, [item for call in calls for item in call if isinstance(item, int)])

    def test_owned_nvidia_compute_is_terminal_and_owned_only(self) -> None:
        process = self.Process(1)
        metadata = self.metadata()
        owned = self.safe_audit()
        owned["owned_nvidia_compute_pids"] = [100]
        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.PAUSE_MEMORY_FLOOR_KIB - 1,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(controller, "audit_runtime_state", return_value=owned),
            mock.patch.object(controller, "terminate_owned", return_value=75) as terminate,
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (75, True))
        terminate.assert_called_once_with(
            process, metadata, ROOT / "runs/fixture.json"
        )

    def test_external_compute_is_observed_but_never_signalled(self) -> None:
        process = self.Process(1)
        metadata = self.metadata()
        external = self.safe_audit()
        external["external_nvidia_compute_pids"] = [888]
        with (
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                return_value=contract.RESUME_MEMORY_FLOOR_KIB,
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=1 << 50),
            ),
            mock.patch.object(controller, "audit_runtime_state", return_value=external),
            mock.patch.object(controller.stop_run, "signal_verified") as signal_verified,
            mock.patch.object(controller, "terminate_owned") as terminate_owned,
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (0, False))
        signal_verified.assert_not_called()
        terminate_owned.assert_not_called()

    def test_protected_runtime_unreadable_and_workspace_compute_are_pause_reasons(
        self,
    ) -> None:
        for field, expected_reason in (
            (
                "protected_nvidia_runtime_mapping_pids",
                "protected-nvidia-runtime-became-active",
            ),
            ("unreadable_protected_maps", "protected-maps-became-unreadable"),
            (
                "workspace_nvidia_compute_pids",
                "unattributed-workspace-nvidia-compute-became-active",
            ),
        ):
            audit = self.safe_audit()
            audit[field] = [77]
            audit["nvidia_compute_pids"] = [77] if "compute" in field else []
            reason = controller.pressure_reason(
                available_kib=contract.RESUME_MEMORY_FLOOR_KIB,
                disk_free_bytes=1 << 50,
                minimum_free_disk_bytes=1,
                audit=audit,
                observation_errors={},
            )
            self.assertEqual(reason["reason"], expected_reason)

    def test_timeout_wins_over_simultaneous_observation_failures(self) -> None:
        process = self.Process(1)
        metadata = self.metadata()
        owned = self.safe_audit()
        owned["nvidia_compute_pids"] = [100]
        owned["owned_nvidia_compute_pids"] = [100]
        with (
            mock.patch.object(controller.time, "monotonic", side_effect=[0.0, 61.0]),
            mock.patch.object(
                controller.preflight,
                "mem_available_kib",
                side_effect=RuntimeError("memory unavailable"),
            ),
            mock.patch.object(
                controller.shutil,
                "disk_usage",
                side_effect=RuntimeError("disk unavailable"),
            ),
            mock.patch.object(controller, "audit_runtime_state", return_value=owned),
            mock.patch.object(controller, "terminate_owned", return_value=75) as terminate,
            mock.patch.object(controller.base, "atomic_json"),
        ):
            result = controller.wait_with_watchdog(
                process,
                metadata,
                timeout_seconds=60,
                minimum_free_disk_bytes=1,
                metadata_path=ROOT / "runs/fixture.json",
            )
        self.assertEqual(result, (75, True))
        self.assertEqual(metadata["coexistence_abort"]["reason"], "owned-run-timeout")
        terminate.assert_called_once_with(
            process, metadata, ROOT / "runs/fixture.json"
        )

    def test_termination_verification_ambiguity_is_recorded_without_fallback(self) -> None:
        process = self.Process(0)
        process.returncode = 0
        metadata = self.metadata()
        with (
            mock.patch.object(
                controller,
                "exact_owned_group_members",
                side_effect=RuntimeError("exact group ambiguous"),
            ),
            mock.patch.object(
                controller.base, "terminate_owned_process"
            ) as base_terminate,
            mock.patch.object(controller.base, "atomic_json"),
        ):
            code = controller.terminate_owned(
                process, metadata, ROOT / "runs/fixture.json"
            )
        self.assertEqual(code, 75)
        base_terminate.assert_not_called()
        self.assertEqual(metadata["status"], "group-ownership-ambiguous")


class TerminateOwnedTest(unittest.TestCase):
    class Process:
        def __init__(self, returncode: int | None = None):
            self.pid = 1234
            self.returncode = returncode

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout=None) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired(["fixture"], timeout)
            return self.returncode

    @staticmethod
    def metadata() -> dict[str, object]:
        return {
            "run_id": "fixture",
            "workspace": str(ROOT),
            "pid": 100,
            "pgid": 100,
            "start_ticks": 1,
            "status": "running",
        }

    def test_exact_owned_divergence_is_ambiguous_and_never_signals(self) -> None:
        process = self.Process()
        metadata = self.metadata()
        with (
            mock.patch.object(
                controller.stop_run,
                "exact_process_group_members",
                return_value=[100],
            ),
            mock.patch.object(
                controller.stop_run, "owned_group_members", return_value=[100, 101]
            ) as owned_members,
            mock.patch.object(
                controller.stop_run, "signal_verified"
            ) as signal_verified,
            mock.patch.object(controller.base, "atomic_json"),
        ):
            code = controller.terminate_owned(process, metadata, ROOT / "runs/f.json")
        self.assertEqual(code, 75)
        signal_verified.assert_not_called()
        owned_members.assert_called_once()
        self.assertEqual(metadata["status"], "group-ownership-ambiguous")

    def test_term_cont_surviving_members_end_paused_never_killed(self) -> None:
        process = self.Process()
        metadata = self.metadata()
        signals: list[tuple[str, object]] = []

        def signal_side_effect(_metadata, requested):
            signals.append(("group", requested))
            return 100, 100

        def followup_side_effect(_metadata, requested, _pgid):
            signals.append(("followup", requested))
            return True

        with (
            mock.patch.object(
                controller.stop_run,
                "exact_process_group_members",
                return_value=[100],
            ),
            mock.patch.object(
                controller.stop_run, "owned_group_members", return_value=[100]
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(
                controller.stop_run,
                "signal_verified_followup",
                side_effect=followup_side_effect,
            ),
            mock.patch.object(
                controller.time,
                "monotonic",
                side_effect=[0.0, 1.0, 2.0, 100.0, 100.0],
            ),
            mock.patch.object(controller.time, "sleep"),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            code = controller.terminate_owned(process, metadata, ROOT / "runs/f.json")
        self.assertEqual(code, 75)
        self.assertEqual(
            signals,
            [
                ("group", signal.SIGTERM),
                ("followup", signal.SIGCONT),
                ("followup", signal.SIGSTOP),
            ],
        )
        self.assertEqual(metadata["status"], "paused")
        self.assertEqual(metadata["termination_surviving_group_pids"], [100])

    def test_group_emptied_after_term_returns_child_code(self) -> None:
        process = self.Process()
        metadata = self.metadata()
        signals: list[object] = []

        def signal_side_effect(_metadata, requested):
            signals.append(requested)
            process.returncode = 9
            return 100, 100

        def followup_side_effect(_metadata, requested, _pgid):
            signals.append(requested)
            return False

        with (
            mock.patch.object(
                controller.stop_run,
                "exact_process_group_members",
                return_value=[100],
            ),
            mock.patch.object(
                controller.stop_run, "owned_group_members", return_value=[100]
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(
                controller.stop_run,
                "signal_verified_followup",
                side_effect=followup_side_effect,
            ),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            code = controller.terminate_owned(process, metadata, ROOT / "runs/f.json")
        self.assertEqual(code, 9)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGCONT])

    def test_survivor_stop_revalidation_failure_is_ambiguous_without_signal(self):
        process = self.Process()
        metadata = self.metadata()
        signals: list[object] = []

        def signal_side_effect(_metadata, requested):
            signals.append(requested)
            return 100, 100

        def followup_side_effect(_metadata, requested, _pgid):
            signals.append(requested)
            if requested == signal.SIGSTOP:
                raise controller.stop_run.GroupRevalidationError(
                    "group changed before stop"
                )
            return True

        with (
            mock.patch.object(
                controller.stop_run,
                "exact_process_group_members",
                return_value=[100],
            ),
            mock.patch.object(
                controller.stop_run, "owned_group_members", return_value=[100]
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified", side_effect=signal_side_effect
            ),
            mock.patch.object(
                controller.stop_run,
                "signal_verified_followup",
                side_effect=followup_side_effect,
            ),
            mock.patch.object(
                controller.time,
                "monotonic",
                side_effect=[0.0, 1.0, 2.0, 100.0, 100.0],
            ),
            mock.patch.object(controller.time, "sleep"),
            mock.patch.object(controller.base, "atomic_json"),
        ):
            code = controller.terminate_owned(process, metadata, ROOT / "runs/f.json")
        self.assertEqual(code, 75)
        self.assertEqual(
            signals, [signal.SIGTERM, signal.SIGCONT, signal.SIGSTOP]
        )
        self.assertEqual(metadata["status"], "group-ownership-ambiguous")

    def test_already_reaped_empty_group_shortcut_returns_child_code(self) -> None:
        process = self.Process(returncode=0)
        metadata = self.metadata()
        with (
            mock.patch.object(
                controller.stop_run,
                "exact_process_group_members",
                return_value=[],
            ),
            mock.patch.object(
                controller.stop_run, "signal_verified"
            ) as signal_verified,
            mock.patch.object(controller.base, "atomic_json"),
        ):
            code = controller.terminate_owned(process, metadata, ROOT / "runs/f.json")
        self.assertEqual(code, 0)
        signal_verified.assert_not_called()


class StructureTest(unittest.TestCase):
    def test_sources_are_additive_and_do_not_rewrite_base_globals(self) -> None:
        source = "\n".join(
            (ROOT / path).read_text()
            for path in (
                contract.PREFLIGHT_RELATIVE_PATH,
                contract.CONTROLLER_RELATIVE_PATH,
                contract.CONTROL_VALIDATOR_RELATIVE_PATH,
            )
        )
        for forbidden in (
            "base.COEXISTENCE_RESUME_MEMORY_KIB =",
            "base.COEXISTENCE_ABORT_MEMORY_KIB =",
            "base.preflight_tool =",
            "subprocess.Popen =",
            "os.kill(",
            "os.killpg(",
            'sys.modules["preflight"] =',
            'sys.modules["stop_run"] =',
            'sys.modules["_pypto_nvidia',
            "shell=True",
        ):
            self.assertNotIn(forbidden, source)
        manifest_path = ROOT / contract.MANIFEST_RELATIVE_PATH
        if manifest_path.exists():
            identity = control.validate_control_manifest(ROOT)
            self.assertEqual(
                identity["manifest_path"],
                contract.MANIFEST_RELATIVE_PATH.as_posix(),
            )
            self.assertEqual(identity["root_clean"], True)
        else:
            self.assertFalse(manifest_path.exists())

    def test_exact_base_dependency_manifest_and_accepted_evidence_are_preserved(
        self,
    ) -> None:
        expected = {
            "tools/preflight.py": "0b9884f8dbd34337a85f62c351b1e19dda3a8b84ec9a88c835d8701af053e3d1",
            "tools/run_isolated.py": "978686ac09743a98233c9616d23b04e57d3a257bd643d5db3b8a71eaac7465c8",
            "tools/stop_run.py": "879a2e3863671531a548c71d788d56298500eab989bd1420d2c7ae01717ddfe4",
            "tools/_pypto_nvidia_executable_sm120_contract.py": "fa477d91933df765e9163bf3081ed6d41f323bb49285106dcdbd4bee554113bf",
            "tools/_pypto_nvidia_sm120_control_manifest.py": "bfa0e5c66ffad9435c0c31dc82ed7581d6bee608f1487e3fb2932cabbb2b597a",
            "state/contracts/pypto_nvidia_executable_sm120_v4.json": "a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf",
            "state/contracts/pypto_fused_pointwise_sm120_v2.json": "d3b16079c811dd2fbe610ba264d81117e8c4a44886b74caaddb684df2d467036",
            "state/contracts/pypto_row_reduction_sm120_v1.json": "7ae64b2273e4906f05e26a070460b713e3d3e5de74194329663dd76dd68ccc31",
            "state/evidence/EV-0060.json": "984f59acbdfc95ca96f57d97f38986bb581c0fbb59214a29976d519cf0215622",
            "state/evidence/EV-0061.json": "1cc93511c73e116a1be341a4dacab47a7dbae9270eb7875e99063f5c44b29bab",
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest
            )
        self.assertEqual(
            control.validate_base_dependencies(ROOT),
            contract.policy_document()["base_dependencies"],
        )

    def test_runtime_audit_schema_is_exact_and_owned_external_disjoint(self) -> None:
        value = WatchdogTest.safe_audit()
        controller.validate_runtime_audit(value)
        value = WatchdogTest.safe_audit()
        value.update(
            {
                "nvidia_compute_pids": [1, 2, 3, 4],
                "owned_nvidia_compute_pids": [1],
                "protected_nvidia_compute_pids": [2],
                "workspace_nvidia_compute_pids": [3],
                "external_nvidia_compute_pids": [4],
            }
        )
        controller.validate_runtime_audit(value)
        candidate = copy.deepcopy(value)
        candidate["unexpected"] = []
        with self.assertRaises(controller.ControllerError):
            controller.validate_runtime_audit(candidate)

    def test_git_environment_is_hard_pinned(self) -> None:
        self.assertEqual(control.GIT_ENVIRONMENT, {"PATH": "/usr/bin:/bin"})
        self.assertTrue(pathlib.Path("/usr/bin/git").is_file())

    def test_main_wrapper_converts_unexpected_refusal_to_75(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            run_id_file = pathlib.Path(directory) / "run-id.json"
            completed = subprocess.run(
                [
                    str(ROOT / "envs/pypto-nvidia/bin/python"),
                    "-E",
                    "-B",
                    "-S",
                    str(ROOT / contract.CONTROLLER_RELATIVE_PATH),
                    "--run-id-file",
                    str(run_id_file),
                    "--",
                    "",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 75)
            self.assertIn("Traceback (most recent call last):", completed.stderr)
            self.assertIn("ContractError", completed.stderr)
            self.assertFalse(run_id_file.exists())

    def test_runtime_audit_builds_disjoint_complete_pid_partition(self) -> None:
        def process(pid: int, cwd: str) -> object:
            return preflight.ProcessInfo(
                pid=pid,
                ppid=1,
                start_ticks=pid * 10,
                rss_kib=1,
                command="fixture",
                cwd=cwd,
            )

        protected = process(2, "/home/zhaosiying/amdgpu-sim")
        workspace = process(3, str(ROOT))
        all_processes = [process(1, str(ROOT)), protected, workspace, process(4, "/tmp")]
        with (
            mock.patch.object(
                controller.preflight,
                "nvidia_compute_pids",
                return_value={1, 2, 3, 4},
            ),
            mock.patch.object(
                controller.preflight,
                "process_table",
                return_value=(all_processes, [protected], [workspace]),
            ),
            mock.patch.object(
                controller.preflight,
                "protected_nvidia_runtime_mappings",
                return_value=([], []),
            ),
            mock.patch.object(
                controller.base,
                "partition_compute_pids",
                return_value=([1], [2, 3, 4]),
            ),
        ):
            value = controller.audit_runtime_state(WatchdogTest.metadata())
        self.assertEqual(value["owned_nvidia_compute_pids"], [1])
        self.assertEqual(value["protected_nvidia_compute_pids"], [2])
        self.assertEqual(value["workspace_nvidia_compute_pids"], [3])
        self.assertEqual(value["external_nvidia_compute_pids"], [4])
        candidate = copy.deepcopy(value)
        candidate["owned_nvidia_compute_pids"] = [9]
        candidate["external_nvidia_compute_pids"] = [9]
        with self.assertRaises(controller.ControllerError):
            controller.validate_runtime_audit(candidate)


class ControlManifestNegativeTest(unittest.TestCase):
    GIT_ENV = {"PATH": "/usr/bin:/bin"}

    def _git(self, root: pathlib.Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=self.GIT_ENV,
        ).stdout.strip()

    def _build_fake_workspace(
        self, root: pathlib.Path
    ) -> tuple[str, str]:
        for relative in (
            *contract.CONTROL_PATHS,
            contract.BASE_PREFLIGHT_RELATIVE_PATH.as_posix(),
            contract.BASE_ISOLATION_RELATIVE_PATH.as_posix(),
            contract.BASE_STOP_RELATIVE_PATH.as_posix(),
            contract.BASE_NVIDIA_CONTRACT_RELATIVE_PATH.as_posix(),
            contract.BASE_NVIDIA_CONTROL_RELATIVE_PATH.as_posix(),
            contract.BASE_NVIDIA_MANIFEST_RELATIVE_PATH.as_posix(),
        ):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        self._git(root, "init", "-q")
        self._git(root, "add", "-A")
        self._git(
            root,
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=fixture",
            "commit",
            "-q",
            "--no-gpg-sign",
            "-m",
            "fixture",
        )
        return self._git(root, "rev-parse", "HEAD"), self._git(
            root, "rev-parse", "HEAD^{tree}"
        )

    def _commit_all(self, root: pathlib.Path, message: str) -> None:
        self._git(root, "add", "-A")
        self._git(
            root,
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=fixture",
            "commit",
            "-q",
            "--no-gpg-sign",
            "-m",
            message,
        )

    def _manifest_document(
        self, root: pathlib.Path, commit: str, tree: str
    ) -> dict[str, object]:
        files = []
        for relative in contract.CONTROL_PATHS:
            path = root / relative
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "mode": stat.S_IMODE(path.stat().st_mode),
                }
            )
        return {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": contract.POLICY_KIND,
            "implementation_commit": commit,
            "implementation_tree": tree,
            "base_dependencies": control.validate_base_dependencies(root),
            "files": files,
        }

    def _validate(self, root: pathlib.Path) -> dict[str, object]:
        fake_nvidia_control = SimpleNamespace(
            validate_control_manifest=lambda _root: {
                "manifest_sha256": contract.BASE_NVIDIA_MANIFEST_SHA256
            }
        )
        with mock.patch.object(control, "base_nvidia_control", fake_nvidia_control):
            return control.validate_control_manifest(root)

    def _workspace_with_manifest(self) -> pathlib.Path:
        directory = tempfile.mkdtemp(dir=ROOT / ".cache")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        root = pathlib.Path(directory).resolve()
        commit, tree = self._build_fake_workspace(root)
        document = self._manifest_document(root, commit, tree)
        manifest_path = root / contract.MANIFEST_RELATIVE_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(control.canonical_json(document))
        self._commit_all(root, "manifest")
        return root

    def test_canonical_manifest_validates_in_fake_workspace(self) -> None:
        root = self._workspace_with_manifest()
        identity = self._validate(root)
        self.assertEqual(identity["root_clean"], True)
        self.assertEqual(identity["manifest_path"], contract.MANIFEST_RELATIVE_PATH.as_posix())
        self.assertEqual(len(identity["files"]), len(contract.CONTROL_PATHS))

    def test_missing_manifest_refused(self) -> None:
        root = self._workspace_with_manifest()
        (root / contract.MANIFEST_RELATIVE_PATH).unlink()
        with (
            mock.patch.object(control, "base_nvidia_control", SimpleNamespace()),
            self.assertRaises(control.ControlManifestError) as raised,
        ):
            self._validate(root)
        self.assertIn("missing", str(raised.exception))

    def test_noncanonical_bytes_refused(self) -> None:
        root = self._workspace_with_manifest()
        document = self._manifest_document(
            root,
            self._git(root, "rev-parse", "HEAD~1"),
            self._git(root, "rev-parse", "HEAD~1^{tree}"),
        )
        (root / contract.MANIFEST_RELATIVE_PATH).write_bytes(
            (json.dumps(document, ensure_ascii=True, indent=2) + "\n").encode()
        )
        with (
            mock.patch.object(control, "base_nvidia_control", SimpleNamespace()),
            self.assertRaises(control.ControlManifestError) as raised,
        ):
            self._validate(root)
        self.assertIn("not canonical JSON", str(raised.exception))

    def test_duplicate_keys_refused(self) -> None:
        root = self._workspace_with_manifest()
        manifest_path = root / contract.MANIFEST_RELATIVE_PATH
        duplicated = manifest_path.read_text()
        duplicated = duplicated.replace(
            "{", '{"implementation_commit": "duplicate", ', 1
        )
        manifest_path.write_text(duplicated)
        with (
            mock.patch.object(control, "base_nvidia_control", SimpleNamespace()),
            self.assertRaises(control.ControlManifestError) as raised,
        ):
            self._validate(root)
        self.assertIn("duplicate JSON key", str(raised.exception))

    def test_wrong_key_set_refused(self) -> None:
        root = self._workspace_with_manifest()
        document = self._manifest_document(
            root,
            self._git(root, "rev-parse", "HEAD~1"),
            self._git(root, "rev-parse", "HEAD~1^{tree}"),
        )
        document.pop("files")
        (root / contract.MANIFEST_RELATIVE_PATH).write_bytes(
            control.canonical_json(document)
        )
        with (
            mock.patch.object(control, "base_nvidia_control", SimpleNamespace()),
            self.assertRaises(control.ControlManifestError) as raised,
        ):
            self._validate(root)
        self.assertIn("key set differs", str(raised.exception))

    def test_dirty_root_refused(self) -> None:
        root = self._workspace_with_manifest()
        (root / "extra.txt").write_text("dirty")
        with (
            mock.patch.object(control, "base_nvidia_control", SimpleNamespace()),
            self.assertRaises(control.ControlManifestError) as raised,
        ):
            self._validate(root)
        self.assertIn("not clean", str(raised.exception))

    def test_controls_changed_after_implementation_refused(self) -> None:
        root = self._workspace_with_manifest()
        target = root / contract.CONTROL_PATHS[0]
        target.write_bytes(target.read_bytes() + b"\n# drift\n")
        self._git(root, "add", "-A")
        self._git(
            root,
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=fixture",
            "commit",
            "-q",
            "--no-gpg-sign",
            "-m",
            "drift",
        )
        with (
            mock.patch.object(control, "base_nvidia_control", SimpleNamespace()),
            self.assertRaises(control.ControlManifestError) as raised,
        ):
            self._validate(root)
        self.assertIn("changed after implementation", str(raised.exception))

    def test_bytecode_cache_refused(self) -> None:
        root = self._workspace_with_manifest()
        cache = root / "tools/__pycache__"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "_pypto_cpu_coexistence_v2_contract.cpython-314.pyc").write_bytes(b"x")
        with (
            mock.patch.object(control, "base_nvidia_control", SimpleNamespace()),
            self.assertRaises(control.ControlManifestError) as raised,
        ):
            self._validate(root)
        self.assertIn("bytecode/cache entries are forbidden", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
