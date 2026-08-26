from __future__ import annotations

import copy
import hashlib
import os
import pathlib
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
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


contract = load_source(
    "test_fused_v2_contract", "tools/_pypto_fused_pointwise_sm120_contract_v2.py"
)
preflight = load_source("test_fused_v2_preflight", "tools/preflight_gpu_smoke_v2.py")
control = load_source(
    "test_fused_v2_control",
    "tools/_pypto_fused_pointwise_sm120_control_manifest_v2.py",
)
controller = load_source(
    "test_fused_v2_controller",
    "tools/run_pypto_fused_pointwise_sm120_v2_isolated.py",
)
child = load_source(
    "test_fused_v2_child",
    "benchmarks/operators/pypto_fused_pointwise_sm120_v2.py",
)
finalizer = load_source(
    "test_fused_v2_finalizer",
    "tools/finalize_pypto_fused_pointwise_sm120_v2.py",
)


class FrozenV1AndContractTest(unittest.TestCase):
    def test_v1_bytes_are_unchanged(self) -> None:
        expected = {
            "tools/preflight.py": "0b9884f8dbd34337a85f62c351b1e19dda3a8b84ec9a88c835d8701af053e3d1",
            "tools/run_isolated.py": "978686ac09743a98233c9616d23b04e57d3a257bd643d5db3b8a71eaac7465c8",
            "tools/stop_run.py": "879a2e3863671531a548c71d788d56298500eab989bd1420d2c7ae01717ddfe4",
            "benchmarks/operators/pypto_fused_pointwise_sm120.py": "b7960cc894834b3ba05476943e774cfc8602891faa5b9137b3d97a6aac40ab15",
            "tools/run_pypto_fused_pointwise_sm120_isolated.py": "484ec6e5c773f1cc912ae447b1319e3f3b7610fd9cf5d8e7eafb29a71e2b5e32",
            "tools/finalize_pypto_fused_pointwise_sm120.py": "c1724d138a6385d293ba5e79dcbf3208ebb0bac1f0dd734af738dddda5d26a37",
            "state/contracts/pypto_fused_pointwise_sm120_v1.json": "ce20dd3ac6796bee16235913b8b296ae8c4781167c35f08de7c19ac7977a6896",
            "tools/_pypto_fused_pointwise_sm120_contract.py": "7c812ccd3d9a76f2e5a258cf53fd029df776a67dfaf42c631363332fb9f8811c",
            "tools/generate_pypto_fused_pointwise_anchors.py": "89f06a416622e1d78595c0a086db4dce66bebbf70f3867b2601885767e85c54e",
            "state/contracts/pypto_fused_pointwise_compile_anchors_v1.json": "584f6755bbd248de5bb6ddd3ff610da8082667bc892a6cff6583ea42d4c44c97",
            "tools/_pypto_fused_pointwise_sm120_control_manifest.py": "299356cf6361fd1372e1fb77ddd626c2d4f84609abd565a3ea3be0bbe26c98c9",
            "tests/test_pypto_fused_pointwise_sm120.py": "8ab9ac5ff7312f8583f7a970f7c060ea00f22637348755dce42cee747cd88c3f",
            "docs/pypto_fused_pointwise_sm120_smoke.md": "c35772024d22956668d2637d489fd4af5a3993d664f21272b0d214bfaf4da70d",
        }
        for relative, digest in expected.items():
            with self.subTest(path=relative):
                self.assertEqual(
                    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest
                )

    def test_v2_contract_is_an_admission_only_layer(self) -> None:
        self.assertEqual(contract.SMOKE_SCHEMA_VERSION, 2)
        self.assertEqual(contract.GPU_SMOKE_POLICY_VERSION, 2)
        self.assertEqual(
            contract.PROTECTED_GPU_SMOKE_MEMORY_FLOOR_KIB, 22 * 1024 * 1024
        )
        self.assertEqual(
            contract.EXCLUSIVE_GPU_SMOKE_MEMORY_FLOOR_KIB, 32 * 1024 * 1024
        )
        self.assertEqual(contract.OWNED_RUN_ABORT_MEMORY_FLOOR_KIB, 16 * 1024 * 1024)
        self.assertEqual(contract.GPU_FREE_MEMORY_FLOOR_MIB, 4 * 1024)
        runner = ROOT / contract.RUNNER_RELATIVE_PATH
        self.assertEqual(runner.stat().st_size, contract.RUNNER_SIZE)
        self.assertEqual(
            hashlib.sha256(runner.read_bytes()).hexdigest(), contract.RUNNER_SHA256
        )
        self.assertEqual(contract.CASE_ORDER, contract._BASE.CASE_ORDER)
        self.assertEqual(
            contract.COMPILE_ANCHORS_SHA256, contract._BASE.COMPILE_ANCHORS_SHA256
        )


class AdmissionBoundaryTest(unittest.TestCase):
    def test_exact_protected_boundary(self) -> None:
        floor = 22 * 1024 * 1024
        below = preflight.gpu_smoke_policy_failures(
            protected_authorized=True,
            protected_heavy=[],
            available_kib=floor - 1,
            protected_nvidia_compute_pids=[],
            protected_nvidia_runtime_pids=[],
            unreadable_protected_maps=[],
        )
        admitted = preflight.gpu_smoke_policy_failures(
            protected_authorized=True,
            protected_heavy=[],
            available_kib=floor,
            protected_nvidia_compute_pids=[],
            protected_nvidia_runtime_pids=[],
            unreadable_protected_maps=[],
        )
        self.assertTrue(any("below 22 GiB" in item for item in below))
        self.assertEqual(admitted, [])
        self.assertEqual(floor - 1, 23_068_671)
        self.assertEqual(floor, 23_068_672)

    def test_exclusive_boundary_remains_32_gib(self) -> None:
        floor = 32 * 1024 * 1024
        self.assertTrue(
            preflight.gpu_smoke_policy_failures(
                protected_authorized=False,
                protected_heavy=[],
                available_kib=floor - 1,
                protected_nvidia_compute_pids=[],
                protected_nvidia_runtime_pids=[],
                unreadable_protected_maps=[],
            )
        )
        self.assertEqual(
            preflight.gpu_smoke_policy_failures(
                protected_authorized=False,
                protected_heavy=[],
                available_kib=floor,
                protected_nvidia_compute_pids=[],
                protected_nvidia_runtime_pids=[],
                unreadable_protected_maps=[],
            ),
            [],
        )

    def _report(self, *, available: int, requested: bool = True) -> dict[str, object]:
        floor = 22 * 1024 * 1024 if requested else 32 * 1024 * 1024
        return {
            "policy_version": 3,
            "coexistence_policy_version": 1,
            "workspace": str(ROOT),
            "cwd": str(ROOT),
            "mode": "gpu-smoke",
            "protected_cpu_only_coexistence_requested": False,
            "protected_zero_nvidia_gpu_smoke_requested": requested,
            "protected_activity_waiver_applied": False,
            "protected_gpu_smoke_waiver_applied": False,
            "ok": True,
            "failures": [],
            "mem_available_kib": available,
            "memory_floor_kib": floor,
            "gpu_smoke_policy_version": 2,
            "gpu_smoke_free_memory_floor_mib": 4096,
            "gpu_smoke_admission_policy": preflight.policy_document(),
            "gpu": {
                "name": contract.EXPECTED_DEVICE_NAME,
                "compute_capability": "12.0",
                "memory_mib": "24576",
                "used_mib": "1024",
                "driver": contract.EXPECTED_DRIVER_RELEASE,
            },
            "torch": {
                "source": "static ENVIRONMENT.lock and selected-prefix file audit",
                "environment_lock_sha256": contract.ENVIRONMENT_LOCK_SHA256,
                "version": contract.EXPECTED_TORCH_VERSION,
                "git_version": contract.EXPECTED_TORCH_GIT,
                "cuda": contract.EXPECTED_TORCH_CUDA,
                "hip": None,
                "python_executable": str(
                    (ROOT / contract.PYTHON_REAL_RELATIVE_PATH).resolve(strict=True)
                ),
                "libcudart_path": str(
                    (ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True)
                ),
                "libcudart_size": contract.CUDA_RUNTIME_SIZE,
                "libcudart_sha256": contract.CUDA_RUNTIME_SHA256,
                "libcudart_record_owned": True,
                "nvidia_runtime_mappings": [],
                "cuda_initialized": False,
                "forbidden_dsos": [],
            },
            "protected_processes": [],
            "protected_heavy_processes": [],
            "nvidia_compute_pids": [],
            "nvidia_compute_audit_ok": True,
            "protected_nvidia_compute_pids": [],
            "protected_nvidia_runtime_mapping_pids": [],
            "unreadable_protected_maps": [],
            "workspace_processes": [],
            "policy": "observation-only; no external process is ever signalled",
        }

    def test_controller_and_finalizer_accept_only_exact_boundary(self) -> None:
        floor = 23_068_672
        report = self._report(available=floor)
        self.assertIs(
            controller.validate_preflight_report(
                report, allow_protected=True, description="fixture"
            ),
            report,
        )
        self.assertTrue(finalizer.validate_preflight(report, description="fixture"))
        below = self._report(available=floor - 1)
        with self.assertRaises(controller.ControllerV2Error):
            controller.validate_preflight_report(
                below, allow_protected=True, description="fixture"
            )
        with self.assertRaises(finalizer.FinalizeV2Error):
            finalizer.validate_preflight(below, description="fixture")

    def test_child_gate_records_host_observation_and_floor(self) -> None:
        identity = {"manifest": "fixture"}
        gate = {
            "static_identity": {},
            "gpu": {
                "name": contract.EXPECTED_DEVICE_NAME,
                "compute_capability": "12.0",
                "memory_mib": "24576",
                "used_mib": "1024",
                "driver": contract.EXPECTED_DRIVER_RELEASE,
            },
            "free_memory_mib": 23_552,
            "mem_available_kib": 23_068_672,
            "host_memory_floor_kib": 23_068_672,
            "admission_policy": preflight.policy_document(),
            "protected_heavy_pids": [],
            "protected_runtime_pids": [],
            "unreadable_protected_maps": [],
            "nvidia_compute_pids": [],
            "control_manifest": identity,
            "base_runner": {
                "path": contract.BASE_RUNNER_RELATIVE_PATH.as_posix(),
                "bytes": contract.BASE_RUNNER_SIZE,
                "sha256": contract.BASE_RUNNER_SHA256,
            },
        }
        finalizer.validate_child_gate(gate, requested=True, control_identity=identity)
        gate["mem_available_kib"] = 23_068_671
        with self.assertRaises(finalizer.FinalizeV2Error):
            finalizer.validate_child_gate(
                gate, requested=True, control_identity=identity
            )


class CompositionAndPublicationTest(unittest.TestCase):
    def test_exact_case_helpers_and_injected_dependencies_are_frozen(self) -> None:
        self.assertIs(contract.CASE_SPECS, contract._BASE.CASE_SPECS)
        self.assertIs(contract.CASE_ORDER, contract._BASE.CASE_ORDER)
        for helper in (
            child.base.make_program,
            child.base.canonical_tensor_ir_source,
            child.base.validate_structured_result,
            child.base.execute_case,
        ):
            self.assertIs(helper.__globals__, child.base.__dict__)
        for helper in (
            finalizer.base.validate_frontend_results,
            finalizer.base.validate_audit,
            finalizer.base._input_words,
            finalizer.base._cpu_reference_words,
            finalizer.base._compare_words,
        ):
            self.assertIs(helper.__globals__, finalizer.base.__dict__)
        self.assertIs(controller.isolation.preflight_tool, controller.preflight)
        self.assertIs(controller.isolation.stop_run, controller.stop_run)
        self.assertIs(controller.isolation.nvidia_smoke_contract, controller.contract)
        self.assertIs(controller.isolation.nvidia_smoke_control, controller.control)

    def test_full_controller_sequence_is_ordered(self) -> None:
        source = (
            ROOT / "tools/run_pypto_fused_pointwise_sm120_v2_isolated.py"
        ).read_text()
        markers = (
            'description="initial v2 preflight"',
            'description="action-boundary v2 preflight"',
            "process = subprocess.Popen(",
            "metadata = isolation.build_run_metadata(",
            "released, early_code = _gate_and_release(",
            "isolation.wait_with_gpu_smoke_watchdog(",
            "post_snapshot, post_violation = isolation.audit_gpu_smoke_runtime_state(",
            "enforce_no_surviving_owned_processes(process, metadata)",
        )
        positions = [source.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_survivor_path_uses_only_verified_owned_helpers(self) -> None:
        process = mock.Mock()
        metadata: dict[str, object] = {"pgid": 7}
        with (
            mock.patch.object(
                controller.stop_run, "owned_group_members", return_value=[101]
            ),
            mock.patch.object(
                controller.isolation, "terminate_owned_process", return_value=0
            ) as terminate,
        ):
            self.assertFalse(
                controller.enforce_no_surviving_owned_processes(process, metadata)
            )
        terminate.assert_called_once_with(process, metadata, wait_seconds=5)
        self.assertEqual(metadata["surviving_group_pids"], [101])

    def test_environment_lock_acquisition_blocks_and_restores_signals(self) -> None:
        calls: list[tuple[object, object]] = []

        def mask(how, values):
            calls.append((how, values))
            return {"old-mask"}

        with (
            mock.patch.object(controller.signal, "pthread_sigmask", side_effect=mask),
            mock.patch.object(
                controller.isolation,
                "acquire_environment_lock",
                side_effect=controller.isolation.EnvironmentLockBusy("busy"),
            ),
            mock.patch.object(controller.sys, "stderr"),
        ):
            self.assertIsNone(controller.acquire_shared_environment_lease())
        self.assertEqual(calls[0][0], controller.signal.SIG_BLOCK)
        self.assertEqual(calls[1], (controller.signal.SIG_SETMASK, {"old-mask"}))

    def test_direct_child_gate_rechecks_exact_boundary(self) -> None:
        identity = {"v2": True}
        fake_control = SimpleNamespace(
            reject_control_bytecode_cache=lambda _root: None,
            validate_control_manifest=lambda _root: identity,
        )
        fake_preflight = SimpleNamespace(
            static_torch_identity=lambda: {},
            mem_available_kib=lambda: 23_068_672,
            nvidia_identity=lambda: {
                "name": contract.EXPECTED_DEVICE_NAME,
                "compute_capability": "12.0",
                "memory_mib": "24576",
                "used_mib": "1024",
                "driver": contract.EXPECTED_DRIVER_RELEASE,
            },
            nvidia_compute_pids=lambda: set(),
            process_table=lambda: ([], [], []),
            protected_nvidia_runtime_mappings=lambda _items: ([], []),
            is_heavy_command=lambda _command: False,
            policy_document=preflight.policy_document,
        )

        def dispatch(name, _path, **_kwargs):
            if "control_manifest" in name:
                return fake_control
            if "contract" in name:
                return contract
            return fake_preflight

        environment = {
            "PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED": "1",
            "PYPTO_GPU_SMOKE_AUTHORIZATION": contract.GPU_SMOKE_AUTHORIZATION,
        }
        with (
            mock.patch.object(child, "load_exact", side_effect=dispatch),
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(sys, "orig_argv", contract.fixed_child_command(ROOT)),
        ):
            _, gate = child.load_contract_and_child_gate(
                ROOT, {"control_manifest": identity}
            )
        self.assertEqual(gate["mem_available_kib"], 23_068_672)
        self.assertEqual(gate["host_memory_floor_kib"], 23_068_672)

    def test_no_prohibited_runtime_rewriting_or_direct_signalling(self) -> None:
        sources = [
            (ROOT / "tools/run_pypto_fused_pointwise_sm120_v2_isolated.py").read_text(),
            (
                ROOT / "benchmarks/operators/pypto_fused_pointwise_sm120_v2.py"
            ).read_text(),
            (ROOT / "tools/finalize_pypto_fused_pointwise_sm120_v2.py").read_text(),
        ]
        joined = "\n".join(sources)
        for forbidden in (
            "sys.orig_argv =",
            "subprocess.run =",
            ".subprocess =",
            "os.kill(",
            "os.killpg(",
            "22 * 1024 * 1024 if",  # no 22-to-24 normalization shim
        ):
            self.assertNotIn(forbidden, joined)
        self.assertIn("isolation.wait_with_gpu_smoke_watchdog", joined)
        self.assertIn("base.validate_frontend_results", joined)
        self.assertIn("base._cpu_reference_words", joined)
        self.assertIn("base.REPLAY_AUDIT_PROGRAM", joined)
        self.assertIn("base.publish_no_replace", joined)

    def test_manifest_publication_is_a_separate_fail_closed_step(self) -> None:
        manifest = ROOT / control.MANIFEST_RELATIVE_PATH
        source = (
            ROOT / "tools/run_pypto_fused_pointwise_sm120_v2_isolated.py"
        ).read_text()
        self.assertLess(
            source.index("control.validate_control_manifest(ROOT)"),
            source.index("subprocess.Popen("),
        )
        self.assertEqual(len(control.CONTROL_PATHS), 6)
        if not manifest.is_file():
            with (
                mock.patch.object(control, "_base_identity", return_value={}),
                self.assertRaisesRegex(
                    control.ControlManifestV2Error, "manifest is missing"
                ),
            ):
                control.validate_control_manifest(ROOT)
            return
        dirty = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        if dirty:
            with self.assertRaises(RuntimeError):
                control.validate_control_manifest(ROOT)
        else:
            identity = control.validate_control_manifest(ROOT)
            self.assertTrue(identity["root_clean"])
            self.assertEqual(
                [record["path"] for record in identity["files"]],
                list(control.CONTROL_PATHS),
            )


class FinalizerOutputJoinRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = b"frozen-reference-bytes"
        self.actual = b"frozen-actual-bytes"
        self.metrics = {
            "observed_max_ulp": 1,
            "observed_max_relative_error": 0.001,
            "observed_max_absolute_error": 0.0005,
        }
        self.execution = {
            "expected_logical_bytes_sha256": finalizer.sha256_bytes(self.reference),
            "actual_logical_bytes_sha256": finalizer.sha256_bytes(self.actual),
            "comparison": {
                "policy": "ulp-and-relative",
                "max_ulp_limit": 4,
                "rtol": 2.0e-6,
                "atol": 0.0,
                **self.metrics,
                "special_classification_and_sign_passed": True,
                "negative_zero_fma_discriminator_passed": False,
                "no_subnormals": True,
                "comparison_passed": True,
            },
        }

    def test_full_comparison_record_accepts_reconstructed_metric_subset(self) -> None:
        finalizer.validate_execution_outputs(
            self.execution, self.reference, self.actual, self.metrics
        )

    def test_reference_raw_tamper_is_rejected(self) -> None:
        with self.assertRaisesRegex(finalizer.FinalizeV2Error, "raw output"):
            finalizer.validate_execution_outputs(
                self.execution, self.reference + b"!", self.actual, self.metrics
            )

    def test_actual_raw_tamper_is_rejected(self) -> None:
        with self.assertRaisesRegex(finalizer.FinalizeV2Error, "raw output"):
            finalizer.validate_execution_outputs(
                self.execution, self.reference, self.actual + b"!", self.metrics
            )


class AuditorParityRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = AdmissionBoundaryTest()._report(
            available=23_068_672, requested=True
        )
        self.run_id = "pypto-20260826T120000Z-123456-abcdef"

    def test_preflight_rejects_policy_cpu_flag_and_inventory_drift(self) -> None:
        mutations = (
            ("coexistence version", ("coexistence_policy_version",), 2),
            (
                "CPU-only request",
                ("protected_cpu_only_coexistence_requested",),
                True,
            ),
            ("policy", ("policy",), "signals may be sent"),
            ("inventory", ("protected_processes",), [{"pid": 1}]),
        )
        for name, path, value in mutations:
            candidate = copy.deepcopy(self.preflight)
            candidate[path[0]] = value
            with self.subTest(name=name), self.assertRaises(finalizer.FinalizeV2Error):
                finalizer.validate_preflight(candidate, description="fixture")

    def _process(self) -> dict[str, object]:
        floor = 23_068_672
        return {
            "schema": 4,
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
                "memory_floor_kib": floor,
                "protected_heavy_processes": [],
                "protected_nvidia_compute_pids": [],
            },
            "gpu_smoke": {
                "policy_version": 2,
                "requested": True,
                "waiver_applied": False,
                "authorization": contract.GPU_SMOKE_AUTHORIZATION,
                "start_barrier_path": str(
                    ROOT / "runs" / self.run_id / "gpu-smoke-start-barrier.json"
                ),
                "gate_path": str(ROOT / "runs" / self.run_id / "gpu-smoke-gate.json"),
                "memory_floor_kib": floor,
                "gpu_free_memory_floor_mib": 4096,
                "protected_heavy_processes": [],
                "protected_nvidia_compute_pids": [],
                "protected_nvidia_runtime_mapping_pids": [],
                "unreadable_protected_maps": [],
                "gate_sha256": "1" * 64,
                "start_barrier_sha256": "2" * 64,
                "release_authorized_at": "20260826T120001Z",
            },
            "initial_preflight": {},
            "preflight": {},
            "resource_policy": {},
            "command": [],
            "pid": 10,
            "pgid": 10,
            "start_ticks": 20,
            "started_at": "20260826T120000Z",
            "status": "exited",
            "gpu_smoke_pre_release_audit": {},
            "gpu_smoke_last_audit": {},
            "gpu_smoke_post_exit_audit": {},
            "return_code": 0,
            "finished_at": "20260826T120002Z",
        }

    def test_process_rejects_environment_lock_framework_timestamp_and_gpu_drift(
        self,
    ) -> None:
        process = self._process()
        finalizer.validate_process_policy(
            process,
            run_id=self.run_id,
            requested=True,
            preflight=self.preflight,
        )
        mutations = (
            ("environment", ("environment",), "/wrong"),
            ("lock", ("environment_access_lock", "mode"), "exclusive"),
            ("framework", ("framework_launch",), True),
            ("timestamp", ("finished_at",), "bad"),
            ("GPU floor", ("gpu_smoke", "gpu_free_memory_floor_mib"), 4095),
        )
        for name, path, value in mutations:
            candidate = copy.deepcopy(process)
            target = candidate
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(name=name), self.assertRaises(finalizer.FinalizeV2Error):
                finalizer.validate_process_policy(
                    candidate,
                    run_id=self.run_id,
                    requested=True,
                    preflight=self.preflight,
                )

    def _runtime_provisional(self) -> tuple[dict[str, object], dict[str, object]]:
        trait_names = {
            "compute_capability",
            "multiprocessor_count",
            "warp_size",
            "max_threads_per_block",
            "max_threads_per_multiprocessor",
            "max_blocks_per_multiprocessor",
            "max_block_dim_x",
            "max_block_dim_y",
            "max_block_dim_z",
            "max_grid_dim_x",
            "max_grid_dim_y",
            "max_grid_dim_z",
            "l1_cache_line_bytes",
            "default_shared_memory_per_cta_bytes",
            "max_shared_memory_per_cta_bytes",
            "shared_memory_per_multiprocessor_bytes",
            "registers_per_cta",
            "max_registers_per_thread",
            "registers_per_multiprocessor",
            "l2_cache_size_bytes",
            "total_global_memory_bytes",
        }
        traits = {name: 1 for name in trait_names}
        traits["compute_capability"] = 120
        traits["multiprocessor_count"] = contract.EXPECTED_SM_COUNT
        torch = {
            "version": contract.EXPECTED_TORCH_VERSION,
            "git_version": contract.EXPECTED_TORCH_GIT,
            "cuda": contract.EXPECTED_TORCH_CUDA,
            "hip": None,
            "module_path": str(
                (
                    ROOT
                    / "envs/pypto-nvidia/lib/python3.14/site-packages/torch/__init__.py"
                ).resolve(strict=True)
            ),
        }
        runtime = {
            "torch": torch,
            "libcudart_paths": [
                str((ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True))
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
                "cuda_driver_api_version": contract.MINIMUM_CUDA_DRIVER_API_VERSION,
                "cuda_runtime_api_version": contract.MINIMUM_CUDA_RUNTIME_API_VERSION,
                "cuda_runtime_library_path": str(
                    (ROOT / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve(strict=True)
                ),
                "context_address": 1,
                "context_id": 2,
            },
            "compile_request": {
                "byte_identity_digest": "1" * 64,
                "loader_compatibility_input_digest": "2" * 64,
                "device_autotune_identity_digest": "3" * 64,
            },
        }
        return {"runtime": runtime}, {"static_identity": self.preflight["torch"]}

    def test_runtime_rejects_trait_uuid_dtype_and_context_drift(self) -> None:
        provisional, gate = self._runtime_provisional()
        finalizer.validate_runtime_identity(provisional, self.preflight, gate)
        mutations = (
            ("trait", ("traits", "warp_size"), 0),
            ("UUID", ("device_uuid",), "GPU-bad"),
            ("dtype", ("supported_compute_dtypes",), ["FP32"]),
            ("context", ("context_id",), 0),
        )
        for name, path, value in mutations:
            candidate = copy.deepcopy(provisional)
            observation = candidate["runtime"]["observation"]
            target = observation
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(name=name), self.assertRaises(finalizer.FinalizeV2Error):
                finalizer.validate_runtime_identity(candidate, self.preflight, gate)

    def test_provisional_input_identity_rejects_source_lineage_tamper(self) -> None:
        identity = {"control": "v2"}
        inputs = {
            "integrity": {},
            "pypto": {
                "head": contract.PYPTO_HEAD,
                "tree": contract.PYPTO_TREE,
                "clean": True,
            },
            "tensor_ir_head": contract.TENSOR_IR_HEAD,
            "cuda_tile_head": contract.CUDA_TILE_HEAD,
            "llvm_head": contract.LLVM_HEAD,
            "replay_files": [],
            "control_manifest": identity,
        }
        finalizer.validate_provisional_input_identity(inputs, identity)
        for name in ("pypto", "tensor_ir_head", "cuda_tile_head", "llvm_head"):
            candidate = copy.deepcopy(inputs)
            if name == "pypto":
                candidate[name]["head"] = "0" * 40
            else:
                candidate[name] = "0" * 40
            with self.subTest(name=name), self.assertRaises(finalizer.FinalizeV2Error):
                finalizer.validate_provisional_input_identity(candidate, identity)

    def test_integrity_rejects_symlink_and_nonregular_paths_before_resolve(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            root = pathlib.Path(directory)
            regular = root / "regular"
            regular.write_bytes(b"value")
            symlink = root / "symlink"
            symlink.symlink_to(regular)
            with self.assertRaisesRegex(finalizer.FinalizeV2Error, "non-symlink"):
                finalizer.exact_integrity_record(symlink, "symlink fixture")
            with self.assertRaisesRegex(finalizer.FinalizeV2Error, "regular"):
                finalizer.exact_integrity_record(root, "directory fixture")


class DocumentationTest(unittest.TestCase):
    def test_v2_runbook_freezes_boundaries_and_nonclaims(self) -> None:
        text = (ROOT / "docs/pypto_fused_pointwise_sm120_smoke_v2.md").read_text()
        for marker in (
            "23,068,672 KiB",
            "16 GiB",
            "4 GiB",
            "32 GiB",
            "v1 remains byte-for-byte unchanged",
            "manifest-only",
            "does not claim",
        ):
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
