from __future__ import annotations

import ast
import copy
import contextlib
import importlib.util
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
    "_pypto_frontend_vector_add_sm120_contract",
    ROOT / "tools/_pypto_frontend_vector_add_sm120_contract.py",
)
control_manifest = load_tool(
    "_pypto_frontend_sm120_control_manifest",
    ROOT / "tools/_pypto_frontend_sm120_control_manifest.py",
)
runner = load_tool(
    "frontend_smoke_test_runner",
    ROOT / "benchmarks/operators/pypto_frontend_vector_add_sm120.py",
)
controller = load_tool(
    "frontend_smoke_test_controller",
    ROOT / "tools/run_pypto_frontend_sm120_isolated.py",
)
finalizer = load_tool(
    "frontend_smoke_test_finalizer",
    ROOT / "tools/finalize_pypto_frontend_vector_add_sm120.py",
)


def function_node(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def called_attribute_lines(node: ast.AST, attribute: str) -> list[int]:
    return sorted(
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == attribute
    )


class FrontendSmokeContractTest(unittest.TestCase):
    def test_case_matrix_is_exact(self) -> None:
        self.assertEqual(
            [
                (
                    case.name,
                    case.dtype,
                    case.shape,
                    case.strides,
                    case.tile_sizes,
                    case.expected_grid,
                    case.expected_kernel_arguments,
                    case.repetitions,
                )
                for case in contract.CASE_SPECS
            ],
            [
                (
                    "fp32_8x8",
                    "float32",
                    (8, 8),
                    (8, 1),
                    (16,),
                    (4, 1, 1),
                    3,
                    2,
                ),
                (
                    "bf16_128",
                    "bfloat16",
                    (128,),
                    (1,),
                    (16,),
                    (8, 1, 1),
                    3,
                    2,
                ),
            ],
        )
        self.assertEqual(
            [
                (case.expected_hir_bytes, case.expected_hir_sha256)
                for case in contract.CASE_SPECS
            ],
            [
                (
                    2_077,
                    "2a4bf1b74adb2e160ba9a1f6c1596238095d73a1572fd911417f43465b6fced5",
                ),
                (
                    1_922,
                    "cc470968297ff4430fdcf43f25bf6cb4b9054e6ba4d67ddd0177040101044baa",
                ),
            ],
        )
        self.assertEqual(
            [
                (
                    case.expected_source_ir_digest,
                    case.expected_static_specialization_digest,
                    case.expected_symbolic_specialization_digest,
                    case.expected_argument_abi_digest,
                    case.expected_result_abi_digest,
                    case.expected_mutation_abi_digest,
                    case.expected_callable_abi_digest,
                    case.expected_device_code_bytes,
                    case.expected_device_code_sha256,
                )
                for case in contract.CASE_SPECS
            ],
            [
                (
                    "0e5fbaf1cd70dffa0c81d43a1d2cad454f97cbf9a57ae5247da1cb27f6a049d3",
                    "0bad0b86c36c86808f12b908692925e6493b4c78027da334d612b57bac5459aa",
                    "1a2d6d32c956e86f41c0dc50dbafe53d33e3f63bd58bd683e78a06595b6ff58c",
                    "2bf59cefa95a2e95ac4e4647a654118b6d8c6fbfaa91a294621a9160d68dbd9b",
                    "5a786136731e5c62c80597edbb57551cd010d344bfeab14456a9f6b263e99ea5",
                    "4adb7fa8fdbdee33582778c543686a6c63953852be40079649e4d2d6f07c766d",
                    "4eee741cbe0c7322f938b83b01b34844c5f12fb1b01da52c2029e6687e24c640",
                    13_784,
                    "dcc529fc856a508642c8b5a98c6fc4e223e10a49cc9f8a200b8984f92b6483ab",
                ),
                (
                    "c22f2459ad794e89f88de1bbd427f17876c6059b9fd222be706dd5ce300a0a7f",
                    "146fbdd823eb18c77190894a272a39d1298b557acf8d3156226d44d5ce7a6051",
                    "1a2d6d32c956e86f41c0dc50dbafe53d33e3f63bd58bd683e78a06595b6ff58c",
                    "5408cc8d6adb1f52d11d850c745a33dbe6a43c470fa907a238b71371f4bf04c1",
                    "e8111628d00263e8f568dc8844a2e0fe5d08576bf77d81682b338dbeb977460a",
                    "4adb7fa8fdbdee33582778c543686a6c63953852be40079649e4d2d6f07c766d",
                    "e3d31183b4ba5b2f09f01ef777043b9f51818e33d6bdbf18333257211d57c0e8",
                    13_784,
                    "83afb2df234ad90167351d608052d44f86e26a8ca73959369992cd139943bc13",
                ),
            ],
        )

    def test_exact_product_identity_includes_stale_directory_label(self) -> None:
        self.assertEqual(
            contract.PYPTO_HEAD,
            "642ff5bd79ee96b9e5a279a2bc945ad7a78362b7",
        )
        self.assertEqual(
            contract.PYPTO_TREE,
            "77d8078d8df84dd7cf8544350918e25b8282976d",
        )
        self.assertEqual(contract.PYPTO_DSO_SIZE, 595_300_112)
        self.assertEqual(
            contract.PYPTO_DSO_SHA256,
            "4b796b1e1c53386356217f9ea6368468f885e68fb98fa715f90a081031ecc6fb",
        )
        self.assertIn("on-c4cf755", contract.PYPTO_DSO_RELATIVE_PATH.as_posix())

    def test_contract_covers_every_unchanged_primitive_dependency(self) -> None:
        for relative in ("tools/preflight.py", "tools/run_isolated.py"):
            tree = ast.parse((ROOT / relative).read_text())
            required = {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "nvidia_smoke_contract"
            }
            self.assertFalse(required - set(vars(contract)), relative)
        self.assertEqual(contract.CUDA_RUNTIME_DISTRIBUTION, "nvidia-cuda-runtime")
        self.assertEqual(contract.CUDA_RUNTIME_VERSION, "13.0.96")

    def test_importing_runner_adds_no_gpu_or_framework_module(self) -> None:
        path = ROOT / contract.RUNNER_RELATIVE_PATH
        before = set(sys.modules)
        load_tool("frontend_smoke_runner_second_import", path)
        added = set(sys.modules) - before
        self.assertTrue(
            {"torch", "pypto", "triton", "sglang", "flashinfer"}.isdisjoint(added)
        )

    def test_fixed_child_and_runner_blob_are_exact(self) -> None:
        command = contract.fixed_child_command(ROOT)
        self.assertEqual(command[1:4], ["-I", "-B", "-S"])
        self.assertEqual(command[-1], str(ROOT / contract.RUNNER_RELATIVE_PATH))
        runner_path = ROOT / contract.RUNNER_RELATIVE_PATH
        self.assertEqual(runner_path.stat().st_size, contract.RUNNER_SIZE)
        self.assertEqual(runner.sha256_file(runner_path), contract.RUNNER_SHA256)

    def test_new_manifest_reuses_exact_v4_primitives(self) -> None:
        v4 = json.loads(
            (ROOT / "state/contracts/pypto_nvidia_executable_sm120_v4.json").read_text()
        )
        v4_records = {record["path"]: record for record in v4["files"]}
        for relative in (
            "tools/preflight.py",
            "tools/run_isolated.py",
            "tools/stop_run.py",
        ):
            record = v4_records[relative]
            path = ROOT / relative
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(control_manifest.sha256_file(path), record["sha256"])
        self.assertEqual(
            control_manifest.CONTROL_PATHS[-3:],
            ("tools/preflight.py", "tools/run_isolated.py", "tools/stop_run.py"),
        )
        parent = control_manifest.validate_parent_control_manifest(ROOT)
        self.assertEqual(
            parent["sha256"],
            "a079c4d252aa346bb19a64a6ad3947867b76e7c778f7234125078fb16b2598bf",
        )
        self.assertEqual(
            [record["path"] for record in parent["primitive_files"]],
            list(control_manifest.CONTROL_PATHS[-3:]),
        )

    def test_control_manifest_is_fail_closed_or_fully_valid(self) -> None:
        manifest = ROOT / control_manifest.MANIFEST_RELATIVE_PATH
        root_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        )
        if not manifest.is_file():
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "manifest is missing"
            ):
                control_manifest.validate_control_manifest(ROOT)
        elif root_dirty:
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "not clean"
            ):
                control_manifest.validate_control_manifest(ROOT)
        else:
            identity = control_manifest.validate_control_manifest(ROOT)
            self.assertTrue(identity["root_clean"])
            self.assertEqual(
                [record["path"] for record in identity["files"]],
                list(control_manifest.CONTROL_PATHS),
            )

    def test_manifest_validator_joins_real_commit_and_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Frontend Smoke Fixture"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "smoke@example.invalid"],
                cwd=repository,
                check=True,
            )
            parent_manifest = (
                ROOT / control_manifest.PARENT_MANIFEST_RELATIVE_PATH
            ).read_bytes()
            parent_path = repository / control_manifest.PARENT_MANIFEST_RELATIVE_PATH
            parent_path.parent.mkdir(parents=True, exist_ok=True)
            parent_path.write_bytes(parent_manifest)
            parent_records = {
                record["path"]: record
                for record in json.loads(parent_manifest)["files"]
            }
            records = []
            for relative in control_manifest.CONTROL_PATHS:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative in parent_records:
                    path.write_bytes((ROOT / relative).read_bytes())
                    path.chmod(parent_records[relative]["mode"])
                else:
                    path.write_bytes(f"reviewed:{relative}\n".encode())
                    path.chmod(0o644)
                records.append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": control_manifest.sha256_file(path),
                        "mode": path.stat().st_mode & 0o777,
                    }
                )
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
            manifest = {
                "schema_version": control_manifest.MANIFEST_SCHEMA_VERSION,
                "kind": control_manifest.MANIFEST_KIND,
                "implementation_commit": implementation,
                "implementation_tree": tree,
                "files": records,
            }
            path = repository / control_manifest.MANIFEST_RELATIVE_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(control_manifest.canonical_json(manifest))
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "publish manifest"],
                cwd=repository,
                check=True,
            )
            identity = control_manifest.validate_control_manifest(repository)
            self.assertEqual(identity["implementation_commit"], implementation)
            (repository / control_manifest.CONTROL_PATHS[0]).write_text("tampered\n")
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "not clean"
            ):
                control_manifest.validate_control_manifest(repository)


class FrontendRunnerStructureTest(unittest.TestCase):
    def test_hir_roundtrip_precedes_single_compile_call(self) -> None:
        path = ROOT / contract.RUNNER_RELATIVE_PATH
        run_smoke = function_node(path, "run_smoke")
        serialize_lines = called_attribute_lines(run_smoke, "serialize")
        deserialize_lines = called_attribute_lines(run_smoke, "deserialize")
        compile_lines = called_attribute_lines(run_smoke, "compile_structured_strict")
        self.assertEqual(len(compile_lines), 1)
        self.assertTrue(any(line < compile_lines[0] for line in serialize_lines))
        self.assertTrue(any(line < compile_lines[0] for line in deserialize_lines))
        source = path.read_text()
        self.assertIn("This is the one and only producer invocation", source)
        self.assertIn('"compile_invocations_per_case": 1', source)
        self.assertIn("pypto_module.ir.deserialize(hir_bytes)", source)
        self.assertIn("(lhs, ir.ParamDirection.In)", source)
        self.assertIn("(rhs, ir.ParamDirection.In)", source)
        self.assertIn(
            "(left_cpu.float() + right_cpu.float()).to(torch.bfloat16)", source
        )

    def test_each_execution_has_sync_release_and_unload(self) -> None:
        execute = function_node(ROOT / contract.RUNNER_RELATIVE_PATH, "execute_case")
        constructors = [
            node
            for node in ast.walk(execute)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "NvidiaExecutable"
        ]
        self.assertEqual(len(constructors), 1)
        sync = called_attribute_lines(execute, "synchronize")
        unload = called_attribute_lines(execute, "unload")
        deletes = [
            node.lineno
            for node in ast.walk(execute)
            if isinstance(node, ast.Delete)
            and any(
                isinstance(target, ast.Name) and target.id == "packet"
                for target in node.targets
            )
        ]
        self.assertEqual(len(sync), 1)
        self.assertEqual(len(deletes), 1)
        self.assertEqual(len(unload), 1)
        self.assertLess(sync[0], deletes[0])
        self.assertLess(deletes[0], unload[0])
        self.assertGreaterEqual(
            len(called_attribute_lines(execute, "is_current_stream_capturing")), 2
        )

    def test_controller_delegates_only_fixed_gpu_smoke(self) -> None:
        argv = controller.delegated_argv(
            contract,
            run_id_file=ROOT / "runs/frontend-next.json",
            allow_protected_zero_nvidia_gpu_smoke=True,
        )
        self.assertIn("--exact-pypto-nvidia-smoke", argv)
        self.assertIn("--allow-protected-zero-nvidia-gpu-smoke", argv)
        self.assertEqual(argv[-5:], contract.fixed_child_command(ROOT))
        self.assertNotIn("gpu-benchmark", argv)

    def test_artifact_target_surface_uses_direct_compute_capability(self) -> None:
        artifact = types.SimpleNamespace(
            actual_target=types.SimpleNamespace(compute_capability=120)
        )
        runner.require_sm120_artifact_target(artifact)
        artifact.actual_target.compute_capability = 100
        with self.assertRaisesRegex(runner.SmokeError, "not SM120"):
            runner.require_sm120_artifact_target(artifact)
        runner_source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text()
        self.assertNotIn("actual_target.traits.compute_capability", runner_source)
        self.assertNotIn(
            "actual_target.traits.compute_capability",
            finalizer.REPLAY_AUDIT_PROGRAM,
        )


def synthetic_provisional() -> dict[str, object]:
    hir_programs = []
    artifacts = []
    executions = []
    for index, case in enumerate(contract.CASE_SPECS):
        artifact_sha = f"{index + 3:x}" * 64
        hir_programs.append(
            {
                "case": case.name,
                "bytes": case.expected_hir_bytes,
                "sha256": case.expected_hir_sha256,
                "serialized_once": True,
                "deserialized_before_compile": True,
                "canonical_reserialization_equal": True,
                "structural_equal": True,
                "parameter_directions": ["In", "In"],
            }
        )
        artifacts.append(
            {
                "case": case.name,
                "compile_api": "pypto.compiler.compile_structured_strict",
                "compiler_invocations": 1,
                "build_spec_identity_digest": "a" * 64,
                "source_ir_digest": case.expected_source_ir_digest,
                "callable_abi_digest": case.expected_callable_abi_digest,
                "static_specialization_digest": (
                    case.expected_static_specialization_digest
                ),
                "symbolic_specialization_digest": (
                    case.expected_symbolic_specialization_digest
                ),
                "argument_abi_digest": case.expected_argument_abi_digest,
                "result_abi_digest": case.expected_result_abi_digest,
                "mutation_abi_digest": case.expected_mutation_abi_digest,
                "artifact_identity_digest": artifact_sha,
                "cache_key_digest": "4" * 64,
                "loader_compatibility_digest": "5" * 64,
                "device_code_bytes": case.expected_device_code_bytes,
                "device_code_sha256": case.expected_device_code_sha256,
                "kernel_abi_identity_digest": case.expected_callable_abi_digest,
                "entry_function_name": "pypto_vector_add",
                "fallback_used": False,
                "expected_grid": list(case.expected_grid),
                "expected_kernel_arguments": case.expected_kernel_arguments,
                "expected_device_code_bytes": case.expected_device_code_bytes,
                "expected_device_code_sha256": case.expected_device_code_sha256,
                "hir_sha256": case.expected_hir_sha256,
                "hir_bytes": case.expected_hir_bytes,
                "hir_roundtrip_exact": True,
            }
        )
        for repetition in range(case.repetitions):
            executions.append(
                {
                    "case": case.name,
                    "repetition": repetition,
                    "lifetime_ordinal": repetition,
                    "fresh_executable": True,
                    "artifact_identity_digest": artifact_sha,
                    "dtype": case.dtype,
                    "shape": list(case.shape),
                    "strides": list(case.strides),
                    "grid": list(case.expected_grid),
                    "kernel_argument_count": case.expected_kernel_arguments,
                    "raw_current_stream": 100 + index,
                    "non_default_stream": True,
                    "capture_free_before": True,
                    "capture_free_at_launch": True,
                    "external_stream_synchronized": True,
                    "expected_logical_bytes_sha256": "7" * 64,
                    "actual_logical_bytes_sha256": "7" * 64,
                    "input_bytes_sha256": "8" * 64,
                    "input_unchanged": True,
                    "torch_equal": True,
                    "packet_released_after_synchronization": True,
                    "explicit_unload": True,
                    "terminal_state": "Unloaded",
                    "bound_context_before_unload": 123,
                    "bound_context_id_before_unload": 456,
                    "bound_context_after_unload": 0,
                    "bound_context_id_after_unload": 0,
                }
            )
    return {
        "scope": {
            "provider": "pypto.tensorir",
            "frontend_hir": True,
            "runtime_object": "NvidiaExecutable",
            "operator_correctness": True,
            "model_forward": False,
            "strict_coverage_result": False,
            "performance_result": False,
            "cuda_graph_result": False,
        },
        "runtime": {
            "observation": {"context_address": 123, "context_id": 456},
            "hir_programs": hir_programs,
            "artifacts": artifacts,
            "executions": executions,
            "case_order": [case.name for case in contract.CASE_SPECS],
            "compile_invocations_per_case": 1,
            "repetitions_per_case": 2,
            "module_lifetimes": 4,
            "explicit_packet_releases": 4,
            "explicit_unloads": 4,
            "non_default_current_stream": True,
            "external_synchronization": True,
            "fallback_used": False,
            "forbidden_provider_imports": [],
        },
    }


class FrontendFinalizerTest(unittest.TestCase):
    def test_finalizer_replay_is_deserialization_only(self) -> None:
        compile(finalizer.REPLAY_AUDIT_PROGRAM, "<frontend-replay-audit>", "exec")
        forbidden = (
            "compile_structured_strict",
            "Artifact.compile_strict",
            "NvidiaExecutable(",
            "observe_current_nvidia_runtime",
            "nvidia-smi",
        )
        for marker in forbidden:
            self.assertNotIn(marker, finalizer.REPLAY_AUDIT_PROGRAM)
        self.assertIn("pypto.ir.deserialize(hir_bytes)", finalizer.REPLAY_AUDIT_PROGRAM)
        self.assertIn(
            "assert not torch.cuda.is_initialized()", finalizer.REPLAY_AUDIT_PROGRAM
        )
        tree = ast.parse(
            (ROOT / "tools/finalize_pypto_frontend_vector_add_sm120.py").read_text()
        )
        forbidden_calls = {
            "compile_structured_strict",
            "compile_strict",
            "observe_current_nvidia_runtime",
            "NvidiaExecutable",
        }
        self.assertFalse(
            [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and (
                    isinstance(node.func, ast.Name)
                    and node.func.id in forbidden_calls
                    or isinstance(node.func, ast.Attribute)
                    and node.func.attr in forbidden_calls
                )
            ]
        )

    def test_frontend_lifetime_contract_and_mutations(self) -> None:
        provisional = synthetic_provisional()
        finalizer.validate_scope(provisional)
        finalizer.validate_frontend_results(provisional)
        mutations = (
            ("compile_invocations_per_case", 2, "aggregate"),
            ("explicit_packet_releases", 3, "aggregate"),
            ("module_lifetimes", 3, "aggregate"),
        )
        for field, value, message in mutations:
            candidate = copy.deepcopy(provisional)
            candidate["runtime"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(finalizer.FinalizeError, message),
            ):
                finalizer.validate_frontend_results(candidate)
        candidate = copy.deepcopy(provisional)
        candidate["runtime"]["executions"][0]["raw_current_stream"] = 1
        with self.assertRaisesRegex(finalizer.FinalizeError, "default stream"):
            finalizer.validate_frontend_results(candidate)
        candidate = copy.deepcopy(provisional)
        candidate["runtime"]["executions"][0]["fresh_executable"] = False
        with self.assertRaisesRegex(finalizer.FinalizeError, "prove correctness"):
            finalizer.validate_frontend_results(candidate)

    def test_scope_rejects_broader_claims(self) -> None:
        provisional = synthetic_provisional()
        provisional["runtime"]["latency"] = 1.0
        with self.assertRaisesRegex(finalizer.FinalizeError, "unaccepted claim"):
            finalizer.validate_scope(provisional)
        provisional = synthetic_provisional()
        provisional["scope"]["model_forward"] = True
        with self.assertRaisesRegex(finalizer.FinalizeError, "scope exceeds"):
            finalizer.validate_scope(provisional)

    def test_duplicate_and_noncanonical_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = pathlib.Path(directory).resolve()
            duplicate = workspace / "duplicate.json"
            duplicate.write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaisesRegex(finalizer.FinalizeError, "duplicate"):
                finalizer.load_canonical(duplicate, workspace, "fixture")
            noncanonical = workspace / "noncanonical.json"
            noncanonical.write_bytes(b'{"a": 1}\n')
            with self.assertRaisesRegex(finalizer.FinalizeError, "canonical"):
                finalizer.load_canonical(noncanonical, workspace, "fixture")


class FrontendFinalizerFullFixtureTest(unittest.TestCase):
    """Adapt the accepted v4 full-transaction fixture to the frontend schema."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = pathlib.Path(self.temporary.name).resolve()
        self.run_id = f"pypto-20990101T000000Z-{os.getpid()}-{secrets.token_hex(3)}"

        self.relative_files = {
            "runner": pathlib.Path("fixtures/runner.py"),
            "pypto_dso": pathlib.Path("fixtures/pypto_core.so"),
            "cuda_runtime": pathlib.Path("fixtures/libcudart.so.13"),
            "python": pathlib.Path("fixtures/python3.14"),
            "contract": pathlib.Path(
                "tools/_pypto_frontend_vector_add_sm120_contract.py"
            ),
            "controller": pathlib.Path("tools/run_pypto_frontend_sm120_isolated.py"),
            "finalizer": pathlib.Path(
                "tools/finalize_pypto_frontend_vector_add_sm120.py"
            ),
            "environment_lock": pathlib.Path("ENVIRONMENT.lock"),
            "versions_lock": pathlib.Path("VERSIONS.lock"),
            "workspace_lock": pathlib.Path("WORKSPACE.lock"),
            "torch": pathlib.Path(
                "envs/pypto-nvidia/lib/python3.14/site-packages/torch/__init__.py"
            ),
        }
        for name, relative in self.relative_files.items():
            path = self.workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture:{name}\n".encode())
        (self.workspace / "projects/pypto").mkdir(parents=True)
        (self.workspace / "runs").mkdir()

        runner_path = self.workspace / self.relative_files["runner"]
        dso_path = self.workspace / self.relative_files["pypto_dso"]
        runtime_path = self.workspace / self.relative_files["cuda_runtime"]
        python_path = self.workspace / self.relative_files["python"]
        environment_lock = self.workspace / self.relative_files["environment_lock"]
        self.patchers = [
            mock.patch.object(finalizer, "ROOT", self.workspace),
            mock.patch.object(
                finalizer,
                "__file__",
                str(self.workspace / self.relative_files["finalizer"]),
            ),
            mock.patch.object(
                contract, "RUNNER_RELATIVE_PATH", self.relative_files["runner"]
            ),
            mock.patch.object(contract, "RUNNER_SIZE", runner_path.stat().st_size),
            mock.patch.object(
                contract, "RUNNER_SHA256", finalizer.sha256_file(runner_path)
            ),
            mock.patch.object(
                contract, "PYPTO_DSO_RELATIVE_PATH", self.relative_files["pypto_dso"]
            ),
            mock.patch.object(contract, "PYPTO_DSO_SIZE", dso_path.stat().st_size),
            mock.patch.object(
                contract, "PYPTO_DSO_SHA256", finalizer.sha256_file(dso_path)
            ),
            mock.patch.object(
                contract,
                "CUDA_RUNTIME_RELATIVE_PATH",
                self.relative_files["cuda_runtime"],
            ),
            mock.patch.object(
                contract, "CUDA_RUNTIME_SIZE", runtime_path.stat().st_size
            ),
            mock.patch.object(
                contract, "CUDA_RUNTIME_SHA256", finalizer.sha256_file(runtime_path)
            ),
            mock.patch.object(
                contract, "PYTHON_REAL_RELATIVE_PATH", self.relative_files["python"]
            ),
            mock.patch.object(contract, "PYTHON_SIZE", python_path.stat().st_size),
            mock.patch.object(
                contract, "PYTHON_SHA256", finalizer.sha256_file(python_path)
            ),
            mock.patch.object(
                contract,
                "ENVIRONMENT_LOCK_SHA256",
                finalizer.sha256_file(environment_lock),
            ),
            mock.patch.object(
                contract, "FINAL_REPORT_DIRECTORY", pathlib.Path("reports/data")
            ),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.control_identity = {
            "manifest_path": "state/contracts/frontend-fixture.json",
            "manifest_bytes": 1,
            "manifest_sha256": "f" * 64,
            "implementation_commit": "1" * 40,
            "implementation_tree": "2" * 40,
            "current_head": "3" * 40,
            "current_tree": "4" * 40,
            "root_clean": True,
            "parent_control_manifest": {"sha256": "e" * 64},
            "files": [],
        }
        self.pypto_identity = {
            "head": contract.PYPTO_HEAD,
            "tree": contract.PYPTO_TREE,
            "clean": True,
        }
        self.gpu = {
            "name": contract.EXPECTED_DEVICE_NAME,
            "compute_capability": "12.0",
            "memory_mib": "24463",
            "used_mib": "1024",
            "driver": contract.EXPECTED_DRIVER_RELEASE,
        }
        self.protected_process = {
            "pid": 10,
            "ppid": 1,
            "start_ticks": 100,
            "rss_kib": 1,
            "command": "gem5.opt",
            "cwd": "/home/zhaosiying/zcode-lane",
        }
        self.static_identity = {
            "source": "static ENVIRONMENT.lock and selected-prefix file audit",
            "environment_lock_sha256": contract.ENVIRONMENT_LOCK_SHA256,
            "version": contract.EXPECTED_TORCH_VERSION,
            "git_version": contract.EXPECTED_TORCH_GIT,
            "cuda": contract.EXPECTED_TORCH_CUDA,
            "hip": None,
            "python_executable": str(
                (self.workspace / contract.PYTHON_REAL_RELATIVE_PATH).resolve()
            ),
            "libcudart_path": str(
                (self.workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve()
            ),
            "libcudart_size": contract.CUDA_RUNTIME_SIZE,
            "libcudart_sha256": contract.CUDA_RUNTIME_SHA256,
            "libcudart_record_owned": True,
            "nvidia_runtime_mappings": [],
            "cuda_initialized": False,
            "forbidden_dsos": [],
        }

    def write_json(self, path: pathlib.Path, value: object, mode: int = 0o600) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = finalizer.canonical_json(value)
        path.write_bytes(raw)
        path.chmod(mode)
        return finalizer.sha256_bytes(raw)

    def load_json(self, path: pathlib.Path) -> dict[str, object]:
        value = json.loads(path.read_text())
        assert isinstance(value, dict)
        return value

    def audit(self, owned: list[int]) -> dict[str, object]:
        return {
            "owned_nvidia_compute_pids": owned,
            "external_nvidia_compute_pids": [],
            "protected_nvidia_compute_pids": [],
            "protected_nvidia_runtime_mapping_pids": [],
            "unreadable_protected_maps": [],
            "protected_heavy_pids": [10],
            "protected_cpu_lane_authorized": True,
            "free_memory_mib": 20_000,
            "gpu": self.gpu,
        }

    def reset_run(self) -> None:
        self.run_dir = self.workspace / "runs" / self.run_id
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.run_dir.mkdir()
        self.replay = contract.replay_directory(self.workspace, self.run_id)
        self.replay.mkdir(mode=0o700)
        self.process_path = self.run_dir / "process.json"
        self.preflight_path = self.run_dir / "preflight.json"
        self.gate_path = self.run_dir / "gpu-smoke-gate.json"
        self.barrier_path = self.run_dir / "gpu-smoke-start-barrier.json"
        self.provisional_path = contract.provisional_path(self.workspace, self.run_id)

    def build_fixture(self) -> tuple[dict[str, object], str, dict[str, object]]:
        self.reset_run()
        protected = [self.protected_process]
        memory_floor = 24 * 1024 * 1024
        preflight = {
            "coexistence_policy_version": 1,
            "cwd": str(self.workspace),
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
            "protected_gpu_smoke_waiver_applied": True,
            "protected_heavy_processes": protected,
            "protected_nvidia_compute_pids": [],
            "protected_nvidia_runtime_mapping_pids": [],
            "protected_processes": protected,
            "protected_zero_nvidia_gpu_smoke_requested": True,
            "torch": self.static_identity,
            "unreadable_protected_maps": [],
            "workspace": str(self.workspace),
            "workspace_processes": [],
        }
        preflight_sha = self.write_json(self.preflight_path, preflight)
        preflight_anchor = {"path": str(self.preflight_path), "sha256": preflight_sha}
        pre_release = self.audit([])
        gate = {
            "schema": 1,
            "run_id": self.run_id,
            "pid": 99,
            "pgid": 99,
            "start_ticks": 990,
            "command": contract.fixed_child_command(self.workspace),
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
            "workspace": str(self.workspace),
            "environment": str(self.workspace / "envs/pypto-nvidia"),
            "environment_access_lock": {
                "path": str(self.workspace / "runs/environment-pypto-nvidia.lock"),
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
                "requested": True,
                "waiver_applied": True,
                "authorization": contract.GPU_SMOKE_AUTHORIZATION,
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
                "timeout_seconds": contract.GPU_SMOKE_TIMEOUT_SECONDS,
                "minimum_free_disk_bytes": contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB
                << 30,
                "owned_run_pause_memory_kib": 16 * 1024 * 1024,
            },
            "command": contract.fixed_child_command(self.workspace),
            "pid": 99,
            "pgid": 99,
            "start_ticks": 990,
            "started_at": "20990101T000000Z",
            "status": "exited",
            "gpu_smoke_pre_release_audit": pre_release,
            "gpu_smoke_last_audit": self.audit([99]),
            "gpu_smoke_post_exit_audit": self.audit([]),
            "return_code": 0,
            "finished_at": "20990101T000100Z",
        }
        self.write_json(self.process_path, process)

        replay_files = []
        names = ["compile-request.msgpack"]
        for case in contract.CASE_SPECS:
            names.extend(
                [
                    f"{case.name}.hir.msgpack",
                    f"{case.name}.build-spec.msgpack",
                    f"{case.name}.artifact.msgpack",
                ]
            )
        for index, name in enumerate(names):
            path = self.replay / name
            payload = f"replay-{index}".encode()
            path.write_bytes(payload)
            path.chmod(0o444)
            replay_files.append(
                {
                    "path": path.relative_to(self.workspace).as_posix(),
                    "bytes": len(payload),
                    "sha256": finalizer.sha256_bytes(payload),
                }
            )

        integrity_paths = {
            "contract": self.workspace / self.relative_files["contract"],
            "runner": self.workspace / contract.RUNNER_RELATIVE_PATH,
            "controller": self.workspace / self.relative_files["controller"],
            "environment_lock": self.workspace / "ENVIRONMENT.lock",
            "versions_lock": self.workspace / "VERSIONS.lock",
            "workspace_lock": self.workspace / "WORKSPACE.lock",
            "pypto_dso": self.workspace / contract.PYPTO_DSO_RELATIVE_PATH,
            "cuda_runtime": self.workspace / contract.CUDA_RUNTIME_RELATIVE_PATH,
        }
        integrity = {
            name: {
                "path": path.relative_to(self.workspace).as_posix(),
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
            "shared_memory_per_multiprocessor_bytes": 102400,
            "registers_per_cta": 65536,
            "max_registers_per_thread": 255,
            "registers_per_multiprocessor": 65536,
            "l2_cache_size_bytes": 64 * 1024 * 1024,
            "total_global_memory_bytes": 25650855936,
        }
        partial = synthetic_provisional()
        runtime = partial["runtime"]
        assert isinstance(runtime, dict)
        compile_request = {
            "byte_identity_digest": "1" * 64,
            "loader_compatibility_input_digest": "2" * 64,
            "device_autotune_identity_digest": "3" * 64,
        }
        child_gate = {
            "static_identity": self.static_identity,
            "gpu": self.gpu,
            "free_memory_mib": 20_000,
            "protected_heavy_pids": [10],
            "protected_runtime_pids": [],
            "unreadable_protected_maps": [],
            "nvidia_compute_pids": [],
            "control_manifest": self.control_identity,
        }
        runtime.update(
            {
                "torch": {
                    "version": contract.EXPECTED_TORCH_VERSION,
                    "git_version": contract.EXPECTED_TORCH_GIT,
                    "cuda": contract.EXPECTED_TORCH_CUDA,
                    "hip": None,
                    "module_path": str(
                        (self.workspace / self.relative_files["torch"]).resolve()
                    ),
                },
                "child_pre_cuda_gate": child_gate,
                "libcudart_paths": [
                    str(
                        (self.workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve()
                    )
                ],
                "observation": {
                    "device_ordinal": 0,
                    "device_name": contract.EXPECTED_DEVICE_NAME,
                    "device_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                    "pci_device_id": "0000:02:00.0",
                    "traits": traits,
                    "cuda_toolkit_version": contract.EXPECTED_CUDA_TOOLKIT_VERSION,
                    "cuda_driver_version": contract.EXPECTED_DRIVER_RELEASE,
                    "tensor_ir_revision": contract.TENSOR_IR_HEAD,
                    "cuda_tile_revision": contract.CUDA_TILE_HEAD,
                    "supported_compute_dtypes": list(
                        contract.EXPECTED_SUPPORTED_COMPUTE_DTYPES
                    ),
                    "cuda_driver_release_provenance": contract.EXPECTED_DRIVER_RELEASE,
                    "cuda_driver_api_version": 13030,
                    "cuda_runtime_api_version": 13000,
                    "cuda_runtime_library_path": str(
                        (self.workspace / contract.CUDA_RUNTIME_RELATIVE_PATH).resolve()
                    ),
                    "context_address": 123,
                    "context_id": 456,
                },
                "compile_request": compile_request,
            }
        )
        provisional = {
            "schema_version": contract.SMOKE_SCHEMA_VERSION,
            "smoke": contract.SMOKE_NAME,
            "acceptance": "gpu-execution-complete-awaiting-run-finalization",
            "scope": partial["scope"],
            "inputs": {
                "integrity": integrity,
                "pypto": self.pypto_identity,
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
                    "path": self.preflight_path.relative_to(self.workspace).as_posix(),
                    "sha256": preflight_sha,
                },
                "gate": {
                    "path": str(self.gate_path),
                    "sha256": gate_sha,
                    "document": gate,
                },
                "start_barrier_sha256": barrier_sha,
                "protected_zero_nvidia_policy": True,
            },
            "runtime": runtime,
        }
        provisional_sha = self.write_json(
            self.provisional_path, provisional, mode=0o444
        )
        return provisional, provisional_sha, self.semantic_audit_document(provisional)

    def semantic_audit_document(
        self, provisional: dict[str, object]
    ) -> dict[str, object]:
        runtime = provisional["runtime"]
        assert isinstance(runtime, dict)
        observation = runtime["observation"]
        assert isinstance(observation, dict)
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
        artifact_fields = (
            "case",
            "build_spec_identity_digest",
            "source_ir_digest",
            "callable_abi_digest",
            "static_specialization_digest",
            "symbolic_specialization_digest",
            "argument_abi_digest",
            "result_abi_digest",
            "mutation_abi_digest",
            "artifact_identity_digest",
            "cache_key_digest",
            "loader_compatibility_digest",
            "device_code_bytes",
            "device_code_sha256",
            "kernel_abi_identity_digest",
            "entry_function_name",
            "fallback_used",
        )
        return {
            "compile_request": runtime["compile_request"],
            "target_info": {name: observation[name] for name in target_fields},
            "hir_programs": [
                {
                    "case": record["case"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                    "canonical_reserialization_equal": True,
                    "parameter_directions": ["In", "In"],
                }
                for record in runtime["hir_programs"]
            ],
            "artifacts": [
                {name: record[name] for name in artifact_fields}
                for record in runtime["artifacts"]
            ],
        }

    @contextlib.contextmanager
    def finalize_patches(
        self,
        *,
        semantic_audit: dict[str, object],
        control_identity: dict[str, object] | None = None,
    ):
        with (
            mock.patch.object(finalizer, "require_no_site_finalizer"),
            mock.patch.object(
                finalizer.control_manifest,
                "validate_control_manifest",
                return_value=(control_identity or self.control_identity),
            ),
            mock.patch.object(
                finalizer,
                "audit_replay_semantics",
                return_value=semantic_audit,
            ),
            mock.patch.object(
                finalizer, "git_identity", return_value=self.pypto_identity
            ),
        ):
            yield

    def call_finalize(
        self,
        provisional_sha: str,
        semantic_audit: dict[str, object],
        *,
        control_identity: dict[str, object] | None = None,
    ):
        with self.finalize_patches(
            semantic_audit=semantic_audit, control_identity=control_identity
        ):
            return finalizer.finalize(
                workspace=self.workspace,
                run_id=self.run_id,
                expected_provisional_sha256=provisional_sha,
            )

    def mutate_json(
        self, path: pathlib.Path, keys: tuple[str, ...], value: object
    ) -> str:
        document = self.load_json(path)
        target: dict[str, object] = document
        for key in keys[:-1]:
            child = target[key]
            assert isinstance(child, dict)
            target = child
        target[keys[-1]] = value
        path.chmod(0o600)
        return self.write_json(
            path, document, mode=0o444 if path == self.provisional_path else 0o600
        )

    def test_complete_synthetic_finalization_and_no_replace(self) -> None:
        provisional, provisional_sha, semantic_audit = self.build_fixture()
        report, output, digest = self.call_finalize(provisional_sha, semantic_audit)
        self.assertEqual(
            report["status"],
            "accepted-real-sm120-frontend-vector-add-correctness-smoke",
        )
        self.assertEqual(report["result"], provisional["runtime"])
        self.assertIs(
            report["finalizer"]["source_audit_compiler_entrypoints_absent"], True
        )
        self.assertIs(report["finalizer"]["torch_cuda_initialized"], False)
        self.assertEqual(digest, finalizer.sha256_file(output))
        self.assertEqual(output.stat().st_mode & 0o777, 0o444)
        with self.assertRaisesRegex(finalizer.FinalizeError, "already exists"):
            self.call_finalize(provisional_sha, semantic_audit)

    def test_replay_subprocess_success_and_failure(self) -> None:
        provisional, _sha, _semantic = self.build_fixture()
        audited = self.semantic_audit_document(provisional)
        completed = types.SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(audited, sort_keys=True, separators=(",", ":")) + "\n",
        )
        with mock.patch.object(finalizer.subprocess, "run", return_value=completed):
            result = finalizer.audit_replay_semantics(
                provisional, self.workspace, self.run_id
            )
        self.assertEqual(result["artifacts"], audited["artifacts"])
        failed = types.SimpleNamespace(
            returncode=1, stderr="fixture failure", stdout=""
        )
        with (
            mock.patch.object(finalizer.subprocess, "run", return_value=failed),
            self.assertRaisesRegex(finalizer.FinalizeError, "replay audit failed"),
        ):
            finalizer.audit_replay_semantics(provisional, self.workspace, self.run_id)

    def test_exact_nested_schemas_and_recursive_claim_scan(self) -> None:
        provisional, _sha, _semantic = self.build_fixture()
        runtime = provisional["runtime"]
        child = runtime["child_pre_cuda_gate"]
        traits = runtime["observation"]["traits"]
        finalizer.validate_child_gate_schema(child)
        finalizer.validate_target_traits_schema(traits)
        for value, validator, description in (
            (child, finalizer.validate_child_gate_schema, "child pre-CUDA gate"),
            (traits, finalizer.validate_target_traits_schema, "target traits"),
        ):
            candidate = copy.deepcopy(value)
            candidate["model_result"] = True
            with (
                self.subTest(description=description),
                self.assertRaisesRegex(finalizer.FinalizeError, "key set differs"),
            ):
                validator(candidate)
        candidate = copy.deepcopy(provisional)
        candidate["runtime"]["child_pre_cuda_gate"]["static_identity"][
            "nested_latency_ns"
        ] = 7
        with self.assertRaisesRegex(finalizer.FinalizeError, "claim field"):
            finalizer.validate_scope(candidate)

    def test_table_driven_transaction_join_mutations(self) -> None:
        mutation_names = (
            "process",
            "preflight",
            "gate",
            "barrier",
            "audit",
            "control",
            "integrity",
            "replay",
        )
        for name in mutation_names:
            with self.subTest(name=name):
                provisional, provisional_sha, semantic_audit = self.build_fixture()
                live_control = self.control_identity
                if name == "process":
                    self.mutate_json(self.process_path, ("status",), "running")
                elif name == "preflight":
                    self.mutate_json(self.preflight_path, ("ok",), False)
                elif name == "gate":
                    self.mutate_json(self.gate_path, ("pid",), 100)
                elif name == "barrier":
                    self.mutate_json(self.barrier_path, ("pid",), 100)
                elif name == "audit":
                    self.mutate_json(
                        self.process_path,
                        (
                            "gpu_smoke_post_exit_audit",
                            "external_nvidia_compute_pids",
                        ),
                        [777],
                    )
                elif name == "control":
                    live_control = copy.deepcopy(self.control_identity)
                    live_control["manifest_sha256"] = "0" * 64
                elif name == "integrity":
                    provisional_sha = self.mutate_json(
                        self.provisional_path,
                        ("inputs", "integrity", "runner", "sha256"),
                        "0" * 64,
                    )
                elif name == "replay":
                    path = self.replay / "compile-request.msgpack"
                    path.chmod(0o600)
                    path.write_bytes(b"tampered")
                    path.chmod(0o444)
                with self.assertRaises(finalizer.FinalizeError):
                    self.call_finalize(
                        provisional_sha,
                        semantic_audit,
                        control_identity=live_control,
                    )

    def test_replay_directory_rejects_extra_entry(self) -> None:
        provisional, _sha, _semantic = self.build_fixture()
        extra = self.replay / "unreviewed.bin"
        extra.write_bytes(b"extra")
        extra.chmod(0o444)
        with self.assertRaisesRegex(finalizer.FinalizeError, "missing or extra"):
            finalizer.validate_replay(provisional, self.workspace, self.run_id)


if __name__ == "__main__":
    unittest.main()
