from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import json
import os
import pathlib
import py_compile
import shutil
import secrets
import stat
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool(name: str, path: pathlib.Path):
    resolved = path.resolve(strict=True)
    if resolved != path or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"exact test source is noncanonical: {path}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__["__exact_source_bytes__"] = len(raw)
    module.__dict__["__exact_source_sha256__"] = hashlib.sha256(raw).hexdigest()
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


control_manifest = load_tool(
    "_pypto_fused_pointwise_sm120_control_manifest",
    ROOT / "tools/_pypto_fused_pointwise_sm120_control_manifest.py",
)
control_manifest.reject_control_bytecode_cache(ROOT)
contract = load_tool(
    "_pypto_fused_pointwise_sm120_contract",
    ROOT / "tools/_pypto_fused_pointwise_sm120_contract.py",
)
runner = load_tool(
    "fused_pointwise_sm120_test_runner",
    ROOT / contract.RUNNER_RELATIVE_PATH,
)
controller = load_tool(
    "fused_pointwise_sm120_test_controller",
    ROOT / "tools/run_pypto_fused_pointwise_sm120_isolated.py",
)
finalizer = load_tool(
    "fused_pointwise_sm120_test_finalizer",
    ROOT / "tools/finalize_pypto_fused_pointwise_sm120.py",
)
# The finalizer deliberately source-loads fresh exact control modules.  Use
# those same objects for patch-based synthetic transactions.
contract = finalizer.contract
control_manifest = finalizer.control_manifest


def function_node(path: pathlib.Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
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


def pack_words(words: list[int], dtype: str) -> bytes:
    import struct

    code = "I" if dtype == "float32" else "H"
    return struct.pack(f"<{len(words)}{code}", *words)


def expected_replay_names() -> list[str]:
    names = ["compile-request.msgpack"]
    for case in contract.CASE_SPECS:
        names.extend(
            [
                f"{case.name}.hir.msgpack",
                f"{case.name}.source.mlir",
                f"{case.name}.build-spec.msgpack",
                f"{case.name}.artifact.msgpack",
                f"{case.name}.cubin",
            ]
        )
    for case in contract.CASE_SPECS:
        for repetition in range(case.repetitions):
            names.extend(
                f"{case.name}.r{repetition}.input{ordinal}.bin"
                for ordinal in range(case.input_count)
            )
            names.extend(
                [
                    f"{case.name}.r{repetition}.reference.bin",
                    f"{case.name}.r{repetition}.actual.bin",
                ]
            )
    return names


class ContractAndAnchorTest(unittest.TestCase):
    def test_exact_source_loader_ignores_a_valid_alternate_pyc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = (root / "cached_module.py").resolve()
            cache = root / "__pycache__/cached_module.cpython-314.pyc"
            cache.parent.mkdir()
            original = b"VALUE = 'source'\n"
            alternate = b"VALUE = 'cached'\n"
            self.assertEqual(len(original), len(alternate))
            source.write_bytes(alternate)
            source.touch()
            timestamp = source.stat().st_mtime
            py_compile.compile(str(source), cfile=str(cache), doraise=True)
            source.write_bytes(original)
            os.utime(source, (timestamp, timestamp))
            loaded = runner._load_module("fused_exact_source_fixture", source)
            self.assertEqual(loaded.VALUE, "source")
            self.assertEqual(
                loaded.__exact_source_sha256__, hashlib.sha256(original).hexdigest()
            )

    def test_validator_rejects_valid_ignored_pyc_with_clean_tracked_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "tools/_pypto_fused_pointwise_sm120_control_manifest.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n")
            (root / ".gitignore").write_text("__pycache__/\n*.pyc\n")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Cache Fixture"], cwd=root, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "cache@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "source"], cwd=root, check=True
            )
            py_compile.compile(str(source), doraise=True)
            status = subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
            self.assertEqual(status, "")
            entries = control_manifest.control_bytecode_cache_entries(root)
            self.assertEqual(
                entries,
                [
                    "tools/__pycache__/"
                    "_pypto_fused_pointwise_sm120_control_manifest.cpython-314.pyc"
                ],
            )
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "bytecode/cache"
            ):
                control_manifest.reject_control_bytecode_cache(root)

    def test_nine_case_matrix_is_exact(self) -> None:
        actual = [
            (
                case.name,
                case.dtype,
                case.shape,
                case.strides,
                case.tile_sizes,
                case.expected_grid,
                case.input_count,
                case.assignment_count,
                case.expected_kernel_arguments,
            )
            for case in contract.CASE_SPECS
        ]
        self.assertEqual(
            actual,
            [
                (
                    "arith_fp32_tail",
                    "float32",
                    (3, 5),
                    (5, 1),
                    (8,),
                    (2, 1, 1),
                    4,
                    8,
                    5,
                ),
                (
                    "arith_bf16_rank3_tail",
                    "bfloat16",
                    (2, 3, 5),
                    (15, 5, 1),
                    (16,),
                    (2, 1, 1),
                    4,
                    8,
                    5,
                ),
                ("exp_fp32_tail1", "float32", (17,), (1,), (16,), (2, 1, 1), 1, 1, 2),
                (
                    "exp_bf16_exact_tile",
                    "bfloat16",
                    (8, 8),
                    (8, 1),
                    (16,),
                    (4, 1, 1),
                    1,
                    1,
                    2,
                ),
                (
                    "recip_fp32_tail",
                    "float32",
                    (3, 5),
                    (5, 1),
                    (8,),
                    (2, 1, 1),
                    1,
                    1,
                    2,
                ),
                (
                    "recip_bf16_tail1",
                    "bfloat16",
                    (17,),
                    (1,),
                    (16,),
                    (2, 1, 1),
                    1,
                    1,
                    2,
                ),
                (
                    "rsqrt_fp32_rank3_tail",
                    "float32",
                    (2, 3, 5),
                    (15, 5, 1),
                    (16,),
                    (2, 1, 1),
                    1,
                    1,
                    2,
                ),
                (
                    "rsqrt_bf16_exact_tile",
                    "bfloat16",
                    (8, 8),
                    (8, 1),
                    (16,),
                    (4, 1, 1),
                    1,
                    1,
                    2,
                ),
                (
                    "max16x64_fp32_tail1",
                    "float32",
                    (17,),
                    (1,),
                    (16,),
                    (2, 1, 1),
                    16,
                    64,
                    17,
                ),
            ],
        )
        self.assertEqual(
            contract.CASE_SPECS[0].scalar_literals, (8_388_608.0, 0.3, 0.3)
        )
        self.assertEqual(contract.CASE_SPECS[1].scalar_literals, (128.0, 0.3, 0.3))
        self.assertEqual(
            contract.CASE_SPECS[-1].operator_sequence.count("tensor.add"), 15
        )
        self.assertEqual(
            contract.CASE_SPECS[-1].operator_sequence.count("tensor.neg"), 49
        )

    def test_numerical_policy_is_frozen(self) -> None:
        by_name = {case.name: case for case in contract.CASE_SPECS}
        for name in ("arith_fp32_tail", "arith_bf16_rank3_tail", "max16x64_fp32_tail1"):
            self.assertEqual(by_name[name].comparison, "exact")
        for name in ("recip_fp32_tail", "recip_bf16_tail1"):
            self.assertEqual(
                by_name[name].comparison, "exact-with-special-classification"
            )
        for name in ("exp_fp32_tail1", "rsqrt_fp32_rank3_tail"):
            self.assertEqual(
                (by_name[name].max_ulp, by_name[name].rtol, by_name[name].atol),
                (4, 2e-6, 0.0),
            )
        for name in ("exp_bf16_exact_tile", "rsqrt_bf16_exact_tile"):
            self.assertEqual(
                (by_name[name].max_ulp, by_name[name].rtol, by_name[name].atol),
                (1, 1 / 128, 0.0),
            )
        self.assertFalse(contract.HIGH_PRECISION_ALLOWED)
        self.assertTrue(contract.NO_SUBNORMAL_INPUTS)
        self.assertEqual(contract.GUARD_ELEMENTS, 16)

    def test_exact_product_and_anchor_manifest(self) -> None:
        self.assertEqual(
            contract.PYPTO_HEAD, "b83fcd3ddc497d585bcc45883eede179aff7d4d2"
        )
        self.assertEqual(
            contract.PYPTO_TREE, "49eda98f3ed8d72bfd14d5a5900cdc0e71ca699d"
        )
        dso = ROOT / contract.PYPTO_DSO_RELATIVE_PATH
        self.assertEqual(dso.stat().st_size, contract.PYPTO_DSO_SIZE)
        self.assertEqual(runner.sha256_file(dso), contract.PYPTO_DSO_SHA256)
        generator = ROOT / contract.ANCHOR_GENERATOR_RELATIVE_PATH
        self.assertEqual(generator.stat().st_size, contract.ANCHOR_GENERATOR_SIZE)
        self.assertEqual(
            runner.sha256_file(generator), contract.ANCHOR_GENERATOR_SHA256
        )
        path = ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH
        raw = path.read_bytes()
        anchors = json.loads(raw)
        self.assertEqual(raw, control_manifest.canonical_json(anchors))
        self.assertEqual(len(raw), contract.COMPILE_ANCHORS_SIZE)
        self.assertEqual(runner.sha256_bytes(raw), contract.COMPILE_ANCHORS_SHA256)
        self.assertEqual(
            anchors["records_sha256"],
            "01e8f99dfb0a1aa0e5788177b7d41cc6ed62983037aa57ed8628bb0a1594b844",
        )
        self.assertEqual(
            [record["run_id"] for record in anchors["anchor_runs"]],
            [
                "pypto-20260826T042728Z-1382280-ce1fa0",
                "pypto-20260826T042750Z-1382496-07c3e7",
            ],
        )
        identity = control_manifest.validate_compile_anchors(ROOT)
        self.assertEqual(identity["sha256"], contract.COMPILE_ANCHORS_SHA256)

    def test_all_anchor_records_join_contract(self) -> None:
        anchors = json.loads(
            (ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH).read_text()
        )
        self.assertEqual(
            [record["case"] for record in anchors["records"]],
            list(contract.CASE_ORDER),
        )
        for record, case in zip(anchors["records"], contract.CASE_SPECS, strict=True):
            self.assertEqual(record["hir_bytes"], case.expected_hir_bytes)
            self.assertEqual(record["hir_sha256"], case.expected_hir_sha256)
            self.assertEqual(record["source_ir_bytes"], case.expected_source_ir_bytes)
            self.assertEqual(record["source_ir_digest"], case.expected_source_ir_digest)
            self.assertEqual(
                record["static_specialization_digest"],
                case.expected_static_specialization_digest,
            )
            self.assertEqual(
                record["symbolic_specialization_digest"],
                case.expected_symbolic_specialization_digest,
            )
            self.assertEqual(
                record["argument_abi_digest"], case.expected_argument_abi_digest
            )
            self.assertEqual(
                record["result_abi_digest"], case.expected_result_abi_digest
            )
            self.assertEqual(
                record["mutation_abi_digest"], case.expected_mutation_abi_digest
            )
            self.assertEqual(
                record["callable_abi_digest"], case.expected_callable_abi_digest
            )
            self.assertEqual(
                record["build_spec_identity_digest"],
                case.expected_build_spec_identity_digest,
            )
            self.assertEqual(
                record["artifact_identity_digest"],
                case.expected_artifact_identity_digest,
            )
            self.assertEqual(
                record["device_code_bytes"], case.expected_device_code_bytes
            )
            self.assertEqual(
                record["device_code_sha256"], case.expected_device_code_sha256
            )
            self.assertEqual(record["expected_grid"], list(case.expected_grid))
            self.assertEqual(
                record["kernel_argument_count"], case.expected_kernel_arguments
            )
            self.assertEqual(record["operator_sequence"], list(case.operator_sequence))
            self.assertEqual(record["entry_function_name"], "pypto_fused_pointwise_v2")
            self.assertTrue(record["pointer_only"])
            self.assertEqual(record["workspace_bytes"], 0)

    def test_compile_anchor_sidecars_fail_closed_on_tamper_and_mode(self) -> None:
        source_manifest = ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH
        document = json.loads(source_manifest.read_text())
        with tempfile.TemporaryDirectory() as directory:
            fixture = pathlib.Path(directory)
            manifest = fixture / contract.COMPILE_ANCHORS_RELATIVE_PATH
            manifest.parent.mkdir(parents=True)
            shutil.copy2(source_manifest, manifest)
            for run in document["anchor_runs"]:
                for name in ("preflight", "process", "record"):
                    source = ROOT / run[name]["path"]
                    target = fixture / run[name]["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                run_record = json.loads((ROOT / run["record"]["path"]).read_text())
                for replay_record in run_record["replay_files"]:
                    source = ROOT / replay_record["path"]
                    target = fixture / replay_record["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            control_manifest.validate_compile_anchors(fixture)
            manifest_original = manifest.read_bytes()

            def validate_rebound(candidate: dict[str, object]) -> None:
                raw = control_manifest.canonical_json(candidate)
                manifest.write_bytes(raw)
                with (
                    mock.patch.object(
                        control_manifest, "COMPILE_ANCHORS_SIZE", len(raw)
                    ),
                    mock.patch.object(
                        control_manifest,
                        "COMPILE_ANCHORS_SHA256",
                        hashlib.sha256(raw).hexdigest(),
                    ),
                ):
                    control_manifest.validate_compile_anchors(fixture)

            digest_mutation = copy.deepcopy(document)
            digest_mutation["records"][0]["source_ir_bytes"] += 1
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "record digest"
            ):
                validate_rebound(digest_mutation)
            manifest.write_bytes(manifest_original)

            process_record = document["anchor_runs"][0]["process"]
            process_path = fixture / process_record["path"]
            process_original = process_path.read_bytes()
            for field, value in (
                ("run_id", "pypto-20990101T000000Z-1-abcdef"),
                ("preflight", {"path": "/wrong", "sha256": "0" * 64}),
            ):
                process_document = json.loads(process_original)
                process_document[field] = value
                process_raw = control_manifest.canonical_json(process_document)
                process_path.write_bytes(process_raw)
                process_path.chmod(0o600)
                rebound = copy.deepcopy(document)
                rebound["anchor_runs"][0]["process"]["bytes"] = len(process_raw)
                rebound["anchor_runs"][0]["process"]["sha256"] = hashlib.sha256(
                    process_raw
                ).hexdigest()
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(
                        control_manifest.ControlManifestError,
                        "isolation evidence",
                    ),
                ):
                    validate_rebound(rebound)
                process_path.write_bytes(process_original)
                process_path.chmod(0o600)
                manifest.write_bytes(manifest_original)

            replay_record = json.loads(
                (fixture / document["anchor_runs"][0]["record"]["path"]).read_text()
            )["replay_files"][0]
            replay_path = fixture / replay_record["path"]
            replay_original = replay_path.read_bytes()
            replay_path.chmod(0o644)
            replay_path.write_bytes(replay_original + b"tamper")
            with self.assertRaisesRegex(
                control_manifest.ControlManifestError, "replay bytes"
            ):
                control_manifest.validate_compile_anchors(fixture)
            replay_path.write_bytes(replay_original)
            replay_path.chmod(0o444)

            record = fixture / document["anchor_runs"][0]["record"]["path"]
            original = record.read_bytes()
            record.chmod(0o644)
            record.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(control_manifest.ControlManifestError):
                control_manifest.validate_compile_anchors(fixture)
            record.write_bytes(original)
            record.chmod(0o600)
            with self.assertRaises(control_manifest.ControlManifestError):
                control_manifest.validate_compile_anchors(fixture)

    def test_accepted_v4_cp43_and_cp44_are_bound_unchanged(self) -> None:
        parent = control_manifest.validate_parent_control_manifest(ROOT)
        legacy, report = control_manifest.validate_legacy_controls(ROOT, parent)
        self.assertEqual(parent["sha256"], control_manifest.PARENT_MANIFEST_SHA256)
        self.assertEqual(legacy["sha256"], control_manifest.LEGACY_MANIFEST_SHA256)
        self.assertEqual(report["sha256"], control_manifest.LEGACY_REPORT_SHA256)
        self.assertEqual(report["mode"], 0o444)
        self.assertEqual(
            [record["path"] for record in legacy["files"][-3:]],
            list(control_manifest.CONTROL_PATHS[-3:]),
        )

    def test_control_manifest_transition_is_fail_closed_or_fully_valid(self) -> None:
        manifest = ROOT / control_manifest.MANIFEST_RELATIVE_PATH
        dirty = bool(
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
        elif dirty:
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

    def test_runner_import_is_pre_barrier_pure(self) -> None:
        before = set(sys.modules)
        load_tool(
            "fused_pointwise_sm120_runner_second_import",
            ROOT / contract.RUNNER_RELATIVE_PATH,
        )
        added = set(sys.modules) - before
        self.assertTrue(
            {"torch", "pypto", "triton", "sglang", "flashinfer"}.isdisjoint(added)
        )

    def test_fixed_child_and_control_paths(self) -> None:
        command = contract.fixed_child_command(ROOT)
        self.assertEqual(command[1:4], ["-I", "-B", "-S"])
        self.assertEqual(command[-1], str(ROOT / contract.RUNNER_RELATIVE_PATH))
        self.assertIn(
            contract.COMPILE_ANCHORS_RELATIVE_PATH.as_posix(),
            control_manifest.CONTROL_PATHS,
        )
        self.assertIn(
            contract.ANCHOR_GENERATOR_RELATIVE_PATH.as_posix(),
            control_manifest.CONTROL_PATHS,
        )
        self.assertEqual(
            control_manifest.CONTROL_PATHS[-3:],
            ("tools/preflight.py", "tools/run_isolated.py", "tools/stop_run.py"),
        )
        runner_path = ROOT / contract.RUNNER_RELATIVE_PATH
        self.assertEqual(runner_path.stat().st_size, contract.RUNNER_SIZE)
        self.assertEqual(runner.sha256_file(runner_path), contract.RUNNER_SHA256)


class ManifestValidatorSyntheticTest(unittest.TestCase):
    def test_commit_tree_order_and_tamper_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Fused Gate Fixture"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "fixture@example.invalid"],
                cwd=root,
                check=True,
            )
            records = []
            for index, relative in enumerate(control_manifest.CONTROL_PATHS):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"control-{index}-{relative}\n".encode())
                path.chmod(
                    0o755 if relative in control_manifest.CONTROL_PATHS[-3:] else 0o644
                )
                records.append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": control_manifest.sha256_file(path),
                        "mode": stat.S_IMODE(path.stat().st_mode),
                    }
                )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "implementation"], cwd=root, check=True
            )
            implementation = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            tree = subprocess.run(
                ["git", "rev-parse", "HEAD^{tree}"],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            parent = {"primitive_files": records[-3:]}
            manifest = {
                "schema_version": 1,
                "kind": control_manifest.MANIFEST_KIND,
                "implementation_commit": implementation,
                "implementation_tree": tree,
                "files": records,
            }
            manifest_path = root / control_manifest.MANIFEST_RELATIVE_PATH
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_bytes(control_manifest.canonical_json(manifest))
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "manifest"], cwd=root, check=True
            )
            patches = (
                mock.patch.object(
                    control_manifest,
                    "validate_parent_control_manifest",
                    return_value=parent,
                ),
                mock.patch.object(
                    control_manifest,
                    "validate_legacy_controls",
                    return_value=({"sha256": "1" * 64}, {"sha256": "2" * 64}),
                ),
                mock.patch.object(
                    control_manifest,
                    "validate_compile_anchors",
                    return_value={"sha256": "3" * 64},
                ),
            )
            with patches[0], patches[1], patches[2]:
                identity = control_manifest.validate_control_manifest(root)
                self.assertEqual(identity["implementation_commit"], implementation)
                bad = copy.deepcopy(manifest)
                bad["files"] = list(reversed(bad["files"]))
                manifest_path.write_bytes(control_manifest.canonical_json(bad))
                subprocess.run(["git", "add", str(manifest_path)], cwd=root, check=True)
                subprocess.run(
                    ["git", "commit", "-q", "-m", "bad order"], cwd=root, check=True
                )
                with self.assertRaisesRegex(
                    control_manifest.ControlManifestError, "order differs"
                ):
                    control_manifest.validate_control_manifest(root)
                manifest_path.write_bytes(control_manifest.canonical_json(manifest))
                subprocess.run(["git", "add", str(manifest_path)], cwd=root, check=True)
                subprocess.run(
                    ["git", "commit", "-q", "-m", "restore order"], cwd=root, check=True
                )
                (root / control_manifest.CONTROL_PATHS[0]).write_text("tampered\n")
                with self.assertRaisesRegex(
                    control_manifest.ControlManifestError, "not clean"
                ):
                    control_manifest.validate_control_manifest(root)


class RunnerStructureAndNumericsTest(unittest.TestCase):
    def test_tracked_generator_is_cuda_hidden_single_compile_and_no_replace(
        self,
    ) -> None:
        path = ROOT / contract.ANCHOR_GENERATOR_RELATIVE_PATH
        node = function_node(path, "run")
        self.assertEqual(
            len(called_attribute_lines(node, "compile_structured_strict")), 1
        )
        source = path.read_text()
        self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES") != ""', source)
        self.assertIn('os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"', source)
        self.assertGreaterEqual(source.count("torch.cuda.is_initialized()"), 2)
        self.assertIn("runner.publish_no_replace", source)
        self.assertNotIn("NvidiaExecutable", source)

    def test_canonical_sources_are_exact_and_v2_only(self) -> None:
        for case in contract.CASE_SPECS:
            source = runner.canonical_tensor_ir_source(case)
            self.assertEqual(len(source), case.expected_source_ir_bytes)
            self.assertEqual(
                runner.sha256_bytes(source), case.expected_source_ir_digest
            )
            self.assertIn(b"@pypto_fused_pointwise_v2", source)
            self.assertNotIn(b"@pypto_vector_add", source)
            self.assertNotIn(b"high_precision", source)
        arithmetic = runner.canonical_tensor_ir_source(contract.CASE_SPECS[0])
        self.assertIn(b"constant 8388608.0 : f32", arithmetic)
        self.assertIn(b"constant 0.29999999999999999 : f32", arithmetic)
        maximum = runner.canonical_tensor_ir_source(contract.CASE_SPECS[-1])
        self.assertIn(b"%arg15", maximum)
        self.assertIn(b"%value63", maximum)

    def test_hir_roundtrip_precedes_the_single_facade_call(self) -> None:
        node = function_node(ROOT / contract.RUNNER_RELATIVE_PATH, "run_smoke")
        compile_lines = called_attribute_lines(node, "compile_structured_strict")
        self.assertEqual(len(compile_lines), 1)
        self.assertTrue(
            any(
                line < compile_lines[0]
                for line in called_attribute_lines(node, "deserialize")
            )
        )
        source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text()
        self.assertIn("This is the one and only producer invocation", source)
        self.assertIn('restored.get_function("fused_main")', source)
        self.assertIn("[ir.ParamDirection.In] * case.input_count", source)
        self.assertNotIn("high_precision=True", source)

    def test_reference_and_candidate_stream_lifetimes_are_ordered(self) -> None:
        node = function_node(ROOT / contract.RUNNER_RELATIVE_PATH, "execute_case")
        sync_lines = called_attribute_lines(node, "synchronize")
        launch_lines = called_attribute_lines(node, "launch")
        unload_lines = called_attribute_lines(node, "unload")
        delete_lines = [
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, ast.Delete)
            and any(
                isinstance(target, ast.Name) and target.id == "packet"
                for target in child.targets
            )
        ]
        self.assertEqual(
            (len(sync_lines), len(launch_lines), len(delete_lines), len(unload_lines)),
            (2, 1, 1, 1),
        )
        self.assertLess(sync_lines[0], launch_lines[0])
        self.assertLess(launch_lines[0], sync_lines[1])
        self.assertLess(sync_lines[1], delete_lines[0])
        self.assertLess(delete_lines[0], unload_lines[0])
        source = ast.unparse(node)
        for marker in (
            "guarded_inputs",
            "prefix_before_sha256",
            "before_sha256",
            "raw_reference_stream",
        ):
            self.assertIn(marker, source)

    def test_cpu_input_generation_matches_independent_finalizer(self) -> None:
        site = ROOT / "envs/pypto-nvidia/lib/python3.14/site-packages"
        sys.path.insert(0, str(site))
        try:
            import torch

            self.assertFalse(torch.cuda.is_initialized())
            for case in contract.CASE_SPECS:
                for repetition in range(case.repetitions):
                    for ordinal in range(case.input_count):
                        tensor = runner.input_tensor(torch, case, repetition, ordinal)
                        self.assertEqual(
                            runner.logical_tensor_bytes(torch, tensor),
                            pack_words(
                                finalizer._input_words(case, repetition, ordinal),
                                case.dtype,
                            ),
                        )
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            sys.path.remove(str(site))

    def test_cpu_reference_policies_cover_signed_zero_specials_and_boundary(
        self,
    ) -> None:
        for case in contract.CASE_SPECS:
            for repetition in range(case.repetitions):
                inputs = [
                    finalizer._input_words(case, repetition, ordinal)
                    for ordinal in range(case.input_count)
                ]
                result = finalizer._cpu_reference_words(case, inputs)
                self.assertFalse(
                    any(
                        finalizer._classify(word, case.dtype)[0] == "subnormal"
                        for word in result
                    )
                )
                finalizer._compare_words(case, result, result)
                if case.family == "arithmetic":
                    self.assertEqual(
                        result[0], 0x80000000 if case.dtype == "float32" else 0x8000
                    )
        maximum = contract.CASE_SPECS[-1]
        self.assertEqual(
            len(
                finalizer._cpu_reference_words(
                    maximum,
                    [finalizer._input_words(maximum, 0, index) for index in range(16)],
                )
            ),
            17,
        )

    def test_ulp_and_exact_limits_reject_one_step_over(self) -> None:
        exp32 = contract.CASE_SPECS[2]
        reference = [0x3F800000]
        finalizer._compare_words(exp32, [0x3F800004], reference)
        with self.assertRaises(finalizer.FinalizeError):
            finalizer._compare_words(exp32, [0x3F800005], reference)
        exp_bf16 = contract.CASE_SPECS[3]
        finalizer._compare_words(exp_bf16, [0x3F81], [0x3F80])
        with self.assertRaises(finalizer.FinalizeError):
            finalizer._compare_words(exp_bf16, [0x3F82], [0x3F80])
        exact = contract.CASE_SPECS[0]
        with self.assertRaises(finalizer.FinalizeError):
            finalizer._compare_words(exact, [0x3F800001], [0x3F800000])

    def test_candidate_cpu_direct_check_blocks_nontransitive_ulp_acceptance(
        self,
    ) -> None:
        case = contract.CASE_SPECS[2]
        cpu = [0x3F800000]
        torch_reference = [0x3F800004]
        candidate = [0x3F800008]
        finalizer._compare_words(case, torch_reference, cpu)
        finalizer._compare_words(case, candidate, torch_reference)
        with self.assertRaises(finalizer.FinalizeError):
            finalizer._compare_words(case, candidate, cpu)

    def test_tail1_full_tile_overwrite_is_contained_and_detected(self) -> None:
        guard = contract.GUARD_ELEMENTS
        tile = max(case.tile_sizes[0] for case in contract.CASE_SPECS)
        self.assertGreaterEqual(guard, tile - 1)
        logical_count = 17
        backing = [-1] * guard + [0] * logical_count + [1] * guard
        suffix_before = tuple(backing[guard + logical_count :])
        emitted_extent = 2 * tile
        for index in range(emitted_extent):
            backing[guard + index] = 99
        suffix_after = tuple(backing[guard + logical_count :])
        self.assertEqual(emitted_extent - logical_count, 15)
        self.assertEqual(len(backing), logical_count + 2 * guard)
        self.assertNotEqual(suffix_after, suffix_before)
        self.assertEqual(suffix_after[:15], (99,) * 15)
        self.assertEqual(suffix_after[15:], (1,))

    def test_controller_delegates_only_the_fixed_gpu_smoke(self) -> None:
        argv = controller.delegated_argv(
            contract,
            run_id_file=ROOT / "runs/fused-pointwise-next.json",
            allow_protected_zero_nvidia_gpu_smoke=True,
        )
        self.assertIn("--exact-pypto-nvidia-smoke", argv)
        self.assertIn("--allow-protected-zero-nvidia-gpu-smoke", argv)
        self.assertEqual(argv[-5:], contract.fixed_child_command(ROOT))
        self.assertNotIn("gpu-benchmark", argv)


def synthetic_provisional() -> dict[str, object]:
    hir_programs: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    executions: list[dict[str, object]] = []
    for case_index, case in enumerate(contract.CASE_SPECS):
        artifact_identity = case.expected_artifact_identity_digest
        hir_programs.append(
            {
                "case": case.name,
                "bytes": case.expected_hir_bytes,
                "sha256": case.expected_hir_sha256,
                "serialized_once": True,
                "deserialized_before_compile": True,
                "canonical_reserialization_equal": True,
                "structural_equal": True,
                "parameter_directions": ["In"] * case.input_count,
                "input_count": case.input_count,
                "assignment_count": case.assignment_count,
                "operator_sequence": list(case.operator_sequence),
            }
        )
        artifacts.append(
            {
                "case": case.name,
                "compile_api": "pypto.compiler.compile_structured_strict",
                "compiler_invocations": 1,
                "build_spec_identity_digest": case.expected_build_spec_identity_digest,
                "source_ir_digest": case.expected_source_ir_digest,
                "source_ir_bytes": case.expected_source_ir_bytes,
                "callable_abi_digest": case.expected_callable_abi_digest,
                "static_specialization_digest": case.expected_static_specialization_digest,
                "symbolic_specialization_digest": case.expected_symbolic_specialization_digest,
                "argument_abi_digest": case.expected_argument_abi_digest,
                "result_abi_digest": case.expected_result_abi_digest,
                "mutation_abi_digest": case.expected_mutation_abi_digest,
                "artifact_identity_digest": artifact_identity,
                "cache_key_digest": "b" * 64,
                "loader_compatibility_digest": "c" * 64,
                "device_code_bytes": case.expected_device_code_bytes,
                "device_code_sha256": case.expected_device_code_sha256,
                "kernel_abi_identity_digest": case.expected_callable_abi_digest,
                "entry_function_name": "pypto_fused_pointwise_v2",
                "fallback_used": False,
                "expected_grid": list(case.expected_grid),
                "expected_kernel_arguments": case.expected_kernel_arguments,
                "expected_device_code_bytes": case.expected_device_code_bytes,
                "expected_device_code_sha256": case.expected_device_code_sha256,
                "hir_sha256": case.expected_hir_sha256,
                "hir_bytes": case.expected_hir_bytes,
                "hir_roundtrip_exact": True,
                "input_operand_count": case.input_count,
                "assignment_count": case.assignment_count,
                "operator_sequence": list(case.operator_sequence),
            }
        )
        for repetition in range(case.repetitions):
            input_hashes = [
                {
                    "ordinal": ordinal,
                    "before_sha256": f"{100 + case_index * 16 + ordinal:064x}",
                    "after_sha256": f"{100 + case_index * 16 + ordinal:064x}",
                    "unchanged": True,
                }
                for ordinal in range(case.input_count)
            ]
            guard_hashes = []
            for ordinal in range(case.input_count + 1):
                if ordinal < case.input_count:
                    prefix = finalizer.expected_guard_sha256(
                        case.dtype, contract.INPUT_GUARD_PREFIX_BASE + ordinal
                    )
                    suffix = finalizer.expected_guard_sha256(
                        case.dtype, contract.INPUT_GUARD_SUFFIX_BASE + ordinal
                    )
                else:
                    prefix = finalizer.expected_guard_sha256(
                        case.dtype, contract.OUTPUT_GUARD_PREFIX
                    )
                    suffix = finalizer.expected_guard_sha256(
                        case.dtype, contract.OUTPUT_GUARD_SUFFIX
                    )
                guard_hashes.append(
                    {
                        "allocation": f"input{ordinal}"
                        if ordinal < case.input_count
                        else "output",
                        "prefix_before_sha256": prefix,
                        "prefix_after_sha256": prefix,
                        "suffix_before_sha256": suffix,
                        "suffix_after_sha256": suffix,
                        "unchanged": True,
                    }
                )
            output_sha = f"{900 + case_index * 2 + repetition:064x}"
            executions.append(
                {
                    "case": case.name,
                    "repetition": repetition,
                    "lifetime_ordinal": repetition,
                    "fresh_executable": True,
                    "artifact_identity_digest": artifact_identity,
                    "dtype": case.dtype,
                    "shape": list(case.shape),
                    "strides": list(case.strides),
                    "grid": list(case.expected_grid),
                    "kernel_argument_count": case.expected_kernel_arguments,
                    "raw_current_stream": 1000 + case_index,
                    "raw_reference_stream": 2000 + case_index,
                    "non_default_stream": True,
                    "distinct_nondefault_reference_stream": True,
                    "reference_stream_synchronized_before_candidate": True,
                    "reference_stream_policy": contract.REFERENCE_STREAM_POLICY,
                    "candidate_stream_policy": contract.CANDIDATE_STREAM_POLICY,
                    "reference_compute_boundary": contract.REFERENCE_COMPUTE_BOUNDARY,
                    "capture_free_before": True,
                    "capture_free_at_launch": True,
                    "external_stream_synchronized": True,
                    "expected_logical_bytes_sha256": output_sha,
                    "actual_logical_bytes_sha256": output_sha,
                    "input_hashes": input_hashes,
                    "guard_hashes": guard_hashes,
                    "guard_elements": contract.GUARD_ELEMENTS,
                    "input_unchanged": True,
                    "guards_unchanged": True,
                    "comparison": {
                        "policy": case.comparison,
                        "max_ulp_limit": case.max_ulp,
                        "rtol": case.rtol,
                        "atol": case.atol,
                        "observed_max_ulp": 0,
                        "observed_max_relative_error": 0.0,
                        "observed_max_absolute_error": 0.0,
                        "special_classification_and_sign_passed": True,
                        "negative_zero_fma_discriminator_passed": case.family
                        == "arithmetic",
                        "no_subnormals": True,
                    },
                    "comparison_passed": True,
                    "packet_released_after_synchronization": True,
                    "explicit_unload": True,
                    "terminal_state": "Unloaded",
                    "bound_context_before_unload": 123,
                    "bound_context_id_before_unload": 456,
                    "bound_context_after_unload": 0,
                    "bound_context_id_after_unload": 0,
                }
            )
    lifetime_count = len(executions)
    return {
        "scope": {
            "frontend_family": "FusedPointwiseV2",
            "fixed_fixture_set": "full-nine-case-numerical-v1",
            "fixed_fixture_correctness": True,
            "general_operator_correctness": False,
            "legacy_cp44_unchanged": True,
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
            "case_order": list(contract.CASE_ORDER),
            "compile_invocations_per_case": 1,
            "repetitions_per_case": 2,
            "module_lifetimes": lifetime_count,
            "explicit_packet_releases": lifetime_count,
            "explicit_unloads": lifetime_count,
            "non_default_current_stream": True,
            "distinct_nondefault_reference_stream": True,
            "reference_compute_outside_candidate_coverage": True,
            "external_reference_synchronizations": lifetime_count,
            "external_synchronization": True,
            "fallback_used": False,
            "forbidden_provider_imports": [],
        },
    }


class FinalizerValidationTest(unittest.TestCase):
    def test_frontend_result_schema_and_table_mutations(self) -> None:
        provisional = synthetic_provisional()
        finalizer.validate_scope(provisional)
        finalizer.validate_frontend_results(provisional)
        mutations = (
            (("runtime", "compile_invocations_per_case"), 2),
            (("runtime", "module_lifetimes"), 17),
            (("runtime", "distinct_nondefault_reference_stream"), False),
            (("runtime", "executions", 0, "fresh_executable"), False),
            (("runtime", "executions", 0, "guards_unchanged"), False),
            (("runtime", "executions", 0, "raw_current_stream"), 1),
            (("runtime", "executions", 0, "raw_reference_stream"), 1000),
            (("runtime", "executions", 0, "comparison", "max_ulp_limit"), 99),
            (("runtime", "artifacts", 0, "entry_function_name"), "pypto_vector_add"),
            (("runtime", "hir_programs", 0, "parameter_directions"), ["In", "In"]),
        )
        for path, value in mutations:
            candidate = copy.deepcopy(provisional)
            target: object = candidate
            for key in path[:-1]:
                target = target[key]  # type: ignore[index]
            target[path[-1]] = value  # type: ignore[index]
            with self.subTest(path=path), self.assertRaises(finalizer.FinalizeError):
                finalizer.validate_frontend_results(candidate)

    def test_scope_rejects_broader_and_nested_claims(self) -> None:
        provisional = synthetic_provisional()
        provisional["scope"]["general_operator_correctness"] = True
        with self.assertRaisesRegex(finalizer.FinalizeError, "scope exceeds"):
            finalizer.validate_scope(provisional)
        provisional = synthetic_provisional()
        provisional["runtime"]["latency_ns"] = 1
        with self.assertRaisesRegex(finalizer.FinalizeError, "unaccepted claim"):
            finalizer.validate_scope(provisional)

    def test_finalizer_replay_child_is_deserialization_only(self) -> None:
        compile(finalizer.REPLAY_AUDIT_PROGRAM, "<fused-replay>", "exec")
        for marker in (
            "compile_structured_strict",
            "Artifact.compile_strict",
            "NvidiaExecutable(",
            "observe_current_nvidia_runtime",
        ):
            self.assertNotIn(marker, finalizer.REPLAY_AUDIT_PROGRAM)
        self.assertIn("pypto.ir.deserialize(hir_bytes)", finalizer.REPLAY_AUDIT_PROGRAM)
        self.assertIn(
            "assert not torch.cuda.is_initialized()", finalizer.REPLAY_AUDIT_PROGRAM
        )

    def test_exact_cpu_replay_child_deserializes_final_anchor_evidence(self) -> None:
        anchors = json.loads(
            (ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH).read_text()
        )
        run_id = anchors["anchor_runs"][0]["run_id"]
        replay = ROOT / "runs" / run_id / "fused-pointwise-compile-anchor-replay"
        command = [
            str((ROOT / contract.PYTHON_REAL_RELATIVE_PATH).resolve(strict=True)),
            "-I",
            "-B",
            "-S",
            "-c",
            finalizer.REPLAY_AUDIT_PROGRAM,
            str(ROOT),
            str(replay),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "void",
            },
            check=False,
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2048:])
        audited = json.loads(completed.stdout)
        self.assertEqual(
            [record["case"] for record in audited["hir_programs"]],
            list(contract.CASE_ORDER),
        )
        self.assertEqual(
            [record["device_code_sha256"] for record in audited["artifacts"]],
            [case.expected_device_code_sha256 for case in contract.CASE_SPECS],
        )

    def _numerical_fixture(
        self, workspace: pathlib.Path, run_id: str
    ) -> dict[str, object]:
        replay = contract.replay_directory(workspace, run_id)
        replay.mkdir(parents=True)
        provisional = synthetic_provisional()
        executions = provisional["runtime"]["executions"]
        index = 0
        for case in contract.CASE_SPECS:
            for repetition in range(case.repetitions):
                execution = executions[index]
                index += 1
                inputs = []
                for ordinal in range(case.input_count):
                    words = finalizer._input_words(case, repetition, ordinal)
                    raw = pack_words(words, case.dtype)
                    (
                        replay / f"{case.name}.r{repetition}.input{ordinal}.bin"
                    ).write_bytes(raw)
                    execution["input_hashes"][ordinal]["before_sha256"] = (
                        finalizer.sha256_bytes(raw)
                    )
                    execution["input_hashes"][ordinal]["after_sha256"] = (
                        finalizer.sha256_bytes(raw)
                    )
                    inputs.append(words)
                reference = pack_words(
                    finalizer._cpu_reference_words(case, inputs), case.dtype
                )
                (replay / f"{case.name}.r{repetition}.reference.bin").write_bytes(
                    reference
                )
                (replay / f"{case.name}.r{repetition}.actual.bin").write_bytes(
                    reference
                )
                digest = finalizer.sha256_bytes(reference)
                execution["expected_logical_bytes_sha256"] = digest
                execution["actual_logical_bytes_sha256"] = digest
                execution["comparison"].update(
                    finalizer._compare_words(
                        case,
                        finalizer._decode_words(reference, case.dtype),
                        finalizer._decode_words(reference, case.dtype),
                    )
                )
        return provisional

    def test_independent_numerical_replay_and_raw_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = pathlib.Path(directory)
            run_id = "pypto-20990101T000000Z-123456-abcdef"
            provisional = self._numerical_fixture(workspace, run_id)
            audited = finalizer.audit_numerical_replay(provisional, workspace, run_id)
            self.assertEqual(len(audited), 18)
            self.assertTrue(
                all(
                    record["independent_cpu_reference_reconstruction"]
                    for record in audited
                )
            )
            self.assertTrue(
                all(
                    set(record)
                    == {
                        "case",
                        "repetition",
                        "independent_cpu_input_reconstruction",
                        "independent_cpu_reference_reconstruction",
                        "actual_sha256",
                        "candidate_vs_cpu",
                        "candidate_vs_torch",
                        "cpu_reference_sha256",
                        "torch_reference_sha256",
                        "torch_vs_cpu",
                    }
                    for record in audited
                )
            )
            replay = contract.replay_directory(workspace, run_id)
            targets = (
                replay / "arith_fp32_tail.r0.input0.bin",
                replay / "exp_fp32_tail1.r0.reference.bin",
                replay / "max16x64_fp32_tail1.r1.actual.bin",
            )
            for target in targets:
                original = target.read_bytes()
                target.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                with (
                    self.subTest(target=target.name),
                    self.assertRaises(finalizer.FinalizeError),
                ):
                    finalizer.audit_numerical_replay(provisional, workspace, run_id)
                target.write_bytes(original)

    def test_replay_exact_set_modes_and_extra_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = pathlib.Path(directory)
            run_id = "pypto-20990101T000000Z-123456-abcdef"
            replay = contract.replay_directory(workspace, run_id)
            replay.mkdir(parents=True)
            records = []
            for index, name in enumerate(expected_replay_names()):
                raw = f"replay-{index}".encode()
                path = replay / name
                path.write_bytes(raw)
                path.chmod(0o444)
                records.append(
                    {
                        "path": path.relative_to(workspace).as_posix(),
                        "bytes": len(raw),
                        "sha256": finalizer.sha256_bytes(raw),
                    }
                )
            (replay / contract.PROVISIONAL_NAME).write_bytes(b"{}\n")
            provisional = {"inputs": {"replay_files": records}}
            self.assertEqual(
                len(finalizer.validate_replay(provisional, workspace, run_id)),
                len(records),
            )
            extra = replay / "unexpected.bin"
            extra.write_bytes(b"x")
            with self.assertRaisesRegex(finalizer.FinalizeError, "missing or extra"):
                finalizer.validate_replay(provisional, workspace, run_id)
            extra.unlink()
            first = replay / expected_replay_names()[0]
            first.chmod(0o644)
            with self.assertRaisesRegex(finalizer.FinalizeError, "bytes differ"):
                finalizer.validate_replay(provisional, workspace, run_id)

    def test_canonical_json_duplicate_keys_and_no_replace(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            parent = pathlib.Path(directory)
            duplicate = parent / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}\n')
            with self.assertRaisesRegex(runner.SmokeError, "duplicate"):
                runner.load_canonical_json(duplicate, "duplicate fixture")
            noncanonical = parent / "noncanonical.json"
            noncanonical.write_text('{"a": 1}\n')
            with self.assertRaisesRegex(runner.SmokeError, "canonical"):
                runner.load_canonical_json(noncanonical, "noncanonical fixture")
            output = parent / "accepted.json"
            digest = finalizer.publish_no_replace(output, {"accepted": True})
            original = output.read_bytes()
            self.assertEqual(finalizer.sha256_bytes(original), digest)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
            with self.assertRaisesRegex(finalizer.FinalizeError, "already exists"):
                finalizer.publish_no_replace(output, {"accepted": False})
            self.assertEqual(output.read_bytes(), original)


class FullSyntheticFinalizationTest(unittest.TestCase):
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
            "contract": pathlib.Path("tools/_pypto_fused_pointwise_sm120_contract.py"),
            "controller": pathlib.Path(
                "tools/run_pypto_fused_pointwise_sm120_isolated.py"
            ),
            "finalizer": pathlib.Path("tools/finalize_pypto_fused_pointwise_sm120.py"),
            "anchor_generator": contract.ANCHOR_GENERATOR_RELATIVE_PATH,
            "compile_anchors": contract.COMPILE_ANCHORS_RELATIVE_PATH,
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
        generator_path = self.workspace / self.relative_files["anchor_generator"]
        anchors_path = self.workspace / self.relative_files["compile_anchors"]
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
                contract,
                "CUDA_RUNTIME_SHA256",
                finalizer.sha256_file(runtime_path),
            ),
            mock.patch.object(
                contract, "PYTHON_REAL_RELATIVE_PATH", self.relative_files["python"]
            ),
            mock.patch.object(contract, "PYTHON_SIZE", python_path.stat().st_size),
            mock.patch.object(
                contract, "PYTHON_SHA256", finalizer.sha256_file(python_path)
            ),
            mock.patch.object(
                contract, "ANCHOR_GENERATOR_SIZE", generator_path.stat().st_size
            ),
            mock.patch.object(
                contract,
                "ANCHOR_GENERATOR_SHA256",
                finalizer.sha256_file(generator_path),
            ),
            mock.patch.object(
                contract, "COMPILE_ANCHORS_SIZE", anchors_path.stat().st_size
            ),
            mock.patch.object(
                contract,
                "COMPILE_ANCHORS_SHA256",
                finalizer.sha256_file(anchors_path),
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
            "manifest_path": "state/contracts/fused-fixture.json",
            "manifest_bytes": 1,
            "manifest_sha256": "f" * 64,
            "implementation_commit": "1" * 40,
            "implementation_tree": "2" * 40,
            "current_head": "3" * 40,
            "current_tree": "4" * 40,
            "root_clean": True,
            "parent_control_manifest": {"sha256": "e" * 64},
            "legacy_control_manifest": {"sha256": "d" * 64},
            "legacy_accepted_report": {"sha256": "c" * 64},
            "compile_anchors": {"sha256": "b" * 64},
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

    def build_fixture(self) -> tuple[dict[str, object], str]:
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

        partial = synthetic_provisional()
        runtime = partial["runtime"]
        executions = runtime["executions"]
        replay_files: list[dict[str, object]] = []

        def replay_file(name: str, payload: bytes) -> None:
            path = self.replay / name
            path.write_bytes(payload)
            path.chmod(0o444)
            replay_files.append(
                {
                    "path": path.relative_to(self.workspace).as_posix(),
                    "bytes": len(payload),
                    "sha256": finalizer.sha256_bytes(payload),
                }
            )

        replay_file("compile-request.msgpack", b"synthetic-request")
        for case in contract.CASE_SPECS:
            replay_file(f"{case.name}.hir.msgpack", b"synthetic-hir")
            replay_file(f"{case.name}.source.mlir", b"synthetic-source")
            replay_file(f"{case.name}.build-spec.msgpack", b"synthetic-spec")
            replay_file(f"{case.name}.artifact.msgpack", b"synthetic-artifact")
            replay_file(f"{case.name}.cubin", b"synthetic-cubin")
        execution_index = 0
        for case in contract.CASE_SPECS:
            for repetition in range(case.repetitions):
                execution = executions[execution_index]
                execution_index += 1
                inputs = []
                for ordinal in range(case.input_count):
                    words = finalizer._input_words(case, repetition, ordinal)
                    raw = pack_words(words, case.dtype)
                    replay_file(f"{case.name}.r{repetition}.input{ordinal}.bin", raw)
                    digest = finalizer.sha256_bytes(raw)
                    execution["input_hashes"][ordinal]["before_sha256"] = digest
                    execution["input_hashes"][ordinal]["after_sha256"] = digest
                    inputs.append(words)
                reference = pack_words(
                    finalizer._cpu_reference_words(case, inputs), case.dtype
                )
                replay_file(f"{case.name}.r{repetition}.reference.bin", reference)
                replay_file(f"{case.name}.r{repetition}.actual.bin", reference)
                digest = finalizer.sha256_bytes(reference)
                execution["expected_logical_bytes_sha256"] = digest
                execution["actual_logical_bytes_sha256"] = digest
                execution["comparison"].update(
                    finalizer._compare_words(
                        case,
                        finalizer._decode_words(reference, case.dtype),
                        finalizer._decode_words(reference, case.dtype),
                    )
                )

        integrity_paths = {
            "anchor_generator": self.workspace
            / self.relative_files["anchor_generator"],
            "compile_anchors": self.workspace / self.relative_files["compile_anchors"],
            "contract": self.workspace / self.relative_files["contract"],
            "runner": self.workspace / contract.RUNNER_RELATIVE_PATH,
            "controller": self.workspace / self.relative_files["controller"],
            "environment_lock": self.workspace
            / self.relative_files["environment_lock"],
            "versions_lock": self.workspace / self.relative_files["versions_lock"],
            "workspace_lock": self.workspace / self.relative_files["workspace_lock"],
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
                "child_pre_cuda_gate": {
                    "static_identity": self.static_identity,
                    "gpu": self.gpu,
                    "free_memory_mib": 20_000,
                    "protected_heavy_pids": [10],
                    "protected_runtime_pids": [],
                    "unreadable_protected_maps": [],
                    "nvidia_compute_pids": [],
                    "control_manifest": self.control_identity,
                },
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
                "compile_request": {
                    "byte_identity_digest": "1" * 64,
                    "loader_compatibility_input_digest": "2" * 64,
                    "device_autotune_identity_digest": "3" * 64,
                },
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
        return provisional, provisional_sha

    @contextlib.contextmanager
    def finalize_patches(self, *, control_identity: dict[str, object] | None = None):
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
                return_value={"exact_cpu_replay_child": True},
            ),
            mock.patch.object(
                finalizer, "git_identity", return_value=self.pypto_identity
            ),
        ):
            yield

    def call_finalize(
        self,
        provisional_sha: str,
        *,
        control_identity: dict[str, object] | None = None,
    ):
        with self.finalize_patches(control_identity=control_identity):
            return finalizer.finalize(
                workspace=self.workspace,
                run_id=self.run_id,
                expected_provisional_sha256=provisional_sha,
            )

    def mutate_json(
        self, path: pathlib.Path, keys: tuple[str, ...], value: object
    ) -> str:
        document = json.loads(path.read_text())
        target = document
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        path.chmod(0o600)
        return self.write_json(
            path, document, mode=0o444 if path == self.provisional_path else 0o600
        )

    def test_complete_finalization_publication_and_no_replace(self) -> None:
        provisional, provisional_sha = self.build_fixture()
        report, output, digest = self.call_finalize(provisional_sha)
        self.assertEqual(
            report["status"],
            "accepted-real-sm120-fused-pointwise-nine-case-correctness-gate",
        )
        self.assertEqual(report["result"], provisional["runtime"])
        self.assertEqual(digest, finalizer.sha256_file(output))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        original = output.read_bytes()
        with self.assertRaisesRegex(finalizer.FinalizeError, "already exists"):
            self.call_finalize(provisional_sha)
        self.assertEqual(output.read_bytes(), original)

    def test_table_driven_full_transaction_tampering(self) -> None:
        for name in (
            "process",
            "preflight",
            "gate",
            "barrier",
            "audit",
            "control",
            "integrity",
            "replay",
            "raw",
        ):
            with self.subTest(name=name):
                _provisional, provisional_sha = self.build_fixture()
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
                        ("gpu_smoke_post_exit_audit", "external_nvidia_compute_pids"),
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
                elif name == "raw":
                    path = self.replay / "exp_fp32_tail1.r0.actual.bin"
                    original = path.read_bytes()
                    path.chmod(0o600)
                    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                    path.chmod(0o444)
                with self.assertRaises(finalizer.FinalizeError):
                    self.call_finalize(provisional_sha, control_identity=live_control)


class DocumentationTest(unittest.TestCase):
    def test_runbook_preserves_claim_boundaries(self) -> None:
        text = (ROOT / "docs/pypto_fused_pointwise_sm120_smoke.md").read_text()
        for marker in (
            "full nine-case",
            "arith_fp32_tail",
            "max16x64_fp32_tail1",
            "4 ULP",
            "1 ULP",
            "prefix and suffix",
            "distinct non-default",
            "CP44",
            "must not run",
        ):
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
