from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType


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


contract = load_source(
    "test_row_reduction_contract", "tools/_pypto_row_reduction_sm120_contract.py"
)
runner = load_source(
    "test_row_reduction_runner", "benchmarks/operators/pypto_row_reduction_sm120.py"
)
control = load_source(
    "test_row_reduction_control", "tools/_pypto_row_reduction_sm120_control_manifest.py"
)
controller = load_source(
    "test_row_reduction_controller", "tools/run_pypto_row_reduction_sm120_isolated.py"
)
finalizer = load_source(
    "test_row_reduction_finalizer", "tools/finalize_pypto_row_reduction_sm120.py"
)


class ContractAndAnchorTest(unittest.TestCase):
    def test_exact_runtime_and_anchor_source_identities(self) -> None:
        for path, size, digest in (
            (
                ROOT / contract.RUNNER_RELATIVE_PATH,
                contract.RUNNER_SIZE,
                contract.RUNNER_SHA256,
            ),
            (
                ROOT / contract.ANCHOR_GENERATOR_RELATIVE_PATH,
                contract.ANCHOR_GENERATOR_SIZE,
                contract.ANCHOR_GENERATOR_SHA256,
            ),
            (
                ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH,
                contract.COMPILE_ANCHORS_SIZE,
                contract.COMPILE_ANCHORS_SHA256,
            ),
        ):
            raw = path.read_bytes()
            self.assertEqual(len(raw), size)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)

    def test_exact_ten_case_matrix_and_twenty_lifetimes(self) -> None:
        actual = [
            (
                case.name,
                case.dtype,
                case.shape,
                case.op_name,
                case.row_tile,
                case.grid,
                contract.contraction_tile(case),
                contract.contraction_chunks(case),
            )
            for case in contract.CASE_SPECS
        ]
        self.assertEqual(
            actual,
            [
                (
                    "rank1_fp32_sum_n1",
                    "float32",
                    (1,),
                    "tensor.row_sum",
                    1,
                    (1, 1, 1),
                    1,
                    1,
                ),
                (
                    "rank1_fp32_sum_n7",
                    "float32",
                    (7,),
                    "tensor.row_sum",
                    1,
                    (1, 1, 1),
                    1,
                    7,
                ),
                (
                    "rank2_bf16_sum_n256_tail",
                    "bfloat16",
                    (17, 256),
                    "tensor.row_sum",
                    16,
                    (2, 1, 1),
                    128,
                    2,
                ),
                (
                    "rank2_bf16_max_n17",
                    "bfloat16",
                    (2, 17),
                    "tensor.row_max",
                    2,
                    (1, 1, 1),
                    1,
                    17,
                ),
                (
                    "rank2_fp32_max_n128_tail",
                    "float32",
                    (5, 128),
                    "tensor.row_max",
                    4,
                    (2, 1, 1),
                    128,
                    1,
                ),
                (
                    "rank2_fp32_max_n96_tail",
                    "float32",
                    (5, 96),
                    "tensor.row_max",
                    4,
                    (2, 1, 1),
                    32,
                    3,
                ),
                (
                    "rank2_bf16_sum_n129",
                    "bfloat16",
                    (8, 129),
                    "tensor.row_sum",
                    8,
                    (1, 1, 1),
                    1,
                    129,
                ),
                (
                    "rank3_fp32_sum_n17_tail",
                    "float32",
                    (2, 3, 17),
                    "tensor.row_sum",
                    4,
                    (2, 1, 1),
                    1,
                    17,
                ),
                (
                    "rank3_bf16_max_n17",
                    "bfloat16",
                    (2, 16, 17),
                    "tensor.row_max",
                    16,
                    (2, 1, 1),
                    1,
                    17,
                ),
                (
                    "rank3_fp32_max_n257_tail",
                    "float32",
                    (2, 3, 257),
                    "tensor.row_max",
                    4,
                    (2, 1, 1),
                    1,
                    257,
                ),
            ],
        )
        self.assertEqual(len(contract.CASE_SPECS) * contract.REPETITIONS, 20)

    def test_guard_bounds_are_derived_and_sentinels_are_exact(self) -> None:
        self.assertEqual(contract.MAXIMUM_REQUIRED_INPUT_GUARD_ELEMENTS, 3967)
        self.assertEqual(contract.INPUT_GUARD_ELEMENTS, 4096)
        self.assertEqual(contract.MAXIMUM_REQUIRED_OUTPUT_GUARD_ELEMENTS, 15)
        self.assertEqual(contract.OUTPUT_GUARD_ELEMENTS, 16)
        worst = next(
            case
            for case in contract.CASE_SPECS
            if case.name == "rank2_bf16_sum_n256_tail"
        )
        self.assertEqual(contract.required_input_guard_elements(worst), 3967)
        for dtype, expected in contract.SENTINEL_WORDS.items():
            encoded = tuple(
                runner.encode_word(value, dtype)
                for value in (
                    contract.INPUT_GUARD_PREFIX,
                    contract.INPUT_GUARD_SUFFIX,
                    contract.OUTPUT_GUARD_PREFIX,
                    contract.OUTPUT_GUARD_SUFFIX,
                )
            )
            self.assertEqual(encoded, expected)
            self.assertEqual(len(set(encoded)), 4)

    def test_compile_anchor_manifest_binds_two_identical_isolated_runs(self) -> None:
        path = ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH
        raw = path.read_bytes()
        self.assertEqual(len(raw), contract.COMPILE_ANCHORS_SIZE)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), contract.COMPILE_ANCHORS_SHA256
        )
        anchors = json.loads(raw)
        self.assertEqual(len(anchors["records"]), 10)
        self.assertEqual(len(anchors["anchor_runs"]), 2)
        self.assertNotEqual(
            anchors["anchor_runs"][0]["run_id"], anchors["anchor_runs"][1]["run_id"]
        )
        self.assertEqual(anchors["matrix_policy"]["executions"], 20)
        self.assertEqual(
            anchors["matrix_policy"]["input_guard_elements_per_side"], 4096
        )
        self.assertEqual(anchors["matrix_policy"]["output_guard_elements_per_side"], 16)
        normalized = {
            name: anchors[name]
            for name in (
                "schema_version",
                "kind",
                "pypto",
                "dso",
                "generator",
                "base_runner",
                "anchor_request",
                "cp48_report",
                "matrix_policy",
                "records",
                "records_sha256",
            )
        }
        normalized_raw = control.canonical_json(normalized)
        self.assertEqual(len(normalized_raw), anchors["normalized_records_bytes"])
        self.assertEqual(
            hashlib.sha256(normalized_raw).hexdigest(),
            anchors["normalized_records_sha256"],
        )
        identity = control.validate_compile_anchors(ROOT, contract)
        self.assertEqual(len(identity["anchor_run_ids"]), 2)

    def test_cp48_sources_and_cubins_join_all_four_overlap_cases(self) -> None:
        anchors = json.loads(
            (ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH).read_text()
        )
        cp48 = json.loads((ROOT / contract.CP48_REPORT_RELATIVE_PATH).read_text())
        old = {record["case"]: record for record in cp48["records"]}
        joined = 0
        for record in anchors["records"]:
            if record["cp48_case"] is None:
                continue
            frozen = old[record["cp48_case"]]
            self.assertEqual(record["source_ir_digest"], frozen["source_ir_digest"])
            self.assertEqual(record["device_code_sha256"], frozen["device_code_sha256"])
            joined += 1
        self.assertEqual(joined, 4)

    def test_bf16_sources_widen_reduce_and_demote(self) -> None:
        for case in contract.CASE_SPECS:
            source = contract.canonical_tensor_ir_source(case).decode()
            if case.dtype == "bfloat16":
                self.assertIn("%wide0 = convert %arg0", source)
                self.assertIn("xf32>", source)
                self.assertIn("%result0 = convert", source)
                self.assertNotIn("reduce(%arg0)", source)
            self.assertEqual(
                hashlib.sha256(source.encode()).hexdigest(),
                next(
                    record["source_sha256"]
                    for record in json.loads(
                        (ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH).read_text()
                    )["records"]
                    if record["case"] == case.name
                ),
            )


class NumericalPolicyTest(unittest.TestCase):
    def test_bf16_fp32_accumulation_discriminators_are_exact(self) -> None:
        for contraction, expected in ((129, 0x3FC0), (256, 0x4000)):
            case = next(
                case
                for case in contract.CASE_SPECS
                if case.dtype == "bfloat16"
                and case.op_name == "tensor.row_sum"
                and case.contraction == contraction
            )
            cpu = runner.cpu_reference_words(case, 0)
            self.assertEqual(cpu, [expected] * case.rows)
            accumulator = runner.bfloat16_word(1.0)
            increment = runner.word_value(runner.bfloat16_word(2.0**-8), "bfloat16")
            for _ in range(contraction - 1):
                accumulator = runner.bfloat16_word(
                    runner.word_value(accumulator, "bfloat16") + increment
                )
            self.assertEqual(accumulator, 0x3F80)
            self.assertNotEqual(accumulator, expected)

    def test_sum_and_max_special_semantics_are_explicit(self) -> None:
        sum_case = next(
            case
            for case in contract.CASE_SPECS
            if case.op_name == "tensor.row_sum" and case.rows >= 5
        )
        sum_words = runner.cpu_reference_words(sum_case, 1)
        classes = [
            runner.base._classification(word, sum_case.dtype)[0]
            for word in sum_words[:5]
        ]
        self.assertEqual(classes, ["zero", "inf", "inf", "nan", "nan"])
        self.assertEqual(sum_words[0], 0)
        max_case = next(
            case
            for case in contract.CASE_SPECS
            if case.op_name == "tensor.row_max" and case.rows >= 5
        )
        max_words = runner.cpu_reference_words(max_case, 1)
        max_inputs = [
            runner.encode_word(value, max_case.dtype)
            for value in runner.input_values(max_case, 1)[:2]
        ]
        self.assertEqual(max_inputs, [0x80000000, 0x00000000])
        self.assertEqual(max_words[0], 0)
        self.assertEqual(max_words[1], 0x80000000)
        self.assertEqual(
            runner.base._classification(max_words[2], max_case.dtype)[0], "inf"
        )
        self.assertEqual(
            runner.base._classification(max_words[4], max_case.dtype)[0], "nan"
        )

    def test_exact_and_tolerance_comparisons_are_separate(self) -> None:
        sum_case = next(
            case
            for case in contract.CASE_SPECS
            if case.name == "rank3_fp32_sum_n17_tail"
        )
        word = runner.float32_word(1.0)
        runner.compare_word_sequences(sum_case, [word], [word], exact_required=True)
        with self.assertRaises(runner.SmokeError):
            runner.compare_word_sequences(
                sum_case, [word + 1], [word], exact_required=True
            )
        runner.compare_word_sequences(
            sum_case, [word + 1], [word], exact_required=False
        )
        self.assertEqual(contract.FP32_SUM_MAX_ULP, 16)

    def test_runner_and_finalizer_independently_reconstruct_all_inputs_and_cpu_outputs(
        self,
    ) -> None:
        for case in contract.CASE_SPECS:
            for repetition in range(2):
                self.assertEqual(
                    runner.input_values(case, repetition),
                    finalizer.input_values(case, repetition),
                )
                self.assertEqual(
                    runner.cpu_reference_words(case, repetition),
                    finalizer.cpu_reference(case, repetition),
                )

    def test_comparison_metadata_and_bf16_negative_control_are_frozen(self) -> None:
        case = next(
            case
            for case in contract.CASE_SPECS
            if case.name == "rank2_bf16_sum_n256_tail"
        )
        metadata = finalizer.comparison_metadata(case, 0)
        value = {
            **metadata,
            "candidate_vs_torch": {},
            "candidate_vs_cpu": {},
            "torch_vs_cpu": {},
        }
        self.assertEqual(
            set(value),
            {
                "policy",
                "max_ulp_limit",
                "rtol",
                "atol",
                "candidate_vs_torch",
                "candidate_vs_cpu",
                "torch_vs_cpu",
                "special_classification_and_sign_passed",
                "bf16_fp32_accumulation_discriminator_passed",
                "bf16_expected_output_word",
                "bf16_sequential_accumulator_word",
            },
        )
        finalizer.validate_comparison_metadata(value, case, 0)
        self.assertEqual(value["bf16_expected_output_word"], 0x4000)
        self.assertEqual(value["bf16_sequential_accumulator_word"], 0x3F80)
        for name in (
            "policy",
            "max_ulp_limit",
            "rtol",
            "atol",
            "special_classification_and_sign_passed",
            "bf16_fp32_accumulation_discriminator_passed",
            "bf16_expected_output_word",
            "bf16_sequential_accumulator_word",
        ):
            candidate = copy.deepcopy(value)
            candidate[name] = None if value[name] is not None else 1
            with self.subTest(name=name), self.assertRaises(finalizer.FinalizeError):
                finalizer.validate_comparison_metadata(candidate, case, 0)
        ordinary = next(
            case
            for case in contract.CASE_SPECS
            if case.name == "rank2_fp32_max_n96_tail"
        )
        ordinary_value = {
            **finalizer.comparison_metadata(ordinary, 1),
            "candidate_vs_torch": {},
            "candidate_vs_cpu": {},
            "torch_vs_cpu": {},
        }
        self.assertIsNone(ordinary_value["bf16_expected_output_word"])
        self.assertIsNone(ordinary_value["bf16_sequential_accumulator_word"])
        finalizer.validate_comparison_metadata(ordinary_value, ordinary, 1)
        extra = copy.deepcopy(value)
        extra["policy_duplicate"] = value["policy"]
        with self.assertRaises(finalizer.FinalizeError):
            finalizer.validate_comparison_metadata(extra, case, 0)


class StructureAndSafetyTest(unittest.TestCase):
    def test_direct_child_admission_precedes_torch(self) -> None:
        source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text()
        run = source[source.index("def run_smoke()") :]
        markers = (
            "workspace_from_environment()",
            "wait_for_start_barrier(workspace, run_id)",
            "load_contract_and_child_gate(",
            "load_anchors(workspace, contract_module)",
            "import torch",
        )
        positions = [run.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        gate = source[
            source.index("def load_contract_and_child_gate(") : source.index(
                "def load_anchors("
            )
        ]
        for marker in (
            "control.validate_control_manifest(workspace)",
            "preflight.mem_available_kib()",
            "preflight.nvidia_identity()",
            "preflight.nvidia_compute_pids()",
            "protected_runtime, unreadable = preflight.protected_nvidia_runtime_mappings(",
        ):
            self.assertIn(marker, gate)
        self.assertNotIn("import torch", gate)

    def test_runner_lifecycle_and_controller_sequence_are_complete(self) -> None:
        runner_source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text()
        for marker in (
            "reference_stream.synchronize()",
            "executable.launch(packet, raw_stream)",
            "candidate_stream.synchronize()",
            "del packet",
            "executable.unload()",
            '"module_lifetimes": 20',
            '"explicit_packet_releases": 20',
            '"explicit_unloads": 20',
        ):
            self.assertIn(marker, runner_source)
        controller_source = (ROOT / contract.CONTROLLER_RELATIVE_PATH).read_text()
        controller_main = controller_source[controller_source.index("def main()") :]
        markers = (
            'description="initial row preflight"',
            'description="action-boundary row preflight"',
            "process = subprocess.Popen(",
            "metadata = isolation.build_run_metadata(",
            "gate_and_release(",
            "isolation.wait_with_gpu_smoke_watchdog(",
            "post_snapshot, post_violation = isolation.audit_gpu_smoke_runtime_state(",
            "enforce_no_survivors(process, metadata)",
        )
        positions = [controller_main.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_manifest_transition_and_controller_fail_closed_order(self) -> None:
        manifest = ROOT / control.MANIFEST_RELATIVE_PATH
        source = (ROOT / contract.CONTROLLER_RELATIVE_PATH).read_text()
        self.assertLess(
            source.index("control.validate_control_manifest(ROOT)"),
            source.index("process = subprocess.Popen("),
        )
        if not manifest.is_file():
            with self.assertRaises(RuntimeError):
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
            self.assertTrue(control.validate_control_manifest(ROOT)["root_clean"])

    def test_no_runtime_rewriting_or_direct_external_signal_calls(self) -> None:
        joined = "\n".join(
            (ROOT / path).read_text()
            for path in (
                contract.RUNNER_RELATIVE_PATH,
                contract.CONTROLLER_RELATIVE_PATH,
                contract.FINALIZER_RELATIVE_PATH,
            )
        )
        for forbidden in (
            "sys.orig_argv =",
            "subprocess.run =",
            ".subprocess =",
            "os.kill(",
            "os.killpg(",
        ):
            self.assertNotIn(forbidden, joined)

    def test_no_replace_finalizer_primitive(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            output = pathlib.Path(directory) / "report.json"
            original_root = finalizer.base.base.ROOT
            finalizer.base.base.ROOT = pathlib.Path(directory).resolve()
            try:
                finalizer.base.base.publish_no_replace(output, {"accepted": True})
                with self.assertRaises(finalizer.base.base.FinalizeError):
                    finalizer.base.base.publish_no_replace(output, {"accepted": False})
            finally:
                finalizer.base.base.ROOT = original_root


class FinalizerAdversarialTest(unittest.TestCase):
    def test_exact_provisional_scope_and_runtime_key_schemas(self) -> None:
        provisional = {
            "schema_version": 1,
            "smoke": contract.SMOKE_NAME,
            "acceptance": "gpu-execution-complete-awaiting-run-finalization",
            "scope": {
                "frontend_family": "RowReductionV3",
                "fixed_case_count": 10,
                "fixed_case_correctness": True,
                "general_reduction_correctness": False,
                "performance_result": False,
                "framework_or_model_result": False,
            },
            "inputs": {
                "integrity": {},
                "control_manifest": {},
                "pypto": {},
                "tensor_ir_head": "",
                "cuda_tile_head": "",
                "llvm_head": "",
                "replay_files": [],
            },
            "run_context": {
                "run_id": "",
                "mode": "gpu-smoke",
                "pid": 1,
                "pgid": 1,
                "start_ticks": 1,
                "initial_preflight": {},
                "preflight": {},
                "gate": {},
                "start_barrier_sha256": "",
                "protected_zero_nvidia_policy": True,
                "admission_policy": {},
            },
            "runtime": {
                "torch": {},
                "child_pre_cuda_gate": {},
                "libcudart_paths": [],
                "observation": {},
                "compile_request": {},
                "hir_programs": [],
                "artifacts": [],
                "executions": [],
                "case_order": [],
                "case_count": 10,
                "compile_invocations_per_case": 1,
                "repetitions_per_case": 2,
                "module_lifetimes": 20,
                "explicit_packet_releases": 20,
                "explicit_unloads": 20,
                "non_default_current_stream": True,
                "distinct_nondefault_reference_stream": True,
                "reference_compute_outside_candidate_coverage": True,
                "external_reference_synchronizations": 20,
                "external_synchronization": True,
                "fallback_used": False,
                "forbidden_provider_imports": [],
            },
        }
        finalizer.validate_provisional_schema(provisional)
        for location in ("top", "scope", "runtime"):
            candidate = copy.deepcopy(provisional)
            target = candidate if location == "top" else candidate[location]
            target["unexpected"] = True
            with (
                self.subTest(location=location),
                self.assertRaises(finalizer.FinalizeError),
            ):
                finalizer.validate_provisional_schema(candidate)

    def _provisional(self) -> tuple[dict[str, object], dict[str, object]]:
        anchors = json.loads(
            (ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH).read_text()
        )
        anchor_records = {record["case"]: record for record in anchors["records"]}
        artifacts = [
            {
                "case": case.name,
                "build_spec_identity_digest": anchor_records[case.name][
                    "build_spec_sha256"
                ],
                "artifact_identity_digest": anchor_records[case.name][
                    "artifact_sha256"
                ],
                "device_code_sha256": anchor_records[case.name]["device_code_sha256"],
                "fallback_used": False,
            }
            for case in contract.CASE_SPECS
        ]
        executions: list[dict[str, object]] = []
        for case in contract.CASE_SPECS:
            sentinels = contract.SENTINEL_WORDS[case.dtype]
            guard_hashes = {
                "input_prefix_before_sha256": hashlib.sha256(
                    finalizer.pack_words(
                        [sentinels[0]] * contract.INPUT_GUARD_ELEMENTS, case.dtype
                    )
                ).hexdigest(),
                "input_suffix_before_sha256": hashlib.sha256(
                    finalizer.pack_words(
                        [sentinels[1]] * contract.INPUT_GUARD_ELEMENTS, case.dtype
                    )
                ).hexdigest(),
                "output_prefix_before_sha256": hashlib.sha256(
                    finalizer.pack_words(
                        [sentinels[2]] * contract.OUTPUT_GUARD_ELEMENTS, case.dtype
                    )
                ).hexdigest(),
                "output_suffix_before_sha256": hashlib.sha256(
                    finalizer.pack_words(
                        [sentinels[3]] * contract.OUTPUT_GUARD_ELEMENTS, case.dtype
                    )
                ).hexdigest(),
            }
            guard_hashes.update(
                {
                    name.replace("before", "after"): value
                    for name, value in list(guard_hashes.items())
                }
            )
            for repetition in range(2):
                executions.append(
                    {
                        "case": case.name,
                        "repetition": repetition,
                        "artifact_identity_digest": anchor_records[case.name][
                            "artifact_sha256"
                        ],
                        "fresh_executable": True,
                        "input_unchanged": True,
                        "input_before_sha256": "1" * 64,
                        "input_after_sha256": "1" * 64,
                        "guards_unchanged": True,
                        "input_guard_elements": 4096,
                        "output_guard_elements": 16,
                        "comparison_passed": True,
                        "non_default_stream": True,
                        "current_stream_launch": True,
                        "raw_current_stream": 10,
                        "raw_reference_stream": 11,
                        "distinct_nondefault_reference_stream": True,
                        "reference_stream_synchronized_before_candidate": True,
                        "reference_stream_policy": contract.REFERENCE_STREAM_POLICY,
                        "candidate_stream_policy": contract.CANDIDATE_STREAM_POLICY,
                        "reference_compute_boundary": contract.REFERENCE_COMPUTE_BOUNDARY,
                        "capture_free_before": True,
                        "capture_free_at_launch": True,
                        "external_stream_synchronized": True,
                        "packet_released_after_synchronization": True,
                        "explicit_unload": True,
                        "terminal_state": "Unloaded",
                        "bound_context_before_unload": 7,
                        "bound_context_id_before_unload": 8,
                        "bound_context_after_unload": 0,
                        "bound_context_id_after_unload": 0,
                        **guard_hashes,
                    }
                )
        provisional = {
            "runtime": {
                "case_order": list(contract.CASE_ORDER),
                "case_count": 10,
                "compile_invocations_per_case": 1,
                "repetitions_per_case": 2,
                "module_lifetimes": 20,
                "explicit_packet_releases": 20,
                "explicit_unloads": 20,
                "external_reference_synchronizations": 20,
                "fallback_used": False,
                "forbidden_provider_imports": [],
                "observation": {"context_address": 7, "context_id": 8},
                "artifacts": artifacts,
                "executions": executions,
            }
        }
        return provisional, anchor_records

    def test_canary_input_and_lifecycle_tampering_is_rejected(self) -> None:
        provisional, anchors = self._provisional()
        finalizer.validate_frontend(provisional, anchors)
        mutations = (
            "input_prefix_after_sha256",
            "input_after_sha256",
            "reference_stream_synchronized_before_candidate",
            "capture_free_at_launch",
            "external_stream_synchronized",
            "bound_context_before_unload",
            "raw_reference_stream",
        )
        for name in mutations:
            candidate = copy.deepcopy(provisional)
            execution = candidate["runtime"]["executions"][0]
            if name.endswith("sha256"):
                execution[name] = "0" * 64
            elif name == "bound_context_before_unload":
                execution[name] = 99
            elif name == "raw_reference_stream":
                execution[name] = execution["raw_current_stream"]
            else:
                execution[name] = False
            with self.subTest(name=name), self.assertRaises(finalizer.FinalizeError):
                finalizer.validate_frontend(candidate, anchors)


class PreservationTest(unittest.TestCase):
    def test_cp47_cp48_artifacts_are_unchanged(self) -> None:
        expected = {
            "state/contracts/pypto_fused_pointwise_sm120_v2.json": "d3b16079c811dd2fbe610ba264d81117e8c4a44886b74caaddb684df2d467036",
            "reports/data/pypto-row-reduction-v3-compiler-cubin-records.json": "d06765beaf4fd3ebec3c023b473a904bc704f6ae3a3491b157913ff49e338abb",
            "state/evidence/EV-0061.json": "1cc93511c73e116a1be341a4dacab47a7dbae9270eb7875e99063f5c44b29bab",
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest
            )


if __name__ == "__main__":
    unittest.main()
