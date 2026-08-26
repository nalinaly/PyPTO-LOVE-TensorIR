from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import os
import pathlib
import secrets
import shutil
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
        self.assertEqual(
            {case.name: case.repetition0_policy for case in contract.CASE_SPECS},
            {
                "rank1_fp32_sum_n1": contract.REPETITION0_EXACT_ALL,
                "rank1_fp32_sum_n7": contract.REPETITION0_EXACT_ALL,
                "rank2_bf16_sum_n256_tail": (
                    contract.REPETITION0_BF16_DISCRIMINATOR_THEN_TOLERANCE
                ),
                "rank2_bf16_max_n17": contract.REPETITION0_EXACT_ALL,
                "rank2_fp32_max_n128_tail": contract.REPETITION0_EXACT_ALL,
                "rank2_fp32_max_n96_tail": contract.REPETITION0_EXACT_ALL,
                "rank2_bf16_sum_n129": (
                    contract.REPETITION0_BF16_DISCRIMINATOR_THEN_TOLERANCE
                ),
                "rank3_fp32_sum_n17_tail": (
                    contract.REPETITION0_FINITE_TOLERANCE_ALL
                ),
                "rank3_bf16_max_n17": contract.REPETITION0_EXACT_ALL,
                "rank3_fp32_max_n257_tail": contract.REPETITION0_EXACT_ALL,
            },
        )

    def test_guard_bounds_are_derived_and_sentinels_are_exact(self) -> None:
        self.assertEqual(contract.MAXIMUM_REQUIRED_INPUT_GUARD_ELEMENTS, 3840)
        self.assertEqual(contract.INPUT_GUARD_ELEMENTS, 4096)
        self.assertEqual(contract.MAXIMUM_REQUIRED_OUTPUT_GUARD_ELEMENTS, 15)
        self.assertEqual(contract.OUTPUT_GUARD_ELEMENTS, 16)
        worst = next(
            case
            for case in contract.CASE_SPECS
            if case.name == "rank2_bf16_sum_n256_tail"
        )
        self.assertEqual(contract.required_input_guard_elements(worst), 3840)
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
        self.assertEqual(
            [
                (
                    case.name,
                    contract.required_input_guard_elements(case),
                    contract.required_output_guard_elements(case),
                )
                for case in contract.CASE_SPECS
            ],
            [
                ("rank1_fp32_sum_n1", 0, 0),
                ("rank1_fp32_sum_n7", 0, 0),
                ("rank2_bf16_sum_n256_tail", 3840, 15),
                ("rank2_bf16_max_n17", 0, 0),
                ("rank2_fp32_max_n128_tail", 384, 3),
                ("rank2_fp32_max_n96_tail", 288, 3),
                ("rank2_bf16_sum_n129", 0, 0),
                ("rank3_fp32_sum_n17_tail", 34, 2),
                ("rank3_bf16_max_n17", 0, 0),
                ("rank3_fp32_max_n257_tail", 514, 2),
            ],
        )
        for case in contract.CASE_SPECS:
            projection = contract.lowered_access_projection(case)
            input_addresses = {
                row * case.contraction
                + chunk * contract.contraction_tile(case)
                + column
                for row in range(projection["materialized_rows"])
                for chunk in range(contract.contraction_chunks(case))
                for column in range(contract.contraction_tile(case))
            }
            self.assertEqual(
                max(input_addresses),
                projection["materialized_rows"] * case.contraction - 1,
            )
            input_guard_indices = {
                address - projection["input_logical_elements"]
                for address in input_addresses
                if address >= projection["input_logical_elements"]
            }
            expected_input_max = max(input_guard_indices, default=-1)
            self.assertEqual(expected_input_max, projection["input_guard_max_index"])
            self.assertEqual(
                projection["required_input_guard_elements"],
                expected_input_max + 1,
            )
            self.assertNotIn(
                projection["required_input_guard_elements"], input_guard_indices
            )
            self.assertLessEqual(
                projection["required_input_guard_elements"],
                contract.INPUT_GUARD_ELEMENTS,
            )
            output_guard_indices = {
                row - case.rows
                for row in range(projection["materialized_rows"])
                if row >= case.rows
            }
            expected_output_max = max(output_guard_indices, default=-1)
            self.assertEqual(expected_output_max, projection["output_guard_max_index"])
            self.assertEqual(
                projection["required_output_guard_elements"],
                expected_output_max + 1,
            )
            self.assertNotIn(
                projection["required_output_guard_elements"], output_guard_indices
            )
            self.assertLessEqual(
                projection["required_output_guard_elements"],
                contract.OUTPUT_GUARD_ELEMENTS,
            )

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
        for location in ("top-level", "matrix-policy"):
            candidate = copy.deepcopy(anchors)
            if location == "top-level":
                candidate["unexpected"] = True
            else:
                candidate["matrix_policy"]["unexpected"] = True
            with self.subTest(location=location), mock.patch.object(
                control, "load_canonical", return_value=candidate
            ), self.assertRaises(control.ControlManifestError):
                control.validate_compile_anchors(ROOT, contract)

    def test_anchor_replay_closure_rejects_delete_tamper_and_extra(self) -> None:
        anchors = json.loads(
            (ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH).read_text()
        )
        run = anchors["anchor_runs"][0]
        run_id = run["run_id"]
        source_replay = (
            ROOT / "runs" / run_id / "row-reduction-compile-anchor-replay"
        )
        record = json.loads((ROOT / run["record"]["path"]).read_text())
        for mutation in (
            "delete",
            "tamper",
            "extra",
            "request-metadata",
            "source-metadata",
            "artifact-metadata",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                dir=ROOT / ".cache"
            ) as directory:
                fixture_root = pathlib.Path(directory).resolve()
                candidate_record = copy.deepcopy(record)
                replay = (
                    fixture_root
                    / "runs"
                    / run_id
                    / "row-reduction-compile-anchor-replay"
                )
                shutil.copytree(source_replay, replay, copy_function=shutil.copy2)
                control.validate_anchor_replay_closure(
                    fixture_root, run_id, candidate_record, contract
                )
                first = replay / "compile-request.msgpack"
                if mutation == "delete":
                    first.unlink()
                elif mutation == "tamper":
                    first.chmod(0o600)
                    first.write_bytes(b"tampered")
                    first.chmod(0o444)
                elif mutation == "extra":
                    extra = replay / "unexpected.bin"
                    extra.write_bytes(b"extra")
                    extra.chmod(0o444)
                elif mutation == "request-metadata":
                    candidate_record["anchor_request"]["derived_sha256"] = "0" * 64
                elif mutation == "source-metadata":
                    candidate_record["records"][0]["source_sha256"] = "0" * 64
                else:
                    candidate_record["records"][0]["artifact_bytes"] += 1
                with self.assertRaises(control.ControlManifestError):
                    control.validate_anchor_replay_closure(
                        fixture_root, run_id, candidate_record, contract
                    )

    def test_anchor_semantic_abi_inner_joins_are_fail_closed(self) -> None:
        anchors = json.loads(
            (ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH).read_text()
        )
        record = anchors["records"][7]
        case = contract.CASE_SPECS[7]
        control.validate_semantic_abi_projection(
            record["semantic_abi"], record, case, contract
        )
        mutations = (
            ("grid_abi", "static_dimensions", [99, 1, 1]),
            ("build_spec", "argument_abi_digest", "0" * 64),
            ("artifact_identities", "callable_abi_digest", "0" * 64),
            ("producer", "environment_overrides_enabled", True),
            ("argument_layout", "uniform_signature", True),
            ("lowered_access", "required_input_guard_elements", 0),
        )
        for section, field, value in mutations:
            candidate = copy.deepcopy(record["semantic_abi"])
            candidate[section][field] = value
            with self.subTest(field=field), self.assertRaises(
                control.ControlManifestError
            ):
                control.validate_semantic_abi_projection(
                    candidate, record, case, contract
                )
        contract_source = (
            ROOT / "tools/_pypto_row_reduction_sm120_contract.py"
        ).read_text()
        self.assertIn("build_spec.catalog_provenance is not None", contract_source)

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
            self.assertEqual(cpu[0], expected)
            accumulator = runner.bfloat16_word(1.0)
            increment = runner.word_value(runner.bfloat16_word(2.0**-8), "bfloat16")
            for _ in range(contraction - 1):
                accumulator = runner.bfloat16_word(
                    runner.word_value(accumulator, "bfloat16") + increment
                )
            self.assertEqual(accumulator, 0x3F80)
            self.assertNotEqual(accumulator, expected)

    def test_sum_and_max_special_semantics_are_explicit(self) -> None:
        for dtype in ("float32", "bfloat16"):
            sum_case = next(
                case
                for case in contract.CASE_SPECS
                if case.dtype == dtype
                and case.op_name == "tensor.row_sum"
                and case.rows >= 5
            )
            sum_words = runner.cpu_reference_words(sum_case, 1)
            classes = [
                runner.base._classification(word, dtype)[0]
                for word in sum_words[:5]
            ]
            self.assertEqual(classes, ["zero", "inf", "inf", "nan", "nan"])
            self.assertEqual(sum_words[0], 0)

            max_case = next(
                case
                for case in contract.CASE_SPECS
                if case.dtype == dtype
                and case.op_name == "tensor.row_max"
                and case.rows >= 5
            )
            max_words = runner.cpu_reference_words(max_case, 1)
            max_inputs = [
                runner.encode_word(value, dtype)
                for value in runner.input_values(max_case, 1)[:2]
            ]
            sign = 0x80000000 if dtype == "float32" else 0x8000
            self.assertEqual(max_inputs, [sign, 0])
            self.assertEqual(max_words[0], 0)
            self.assertEqual(max_words[1], sign)
            self.assertEqual(
                runner.base._classification(max_words[2], dtype), ("inf", 0)
            )
            self.assertEqual(
                runner.base._classification(max_words[3], dtype), ("inf", 1)
            )
            self.assertEqual(
                runner.base._classification(max_words[4], dtype)[0], "nan"
            )

    def test_exact_and_tolerance_comparisons_are_separate(self) -> None:
        sum_case = next(
            case for case in contract.CASE_SPECS if case.name == "rank1_fp32_sum_n1"
        )
        word = runner.float32_word(1.0)
        runner.compare_word_sequences(
            sum_case,
            [word],
            [word],
            comparison_modes=(contract.COMPARISON_MODE_EXACT,),
        )
        with self.assertRaises(runner.SmokeError):
            runner.compare_word_sequences(
                sum_case,
                [word + 1],
                [word],
                comparison_modes=(contract.COMPARISON_MODE_EXACT,),
            )
        runner.compare_word_sequences(
            sum_case,
            [word + 1],
            [word],
            comparison_modes=(contract.COMPARISON_MODE_TOLERANCE,),
        )
        negative_zero = runner.float32_word(-0.0)
        positive_zero = runner.float32_word(0.0)
        runner.compare_word_sequences(
            sum_case,
            [negative_zero],
            [negative_zero],
            comparison_modes=(contract.COMPARISON_MODE_SPECIAL,),
        )
        with self.assertRaises(runner.SmokeError):
            runner.compare_word_sequences(
                sum_case,
                [positive_zero],
                [negative_zero],
                comparison_modes=(contract.COMPARISON_MODE_SPECIAL,),
            )
        self.assertEqual(contract.FP32_SUM_MAX_ULP, 16)

    def test_tolerance_boundaries_and_subnormals_are_fail_closed(self) -> None:
        case = next(
            case for case in contract.CASE_SPECS if case.name == "rank1_fp32_sum_n1"
        )
        near_two = runner.float32_word(1.875)
        runner.compare_word_sequences(
            case,
            [near_two + contract.FP32_SUM_MAX_ULP],
            [near_two],
            comparison_modes=(contract.COMPARISON_MODE_TOLERANCE,),
        )
        with self.assertRaises(runner.SmokeError):
            runner.compare_word_sequences(
                case,
                [near_two + contract.FP32_SUM_MAX_ULP + 1],
                [near_two],
                comparison_modes=(contract.COMPARISON_MODE_TOLERANCE,),
            )
        subnormal = 1_000
        runner.compare_word_sequences(
            case,
            [subnormal],
            [subnormal],
            comparison_modes=(contract.COMPARISON_MODE_TOLERANCE,),
        )
        with self.assertRaises(runner.SmokeError):
            runner.compare_word_sequences(
                case,
                [subnormal + 1],
                [subnormal],
                comparison_modes=(contract.COMPARISON_MODE_TOLERANCE,),
            )
        max_probe = contract.CaseSpec(
            "subnormal_max_probe",
            "float32",
            (1, 1),
            "tensor.row_max",
            1,
        )
        runner.compare_word_sequences(
            max_probe,
            [subnormal],
            [subnormal],
            comparison_modes=(contract.COMPARISON_MODE_EXACT,),
        )
        with self.assertRaises(runner.SmokeError):
            runner.compare_word_sequences(
                max_probe,
                [subnormal + 1],
                [subnormal],
                comparison_modes=(contract.COMPARISON_MODE_EXACT,),
            )
        with self.assertRaises(runner.SmokeError):
            runner.compare_word_sequences(
                max_probe,
                [subnormal],
                [subnormal],
                comparison_modes=(contract.COMPARISON_MODE_SPECIAL,),
            )
        finalizer.compare(
            case,
            [subnormal],
            [subnormal],
            modes=(contract.COMPARISON_MODE_TOLERANCE,),
        )
        with self.assertRaises(finalizer.FinalizeError):
            finalizer.compare(
                case,
                [subnormal + 1],
                [subnormal],
                modes=(contract.COMPARISON_MODE_TOLERANCE,),
            )
        finalizer.compare(
            max_probe,
            [subnormal],
            [subnormal],
            modes=(contract.COMPARISON_MODE_EXACT,),
        )
        with self.assertRaises(finalizer.FinalizeError):
            finalizer.compare(
                max_probe,
                [subnormal + 1],
                [subnormal],
                modes=(contract.COMPARISON_MODE_EXACT,),
            )

    def test_real_fp32_and_bf16_tolerance_rows_accept_one_ulp_only(self) -> None:
        for dtype in ("float32", "bfloat16"):
            case = next(
                case
                for case in contract.CASE_SPECS
                if case.dtype == dtype and case.tolerance_output_indices(0)
            )
            reference = runner.cpu_reference_words(case, 0)
            index = case.tolerance_output_indices(0)[0]
            actual = list(reference)
            actual[index] += 1
            runner.compare_word_sequences(
                case,
                actual,
                reference,
                comparison_modes=case.output_comparison_modes(0),
            )
            actual[index] += case.max_ulp
            with self.assertRaises(runner.SmokeError):
                runner.compare_word_sequences(
                    case,
                    actual,
                    reference,
                    comparison_modes=case.output_comparison_modes(0),
                )

    def test_finalizer_rejects_misaligned_logical_bytes_as_finalize_error(self) -> None:
        for dtype, raw in (("float32", b"\x00"), ("bfloat16", b"\x00")):
            with self.subTest(dtype=dtype), self.assertRaises(
                finalizer.FinalizeError
            ):
                finalizer.unpack_words(raw, dtype)

    def test_fp32_and_bf16_finite_tolerance_rows_are_reachable(self) -> None:
        reached = {
            case.dtype
            for case in contract.CASE_SPECS
            if case.tolerance_output_indices(0)
            and all(
                runner.base._classification(word, case.dtype)[0] == "finite"
                for index, word in enumerate(runner.cpu_reference_words(case, 0))
                if index in case.tolerance_output_indices(0)
            )
        }
        self.assertEqual(reached, {"float32", "bfloat16"})
        for case in contract.CASE_SPECS:
            self.assertEqual(
                case.exact_output_indices(0),
                finalizer.exact_output_indices(case, 0),
            )
            self.assertEqual(
                case.tolerance_output_indices(0),
                finalizer.tolerance_output_indices(case, 0),
            )
            self.assertEqual(
                case.special_output_indices(1),
                finalizer.special_output_indices(case, 1),
            )
            self.assertEqual(
                case.output_comparison_modes(0),
                finalizer.comparison_modes(case, 0),
            )
            self.assertEqual(
                case.output_comparison_modes(1),
                finalizer.comparison_modes(case, 1),
            )

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
                "repetition0_policy",
                "comparison_modes",
                "exact_output_indices",
                "tolerance_output_indices",
                "special_output_indices",
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
            "repetition0_policy",
            "comparison_modes",
            "exact_output_indices",
            "tolerance_output_indices",
            "special_output_indices",
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
    def _run_mocked_controller(
        self, scenario: str
    ) -> tuple[int, list[str], dict[str, object]]:
        events: list[str] = []
        writes: dict[str, list[dict[str, object]]] = {}
        transaction: dict[str, object] = {"writes": writes}
        report = {
            "ok": True,
            "protected_gpu_smoke_waiver_applied": True,
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

        process = Process()
        metadata = {
            "run_id": "fixture",
            "pgid": 1234,
            "start_ticks": 55,
            "command": contract.fixed_child_command(ROOT),
            "gpu_smoke": {},
            "preflight": {"path": "fixture-preflight", "sha256": "a" * 64},
            "status": "running",
        }

        def preflight_side_effect(**kwargs):
            description = kwargs["description"]
            events.append(
                "initial-preflight"
                if description.startswith("initial")
                else "action-preflight"
            )
            return 0, copy.deepcopy(report), b"preflight"

        def watchdog_side_effect(*_args, **_kwargs):
            events.append("watchdog")
            if scenario == "watchdog-abort":
                process.returncode = 137
                return 137, True
            process.returncode = 0
            return 0, False

        def survivor_side_effect(_process, _metadata):
            events.append("survivor-check")
            return scenario != "survivor-failure"

        input_checks = 0

        def validate_inputs_side_effect() -> None:
            nonlocal input_checks
            input_checks += 1
            events.append("initial-inputs" if input_checks == 1 else "gate-inputs")

        audit_checks = 0

        def audit_side_effect(_metadata):
            nonlocal audit_checks
            audit_checks += 1
            if audit_checks == 1:
                events.append("pre-release-audit")
                owned = [777] if scenario == "gate-failure" else []
                return {"audit": "pre-release", "owned_nvidia_compute_pids": owned}, None
            events.append("post-exit-audit")
            return {"audit": "post-exit", "owned_nvidia_compute_pids": []}, None

        def terminate_side_effect(_process, _metadata, **_kwargs):
            events.append("terminate-owned")
            process.returncode = 75
            return 75

        def atomic_side_effect(path, value):
            events.append(f"metadata:{path.name}")
            writes.setdefault(path.name, []).append(copy.deepcopy(value))

        def popen_side_effect(*args, **kwargs):
            events.append("popen")
            transaction["popen_args"] = args
            transaction["popen_kwargs"] = kwargs
            return process

        def build_metadata_side_effect(*_args, **kwargs):
            events.append("build-metadata")
            value = copy.deepcopy(metadata)
            value["preflight"] = {
                "path": kwargs["preflight_report_path"],
                "sha256": kwargs["preflight_report_sha256"],
            }
            return value

        fake_flags = SimpleNamespace(
            ignore_environment=1, no_site=1, dont_write_bytecode=1
        )
        fake_disk = SimpleNamespace(free=1 << 50)
        argv = [
            "run_pypto_row_reduction_sm120_isolated.py",
            "--allow-protected-zero-nvidia-gpu-smoke",
            "--run-id-file",
            str(ROOT / "runs/mock-row-run-id.json"),
        ]
        patches = (
            mock.patch.object(
                controller.control,
                "validate_control_manifest",
                side_effect=lambda _root: events.append("manifest") or {"row": True},
            ),
            mock.patch.object(
                controller.isolation,
                "validate_exact_nvidia_smoke_command",
                side_effect=lambda _command: events.append("command"),
            ),
            mock.patch.object(
                controller,
                "acquire_shared_environment_lease",
                side_effect=lambda: events.append("lease") or Lease(),
            ),
            mock.patch.object(
                controller.isolation,
                "validate_exact_nvidia_smoke_inputs",
                side_effect=validate_inputs_side_effect,
            ),
            mock.patch.object(
                controller.base_controller,
                "run_preflight",
                side_effect=preflight_side_effect,
            ),
            mock.patch.object(controller.base_controller, "_write_run_id"),
            mock.patch.object(controller.pathlib.Path, "mkdir"),
            mock.patch.object(
                controller.isolation,
                "isolated_environment",
                side_effect=lambda *_args, **_kwargs: events.append("environment")
                or {"SYNTHETIC_BASE_ENV": "fixed"},
            ),
            mock.patch.object(
                controller.isolation, "environment_lock_markers", return_value={}
            ),
            mock.patch.object(
                controller.isolation,
                "atomic_json",
                side_effect=atomic_side_effect,
            ),
            mock.patch.object(
                controller.isolation,
                "sha256_file",
                side_effect=lambda path: (
                    "c" * 64 if path.name == "gpu-smoke-gate.json" else "a" * 64
                ),
            ),
            mock.patch.object(controller.shutil, "disk_usage", return_value=fake_disk),
            mock.patch.object(
                controller.subprocess,
                "Popen",
                side_effect=popen_side_effect,
            ),
            mock.patch.object(
                controller.isolation,
                "build_run_metadata",
                side_effect=build_metadata_side_effect,
            ),
            mock.patch.object(
                controller.preflight,
                "static_torch_identity",
                side_effect=lambda: events.append("static-identity")
                or {"static": True},
            ),
            mock.patch.object(
                controller.preflight,
                "policy_document",
                side_effect=lambda: events.append("admission-policy")
                or {"policy": 2},
            ),
            mock.patch.object(
                controller.stop_run,
                "verify",
                side_effect=lambda _metadata: events.append("verify-owned-process"),
            ),
            mock.patch.object(
                controller.isolation,
                "terminate_owned_process",
                side_effect=terminate_side_effect,
            ),
            mock.patch.object(
                controller.isolation,
                "wait_with_gpu_smoke_watchdog",
                side_effect=watchdog_side_effect,
            ),
            mock.patch.object(
                controller.isolation,
                "audit_gpu_smoke_runtime_state",
                side_effect=audit_side_effect,
            ),
            mock.patch.object(
                controller,
                "enforce_no_survivors",
                side_effect=survivor_side_effect,
            ),
            mock.patch.object(controller.stop_run, "process_group_members", return_value=[]),
            mock.patch.object(controller.signal, "getsignal", return_value=None),
            mock.patch.object(controller.signal, "signal"),
            mock.patch.object(controller.signal, "pthread_sigmask", return_value=set()),
            mock.patch.object(controller, "print", create=True),
            mock.patch.object(controller.sys, "flags", fake_flags),
            mock.patch.object(controller.sys, "argv", argv),
        )
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            return controller.main(), events, transaction

    def test_controller_main_executes_full_transaction_in_order(self) -> None:
        code, events, transaction = self._run_mocked_controller("success")
        self.assertEqual(code, 0)
        expected = (
            "manifest",
            "command",
            "lease",
            "initial-inputs",
            "initial-preflight",
            "metadata:initial-preflight.json",
            "environment",
            "action-preflight",
            "metadata:preflight.json",
            "popen",
            "build-metadata",
            "metadata:process.json",
            "gate-inputs",
            "static-identity",
            "verify-owned-process",
            "pre-release-audit",
            "admission-policy",
            "metadata:gpu-smoke-gate.json",
            "metadata:process.json",
            "metadata:gpu-smoke-start-barrier.json",
            "watchdog",
            "post-exit-audit",
            "survivor-check",
            "metadata:process.json",
            "lease-close",
        )
        self.assertEqual(tuple(events), expected)
        writes = transaction["writes"]
        gate = writes["gpu-smoke-gate.json"][0]
        barrier = writes["gpu-smoke-start-barrier.json"][0]
        final_metadata = writes["process.json"][-1]
        self.assertEqual(gate["schema"], 2)
        self.assertEqual(gate["control_manifest"], {"row": True})
        self.assertEqual(gate["runtime_isolation"]["audit"], "pre-release")
        self.assertEqual(gate["initial_preflight"], final_metadata["initial_preflight"])
        self.assertEqual(gate["preflight"], final_metadata["preflight"])
        for name in ("schema", "run_id", "pid", "pgid", "start_ticks"):
            self.assertEqual(barrier[name], gate[name])
        self.assertEqual(barrier["gate_sha256"], "c" * 64)
        self.assertEqual(final_metadata["gpu_smoke"]["gate_sha256"], "c" * 64)
        self.assertEqual(
            final_metadata["gpu_smoke"]["start_barrier_sha256"],
            hashlib.sha256(controller.isolation.canonical_json_bytes(barrier)).hexdigest(),
        )
        self.assertIs(transaction["popen_kwargs"]["start_new_session"], True)
        self.assertEqual(transaction["popen_kwargs"]["pass_fds"], (9,))
        self.assertEqual(
            transaction["popen_args"], (contract.fixed_child_command(ROOT),)
        )
        self.assertEqual(transaction["popen_kwargs"]["cwd"], ROOT)
        self.assertTrue(callable(transaction["popen_kwargs"]["preexec_fn"]))
        environment = transaction["popen_kwargs"]["env"]
        run_dir = pathlib.Path(environment["PYPTO_PREFLIGHT_REPORT_PATH"]).parent
        self.assertEqual(run_dir.parent, ROOT / "runs")
        self.assertRegex(run_dir.name, control.RUN_ID_PATTERN)
        self.assertEqual(
            environment,
            {
                "SYNTHETIC_BASE_ENV": "fixed",
                "PYPTO_PROTECTED_CPU_ONLY_COEXISTENCE_REQUESTED": "0",
                "PYPTO_PROTECTED_ACTIVITY_WAIVER_APPLIED": "0",
                "PYPTO_PROTECTED_ZERO_NVIDIA_GPU_SMOKE_REQUESTED": "1",
                "PYPTO_PROTECTED_GPU_SMOKE_WAIVER_APPLIED": "1",
                "PYPTO_GPU_SMOKE_START_BARRIER": str(
                    run_dir / "gpu-smoke-start-barrier.json"
                ),
                "PYPTO_PREFLIGHT_REPORT_PATH": str(run_dir / "preflight.json"),
                "PYPTO_PREFLIGHT_REPORT_SHA256": "a" * 64,
                "PYPTO_INITIAL_PREFLIGHT_REPORT_PATH": str(
                    run_dir / "initial-preflight.json"
                ),
                "PYPTO_INITIAL_PREFLIGHT_REPORT_SHA256": "a" * 64,
                "PYPTO_RUN_MODE": "gpu-smoke",
            },
        )
        self.assertEqual(final_metadata["status"], "exited")
        self.assertEqual(final_metadata["return_code"], 0)

    def test_controller_gate_watchdog_and_survivor_failures_are_fail_closed(self) -> None:
        for scenario in ("gate-failure", "watchdog-abort", "survivor-failure"):
            with self.subTest(scenario=scenario):
                code, events, _transaction = self._run_mocked_controller(scenario)
                self.assertEqual(code, 75)
                self.assertIn("lease-close", events)
                if scenario == "gate-failure":
                    self.assertNotIn("watchdog", events)
                else:
                    self.assertIn("post-exit-audit", events)

    def test_survivor_cleanup_uses_verified_owned_process(self) -> None:
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
            self.assertFalse(controller.enforce_no_survivors(process, metadata))
        terminate.assert_called_once_with(process, metadata, wait_seconds=5)
        self.assertEqual(metadata["surviving_group_pids"], [101])

    def test_direct_child_admission_precedes_torch(self) -> None:
        source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text()
        run = source[source.index("def run_smoke()") :]
        markers = (
            "workspace_from_environment()",
            "wait_for_start_barrier(workspace, run_id)",
            "load_contract_and_child_gate(",
            "load_anchors(workspace, contract_module)",
            "import torch",
            "torch.cuda.init()",
            "sha256_file(dso) != contract_module.PYPTO_DSO_SHA256",
            "pypto = base.bootstrap_exact_pypto(workspace, dso.parent)",
            "from pypto import compiler",
            "from pypto.runtime import nvidia as runtime",
            "validate_exact_module_origins(",
            "not info.compiled",
            "compiler.compile_structured_strict(",
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

    def test_exact_pypto_module_origins_and_preexisting_modules_are_rejected(
        self,
    ) -> None:
        package = ROOT / "projects/pypto/python/pypto"
        dso = ROOT / contract.PYPTO_DSO_RELATIVE_PATH
        modules = {
            "pypto": ModuleType("pypto"),
            "pypto.pypto_core": ModuleType("pypto.pypto_core"),
            "pypto.compiler": ModuleType("pypto.compiler"),
            "pypto.runtime.nvidia": ModuleType("pypto.runtime.nvidia"),
            "torch": ModuleType("torch"),
        }
        modules["pypto"].__file__ = str(package / "__init__.py")
        modules["pypto"].__path__ = [str(package)]
        modules["pypto.pypto_core"].__file__ = str(dso)
        modules["pypto.compiler"].__file__ = str(package / "compiler/__init__.py")
        modules["pypto.runtime.nvidia"].__file__ = str(
            package / "runtime/nvidia.py"
        )
        modules["torch"].__file__ = str(
            ROOT
            / "envs/pypto-nvidia/lib/python3.14/site-packages/torch/__init__.py"
        )
        with mock.patch.dict(sys.modules, modules, clear=False):
            runner.validate_exact_module_origins(
                ROOT,
                dso,
                modules["pypto"],
                modules["pypto.compiler"],
                modules["pypto.runtime.nvidia"],
                modules["torch"],
            )
            modules["pypto.compiler"].__file__ = str(package / "ambient.py")
            with self.assertRaises(runner.SmokeError):
                runner.validate_exact_module_origins(
                    ROOT,
                    dso,
                    modules["pypto"],
                    modules["pypto.compiler"],
                    modules["pypto.runtime.nvidia"],
                    modules["torch"],
                )
        with mock.patch.dict(sys.modules, {"pypto.ambient": ModuleType("ambient")}):
            with self.assertRaises(runner.SmokeError):
                runner.reject_preexisting_pypto_modules()

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

    def test_reused_helper_globals_and_injected_dependencies_are_frozen(self) -> None:
        for helper in (
            runner.base.bootstrap_exact_pypto,
            runner.base.validate_pypto_python_source,
            runner.base.tensor_argument,
            runner.base.target_traits_document,
        ):
            self.assertIs(helper.__globals__, runner.base.__dict__)
        for helper in (
            finalizer.base.validate_preflight,
            finalizer.base.validate_audit,
            finalizer.base.validate_child_gate,
            finalizer.base.validate_child_parent_joins,
            finalizer.base.validate_runtime_identity,
        ):
            self.assertIs(helper.__globals__, finalizer.base.__dict__)
        self.assertIs(
            controller.base_controller.run_preflight.__globals__,
            controller.base_controller.__dict__,
        )
        self.assertIs(
            controller.base_controller._write_run_id.__globals__,
            controller.base_controller.__dict__,
        )
        self.assertIs(controller.isolation.preflight_tool, controller.preflight)
        self.assertIs(controller.isolation.stop_run, controller.stop_run)
        self.assertIs(controller.isolation.nvidia_smoke_contract, controller.contract)
        self.assertIs(controller.isolation.nvidia_smoke_control, controller.control)

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
        hir_programs = [
            {
                "case": case.name,
                "bytes": anchor_records[case.name]["hir_bytes"],
                "sha256": anchor_records[case.name]["hir_sha256"],
                "canonical_roundtrip": True,
                "input_count": 1,
                "operator": case.op_name,
            }
            for case in contract.CASE_SPECS
        ]
        artifacts = [
            {
                "case": case.name,
                "hir_bytes": anchor_records[case.name]["hir_bytes"],
                "hir_sha256": anchor_records[case.name]["hir_sha256"],
                "source_ir_bytes": anchor_records[case.name]["source_bytes"],
                "source_ir_sha256": anchor_records[case.name]["source_sha256"],
                "source_ir_digest": anchor_records[case.name]["source_ir_digest"],
                "build_spec_bytes": anchor_records[case.name]["build_spec_bytes"],
                "build_spec_identity_digest": anchor_records[case.name][
                    "build_spec_sha256"
                ],
                "artifact_bytes": anchor_records[case.name]["artifact_bytes"],
                "artifact_identity_digest": anchor_records[case.name][
                    "artifact_sha256"
                ],
                "device_code_bytes": anchor_records[case.name]["device_code_bytes"],
                "device_code_sha256": anchor_records[case.name]["device_code_sha256"],
                "grid": list(case.grid),
                "row_tile": case.row_tile,
                "semantic_abi": anchor_records[case.name]["semantic_abi"],
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
                        "expected_logical_bytes_sha256": "2" * 64,
                        "actual_logical_bytes_sha256": "2" * 64,
                        "cpu_reference_bytes_sha256": "2" * 64,
                        "frozen_numerical_oracle_passed": True,
                        "fresh_executable": True,
                        "input_unchanged": True,
                        "input_before_sha256": "1" * 64,
                        "input_after_sha256": "1" * 64,
                        "guards_unchanged": True,
                        "input_guard_elements": 4096,
                        "output_guard_elements": 16,
                        "comparison_passed": True,
                        "comparison": {
                            **finalizer.comparison_metadata(case, repetition),
                            "candidate_vs_torch": {},
                            "candidate_vs_cpu": {},
                            "torch_vs_cpu": {},
                        },
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
                "non_default_current_stream": True,
                "distinct_nondefault_reference_stream": True,
                "reference_compute_outside_candidate_coverage": True,
                "external_reference_synchronizations": 20,
                "external_synchronization": True,
                "fallback_used": False,
                "forbidden_provider_imports": [],
                "observation": {"context_address": 7, "context_id": 8},
                "compile_request": {
                    "byte_identity_digest": (
                        contract.EXPECTED_COMPILE_REQUEST_BYTE_IDENTITY_DIGEST
                    ),
                    "loader_compatibility_input_digest": (
                        contract.EXPECTED_LOADER_COMPATIBILITY_INPUT_DIGEST
                    ),
                    "device_autotune_identity_digest": (
                        contract.EXPECTED_DEVICE_AUTOTUNE_IDENTITY_DIGEST
                    ),
                },
                "hir_programs": hir_programs,
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
            "frozen_numerical_oracle_passed",
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

    def test_hir_artifact_execution_schemas_and_semantic_fields_are_rejected(self) -> None:
        provisional, anchors = self._provisional()
        mutations = (
            ("hir", "bytes", 1),
            ("hir", "operator", "tensor.add"),
            ("artifact", "grid", [99, 1, 1]),
            ("artifact", "row_tile", 99),
            ("artifact", "device_code_bytes", 1),
            ("artifact", "source_ir_sha256", "0" * 64),
            ("semantic", "runtime_kernel_name", "ambient"),
            ("comparison", "repetition0_policy", "exact-all"),
            ("comparison", "comparison_modes", ["ambient"]),
            ("comparison", "special_output_indices", [0]),
        )
        for location, field, value in mutations:
            candidate = copy.deepcopy(provisional)
            if location == "hir":
                target = candidate["runtime"]["hir_programs"][0]
            elif location == "artifact":
                target = candidate["runtime"]["artifacts"][0]
            elif location == "semantic":
                target = candidate["runtime"]["artifacts"][0]["semantic_abi"]
            else:
                target = candidate["runtime"]["executions"][12]["comparison"]
            target[field] = value
            with self.subTest(location=location, field=field), self.assertRaises(
                finalizer.FinalizeError
            ):
                finalizer.validate_frontend(candidate, anchors)
        for collection in ("hir_programs", "artifacts", "executions"):
            candidate = copy.deepcopy(provisional)
            candidate["runtime"][collection][0]["unexpected"] = True
            with self.subTest(extra=collection), self.assertRaises(
                finalizer.FinalizeError
            ):
                finalizer.validate_frontend(candidate, anchors)


class SyntheticFinalizerTransactionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor_manifest = json.loads(
            (ROOT / contract.COMPILE_ANCHORS_RELATIVE_PATH).read_text()
        )
        self.anchor_records = {
            record["case"]: record for record in self.anchor_manifest["records"]
        }
        self.control_identity = {
            "manifest_path": "state/contracts/synthetic-row-controls.json",
            "manifest_bytes": 1,
            "manifest_sha256": "f" * 64,
            "implementation_commit": "1" * 40,
            "implementation_tree": "2" * 40,
            "current_head": "3" * 40,
            "current_tree": "4" * 40,
            "root_clean": True,
            "base_admission": {"manifest_sha256": "e" * 64},
            "compile_anchors": {"sha256": contract.COMPILE_ANCHORS_SHA256},
            "files": [],
        }
        self.pypto_identity = {
            "head": contract.PYPTO_HEAD,
            "tree": contract.PYPTO_TREE,
            "clean": True,
        }

    @staticmethod
    def _write_json(path: pathlib.Path, value: object, mode: int) -> str:
        raw = finalizer.base.base.canonical_json(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.chmod(0o600)
        path.write_bytes(raw)
        path.chmod(mode)
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _integrity_record(path: pathlib.Path) -> dict[str, object]:
        resolved = path.resolve(strict=True)
        return {
            "path": resolved.relative_to(ROOT).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": finalizer.sha256_file(resolved),
        }

    @contextlib.contextmanager
    def _contract_patches(self, fixture: dict[str, object]):
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    finalizer.contract,
                    "PYPTO_DSO_RELATIVE_PATH",
                    fixture["dso_relative"],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    finalizer.contract,
                    "PYTHON_REAL_RELATIVE_PATH",
                    fixture["python_relative"],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    finalizer.contract,
                    "CUDA_RUNTIME_RELATIVE_PATH",
                    fixture["cuda_runtime_relative"],
                )
            )
            for owner in (finalizer.base.contract,):
                stack.enter_context(
                    mock.patch.object(
                        owner,
                        "PYTHON_REAL_RELATIVE_PATH",
                        fixture["python_relative"],
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        owner,
                        "CUDA_RUNTIME_RELATIVE_PATH",
                        fixture["cuda_runtime_relative"],
                    )
                )
            stack.enter_context(
                mock.patch.object(
                    finalizer.contract,
                    "FINAL_REPORT_DIRECTORY",
                    fixture["final_report_directory"],
                )
            )
            for prefix in ("PYPTO_DSO", "PYTHON", "CUDA_RUNTIME"):
                size, digest = fixture["critical_pins"][prefix]
                stack.enter_context(
                    mock.patch.object(
                        finalizer.contract, f"{prefix}_SIZE", size
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        finalizer.contract, f"{prefix}_SHA256", digest
                    )
                )
                if prefix in {"PYTHON", "CUDA_RUNTIME"}:
                    stack.enter_context(
                        mock.patch.object(
                            finalizer.base.contract,
                            f"{prefix}_SIZE",
                            size,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            finalizer.base.contract,
                            f"{prefix}_SHA256",
                            digest,
                        )
                    )
            yield

    @staticmethod
    def _audit(gpu: dict[str, object], owned: list[int]) -> dict[str, object]:
        return {
            "owned_nvidia_compute_pids": owned,
            "external_nvidia_compute_pids": [],
            "protected_nvidia_compute_pids": [],
            "protected_nvidia_runtime_mapping_pids": [],
            "unreadable_protected_maps": [],
            "protected_heavy_pids": [],
            "protected_cpu_lane_authorized": False,
            "free_memory_mib": int(gpu["memory_mib"]) - int(gpu["used_mib"]),
            "gpu": gpu,
        }

    def _semantic_stdout(self) -> str:
        records = [
            {
                "case": case.name,
                "hir_sha256": self.anchor_records[case.name]["hir_sha256"],
                "source_sha256": self.anchor_records[case.name]["source_sha256"],
                "source_ir_digest": self.anchor_records[case.name][
                    "source_ir_digest"
                ],
                "build_spec_sha256": self.anchor_records[case.name][
                    "build_spec_sha256"
                ],
                "artifact_sha256": self.anchor_records[case.name][
                    "artifact_sha256"
                ],
                "device_code_sha256": self.anchor_records[case.name][
                    "device_code_sha256"
                ],
                "semantic_abi": self.anchor_records[case.name]["semantic_abi"],
            }
            for case in contract.CASE_SPECS
        ]
        return json.dumps(
            {
                "torch_cuda_initialized": False,
                "compile_request": {
                    "byte_identity_digest": (
                        contract.EXPECTED_COMPILE_REQUEST_BYTE_IDENTITY_DIGEST
                    ),
                    "loader_compatibility_input_digest": (
                        contract.EXPECTED_LOADER_COMPATIBILITY_INPUT_DIGEST
                    ),
                    "device_autotune_identity_digest": (
                        contract.EXPECTED_DEVICE_AUTOTUNE_IDENTITY_DIGEST
                    ),
                },
                "records": records,
            },
            sort_keys=True,
        ) + "\n"

    def _build_fixture(self) -> dict[str, object]:
        run_id = (
            f"pypto-20990101T000000Z-{os.getpid()}-{secrets.token_hex(3)}"
        )
        run_dir = ROOT / "runs" / run_id
        run_dir.mkdir()
        self.addCleanup(shutil.rmtree, run_dir, True)
        replay = run_dir / contract.REPLAY_DIRECTORY_NAME
        replay.mkdir(mode=0o700)
        fixture_inputs = run_dir / "synthetic-finalizer-inputs"
        fixture_inputs.mkdir()
        python_path = fixture_inputs / "python3.14"
        dso_path = fixture_inputs / "pypto_core.so"
        cuda_runtime_path = fixture_inputs / "libcudart.so.13"
        python_path.write_bytes(b"synthetic-cpython-for-cpu-replay\n")
        python_path.chmod(0o755)
        dso_path.write_bytes(b"synthetic-pypto-dso\n")
        cuda_runtime_path.write_bytes(b"synthetic-libcudart\n")
        fixture: dict[str, object] = {
            "run_id": run_id,
            "run_dir": run_dir,
            "replay": replay,
            "python_path": python_path,
            "dso_path": dso_path,
            "cuda_runtime_path": cuda_runtime_path,
            "python_relative": python_path.relative_to(ROOT),
            "dso_relative": dso_path.relative_to(ROOT),
            "cuda_runtime_relative": cuda_runtime_path.relative_to(ROOT),
            "critical_pins": {
                "PYPTO_DSO": (
                    dso_path.stat().st_size,
                    finalizer.sha256_file(dso_path),
                ),
                "PYTHON": (
                    python_path.stat().st_size,
                    finalizer.sha256_file(python_path),
                ),
                "CUDA_RUNTIME": (
                    cuda_runtime_path.stat().st_size,
                    finalizer.sha256_file(cuda_runtime_path),
                ),
            },
            "final_report_directory": pathlib.Path("runs") / run_id / "final",
        }
        with self._contract_patches(fixture):
            partial, _anchors = FinalizerAdversarialTest()._provisional()
            runtime = partial["runtime"]
            replay_files: list[dict[str, object]] = []

            def record_file(path: pathlib.Path) -> None:
                path.chmod(0o444)
                replay_files.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": finalizer.sha256_file(path),
                    }
                )

            compile_names = ["compile-request.msgpack"]
            for case in contract.CASE_SPECS:
                compile_names.extend(
                    [
                        f"{case.name}.hir.msgpack",
                        f"{case.name}.source.mlir",
                        f"{case.name}.build-spec.msgpack",
                        f"{case.name}.artifact.msgpack",
                        f"{case.name}.cubin",
                    ]
                )
            anchor_run_id = self.anchor_manifest["anchor_runs"][0]["run_id"]
            retained = (
                ROOT
                / "runs"
                / anchor_run_id
                / "row-reduction-compile-anchor-replay"
            )
            for name in compile_names:
                destination = replay / name
                shutil.copy2(retained / name, destination)
                record_file(destination)

            execution_index = 0
            for case in contract.CASE_SPECS:
                for repetition in range(contract.REPETITIONS):
                    execution = runtime["executions"][execution_index]
                    execution_index += 1
                    input_raw = finalizer.pack_words(
                        [
                            finalizer.encode_word(value, case.dtype)
                            for value in finalizer.input_values(case, repetition)
                        ],
                        case.dtype,
                    )
                    cpu_words = finalizer.cpu_reference(case, repetition)
                    output_raw = finalizer.pack_words(cpu_words, case.dtype)
                    for suffix, payload in (
                        ("input", input_raw),
                        ("reference", output_raw),
                        ("actual", output_raw),
                        ("cpu-reference", output_raw),
                    ):
                        path = replay / f"{case.name}.r{repetition}.{suffix}.bin"
                        path.write_bytes(payload)
                        record_file(path)
                    input_sha = hashlib.sha256(input_raw).hexdigest()
                    output_sha = hashlib.sha256(output_raw).hexdigest()
                    metrics = finalizer.compare(
                        case,
                        cpu_words,
                        cpu_words,
                        modes=finalizer.comparison_modes(case, repetition),
                    )
                    execution.update(
                        {
                            "input_before_sha256": input_sha,
                            "input_after_sha256": input_sha,
                            "expected_logical_bytes_sha256": output_sha,
                            "actual_logical_bytes_sha256": output_sha,
                            "cpu_reference_bytes_sha256": output_sha,
                            "comparison": {
                                **finalizer.comparison_metadata(case, repetition),
                                "candidate_vs_torch": metrics,
                                "candidate_vs_cpu": metrics,
                                "torch_vs_cpu": metrics,
                            },
                        }
                    )

            base_contract = finalizer.base.contract
            gpu = {
                "name": base_contract.EXPECTED_DEVICE_NAME,
                "compute_capability": "12.0",
                "memory_mib": "24576",
                "used_mib": "1024",
                "driver": base_contract.EXPECTED_DRIVER_RELEASE,
            }
            static_identity = {
                "source": "static ENVIRONMENT.lock and selected-prefix file audit",
                "environment_lock_sha256": base_contract.ENVIRONMENT_LOCK_SHA256,
                "version": base_contract.EXPECTED_TORCH_VERSION,
                "git_version": base_contract.EXPECTED_TORCH_GIT,
                "cuda": base_contract.EXPECTED_TORCH_CUDA,
                "hip": None,
                "python_executable": str(python_path.resolve(strict=True)),
                "libcudart_path": str(cuda_runtime_path.resolve(strict=True)),
                "libcudart_size": cuda_runtime_path.stat().st_size,
                "libcudart_sha256": finalizer.sha256_file(cuda_runtime_path),
                "libcudart_record_owned": True,
                "nvidia_runtime_mappings": [],
                "cuda_initialized": False,
                "forbidden_dsos": [],
            }
            admission_policy = finalizer.base.preflight_adapter.policy_document()
            memory_floor = finalizer.base.expected_floor(False)
            preflight = {
                "coexistence_policy_version": 1,
                "cwd": str(ROOT),
                "failures": [],
                "gpu": gpu,
                "gpu_smoke_admission_policy": admission_policy,
                "gpu_smoke_free_memory_floor_mib": base_contract.GPU_FREE_MEMORY_FLOOR_MIB,
                "gpu_smoke_policy_version": 2,
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
                "protected_gpu_smoke_waiver_applied": False,
                "protected_heavy_processes": [],
                "protected_nvidia_compute_pids": [],
                "protected_nvidia_runtime_mapping_pids": [],
                "protected_processes": [],
                "protected_zero_nvidia_gpu_smoke_requested": False,
                "torch": static_identity,
                "unreadable_protected_maps": [],
                "workspace": str(ROOT),
                "workspace_processes": [],
            }
            initial_path = run_dir / "initial-preflight.json"
            preflight_path = run_dir / "preflight.json"
            initial_sha = self._write_json(initial_path, preflight, 0o600)
            preflight_sha = self._write_json(preflight_path, preflight, 0o600)
            command = finalizer.contract.fixed_child_command(ROOT)
            identity = {"schema": 2, "run_id": run_id, "pid": 99, "pgid": 99, "start_ticks": 990}
            pre_release = self._audit(gpu, [])
            gate_path = run_dir / "gpu-smoke-gate.json"
            gate = {
                **identity,
                "command": command,
                "initial_preflight": {"path": str(initial_path), "sha256": initial_sha},
                "preflight": {"path": str(preflight_path), "sha256": preflight_sha},
                "static_identity": static_identity,
                "control_manifest": self.control_identity,
                "runtime_isolation": pre_release,
                "admission_policy": admission_policy,
            }
            gate_sha = self._write_json(gate_path, gate, 0o600)
            barrier_path = run_dir / "gpu-smoke-start-barrier.json"
            barrier = {
                **identity,
                "gate_path": str(gate_path),
                "gate_sha256": gate_sha,
            }
            barrier_sha = self._write_json(barrier_path, barrier, 0o600)
            process = {
                "schema": 4,
                "run_id": run_id,
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
                    "protected_heavy_processes": [],
                    "protected_nvidia_compute_pids": [],
                },
                "gpu_smoke": {
                    "policy_version": 2,
                    "requested": False,
                    "waiver_applied": False,
                    "authorization": None,
                    "start_barrier_path": str(barrier_path),
                    "gate_path": str(gate_path),
                    "memory_floor_kib": memory_floor,
                    "gpu_free_memory_floor_mib": base_contract.GPU_FREE_MEMORY_FLOOR_MIB,
                    "protected_heavy_processes": [],
                    "protected_nvidia_compute_pids": [],
                    "protected_nvidia_runtime_mapping_pids": [],
                    "unreadable_protected_maps": [],
                    "gate_sha256": gate_sha,
                    "start_barrier_sha256": barrier_sha,
                    "release_authorized_at": "20990101T000000Z",
                },
                "initial_preflight": {"path": str(initial_path), "sha256": initial_sha},
                "preflight": {"path": str(preflight_path), "sha256": preflight_sha},
                "resource_policy": {
                    "timeout_seconds": finalizer.contract.GPU_SMOKE_TIMEOUT_SECONDS,
                    "minimum_free_disk_bytes": finalizer.contract.GPU_SMOKE_MINIMUM_FREE_DISK_GIB << 30,
                    "owned_run_pause_memory_kib": finalizer.contract.OWNED_RUN_ABORT_MEMORY_FLOOR_KIB,
                },
                "command": command,
                "pid": 99,
                "pgid": 99,
                "start_ticks": 990,
                "started_at": "20990101T000000Z",
                "status": "exited",
                "gpu_smoke_pre_release_audit": pre_release,
                "gpu_smoke_last_audit": self._audit(gpu, [99]),
                "gpu_smoke_post_exit_audit": self._audit(gpu, []),
                "return_code": 0,
                "finished_at": "20990101T000100Z",
            }
            process_path = run_dir / "process.json"
            self._write_json(process_path, process, 0o600)

            child_gate = {
                "static_identity": static_identity,
                "gpu": gpu,
                "free_memory_mib": int(gpu["memory_mib"]) - int(gpu["used_mib"]),
                "mem_available_kib": 40 * 1024 * 1024,
                "host_memory_floor_kib": memory_floor,
                "admission_policy": admission_policy,
                "protected_heavy_pids": [],
                "protected_runtime_pids": [],
                "unreadable_protected_maps": [],
                "nvidia_compute_pids": [],
                "control_manifest": self.control_identity,
                "base_runner": {
                    "path": base_contract.BASE_RUNNER_RELATIVE_PATH.as_posix(),
                    "bytes": base_contract.BASE_RUNNER_SIZE,
                    "sha256": base_contract.BASE_RUNNER_SHA256,
                },
            }
            traits = {
                "compute_capability": 120,
                "multiprocessor_count": base_contract.EXPECTED_SM_COUNT,
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
                "total_global_memory_bytes": 24 * 1024 * 1024 * 1024,
            }
            runtime.update(
                {
                    "torch": {
                        "version": base_contract.EXPECTED_TORCH_VERSION,
                        "git_version": base_contract.EXPECTED_TORCH_GIT,
                        "cuda": base_contract.EXPECTED_TORCH_CUDA,
                        "hip": None,
                        "module_path": str(
                            (
                                ROOT
                                / "envs/pypto-nvidia/lib/python3.14/site-packages/torch/__init__.py"
                            ).resolve(strict=True)
                        ),
                    },
                    "child_pre_cuda_gate": child_gate,
                    "libcudart_paths": [str(cuda_runtime_path.resolve(strict=True))],
                    "observation": {
                        "device_ordinal": 0,
                        "device_name": base_contract.EXPECTED_DEVICE_NAME,
                        "device_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                        "pci_device_id": "0000:01:00.0",
                        "traits": traits,
                        "cuda_toolkit_version": base_contract.EXPECTED_CUDA_TOOLKIT_VERSION,
                        "cuda_driver_version": base_contract.EXPECTED_DRIVER_RELEASE,
                        "tensor_ir_revision": base_contract.TENSOR_IR_HEAD,
                        "cuda_tile_revision": base_contract.CUDA_TILE_HEAD,
                        "supported_compute_dtypes": list(
                            base_contract.EXPECTED_SUPPORTED_COMPUTE_DTYPES
                        ),
                        "cuda_driver_release_provenance": base_contract.EXPECTED_DRIVER_RELEASE,
                        "cuda_driver_api_version": 13030,
                        "cuda_runtime_api_version": 13000,
                        "cuda_runtime_library_path": str(
                            cuda_runtime_path.resolve(strict=True)
                        ),
                        "context_address": 7,
                        "context_id": 8,
                    },
                    "compile_request": {
                        "byte_identity_digest": (
                            contract.EXPECTED_COMPILE_REQUEST_BYTE_IDENTITY_DIGEST
                        ),
                        "loader_compatibility_input_digest": (
                            contract.EXPECTED_LOADER_COMPATIBILITY_INPUT_DIGEST
                        ),
                        "device_autotune_identity_digest": (
                            contract.EXPECTED_DEVICE_AUTOTUNE_IDENTITY_DIGEST
                        ),
                    },
                }
            )
            integrity_paths = {
                "runner": ROOT / finalizer.contract.RUNNER_RELATIVE_PATH,
                "contract": ROOT / "tools/_pypto_row_reduction_sm120_contract.py",
                "anchor_generator": ROOT
                / finalizer.contract.ANCHOR_GENERATOR_RELATIVE_PATH,
                "compile_anchors": ROOT
                / finalizer.contract.COMPILE_ANCHORS_RELATIVE_PATH,
                "controller": ROOT / finalizer.contract.CONTROLLER_RELATIVE_PATH,
                "control_validator": ROOT
                / finalizer.contract.CONTROL_VALIDATOR_RELATIVE_PATH,
                "preflight": ROOT
                / finalizer.contract.PREFLIGHT_ADAPTER_RELATIVE_PATH,
                "pypto_dso": dso_path,
                "python": python_path,
                "cuda_runtime": cuda_runtime_path,
                "environment_lock": ROOT / "ENVIRONMENT.lock",
                "versions_lock": ROOT / "VERSIONS.lock",
                "workspace_lock": ROOT / "WORKSPACE.lock",
            }
            provisional = {
                "schema_version": 1,
                "smoke": finalizer.contract.SMOKE_NAME,
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
                    "integrity": {
                        name: self._integrity_record(path)
                        for name, path in integrity_paths.items()
                    },
                    "control_manifest": self.control_identity,
                    "pypto": self.pypto_identity,
                    "tensor_ir_head": finalizer.contract.TENSOR_IR_HEAD,
                    "cuda_tile_head": finalizer.contract.CUDA_TILE_HEAD,
                    "llvm_head": finalizer.contract.LLVM_HEAD,
                    "replay_files": replay_files,
                },
                "run_context": {
                    "run_id": run_id,
                    "mode": "gpu-smoke",
                    "pid": 99,
                    "pgid": 99,
                    "start_ticks": 990,
                    "initial_preflight": {
                        "path": initial_path.relative_to(ROOT).as_posix(),
                        "sha256": initial_sha,
                    },
                    "preflight": {
                        "path": preflight_path.relative_to(ROOT).as_posix(),
                        "sha256": preflight_sha,
                    },
                    "gate": {
                        "path": str(gate_path),
                        "sha256": gate_sha,
                        "document": gate,
                    },
                    "start_barrier_sha256": barrier_sha,
                    "protected_zero_nvidia_policy": False,
                    "admission_policy": admission_policy,
                },
                "runtime": runtime,
            }
            provisional_path = replay / finalizer.contract.PROVISIONAL_NAME
            provisional_sha = self._write_json(provisional_path, provisional, 0o444)
        fixture.update(
            {
                "process_path": process_path,
                "initial_path": initial_path,
                "preflight_path": preflight_path,
                "gate_path": gate_path,
                "barrier_path": barrier_path,
                "provisional_path": provisional_path,
                "provisional": provisional,
                "provisional_sha": provisional_sha,
                "semantic_stdout": self._semantic_stdout(),
            }
        )
        return fixture

    @contextlib.contextmanager
    def _finalize_patches(
        self,
        fixture: dict[str, object],
        *,
        control_identity: dict[str, object] | None = None,
        semantic_stdout: str | None = None,
    ):
        completed = SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=semantic_stdout or fixture["semantic_stdout"],
        )
        with self._contract_patches(fixture), contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(finalizer.base.base, "require_no_site_finalizer")
            )
            stack.enter_context(
                mock.patch.object(
                    finalizer.control,
                    "validate_control_manifest",
                    return_value=control_identity or self.control_identity,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    finalizer.base.base,
                    "git_identity",
                    return_value=self.pypto_identity,
                )
            )
            run = stack.enter_context(
                mock.patch.object(finalizer.subprocess, "run", return_value=completed)
            )
            yield run

    def _call_finalize(
        self,
        fixture: dict[str, object],
        *,
        control_identity: dict[str, object] | None = None,
        semantic_stdout: str | None = None,
    ):
        with self._finalize_patches(
            fixture,
            control_identity=control_identity,
            semantic_stdout=semantic_stdout,
        ) as run:
            result = finalizer.finalize(
                workspace=ROOT,
                run_id=fixture["run_id"],
                expected_provisional_sha256=fixture["provisional_sha"],
            )
        return result, run

    def _mutate_json(
        self,
        path: pathlib.Path,
        keys: tuple[str | int, ...],
        value: object,
        *,
        mode: int,
    ) -> str:
        document = json.loads(path.read_text())
        target = document
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        return self._write_json(path, document, mode)

    def test_complete_synthetic_finalization_and_exact_no_replace_retry(self) -> None:
        fixture = self._build_fixture()
        (report, output, digest), run = self._call_finalize(fixture)
        self.assertEqual(
            report["status"],
            "accepted-real-sm120-row-reduction-ten-case-correctness-gate",
        )
        self.assertEqual(report["result"], fixture["provisional"]["runtime"])
        self.assertIs(
            report["inputs"]["replay_semantics"]["torch_cuda_initialized"], False
        )
        self.assertEqual(len(report["inputs"]["replay_semantics"]["records"]), 10)
        self.assertEqual(len(report["inputs"]["numerical_replay"]), 20)
        self.assertTrue(
            all(
                record["frozen_oracle_joined"]
                for record in report["inputs"]["numerical_replay"]
            )
        )
        self.assertIs(report["finalizer"]["cpu_only_deserialization"], True)
        self.assertIs(report["finalizer"]["torch_cuda_initialized"], False)
        self.assertEqual(digest, finalizer.sha256_file(output))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(run.call_args.kwargs["env"]["NVIDIA_VISIBLE_DEVICES"], "void")
        original = output.read_bytes()
        with self.assertRaisesRegex(finalizer.FinalizeError, "already exists"):
            self._call_finalize(fixture)
        self.assertEqual(output.read_bytes(), original)

    def test_outer_transaction_and_cpu_replay_mutation_matrix(self) -> None:
        names = (
            "process",
            "audit",
            "initial_preflight",
            "preflight",
            "gate",
            "barrier",
            "runtime",
            "runtime_aggregate",
            "compile_request",
            "child_static_join",
            "integrity",
            "replay",
            "replay_delete",
            "replay_extra",
            "replay_mode",
            "numerical_input",
            "numerical_reference",
            "numerical_actual",
            "numerical_cpu",
            "numerical_metrics",
            "dso_integrity",
            "python_integrity",
            "cuda_runtime_integrity",
            "control",
            "cpu_replay_empty",
            "cpu_replay_order",
            "cpu_replay_request",
            "cpu_replay_extra",
        )
        for name in names:
            with self.subTest(name=name):
                fixture = self._build_fixture()
                control_identity = None
                semantic_stdout = None
                if name == "process":
                    self._mutate_json(
                        fixture["process_path"], ("status",), "running", mode=0o600
                    )
                elif name == "audit":
                    self._mutate_json(
                        fixture["process_path"],
                        ("gpu_smoke_post_exit_audit", "external_nvidia_compute_pids"),
                        [42],
                        mode=0o600,
                    )
                elif name == "initial_preflight":
                    self._mutate_json(
                        fixture["initial_path"], ("ok",), False, mode=0o600
                    )
                elif name == "preflight":
                    self._mutate_json(
                        fixture["preflight_path"], ("ok",), False, mode=0o600
                    )
                elif name == "gate":
                    self._mutate_json(
                        fixture["gate_path"], ("pid",), 100, mode=0o600
                    )
                elif name == "barrier":
                    self._mutate_json(
                        fixture["barrier_path"], ("pid",), 100, mode=0o600
                    )
                elif name in {
                    "runtime",
                    "runtime_aggregate",
                    "compile_request",
                    "child_static_join",
                    "integrity",
                }:
                    document = json.loads(fixture["provisional_path"].read_text())
                    if name == "runtime":
                        document["runtime"]["torch"]["version"] = "ambient"
                    elif name == "runtime_aggregate":
                        document["runtime"]["external_synchronization"] = False
                    elif name == "compile_request":
                        document["runtime"]["compile_request"][
                            "byte_identity_digest"
                        ] = "0" * 64
                    elif name == "child_static_join":
                        document["runtime"]["child_pre_cuda_gate"]["static_identity"] = {
                            "static": "ambient"
                        }
                    else:
                        document["inputs"]["integrity"]["runner"]["sha256"] = "0" * 64
                    fixture["provisional_sha"] = self._write_json(
                        fixture["provisional_path"], document, 0o444
                    )
                elif name == "replay":
                    path = fixture["replay"] / "compile-request.msgpack"
                    path.chmod(0o600)
                    path.write_bytes(b"tampered")
                    path.chmod(0o444)
                elif name == "replay_delete":
                    (fixture["replay"] / "compile-request.msgpack").unlink()
                elif name == "replay_extra":
                    extra = fixture["replay"] / "unexpected.bin"
                    extra.write_bytes(b"unexpected")
                    extra.chmod(0o444)
                elif name == "replay_mode":
                    fixture["replay"].chmod(0o755)
                elif name.startswith("numerical_"):
                    document = json.loads(fixture["provisional_path"].read_text())
                    execution = document["runtime"]["executions"][0]
                    if name == "numerical_metrics":
                        execution["comparison"]["candidate_vs_torch"][
                            "observed_max_ulp"
                        ] = 1
                    else:
                        kind = name.removeprefix("numerical_")
                        suffix = {
                            "input": "input.bin",
                            "reference": "reference.bin",
                            "actual": "actual.bin",
                            "cpu": "cpu-reference.bin",
                        }[kind]
                        path = (
                            fixture["replay"]
                            / f"rank1_fp32_sum_n1.r0.{suffix}"
                        )
                        raw = finalizer.pack_words(
                            [finalizer.float32_word(999.0)], "float32"
                        )
                        path.chmod(0o600)
                        path.write_bytes(raw)
                        path.chmod(0o444)
                        digest = hashlib.sha256(raw).hexdigest()
                        relative = path.relative_to(ROOT).as_posix()
                        record = next(
                            item
                            for item in document["inputs"]["replay_files"]
                            if item["path"] == relative
                        )
                        record.update({"bytes": len(raw), "sha256": digest})
                        if kind == "input":
                            execution["input_before_sha256"] = digest
                            execution["input_after_sha256"] = digest
                        elif kind == "reference":
                            execution["expected_logical_bytes_sha256"] = digest
                        elif kind == "actual":
                            execution["actual_logical_bytes_sha256"] = digest
                        else:
                            execution["cpu_reference_bytes_sha256"] = digest
                    fixture["provisional_sha"] = self._write_json(
                        fixture["provisional_path"], document, 0o444
                    )
                elif name in {
                    "dso_integrity",
                    "python_integrity",
                    "cuda_runtime_integrity",
                }:
                    integrity_name, path_name, payload = {
                        "dso_integrity": (
                            "pypto_dso",
                            "dso_path",
                            b"drifted-dso\n",
                        ),
                        "python_integrity": (
                            "python",
                            "python_path",
                            b"drifted-python\n",
                        ),
                        "cuda_runtime_integrity": (
                            "cuda_runtime",
                            "cuda_runtime_path",
                            b"drifted-cudart\n",
                        ),
                    }[name]
                    path = fixture[path_name]
                    path.write_bytes(payload)
                    document = json.loads(fixture["provisional_path"].read_text())
                    document["inputs"]["integrity"][integrity_name] = (
                        self._integrity_record(path)
                    )
                    fixture["provisional_sha"] = self._write_json(
                        fixture["provisional_path"], document, 0o444
                    )
                elif name == "control":
                    control_identity = copy.deepcopy(self.control_identity)
                    control_identity["manifest_sha256"] = "0" * 64
                else:
                    semantic = json.loads(fixture["semantic_stdout"])
                    if name == "cpu_replay_empty":
                        semantic["records"] = []
                    elif name == "cpu_replay_order":
                        semantic["records"].reverse()
                    elif name == "cpu_replay_request":
                        semantic["compile_request"][
                            "device_autotune_identity_digest"
                        ] = "0" * 64
                    else:
                        semantic["unexpected"] = True
                    semantic_stdout = json.dumps(semantic, sort_keys=True) + "\n"
                with self.assertRaises(finalizer.FinalizeError):
                    self._call_finalize(
                        fixture,
                        control_identity=control_identity,
                        semantic_stdout=semantic_stdout,
                    )


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
