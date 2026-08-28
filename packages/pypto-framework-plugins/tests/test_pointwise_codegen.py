"""Focused native tile pointwise codegen tests against the exact DSO."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from pypto_plugins.errors import StrictCoverageError
from pypto_plugins.torch import pointwise_codegen as pc


class PointwiseCodegenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modules = pc.bootstrap_pypto()

    def test_fp32_add_mul_chain_compiles_deterministically(self) -> None:
        builder = pc.PointwiseProgramBuilder((256,), "float32")
        x = builder.add_input("x")
        y = builder.add_input("y")
        first = builder.emit("tensor.add", [x, y])
        second = builder.emit("tensor.muls", [first, builder.scalar(2.0)])
        third = builder.emit("tensor.neg", [second])
        builder.mark_output(third)
        program = builder.build()
        self.assertIsInstance(program, pc.NativePointwiseProgram)
        native_source = program.native_source(128)
        self.assertIn("@pl.jit", native_source)
        self.assertIn("pl.load", native_source)
        self.assertIn("pl.store", native_source)
        rendered = str(program.specialize(128))
        self.assertIn("pl.at(level=pl.Level.CORE_GROUP)", rendered)
        self.assertIn("pl.range(2)", rendered)
        self.assertEqual(rendered.count("pl.tile.load"), 2)
        self.assertIn("pl.tile.store", rendered)
        self.assertNotIn("tensor.", rendered)
        artifact = pc.compile_pointwise(program, tile=128)
        self.assertTrue(artifact.entry_name.startswith("pypto_fused_pointwise"))
        self.assertGreater(artifact.cubin_bytes, 0)
        self.assertEqual(artifact.argument_count, 3)
        self.assertEqual(artifact.workspace_bytes, 0)
        self.assertFalse(artifact.fallback_used)
        self.assertEqual(artifact.dso_sha256, pc.pypto_dso_sha256())
        with pc.capture_pointwise_artifacts() as capture:
            again = pc.compile_pointwise(builder.build(), tile=128)
        self.assertEqual(capture.single_artifact(), artifact)
        self.assertEqual(artifact.cubin_sha256, again.cubin_sha256)
        self.assertEqual(artifact.artifact_sha256, again.artifact_sha256)

    def test_bf16_chain_compiles_and_widens(self) -> None:
        builder = pc.PointwiseProgramBuilder((64,), "bfloat16")
        x = builder.add_input("x")
        first = builder.emit("tensor.exp", [x])
        builder.mark_output(first)
        artifact = pc.compile_pointwise(builder.build(), tile=64)
        self.assertFalse(artifact.fallback_used)
        self.assertGreater(artifact.cubin_bytes, 0)
        self.assertIn("pypto_inductor_", artifact.kernel_name)
        self.assertTrue(artifact.source_node.startswith("torch-inductor:"))

    def test_native_row_reduction_compiles(self) -> None:
        program = pc.NativeReductionProgram((256, 128), "float32", "sum")
        source = program.native_source()
        self.assertIn("@pl.jit", source)
        self.assertIn("pl.range(256)", source)
        self.assertIn("pl.row_sum", source)
        self.assertIn("pl.store", source)
        rendered = str(program.specialize())
        self.assertIn("pl.tile.row_sum", rendered)
        self.assertNotIn("tensor.", rendered)
        artifact = pc.compile_pointwise(program, tile=program.row_tile)
        self.assertTrue(artifact.entry_name.startswith("pypto_row_reduction"))
        self.assertEqual(artifact.argument_count, 2)
        self.assertFalse(artifact.fallback_used)

    def test_row_pitched_fp32_swiglu_compiles_with_explicit_casts(self) -> None:
        shape = (19, 3584)
        row_pitched = pc.PointwiseTensorSpec(
            shape,
            (7168, 1),
            "bfloat16",
            "cuda",
            0,
        )
        output_spec = pc.PointwiseTensorSpec(
            shape,
            (4096, 1),
            "bfloat16",
            "cuda",
            0,
        )
        builder = pc.PointwiseProgramBuilder(
            shape,
            "bfloat16",
            output_spec=output_spec,
        )
        gate = builder.add_input("gate", specialization=row_pitched)
        up = builder.add_input("up", specialization=row_pitched)
        gate_wide = builder.emit(
            "tensor.cast", [gate, builder.dtype("float32")]
        )
        up_wide = builder.emit("tensor.cast", [up, builder.dtype("float32")])
        negative = builder.emit(
            "tensor.muls", [gate_wide, builder.scalar(-1.0)]
        )
        exponential = builder.emit("tensor.exp", [negative])
        denominator = builder.emit(
            "tensor.adds", [exponential, builder.scalar(1.0)]
        )
        sigmoid = builder.emit("tensor.recip", [denominator])
        silu = builder.emit("tensor.mul", [gate_wide, sigmoid])
        product = builder.emit("tensor.mul", [silu, up_wide])
        result = builder.emit(
            "tensor.cast", [product, builder.dtype("bfloat16")]
        )
        builder.mark_output(result)
        program = builder.build()
        source = program.native_source(128)
        self.assertEqual(source.count("pl.cast"), 3)
        self.assertIn("target_type=pl.FP32", source)
        self.assertIn("target_type=pl.BF16", source)
        samples = program.specialization_samples()
        self.assertEqual(tuple(samples[0].stride()), (7168, 1))
        self.assertEqual(tuple(samples[1].stride()), (7168, 1))
        self.assertEqual(tuple(samples[2].stride()), (4096, 1))
        artifact = pc.compile_pointwise(program, tile=128)
        self.assertFalse(artifact.fallback_used)
        self.assertEqual(artifact.argument_count, 3)
        self.assertTrue(artifact.source_node.startswith("torch-inductor:"))

    def test_real_dso_digest_and_directory_uniqueness(self) -> None:
        dso = pc.pypto_dso_path()
        with dso.open("rb") as stream:
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
        self.assertEqual(pc.pypto_dso_sha256(), expected)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "pypto_core.first.so"
            first.write_bytes(b"first")
            self.assertEqual(pc._resolve_dso_override(root), first)
            (root / "pypto_core.second.so").write_bytes(b"second")
            with self.assertRaisesRegex(StrictCoverageError, "exactly one"):
                pc._resolve_dso_override(root)

    def test_compiler_cache_fails_closed_after_fork(self) -> None:
        original = pc._OWNER_PID
        try:
            pc._OWNER_PID = original + 1
            with self.assertRaisesRegex(StrictCoverageError, "inherited across fork"):
                pc.compile_cache_snapshot()
        finally:
            pc._OWNER_PID = original

    def test_bounds_and_dtype_fail_closed(self) -> None:
        with self.assertRaises(StrictCoverageError):
            pc.PointwiseProgramBuilder((8,), "float64")
        builder = pc.PointwiseProgramBuilder((8,), "float32")
        x = builder.add_input("x")
        builder.mark_output(x)
        with self.assertRaises(StrictCoverageError):
            builder.build()
        bounded = pc.PointwiseProgramBuilder((8,), "float32")
        y = bounded.add_input("y")
        with self.assertRaises(StrictCoverageError):
            for _ in range(65):
                bounded.emit("tensor.neg", [y])
        overflow_inputs = pc.PointwiseProgramBuilder((8,), "float32")
        with self.assertRaises(StrictCoverageError):
            for index in range(17):
                overflow_inputs.add_input(f"i{index}")


if __name__ == "__main__":
    unittest.main()
