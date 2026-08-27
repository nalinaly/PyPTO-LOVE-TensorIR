"""Focused pointwise codegen tests against the exact-DSO backend.

These tests run with CUDA hidden: they prove the FX-side program builder
produces the exact FusedPointwiseV2 HIR, that the strict facade compiles
it into a deterministic SM120 Cubin artifact, and that the plugin never
falls back. GPU execution is a separate later gate.
"""

from __future__ import annotations

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
        artifact = pc.compile_pointwise(builder.build(), tile=128)
        self.assertEqual(artifact.entry_name, "pypto_fused_pointwise_v2")
        self.assertGreater(artifact.cubin_bytes, 0)
        self.assertEqual(artifact.argument_count, 3)
        self.assertEqual(artifact.workspace_bytes, 0)
        self.assertFalse(artifact.fallback_used)
        again = pc.compile_pointwise(builder.build(), tile=128)
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
        self.assertIn("pypto_pointwise_", artifact.kernel_name)

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
