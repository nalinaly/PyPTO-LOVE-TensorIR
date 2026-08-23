from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class BaselineLauncherTest(unittest.TestCase):
    def launcher(self, name: str) -> str:
        return (ROOT / "baseline" / name).read_text()

    def test_launchers_use_the_plugin_free_baseline_profile(self) -> None:
        for name in ("launch_0p8b.sh", "launch_9b.sh"):
            text = self.launcher(name)
            self.assertIn("--environment sglang-baseline-py312", text)
            self.assertIn("--framework-profile baseline", text)
            self.assertIn("--framework-launch", text)
            self.assertNotIn("--require-framework-plugins", text)
            self.assertNotIn("backend pypto", text)
            self.assertNotIn("--language-model-only", text)
            self.assertIn("--json-model-override-args", text)
            self.assertIn('"language_model_only":true', text)

    def test_initial_baseline_is_single_request_without_graphs(self) -> None:
        required = (
            "--tp-size 1",
            "--max-running-requests 1",
            "--chunked-prefill-size -1",
            "--attention-backend flashinfer",
            "--linear-attn-backend triton",
            "--linear-attn-decode-backend triton",
            "--linear-attn-prefill-backend triton",
            "--mamba-ssm-dtype float32",
            "--cuda-graph-backend-decode disabled",
            "--cuda-graph-backend-prefill disabled",
        )
        for name in ("launch_0p8b.sh", "launch_9b.sh"):
            text = self.launcher(name)
            for option in required:
                self.assertIn(option, text)

    def test_models_and_ports_are_disjoint(self) -> None:
        small = self.launcher("launch_0p8b.sh")
        large = self.launcher("launch_9b.sh")
        self.assertIn("models/Qwen3.5-0.8B", small)
        self.assertIn("models/Qwen3.5-9B", large)
        for port in ("43180", "44180", "45180"):
            self.assertIn(port, small)
            self.assertNotIn(port, large)
        for port in ("43190", "44190", "45190"):
            self.assertIn(port, large)
            self.assertNotIn(port, small)


if __name__ == "__main__":
    unittest.main()
