from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stop_run = load_tool("stop_run")
sys.modules["stop_run"] = stop_run
run_isolated = load_tool("run_isolated")
preflight = load_tool("preflight")
audit_environment = load_tool("audit_python_environment")


class IsolationEnvironmentTest(unittest.TestCase):
    def make_environment(self, profile: str, environment: str) -> dict[str, str]:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            return run_isolated.isolated_environment(
                "test-run",
                pathlib.Path(directory),
                environment_prefix=run_isolated.ENVIRONMENTS[environment],
                framework_profile=profile,
            )

    def test_pypto_profile_contains_only_workspace_sources(self) -> None:
        environment = self.make_environment("pypto", "pypto-nvidia")
        paths = environment["PYTHONPATH"].split(":")
        self.assertIn(str(ROOT / "projects" / "pypto-kernels" / "src"), paths)
        self.assertIn(str(ROOT / "upstream" / "sglang" / "python"), paths)
        self.assertEqual(environment["SGLANG_PLUGINS"], "pypto")
        self.assertEqual(environment["PYPTO_FRAMEWORK_PROFILE"], "pypto")

    def test_baseline_profile_excludes_every_pypto_project(self) -> None:
        environment = self.make_environment(
            "baseline", "sglang-baseline-py312"
        )
        self.assertEqual(
            environment["PYTHONPATH"],
            str(ROOT / "upstream" / "sglang" / "python"),
        )
        self.assertEqual(
            environment["SGLANG_PLUGINS"], "__pypto_baseline_no_plugins__"
        )
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(
            environment["PYPTO_ENV_PREFIX"],
            str(ROOT / "envs" / "sglang-baseline-py312"),
        )

    def test_amd_and_simulator_environment_is_removed(self) -> None:
        protected = {
            "ROCM_PATH": "/bad",
            "HIP_VISIBLE_DEVICES": "0",
            "HSA_TEST": "bad",
            "ROCR_TEST": "bad",
            "GEMSIM_TEST": "bad",
            "LLVM_SYSPATH": "/home/zhaosiying/.triton/external-llvm",
            "JSON_SYSPATH": "/external/json",
            "TRITON_OFFLINE_BUILD": "1",
            "TRITON_BUILD_WITH_CLANG_LLD": "1",
        }
        original = {name: run_isolated.os.environ.get(name) for name in protected}
        try:
            run_isolated.os.environ.update(protected)
            environment = self.make_environment("pypto", "pypto-nvidia")
        finally:
            for name, value in original.items():
                if value is None:
                    run_isolated.os.environ.pop(name, None)
                else:
                    run_isolated.os.environ[name] = value
        self.assertTrue(set(protected).isdisjoint(environment))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("triton-dev", environment["PATH"])
        self.assertEqual(
            environment["CONDA_PREFIX"],
            str(ROOT / "envs" / "pypto-nvidia"),
        )

    def test_all_mutable_build_and_runtime_caches_are_workspace_owned(self) -> None:
        environment = self.make_environment("pypto", "pypto-nvidia")
        for name in (
            "CCACHE_DIR",
            "CONDA_PKGS_DIRS",
            "CUDA_CACHE_PATH",
            "HF_HOME",
            "PIP_CACHE_DIR",
            "PYPTO_CACHE_DIR",
            "PYPTO_PROG_BUILD_DIR",
            "SGLANG_CACHE_DIR",
            "TMPDIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "TRITON_HOME",
            "UV_CACHE_DIR",
            "XDG_CACHE_HOME",
        ):
            path = pathlib.Path(environment[name]).resolve()
            self.assertTrue(
                path == ROOT or ROOT in path.parents,
                f"{name} escaped the workspace: {path}",
            )

    def test_candidate_controls_are_exact_and_baseline_discards_them(self) -> None:
        controls = {
            "PTO_BACKTRACE": "1",
            "PYPTO_ALLOW_FALLBACK": "0",
            "PYPTO_INDUCTOR_CUDA_BACKEND": "pypto",
            "PYPTO_STRICT_COVERAGE": "1",
            "PYPTO_VERIFY_LEVEL": "roundtrip",
        }
        original = {name: run_isolated.os.environ.get(name) for name in controls}
        try:
            run_isolated.os.environ.update(controls)
            candidate = self.make_environment("pypto", "pypto-nvidia")
            baseline = self.make_environment(
                "baseline", "sglang-baseline-py312"
            )
        finally:
            for name, value in original.items():
                if value is None:
                    run_isolated.os.environ.pop(name, None)
                else:
                    run_isolated.os.environ[name] = value
        for name, value in controls.items():
            self.assertEqual(candidate[name], value)
            self.assertNotIn(name, baseline)

    def test_external_path_controls_fail_closed(self) -> None:
        original = run_isolated.os.environ.get("PTOAS_ROOT")
        try:
            run_isolated.os.environ["PTOAS_ROOT"] = "/opt/external-ptoas"
            with self.assertRaisesRegex(ValueError, "below the workspace"):
                self.make_environment("pypto", "pypto-nvidia")
        finally:
            if original is None:
                run_isolated.os.environ.pop("PTOAS_ROOT", None)
            else:
                run_isolated.os.environ["PTOAS_ROOT"] = original

    def test_unknown_direct_profile_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            with self.assertRaisesRegex(ValueError, "unknown framework profile"):
                run_isolated.isolated_environment(
                    "test-run",
                    pathlib.Path(directory),
                    environment_prefix=run_isolated.ENVIRONMENTS["pypto-nvidia"],
                    framework_profile="unknown",
                )

    def test_direct_profile_environment_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            with self.assertRaisesRegex(ValueError, "requires environment"):
                run_isolated.isolated_environment(
                    "test-run",
                    pathlib.Path(directory),
                    environment_prefix=run_isolated.ENVIRONMENTS[
                        "sglang-baseline-py312"
                    ],
                    framework_profile="pypto",
                )


class ProtectedProcessClassificationTest(unittest.TestCase):
    def test_cwd_binds_a_process_to_protected_scope(self) -> None:
        self.assertTrue(
            preflight.belongs_to_roots(
                "python worker.py",
                "/home/zhaosiying/zcode-lane/run",
                preflight.PROTECTED_ROOTS,
            )
        )

    def test_framework_module_launches_are_heavy(self) -> None:
        for command in (
            "python -m sglang.launch_server --model x",
            "python -m vllm.entrypoints.openai.api_server --model x",
            "python examples/vllm/qwen35_inference.py",
        ):
            self.assertTrue(preflight.is_heavy_command(command), command)

    def test_unrelated_sleep_is_not_heavy(self) -> None:
        self.assertFalse(preflight.is_heavy_command("/usr/bin/sleep 30"))


class StopRunVerificationTest(unittest.TestCase):
    def test_signal_verified_rechecks_before_killpg(self) -> None:
        metadata = {"run_id": "test"}
        with mock.patch.object(
            stop_run, "verify", return_value=(123, 456)
        ) as verify, mock.patch.object(stop_run.os, "killpg") as killpg:
            self.assertEqual(
                stop_run.signal_verified(metadata, stop_run.signal.SIGTERM),
                (123, 456),
            )
        verify.assert_called_once_with(metadata)
        killpg.assert_called_once_with(456, stop_run.signal.SIGTERM)


class PythonImportAuditTest(unittest.TestCase):
    def test_baseline_allows_only_the_official_sglang_source(self) -> None:
        self.assertTrue(
            audit_environment.is_allowed_source(
                ROOT / "upstream" / "sglang" / "python" / "sglang",
                "baseline",
            )
        )
        for candidate in (
            ROOT / "projects" / "pypto" / "python",
            ROOT / "projects" / "pypto-kernels" / "src",
            ROOT / "projects" / "pypto-framework-plugins" / "src",
        ):
            self.assertFalse(
                audit_environment.is_allowed_source(candidate, "baseline")
            )

    def test_executable_pth_allowlist_rejects_editable_triton(self) -> None:
        triton = pathlib.Path("__editable__.triton-3.7.1.pth")
        line = "import __editable___triton_3_7_1_finder; __editable___triton_3_7_1_finder.install()"
        self.assertFalse(
            audit_environment.executable_pth_is_allowed(triton, line, "pypto")
        )
        self.assertTrue(
            audit_environment.executable_pth_is_allowed(
                pathlib.Path("_editable_skbc_pypto.pth"),
                "import _editable_skbc_pypto",
                "pypto",
            )
        )
        self.assertFalse(
            audit_environment.executable_pth_is_allowed(
                pathlib.Path("_editable_skbc_pypto.pth"),
                "import _editable_skbc_pypto",
                "baseline",
            )
        )
        self.assertFalse(
            audit_environment.executable_pth_is_allowed(
                pathlib.Path("distutils-precedence.pth"),
                "import os; os.system('unexpected')",
                "pypto",
            )
        )
        self.assertTrue(
            audit_environment.executable_pth_is_allowed(
                pathlib.Path("distutils-precedence.pth"),
                audit_environment.DISTUTILS_PRECEDENCE_PTH,
                "baseline",
            )
        )

if __name__ == "__main__":
    unittest.main()
