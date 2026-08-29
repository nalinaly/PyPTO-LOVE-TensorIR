from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
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


source_release = load_tool("verify_source_release")
sys.modules["verify_source_release"] = source_release
bootstrap_release = load_tool("bootstrap_release")
environment_bootstrap = load_tool("bootstrap_release_environment")
nested_git = load_tool("verify_no_nested_git")


class SourceReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = source_release.load_lock()

    def test_exact_release_boundaries_and_gitlinks_are_locked(self) -> None:
        repositories = self.lock["repositories"]
        self.assertEqual(repositories["pypto"]["commit_count"], 222)
        self.assertEqual(repositories["tensor_ir"]["commit_count"], 84)
        self.assertEqual(
            repositories["pypto"]["head_commit"],
            "b9d24b470c9a113e382916b586785112274c373f",
        )
        self.assertEqual(
            repositories["tensor_ir"]["head_commit"],
            "eb5fc509d9ac6d7a015a29f8b8330f6d9d15fa6b",
        )
        self.assertEqual(
            repositories["pypto"]["gitlinks"],
            {entry["path"]: entry["commit"] for entry in self.lock["pypto_submodules"]},
        )

    def test_package_subtrees_match_their_original_commits(self) -> None:
        report = source_release.verify_release_artifacts(ROOT, self.lock)
        self.assertEqual(
            report["packages"]["pypto-kernels"]["source_commit"],
            "bcd1a16f0cacdf4e5c196f6d7a7375fe117a61ab",
        )
        self.assertEqual(
            report["packages"]["pypto-framework-plugins"]["source_commit"],
            "6c363a1cddebc73d0f4134f198c921ec9f1d0e7c",
        )
        for package in report["packages"].values():
            self.assertEqual(package["source_tree"], package["prefix_tree"])

    def test_bundles_are_hash_locked_and_materializable(self) -> None:
        report = source_release.verify_release_artifacts(ROOT, self.lock)
        self.assertTrue(report["pypto"]["bundle"]["clone_probe"])
        self.assertTrue(report["tensor_ir"]["bundle"]["clone_probe"])
        self.assertEqual(report["pypto"]["patch_series"]["patch_count"], 222)
        self.assertEqual(report["tensor_ir"]["patch_series"]["patch_count"], 84)

    def test_every_patch_replays_to_the_locked_tree(self) -> None:
        report = source_release.replay_all_patch_series(ROOT, self.lock)
        self.assertEqual(
            report["pypto"]["result_tree"],
            self.lock["repositories"]["pypto"]["head_tree"],
        )
        self.assertEqual(
            report["tensor_ir"]["result_tree"],
            self.lock["repositories"]["tensor_ir"]["head_tree"],
        )

    def test_bootstrap_refuses_to_overwrite_any_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                bootstrap_release.bootstrap(
                    ROOT,
                    pathlib.Path(temporary),
                    self.lock,
                    jobs=24,
                )

    def test_tensor_ir_bundle_materializes_without_global_git_config(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            destination = pathlib.Path(temporary) / "tensor-ir"
            spec = self.lock["repositories"]["tensor_ir"]
            bootstrap_release.clone_bundled_repository(ROOT, destination, spec)
            identity = source_release.verify_checkout(destination, spec, "tensor_ir")
            self.assertEqual(identity["head_tree"], spec["head_tree"])
            self.assertEqual(
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(destination),
                        "rev-parse",
                        "--is-shallow-repository",
                    ],
                    text=True,
                ).strip(),
                "true",
            )

    def test_tampered_series_manifest_is_rejected(self) -> None:
        changed = json.loads(json.dumps(self.lock))
        changed["repositories"]["pypto"]["patch_series"]["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(source_release.SourceReleaseError, "SHA-256"):
            source_release.verify_patch_series(
                ROOT, "pypto", changed["repositories"]["pypto"]
            )

    def test_publishable_tree_has_no_nested_git_state(self) -> None:
        report = nested_git.verify(ROOT)
        self.assertEqual(report["status"], "clean")
        self.assertEqual(report["gitlinks"], 0)
        self.assertEqual(report["nested_git_entries"], 0)

    def test_environment_creation_has_a_lock_driven_fresh_path(self) -> None:
        artifact_lock = environment_bootstrap.load_artifact_lock()
        self.assertEqual(
            artifact_lock["fresh_creation"]["status"],
            "artifact-complete-formal-base-created-pip-check-clean",
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            destination = pathlib.Path(temporary) / "formal-environment"
            with mock.patch.object(environment_bootstrap, "command") as invoked:
                environment_bootstrap.create_conda_prefix(
                    pathlib.Path("/usr/bin/conda"),
                    destination,
                    base_prefix=None,
                )
            arguments = invoked.call_args.args[0]
            self.assertIn("--file", arguments)
            self.assertIn(str(ROOT / "environment" / "conda-linux-64.lock"), arguments)
            self.assertNotIn("--clone", arguments)

    def test_environment_acceleration_is_explicitly_optional(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            destination = root / "destination"
            (source / "bin").mkdir(parents=True)
            (source / "bin" / "python").touch()
            with mock.patch.object(environment_bootstrap, "command") as invoked:
                environment_bootstrap.create_conda_prefix(
                    pathlib.Path("/usr/bin/conda"),
                    destination,
                    base_prefix=source,
                )
            arguments = invoked.call_args.args[0]
            self.assertIn("--clone", arguments)
            self.assertNotIn("--file", arguments)

    def test_nested_git_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            fixture = pathlib.Path(temporary)
            subprocess.run(
                ["git", "init", "--quiet", "--initial-branch=release"],
                cwd=fixture,
                check=True,
            )
            (fixture / "vendor" / "package" / ".git").mkdir(parents=True)
            with self.assertRaisesRegex(nested_git.NestedGitError, "nested_git"):
                nested_git.verify(fixture, ("vendor",))


if __name__ == "__main__":
    unittest.main()
