from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "build_triton_wheel_offline.sh"
RUNBOOK = ROOT / "docs" / "triton_workspace_wheel_acceptance_gate.md"
MANIFEST_SHA = "29c0736211ba0b286acd562ba097d7f1dea989671003c63a7b988de5afb0fe7d"


class OfflineTritonWheelRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = RUNNER.read_text(encoding="utf-8")
        cls.runbook = RUNBOOK.read_text(encoding="utf-8")
        cls.section_three = cls.runbook.split(
            "## 3. Frozen wheel recipe", 1
        )[1].split("## 4. Wheel and dependency audit", 1)[0]

    def test_runner_and_documented_invocation_are_valid_bash(self) -> None:
        subprocess.run(
            ["/bin/bash", "-n", str(RUNNER)],
            cwd=ROOT,
            check=True,
        )
        fences = re.findall(r"```bash\n(.*?)\n```", self.section_three, re.DOTALL)
        self.assertEqual(len(fences), 1)
        subprocess.run(
            ["/bin/bash", "-n"],
            cwd=ROOT,
            input=fences[0],
            text=True,
            check=True,
        )

    def test_runbook_calls_tracked_runner_directly_with_exact_inputs(self) -> None:
        section = self.section_three
        self.assertNotIn("/bin/bash -c", section)
        self.assertNotIn("REPLACE_WITH_REVIEWED", section)
        self.assertIn("--run-id-file builds/triton-wheel-build-run-id.json", section)
        self.assertIn(
            "/bin/bash /home/zhaosiying/pypto-love-tensor-ir/tools/"
            "build_triton_wheel_offline.sh",
            section,
        )
        self.assertIn("--reviewed-manifest-sha256", section)
        self.assertIn(MANIFEST_SHA, section)

    def test_runner_clears_ambient_pythonpath_and_isolates_build_python(self) -> None:
        script = self.script
        self.assertIn("unset PYTHONPATH PYTHONHOME", script)
        self.assertNotRegex(script, r"(?m)^\s*PYTHONPATH=")
        self.assertIn("reviewed-build-dependencies.pth", script)
        self.assertGreaterEqual(script.count('"$build_py" -I -B'), 6)
        for unsafe in (
            '"$build_py" -c',
            '"$build_py" -B',
            '"$build_py" -m',
            '"$build_py" - <<',
        ):
            self.assertNotIn(unsafe, script)
        pth = script.index("reviewed-build-dependencies.pth")
        metadata_probe = script.index("import importlib.metadata as metadata")
        self.assertLess(pth, metadata_probe)
        self.assertIn('.lower().replace("_", "-")', script)
        self.assertIn("assert found == set(expected), found", script)
        self.assertIn(
            "printf '#!/bin/sh\\nexec %s \"$@\"\\n' \"$cmake_payload\"",
            script,
        )
        self.assertIn(
            "aadd40ffd6b8bc9dac19f6dadc7ee0800cdbb3cf72f5b1f1b8b24e37f61e97da",
            script,
        )
        self.assertNotIn('cp --reflink=auto "$cmake_payload"', script)

    def test_exact_build_resource_sandbox_audit_and_probe_gates_remain(self) -> None:
        script = self.script
        required = (
            "5d6048aa0a324e090ada215b609ea76620133845",
            "448265acc1eff726c2e528813552865b33546cc9",
            "2ebfd3f7e98dee2e8524b9b210716fbe1f07759b6d89307280a9b10ae359b43e",
            "reviewed manifest does not match VERSIONS.lock",
            'require_absent_output "$root"',
            'require_absent_output "$log"',
            "set -o noclobber",
            ': > "$log"',
            'tee -a "$log"',
            "--verify --require-reviewed",
            "--probe-reviewed-tools",
            "--unshare-net",
            "/usr/bin/env -i",
            "PIP_NO_INDEX=1",
            "TRITON_OFFLINE_BUILD=1",
            "TRITON_EXT_ENABLED=ON",
            "TRITON_BUILD_PROTON=ON",
            "TRITON_PARALLEL_LINK_JOBS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2",
            "--skip-dependency-check",
            "ulimit -S -v $((24 * 1024 * 1024))",
            "generated-source-metadata",
            '"PKG-INFO"',
            '"SOURCES.txt"',
            '"dependency_links.txt"',
            '"entry_points.txt"',
            '"not-zip-safe"',
            '"requires.txt"',
            '"top_level.txt"',
            "_rename_no_replace(source, destination, require_same_parent=False)",
            "verify_reference_tree_exact",
            "audit_triton_wheel.py",
            "--expected-producer-identity-sha256",
            "probe_triton_wheel.py",
            "--expected-wheel-audit-evidence-sha256",
            "verify_source_input",
            'test "$(tree_sha "$producer_site")" = "$producer_site_sha"',
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, script)
        self.assertLess(script.index("-m build --wheel"), script.index("audit_triton_wheel.py"))
        self.assertLess(
            script.index("_rename_no_replace(source, destination"),
            script.index("verify_reference_tree_exact"),
        )
        self.assertLess(script.index("audit_triton_wheel.py"), script.index("probe_triton_wheel.py"))

    def test_generated_source_metadata_is_exact_and_atomically_retained(self) -> None:
        invocation = self.script.index(
            '"$base_py" -I -B - "$ws/tools" "$generated_source_metadata"'
        )
        body_start = self.script.index("<<'PY'\n", invocation) + len("<<'PY'\n")
        body_end = self.script.index("\nPY", body_start)
        program = self.script[body_start:body_end]
        expected_files = (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "not-zip-safe",
            "requires.txt",
            "top_level.txt",
        )

        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            run_root = pathlib.Path(directory)
            build_input = run_root / "build-input"
            (build_input / "python").mkdir(parents=True)

            source = run_root / "source" / "python" / "triton.egg-info"
            source.mkdir(parents=True)
            source.chmod(0o755)
            for name in expected_files:
                member = source / name
                member.write_text(f"retained {name}\n", encoding="utf-8")
                member.chmod(0o644)
            destination = run_root / "generated-source-metadata"
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-",
                    str(ROOT / "tools"),
                    str(source),
                    str(destination),
                    str(build_input),
                ],
                input=program,
                text=True,
                check=True,
            )
            self.assertFalse(source.exists())
            self.assertEqual(
                tuple(sorted(path.name for path in destination.iterdir())),
                expected_files,
            )

            rejected_source = (
                run_root / "rejected-source" / "python" / "triton.egg-info"
            )
            rejected_source.mkdir(parents=True)
            rejected_source.chmod(0o755)
            for name in (*expected_files, "arbitrary-extra"):
                member = rejected_source / name
                member.write_text("must not be deleted\n", encoding="utf-8")
                member.chmod(0o644)
            rejected_destination = run_root / "rejected-generated-source-metadata"
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-",
                    str(ROOT / "tools"),
                    str(rejected_source),
                    str(rejected_destination),
                    str(build_input),
                ],
                input=program,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("member set drift", result.stderr)
            self.assertTrue((rejected_source / "arbitrary-extra").is_file())
            self.assertFalse(rejected_destination.exists())

    def test_lit_wrapper_heredoc_preserves_required_single_quotes(self) -> None:
        match = re.search(
            r'"\$base_py" -I -B - "\$producer_bin/lit" "\$build_py" '
            r"<<'PY'\n(.*?)\nPY",
            self.script,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            wrapper = root / "lit"
            python = root / "build-venv" / "bin" / "python"
            subprocess.run(
                [sys.executable, "-", str(wrapper), str(python)],
                input=match.group(1),
                text=True,
                check=True,
            )
            self.assertEqual(
                wrapper.read_text(encoding="utf-8").splitlines(),
                [
                    f"#!{python.absolute()}",
                    "import sys",
                    "from lit.main import main",
                    "if __name__ == '__main__':",
                    "    sys.argv[0] = sys.argv[0].removesuffix('.exe')",
                    "    sys.exit(main())",
                ],
            )

    def test_existing_output_failures_do_not_overwrite(self) -> None:
        source_and_check = 'source "$1"\nrequire_absent_output "$2"'
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            missing = root / "missing"
            subprocess.run(
                ["/bin/bash", "-c", source_and_check, "bash", str(RUNNER), str(missing)],
                check=True,
            )
            self.assertFalse(missing.exists())

            existing_file = root / "existing.log"
            existing_file.write_text("retained evidence\n", encoding="utf-8")
            existing_dir = root / "existing-build"
            existing_dir.mkdir()
            dangling = root / "dangling"
            dangling.symlink_to(root / "absent-target")
            for existing in (existing_file, existing_dir, dangling):
                before = (
                    existing_file.read_bytes()
                    if existing == existing_file
                    else None
                )
                result = subprocess.run(
                    [
                        "/bin/bash",
                        "-c",
                        source_and_check,
                        "bash",
                        str(RUNNER),
                        str(existing),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                with self.subTest(path=existing.name):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("refusing to overwrite output", result.stderr)
                    if before is not None:
                        self.assertEqual(existing_file.read_bytes(), before)
            self.assertTrue(existing_dir.is_dir())
            self.assertTrue(dangling.is_symlink())


if __name__ == "__main__":
    unittest.main()
