from __future__ import annotations

import base64
import copy
import csv
from dataclasses import replace
import hashlib
import importlib.util
import io
import pathlib
import shutil
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_auditor():
    path = ROOT / "tools" / "audit_triton_wheel.py"
    spec = importlib.util.spec_from_file_location(
        "test_triton_wheel_audit_tool", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = load_auditor()


class TritonWheelAuditTest(unittest.TestCase):
    libdevice = b"synthetic libdevice for the local wheel auditor tests\n"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="triton-wheel-audit-test-", dir=ROOT / "runs"
        )
        self.addCleanup(self.temporary.cleanup)
        self.workspace = pathlib.Path(self.temporary.name).resolve()
        self.temp_root = self.workspace / "tmp"
        self.evidence_dir = self.workspace / "evidence"
        self.tools_dir = self.workspace / "tools"
        for directory in (self.temp_root, self.evidence_dir, self.tools_dir):
            directory.mkdir()

        self.readelf = self.make_executable("readelf")
        self.ldd = self.make_executable("ldd")
        self.bwrap = self.make_executable("bwrap")
        tool_paths = {
            "bwrap": self.bwrap.resolve(),
            "ldd": self.ldd.resolve(),
            "readelf": self.readelf.resolve(),
        }
        tool_sha256 = {
            name: audit.sha256_file(path) for name, path in tool_paths.items()
        }
        path_patch = mock.patch.object(audit, "AUDIT_TOOL_PATHS", tool_paths)
        sha_patch = mock.patch.object(audit, "AUDIT_TOOL_SHA256", tool_sha256)
        path_patch.start()
        sha_patch.start()
        self.addCleanup(path_patch.stop)
        self.addCleanup(sha_patch.stop)
        self.producer_site = self.workspace / "producer-site"
        self.producer_bin = self.workspace / "producer-bin"
        self.producer_site.mkdir()
        self.producer_bin.mkdir()
        (self.producer_site / "selected.txt").write_text("selected producer\n")
        for name, payload in (
            ("cmake", b"synthetic cmake\n"),
            ("ninja", b"synthetic ninja\n"),
        ):
            path = self.producer_bin / name
            path.write_bytes(payload)
            path.chmod(0o755)
        lit = self.producer_bin / "lit"
        lit.write_text(
            f"#!{self.workspace}/build-venv/bin/python\n"
            "import sys\n"
            "from lit.main import main\n"
            "if __name__ == '__main__':\n"
            "    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
            "    sys.exit(main())\n"
        )
        lit.chmod(0o755)
        cmake_patch = mock.patch.object(
            audit,
            "PRODUCER_CMAKE_SHA256",
            audit.sha256_file(self.producer_bin / "cmake"),
        )
        ninja_patch = mock.patch.object(
            audit,
            "PRODUCER_NINJA_SHA256",
            audit.sha256_file(self.producer_bin / "ninja"),
        )
        cmake_patch.start()
        ninja_patch.start()
        self.addCleanup(cmake_patch.stop)
        self.addCleanup(ninja_patch.stop)
        self.producer_site_document = {
            **audit.tree_identity(self.producer_site),
            "distributions": [
                "build",
                "lit",
                "packaging",
                "pyproject-hooks",
                "setuptools",
                "wheel",
            ],
        }
        self.producer_site_identity = self.workspace / "producer-site.json"
        self.write_json(
            self.producer_site_identity, self.producer_site_document
        )
        projection_patch = mock.patch.object(
            audit,
            "PRODUCER_SITE_IDENTITY",
            dict(self.producer_site_document),
        )
        projection_patch.start()
        self.addCleanup(projection_patch.stop)
        self.source_archive = self.workspace / "source.tar"
        self.source_input = self.workspace / "source-input"
        self.build_input = self.workspace / "build-input"
        self.built_source = self.workspace / "built-source"
        self.source_input.mkdir()
        (self.source_input / "setup.py").write_text("# pinned source\n")
        with tarfile.open(self.source_archive, "w") as source_archive:
            source_archive.add(
                self.source_input / "setup.py",
                arcname="setup.py",
                recursive=False,
            )
        shutil.copytree(self.source_input, self.build_input, dirs_exist_ok=True)
        overlay = self.build_input / "third_party/nvidia/backend/bin"
        overlay.mkdir(parents=True)
        (overlay / "ptxas").write_bytes(b"reviewed overlay\n")
        self.nvidia_overlay = self.workspace / "reviewed-overlay"
        (self.nvidia_overlay / "bin").mkdir(parents=True)
        (self.nvidia_overlay / "bin/ptxas").write_bytes(b"reviewed overlay\n")
        shutil.copytree(self.build_input, self.built_source, dirs_exist_ok=True)
        self.expectations = audit.AuditExpectations(
            libdevice_sha256=audit.sha256_bytes(self.libdevice),
            source_archive_sha256=audit.sha256_file(self.source_archive),
        )

        self.source_document = {
            "archive_sha256": self.expectations.source_archive_sha256,
            "build_input_tree_sha256": audit.tree_identity(self.build_input)[
                "sha256"
            ],
            "commit": self.expectations.commit,
            "extracted_tree_sha256": audit.tree_identity(self.source_input)[
                "sha256"
            ],
            "kind": "triton-git-archive",
            "module_version": self.expectations.module_version,
            "repository": audit.TRITON_REPOSITORY,
            "schema_version": 1,
            "tree": self.expectations.tree,
        }
        self.source_path = self.workspace / "source-provenance.json"
        self.write_json(self.source_path, self.source_document)

        producer_unsigned = {
            "distribution_set_sha256": "c" * 64,
            "dynamic_libraries": [{"path": "/lib/synthetic.so", "sha256": "d" * 64}],
            "executables": [
                {
                    "name": "python",
                    "path": str(self.workspace / "python"),
                    "sha256": "e" * 64,
                }
            ],
            "identity_schema": audit.PRODUCER_IDENTITY_SCHEMA,
            "identity_scope": audit.PRODUCER_IDENTITY_SCOPE,
            "package_distributions": {
                name: {} for name in audit.PRODUCER_PACKAGE_VERSIONS
            },
            "package_versions": dict(audit.PRODUCER_PACKAGE_VERSIONS),
            "python_version": "3.14.6",
            "record_rewrite_count": 6,
            "record_rewrite_policy_sha256": "f" * 64,
        }
        self.producer_identity = audit.compact_json_sha256(producer_unsigned)
        self.producer_document = {
            **producer_unsigned,
            "selected_producer_identity_sha256": self.producer_identity,
        }
        self.producer_path = self.workspace / "producer-provenance.json"
        self.write_json(self.producer_path, self.producer_document)

        self.dependency_document = {
            "build_inputs": {
                "nvidia_backend_overlay": "nvidia-backend-overlay",
                "nvidia_backend_overlay_tree": audit.tree_identity(
                    self.nvidia_overlay
                ),
            },
            "packages": [
                {
                    "archive_bytes": 17,
                    "archive_sha256": "1" * 64,
                    "expanded_tree": {"sha256": "2" * 64},
                    "name": "synthetic-llvm",
                }
            ],
            "schema_version": 1,
            "status": "materialized-unreviewed",
            "triton_commit": self.expectations.commit,
            "triton_llvm_commit": self.expectations.llvm_commit,
            "triton_tree": self.expectations.tree,
        }
        dependency_raw = audit.canonical_json(self.dependency_document).encode("ascii")
        self.dependency_sha256 = audit.sha256_bytes(dependency_raw)
        dependency_dir = (
            self.workspace / "dependencies" / self.dependency_sha256
        )
        dependency_dir.mkdir(parents=True)
        shutil.copytree(
            self.nvidia_overlay,
            dependency_dir / "nvidia-backend-overlay",
        )
        self.dependency_path = dependency_dir / "manifest.json"
        self.dependency_path.write_bytes(dependency_raw)
        self.review_document = self.make_review_document(
            self.dependency_sha256
        )
        self.review_path = dependency_dir / "review.json"
        self.write_json(self.review_path, self.review_document)

        self.wheel_path = self.workspace / (
            f"triton-{self.expectations.distribution_version}"
            "-cp314-cp314-linux_x86_64.whl"
        )
        self.build_wheel(self.wheel_path)
        self.evidence_path = self.evidence_dir / "audit.json"
        self.request = audit.AuditRequest(
            workspace=self.workspace,
            wheel=self.wheel_path,
            dependency_manifest=self.dependency_path,
            reviewed_dependency_manifest_sha256=self.dependency_sha256,
            source_provenance=self.source_path,
            source_archive=self.source_archive,
            source_input=self.source_input,
            build_input=self.build_input,
            built_source=self.built_source,
            producer_provenance=self.producer_path,
            producer_site_identity=self.producer_site_identity,
            producer_site=self.producer_site,
            producer_bin=self.producer_bin,
            expected_producer_identity_sha256=self.producer_identity,
            evidence=self.evidence_path,
            temp_root=self.temp_root,
            tools=audit.AuditToolPaths(
                readelf=self.readelf,
                ldd=self.ldd,
                bwrap=self.bwrap,
            ),
        )

    def make_executable(self, name: str) -> pathlib.Path:
        path = self.tools_dir / name
        path.write_bytes(b"#!/bin/sh\nexit 97\n")
        path.chmod(0o755)
        return path

    def write_json(self, path: pathlib.Path, value: object) -> None:
        path.write_text(audit.canonical_json(value), encoding="ascii")

    def make_review_document(self, manifest_sha256: str) -> dict[str, object]:
        return {
            "archives": [
                {
                    "name": package["name"],
                    "sha256": package["archive_sha256"],
                }
                for package in self.dependency_document["packages"]
            ],
            "manifest_sha256": manifest_sha256,
            "schema_version": 1,
            "status": "reviewed",
            "triton_commit": self.expectations.commit,
            "triton_llvm_commit": self.expectations.llvm_commit,
            "triton_tree": self.expectations.tree,
        }

    @staticmethod
    def zip_info(
        name: str,
        *,
        mode: int = 0o644,
        file_type: int = stat.S_IFREG,
    ) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = (file_type | mode) << 16
        return info

    @staticmethod
    def record_bytes(members: dict[str, tuple[bytes, int]]) -> bytes:
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        for path in sorted((*members, audit.RECORD_PATH)):
            if path == audit.RECORD_PATH:
                writer.writerow((path, "", ""))
                continue
            data = members[path][0]
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            writer.writerow(
                (
                    path,
                    f"sha256={digest.rstrip(b'=').decode('ascii')}",
                    str(len(data)),
                )
            )
        return output.getvalue().encode("utf-8")

    def wheel_members(self) -> dict[str, tuple[bytes, int]]:
        members: dict[str, tuple[bytes, int]] = {
            audit.METADATA_PATH: (
                (
                    "Metadata-Version: 2.4\n"
                    "Name: triton\n"
                    f"Version: {self.expectations.distribution_version}\n\n"
                ).encode("ascii"),
                0o644,
            ),
            audit.WHEEL_PATH: (
                (
                    "Wheel-Version: 1.0\n"
                    "Generator: synthetic-local-test\n"
                    "Root-Is-Purelib: false\n"
                    "Tag: cp314-cp314-linux_x86_64\n\n"
                ).encode("ascii"),
                0o644,
            ),
            audit.MODULE_PATH: (
                f'__version__ = "{self.expectations.module_version}"\n'.encode(
                    "ascii"
                ),
                0o644,
            ),
            audit.FILECHECK_PATH: (b"\x7fELFsynthetic FileCheck\n", 0o755),
            audit.LIBDEVICE_PATH: (self.libdevice, 0o644),
            "triton/backends/nvidia/lib/cupti/libcupti.so.13": (
                b"\x7fELFsynthetic CUPTI\n",
                0o644,
            ),
        }
        for header in audit.REQUIRED_HEADERS:
            members[header] = (b"/* synthetic NVIDIA header */\n", 0o644)
        for tool in self.expectations.tool_versions():
            members[f"{audit.NVIDIA_BIN_PREFIX}/{tool}"] = (
                f"\x7fELFsynthetic {tool}\n".encode("utf-8"),
                0o755,
            )
        return members

    def build_wheel(
        self,
        path: pathlib.Path,
        *,
        tamper_after_record: str | None = None,
    ) -> None:
        members = self.wheel_members()
        record = self.record_bytes(members)
        if tamper_after_record is not None:
            original, mode = members[tamper_after_record]
            members[tamper_after_record] = (original + b"# tampered\n", mode)
        members[audit.RECORD_PATH] = (record, 0o644)
        with zipfile.ZipFile(path, "w") as wheel:
            for name in sorted(members):
                data, mode = members[name]
                wheel.writestr(self.zip_info(name, mode=mode), data)

    def append_member(
        self,
        name: str,
        data: bytes,
        *,
        mode: int = 0o644,
        file_type: int = stat.S_IFREG,
    ) -> None:
        with zipfile.ZipFile(self.wheel_path, "a") as wheel:
            wheel.writestr(
                self.zip_info(name, mode=mode, file_type=file_type), data
            )

    def fake_command(
        self,
        argv: list[str],
        *,
        temp_root: pathlib.Path,
        limits: object,
        description: str,
    ) -> str:
        self.assertEqual(temp_root, self.temp_root)
        self.assertIsInstance(limits, audit.AuditLimits)
        if description.startswith("readelf "):
            self.assertEqual(argv[0], str(self.readelf.resolve()))
            return (
                "ELF Header:\n"
                "  Class:                             ELF64\n"
                "  Data:                              2's complement, little endian\n"
                "  Type:                              DYN (Shared object file)\n"
                "  Machine:                           X86-64\n"
                "  Build ID: 0123456789abcdef\n"
                " 0x0000000000000001 (NEEDED) Shared library: [libc.so.6]\n"
            )
        if description.startswith("ldd "):
            self.assertEqual(argv[0], str(self.bwrap.resolve()))
            self.assertIn("--unshare-net", argv)
            self.assertIn("--ro-bind", argv)
            self.assertEqual(argv[-2], str(self.ldd.resolve()))
            self.assertTrue(argv[-1].startswith("/wheel/"))
            return (
                "linux-vdso.so.1 (0x0000000000000000)\n"
                "libc.so.6 => /lib64/libc.so.6 (0x0000000000000000)\n"
            )
        if description.startswith("sandboxed "):
            self.assertEqual(argv[0], str(self.bwrap.resolve()))
            self.assertIn("--unshare-net", argv)
            self.assertEqual(argv[-1], "--version")
            name = description.removeprefix("sandboxed ").removesuffix(
                " --version"
            )
            if name == "FileCheck":
                return "LLVM version 20.1.0\n"
            version = self.expectations.tool_versions()[name]
            release = ".".join(version.split(".")[:2])
            return f"{name}: release {release}, V{version}\n"
        raise AssertionError(f"unexpected command boundary: {description}: {argv}")

    def assert_rejected_before_commands(
        self,
        pattern: str,
        request: object | None = None,
    ) -> None:
        selected = self.request if request is None else request
        with mock.patch.object(
            audit,
            "run_limited_command",
            side_effect=AssertionError("rejected audit unexpectedly ran a command"),
        ) as command:
            with self.assertRaisesRegex(audit.AuditError, pattern):
                audit.run_audit(selected, expectations=self.expectations)
        command.assert_not_called()
        self.assertFalse(selected.evidence.exists())

    def test_success_runs_static_native_probe_and_no_replace_publish(self) -> None:
        self.assertEqual(self.dependency_path.parent.name, self.dependency_sha256)
        self.assertEqual(
            self.dependency_path.read_bytes(),
            audit.canonical_json(self.dependency_document).encode("ascii"),
        )
        self.assertEqual(
            self.review_path.read_bytes(),
            audit.canonical_json(self.review_document).encode("ascii"),
        )
        self.assertEqual(
            self.review_document["manifest_sha256"], self.dependency_sha256
        )
        self.assertEqual(
            self.review_document["archives"],
            [
                {
                    "name": package["name"],
                    "sha256": package["archive_sha256"],
                }
                for package in self.dependency_document["packages"]
            ],
        )
        self.assertEqual(
            self.review_document["triton_tree"], self.source_document["tree"]
        )

        with mock.patch.object(
            audit, "run_limited_command", side_effect=self.fake_command
        ) as command:
            evidence, evidence_sha256 = audit.run_audit(
                self.request, expectations=self.expectations
            )

        published, published_raw = audit.load_canonical_json(
            self.evidence_path, "published evidence"
        )
        self.assertEqual(published, evidence)
        self.assertEqual(evidence_sha256, audit.sha256_bytes(published_raw))
        self.assertEqual(stat.S_IMODE(self.evidence_path.stat().st_mode), 0o444)
        self.assertFalse(
            list(self.evidence_dir.glob(f".{self.evidence_path.name}.*.partial"))
        )

        wheel = evidence["wheel"]
        archive_paths = {
            member["path"] for member in wheel["archive"]["members"]
        }
        record_paths = {entry["path"] for entry in wheel["record"]["entries"]}
        self.assertEqual(record_paths, archive_paths)
        self.assertEqual(
            wheel["distribution_metadata"],
            {
                "metadata_version": "2.4",
                "name": "triton",
                "version": self.expectations.distribution_version,
            },
        )
        self.assertEqual(
            wheel["wheel_metadata"]["tags"],
            ["cp314-cp314-linux_x86_64"],
        )
        self.assertEqual(
            wheel["required_resources"]["libdevice"]["sha256"],
            self.expectations.libdevice_sha256,
        )
        self.assertIn(
            "triton/backends/nvidia/lib/cupti/libcupti.so.13",
            wheel["elf_paths"],
        )
        self.assertEqual(
            len(wheel["native_manifest"]), len(wheel["elf_paths"])
        )
        self.assertEqual(
            command.call_count,
            len(wheel["native_manifest"]) * 2 + len(wheel["tool_probes"]),
        )
        descriptions = [call.kwargs["description"] for call in command.call_args_list]
        self.assertEqual(
            sum(description.startswith("ldd ") for description in descriptions),
            len(wheel["native_manifest"]),
        )
        dependency_evidence = evidence["provenance"]["dependency_manifest"]
        self.assertEqual(
            dependency_evidence["reviewed_manifest_sha256"],
            self.dependency_sha256,
        )
        self.assertEqual(
            dependency_evidence["review"]["document"], self.review_document
        )
        self.assertEqual(
            evidence["provenance"]["source"]["document"], self.source_document
        )
        self.assertEqual(
            evidence["provenance"]["producer"]["selected_identity_sha256"],
            self.producer_identity,
        )

    def test_duplicate_zip_member_is_rejected(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.append_member(audit.MODULE_PATH, b"duplicate\n")
        self.assert_rejected_before_commands("duplicate ZIP members")

    def test_noncanonical_parent_path_is_rejected(self) -> None:
        self.append_member("../escape.py", b"escape\n")
        self.assert_rejected_before_commands("path is unsafe or non-canonical")

    def test_archive_symlink_is_rejected(self) -> None:
        self.append_member(
            "triton/alias",
            b"__init__.py",
            mode=0o777,
            file_type=stat.S_IFLNK,
        )
        self.assert_rejected_before_commands("archive symlink is forbidden")

    def test_record_tamper_is_rejected(self) -> None:
        self.build_wheel(
            self.wheel_path, tamper_after_record=audit.MODULE_PATH
        )
        self.assert_rejected_before_commands(
            rf"RECORD SHA256 mismatch: {audit.MODULE_PATH}"
        )

    def test_native_rpath_is_rejected(self) -> None:
        def command(argv, **kwargs):
            value = self.fake_command(argv, **kwargs)
            if kwargs["description"].startswith("readelf "):
                value += " 0x000000000000001d (RUNPATH) Library runpath: [/external]\n"
            return value

        with mock.patch.object(audit, "run_limited_command", side_effect=command):
            with self.assertRaisesRegex(audit.AuditError, "RPATH/RUNPATH"):
                audit.run_audit(self.request, expectations=self.expectations)

    def test_wrong_source_anchor_is_rejected(self) -> None:
        wrong = {**self.source_document, "commit": "0" * 40}
        self.write_json(self.source_path, wrong)
        self.assert_rejected_before_commands("source provenance commit mismatch")

    def test_source_archive_and_extracted_tree_are_rederived(self) -> None:
        self.source_archive.write_bytes(b"not the frozen archive")
        self.assert_rejected_before_commands("frozen Git-archive anchor")

    def test_self_hashed_extra_build_input_is_rejected(self) -> None:
        (self.build_input / "unexpected_plugin.py").write_text("malicious = True\n")
        self.source_document["build_input_tree_sha256"] = audit.tree_identity(
            self.build_input
        )["sha256"]
        self.write_json(self.source_path, self.source_document)
        self.assert_rejected_before_commands(
            "not exact source plus reviewed NVIDIA overlay"
        )

    def test_post_build_source_extras_are_rejected(self) -> None:
        mutations = {
            "package-source": lambda: (
                self.built_source / "python/triton/runtime/generated.py"
            ),
            "generated-metadata": lambda: (
                self.built_source / "python/triton.egg-info/PKG-INFO"
            ),
            "empty-directory": lambda: self.built_source / "build/generated",
        }
        for name, select in mutations.items():
            with self.subTest(name=name):
                path = select()
                if name == "empty-directory":
                    path.mkdir(parents=True)
                else:
                    path.parent.mkdir(parents=True)
                    path.write_text("unapproved post-build output\n")
                self.assert_rejected_before_commands(
                    "built source has unapproved extra entries"
                )
                top = path.relative_to(self.built_source).parts[0]
                shutil.rmtree(self.built_source / top)

    def test_post_build_source_root_mode_is_exact(self) -> None:
        self.built_source.chmod(0o700)
        self.assert_rejected_before_commands("root directory mode drift")

    def test_empty_directory_and_directory_mode_are_in_build_identity(self) -> None:
        mutations = {
            "empty-directory": lambda: (self.build_input / "unexpected-empty").mkdir(),
            "directory-mode": lambda: (self.build_input / "third_party").chmod(0o700),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                mutate()
                self.source_document["build_input_tree_sha256"] = audit.tree_identity(
                    self.build_input
                )["sha256"]
                self.write_json(self.source_path, self.source_document)
                self.assert_rejected_before_commands("directory/mode tree")
                if name == "empty-directory":
                    (self.build_input / "unexpected-empty").rmdir()
                else:
                    (self.build_input / "third_party").chmod(0o755)

    def test_reviewed_overlay_tree_is_rechecked(self) -> None:
        overlay = self.dependency_path.parent / "nvidia-backend-overlay/bin/ptxas"
        overlay.write_bytes(b"tampered overlay\n")
        self.assert_rejected_before_commands("overlay differs from manifest")

    def test_reviewed_dependency_directory_name_is_an_anchor(self) -> None:
        misplaced_dir = self.workspace / "dependencies" / "misnamed"
        misplaced_dir.mkdir()
        misplaced_manifest = misplaced_dir / "manifest.json"
        misplaced_manifest.write_bytes(self.dependency_path.read_bytes())
        (misplaced_dir / "review.json").write_bytes(self.review_path.read_bytes())
        request = replace(
            self.request, dependency_manifest=misplaced_manifest
        )
        self.assert_rejected_before_commands(
            "directory name must equal manifest SHA256", request
        )

    def test_wrong_dependency_content_anchor_is_rejected(self) -> None:
        wrong_anchor = "0" * 64
        self.assertNotEqual(wrong_anchor, self.dependency_sha256)
        wrong_dir = self.workspace / "dependencies" / wrong_anchor
        wrong_dir.mkdir()
        wrong_manifest = wrong_dir / "manifest.json"
        wrong_manifest.write_bytes(self.dependency_path.read_bytes())
        wrong_review = self.make_review_document(wrong_anchor)
        self.write_json(wrong_dir / "review.json", wrong_review)
        request = replace(
            self.request,
            dependency_manifest=wrong_manifest,
            reviewed_dependency_manifest_sha256=wrong_anchor,
        )
        self.assert_rejected_before_commands(
            "dependency manifest differs from its reviewed SHA256 anchor", request
        )

    def test_dependency_review_binds_archives_and_source_anchor(self) -> None:
        mutations = {
            "archive": lambda document: document["archives"][0].update(
                {"sha256": "3" * 64}
            ),
            "source": lambda document: document.update(
                {"triton_tree": "4" * 40}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                wrong = copy.deepcopy(self.review_document)
                mutate(wrong)
                self.write_json(self.review_path, wrong)
                self.assert_rejected_before_commands(
                    "review record differs from manifest/source anchor"
                )

    def test_wrong_producer_anchor_is_rejected(self) -> None:
        wrong_anchor = "0" * 64
        self.assertNotEqual(wrong_anchor, self.producer_identity)
        request = replace(
            self.request,
            expected_producer_identity_sha256=wrong_anchor,
        )
        self.assert_rejected_before_commands(
            "producer identity differs from its frozen anchor", request
        )

    def test_existing_evidence_is_never_replaced(self) -> None:
        sentinel = b"pre-existing evidence must survive byte-for-byte\n"
        self.evidence_path.write_bytes(sentinel)
        with mock.patch.object(
            audit,
            "run_limited_command",
            side_effect=AssertionError("no-replace failure ran a command"),
        ) as command:
            with self.assertRaisesRegex(audit.AuditError, "evidence already exists"):
                audit.run_audit(self.request, expectations=self.expectations)
        command.assert_not_called()
        self.assertEqual(self.evidence_path.read_bytes(), sentinel)


if __name__ == "__main__":
    unittest.main()
