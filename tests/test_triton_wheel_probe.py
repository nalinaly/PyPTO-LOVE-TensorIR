from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools" / "probe_triton_wheel.py"
    spec = importlib.util.spec_from_file_location("test_probe_triton_wheel_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_tool()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record_digest(value: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


class SyntheticWheel:
    def __init__(
        self,
        workspace: pathlib.Path,
        *,
        extra_members: dict[str, bytes] | None = None,
        symlink_member: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.wheel = (
            workspace
            / "artifacts"
            / "triton-3.7.1+git5d6048aa-cp314-cp314-linux_x86_64.whl"
        )
        self.wheel.parent.mkdir()
        members: dict[str, bytes] = {
            "triton/__init__.py": b'__version__ = "3.7.1"\n',
            "triton/_C/__init__.py": b"",
            "triton/_C/libtriton.so": b"synthetic-libtriton\x00",
            "triton/backends/__init__.py": b"",
            "triton/backends/nvidia/__init__.py": b"",
            "triton/backends/nvidia/bin/ptxas-blackwell": b"synthetic-ptxas\x00",
            f"{probe.DIST_INFO}/METADATA": (
                b"Metadata-Version: 2.4\n"
                b"Name: triton\n"
                b"Version: 3.7.1+git5d6048aa\n\n"
            ),
            f"{probe.DIST_INFO}/WHEEL": (
                b"Wheel-Version: 1.0\n"
                b"Generator: synthetic-test\n"
                b"Root-Is-Purelib: false\n"
                b"Tag: cp314-cp314-linux_x86_64\n\n"
            ),
        }
        members.update(extra_members or {})
        record_buffer = io.StringIO(newline="")
        writer = csv.writer(record_buffer, lineterminator="\n")
        for name, value in sorted(members.items()):
            writer.writerow((name, record_digest(value), str(len(value))))
        writer.writerow((probe.RECORD_PATH, "", ""))
        members[probe.RECORD_PATH] = record_buffer.getvalue().encode("utf-8")
        self.members = members

        with zipfile.ZipFile(self.wheel, "w", compression=zipfile.ZIP_STORED) as wheel:
            for name, value in members.items():
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                if name == symlink_member:
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                else:
                    executable = name == probe.PTXAS_BLACKWELL_PATH
                    mode = 0o755 if executable else 0o644
                    info.external_attr = (stat.S_IFREG | mode) << 16
                wheel.writestr(info, value)

        member_records = []
        with zipfile.ZipFile(self.wheel) as wheel:
            for info in sorted(wheel.infolist(), key=lambda item: item.filename):
                value = wheel.read(info)
                member_records.append(
                    {
                        "compressed_bytes": info.compress_size,
                        "compression": "stored",
                        "crc32": f"{info.CRC:08x}",
                        "mode": f"{stat.S_IMODE((info.external_attr >> 16) & 0xFFFF):04o}",
                        "path": info.filename,
                        "sha256": digest_bytes(value),
                        "size": len(value),
                    }
                )
        by_path = {record["path"]: record for record in member_records}
        ptxas = by_path[probe.PTXAS_BLACKWELL_PATH]
        libtriton = "triton/_C/libtriton.so"
        relative_wheel = self.wheel.relative_to(workspace).as_posix()
        self.audit_document = {
            "acceptance": "accepted",
            "audit": "triton-workspace-wheel",
            "expectations": {
                "distribution_version": "3.7.1+git5d6048aa",
                "module_version": "3.7.1",
            },
            "schema_version": 1,
            "wheel": {
                "archive": {
                    "expanded_bytes": sum(record["size"] for record in member_records),
                    "members": member_records,
                    "members_count": len(member_records),
                },
                "distribution_metadata": {
                    "metadata_version": "2.4",
                    "name": "triton",
                    "version": "3.7.1+git5d6048aa",
                },
                "elf_paths": [libtriton, probe.PTXAS_BLACKWELL_PATH],
                "filename": self.wheel.name,
                "module_version": "3.7.1",
                "path": relative_wheel,
                "record": {
                    "entries": [
                        {
                            "path": record["path"],
                            "sha256": record["sha256"],
                            "size": record["size"],
                        }
                        for record in member_records
                    ],
                    "entries_count": len(member_records),
                    "path": probe.RECORD_PATH,
                    "sha256": by_path[probe.RECORD_PATH]["sha256"],
                    "size": by_path[probe.RECORD_PATH]["size"],
                },
                "required_resources": {
                    "nvidia_tools": {
                        "ptxas-blackwell": {
                            "expected_version": "13.1.80",
                            "path": probe.PTXAS_BLACKWELL_PATH,
                            "sha256": ptxas["sha256"],
                            "size": ptxas["size"],
                        }
                    }
                },
                "sha256": probe.sha256_file(self.wheel),
                "size": self.wheel.stat().st_size,
                "wheel_metadata": {
                    "root_is_purelib": False,
                    "tags": ["cp314-cp314-linux_x86_64"],
                    "wheel_version": "1.0",
                },
            },
        }
        self.audit = workspace / "evidence" / "triton-wheel-audit.json"
        self.audit.parent.mkdir()
        self.audit.write_text(probe.canonical_json(self.audit_document))
        self.audit_sha256 = probe.sha256_file(self.audit)

    def anchor(self) -> dict[str, object]:
        document, raw = probe.load_canonical_json(self.audit, "synthetic audit")
        return probe.validate_audit_anchor(
            document,
            raw,
            expected_evidence_sha256=self.audit_sha256,
            wheel_path=self.wheel,
            workspace=self.workspace,
            limits=probe.ProbeLimits(),
        )


class TritonWheelProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = pathlib.Path(self.temporary.name).resolve()
        self.process_guard = mock.patch.object(
            probe.subprocess,
            "run",
            side_effect=AssertionError("unit test attempted to spawn a process"),
        )
        self.process_guard.start()
        self.addCleanup(self.process_guard.stop)

    def make_scheme(self) -> probe.InstallScheme:
        prefix = self.workspace / "probe"
        prefix.mkdir()
        return probe.InstallScheme(
            prefix=prefix,
            purelib=prefix / "lib/python3.14/site-packages",
            platlib=prefix / "lib/python3.14/site-packages",
            scripts=prefix / "bin",
            headers=prefix / "include/site/python3.14/triton",
            data=prefix,
            python_version=(3, 14, 6),
        )

    def install(
        self,
    ) -> tuple[
        SyntheticWheel,
        probe.InstallScheme,
        dict[str, object],
        dict[str, object],
    ]:
        synthetic = SyntheticWheel(self.workspace)
        anchor = synthetic.anchor()
        scheme = self.make_scheme()
        installation = probe.install_audited_wheel(
            synthetic.wheel,
            scheme,
            anchor,
            limits=probe.ProbeLimits(),
        )
        return synthetic, scheme, anchor, installation

    def valid_runtime_report(
        self,
        scheme: probe.InstallScheme,
        torch_site: pathlib.Path,
        key: str = "3.7.1-synthetic-key",
    ) -> dict[str, object]:
        libtriton = scheme.platlib / "triton/_C/libtriton.so"
        ptxas = scheme.platlib / probe.PTXAS_BLACKWELL_PATH
        return {
            "backend": {
                "arch": "sm120",
                "class": "triton.backends.nvidia.compiler.CUDABackend",
                "target": ["cuda", 120, 32],
            },
            "distribution": {
                "dist_info": str(scheme.platlib / probe.DIST_INFO),
                "name": "triton",
                "version": "3.7.1+git5d6048aa",
            },
            "editable": {
                "carriers": {
                    "meta_path": [],
                    "path_hooks": [],
                    "path_importer_cache": [],
                },
                "loaded_modules": [],
            },
            "keys": {
                "direct": key,
                "torch_compat": key,
                "torch_inductor": key,
            },
            "libtriton_maps": [str(libtriton)],
            "module_paths": {
                "triton": [str(scheme.platlib / "triton/__init__.py")],
                "triton._C.libtriton": [str(libtriton)],
            },
            "native_submodules": [],
            "module_version": "3.7.1",
            "ptxas_blackwell": {
                "path": str(ptxas),
                "reported_release": "13.1",
            },
            "sys_path": [str(scheme.platlib), str(torch_site)],
            "torch": {
                "cuda": "13.0",
                "file": str(torch_site / "torch/__init__.py"),
                "git_version": probe.TORCH_GIT_VERSION,
                "has_triton_package": True,
                "hip": None,
                "triton_compat_has_triton": True,
                "triton_compat_symbols": {
                    "Config": True,
                    "CompiledKernel": True,
                    "GPUTarget": True,
                    "JITFunction": True,
                    "tl": True,
                    "triton_key": True,
                },
                "triton_version": [3, 7],
                "version": "2.13.0+cu130",
            },
        }

    def make_torch_site(self) -> pathlib.Path:
        torch_site = self.workspace / "torch-site"
        (torch_site / "torch").mkdir(parents=True)
        (torch_site / "torch/__init__.py").write_text("")
        torch_dist = torch_site / "torch-2.13.0+cu130.dist-info"
        torch_dist.mkdir()
        (torch_dist / "METADATA").write_text(
            "Metadata-Version: 2.4\n"
            "Name: torch\n"
            "Version: 2.13.0+cu130\n\n"
        )
        return torch_site

    def test_exact_audit_installs_owned_bytes_and_generated_direct_url(self) -> None:
        synthetic, scheme, anchor, installation = self.install()
        verification = probe.verify_installed_wheel(
            scheme, synthetic.wheel, anchor, installation
        )
        self.assertEqual(verification["editable_artifacts"], [])
        self.assertEqual(
            verification["entries_count"],
            len(synthetic.members) + 1,
        )
        direct = installation["direct_url"]["document"]
        self.assertEqual(
            direct["archive_info"]["hashes"]["sha256"],
            probe.sha256_file(synthetic.wheel),
        )
        self.assertEqual(direct["url"], synthetic.wheel.as_uri())
        ptxas = scheme.platlib / probe.PTXAS_BLACKWELL_PATH
        self.assertTrue(ptxas.stat().st_mode & stat.S_IXUSR)
        for member in installation["archive_members"]:
            source = synthetic.members[member["archive_path"]]
            installed = scheme.platlib / member["installed_path"]
            self.assertEqual(installed.read_bytes(), source)

    def test_audit_evidence_sha_is_a_mandatory_anchor(self) -> None:
        synthetic = SyntheticWheel(self.workspace)
        document, raw = probe.load_canonical_json(synthetic.audit, "audit")
        with self.assertRaisesRegex(probe.ProbeError, "SHA256 anchor"):
            probe.validate_audit_anchor(
                document,
                raw,
                expected_evidence_sha256="0" * 64,
                wheel_path=synthetic.wheel,
                workspace=self.workspace,
                limits=probe.ProbeLimits(),
            )

    def test_wheel_tamper_after_audit_is_rejected(self) -> None:
        synthetic = SyntheticWheel(self.workspace)
        document, raw = probe.load_canonical_json(synthetic.audit, "audit")
        with synthetic.wheel.open("ab") as sink:
            sink.write(b"tamper")
        with self.assertRaisesRegex(probe.ProbeError, "wheel bytes differ"):
            probe.validate_audit_anchor(
                document,
                raw,
                expected_evidence_sha256=synthetic.audit_sha256,
                wheel_path=synthetic.wheel,
                workspace=self.workspace,
                limits=probe.ProbeLimits(),
            )

    def test_noncanonical_and_duplicate_key_evidence_are_rejected(self) -> None:
        noncanonical = self.workspace / "noncanonical.json"
        noncanonical.write_text('{"b": 1, "a": 2}\n')
        with self.assertRaisesRegex(probe.ProbeError, "not canonical"):
            probe.load_canonical_json(noncanonical, "evidence")
        duplicate = self.workspace / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}\n')
        with self.assertRaisesRegex(probe.ProbeError, "duplicate JSON key"):
            probe.load_canonical_json(duplicate, "evidence")

    def test_unsafe_archive_path_is_rejected_even_when_claimed(self) -> None:
        synthetic = SyntheticWheel(
            self.workspace, extra_members={"../escape.py": b"bad"}
        )
        document, raw = probe.load_canonical_json(synthetic.audit, "audit")
        with self.assertRaisesRegex(probe.ProbeError, "path is unsafe"):
            probe.validate_audit_anchor(
                document,
                raw,
                expected_evidence_sha256=synthetic.audit_sha256,
                wheel_path=synthetic.wheel,
                workspace=self.workspace,
                limits=probe.ProbeLimits(),
            )

    def test_symlink_wheel_member_is_rejected_by_safe_installer(self) -> None:
        synthetic = SyntheticWheel(
            self.workspace, symlink_member="triton/backends/__init__.py"
        )
        anchor = synthetic.anchor()
        scheme = self.make_scheme()
        with self.assertRaisesRegex(probe.ProbeError, "non-regular wheel member"):
            probe.install_audited_wheel(
                synthetic.wheel,
                scheme,
                anchor,
                limits=probe.ProbeLimits(),
            )

    def test_tampered_installed_byte_is_not_record_owned(self) -> None:
        synthetic, scheme, anchor, installation = self.install()
        (scheme.platlib / "triton/__init__.py").write_text("tampered\n")
        with self.assertRaisesRegex(probe.ProbeError, "byte ownership mismatch"):
            probe.verify_installed_wheel(
                scheme, synthetic.wheel, anchor, installation
            )

    def test_unowned_triton_file_is_rejected(self) -> None:
        synthetic, scheme, anchor, installation = self.install()
        (scheme.platlib / "triton/unowned.py").write_text("bad\n")
        with self.assertRaisesRegex(probe.ProbeError, "not RECORD-owned"):
            probe.verify_installed_wheel(
                scheme, synthetic.wheel, anchor, installation
            )

    def test_editable_pth_and_external_direct_url_are_rejected(self) -> None:
        synthetic, scheme, anchor, installation = self.install()
        editable = scheme.platlib / "__editable__.triton.pth"
        editable.write_text("/external/triton\n")
        with self.assertRaisesRegex(probe.ProbeError, "editable/pth"):
            probe.verify_installed_wheel(
                scheme, synthetic.wheel, anchor, installation
            )
        editable.unlink()
        direct = scheme.platlib / probe.DIRECT_URL_PATH
        direct.write_text(
            probe.canonical_json(
                {
                    "dir_info": {"editable": True},
                    "url": "file:///external/triton",
                }
            )
        )
        with self.assertRaisesRegex(probe.ProbeError, "not the audited workspace wheel"):
            probe.verify_installed_wheel(
                scheme, synthetic.wheel, anchor, installation
            )

    def test_runtime_report_proves_module_maps_torch_and_sm120_tool(self) -> None:
        _, scheme, anchor, _ = self.install()
        torch_site = self.make_torch_site()
        report = self.valid_runtime_report(scheme, torch_site)
        normalized = probe.validate_runtime_report(
            report, scheme=scheme, torch_site=torch_site, anchor=anchor
        )
        self.assertEqual(
            normalized["ptxas_blackwell"]["path"],
            "$PROBE_PREFIX/lib/python3.14/site-packages/"
            "triton/backends/nvidia/bin/ptxas-blackwell",
        )
        self.assertEqual(
            normalized["libtriton_maps"],
            ["$PROBE_PREFIX/lib/python3.14/site-packages/triton/_C/libtriton.so"],
        )
        self.assertTrue(normalized["torch"]["has_triton_package"])

    def test_runtime_rejects_external_module_and_libtriton_maps(self) -> None:
        _, scheme, anchor, _ = self.install()
        torch_site = self.make_torch_site()
        external = self.workspace / "external-libtriton.so"
        external.write_bytes(b"outside")
        report = self.valid_runtime_report(scheme, torch_site)
        report["module_paths"]["triton.bad"] = [str(external)]
        with self.assertRaisesRegex(probe.ProbeError, "module escaped"):
            probe.validate_runtime_report(
                report, scheme=scheme, torch_site=torch_site, anchor=anchor
            )
        report = self.valid_runtime_report(scheme, torch_site)
        report["libtriton_maps"] = [str(external)]
        with self.assertRaisesRegex(probe.ProbeError, "not wheel-owned"):
            probe.validate_runtime_report(
                report, scheme=scheme, torch_site=torch_site, anchor=anchor
            )

    def test_runtime_rejects_wrong_ptxas_and_disagreeing_keys(self) -> None:
        _, scheme, anchor, _ = self.install()
        torch_site = self.make_torch_site()
        report = self.valid_runtime_report(scheme, torch_site)
        report["ptxas_blackwell"]["reported_release"] = "12.8"
        with self.assertRaisesRegex(probe.ProbeError, "ptxas-blackwell"):
            probe.validate_runtime_report(
                report, scheme=scheme, torch_site=torch_site, anchor=anchor
            )
        report = self.valid_runtime_report(scheme, torch_site)
        report["keys"]["torch_inductor"] = "different"
        with self.assertRaisesRegex(probe.ProbeError, "key implementations disagree"):
            probe.validate_runtime_report(
                report, scheme=scheme, torch_site=torch_site, anchor=anchor
            )

    def test_native_submodules_are_explicitly_owned_by_root_libtriton(self) -> None:
        _, scheme, anchor, _ = self.install()
        torch_site = self.make_torch_site()
        report = self.valid_runtime_report(scheme, torch_site)
        native_name = "triton._C.libtriton.nvidia.passes.ttnvgpuir"
        report["native_submodules"] = [native_name]
        report["module_paths"][native_name] = list(
            report["module_paths"]["triton._C.libtriton"]
        )
        normalized = probe.validate_runtime_report(
            report, scheme=scheme, torch_site=torch_site, anchor=anchor
        )
        self.assertEqual(normalized["native_submodules"], [native_name])
        report["module_paths"].pop(native_name)
        with self.assertRaisesRegex(probe.ProbeError, "not attributed"):
            probe.validate_runtime_report(
                report, scheme=scheme, torch_site=torch_site, anchor=anchor
            )

    def test_runtime_probe_uses_two_isolation_layers_and_scrubbed_environment(self) -> None:
        _, scheme, anchor, _ = self.install()
        torch_site = self.make_torch_site()
        torch_runtime_view, _ = probe.create_torch_runtime_view(
            torch_site, scheme.prefix
        )
        python = scheme.prefix / "bin/python"
        python.parent.mkdir(exist_ok=True)
        python.write_bytes(b"synthetic-python")
        python.chmod(0o755)
        raw_report = self.valid_runtime_report(scheme, torch_site)
        with mock.patch.object(
            probe, "run_json_command", return_value=raw_report
        ) as command:
            probe.run_runtime_probe(
                python,
                scheme,
                torch_site,
                torch_runtime_view,
                anchor,
                limits=probe.ProbeLimits(),
            )
        argv = command.call_args.args[0]
        environment = command.call_args.kwargs["environment"]
        self.assertEqual(argv[1:4], ["-I", "-B", "-S"])
        self.assertEqual(
            argv[-4:],
            [
                str(scheme.platlib),
                str(torch_runtime_view),
                str(scheme.prefix),
                str(torch_site),
            ],
        )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("TRITON_PTXAS_PATH", environment)
        self.assertTrue(
            pathlib.Path(environment["TRITON_CACHE_DIR"]).is_relative_to(
                scheme.prefix
            )
        )

    def test_cp314_install_scheme_keeps_pip_compatible_headers_in_prefix(self) -> None:
        prefix = self.workspace / "fresh"
        prefix.mkdir()
        python = prefix / "bin/python"
        report = {
            "executable": str(python),
            "paths": {
                "data": str(prefix),
                "headers": str(prefix / "include/site/python3.14/triton"),
                "platlib": str(prefix / "lib/python3.14/site-packages"),
                "purelib": str(prefix / "lib/python3.14/site-packages"),
                "scripts": str(prefix / "bin"),
            },
            "prefix": str(prefix),
            "version": [3, 14, 6],
        }
        with mock.patch.object(
            probe, "run_json_command", return_value=report
        ) as command:
            scheme = probe.query_install_scheme(
                python, prefix, limits=probe.ProbeLimits()
            )
        self.assertEqual(
            scheme.headers, prefix / "include/site/python3.14/triton"
        )
        self.assertEqual(command.call_args.args[0][1:4], ["-I", "-B", "-S"])
        self.assertIn('pathlib.Path(sys.prefix)', probe.SCHEME_PROGRAM)

    def test_torch_runtime_view_hides_ambient_site_and_triton_carriers(self) -> None:
        scheme = self.make_scheme()
        torch_site = self.make_torch_site()
        (torch_site / "typing_extensions.py").write_text("value = 1\n")
        (torch_site / "__editable__.triton.pth").write_text("/external/triton\n")
        (torch_site / "__editable___triton_finder.py").write_text("bad = 1\n")
        (torch_site / "triton-3.7.1.dist-info").mkdir()
        (torch_site / "conda-site.pth").write_text("import site\n")
        editable_dist = torch_site / "candidate-1.dist-info"
        editable_dist.mkdir()
        (editable_dist / "direct_url.json").write_text(
            json.dumps(
                {
                    "dir_info": {"editable": True},
                    "url": "file:///external/source",
                }
            )
        )
        view, evidence = probe.create_torch_runtime_view(torch_site, scheme.prefix)
        self.assertTrue((view / "torch").is_symlink())
        self.assertTrue((view / "torch-2.13.0+cu130.dist-info").is_symlink())
        self.assertTrue((view / "typing_extensions.py").is_symlink())
        self.assertFalse((view / "triton-3.7.1.dist-info").exists())
        self.assertFalse((view / "__editable__.triton.pth").exists())
        self.assertFalse((view / "candidate-1.dist-info").exists())
        reasons = {
            item["name"]: item["reason"]
            for item in evidence["excluded_entries"]
        }
        self.assertEqual(
            reasons["triton-3.7.1.dist-info"], "ambient-triton-carrier"
        )
        self.assertEqual(
            reasons["candidate-1.dist-info"], "editable-direct-url-metadata"
        )
        self.assertEqual(
            evidence["projection_entries_count"], len(evidence["included_entries"])
        )
        self.assertEqual(
            evidence["projection_entries_sha256"],
            digest_bytes(
                probe.canonical_json(evidence["projection_entries"]).encode("ascii")
            ),
        )

    def test_post_runtime_input_tampering_is_rejected_exactly(self) -> None:
        for name in (
            "base-python",
            "environment-lock",
            "torch-tree",
            "torch-tree-empty-directory",
            "torch-runtime-view",
        ):
            with self.subTest(name=name):
                case = self.workspace / name
                case.mkdir()
                base_python = case / "base-python"
                base_python.write_bytes(b"synthetic-python\x00")
                base_python.chmod(0o755)
                torch_site = case / "site-packages"
                (torch_site / "torch").mkdir(parents=True)
                (torch_site / "torch/__init__.py").write_text("# torch\n")
                dist_info = torch_site / "torch-2.13.0+cu130.dist-info"
                dist_info.mkdir()
                (dist_info / "METADATA").write_text(
                    "Metadata-Version: 2.4\n"
                    "Name: torch\n"
                    "Version: 2.13.0+cu130\n\n"
                )
                tree = probe.torch_tree_identity(
                    torch_site, torch_site / "torch", dist_info
                )
                environment_lock = case / "ENVIRONMENT.lock"
                environment_document = {
                    "cuda": probe.TORCH_CUDA_VERSION,
                    "destination_prefix": case.relative_to(self.workspace).as_posix(),
                    "hip": None,
                    "torch": probe.TORCH_VERSION,
                    "torch_git": probe.TORCH_GIT_VERSION,
                    **tree,
                }
                environment_lock.write_text(
                    probe.canonical_json(environment_document)
                )
                prefix = case / "probe"
                prefix.mkdir()
                view, _ = probe.create_torch_runtime_view(torch_site, prefix)
                with (
                    mock.patch.object(
                        probe, "TORCH_TREE_SHA256", tree["torch_tree_sha256"]
                    ),
                    mock.patch.object(
                        probe, "TORCH_TREE_FILES", tree["torch_tree_files"]
                    ),
                    mock.patch.object(
                        probe, "TORCH_TREE_BYTES", tree["torch_tree_bytes"]
                    ),
                ):
                    _, before = probe.capture_runtime_input_identity(
                        base_python=base_python,
                        environment_lock_path=environment_lock,
                        torch_site=torch_site,
                        torch_runtime_view=view,
                        workspace=self.workspace,
                    )
                    if name == "base-python":
                        with base_python.open("ab") as output:
                            output.write(b"tamper")
                    elif name == "environment-lock":
                        environment_document["unexpected"] = "tamper"
                        environment_lock.write_text(
                            probe.canonical_json(environment_document)
                        )
                    elif name == "torch-tree":
                        with (torch_site / "torch/__init__.py").open("ab") as output:
                            output.write(b"# tamper\n")
                    elif name == "torch-tree-empty-directory":
                        (torch_site / "torch/unexpected-empty").mkdir()
                    else:
                        (view / "torch").unlink()
                        (view / "torch").symlink_to(dist_info, target_is_directory=True)
                    with self.assertRaises(probe.ProbeError):
                        probe.require_runtime_input_identity_unchanged(
                            before,
                            base_python=base_python,
                            environment_lock_path=environment_lock,
                            torch_site=torch_site,
                            torch_runtime_view=view,
                            workspace=self.workspace,
                        )

    def test_run_probe_requires_stable_key_from_two_independent_reports(self) -> None:
        synthetic = SyntheticWheel(self.workspace)
        base_python = self.workspace / "base-python"
        base_python.write_bytes(b"python")
        base_python.chmod(0o755)
        torch_site = self.workspace / "torch-site"
        (torch_site / "torch").mkdir(parents=True)
        (torch_site / "torch/__init__.py").write_text("")
        torch_dist = torch_site / "torch-2.13.0+cu130.dist-info"
        torch_dist.mkdir()
        (torch_dist / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: torch\nVersion: 2.13.0+cu130\n\n"
        )
        prefix = self.workspace / "fresh-probe"
        evidence = self.workspace / "evidence/probe.json"
        environment_lock = self.workspace / "ENVIRONMENT.lock"
        environment_lock.write_text(
            probe.canonical_json(
                {
                    "python": "3.14.6 | synthetic",
                    "python_abi": "cp314",
                    "python_executable": str(base_python.resolve()),
                    "python_implementation": "CPython",
                }
            )
        )
        request = probe.ProbeRequest(
            workspace=self.workspace,
            wheel=synthetic.wheel,
            wheel_audit_evidence=synthetic.audit,
            expected_wheel_audit_evidence_sha256=synthetic.audit_sha256,
            base_python=base_python,
            torch_site_packages=torch_site,
            environment_lock=environment_lock,
            probe_prefix=prefix,
            evidence=evidence,
        )

        def fake_create(_python, created_prefix, *, limits):
            created_prefix.mkdir()
            executable = created_prefix / "bin/python"
            executable.parent.mkdir()
            executable.write_bytes(b"python")
            executable.chmod(0o755)
            return executable

        scheme_holder: dict[str, probe.InstallScheme] = {}

        def fake_scheme(_python, created_prefix, *, limits):
            scheme = probe.InstallScheme(
                prefix=created_prefix,
                purelib=created_prefix / "site",
                platlib=created_prefix / "site",
                scripts=created_prefix / "bin",
                headers=created_prefix / "include",
                data=created_prefix,
                python_version=(3, 14, 6),
            )
            scheme_holder["value"] = scheme
            return scheme

        reports = [
            {"keys": {"torch_inductor": "first"}},
            {"keys": {"torch_inductor": "second"}},
        ]
        with (
            mock.patch.object(probe, "create_fresh_venv", side_effect=fake_create),
            mock.patch.object(probe, "query_install_scheme", side_effect=fake_scheme),
            mock.patch.object(probe, "install_audited_wheel", return_value={}),
            mock.patch.object(probe, "verify_installed_wheel", return_value={}),
            mock.patch.object(
                probe,
                "validate_torch_site",
                return_value={"version": probe.TORCH_VERSION},
            ),
            mock.patch.object(
                probe,
                "BASE_PYTHON_SHA256",
                probe.sha256_file(base_python),
            ),
            mock.patch.object(probe, "run_runtime_probe", side_effect=reports) as runtime,
        ):
            with self.assertRaisesRegex(probe.ProbeError, "not stable"):
                probe.run_probe(request)
        self.assertEqual(runtime.call_count, 2)
        self.assertFalse(evidence.exists())

    def test_fresh_prefix_rejects_existing_symlink_and_workspace_escape(self) -> None:
        existing = self.workspace / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(probe.ProbeError, "not fresh"):
            probe.require_fresh_workspace_prefix(existing, self.workspace)
        target = self.workspace / "target"
        target.mkdir()
        link = self.workspace / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(probe.ProbeError, "not fresh"):
            probe.require_fresh_workspace_prefix(link, self.workspace)
        with self.assertRaisesRegex(probe.ProbeError, "child of the workspace"):
            probe.require_fresh_workspace_prefix(
                self.workspace.parent / "outside", self.workspace
            )

    def test_atomic_evidence_is_canonical_read_only_and_no_replace(self) -> None:
        evidence = self.workspace / "evidence.json"
        value = {"z": [2, 1], "a": "accepted"}
        digest = probe.publish_canonical_json_no_replace(evidence, value)
        self.assertEqual(evidence.read_text(), probe.canonical_json(value))
        self.assertEqual(digest, probe.sha256_file(evidence))
        self.assertEqual(stat.S_IMODE(evidence.stat().st_mode), 0o444)
        with self.assertRaisesRegex(probe.ProbeError, "already exists"):
            probe.publish_canonical_json_no_replace(evidence, {"replacement": True})
        self.assertEqual(evidence.read_text(), probe.canonical_json(value))

    def test_probe_program_has_no_gpu_workload_or_ambient_site_processing(self) -> None:
        compile(probe.RUNTIME_PROBE_PROGRAM, "runtime-probe", "exec")
        self.assertNotIn("torch.cuda", probe.RUNTIME_PROBE_PROGRAM)
        self.assertNotIn("vector", probe.RUNTIME_PROBE_PROGRAM.lower())
        self.assertIn("sys.path.insert(0, str(probe_site))", probe.RUNTIME_PROBE_PROGRAM)
        self.assertIn(
            "sys.path.append(str(torch_runtime_view))",
            probe.RUNTIME_PROBE_PROGRAM,
        )
        self.assertIn("get_ptxas(120)", probe.RUNTIME_PROBE_PROGRAM)
        self.assertIn("torch_triton_key()", probe.RUNTIME_PROBE_PROGRAM)


if __name__ == "__main__":
    unittest.main()
