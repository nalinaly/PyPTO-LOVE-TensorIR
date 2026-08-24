from __future__ import annotations

import base64
import csv
from dataclasses import replace as dataclass_replace
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replace = load_tool(
    "test_replace_triton_environment_tool",
    "tools/replace_triton_environment.py",
)
probe = replace.probe
REAL_QUERY_TARGET_SCHEME = replace.query_target_scheme


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record_digest(value: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


class SyntheticWheel:
    def __init__(self, workspace: pathlib.Path) -> None:
        self.path = workspace / "artifacts" / (
            "triton-3.7.1+git5d6048aa-cp314-cp314-linux_x86_64.whl"
        )
        self.path.parent.mkdir(parents=True)
        members: dict[str, bytes] = {
            "triton/__init__.py": b'__version__ = "3.7.1"\n',
            "triton/_C/__init__.py": b"",
            "triton/_C/libtriton.so": b"new-synthetic-libtriton\x00",
            "triton/backends/__init__.py": b"",
            "triton/backends/compiler.py": b"class GPUTarget: pass\n",
            "triton/backends/nvidia/__init__.py": b"",
            "triton/backends/nvidia/bin/ptxas-blackwell": b"synthetic-ptxas\x00",
            f"{probe.DIST_INFO}/METADATA": (
                b"Metadata-Version: 2.4\n"
                b"Name: triton\n"
                b"Version: 3.7.1+git5d6048aa\n\n"
            ),
            f"{probe.DIST_INFO}/WHEEL": (
                b"Wheel-Version: 1.0\n"
                b"Generator: replacement-test\n"
                b"Root-Is-Purelib: false\n"
                b"Tag: cp314-cp314-linux_x86_64\n\n"
            ),
        }
        record_buffer = io.StringIO(newline="")
        writer = csv.writer(record_buffer, lineterminator="\n")
        for name, value in sorted(members.items()):
            writer.writerow((name, record_digest(value), str(len(value))))
        writer.writerow((probe.RECORD_PATH, "", ""))
        members[probe.RECORD_PATH] = record_buffer.getvalue().encode("utf-8")
        self.members = members
        with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_STORED) as wheel:
            for name, value in members.items():
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                executable = name == probe.PTXAS_BLACKWELL_PATH
                info.external_attr = (
                    stat.S_IFREG | (0o755 if executable else 0o644)
                ) << 16
                wheel.writestr(info, value)

        member_records: list[dict[str, object]] = []
        with zipfile.ZipFile(self.path) as wheel:
            for info in sorted(wheel.infolist(), key=lambda item: item.filename):
                value = wheel.read(info)
                member_records.append(
                    {
                        "compressed_bytes": info.compress_size,
                        "compression": "stored",
                        "crc32": f"{info.CRC:08x}",
                        "mode": f"{stat.S_IMODE((info.external_attr >> 16) & 0xFFFF):04o}",
                        "path": info.filename,
                        "sha256": digest(value),
                        "size": len(value),
                    }
                )
        by_path = {str(item["path"]): item for item in member_records}
        ptxas = by_path[probe.PTXAS_BLACKWELL_PATH]
        libtriton = "triton/_C/libtriton.so"
        self.audit_document = {
            "acceptance": "accepted",
            "audit": probe.AUDIT_NAME,
            "expectations": {
                "distribution_version": probe.TRITON_DISTRIBUTION_VERSION,
                "module_version": probe.TRITON_MODULE_VERSION,
            },
            "schema_version": 1,
            "wheel": {
                "archive": {
                    "expanded_bytes": sum(int(item["size"]) for item in member_records),
                    "members": member_records,
                    "members_count": len(member_records),
                },
                "distribution_metadata": {
                    "metadata_version": "2.4",
                    "name": "triton",
                    "version": probe.TRITON_DISTRIBUTION_VERSION,
                },
                "elf_paths": [libtriton, probe.PTXAS_BLACKWELL_PATH],
                "filename": self.path.name,
                "module_version": probe.TRITON_MODULE_VERSION,
                "path": self.path.relative_to(workspace).as_posix(),
                "record": {
                    "entries": [
                        {
                            "path": item["path"],
                            "sha256": item["sha256"],
                            "size": item["size"],
                        }
                        for item in member_records
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
                "sha256": probe.sha256_file(self.path),
                "size": self.path.stat().st_size,
                "wheel_metadata": {
                    "root_is_purelib": False,
                    "tags": ["cp314-cp314-linux_x86_64"],
                    "wheel_version": "1.0",
                },
            },
        }
        self.audit = workspace / "evidence" / "wheel-audit.json"
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        self.audit.write_text(probe.canonical_json(self.audit_document))
        self.audit_sha = probe.sha256_file(self.audit)

    def anchor(self, workspace: pathlib.Path) -> dict[str, object]:
        document, raw = probe.load_canonical_json(self.audit, "test audit")
        return probe.validate_audit_anchor(
            document,
            raw,
            expected_evidence_sha256=self.audit_sha,
            wheel_path=self.path,
            workspace=workspace,
            limits=probe.ProbeLimits(),
        )


class ReplacementFixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        for name in ("builds", "evidence", "state"):
            (self.workspace / name).mkdir(exist_ok=True)
        (self.workspace / "runs").mkdir()
        self.prefix = self.workspace / "envs" / "pypto-nvidia"
        self.site = self.prefix / "lib/python3.14/site-packages"
        self.site.mkdir(parents=True)
        (self.prefix / "bin").mkdir()
        self.base_python = self.prefix / "bin/python3.14"
        os.link(ROOT / "envs/pypto-nvidia/bin/python3.14", self.base_python)
        (self.prefix / "bin/python").symlink_to(self.base_python)
        self.scheme = probe.InstallScheme(
            prefix=self.prefix,
            purelib=self.site,
            platlib=self.site,
            scripts=self.prefix / "bin",
            headers=self.prefix / "include/site/python3.14/triton",
            data=self.prefix,
            python_version=(3, 14, 6),
        )
        self.external = root / "external-triton"
        self.package = self.external / "python/triton"
        (self.package / "_C").mkdir(parents=True)
        (self.package / "__init__.py").write_text('__version__ = "3.8.0"\n')
        self.old_native = self.package / "_C/libtriton.so"
        self.old_native.write_bytes(b"old-external-native\x00")
        self.old_native.chmod(0o755)
        self.old_native_bytes = self.old_native.read_bytes()
        self._initialize_external_git()
        self._make_old_editable()
        self.wheel = SyntheticWheel(self.workspace)
        self.anchor = self.wheel.anchor(self.workspace)
        self._make_chain()
        self.backup = self.workspace / "builds/replacement-backup"
        self.final_evidence = self.workspace / "evidence/replacement.json"
        self.request = replace.ReplacementRequest(
            workspace=self.workspace,
            prefix=self.prefix,
            wheel=self.wheel.path,
            wheel_audit_evidence=self.wheel.audit,
            expected_wheel_audit_evidence_sha256=self.wheel.audit_sha,
            wheel_probe_evidence=self.probe_evidence,
            expected_wheel_probe_evidence_sha256=self.probe_sha,
            gpu_smoke_evidence=self.smoke_evidence,
            expected_gpu_smoke_evidence_sha256=self.smoke_sha,
            environment_lock=self.environment_lock,
            expected_environment_lock_sha256=self.lock_sha,
            backup_root=self.backup,
            evidence=self.final_evidence,
        )

    def _initialize_external_git(self) -> None:
        for arguments in (
            ["init", "-q"],
            ["config", "user.name", "Synthetic Test"],
            ["config", "user.email", "synthetic@example.invalid"],
            ["add", "."],
            ["commit", "-qm", "synthetic external Triton"],
            ["remote", "add", "origin", "https://example.invalid/triton.git"],
        ):
            subprocess.run(
                ["/usr/bin/git", "-C", str(self.external), *arguments],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={
                    "HOME": str(self.root),
                    "LANG": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                },
            )

    def _make_old_editable(self) -> None:
        self.finder_module = "__editable___triton_3_7_1_git5d6048aa_finder"
        finder = self.site / f"{self.finder_module}.py"
        finder.write_text(
            "MAPPING: dict[str, str] = "
            + repr({"triton": str(self.package)})
            + "\nNAMESPACES: dict[str, list[str]] = {}\n"
            + "def install():\n    pass\n"
        )
        finder.chmod(0o600)
        pth = self.site / "__editable__.triton-3.7.1+git5d6048aa.pth"
        pth.write_text(f"import {self.finder_module}; {self.finder_module}.install()\n")
        pycache = self.site / "__pycache__"
        pycache.mkdir()
        finder_pyc = pycache / f"{self.finder_module}.cpython-314.pyc"
        finder_pyc.write_bytes(b"old-bytecode\x00")
        dist = self.site / probe.DIST_INFO
        (dist / "licenses").mkdir(parents=True)
        files: dict[pathlib.Path, bytes] = {
            finder: finder.read_bytes(),
            pth: pth.read_bytes(),
            finder_pyc: finder_pyc.read_bytes(),
            dist / "METADATA": (
                b"Metadata-Version: 2.4\nName: triton\n"
                b"Version: 3.7.1+git5d6048aa\n\n"
            ),
            dist / "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n\n",
            dist / "direct_url.json": json.dumps(
                {
                    "dir_info": {"editable": True},
                    "url": self.external.as_uri(),
                }
            ).encode(),
            dist / "licenses/LICENSE": b"synthetic old license\n",
        }
        for path, value in files.items():
            if not path.exists():
                path.write_bytes(value)
        self.old_link = dist / "rollback-link"
        self.old_link.symlink_to("licenses/LICENSE")
        record = dist / "RECORD"
        rows: list[tuple[str, str, str]] = []
        for path, value in sorted(files.items(), key=lambda item: str(item[0])):
            relative = pathlib.PurePosixPath(os.path.relpath(path, self.site)).as_posix()
            if path == finder_pyc:
                rows.append((relative, "", ""))
            else:
                rows.append((relative, record_digest(value), str(len(value))))
        rows.append(
            (
                pathlib.PurePosixPath(os.path.relpath(self.old_link, self.site)).as_posix(),
                "",
                "",
            )
        )
        rows.append((f"{probe.DIST_INFO}/RECORD", "", ""))
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(rows)
        record.write_text(output.getvalue())
        self.dist = dist
        self.old_snapshot = self.snapshot_old()

    def snapshot_old(self) -> dict[str, tuple[object, ...]]:
        result: dict[str, tuple[object, ...]] = {}
        for path in sorted(
            [
                item
                for item in self.site.rglob("*")
                if item.is_file() or item.is_symlink()
            ],
            key=str,
        ):
            relative = path.relative_to(self.prefix).as_posix()
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink():
                result[relative] = ("symlink", mode, os.readlink(path))
            else:
                result[relative] = ("regular", mode, path.read_bytes())
        return result

    def identity(self, path: pathlib.Path) -> dict[str, object]:
        return {
            "path": path.relative_to(self.workspace).as_posix(),
            "sha256": probe.sha256_file(path),
            "size": path.stat().st_size,
        }

    def _make_chain(self) -> None:
        self.environment_lock = self.workspace / "ENVIRONMENT.lock"
        lock = {
            "destination_prefix": "envs/pypto-nvidia",
            "python_abi": "cp314",
            "python_executable": str(self.base_python),
            "schema": 1,
        }
        self.environment_lock.write_text(probe.canonical_json(lock))
        self.lock_sha = probe.sha256_file(self.environment_lock)
        (self.workspace / "VERSIONS.lock").write_text(
            "triton.producer.python_sha256=" + probe.BASE_PYTHON_SHA256 + "\n"
        )

        self.probe_prefix = self.workspace / "builds/fresh-probe"
        self.probe_prefix.mkdir()
        probe_site = self.probe_prefix / "lib/python3.14/site-packages"
        probe_scheme = probe.InstallScheme(
            prefix=self.probe_prefix,
            purelib=probe_site,
            platlib=probe_site,
            scripts=self.probe_prefix / "bin",
            headers=self.probe_prefix / "include/site/python3.14/triton",
            data=self.probe_prefix,
            python_version=(3, 14, 6),
        )
        probe.install_audited_wheel(
            self.wheel.path,
            probe_scheme,
            self.anchor,
            limits=probe.ProbeLimits(),
        )
        ptxas_token = (
            "$PROBE_PREFIX/lib/python3.14/site-packages/"
            "triton/backends/nvidia/bin/ptxas-blackwell"
        )
        process = {
            "ptxas_blackwell": {"path": ptxas_token},
            "keys": {"torch_inductor": "stable-key"},
            "torch": {
                "cuda": probe.TORCH_CUDA_VERSION,
                "git_version": probe.TORCH_GIT_VERSION,
                "hip": None,
                "version": probe.TORCH_VERSION,
            },
        }
        self.probe_document = {
            "acceptance": "accepted",
            "inputs": {
                "base_python": self.identity(self.base_python),
                "environment_lock": self.identity(self.environment_lock),
                "torch_site_packages": {
                    "path": self.site.relative_to(self.workspace).as_posix(),
                    "torch_tree_bytes": 123,
                    "torch_tree_files": 4,
                    "torch_tree_sha256": "1" * 64,
                },
                "wheel": {
                    "audit_evidence": self.identity(self.wheel.audit),
                    "filename": self.wheel.path.name,
                    **self.identity(self.wheel.path),
                },
            },
            "installation": {
                "fresh_prefix": True,
                "method": "stdlib-safe-wheel-installer",
                "prefix": self.probe_prefix.relative_to(self.workspace).as_posix(),
            },
            "probe": probe.PROBE_NAME,
            "runtime": {
                "gpu_execution": False,
                "processes": [process, process],
                "processes_count": 2,
                "triton_key": {
                    "sha256": digest(b"stable-key"),
                    "stable_across_processes": True,
                    "value": "stable-key",
                },
            },
            "schema_version": 1,
        }
        self.probe_evidence = self.workspace / "evidence/wheel-probe.json"
        self.probe_evidence.write_text(probe.canonical_json(self.probe_document))
        self.probe_sha = probe.sha256_file(self.probe_evidence)

        runner = self.workspace / "benchmarks/operators/triton_reference_sm120.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("# synthetic reference-only runner\n")
        cache = self.workspace / "builds/reference-cache"
        cache.mkdir()
        cubin = cache / "vector-add.cubin"
        cubin.write_bytes(b"synthetic cubin\x00")
        cache_artifact = {
            "path": "vector-add.cubin",
            "sha256": probe.sha256_file(cubin),
            "size": cubin.stat().st_size,
        }
        ptxas_sha = self.anchor["ptxas_blackwell"]["sha256"]
        provisional = {
            "acceptance": "gpu-execution-complete-awaiting-run-finalization",
            "inputs": {
                "base_python": self.probe_document["inputs"]["base_python"],
                "environment_lock": self.probe_document["inputs"]["environment_lock"],
                "probe_evidence": self.identity(self.probe_evidence),
                "probe_prefix": self.probe_prefix.relative_to(self.workspace).as_posix(),
                "probe_site": probe_site.relative_to(self.workspace).as_posix(),
                "runner": self.identity(runner),
                "torch_runtime_view": "builds/fresh-probe/.torch-runtime-view",
                "torch_site_packages": self.probe_document["inputs"][
                    "torch_site_packages"
                ],
                "wheel": self.probe_document["inputs"]["wheel"],
            },
            "runtime": {
                "compiled_cache": {
                    "artifacts": [cache_artifact],
                    "artifacts_count": 1,
                    "artifacts_sha256": digest(
                        probe.canonical_json([cache_artifact]).encode("ascii")
                    ),
                    "cubin_count": 1,
                    "fresh_before_run": True,
                    "path": cache.relative_to(self.workspace).as_posix(),
                    "total_bytes": cubin.stat().st_size,
                },
                "correctness": {
                    "block_size": 256,
                    "comparison": "torch.equal",
                    "dtype": "float32",
                    "equal": True,
                    "kernel": "masked-vector-add",
                    "n_elements": 65_537,
                    "reference_provider": "torch",
                },
                "device": {
                    "compute_capability": [12, 0],
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
                },
                "gpu_execution": True,
                "integrity": {
                    "before": {
                        "environment_lock": {"sha256": self.lock_sha},
                        "installed_triton": {
                            "native_bytes": 12,
                            "native_entries_count": 1,
                            "native_entries_sha256": "2" * 64,
                            "package_entries_count": 5,
                            "package_entries_sha256": "3" * 64,
                            "record_entries_count": 9,
                            "record_entries_sha256": "4" * 64,
                        },
                        "linked_inputs": {
                            "base_python": self.probe_document["inputs"][
                                "base_python"
                            ],
                            "probe_evidence": self.identity(self.probe_evidence),
                            "wheel": {
                                "audit_evidence_sha256": self.wheel.audit_sha,
                                "sha256": probe.sha256_file(self.wheel.path),
                            },
                        },
                        "torch_tree": {
                            "torch_tree_bytes": 123,
                            "torch_tree_files": 4,
                            "torch_tree_sha256": "1" * 64,
                        },
                    },
                    "stable": True,
                },
                "provenance": {
                    "editable": {
                        "carriers": {
                            "meta_path": [],
                            "path_hooks": [],
                            "path_importer_cache": [],
                        },
                        "loaded_modules": [],
                    },
                    "libtriton_maps": [
                        "$PROBE_PREFIX/lib/python3.14/site-packages/"
                        "triton/_C/libtriton.so"
                    ],
                    "module_paths": {
                        "triton": [
                            "$PROBE_PREFIX/lib/python3.14/site-packages/"
                            "triton/__init__.py"
                        ]
                    },
                    "python": {
                        "executable": "$PROBE_PREFIX/bin/python",
                        "resolved_executable": self.base_python.relative_to(
                            self.workspace
                        ).as_posix(),
                        "resolved_sha256": probe.sha256_file(self.base_python),
                        "resolved_size": self.base_python.stat().st_size,
                    },
                    "sys_path": ["$PROBE_PREFIX/lib/python3.14/site-packages"],
                    "torch_file": "$TORCH_SITE_PACKAGES/torch/__init__.py",
                    "torch_runtime": {
                        "cuda": probe.TORCH_CUDA_VERSION,
                        "file": "$TORCH_SITE_PACKAGES/torch/__init__.py",
                        "git_version": probe.TORCH_GIT_VERSION,
                        "hip": None,
                        "version": probe.TORCH_VERSION,
                    },
                },
                "ptxas_blackwell": {
                    "audited_full_version": "13.1.80",
                    "path": ptxas_token,
                    "reported_release": "13.1",
                    "sha256": ptxas_sha,
                    "wheel_owned": True,
                },
                "synchronization": {
                    "after_comparison": True,
                    "after_kernel": True,
                    "before_launch": True,
                    "error": None,
                },
                "target": {"arch": 120, "backend": "cuda", "warp_size": 32},
            },
            "schema_version": 1,
            "scope": {
                "coverage_result": False,
                "performance_result": False,
                "provider": "triton",
                "pypto_kernel": False,
                "reference_only": True,
            },
            "smoke": replace.GPU_SMOKE_NAME,
        }
        provisional["runtime"]["integrity"]["after"] = json.loads(
            probe.canonical_json(provisional["runtime"]["integrity"]["before"])
        )
        self.provisional_smoke = self.workspace / "evidence/gpu-smoke-provisional.json"
        run_id = "pypto-20990101T000000Z-123456-abcdef"
        run_dir = self.workspace / "runs" / run_id
        run_dir.mkdir()
        preflight = {
            "mode": "gpu-benchmark",
            "nvidia_compute_pids": [],
            "ok": True,
            "protected_cpu_only_coexistence_requested": False,
            "protected_heavy_processes": [],
        }
        preflight_path = run_dir / "preflight.json"
        preflight_path.write_text(probe.canonical_json(preflight))
        provisional["run_context"] = {
            "mode": "gpu-benchmark",
            "pgid": 7777,
            "pid": 7778,
            "preflight": {
                "path": preflight_path.relative_to(self.workspace).as_posix(),
                "sha256": probe.sha256_file(preflight_path),
                "size": preflight_path.stat().st_size,
            },
            "provisional_evidence_path": self.provisional_smoke.relative_to(
                self.workspace
            ).as_posix(),
            "run_id": run_id,
        }
        self.provisional_smoke.write_text(probe.canonical_json(provisional))
        command = [str(runner), "--evidence", str(self.provisional_smoke)]
        process_document = {
            "coexistence": {"requested": False},
            "coexistence_pauses": [],
            "command": command,
            "gpu_benchmark_abort": None,
            "mode": "gpu-benchmark",
            "preflight": {
                "path": str(preflight_path),
                "sha256": probe.sha256_file(preflight_path),
            },
            "pgid": 7777,
            "return_code": 0,
            "run_id": run_id,
            "status": "exited",
        }
        process_path = run_dir / "process.json"
        process_path.write_text(probe.canonical_json(process_document))
        finalizer_path = self.workspace / "tools/finalize_triton_reference_smoke.py"
        finalizer_path.parent.mkdir()
        finalizer_path.write_text("# synthetic finalizer\n")
        self.smoke_document = dict(provisional)
        self.smoke_document["acceptance"] = "accepted"
        self.smoke_document["exclusive_run"] = {
            "coexistence_pauses": [],
            "finalizer": self.identity(finalizer_path),
            "gpu_benchmark_abort": None,
            "preflight": {
                "document_sha256": probe.sha256_file(preflight_path),
                "mode": "gpu-benchmark",
                "ok": True,
                "path": preflight_path.relative_to(self.workspace).as_posix(),
            },
            "process": {
                "command_sha256": digest(
                    probe.canonical_json(command).encode("ascii")
                ),
                "document_sha256": probe.sha256_file(process_path),
                "mode": "gpu-benchmark",
                "path": process_path.relative_to(self.workspace).as_posix(),
                "return_code": 0,
                "status": "exited",
            },
            "run_id": run_id,
        }
        self.smoke_document["provisional_evidence"] = self.identity(
            self.provisional_smoke
        )
        self.smoke_evidence = self.workspace / "evidence/gpu-smoke-final.json"
        self.smoke_evidence.write_text(probe.canonical_json(self.smoke_document))
        self.smoke_sha = probe.sha256_file(self.smoke_evidence)

    def old_runtime(self) -> dict[str, object]:
        return {
            "distribution": {
                "dist_info": str(self.dist),
                "name": "triton",
                "version": probe.TRITON_DISTRIBUTION_VERSION,
            },
            "editable": {
                "carriers": {
                    "meta_path": [self.finder_module],
                    "path_hooks": [],
                    "path_importer_cache": [],
                },
                "loaded_modules": [self.finder_module],
            },
            "libtriton_maps": [str(self.old_native)],
            "module_paths": {
                "triton": [str(self.package / "__init__.py")],
                "triton._C.libtriton": [str(self.old_native)],
            },
        }


class ReplaceTritonEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        # The root suite is itself expected to run under run_isolated, whose
        # real workspace lock markers must not be misapplied to each synthetic
        # fixture workspace. Individual inheritance tests install exact fixture
        # markers explicitly.
        inherited_names = tuple(replace.INHERITED_LOCK_ENVIRONMENT.values())
        inherited_values = {
            name: os.environ.pop(name, None) for name in inherited_names
        }

        def restore_inherited_environment() -> None:
            for name in inherited_names:
                os.environ.pop(name, None)
                value = inherited_values[name]
                if value is not None:
                    os.environ[name] = value

        self.addCleanup(restore_inherited_environment)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="replace-triton-test-", dir=ROOT / "runs"
        )
        self.addCleanup(self.temporary.cleanup)
        self.fixture = ReplacementFixture(pathlib.Path(self.temporary.name).resolve())
        self.query_patch = mock.patch.object(
            replace,
            "query_target_scheme",
            return_value=(self.fixture.prefix / "bin/python", self.fixture.scheme),
        )
        self.torch_patch = mock.patch.object(
            replace,
            "capture_torch_identity",
            return_value={"torch_tree_sha256": "frozen-synthetic-torch"},
        )
        self.query_patch.start()
        self.torch_patch.start()
        self.addCleanup(self.query_patch.stop)
        self.addCleanup(self.torch_patch.stop)

    def test_plan_is_read_only_and_reports_exact_ownership_sets(self) -> None:
        before = self.fixture.snapshot_old()
        prepared = replace.prepare_replacement(self.fixture.request)
        self.assertEqual(prepared.plan_document["mode"], "plan")
        self.assertFalse(prepared.plan_document["mutation"])
        self.assertFalse(self.fixture.backup.exists())
        self.assertFalse(self.fixture.final_evidence.exists())
        self.assertEqual(self.fixture.snapshot_old(), before)
        old = prepared.plan_document["old_inventory"]
        self.assertIn("editable-pth", {role for item in old["files"] for role in item["roles"]})
        self.assertEqual(old["native_files_count"], 1)
        self.assertEqual(
            old["native_files"][0]["role"], "external-native-observation-only"
        )
        self.assertIn(
            "lib/python3.14/site-packages/triton/__init__.py",
            prepared.new_paths,
        )
        external = old["external_source_identity"]
        self.assertEqual(
            external["git_origin"], "https://example.invalid/triton.git"
        )
        self.assertRegex(external["git_head"], r"^[0-9a-f]{40}$")
        self.assertRegex(external["git_tree"], r"^[0-9a-f]{40}$")
        self.assertFalse(external["dirty"])
        self.assertGreater(external["package_tree"]["files"], 0)

    def test_cli_plan_holds_shared_lock_without_publishing_or_backup(self) -> None:
        lock_path = self.fixture.workspace / "runs" / replace.ENVIRONMENT_LOCK_NAME
        lock_path.touch(mode=0o600)
        before = self.fixture.snapshot_old()
        stdout = io.StringIO()
        with (
            mock.patch.object(replace, "_request_from_args", return_value=self.fixture.request),
            mock.patch.object(sys, "stdout", stdout),
        ):
            result = replace.main(
                ["--plan", "--backup-root", str(self.fixture.backup)]
            )
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["mode"], "plan")
        self.assertEqual(self.fixture.snapshot_old(), before)
        self.assertFalse(self.fixture.backup.exists())
        self.assertFalse(self.fixture.final_evidence.exists())

    def test_evidence_sha_mismatch_fails_before_any_write(self) -> None:
        request = replace.ReplacementRequest(
            **{
                field: getattr(self.fixture.request, field)
                for field in self.fixture.request.__dataclass_fields__
                if field != "expected_gpu_smoke_evidence_sha256"
            },
            expected_gpu_smoke_evidence_sha256="0" * 64,
        )
        with self.assertRaisesRegex(replace.ReplacementError, "GPU-smoke evidence differs"):
            replace.prepare_replacement(request)
        self.assertFalse(self.fixture.backup.exists())
        self.assertFalse(self.fixture.final_evidence.exists())

    def test_unowned_new_destination_is_rejected(self) -> None:
        package = self.fixture.site / "triton"
        package.mkdir()
        (package / "unowned.py").write_text("bad = True\n")
        with self.assertRaisesRegex(replace.ReplacementError, "unowned file"):
            replace.prepare_replacement(self.fixture.request)
        self.assertFalse(self.fixture.backup.exists())

    def test_old_record_escape_is_rejected(self) -> None:
        record = self.fixture.dist / "RECORD"
        with record.open("a") as sink:
            sink.write("../../../../../../outside,,\n")
        with self.assertRaisesRegex(replace.ReplacementError, "escaped environment prefix"):
            replace.prepare_replacement(self.fixture.request)
        self.assertFalse(self.fixture.backup.exists())

    def _run_apply(self, post):
        def post_adapter(prepared, installation, *, limits):
            return post(prepared, installation, limits)

        with (
            mock.patch.object(
                replace,
                "run_normal_runtime_audit",
                return_value=self.fixture.old_runtime(),
            ),
            mock.patch.object(replace, "_post_audit", side_effect=post_adapter),
        ):
            return replace.apply_replacement(self.fixture.request)

    def test_apply_uses_stdlib_installer_and_publishes_atomic_evidence(self) -> None:
        def post(prepared, installation, limits):
            verified = replace.verify_new_install(
                prepared, installation, limits=limits
            )
            return {
                "record_verification": verified,
                "torch_after": prepared.torch_before,
                "torch_tree_unchanged": True,
            }

        with mock.patch.object(
            probe,
            "install_audited_wheel",
            wraps=probe.install_audited_wheel,
        ) as installer:
            evidence, evidence_sha = self._run_apply(post)
        installer.assert_called_once()
        self.assertEqual(evidence["status"], "committed")
        self.assertEqual(evidence["installation"]["method"], replace.INSTALL_METHOD)
        self.assertEqual(evidence_sha, probe.sha256_file(self.fixture.final_evidence))
        self.assertEqual(stat.S_IMODE(self.fixture.final_evidence.stat().st_mode), 0o444)
        self.assertFalse(
            (self.fixture.site / "__editable__.triton-3.7.1+git5d6048aa.pth").exists()
        )
        self.assertFalse((self.fixture.site / f"{self.fixture.finder_module}.py").exists())
        self.assertEqual(
            (self.fixture.site / "triton/__init__.py").read_bytes(),
            self.fixture.wheel.members["triton/__init__.py"],
        )
        self.assertEqual(self.fixture.old_native.read_bytes(), self.fixture.old_native_bytes)
        manifest = json.loads((self.fixture.backup / "manifest.json").read_text())
        native = manifest["native_files"][0]
        self.assertEqual(
            (self.fixture.backup / native["blob"]).read_bytes(),
            self.fixture.old_native_bytes,
        )

    def test_post_audit_failure_restores_bytes_modes_and_symlink(self) -> None:
        before = self.fixture.snapshot_old()

        def fail_post(prepared, installation, limits):
            replace.verify_new_install(prepared, installation, limits=limits)
            raise replace.ReplacementError("injected post-audit failure")

        with self.assertRaisesRegex(
            replace.ReplacementError, "old environment restored"
        ) as raised:
            self._run_apply(fail_post)
        self.assertTrue(raised.exception.rollback["verified"])
        self.assertEqual(self.fixture.snapshot_old(), before)
        self.assertEqual(self.fixture.old_native.read_bytes(), self.fixture.old_native_bytes)
        self.assertTrue(self.fixture.old_link.is_symlink())
        self.assertEqual(os.readlink(self.fixture.old_link), "licenses/LICENSE")
        finder = self.fixture.site / f"{self.fixture.finder_module}.py"
        self.assertEqual(stat.S_IMODE(finder.stat().st_mode), 0o600)
        self.assertFalse((self.fixture.site / "triton").exists())
        failure = json.loads(self.fixture.final_evidence.read_text())
        self.assertEqual(failure["status"], "rolled-back")
        self.assertTrue(failure["rollback"]["verified"])
        self.assertEqual(stat.S_IMODE(self.fixture.final_evidence.stat().st_mode), 0o444)

    def test_partial_install_failure_restores_without_deleting_unowned_paths(self) -> None:
        before = self.fixture.snapshot_old()
        unowned = self.fixture.site / "unrelated-package.py"
        unowned.write_bytes(b"keep me\n")

        def partial_install(_wheel, scheme, _anchor, *, limits):
            path = scheme.platlib / "triton/__init__.py"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"partial\n")
            raise probe.ProbeError("synthetic installer failure")

        with (
            mock.patch.object(
                replace,
                "run_normal_runtime_audit",
                return_value=self.fixture.old_runtime(),
            ),
            mock.patch.object(
                probe, "install_audited_wheel", side_effect=partial_install
            ),
            self.assertRaisesRegex(replace.ReplacementError, "old environment restored"),
        ):
            replace.apply_replacement(self.fixture.request)
        self.assertEqual(self.fixture.snapshot_old(), before | {
            unowned.relative_to(self.fixture.prefix).as_posix(): (
                "regular", 0o644, b"keep me\n"
            )
        })
        self.assertEqual(unowned.read_bytes(), b"keep me\n")

    def test_atomic_evidence_never_replaces_existing_path(self) -> None:
        path = self.fixture.workspace / "evidence/atomic.json"
        first = {"status": "first"}
        digest_value = replace.publish_canonical_json_no_replace(path, first)
        self.assertEqual(digest_value, probe.sha256_file(path))
        with self.assertRaisesRegex(replace.ReplacementError, "already exists"):
            replace.publish_canonical_json_no_replace(path, {"status": "second"})
        self.assertEqual(json.loads(path.read_text()), first)

    def test_source_contract_has_no_package_manager_network_or_broad_delete(self) -> None:
        source = (ROOT / "tools/replace_triton_environment.py").read_text()
        self.assertNotIn("pip install", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("urllib.request", source)
        self.assertNotIn("shutil.rmtree", source)
        self.assertIn("probe.install_audited_wheel", source)
        self.assertIn("--plan", source)
        self.assertIn("--apply", source)

    def test_final_smoke_must_be_finalizer_bound(self) -> None:
        document = dict(self.fixture.smoke_document)
        document.pop("exclusive_run")
        path = self.fixture.workspace / "evidence/not-finalized.json"
        path.write_text(probe.canonical_json(document))
        request = dataclass_replace(
            self.fixture.request,
            gpu_smoke_evidence=path,
            expected_gpu_smoke_evidence_sha256=probe.sha256_file(path),
        )
        with self.assertRaisesRegex(replace.ReplacementError, "not finalizer-bound"):
            replace.prepare_replacement(request)

    def test_compiled_cache_drift_is_rejected(self) -> None:
        cache_artifact = self.fixture.workspace / "builds/reference-cache/vector-add.cubin"
        cache_artifact.write_bytes(b"tampered cubin\x00")
        with self.assertRaisesRegex(replace.ReplacementError, "cache artifact bytes changed"):
            replace.prepare_replacement(self.fixture.request)

    def test_finalizer_run_context_join_is_revalidated(self) -> None:
        provisional = json.loads(self.fixture.provisional_smoke.read_text())
        provisional["run_context"]["pgid"] = 9999
        self.fixture.provisional_smoke.write_text(probe.canonical_json(provisional))
        final = dict(provisional)
        final["acceptance"] = "accepted"
        final["exclusive_run"] = self.fixture.smoke_document["exclusive_run"]
        final["provisional_evidence"] = self.fixture.identity(
            self.fixture.provisional_smoke
        )
        path = self.fixture.workspace / "evidence/bad-run-context-final.json"
        path.write_text(probe.canonical_json(final))
        request = dataclass_replace(
            self.fixture.request,
            gpu_smoke_evidence=path,
            expected_gpu_smoke_evidence_sha256=probe.sha256_file(path),
        )
        with self.assertRaisesRegex(replace.ReplacementError, "run_context"):
            replace.prepare_replacement(request)

    def test_versions_lock_interpreter_sha_is_mandatory(self) -> None:
        (self.fixture.workspace / "VERSIONS.lock").write_text(
            "triton.producer.python_sha256=" + "0" * 64 + "\n"
        )
        with self.assertRaisesRegex(replace.ReplacementError, "base-Python SHA256"):
            replace.prepare_replacement(self.fixture.request)

    def test_external_source_package_drift_is_rejected_before_mutation(self) -> None:
        prepared = replace.prepare_replacement(self.fixture.request)
        (self.fixture.package / "__init__.py").write_text("# drift\n")
        with self.assertRaisesRegex(replace.ReplacementError, "source/package tree drifted"):
            replace.verify_old_inventory(prepared)
        self.assertFalse(self.fixture.backup.exists())

    def test_query_uses_workspace_temporary_files_and_exact_probe_python(self) -> None:
        seen: list[pathlib.Path] = []

        def fake_run(_argv, *, environment, limits, description):
            temp_root = pathlib.Path(tempfile.gettempdir()).resolve()
            seen.append(temp_root)
            self.assertIn(self.fixture.workspace / "runs", temp_root.parents)
            self.assertEqual(pathlib.Path(environment["TMPDIR"]), temp_root)
            with tempfile.TemporaryFile() as handle:
                target = os.readlink(f"/proc/self/fd/{handle.fileno()}")
                self.assertIn(str(temp_root), target)
            return {
                "executable": str(self.fixture.prefix / "bin/python"),
                "paths": {
                    "data": str(self.fixture.scheme.data),
                    "headers": str(self.fixture.scheme.headers),
                    "platlib": str(self.fixture.scheme.platlib),
                    "purelib": str(self.fixture.scheme.purelib),
                    "scripts": str(self.fixture.scheme.scripts),
                },
                "prefix": str(self.fixture.prefix),
                "version": [3, 14, 6],
            }

        with mock.patch.object(probe, "run_json_command", side_effect=fake_run):
            python, scheme = REAL_QUERY_TARGET_SCHEME(
                self.fixture.prefix,
                json.loads(self.fixture.environment_lock.read_text()),
                workspace=self.fixture.workspace,
                base_python_identity=self.fixture.probe_document["inputs"][
                    "base_python"
                ],
                limits=replace.ReplacementLimits(),
            )
        self.assertEqual(python, self.fixture.prefix / "bin/python")
        self.assertEqual(scheme.platlib, self.fixture.site)
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].exists())

    def test_workspace_lock_rejects_competing_apply(self) -> None:
        with replace.workspace_transaction_lock(self.fixture.workspace):
            with self.assertRaisesRegex(replace.ReplacementError, "holds the lock"):
                replace.apply_replacement(self.fixture.request)
        self.assertFalse(self.fixture.backup.exists())

    def test_inherited_exclusive_environment_lock_is_verified_without_relock(self) -> None:
        lock_path = self.fixture.workspace / "runs" / replace.ENVIRONMENT_LOCK_NAME
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            identity = os.fstat(descriptor)
            start_ticks = replace._proc_start_ticks(
                pathlib.Path("/proc/self/stat").read_text()
            )
            environment = {
                replace.INHERITED_LOCK_ENVIRONMENT["fd"]: str(descriptor),
                replace.INHERITED_LOCK_ENVIRONMENT["mode"]: "exclusive",
                replace.INHERITED_LOCK_ENVIRONMENT["path"]: str(lock_path),
                replace.INHERITED_LOCK_ENVIRONMENT["dev"]: str(identity.st_dev),
                replace.INHERITED_LOCK_ENVIRONMENT["ino"]: str(identity.st_ino),
                replace.INHERITED_LOCK_ENVIRONMENT["controller_pid"]: str(os.getpid()),
                replace.INHERITED_LOCK_ENVIRONMENT["controller_start_ticks"]: str(
                    start_ticks
                ),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(replace.os, "getppid", return_value=os.getpid()),
            ):
                with replace.workspace_transaction_lock(self.fixture.workspace) as value:
                    self.assertTrue(value["inherited"])
                    self.assertEqual(value["fd"], descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_inherited_shared_or_partial_lock_marker_is_rejected(self) -> None:
        lock_path = self.fixture.workspace / "runs" / replace.ENVIRONMENT_LOCK_NAME
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            identity = os.fstat(descriptor)
            start_ticks = replace._proc_start_ticks(
                pathlib.Path("/proc/self/stat").read_text()
            )
            environment = {
                replace.INHERITED_LOCK_ENVIRONMENT["fd"]: str(descriptor),
                replace.INHERITED_LOCK_ENVIRONMENT["mode"]: "exclusive",
                replace.INHERITED_LOCK_ENVIRONMENT["path"]: str(lock_path),
                replace.INHERITED_LOCK_ENVIRONMENT["dev"]: str(identity.st_dev),
                replace.INHERITED_LOCK_ENVIRONMENT["ino"]: str(identity.st_ino),
                replace.INHERITED_LOCK_ENVIRONMENT["controller_pid"]: str(os.getpid()),
                replace.INHERITED_LOCK_ENVIRONMENT["controller_start_ticks"]: str(
                    start_ticks
                ),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(replace.os, "getppid", return_value=os.getpid()),
                self.assertRaisesRegex(
                    replace.ReplacementError,
                    "identity mismatch|not held exclusively",
                ),
            ):
                with replace.workspace_transaction_lock(self.fixture.workspace):
                    pass
            environment[replace.INHERITED_LOCK_ENVIRONMENT["mode"]] = "shared"
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(replace.os, "getppid", return_value=os.getpid()),
            ):
                with replace.workspace_transaction_lock(
                    self.fixture.workspace,
                    required_mode="shared",
                ) as value:
                    self.assertEqual(value["mode"], "shared")
            partial = {replace.INHERITED_LOCK_ENVIRONMENT["fd"]: str(descriptor)}
            with (
                mock.patch.dict(os.environ, partial, clear=True),
                self.assertRaisesRegex(replace.ReplacementError, "marker set is incomplete"),
            ):
                with replace.workspace_transaction_lock(self.fixture.workspace):
                    pass
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_real_parent_controller_fd_handoff_is_accepted(self) -> None:
        lock_path = self.fixture.workspace / "runs" / replace.ENVIRONMENT_LOCK_NAME
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        read_fd, write_fd = os.pipe()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            identity = os.fstat(descriptor)
            controller_ticks = replace._proc_start_ticks(
                pathlib.Path("/proc/self/stat").read_text()
            )
            child = os.fork()
            if child == 0:
                os.close(read_fd)
                environment = {
                    replace.INHERITED_LOCK_ENVIRONMENT["fd"]: str(descriptor),
                    replace.INHERITED_LOCK_ENVIRONMENT["mode"]: "exclusive",
                    replace.INHERITED_LOCK_ENVIRONMENT["path"]: str(lock_path),
                    replace.INHERITED_LOCK_ENVIRONMENT["dev"]: str(identity.st_dev),
                    replace.INHERITED_LOCK_ENVIRONMENT["ino"]: str(identity.st_ino),
                    replace.INHERITED_LOCK_ENVIRONMENT["controller_pid"]: str(
                        os.getppid()
                    ),
                    replace.INHERITED_LOCK_ENVIRONMENT[
                        "controller_start_ticks"
                    ]: str(controller_ticks),
                }
                try:
                    with mock.patch.dict(os.environ, environment, clear=True):
                        with replace.workspace_transaction_lock(
                            self.fixture.workspace
                        ) as value:
                            os.write(write_fd, b"ok" if value["inherited"] else b"bad")
                    os._exit(0)
                except BaseException as error:
                    os.write(write_fd, f"error:{error}".encode())
                    os._exit(1)
            os.close(write_fd)
            message = os.read(read_fd, 4096)
            _, status = os.waitpid(child, 0)
            self.assertEqual(status, 0, message.decode(errors="replace"))
            self.assertEqual(message, b"ok")
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass
            try:
                os.close(write_fd)
            except OSError:
                pass
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_exec_pass_fd_controller_handoff_is_accepted(self) -> None:
        lock_path = self.fixture.workspace / "runs" / replace.ENVIRONMENT_LOCK_NAME
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            identity = os.fstat(descriptor)
            controller_ticks = replace._proc_start_ticks(
                pathlib.Path("/proc/self/stat").read_text()
            )
            environment = {
                "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                replace.INHERITED_LOCK_ENVIRONMENT["fd"]: str(descriptor),
                replace.INHERITED_LOCK_ENVIRONMENT["mode"]: "exclusive",
                replace.INHERITED_LOCK_ENVIRONMENT["path"]: str(lock_path),
                replace.INHERITED_LOCK_ENVIRONMENT["dev"]: str(identity.st_dev),
                replace.INHERITED_LOCK_ENVIRONMENT["ino"]: str(identity.st_ino),
                replace.INHERITED_LOCK_ENVIRONMENT["controller_pid"]: str(os.getpid()),
                replace.INHERITED_LOCK_ENVIRONMENT["controller_start_ticks"]: str(
                    controller_ticks
                ),
            }
            program = (
                "import importlib.util,json,pathlib,sys;"
                "p=pathlib.Path(sys.argv[1]);"
                "s=importlib.util.spec_from_file_location('exec_replace',p);"
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;"
                "s.loader.exec_module(m);"
                "w=pathlib.Path(sys.argv[2]);"
                "c=m.workspace_transaction_lock(w);v=c.__enter__();"
                "print(json.dumps(v,sort_keys=True));c.__exit__(None,None,None)"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    program,
                    str(ROOT / "tools/replace_triton_environment.py"),
                    str(self.fixture.workspace),
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                pass_fds=(descriptor,),
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["inherited"])
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_manual_exclusive_fallback_conflicts_with_consumer_shared_lock(self) -> None:
        lock_path = self.fixture.workspace / "runs" / replace.ENVIRONMENT_LOCK_NAME
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(replace.ReplacementError, "holds the lock"),
            ):
                with replace.workspace_transaction_lock(self.fixture.workspace):
                    pass
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_proc_scan_detects_foreign_uid_prefix_and_native_use(self) -> None:
        proc_root = self.fixture.root / "synthetic-proc"
        process = proc_root / "4242"
        process.mkdir(parents=True)
        (process / "stat").write_text("4242 (foreign) S 1 9999 0 0 0\n")
        (process / "status").write_text("Name:\tforeign\nUid:\t12345\t12345\n")
        (process / "exe").symlink_to(self.fixture.prefix / "bin/python")
        (process / "cwd").symlink_to(self.fixture.root)
        (process / "maps").write_text(
            "1000-2000 r-xp 00000000 00:00 0 " + str(self.fixture.old_native) + "\n"
        )
        (process / "environ").write_bytes(
            b"PATH=" + os.fsencode(self.fixture.prefix / "bin") + b"\x00"
        )
        blockers = replace.scan_environment_users(
            self.fixture.prefix,
            [
                replace.NativeEntry(
                    path=str(self.fixture.old_native),
                    mode=0o755,
                    size=self.fixture.old_native.stat().st_size,
                    sha256=probe.sha256_file(self.fixture.old_native),
                )
            ],
            proc_root=proc_root,
            allowed_pgid=os.getpgrp(),
        )
        self.assertEqual(blockers[0]["uid"], 12345)
        self.assertIn("prefix-executable", blockers[0]["reasons"])
        self.assertIn("external-libtriton-mapping", blockers[0]["reasons"])

    def test_proc_scan_exempts_only_exact_controller_pid_start_ticks(self) -> None:
        proc_root = self.fixture.root / "controller-proc"
        process = proc_root / "5000"
        process.mkdir(parents=True)
        fields = ["S", "1", "9999"] + ["0"] * 16 + ["424242"]
        (process / "stat").write_text("5000 (controller) " + " ".join(fields) + "\n")
        (process / "status").write_text("Uid:\t1000\t1000\n")
        (process / "exe").symlink_to(self.fixture.prefix / "bin/python")
        (process / "cwd").symlink_to(self.fixture.root)
        (process / "maps").write_text("")
        (process / "environ").write_bytes(b"")
        without = replace.scan_environment_users(
            self.fixture.prefix,
            [],
            proc_root=proc_root,
            allowed_pgid=os.getpgrp(),
        )
        exact = replace.scan_environment_users(
            self.fixture.prefix,
            [],
            proc_root=proc_root,
            allowed_pgid=os.getpgrp(),
            exempt_identities={5000: 424242},
        )
        stale = replace.scan_environment_users(
            self.fixture.prefix,
            [],
            proc_root=proc_root,
            allowed_pgid=os.getpgrp(),
            exempt_identities={5000: 424241},
        )
        self.assertEqual(without[0]["pid"], 5000)
        self.assertEqual(exact, [])
        self.assertEqual(stale[0]["pid"], 5000)

    def test_sigterm_during_post_audit_rolls_back_and_journals(self) -> None:
        before = self.fixture.snapshot_old()

        def interrupted(_prepared, _installation, _limits):
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("signal handler did not interrupt")

        import signal

        with self.assertRaisesRegex(replace.ReplacementError, "old environment restored"):
            self._run_apply(interrupted)
        self.assertEqual(self.fixture.snapshot_old(), before)
        journal = replace.load_journal(self.fixture.backup)
        self.assertEqual(journal["phase"], "rolled-back")
        self.assertTrue(journal["rollback"]["verified"])

    def test_recover_is_idempotent_after_simulated_sigkill_mid_install(self) -> None:
        prepared = replace.prepare_replacement(self.fixture.request)
        self.fixture.backup.mkdir()
        (self.fixture.backup / "scratch").mkdir()
        old_runtime = self.fixture.old_runtime()
        _, manifest_sha = replace.create_backup(prepared, old_runtime=old_runtime)
        replace.update_journal(
            self.fixture.backup,
            phase="removing-old",
            manifest_sha256=manifest_sha,
            mutation_started=True,
        )
        replace._remove_old_inventory(prepared)
        replace.update_journal(
            self.fixture.backup,
            phase="installing-new",
            manifest_sha256=manifest_sha,
            mutation_started=True,
        )
        partial = self.fixture.site / "triton/__init__.py"
        partial.parent.mkdir()
        partial.write_bytes(b"partial before SIGKILL\n")
        old_only = self.fixture.site / "__editable__.triton-3.7.1+git5d6048aa.pth"
        rollback_temporary = old_only.with_name(
            f".{old_only.name}.pypto-triton-rollback.partial"
        )
        rollback_temporary.write_bytes(b"partial rollback before second SIGKILL\n")
        with mock.patch.object(
            replace,
            "run_normal_runtime_audit",
            return_value=old_runtime,
        ):
            first, first_sha = replace.recover_replacement(
                self.fixture.workspace,
                self.fixture.backup,
                force_rollback=False,
            )
            second, second_sha = replace.recover_replacement(
                self.fixture.workspace,
                self.fixture.backup,
                force_rollback=False,
            )
        self.assertEqual(first, second)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first["status"], "rolled-back")
        self.assertEqual(self.fixture.snapshot_old(), self.fixture.old_snapshot)
        self.assertEqual(replace.load_journal(self.fixture.backup)["phase"], "rolled-back")

    def test_prejournal_sigkill_staging_does_not_block_apply_retry(self) -> None:
        # Simulate SIGKILL immediately after sibling staging mkdir, before the
        # initializing journal write. The intended backup root was never made
        # visible, so a fresh apply must be able to proceed without guessing
        # whether the environment was mutated.
        orphan = replace.create_initialization_staging(self.fixture.backup)
        self.assertEqual(list(orphan.iterdir()), [])
        self.assertFalse(self.fixture.backup.exists())

        def post(prepared, installation, limits):
            return {
                "record_verification": replace.verify_new_install(
                    prepared, installation, limits=limits
                ),
                "torch_after": prepared.torch_before,
                "torch_tree_unchanged": True,
            }

        evidence, _ = self._run_apply(post)
        self.assertEqual(evidence["status"], "committed")
        self.assertTrue(self.fixture.backup.is_dir())
        self.assertTrue((self.fixture.backup / "journal.json").is_file())
        self.assertEqual(replace.load_journal(self.fixture.backup)["phase"], "committed")
        self.assertTrue(orphan.is_dir())
        self.assertEqual(list(orphan.iterdir()), [])

    def test_recover_never_trusts_committed_json_over_tampered_prefix(self) -> None:
        def post(prepared, installation, limits):
            verified = replace.verify_new_install(
                prepared, installation, limits=limits
            )
            return {
                "record_verification": verified,
                "torch_after": prepared.torch_before,
                "torch_tree_unchanged": True,
            }

        self._run_apply(post)
        installed = self.fixture.site / "triton/__init__.py"
        installed.write_bytes(b"tampered after commit\n")
        with self.assertRaisesRegex(
            replace.ReplacementError, "installed RECORD bytes differ"
        ):
            replace.recover_replacement(
                self.fixture.workspace,
                self.fixture.backup,
                force_rollback=False,
            )
        self.assertFalse(
            (self.fixture.site / "__editable__.triton-3.7.1+git5d6048aa.pth").exists()
        )

    def test_committed_recover_publishes_separate_idempotent_verification(self) -> None:
        def post(prepared, installation, limits):
            verified = replace.verify_new_install(
                prepared, installation, limits=limits
            )
            return {
                "record_verification": verified,
                "torch_after": prepared.torch_before,
                "torch_tree_unchanged": True,
            }

        terminal, _ = self._run_apply(post)
        recovery_path = self.fixture.workspace / "evidence/commit-recovery.json"
        with mock.patch.object(
            replace,
            "_post_audit",
            return_value=terminal["post_audit"],
        ):
            first, first_sha = replace.recover_replacement(
                self.fixture.workspace,
                self.fixture.backup,
                force_rollback=False,
                evidence=recovery_path,
            )
            second, second_sha = replace.recover_replacement(
                self.fixture.workspace,
                self.fixture.backup,
                force_rollback=False,
                evidence=recovery_path,
            )
        self.assertEqual(first, second)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first["status"], "committed-verified")
        self.assertEqual(json.loads(recovery_path.read_text()), first)
        self.assertNotEqual(recovery_path, self.fixture.final_evidence)

    def test_committed_recover_cli_reports_actual_new_evidence_path_and_sha(self) -> None:
        def post(prepared, installation, limits):
            return {
                "record_verification": replace.verify_new_install(
                    prepared, installation, limits=limits
                ),
                "torch_after": prepared.torch_before,
                "torch_tree_unchanged": True,
            }

        terminal, _ = self._run_apply(post)
        recovery_path = self.fixture.workspace / "evidence/cli-commit-recovery.json"
        stdout = io.StringIO()
        with (
            mock.patch.object(
                replace,
                "_post_audit",
                return_value=terminal["post_audit"],
            ),
            mock.patch.object(sys, "stdout", stdout),
        ):
            result = replace.main(
                [
                    "--workspace",
                    str(self.fixture.workspace),
                    "--backup-root",
                    str(self.fixture.backup),
                    "--evidence",
                    str(recovery_path),
                    "--recover",
                ]
            )
        report = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "committed-verified")
        self.assertEqual(report["evidence"], str(recovery_path))
        self.assertEqual(report["evidence_sha256"], probe.sha256_file(recovery_path))

    def test_no_mutation_recover_cli_publishes_requested_evidence(self) -> None:
        backup = self.fixture.workspace / "builds/no-mutation-backup"
        backup.mkdir()
        replace.update_journal(
            backup,
            phase="initializing",
            manifest_sha256=None,
            mutation_started=False,
        )
        recovery_path = self.fixture.workspace / "evidence/no-mutation-recovery.json"
        stdout = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout):
            result = replace.main(
                [
                    "--workspace",
                    str(self.fixture.workspace),
                    "--backup-root",
                    str(backup),
                    "--evidence",
                    str(recovery_path),
                    "--recover",
                ]
            )
        report = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "no-mutation-before-backup")
        self.assertEqual(report["evidence"], str(recovery_path))
        self.assertEqual(report["evidence_sha256"], probe.sha256_file(recovery_path))

    def test_main_reports_rollback_failed_not_rolled_back(self) -> None:
        fake = replace.ReplacementError(
            "high-risk rollback failure",
            rollback={"verified": False},
            evidence_sha256="a" * 64,
            evidence_status="rollback-failed",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(replace, "_request_from_args", return_value=self.fixture.request),
            mock.patch.object(replace, "apply_replacement", side_effect=fake),
            mock.patch.object(sys, "stdout", stdout),
            mock.patch.object(sys, "stderr", stderr),
        ):
            result = replace.main(
                ["--apply", "--backup-root", str(self.fixture.backup)]
            )
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "rollback-failed")

    def test_real_rollback_failure_publishes_high_risk_terminal_status(self) -> None:
        def fail_post(_prepared, _installation, _limits):
            raise replace.ReplacementError("injected post failure")

        with (
            mock.patch.object(
                replace,
                "restore_old_inventory",
                side_effect=replace.ReplacementError("injected rollback failure"),
            ),
            self.assertRaisesRegex(replace.ReplacementError, "could not be proven") as raised,
        ):
            self._run_apply(fail_post)
        self.assertEqual(raised.exception.evidence_status, "rollback-failed")
        document = json.loads(self.fixture.final_evidence.read_text())
        self.assertEqual(document["status"], "rollback-failed")
        self.assertFalse(document["rollback"]["verified"])


if __name__ == "__main__":
    unittest.main()
