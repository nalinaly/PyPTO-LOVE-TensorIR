from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import pathlib
import stat
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "benchmarks" / "operators" / "triton_reference_sm120.py"
    spec = importlib.util.spec_from_file_location(
        "test_triton_reference_sm120_tool", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = load_tool()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class SyntheticProbe:
    def __init__(self, workspace: pathlib.Path) -> None:
        self.workspace = workspace
        self.prefix = workspace / "runs" / "probe-prefix"
        self.site = self.prefix / "lib" / "python3.14" / "site-packages"
        self.view = self.prefix / ".torch-runtime-view"
        self.python = self.prefix / "bin" / "python"
        self.site.mkdir(parents=True)
        self.view.mkdir()
        self.python.parent.mkdir()
        self.base_python = workspace / "envs" / "base-python"
        self.base_python.parent.mkdir()
        self.base_python.write_bytes(b"synthetic-cpython-3.14.6\x00")
        self.base_python.chmod(0o755)
        self.python.symlink_to(self.base_python)

        triton = self.site / "triton"
        (triton / "_C").mkdir(parents=True)
        (triton / "__init__.py").write_text('__version__ = "3.7.1"\n')
        libtriton = triton / "_C" / "libtriton.so"
        libtriton.write_bytes(b"\x7fELFsynthetic-libtriton\x00")
        ptxas = self.site / smoke.PTXAS_BLACKWELL_RELATIVE
        ptxas.parent.mkdir(parents=True)
        ptxas.write_bytes(b"\x7fELFsynthetic-ptxas-blackwell\x00")
        ptxas.chmod(0o755)
        dist_info = self.site / "triton-3.7.1+git5d6048aa.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: triton\nVersion: 3.7.1+git5d6048aa\n\n"
        )
        (dist_info / "RECORD").write_text("synthetic RECORD\n")

        self.torch_source_site = workspace / "envs" / "torch-site"
        torch_package = self.torch_source_site / "torch"
        torch_dist_info = self.torch_source_site / "torch-2.13.0+cu130.dist-info"
        torch_package.mkdir(parents=True)
        torch_dist_info.mkdir()
        (torch_package / "__init__.py").write_text("# synthetic Torch\n")
        (torch_dist_info / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: torch\nVersion: 2.13.0+cu130\n\n"
        )
        for source in (torch_package, torch_dist_info):
            (self.view / source.name).symlink_to(source, target_is_directory=True)
        included = sorted(path.name for path in self.view.iterdir())

        evidence_dir = workspace / "reports" / "data"
        evidence_dir.mkdir(parents=True)
        artifact_dir = workspace / "builds" / "triton"
        artifact_dir.mkdir(parents=True)
        self.environment_lock = workspace / "ENVIRONMENT.lock"
        torch_tree = smoke._torch_tree_identity(
            self.torch_source_site, torch_package, torch_dist_info
        )
        torch_metadata = (torch_dist_info / "METADATA").read_bytes()
        torch_identity = {
            **torch_tree,
            "dist_info_name": torch_dist_info.name,
            "metadata_sha256": digest(torch_metadata),
            "module_sha256": digest((torch_package / "__init__.py").read_bytes()),
            "version": "2.13.0+cu130",
        }
        self.environment_lock.write_text(
            smoke.canonical_json(
                {
                    "cuda": "13.0",
                    "destination_prefix": "envs",
                    "hip": None,
                    "python": "3.14.6 | synthetic",
                    "python_abi": "cp314",
                    "python_executable": str(self.base_python),
                    "python_implementation": "CPython",
                    "torch": "2.13.0+cu130",
                    "torch_dist_info": str(torch_dist_info),
                    "torch_file": str(torch_package / "__init__.py"),
                    "torch_git": "cf30153c4c131c8164ee7798e5022d810682e2cb",
                    "torch_package_root": str(torch_package),
                    **torch_tree,
                }
            )
        )
        self.wheel = artifact_dir / "triton-synthetic.whl"
        self.wheel.write_bytes(b"synthetic-wheel\x00")
        self.audit = evidence_dir / "triton-wheel-audit.json"
        self.audit.write_text(
            smoke.canonical_json(
                {
                    "acceptance": "accepted",
                    "audit": smoke.WHEEL_AUDIT_NAME,
                    "schema_version": 1,
                    "wheel": {
                        "elf_paths": [
                            "triton/_C/libtriton.so",
                            smoke.PTXAS_BLACKWELL_RELATIVE.as_posix(),
                        ]
                    },
                }
            )
        )

        def identity(path: pathlib.Path) -> dict[str, object]:
            raw = path.read_bytes()
            return {
                "path": path.relative_to(workspace).as_posix(),
                "sha256": digest(raw),
                "size": len(raw),
            }

        prefix_token = "$PROBE_PREFIX/"
        libtriton_token = prefix_token + libtriton.relative_to(self.prefix).as_posix()
        ptxas_token = prefix_token + ptxas.relative_to(self.prefix).as_posix()
        process = {
            "backend": {
                "arch": "sm120",
                "class": "triton.backends.nvidia.compiler.CUDABackend",
                "target": ["cuda", 120, 32],
            },
            "editable": {
                "carriers": {
                    "meta_path": [],
                    "path_hooks": [],
                    "path_importer_cache": [],
                },
                "loaded_modules": [],
            },
            "distribution": {
                "dist_info": prefix_token
                + dist_info.relative_to(self.prefix).as_posix(),
                "name": "triton",
                "version": "3.7.1+git5d6048aa",
            },
            "libtriton_maps": [libtriton_token],
            "module_paths": {
                "triton": [
                    prefix_token
                    + (triton / "__init__.py").relative_to(self.prefix).as_posix()
                ],
                "triton._C.libtriton": [libtriton_token],
            },
            "ptxas_blackwell": {
                "audited_full_version": smoke.EXPECTED_PTXAS_FULL_VERSION,
                "path": ptxas_token,
                "reported_release": smoke.EXPECTED_PTXAS_RELEASE,
                "sha256": digest(ptxas.read_bytes()),
            },
            "torch": {
                "cuda": "13.0",
                "git_version": "cf30153c4c131c8164ee7798e5022d810682e2cb",
                "hip": None,
                "version": "2.13.0+cu130",
            },
        }
        wheel_identity = identity(self.wheel)
        wheel_identity.update(
            {
                "audit_evidence": identity(self.audit),
                "filename": self.wheel.name,
            }
        )
        installed_files = sorted(
            path
            for root in (triton, dist_info)
            for path in root.rglob("*")
            if path.is_file()
        )
        record_entries = [
            {
                "path": path.relative_to(self.site).as_posix(),
                "sha256": digest(path.read_bytes()),
                "size": path.stat().st_size,
            }
            for path in installed_files
        ]
        record_entries.sort(key=lambda record: str(record["path"]))
        self.document = {
            "acceptance": "accepted",
            "inputs": {
                "base_python": identity(self.base_python),
                "environment_lock": identity(self.environment_lock),
                "torch_site_packages": {
                    "path": self.torch_source_site.relative_to(workspace).as_posix(),
                    **torch_identity,
                },
                "wheel": wheel_identity,
            },
            "installation": {
                "archive_members": [
                    {
                        "archive_path": "triton/_C/libtriton.so",
                        "installed_path": "triton/_C/libtriton.so",
                        "sha256": digest(libtriton.read_bytes()),
                        "size": libtriton.stat().st_size,
                    },
                    {
                        "archive_path": smoke.PTXAS_BLACKWELL_RELATIVE.as_posix(),
                        "installed_path": smoke.PTXAS_BLACKWELL_RELATIVE.as_posix(),
                        "sha256": digest(ptxas.read_bytes()),
                        "size": ptxas.stat().st_size,
                    },
                ],
                "fresh_prefix": True,
                "prefix": self.prefix.relative_to(workspace).as_posix(),
                "python": self.python.relative_to(workspace).as_posix(),
                "scheme": {
                    "platlib": self.site.relative_to(workspace).as_posix(),
                },
                "record_verification": {
                    "editable_artifacts": [],
                    "entries": record_entries,
                    "entries_count": len(record_entries),
                },
                "torch_runtime_view": {
                    "included_entries": included,
                    "included_entries_count": len(included),
                    "included_entries_sha256": digest(
                        smoke.canonical_json(included).encode("ascii")
                    ),
                    "path": self.view.relative_to(workspace).as_posix(),
                },
            },
            "probe": smoke.PROBE_NAME,
            "runtime": {
                "gpu_execution": False,
                "processes": [process, process],
                "processes_count": 2,
            },
            "schema_version": 1,
        }
        self.evidence = evidence_dir / "triton-wheel-probe.json"
        self.evidence.write_text(smoke.canonical_json(self.document))
        self.evidence_sha256 = digest(self.evidence.read_bytes())


class TritonReferenceSm120Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = pathlib.Path(self.temporary.name).resolve()
        self.synthetic = SyntheticProbe(self.workspace)
        self.cache = self.workspace / "caches" / "reference-sm120"
        self.cache.parent.mkdir()
        self.output = self.workspace / "reports" / "data" / "smoke.json"
        self.request = smoke.SmokeRequest(
            workspace=self.workspace,
            probe_evidence=self.synthetic.evidence,
            expected_probe_evidence_sha256=self.synthetic.evidence_sha256,
            probe_prefix=self.synthetic.prefix,
            probe_site=self.synthetic.site,
            torch_runtime_view=self.synthetic.view,
            cache_dir=self.cache,
            evidence=self.output,
        )

    def runtime_context(self) -> smoke.RuntimeContext:
        return smoke.RuntimeContext(
            torch=ModuleType("synthetic_torch"),
            triton=ModuleType("synthetic_triton"),
            language=ModuleType("synthetic_triton_language"),
            ptxas_evidence={
                "audited_full_version": smoke.EXPECTED_PTXAS_FULL_VERSION,
                "path": "$PROBE_PREFIX/lib/python3.14/site-packages/"
                "triton/backends/nvidia/bin/ptxas-blackwell",
                "reported_release": smoke.EXPECTED_PTXAS_RELEASE,
                "sha256": digest(b"\x7fELFsynthetic-ptxas-blackwell\x00"),
                "wheel_owned": True,
            },
        )

    @staticmethod
    def execution() -> dict[str, object]:
        return {
            "correctness": {
                "block_size": smoke.BLOCK_SIZE,
                "comparison": "torch.equal",
                "dtype": "float32",
                "equal": True,
                "kernel": "masked-vector-add",
                "n_elements": smoke.VECTOR_ELEMENTS,
                "reference_provider": "torch",
            },
            "device": {
                "compute_capability": [12, 0],
                "index": 0,
                "name": smoke.EXPECTED_DEVICE_NAME,
            },
            "synchronization": {
                "after_comparison": True,
                "after_kernel": True,
                "before_launch": True,
                "error": None,
            },
            "target": {"arch": 120, "backend": "cuda", "warp_size": 32},
        }

    @staticmethod
    def provenance() -> dict[str, object]:
        return {
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
                    "$PROBE_PREFIX/lib/python3.14/site-packages/triton/__init__.py"
                ]
            },
            "python": {
                "executable": "$PROBE_PREFIX/bin/python",
                "resolved_executable": "envs/base-python",
                "resolved_sha256": digest(b"synthetic-cpython-3.14.6\x00"),
                "resolved_size": len(b"synthetic-cpython-3.14.6\x00"),
            },
            "sys_path": ["$PROBE_PREFIX/lib/python3.14/site-packages"],
            "torch_file": "$TORCH_SITE_PACKAGES/torch/__init__.py",
            "torch_runtime": {
                "cuda": "13.0",
                "file": "$TORCH_SITE_PACKAGES/torch/__init__.py",
                "git_version": "cf30153c4c131c8164ee7798e5022d810682e2cb",
                "hip": None,
                "version": "2.13.0+cu130",
            },
        }

    def run_context(self) -> dict[str, object]:
        return {
            "mode": "gpu-benchmark",
            "pgid": 123456,
            "pid": 123457,
            "preflight": {
                "path": "runs/synthetic/preflight.json",
                "sha256": "a" * 64,
                "size": 123,
            },
            "provisional_evidence_path": self.output.relative_to(
                self.workspace
            ).as_posix(),
            "run_id": "pypto-20990101T000000Z-123456-abcdef",
        }

    def test_mocked_success_publishes_bound_reference_only_evidence(self) -> None:
        context = self.runtime_context()

        def execute(_: smoke.RuntimeContext) -> dict[str, object]:
            artifact_dir = self.cache / "0123456789abcdef"
            artifact_dir.mkdir()
            (artifact_dir / "vector_add.cubin").write_bytes(b"synthetic-cubin\x00")
            (artifact_dir / "vector_add.json").write_bytes(b'{"name":"vector_add"}\n')
            return self.execution()

        with (
            mock.patch.dict(smoke.os.environ, {}, clear=True),
            mock.patch.object(smoke, "_validate_isolated_invocation") as invocation,
            mock.patch.object(
                smoke, "_capture_gpu_run_context", return_value=self.run_context()
            ),
            mock.patch.object(smoke, "_load_runtime", return_value=context) as loader,
            mock.patch.object(smoke, "_execute_vector_add", side_effect=execute) as run,
            mock.patch.object(
                smoke,
                "_collect_runtime_provenance",
                return_value=self.provenance(),
            ) as provenance,
        ):
            evidence, evidence_sha256 = smoke.run_smoke(self.request)

        invocation.assert_called_once()
        loader.assert_called_once()
        run.assert_called_once_with(context)
        provenance.assert_called_once()
        self.assertEqual(self.output.read_text(), smoke.canonical_json(evidence))
        self.assertEqual(evidence_sha256, digest(self.output.read_bytes()))
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o444)
        self.assertEqual(evidence["smoke"], smoke.SMOKE_NAME)
        self.assertEqual(
            evidence["scope"],
            {
                "coverage_result": False,
                "performance_result": False,
                "provider": "triton",
                "pypto_kernel": False,
                "reference_only": True,
            },
        )
        self.assertEqual(
            evidence["inputs"]["probe_evidence"]["sha256"],
            self.synthetic.evidence_sha256,
        )
        self.assertEqual(
            evidence["inputs"]["wheel"]["sha256"], digest(self.synthetic.wheel.read_bytes())
        )
        runtime = evidence["runtime"]
        self.assertTrue(runtime["gpu_execution"])
        self.assertTrue(runtime["integrity"]["stable"])
        self.assertEqual(
            runtime["integrity"]["before"], runtime["integrity"]["after"]
        )
        self.assertTrue(runtime["correctness"]["equal"])
        self.assertIsNone(runtime["synchronization"]["error"])
        self.assertTrue(runtime["ptxas_blackwell"]["wheel_owned"])
        self.assertEqual(
            runtime["provenance"]["python"]["resolved_executable"],
            "envs/base-python",
        )
        self.assertEqual(runtime["provenance"]["torch_runtime"]["cuda"], "13.0")
        cache = runtime["compiled_cache"]
        self.assertTrue(cache["fresh_before_run"])
        self.assertEqual(cache["artifacts_count"], 2)
        self.assertEqual(cache["cubin_count"], 1)
        self.assertEqual(
            {record["path"] for record in cache["artifacts"]},
            {
                "0123456789abcdef/vector_add.cubin",
                "0123456789abcdef/vector_add.json",
            },
        )

    def test_mocked_cuda_error_never_publishes_acceptance(self) -> None:
        context = self.runtime_context()
        with (
            mock.patch.dict(smoke.os.environ, {}, clear=True),
            mock.patch.object(smoke, "_validate_isolated_invocation"),
            mock.patch.object(
                smoke, "_capture_gpu_run_context", return_value=self.run_context()
            ),
            mock.patch.object(smoke, "_load_runtime", return_value=context),
            mock.patch.object(
                smoke,
                "_execute_vector_add",
                side_effect=smoke.SmokeError("synthetic asynchronous CUDA error"),
            ),
            mock.patch.object(smoke, "_collect_runtime_provenance") as provenance,
        ):
            with self.assertRaisesRegex(smoke.SmokeError, "asynchronous CUDA error"):
                smoke.run_smoke(self.request)
        provenance.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_evidence_is_no_replace_before_import_or_cache(self) -> None:
        sentinel = b"pre-existing evidence must survive byte-for-byte\n"
        self.output.write_bytes(sentinel)
        with (
            mock.patch.object(smoke, "_validate_isolated_invocation") as invocation,
            mock.patch.object(
                smoke,
                "_load_runtime",
                side_effect=AssertionError("no-replace attempted a runtime import"),
            ) as loader,
        ):
            with self.assertRaisesRegex(smoke.SmokeError, "evidence already exists"):
                smoke.run_smoke(self.request)
        invocation.assert_not_called()
        loader.assert_not_called()
        self.assertFalse(self.cache.exists())
        self.assertEqual(self.output.read_bytes(), sentinel)

    def test_probe_site_rejects_ambient_or_editable_carriers(self) -> None:
        (self.synthetic.site / "ambient.pth").write_text("/external/triton\n")
        with self.assertRaisesRegex(smoke.SmokeError, "ambient/editable carrier"):
            smoke.prepare_smoke(self.request)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.output.exists())

    def test_base_python_tamper_is_rejected_before_runtime(self) -> None:
        with self.synthetic.base_python.open("ab") as sink:
            sink.write(b"tamper")
        with mock.patch.object(
            smoke,
            "_load_runtime",
            side_effect=AssertionError("base-Python tamper attempted runtime import"),
        ) as loader:
            with self.assertRaisesRegex(smoke.SmokeError, "base Python differs"):
                smoke.run_smoke(self.request)
        loader.assert_not_called()
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.output.exists())

    def test_complete_torch_tree_tamper_is_rejected_before_runtime(self) -> None:
        torch_init = self.synthetic.torch_source_site / "torch" / "__init__.py"
        with torch_init.open("ab") as sink:
            sink.write(b"# tamper\n")
        with self.assertRaisesRegex(smoke.SmokeError, "complete Torch package tree"):
            smoke.prepare_smoke(self.request)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.output.exists())

    def test_environment_version_cuda_semantics_are_revalidated(self) -> None:
        environment, _ = smoke.load_canonical_json(
            self.synthetic.environment_lock, "synthetic environment lock"
        )
        environment["cuda"] = "12.9"
        self.synthetic.environment_lock.write_text(smoke.canonical_json(environment))
        environment_raw = self.synthetic.environment_lock.read_bytes()
        self.synthetic.document["inputs"]["environment_lock"] = {
            "path": self.synthetic.environment_lock.relative_to(
                self.workspace
            ).as_posix(),
            "sha256": digest(environment_raw),
            "size": len(environment_raw),
        }
        self.synthetic.evidence.write_text(
            smoke.canonical_json(self.synthetic.document)
        )
        request = replace(
            self.request,
            expected_probe_evidence_sha256=digest(
                self.synthetic.evidence.read_bytes()
            ),
        )
        with self.assertRaisesRegex(
            smoke.SmokeError, "version/git/CUDA/tree identity"
        ):
            smoke.prepare_smoke(request)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.output.exists())

    def test_installed_native_tamper_is_rejected_before_runtime(self) -> None:
        libtriton = self.synthetic.site / "triton" / "_C" / "libtriton.so"
        with libtriton.open("ab") as sink:
            sink.write(b"tamper")
        with self.assertRaisesRegex(smoke.SmokeError, "installed Triton file changed"):
            smoke.prepare_smoke(self.request)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.output.exists())

    def test_post_execution_tree_tamper_prevents_publication(self) -> None:
        context = self.runtime_context()

        def execute(_: smoke.RuntimeContext) -> dict[str, object]:
            artifact_dir = self.cache / "post-tamper"
            artifact_dir.mkdir()
            (artifact_dir / "vector_add.cubin").write_bytes(b"synthetic-cubin\x00")
            triton_init = self.synthetic.site / "triton" / "__init__.py"
            with triton_init.open("ab") as sink:
                sink.write(b"# post-execution tamper\n")
            return self.execution()

        with (
            mock.patch.dict(smoke.os.environ, {}, clear=True),
            mock.patch.object(smoke, "_validate_isolated_invocation"),
            mock.patch.object(
                smoke, "_capture_gpu_run_context", return_value=self.run_context()
            ),
            mock.patch.object(smoke, "_load_runtime", return_value=context),
            mock.patch.object(smoke, "_execute_vector_add", side_effect=execute),
            mock.patch.object(
                smoke,
                "_collect_runtime_provenance",
                return_value=self.provenance(),
            ),
        ):
            with self.assertRaisesRegex(smoke.SmokeError, "installed Triton file changed"):
                smoke.run_smoke(self.request)
        self.assertFalse(self.output.exists())

    def test_source_contract_is_masked_fp32_correctness_not_performance(self) -> None:
        source = (
            ROOT / "benchmarks" / "operators" / "triton_reference_sm120.py"
        ).read_text()
        self.assertIn("offsets < n_elements", source)
        self.assertIn("tl.load", source)
        self.assertIn("tl.store", source)
        self.assertIn("torch.equal", source)
        self.assertIn('"performance_result": False', source)
        self.assertNotIn("torch.cuda.Event", source)
        self.assertNotIn("time.perf_counter", source)


if __name__ == "__main__":
    unittest.main()
