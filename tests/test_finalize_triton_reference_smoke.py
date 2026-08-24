from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import secrets
import shutil
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools/finalize_triton_reference_smoke.py"
    spec = importlib.util.spec_from_file_location("test_smoke_finalizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalizer = load_tool()


class SmokeFinalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = (
            f"pypto-20990101T000000Z-{100000 + (id(self) % 899999)}-"
            f"{secrets.token_hex(3)}"
        )
        self.run_dir = ROOT / "runs" / self.run_id
        self.run_dir.mkdir()
        self.addCleanup(shutil.rmtree, self.run_dir)
        self.provisional = self.run_dir / "provisional.json"
        self.final = self.run_dir / "final.json"
        runner = ROOT / "benchmarks/operators/triton_reference_sm120.py"

        def identity(path: pathlib.Path) -> dict[str, object]:
            return {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": finalizer.sha256_file(path),
                "size": path.stat().st_size,
            }

        self.base_python = self.run_dir / "base-python"
        self.base_python.write_bytes(b"synthetic-cpython-3.14.6\x00")
        self.base_python.chmod(0o755)
        self.environment_lock = self.run_dir / "ENVIRONMENT.lock"
        self.environment_lock.write_text(
            finalizer.canonical_json({"kind": "synthetic-environment-lock"})
        )
        self.audit = self.run_dir / "triton-wheel-audit.json"
        self.audit.write_text(
            finalizer.canonical_json(
                {"acceptance": "accepted", "audit": "triton-workspace-wheel"}
            )
        )
        self.wheel = self.run_dir / "triton-3.7.1+git5d6048aa.whl"
        self.wheel.write_bytes(b"synthetic-wheel\x00")
        self.torch_site = self.run_dir / "torch-site"
        (self.torch_site / "torch").mkdir(parents=True)
        (self.torch_site / "torch/__init__.py").write_text("# synthetic torch\n")
        torch_dist = self.torch_site / "torch-2.13.0+cu130.dist-info"
        torch_dist.mkdir()
        (torch_dist / "METADATA").write_text("Name: torch\nVersion: 2.13.0+cu130\n")

        self.probe_prefix = self.run_dir / "probe"
        self.probe_site = self.probe_prefix / "lib/python3.14/site-packages"
        self.runtime_view = self.probe_prefix / ".torch-runtime-view"
        triton = self.probe_site / "triton"
        libtriton = triton / "_C/libtriton.so"
        ptxas = self.probe_site / finalizer.PTXAS_BLACKWELL_RELATIVE
        self.runtime_view.mkdir(parents=True)
        libtriton.parent.mkdir(parents=True)
        ptxas.parent.mkdir(parents=True)
        (triton / "__init__.py").write_text("__version__ = '3.7.1'\n")
        libtriton.write_bytes(b"synthetic-libtriton\x00")
        ptxas.write_bytes(b"synthetic-ptxas-blackwell\x00")

        base_identity = identity(self.base_python)
        environment_identity = identity(self.environment_lock)
        torch_identity = {
            "dist_info_name": torch_dist.name,
            "metadata_sha256": finalizer.sha256_file(torch_dist / "METADATA"),
            "module_sha256": finalizer.sha256_file(
                self.torch_site / "torch/__init__.py"
            ),
            "path": self.torch_site.relative_to(ROOT).as_posix(),
            "torch_tree_bytes": 47,
            "torch_tree_files": 2,
            "torch_tree_sha256": "3" * 64,
            "version": finalizer.EXPECTED_TORCH_VERSION,
        }
        wheel_identity = {
            **identity(self.wheel),
            "audit_evidence": identity(self.audit),
            "filename": self.wheel.name,
        }
        probe_site_token = finalizer._probe_token(
            self.probe_site, self.probe_prefix
        )
        view_token = finalizer._probe_token(
            self.runtime_view, self.probe_prefix
        )
        libtriton_token = finalizer._probe_token(libtriton, self.probe_prefix)
        ptxas_token = finalizer._probe_token(ptxas, self.probe_prefix)
        torch_file_token = "$TORCH_SITE_PACKAGES/torch/__init__.py"
        self.probe_process = {
            "libtriton_maps": [libtriton_token],
            "ptxas_blackwell": {
                "audited_full_version": finalizer.EXPECTED_PTXAS_FULL_VERSION,
                "path": ptxas_token,
                "reported_release": finalizer.EXPECTED_PTXAS_RELEASE,
                "sha256": finalizer.sha256_file(ptxas),
            },
            "torch": {
                "cuda": finalizer.EXPECTED_TORCH_CUDA,
                "file": torch_file_token,
                "git_version": "cf30153c4c131c8164ee7798e5022d810682e2cb",
                "hip": None,
                "version": finalizer.EXPECTED_TORCH_VERSION,
            },
        }
        self.probe = self.run_dir / "triton-wheel-probe.json"
        probe_document = {
            "acceptance": "accepted",
            "inputs": {
                "base_python": base_identity,
                "environment_lock": environment_identity,
                "torch_site_packages": torch_identity,
                "wheel": wheel_identity,
            },
            "installation": {
                "prefix": self.probe_prefix.relative_to(ROOT).as_posix(),
                "scheme": {
                    "platlib": self.probe_site.relative_to(ROOT).as_posix()
                },
                "torch_runtime_view": {
                    "path": self.runtime_view.relative_to(ROOT).as_posix()
                },
            },
            "probe": finalizer.PROBE,
            "runtime": {
                "processes": [self.probe_process, self.probe_process],
                "processes_count": 2,
            },
            "schema_version": 1,
        }
        self.probe.write_text(finalizer.canonical_json(probe_document))
        probe_identity = identity(self.probe)

        self.cache = self.run_dir / "cache"
        self.cache.mkdir()
        cache_artifact = self.cache / "vector_add.cubin"
        cache_artifact.write_bytes(b"synthetic-cubin\x00")
        artifacts = [
            {
                "path": cache_artifact.relative_to(self.cache).as_posix(),
                "sha256": finalizer.sha256_file(cache_artifact),
                "size": cache_artifact.stat().st_size,
            }
        ]
        integrity_snapshot = {
            "environment_lock": {
                "cuda": finalizer.EXPECTED_TORCH_CUDA,
                "hip": None,
                **environment_identity,
                "torch": finalizer.EXPECTED_TORCH_VERSION,
                "torch_git": self.probe_process["torch"]["git_version"],
            },
            "installed_triton": {
                "native_bytes": libtriton.stat().st_size,
                "native_entries_count": 1,
                "native_entries_sha256": "4" * 64,
                "package_entries_count": 3,
                "package_entries_sha256": "5" * 64,
                "record_entries_count": 3,
                "record_entries_sha256": "6" * 64,
            },
            "linked_inputs": {
                "base_python": base_identity,
                "probe_evidence": probe_identity,
                "wheel": {
                    "audit_evidence_sha256": wheel_identity["audit_evidence"][
                        "sha256"
                    ],
                    "sha256": wheel_identity["sha256"],
                },
            },
            "torch_tree": {
                **torch_identity,
                "cuda": finalizer.EXPECTED_TORCH_CUDA,
                "git_version": self.probe_process["torch"]["git_version"],
                "hip": None,
            },
        }
        provisional = {
            "acceptance": finalizer.PROVISIONAL_STATUS,
            "inputs": {
                "base_python": base_identity,
                "environment_lock": environment_identity,
                "probe_evidence": probe_identity,
                "probe_prefix": self.probe_prefix.relative_to(ROOT).as_posix(),
                "probe_site": self.probe_site.relative_to(ROOT).as_posix(),
                "runner": identity(runner),
                "torch_runtime_view": self.runtime_view.relative_to(ROOT).as_posix(),
                "torch_site_packages": torch_identity,
                "wheel": wheel_identity,
            },
            "runtime": {
                "compiled_cache": {
                    "artifacts": artifacts,
                    "artifacts_count": 1,
                    "artifacts_sha256": hashlib.sha256(
                        finalizer.canonical_json(artifacts).encode("ascii")
                    ).hexdigest(),
                    "cubin_count": 1,
                    "fresh_before_run": True,
                    "path": self.cache.relative_to(ROOT).as_posix(),
                    "total_bytes": cache_artifact.stat().st_size,
                },
                "correctness": {
                    "block_size": finalizer.BLOCK_SIZE,
                    "comparison": "torch.equal",
                    "dtype": "float32",
                    "equal": True,
                    "kernel": "masked-vector-add",
                    "n_elements": finalizer.VECTOR_ELEMENTS,
                    "reference_provider": "torch",
                },
                "device": {
                    "compute_capability": [12, 0],
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 5090 Laptop GPU",
                },
                "gpu_execution": True,
                "integrity": {
                    "before": integrity_snapshot,
                    "after": integrity_snapshot,
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
                    "libtriton_maps": [libtriton_token],
                    "module_paths": {
                        "triton": [
                            finalizer._probe_token(
                                triton / "__init__.py", self.probe_prefix
                            )
                        ],
                        "triton._C.libtriton": [libtriton_token],
                    },
                    "python": {
                        "executable": "$PROBE_PREFIX/bin/python",
                        "resolved_executable": base_identity["path"],
                        "resolved_sha256": base_identity["sha256"],
                        "resolved_size": base_identity["size"],
                    },
                    "sys_path": [probe_site_token, view_token],
                    "torch_file": torch_file_token,
                    "torch_runtime": dict(self.probe_process["torch"]),
                },
                "ptxas_blackwell": {
                    **self.probe_process["ptxas_blackwell"],
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
            "run_context": {
                "mode": "gpu-benchmark",
                "pgid": 123,
                "pid": 124,
                "preflight": {},
                "provisional_evidence_path": self.provisional.relative_to(
                    ROOT
                ).as_posix(),
                "run_id": self.run_id,
            },
            "schema_version": 1,
            "scope": {
                "coverage_result": False,
                "performance_result": False,
                "provider": "triton",
                "pypto_kernel": False,
                "reference_only": True,
            },
            "smoke": finalizer.SMOKE,
        }
        self.provisional.write_text(finalizer.canonical_json(provisional))
        self.provisional_sha = finalizer.sha256_file(self.provisional)
        self.preflight = {
            "mode": "gpu-benchmark",
            "nvidia_compute_pids": [],
            "ok": True,
            "protected_cpu_only_coexistence_requested": False,
            "protected_heavy_processes": [],
        }
        preflight_path = self.run_dir / "preflight.json"
        preflight_path.write_text(finalizer.canonical_json(self.preflight))
        preflight_sha = finalizer.sha256_file(preflight_path)
        self.process = {
            "coexistence": {"requested": False},
            "coexistence_pauses": [],
            "command": [
                str(ROOT / "benchmarks/operators/triton_reference_sm120.py"),
                "--evidence",
                str(self.provisional),
            ],
            "gpu_benchmark_abort": None,
            "mode": "gpu-benchmark",
            "pgid": 123,
            "preflight": {
                "path": str(preflight_path),
                "sha256": preflight_sha,
            },
            "return_code": 0,
            "run_id": self.run_id,
            "status": "exited",
        }
        provisional["run_context"]["preflight"] = {
            "path": preflight_path.relative_to(ROOT).as_posix(),
            "sha256": preflight_sha,
            "size": preflight_path.stat().st_size,
        }
        self.provisional.write_text(finalizer.canonical_json(provisional))
        self.provisional_sha = finalizer.sha256_file(self.provisional)
        (self.run_dir / "process.json").write_text(
            finalizer.canonical_json(self.process)
        )

    def call(self):
        return finalizer.finalize(
            workspace=ROOT,
            provisional_path=self.provisional,
            expected_provisional_sha256=self.provisional_sha,
            run_id=self.run_id,
            output=self.final,
        )

    def rewrite_provisional(self, document: dict[str, object]) -> str:
        self.provisional.write_text(finalizer.canonical_json(document))
        return finalizer.sha256_file(self.provisional)

    def test_finalizes_exclusive_clean_gpu_run(self) -> None:
        document, digest = self.call()
        self.assertEqual(document["acceptance"], "accepted")
        self.assertEqual(document["exclusive_run"]["run_id"], self.run_id)
        self.assertEqual(document["provisional_evidence"]["sha256"], self.provisional_sha)
        self.assertEqual(finalizer.sha256_file(self.final), digest)

    def test_abort_or_preflight_content_drift_is_rejected(self) -> None:
        self.process["gpu_benchmark_abort"] = {"reason": "external-compute"}
        (self.run_dir / "process.json").write_text(
            finalizer.canonical_json(self.process)
        )
        with self.assertRaisesRegex(finalizer.FinalizeError, "did not exit cleanly"):
            self.call()

    def test_no_replace_and_external_anchor_are_enforced(self) -> None:
        self.final.write_bytes(b"sentinel")
        with self.assertRaisesRegex(finalizer.FinalizeError, "already exists"):
            self.call()
        self.assertEqual(self.final.read_bytes(), b"sentinel")
        self.final.unlink()
        with self.assertRaisesRegex(finalizer.FinalizeError, "external anchor"):
            finalizer.finalize(
                workspace=ROOT,
                provisional_path=self.provisional,
                expected_provisional_sha256="0" * 64,
                run_id=self.run_id,
                output=self.final,
            )

    def test_runtime_integrity_cache_and_run_context_are_not_substitutable(self) -> None:
        mutations = {
            "integrity": lambda document: document["runtime"]["integrity"].update(
                {"stable": False}
            ),
            "cache-path": lambda document: document["runtime"][
                "compiled_cache"
            ]["artifacts"][0].update({"path": "../escape.cubin"}),
            "run-context": lambda document: document["run_context"].update(
                {"run_id": "pypto-20990101T000000Z-111111-aaaaaa"}
            ),
        }
        original = json.loads(self.provisional.read_text())
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                document = json.loads(json.dumps(original))
                mutate(document)
                self.provisional.write_text(finalizer.canonical_json(document))
                digest = finalizer.sha256_file(self.provisional)
                with self.assertRaises(finalizer.FinalizeError):
                    finalizer.finalize(
                        workspace=ROOT,
                        provisional_path=self.provisional,
                        expected_provisional_sha256=digest,
                        run_id=self.run_id,
                        output=self.final,
                    )

    def test_input_identities_must_be_complete_live_and_probe_bound(self) -> None:
        mutations = {
            "empty-base": lambda document: document["inputs"].update(
                {"base_python": {}}
            ),
            "wheel-byte-substitution": lambda document: document["inputs"][
                "wheel"
            ].update({"sha256": "0" * 64}),
            "torch-tree-empty": lambda document: document["inputs"][
                "torch_site_packages"
            ].update({"torch_tree_files": 0}),
            "probe-path-substitution": lambda document: document["inputs"].update(
                {"probe_site": document["inputs"]["probe_prefix"]}
            ),
        }
        original = json.loads(self.provisional.read_text())
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                document = json.loads(json.dumps(original))
                mutate(document)
                digest = self.rewrite_provisional(document)
                with self.assertRaises(finalizer.FinalizeError):
                    finalizer.finalize(
                        workspace=ROOT,
                        provisional_path=self.provisional,
                        expected_provisional_sha256=digest,
                        run_id=self.run_id,
                        output=self.final,
                    )

    def test_full_runtime_provenance_ptxas_and_smoke_contract_are_required(self) -> None:
        outside_site = self.probe_prefix / "outside-triton.py"
        outside_site.write_text("# not wheel-site owned\n")
        outside_token = finalizer._probe_token(outside_site, self.probe_prefix)
        mutations = {
            "ptxas-path": lambda document: document["runtime"][
                "ptxas_blackwell"
            ].update({"path": "$PROBE_PREFIX/triton/external-ptxas"}),
            "ptxas-sha": lambda document: document["runtime"][
                "ptxas_blackwell"
            ].update({"sha256": "0" * 64}),
            "missing-python-provenance": lambda document: document["runtime"][
                "provenance"
            ].pop("python"),
            "external-libtriton": lambda document: document["runtime"][
                "provenance"
            ].update({"libtriton_maps": ["/external/libtriton.so"]}),
            "prefix-only-module": lambda document: document["runtime"][
                "provenance"
            ]["module_paths"].update({"triton": [outside_token]}),
            "wrong-shape": lambda document: document["runtime"][
                "correctness"
            ].update({"n_elements": 65_536}),
            "wrong-kernel": lambda document: document["runtime"][
                "correctness"
            ].update({"kernel": "unmasked-vector-add"}),
        }
        original = json.loads(self.provisional.read_text())
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                document = json.loads(json.dumps(original))
                mutate(document)
                digest = self.rewrite_provisional(document)
                with self.assertRaises(finalizer.FinalizeError):
                    finalizer.finalize(
                        workspace=ROOT,
                        provisional_path=self.provisional,
                        expected_provisional_sha256=digest,
                        run_id=self.run_id,
                        output=self.final,
                    )

    def test_integrity_snapshot_schema_and_live_cache_are_exact(self) -> None:
        original = json.loads(self.provisional.read_text())
        mutations = {
            "minimal-equal-snapshot": lambda document: document["runtime"].update(
                {
                    "integrity": {
                        "after": {"tree": "a" * 64},
                        "before": {"tree": "a" * 64},
                        "stable": True,
                    }
                }
            ),
            "empty-installed-digest": lambda document: document["runtime"][
                "integrity"
            ]["before"]["installed_triton"].update(
                {"native_entries_sha256": ""}
            ),
            "unbound-environment": lambda document: document["runtime"][
                "integrity"
            ]["before"]["environment_lock"].update({"sha256": "0" * 64}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                document = json.loads(json.dumps(original))
                mutate(document)
                if name != "minimal-equal-snapshot":
                    document["runtime"]["integrity"]["after"] = json.loads(
                        json.dumps(document["runtime"]["integrity"]["before"])
                    )
                digest = self.rewrite_provisional(document)
                with self.assertRaises(finalizer.FinalizeError):
                    finalizer.finalize(
                        workspace=ROOT,
                        provisional_path=self.provisional,
                        expected_provisional_sha256=digest,
                        run_id=self.run_id,
                        output=self.final,
                    )

        extra = self.cache / "unrecorded.json"
        extra.write_text("{}\n")
        self.rewrite_provisional(original)
        with self.assertRaisesRegex(finalizer.FinalizeError, "cache aggregate"):
            self.call()


if __name__ == "__main__":
    unittest.main()
