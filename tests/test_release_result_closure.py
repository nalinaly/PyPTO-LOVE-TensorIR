from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zlib

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = REPOSITORY / "tools" / name
    spec = importlib.util.spec_from_file_location(f"closure_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = _load_tool("render_release_results.py")
sync = _load_tool("sync_release_docs.py")


def _identity(profile: str) -> dict[str, object]:
    model = {
        "manifest": {"sha256": "1" * 64},
        "files": [
            {"path": "model.bin", "bytes": 16, "sha256": "2" * 64}
        ],
    }
    model["identity_sha256"] = renderer.canonical_json_sha256(model)
    distributions = {
        name: {"file_count": 1, "content_tree_sha256": digest * 64}
        for name, digest in (
            ("pypto", "6"),
            ("pypto-framework-plugins", "a"),
            ("pypto-kernels", "b"),
        )
    }
    global_fields = {
        "schema": 1,
        "kind": "qwen35-release-evidence-identity",
        "model": model,
        "environment_locks": {
            "pypto": {
                "sha256": "3" * 64,
                "identity": {
                    "torch": "2.13.0+cu130",
                    "torch_git": "a" * 40,
                    "cuda": "13.0",
                    "hip": None,
                    "torch_tree_sha256": "c" * 64,
                    "distributions_sha256": "d" * 64,
                },
            },
            "baseline": {
                "sha256": "4" * 64,
                "identity": {
                    "torch": "2.13.0+cu130",
                    "torch_git": "a" * 40,
                    "cuda": "13.0",
                    "hip": None,
                    "torch_tree_sha256": "e" * 64,
                    "distributions_sha256": "f" * 64,
                },
            },
            "manifest": {"sha256": "0" * 64},
        },
        "candidate_packages": {
            "dso": {
                "path": "envs/pypto/pypto_core.so",
                "bytes": 1024,
                "sha256": "5" * 64,
            },
            "distributions": distributions,
        },
        "sources": {
            "pypto": {"commit": "7" * 40, "tree": "1" * 40, "clean": True},
            "tensor_ir": {
                "commit": "8" * 40,
                "tree": "2" * 40,
                "clean": True,
            },
            "sglang": {
                "commit": "9" * 40,
                "tree": "3" * 40,
                "clean": True,
                "version": "0.5.18",
            },
            "cuda_tile": {
                "commit": "c" * 40,
                "tree": "4" * 40,
                "clean": True,
            },
            "llvm": {"commit": "d" * 40, "tree": "5" * 40, "clean": True},
        },
        "expected_compiler": {
            "pypto_revision": "7" * 40,
            "tensor_ir_revision": "8" * 40,
            "cuda_tile_revision": "c" * 40,
            "llvm_revision": "d" * 40,
        },
        "gpu": {
            "name": "RTX 5090",
            "uuid": "GPU-release",
            "compute_capability": "12.0",
            "total_memory_mib": 24576,
            "driver": "590.1",
        },
    }
    identity = {
        **global_fields,
        "selected_environment_lock": profile,
        "runtime": {
            "profile": profile,
            "prefix": f"envs/{profile}",
            "torch": {
                "version": "2.13.0+cu130",
                "git": "a" * 40,
                "cuda_toolkit": "13.0",
                "hip": None,
                "module": f"envs/{profile}/torch/__init__.py",
            },
            "sglang": {
                "version": "0.5.18",
                "module": "sources/sglang/python/sglang/__init__.py",
                "module_sha256": "f" * 64,
                "source_commit": "9" * 40,
                "source_tree": "3" * 40,
            },
        },
        "compiler": (
            {
                "compiled": True,
                "compiler_factory_available": True,
                "pypto_revision": "7" * 40,
                "tensor_ir_revision": "8" * 40,
                "cuda_tile_revision": "c" * 40,
                "llvm_revision": "d" * 40,
                "cuda_toolkit_root": "/usr/local/cuda-13.3",
                "cuda_toolkit_version": "13.3.73",
                "tileiras_real_path": "/usr/local/cuda-13.3/bin/tileiras",
                "tileiras_version": "13.3.36",
                "tileiras_sha256": "e" * 64,
                "sm120_target": "sm_120a",
            }
            if profile == "pypto"
            else None
        ),
    }
    unsigned = copy.deepcopy(identity)
    identity["identity_sha256"] = renderer.canonical_json_sha256(unsigned)
    return identity


def _report_path(root: Path, name: str) -> Path:
    path = root / "runs" / name / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _controller_gpu() -> dict[str, object]:
    return {
        "name": "RTX 5090",
        "compute_capability": "12.0",
        "total_memory_mib": 24576,
        "driver": "590.1",
    }


def test_identity_audit_requires_cross_run_byte_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    audit = renderer.IdentityAudit()
    baseline_path = _report_path(tmp_path, "baseline")
    candidate_path = _report_path(tmp_path, "candidate")
    audit.add(
        {"evidence_identity": _identity("baseline")},
        report_path=baseline_path,
        expected_profile="baseline",
        controller_gpu=_controller_gpu(),
    )
    audit.add(
        {"evidence_identity": _identity("pypto")},
        report_path=candidate_path,
        expected_profile="pypto",
        controller_gpu=_controller_gpu(),
    )
    result = audit.result()
    assert result["gpu"]["uuid"] == "GPU-release"
    assert result["compiler"]["cuda_toolkit_root"] == "cuda-13.3"
    assert result["report_count"] == 2

    drift = _identity("pypto")
    drift["model"]["files"][0]["sha256"] = "f" * 64
    unsigned_model = copy.deepcopy(drift["model"])
    unsigned_model.pop("identity_sha256")
    drift["model"]["identity_sha256"] = renderer.canonical_json_sha256(
        unsigned_model
    )
    unsigned = copy.deepcopy(drift)
    unsigned.pop("identity_sha256")
    drift["identity_sha256"] = renderer.canonical_json_sha256(unsigned)
    with pytest.raises(renderer.ReleaseContractError, match="identity drifted"):
        audit.add(
            {"evidence_identity": drift},
            report_path=candidate_path,
            expected_profile="pypto",
            controller_gpu=_controller_gpu(),
        )


def test_correctness_all_control_kind_selects_reference_and_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    reports = []
    for name in ("reference", "candidate-0", "candidate-1", "candidate-2"):
        path = _report_path(tmp_path, name)
        reports.append(path)
    summary = tmp_path / "runs/control/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "kind": "qwen35-9b-all-control",
                "status": "complete",
                "runs": [
                    {
                        "phase": phase,
                        "return_code": 0,
                        "report": str(path),
                    }
                    for phase, path in zip(
                        ("reference", "candidate", "candidate", "candidate"),
                        reports,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    reference, candidates, inputs = renderer._summary_correctness_paths(summary)
    assert reference == reports[0]
    assert candidates == reports[1:]
    assert inputs[0]["role"] == "correctness-control"


def test_release_summary_sanitizer_rejects_machine_paths_and_placeholders() -> None:
    with pytest.raises(renderer.ReleaseContractError, match="absolute"):
        renderer._require_sanitized({"path": "/machine/private/result.json"})
    with pytest.raises(renderer.ReleaseContractError, match="placeholder"):
        renderer._require_sanitized({"result": "pending formal GPU gate"})


def test_renderer_requires_nonpausing_pytest_cpu_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    run = tmp_path / "runs/cpu-policy"
    run.mkdir(parents=True)
    report = run / "operator-structure.json"
    process = run / "process.json"
    payload = {
        "schema": 2,
        "mode": "cpu-bounded",
        "framework_profile": "pypto",
        "status": "exited",
        "return_code": 0,
        "abort_reason": None,
        "pid": 777,
        "pgid": 777,
        "sid": 777,
        "pauses": [],
        "low_memory_samples": [
            {
                "sample_index": 3,
                "mem_available_kib": 11 * 1024**2,
                "owned_sid_rss_kib": 14 * 1024**2,
                "consecutive_emergency_samples": 0,
            },
            {
                "sample_index": 4,
                "mem_available_kib": 5 * 1024**2,
                "owned_sid_rss_kib": 14 * 1024**2,
                "consecutive_emergency_samples": 1,
            },
        ],
        "low_memory_sample_count": 2,
        "sample_period_ms": 200,
        "maximum_consecutive_emergency_samples": 1,
        "maximum_owned_sid_rss_kib": 14 * 1024**2,
        "session_cleanup": {
            "schema": 1,
            "kind": "pypto-owned-session-cleanup",
            "sid": 777,
            "term_signaled": [],
            "kill_signaled": [],
            "rejected": [],
            "survivors": [],
            "complete": True,
        },
        "policy": {
            "schema": 2,
            "kind": "pypto-cpu-memory-policy",
            "workload_mode": "pytest-resident-workers",
            "launch_admission_floor_kib": None,
            "pause_enabled": False,
            "pause_memory_floor_kib": None,
            "resume_memory_floor_kib": None,
            "low_memory_recording_floor_kib": 12 * 1024**2,
            "emergency_abort_memory_floor_kib": 6 * 1024**2,
            "emergency_abort_consecutive_samples": 3,
            "low_memory_action": "record-and-continue",
            "emergency_action": "terminate-owned-pgid-after-consecutive-samples",
            "parallelism": 24,
            "external_process_signals": False,
            "pause_signal_scope": "verified-owned-pgid-only",
            "termination_signal_scope": (
                "verified-pgid-then-verified-session-residuals"
            ),
            "successful_exit_cleanup": "natural-session-empty",
            "rss_accounting_scope": "owned-session-id",
            "formal_identity_verified": True,
        },
    }
    process.write_text(json.dumps(payload), encoding="utf-8")
    evidence = renderer._cpu_controller_evidence(report)
    assert evidence[0]["role"] == "cpu-controller"

    payload["policy"]["pause_enabled"] = True
    process.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(renderer.ReleaseContractError, match="did not accept"):
        renderer._cpu_controller_evidence(report)


def test_renderer_requires_complete_gpu_session_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    run = tmp_path / "runs/gpu-policy"
    run.mkdir(parents=True)
    report = run / "performance.json"
    gpu = {
        "name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "memory_mib": "24576",
        "driver": "test-driver",
    }
    audit = {
        "gpu": gpu,
        "external_compute_pids": [],
        "protected_compute_pids": [],
        "protected_runtime_mapping_pids": [],
        "unreadable_protected_maps": [],
        "owned_compute_pids": [],
    }
    for name in ("initial-audit.json", "locked-audit.json", "post-identity-audit.json"):
        (run / name).write_text(json.dumps(audit), encoding="utf-8")
    process = run / "process.json"
    payload = {
        "schema": 2,
        "mode": "gpu-bounded",
        "framework_profile": "pypto",
        "status": "exited",
        "return_code": 0,
        "abort_reason": None,
        "pid": 888,
        "pgid": 888,
        "sid": 888,
        "maximum_owned_sid_rss_kib": 1024,
        "policy": {
            "schema": 2,
            "kind": "pypto-gpu-resource-policy",
            "launch_admission_floor_kib": None,
            "host_abort_floor_kib": 12 * 1024**2,
            "host_emergency_abort_floor_kib": 11 * 1024**2,
            "host_floor_consecutive_samples": 3,
            "gpu_free_floor_mib": 4 * 1024,
            "external_process_signals": False,
            "formal_identity_verified": True,
            "termination_signal_scope": (
                "verified-pgid-then-verified-session-residuals"
            ),
            "successful_exit_cleanup": "natural-session-empty",
            "rss_accounting_scope": "owned-session-id",
        },
        "session_cleanup": {
            "schema": 1,
            "kind": "pypto-owned-session-cleanup",
            "sid": 888,
            "term_signaled": [],
            "kill_signaled": [],
            "rejected": [],
            "survivors": [],
            "complete": True,
        },
        "post_audit": audit,
    }
    process.write_text(json.dumps(payload), encoding="utf-8")
    evidence, observed_gpu = renderer._controller_evidence(report, "pypto")
    assert observed_gpu["compute_capability"] == "12.0"
    assert any(item["role"] == "gpu-controller" for item in evidence)

    payload["policy"]["gpu_free_floor_mib"] = 0
    payload["policy"]["gpu_free_floor_mode"] = "disabled-completion-only"
    process.write_text(json.dumps(payload), encoding="utf-8")
    renderer._controller_evidence(
        report,
        "pypto",
        expected_gpu_free_floor_mib=0,
    )
    payload["policy"]["gpu_free_floor_mib"] = 4 * 1024
    payload["policy"].pop("gpu_free_floor_mode")

    payload["session_cleanup"]["complete"] = False
    payload["session_cleanup"]["survivors"] = [{"pid": 42}]
    process.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(renderer.ReleaseContractError, match="policy drifted"):
        renderer._controller_evidence(report, "pypto")

    payload["session_cleanup"]["complete"] = True
    payload["session_cleanup"]["survivors"] = []
    payload["session_cleanup"]["term_signaled"] = [{"pid": 42}]
    process.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(renderer.ReleaseContractError, match="policy drifted"):
        renderer._controller_evidence(report, "pypto")


def test_release_closure_sources_are_workstation_portable() -> None:
    for relative in (
        "benchmarks/release/evidence_identity.py",
        "benchmarks/release/operator_performance_runtime.py",
        "tools/run_operator_performance.py",
        "tools/render_release_results.py",
        "tools/sync_release_docs.py",
    ):
        text = (REPOSITORY / relative).read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "/Users/" not in text


def _png(width: int = 1280, height: int = 720) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    row = b"\x00" + b"\x00\x00\x00" * width
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            chunk(b"tEXt", b"release-evidence\x00" + b"x" * 5000),
            chunk(b"IDAT", zlib.compress(row * height)),
            chunk(b"IEND", b""),
        )
    )


def _docs_fixture(root: Path) -> tuple[dict[str, Path], Path, Path, Path]:
    (root / "reports/local-blog").mkdir(parents=True)
    documents = {
        "readme_zh": root / "README.md",
        "readme_en": root / "README_EN.md",
        "blog": root / "reports/local-blog/blog.md",
    }
    documents["readme_zh"].write_text(
        "<!-- RELEASE_RESULTS:SUMMARY_BEGIN -->old<!-- RELEASE_RESULTS:SUMMARY_END -->\n",
        encoding="utf-8",
    )
    documents["readme_en"].write_text(
        "<!-- RELEASE_RESULTS:SUMMARY_BEGIN -->old<!-- RELEASE_RESULTS:SUMMARY_END -->\n",
        encoding="utf-8",
    )
    blog_markers = "\n".join(
        f"<!-- RELEASE_RESULTS:{marker}_BEGIN -->old<!-- RELEASE_RESULTS:{marker}_END -->"
        for marker in sync.DOCUMENT_FRAGMENTS["blog"]
    )
    documents["blog"].write_text(blog_markers + "\n", encoding="utf-8")

    evidence = root / "runs/release-results-test"
    marker_dir = evidence / "markers"
    marker_dir.mkdir(parents=True)
    summary = evidence / "release-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "qwen35-9b-release-results",
                "release_identity": {"sha256": "a"},
                "operator_correctness": {"pass": True},
                "operator_performance": {"pass": True},
                "model_correctness": {"pass": True},
                "performance": {"pass": True},
                "profile_reconciliation": {"pass": True},
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    fragment_records = {}
    fragment_names = {
        value
        for mapping in sync.DOCUMENT_FRAGMENTS.values()
        for value in mapping.values()
    }
    for name in fragment_names:
        path = marker_dir / f"{name.lower()}.md"
        path.write_text(f"{name}: PASS\n", encoding="utf-8")
        fragment_records[name] = {
            "path": path.relative_to(evidence).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    fragment_manifest = evidence / "marker-fragments.json"
    fragment_manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "qwen35-release-marker-fragments",
                "release_summary": {
                    "path": "release-summary.json",
                    "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                },
                "fragments": fragment_records,
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    assets = root / "docs/assets/screenshots"
    assets.mkdir(parents=True)
    screenshot_records = {}
    for role in sync.EXPECTED_SCREENSHOTS:
        path = assets / f"{role}.png"
        path.write_bytes(_png())
        evidence_payloads = (
            [
                {
                    "kind": "pypto-sm120-release-build",
                    "stage": stage,
                    "status": "complete",
                }
                for stage in ("wheels", "native", "ctest", "install")
            ]
            if role == "build"
            else [
                {
                    "kind": next(iter(sync.SCREENSHOT_EVIDENCE_KINDS[role])),
                    "status": "complete",
                }
            ]
        )
        evidence_records = []
        for index, payload in enumerate(evidence_payloads):
            evidence_path = root / "runs" / f"evidence-{role}-{index}.json"
            evidence_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            evidence_records.append(
                {
                    "path": evidence_path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                }
            )
        screenshot_records[role] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "caption_zh": f"{role} 正式证据",
            "caption_en": f"Formal {role} evidence",
            "evidence": evidence_records,
        }
    screenshot_manifest = evidence / "screenshots.json"
    screenshot_manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "qwen35-release-screenshots",
                "release_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "screenshots": screenshot_records,
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    return documents, summary, fragment_manifest, screenshot_manifest


def _sync_argv(
    documents: dict[str, Path],
    summary: Path,
    fragments: Path,
    screenshots: Path,
    *,
    check: bool,
) -> list[str]:
    argv = [
        "sync_release_docs.py",
        "--release-summary",
        str(summary),
        "--marker-fragments",
        str(fragments),
        "--readme-zh",
        str(documents["readme_zh"]),
        "--readme-en",
        str(documents["readme_en"]),
        "--blog",
        str(documents["blog"]),
        "--screenshots-manifest",
        str(screenshots),
    ]
    if check:
        argv.append("--check")
    return argv


def test_docs_sync_is_evidence_bound_and_updates_only_controlled_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, summary, fragments_path, screenshots_path = _docs_fixture(tmp_path)
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    loaded, digest = sync._load_fragments(summary, fragments_path)
    assert set(loaded) == {
        value for mapping in sync.DOCUMENT_FRAGMENTS.values() for value in mapping.values()
    }
    assert sync._verify_screenshots(screenshots_path, digest).keys() == (
        sync.EXPECTED_SCREENSHOTS
    )
    updates = {}
    for name, path in documents.items():
        text = path.read_text(encoding="utf-8")
        for marker, fragment_name in sync.DOCUMENT_FRAGMENTS[name].items():
            text = sync._replace_marker(text, marker, loaded[fragment_name])
        updates[path] = text
    sync._atomic_replace_many(updates)
    assert "SUMMARY_ZH: PASS" in documents["readme_zh"].read_text(encoding="utf-8")
    assert "SUMMARY_EN: PASS" in documents["readme_en"].read_text(encoding="utf-8")
    assert "CONCLUSION_ZH: PASS" in documents["blog"].read_text(encoding="utf-8")

    gallery = sync._screenshot_markdown(
        documents["readme_en"],
        "en",
        ("build", "model-inference"),
        sync._verify_screenshots(screenshots_path, digest),
    )
    assert "![Formal build evidence](docs/assets/screenshots/build.png)" in gallery
    assert "model-inference.png" in gallery


def test_docs_sync_rejects_placeholder_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _documents, summary, fragments_path, _screenshots_path = _docs_fixture(tmp_path)
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    manifest = json.loads(fragments_path.read_text(encoding="utf-8"))
    record = manifest["fragments"]["SUMMARY_ZH"]
    fragment = fragments_path.parent / record["path"]
    fragment.write_text("待正式 GPU gate\n", encoding="utf-8")
    record["sha256"] = hashlib.sha256(fragment.read_bytes()).hexdigest()
    fragments_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(sync.ReleaseContractError, match="placeholder"):
        sync._load_fragments(summary, fragments_path)


def test_docs_sync_check_is_fail_closed_read_only_and_then_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, summary, fragments, screenshots = _docs_fixture(tmp_path)
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    before = {name: path.read_bytes() for name, path in documents.items()}
    monkeypatch.setattr(
        sys,
        "argv",
        _sync_argv(
            documents, summary, fragments, screenshots, check=True
        ),
    )
    with pytest.raises(sync.ReleaseContractError) as failure:
        sync.main()
    message = str(failure.value)
    assert "not synchronized" in message
    assert "README.md" in message
    assert "README_EN.md" in message
    assert "reports/local-blog/blog.md" in message
    assert {name: path.read_bytes() for name, path in documents.items()} == before

    monkeypatch.setattr(
        sys,
        "argv",
        _sync_argv(
            documents, summary, fragments, screenshots, check=False
        ),
    )
    assert sync.main() == 0
    synchronized = {name: path.read_bytes() for name, path in documents.items()}
    monkeypatch.setattr(
        sys,
        "argv",
        _sync_argv(
            documents, summary, fragments, screenshots, check=True
        ),
    )
    assert sync.main() == 0
    assert {name: path.read_bytes() for name, path in documents.items()} == synchronized


def test_docs_sync_check_detects_fragment_input_drift_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, summary, fragments, screenshots = _docs_fixture(tmp_path)
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        _sync_argv(
            documents, summary, fragments, screenshots, check=False
        ),
    )
    assert sync.main() == 0

    manifest = json.loads(fragments.read_text(encoding="utf-8"))
    record = manifest["fragments"]["SUMMARY_ZH"]
    fragment = fragments.parent / record["path"]
    fragment.write_text("SUMMARY_ZH: PASS UPDATED\n", encoding="utf-8")
    record["sha256"] = hashlib.sha256(fragment.read_bytes()).hexdigest()
    fragments.write_text(json.dumps(manifest), encoding="utf-8")
    before = {name: path.read_bytes() for name, path in documents.items()}

    monkeypatch.setattr(
        sys,
        "argv",
        _sync_argv(
            documents, summary, fragments, screenshots, check=True
        ),
    )
    with pytest.raises(sync.ReleaseContractError) as failure:
        sync.main()
    assert "README.md" in str(failure.value)
    assert "reports/local-blog/blog.md" in str(failure.value)
    assert {name: path.read_bytes() for name, path in documents.items()} == before


def test_docs_sync_check_detects_screenshot_input_drift_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, summary, fragments, screenshots = _docs_fixture(tmp_path)
    monkeypatch.setattr(sync, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        _sync_argv(
            documents, summary, fragments, screenshots, check=False
        ),
    )
    assert sync.main() == 0

    manifest = json.loads(screenshots.read_text(encoding="utf-8"))
    manifest["screenshots"]["build"]["caption_en"] = (
        "Updated formal build evidence"
    )
    screenshots.write_text(json.dumps(manifest), encoding="utf-8")
    before = {name: path.read_bytes() for name, path in documents.items()}

    monkeypatch.setattr(
        sys,
        "argv",
        _sync_argv(
            documents, summary, fragments, screenshots, check=True
        ),
    )
    with pytest.raises(sync.ReleaseContractError) as failure:
        sync.main()
    assert "README_EN.md" in str(failure.value)
    assert "README.md" not in str(failure.value)
    assert "reports/local-blog/blog.md" not in str(failure.value)
    assert {name: path.read_bytes() for name, path in documents.items()} == before
