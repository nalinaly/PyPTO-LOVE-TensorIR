#!/usr/bin/env python3
"""Probe Torch's pinned CUPTI trace-window and annotation contract."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.release.cupti_overlay import activate_overlay  # noqa: E402


def main() -> int:
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    run_dir = ROOT / "runs" / run_id
    report_path = run_dir / "cupti-monitor-probe.json"
    report: dict[str, object] = {"run_id": run_id, "status": "starting"}
    monitor_started = False
    try:
        report["cupti_overlay"] = activate_overlay()
        import torch
        from torch.profiler import _cupti_monitor as monitor_api

        monitor = monitor_api.start_collection(run_dir / "cupti-monitor")
        monitor_started = True
        # Establish the CUDA context after CUPTI starts but before opening a
        # measured window, matching the release model/profile lifecycle.
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        windows = []
        external_ids = []
        popped_ids = []
        result = None
        for index in range(3):
            monitor.begin_trace_window()
            annotation = json.dumps(
                {
                    "index": index,
                    "kind": "probe",
                    "provider": "pypto.tensorir",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            external_ids.append(monitor.push_user_annotation(annotation))
            value = torch.ones(4096, device="cuda")
            result = value + value
            popped_ids.append(monitor.pop_user_annotation())
            torch.cuda.synchronize()
            completed_before = int(monitor.stats()["buffers_completed"])
            monitor.flush(forced=True)
            deadline = time.monotonic() + 1.0
            while (
                int(monitor.stats()["buffers_completed"]) <= completed_before
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            window = monitor.end_trace_window()
            windows.append(window)
        stats = monitor_api.stop_collection()
        monitor_started = False
        report.update(
            {
                "event_counts": [len(window["events"]) for window in windows],
                "external_ids": external_ids,
                "popped_ids": popped_ids,
                "trace_windows": windows,
                "stats": stats,
            }
        )
        empty_windows = [
            index for index, window in enumerate(windows) if not window["events"]
        ]
        if empty_windows:
            raise RuntimeError(
                f"CUPTI trace windows returned no activity records: {empty_windows}"
            )
        report.update(
            {
                "status": "complete",
                "result_sum": float(result.sum().cpu()),
            }
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        if monitor_started:
            try:
                from torch.profiler import _cupti_monitor as monitor_api

                report["stats"] = monitor_api.stop_collection()
            except BaseException as stop_error:
                report["collector_stop_error"] = (
                    f"{type(stop_error).__name__}: {stop_error}"
                )
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
