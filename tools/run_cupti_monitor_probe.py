#!/usr/bin/env python3
"""Probe Torch's pinned CUPTI trace-window and annotation contract."""

from __future__ import annotations

import json
import os
import pathlib
import traceback


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    run_id = os.environ.get("PYPTO_RUN_ID", "manual")
    run_dir = ROOT / "runs" / run_id
    report_path = run_dir / "cupti-monitor-probe.json"
    report: dict[str, object] = {"run_id": run_id, "status": "starting"}
    monitor_started = False
    try:
        import torch
        from torch.profiler import _cupti_monitor as monitor_api

        monitor = monitor_api.start_collection(run_dir / "cupti-monitor")
        monitor_started = True
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
            window = monitor.end_trace_window()
            if not window["events"]:
                raise RuntimeError(
                    f"CUPTI trace window {index} returned no activity records"
                )
            windows.append(window)
        stats = monitor_api.stop_collection()
        monitor_started = False
        report.update(
            {
                "status": "complete",
                "event_counts": [len(window["events"]) for window in windows],
                "external_ids": external_ids,
                "popped_ids": popped_ids,
                "result_sum": float(result.sum().cpu()),
                "trace_windows": windows,
                "stats": stats,
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

                monitor_api.stop_collection()
            except BaseException:
                pass
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
