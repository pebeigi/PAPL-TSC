#!/usr/bin/env python3
"""Summarize LibSignal runtime profiling CSVs.

Runtime profiles are written by trainer/tsc_trainer.py as *_RUNTIME.csv files
next to the normal *_DTL.log files. This script scans those files and reports
wall-clock runtime, real-time factor, DRL action-selection time, and OMNeT CSV
read/observation-building overhead.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_ROOT = Path("/home/exx/Desktop/vtc2026/LibSignal-master/data/output_data")
DEFAULT_OUTPUT = Path("/home/exx/Desktop/vtc2026/LibSignal-master/experiments/results/runtime_profile_summary.csv")


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/*_RUNTIME.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["runtime_file"] = str(path)
                rows.append(row)
    return rows


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "simulated_horizon_s",
        "total_wall_s",
        "loop_wall_s",
        "wall_per_sim_s",
        "real_time_factor",
        "decision_wall_mean_s",
        "action_select_mean_ms",
        "env_step_mean_ms",
        "visible_csv_read_mean_ms",
        "observe_mean_ms",
        "observepart_mean_ms",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("model", "unknown")),
            str(row.get("mode", "unknown")),
            str(row.get("use_omnet", "unknown")),
        )
        groups.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for (model, mode, use_omnet), group_rows in sorted(groups.items()):
        out: dict[str, Any] = {
            "model": model,
            "mode": mode,
            "use_omnet": use_omnet,
            "n_runs": len(group_rows),
        }
        for metric in metrics:
            values = [safe_float(row.get(metric)) for row in group_rows]
            out[f"{metric}_mean"] = mean(values) if values else 0.0
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_min"] = min(values) if values else 0.0
            out[f"{metric}_max"] = max(values) if values else 0.0
        out_rows.append(out)
    return out_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No runtime profile rows found.")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize *_RUNTIME.csv files produced by LibSignal tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.root)
    summary = summarize(rows)
    write_csv(args.output, summary)
    print(f"Read {len(rows)} runtime rows.")
    print(f"Wrote summary: {args.output}")
    for row in summary:
        print(
            "{model} {mode} use_omnet={use_omnet}: "
            "wall/sim={wall:.3f}, RTF={rtf:.3f}x, "
            "action={action:.3f} ms".format(
                model=row["model"],
                mode=row["mode"],
                use_omnet=row["use_omnet"],
                wall=row["wall_per_sim_s_mean"],
                rtf=row["real_time_factor_mean"],
                action=row["action_select_mean_ms_mean"],
            )
        )


if __name__ == "__main__":
    main()
