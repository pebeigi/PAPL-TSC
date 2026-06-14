#!/usr/bin/env python3
"""Combine repeated live-OMNeT slot result folders into one CI CSV.

Default input folders:
  experiments/omnet_1_press
  experiments/omnet_2_press
  experiments/omnet_3_press

The output has one row per OMNeT slot with mean/std/95% CI computed across
the repeated runs. It is intended for plot_penetration_results.py.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIRS = [
    SCRIPT_DIR / "omnet_1_press",
    SCRIPT_DIR / "omnet_2_press",
    SCRIPT_DIR / "omnet_3_press",
]
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "results" / "omnet_combined_presslight_sumohz4x4_3runs_aggregated.csv"
)
METRICS = ("travel_time", "loss", "reward", "queue", "delay", "throughput")
FILENAME_RE = re.compile(
    r"inference_(?P<agent>[^_]+)_(?P<network>.+?)_from_pr_"
    r"(?P<train_pr>\d+\.\d+)_omnet_(?P<slot>\d+)_.*_aggregated\.csv$"
)
UE_TX_POWER_BY_GROUP = {
    1: 26,
    2: 21,
    3: 16,
    4: 11,
    5: 6,
    6: 1,
}


def t_critical_95(n: int) -> float:
    """Two-sided 95% t critical value for n samples."""
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    if n <= 1:
        return 0.0
    return table.get(n, 1.96)


def parse_result_file(path: Path, repeat_idx: int, repeat_name: str) -> dict[str, Any] | None:
    match = FILENAME_RE.match(path.name)
    if match is None:
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    row = df.iloc[0].to_dict()
    slot = int(match.group("slot"))
    group_run = ((slot - 1) % 6) + 1
    out: dict[str, Any] = {
        "repeat_idx": repeat_idx,
        "repeat_name": repeat_name,
        "source_file": str(path),
        "agent": match.group("agent"),
        "network": match.group("network"),
        "world": row.get("world", "sumo"),
        "train_pr": float(match.group("train_pr")),
        "infer_pr": float(row.get("infer_pr", 1.0)),
        "omnet_slot": slot,
        "omnet_group_run": group_run,
        "ue_tx_power_dbm": UE_TX_POWER_BY_GROUP[group_run],
    }

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        raw_col = metric
        if mean_col in row:
            out[metric] = pd.to_numeric(row[mean_col], errors="coerce")
        elif raw_col in row:
            out[metric] = pd.to_numeric(row[raw_col], errors="coerce")
    return out


def load_repeat_rows(input_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat_idx, folder in enumerate(input_dirs, start=1):
        for path in sorted(folder.glob("*_aggregated.csv")):
            row = parse_result_file(path, repeat_idx, folder.name)
            if row is not None:
                rows.append(row)
    if not rows:
        raise SystemExit("No aggregated OMNeT CSV rows found in the input folders.")
    return pd.DataFrame(rows)


def aggregate_repeats(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "agent",
        "network",
        "world",
        "train_pr",
        "infer_pr",
        "omnet_slot",
        "omnet_group_run",
        "ue_tx_power_dbm",
    ]
    rows: list[dict[str, Any]] = []
    for key_values, group in raw.groupby(keys, dropna=False, sort=True):
        out = dict(zip(keys, key_values))
        out["n_success"] = int(len(group))
        out["n_total"] = int(len(group))
        out["source_repeats"] = ",".join(sorted(group["repeat_name"].astype(str).unique()))

        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = (
                t_critical_95(len(values)) * std / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_ci95_low"] = mean - half_width
            out[f"{metric}_ci95_high"] = mean + half_width
        rows.append(out)

    return pd.DataFrame(rows).sort_values(["omnet_slot"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine repeated live-OMNeT slot CSV folders into one CI CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUT_DIRS,
        help="Repeat result folders containing *_aggregated.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Combined aggregated CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dirs]
    raw = load_repeat_rows(input_dirs)
    combined = aggregate_repeats(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Loaded {len(raw)} rows from {len(input_dirs)} repeat folders.")
    print(f"Wrote {len(combined)} combined slot rows:")
    print(f"  {args.output}")


if __name__ == "__main__":
    main()
