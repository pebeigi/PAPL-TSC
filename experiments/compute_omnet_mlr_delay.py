#!/usr/bin/env python3
"""Compute MLR and V2X end-to-end delay from saved OMNeT result folders.

The script scans OMNeT result directories such as:

  /home/exx/Desktop/vtc2026/omnet_files/OMNeT_results/*/Slot*/stats.csv

For each slot, it reads received packet records, summarizes the E2E delay, and
estimates message loss ratio (MLR) from per-vehicle sequence numbers:

  MLR = 1 - unique_received_sequences / observed_sequence_span

where observed_sequence_span is sum(max_sequence - min_sequence + 1) over all
vehicles seen in that slot. This is a conservative receive-log-based estimate:
it captures sequence gaps among observed vehicles, but cannot count vehicles
that never produced any received packet.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_OMNET_RESULTS = Path("/home/exx/Desktop/vtc2026/omnet_files/OMNeT_results")
DEFAULT_OUTPUT_DIR = Path("/home/exx/Desktop/vtc2026/LibSignal-master/experiments/results")

SNR_PENALTY_BY_GROUP = {
    1: 0,
    2: 5,
    3: 10,
    4: 15,
    5: 20,
    6: 25,
}


def slot_group(slot: int) -> int:
    return ((slot - 1) % 6) + 1


def train_pr_for_slot(slot: int) -> float:
    if 1 <= slot <= 6:
        return 0.1
    if 7 <= slot <= 12:
        return 0.5
    if 13 <= slot <= 18:
        return 1.0
    return float("nan")


def infer_agent(result_name: str) -> str:
    lowered = result_name.lower()
    if "mplight" in lowered:
        return "mplight"
    if "presslight" in lowered or "press" in lowered:
        return "presslight"
    return "unknown"


def infer_run_idx(result_name: str) -> int | None:
    match = re.search(r"run[_-]?(\d+)", result_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def choose_stats_file(slot_dir: Path) -> Path | None:
    stats = slot_dir / "stats.csv"
    if stats.exists():
        return stats

    synced = sorted(
        path
        for path in slot_dir.glob("SimulationStats_Nr_0_Vehicles_RealData_*.csv")
        if "Unsynchronized" not in path.name
    )
    if synced:
        return synced[-1]

    unsynced = sorted(slot_dir.glob("SimulationStats_Nr_0_Vehicles_RealData_Unsynchronized_*.csv"))
    return unsynced[-1] if unsynced else None


def read_slot_stats(path: Path) -> dict[str, Any]:
    delays: list[float] = []
    sequence_counts_by_sender: dict[str, int] = defaultdict(int)
    min_sequence_by_sender: dict[str, int] = {}
    max_sequence_by_sender: dict[str, int] = {}
    sim_time_min = float("inf")
    sim_time_max = float("-inf")

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sender = row.get("SenderID") or row.get("VehicleID")
            seq_raw = row.get("SequenceNumber")
            delay_raw = row.get("Delay") or row.get("E2Edelay")
            sim_time_raw = row.get("SimTime")

            if delay_raw not in (None, ""):
                try:
                    delays.append(float(delay_raw))
                except ValueError:
                    pass

            if sender and seq_raw not in (None, ""):
                try:
                    sender = str(sender)
                    seq = int(float(seq_raw))
                except ValueError:
                    seq = None
                if seq is not None:
                    sequence_counts_by_sender[sender] += 1
                    min_sequence_by_sender[sender] = min(
                        seq,
                        min_sequence_by_sender.get(sender, seq),
                    )
                    max_sequence_by_sender[sender] = max(
                        seq,
                        max_sequence_by_sender.get(sender, seq),
                    )

            if sim_time_raw not in (None, ""):
                try:
                    sim_time = float(sim_time_raw)
                except ValueError:
                    continue
                sim_time_min = min(sim_time_min, sim_time)
                sim_time_max = max(sim_time_max, sim_time)

    received_unique = sum(sequence_counts_by_sender.values())
    expected_observed_span = sum(
        max_sequence_by_sender[sender] - min_sequence_by_sender[sender] + 1
        for sender in sequence_counts_by_sender
    )
    mlr = (
        1.0 - received_unique / expected_observed_span
        if expected_observed_span > 0
        else float("nan")
    )

    return {
        "stats_file": str(path),
        "sender_count": len(sequence_counts_by_sender),
        "received_unique_packets": received_unique,
        "expected_packets_observed_span": expected_observed_span,
        "mlr": mlr,
        "delay_count": len(delays),
        "delay_mean_ms": mean(delays) if delays else float("nan"),
        "delay_median_ms": percentile(delays, 0.50),
        "delay_p95_ms": percentile(delays, 0.95),
        "delay_p99_ms": percentile(delays, 0.99),
        "delay_max_ms": max(delays) if delays else float("nan"),
        "sim_time_min": sim_time_min if math.isfinite(sim_time_min) else float("nan"),
        "sim_time_max": sim_time_max if math.isfinite(sim_time_max) else float("nan"),
    }


def summarize_slot(slot_dir: Path) -> dict[str, Any] | None:
    match = re.fullmatch(r"Slot(\d+)", slot_dir.name)
    if not match:
        return None

    stats_file = choose_stats_file(slot_dir)
    if stats_file is None:
        return None

    slot = int(match.group(1))
    result_dir = slot_dir.parent
    group = slot_group(slot)

    row = {
        "result_set": result_dir.name,
        "agent": infer_agent(result_dir.name),
        "run_idx": infer_run_idx(result_dir.name),
        "slot": slot,
        "slot_group": group,
        "snr_penalty_db": SNR_PENALTY_BY_GROUP[group],
        "train_pr": train_pr_for_slot(slot),
    }
    row.update(read_slot_stats(stats_file))
    return row


def t_critical_95(n: int) -> float:
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
    }
    if n <= 1:
        return 0.0
    return table.get(n, 1.96)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "mlr",
        "delay_mean_ms",
        "delay_median_ms",
        "delay_p95_ms",
        "delay_p99_ms",
        "delay_max_ms",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("agent", "train_pr", "snr_penalty_db", "slot_group")

    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    out_rows: list[dict[str, Any]] = []
    for key_values, group_rows in sorted(grouped.items()):
        out: dict[str, Any] = dict(zip(keys, key_values))
        out["n_slots"] = len(group_rows)
        out["result_sets"] = ",".join(sorted({str(row["result_set"]) for row in group_rows}))

        for metric in metrics:
            values = [
                float(row[metric])
                for row in group_rows
                if row.get(metric) not in (None, "") and math.isfinite(float(row[metric]))
            ]
            if not values:
                continue
            metric_mean = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0
            half_width = (
                t_critical_95(len(values)) * metric_std / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            out[f"{metric}_mean"] = metric_mean
            out[f"{metric}_std"] = metric_std
            out[f"{metric}_ci95_low"] = metric_mean - half_width
            out[f"{metric}_ci95_high"] = metric_mean + half_width
        out_rows.append(out)

    return out_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"No rows to write for {path}")

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
        description="Compute OMNeT message loss ratio and E2E delay summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--omnet_results_dir",
        type=Path,
        default=DEFAULT_OMNET_RESULTS,
        help="Root containing copied OMNeT result folders with Slot*/stats.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV summaries.",
    )
    parser.add_argument(
        "--slot_output",
        default="omnet_mlr_delay_slot_summary.csv",
        help="Per-slot output CSV filename.",
    )
    parser.add_argument(
        "--aggregated_output",
        default="omnet_mlr_delay_aggregated.csv",
        help="Aggregated output CSV filename.",
    )
    parser.add_argument(
        "--result_glob",
        default="*",
        help=(
            "Only process result folders matching this glob, e.g. "
            "'PressLight-Run5-*' or 'MPLight-*'."
        ),
    )
    parser.add_argument(
        "--slots",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of slot numbers to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = sorted(path for path in args.omnet_results_dir.glob(args.result_glob) if path.is_dir())
    slot_dirs = sorted(slot_dir for result_dir in result_dirs for slot_dir in result_dir.glob("Slot*"))
    if args.slots is not None:
        wanted = {f"Slot{slot}" for slot in args.slots}
        slot_dirs = [slot_dir for slot_dir in slot_dirs if slot_dir.name in wanted]

    rows = []
    for idx, slot_dir in enumerate(slot_dirs, start=1):
        print(f"[{idx}/{len(slot_dirs)}] Processing {slot_dir}", flush=True)
        row = summarize_slot(slot_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No OMNeT stats found under {args.omnet_results_dir}")

    aggregated = aggregate_rows(rows)
    slot_path = args.output_dir / args.slot_output
    aggregated_path = args.output_dir / args.aggregated_output

    write_csv(slot_path, rows)
    write_csv(aggregated_path, aggregated)

    print(f"Processed {len(rows)} slot result files.")
    print(f"Wrote per-slot summary: {slot_path}")
    print(f"Wrote aggregated summary: {aggregated_path}")
    print(
        "Note: MLR is estimated from gaps in received per-vehicle sequence numbers; "
        "vehicles with no received packets cannot be counted from receive logs alone."
    )


if __name__ == "__main__":
    main()
