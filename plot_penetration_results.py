"""Plot baseline vs penetration-aware inference results.

The script reads CSVs produced by ``experiments/run_inference_sweep.py``:

* baseline: ``inference_{agent}_{network}_from_pr_1.00_*[_aggregated].csv``
* matched:  ``inference_matched_{agent}_{network}_*[_aggregated].csv``
* live OMNeT: ``inference_{agent}_{network}_from_pr_{train_pr}_omnet_{slot:02d}_*[_aggregated].csv``
  (one CSV per slot; by default all slots are combined regardless of train_pr in the filename)

For each metric, the line is the mean and the shaded band is the 95% confidence
interval when available. If only raw per-run CSVs exist, the script computes the
mean/std/95% CI before plotting.

Live OMNeT runs produce one CSV per slot; this script combines them and plots
travel time, throughput, and queue versus OMNeT slot number.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


C_BASELINE = "#4C72B0"
C_MATCHED = "#DD8452"
C_IMPROV = "#55A868"
MIX_THROUGHPUT_BASELINE = "#A6CEE3"
MIX_THROUGHPUT_MATCHED = "#1F78B4"
MIX_TRAVEL_TIME_BASELINE = "#FB9A99"
MIX_TRAVEL_TIME_MATCHED = "#E31A1C"
ALPHA_CI = 0.18
LW = 2.4
MS = 6
TITLE_SIZE = 24
LABEL_SIZE = 21
TICK_SIZE = 18
LEGEND_SIZE = 17
METRICS = ("travel_time", "throughput", "queue")
METRIC_SPECS = [
    ("travel_time", "Travel Time (s)"),
    ("throughput", "Throughput (vehicles)"),
    ("queue", "Queue (vehicles)"),
]
SENSITIVITY_STYLES = {
    "PressLight": ("#1f77b4", "o", ":"),
    "PR-PressLight": ("#ff7f0e", "x", "-"),
    "MPLight": ("#2ca02c", "^", ":"),
    "PR-MPLight": ("#d62728", "s", "-"),
}
OMNET_RATE_STYLES = {
    0.10: ("#1f77b4", "o"),
    0.50: ("#ff7f0e", "s"),
    1.00: ("#2ca02c", "^"),
}
OMNET_GROUP_SNR_PENALTY = {
    1: 0,
    2: 5,
    3: 10,
    4: 15,
    5: 20,
    6: 25,
}
OMNET_SLOT_RE = re.compile(r"_omnet_(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PressLight/MPLight baseline vs PR-aware inference results "
            "with mean lines and shaded 95% confidence intervals."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default="experiments/results")
    parser.add_argument("--out_dir", default="plots_penetration")
    parser.add_argument("--network", default="sumohz4x4")
    parser.add_argument("--baseline_train_pr", type=float, default=1.0)
    parser.add_argument(
        "--agents", nargs="+", default=["presslight", "mplight"],
        choices=["presslight", "mplight"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--skip_omnet", action="store_true",
        help="Do not plot live OMNeT slot sweeps",
    )
    parser.add_argument(
        "--omnet_infer_pr", type=float, default=None,
        help="Filter OMNeT slot plots to this inference penetration rate (default: all)",
    )
    parser.add_argument(
        "--omnet_train_pr", type=float, default=None,
        help=(
            "Only include live OMNeT CSVs whose filename matches this training PR "
            "(e.g. 1.0 → slots trained with from_pr_1.00). Default: include every "
            "omnet_* slot CSV regardless of training PR."
        ),
    )
    parser.add_argument(
        "--skip_omnet_repeat_dirs",
        action="store_true",
        help="Do not also plot individual experiments/omnet_*_* repeat folders.",
    )
    return parser.parse_args()


def has_usable_metrics(path: Path) -> bool:
    try:
        df = pd.read_csv(path, nrows=50)
    except Exception:
        return False

    if is_aggregated(df):
        return any(
            f"{metric}_mean" in df.columns
            and pd.to_numeric(df[f"{metric}_mean"], errors="coerce").notna().any()
            for metric in METRICS
        )

    if "status" in df.columns:
        df = df[df["status"].eq("success")]
    return any(
        metric in df.columns
        and pd.to_numeric(df[metric], errors="coerce").notna().any()
        for metric in METRICS
    )


def latest_usable_csv(results_dir: Path, pattern: str, *, exclude_omnet: bool = False) -> Path | None:
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in matches:
        if exclude_omnet and parse_omnet_slot_from_path(path) is not None:
            continue
        if has_usable_metrics(path):
            return path
    return None


def find_result_files(
    results_dir: Path,
    agent: str,
    network: str,
    baseline_train_pr: float,
) -> tuple[Path | None, Path | None]:
    baseline_tag = f"{baseline_train_pr:.2f}"
    baseline = latest_usable_csv(
        results_dir,
        f"inference_{agent}_{network}_from_pr_{baseline_tag}_*_aggregated.csv",
        exclude_omnet=True,
    )
    matched = latest_usable_csv(
        results_dir,
        f"inference_matched_{agent}_{network}_*_aggregated.csv",
        exclude_omnet=True,
    )

    # Fall back to raw CSVs when aggregated files are not present yet.
    if baseline is None:
        baseline = latest_usable_csv(
            results_dir,
            f"inference_{agent}_{network}_from_pr_{baseline_tag}_*.csv",
            exclude_omnet=True,
        )
    if matched is None:
        matched = latest_usable_csv(
            results_dir,
            f"inference_matched_{agent}_{network}_*.csv",
            exclude_omnet=True,
        )
    return baseline, matched


def infer_train_pr_from_filename(path: Path, agent: str, network: str) -> float | None:
    prefix = f"inference_{agent}_{network}_from_pr_"
    if not path.name.startswith(prefix):
        return None
    remainder = path.name[len(prefix):]
    try:
        return float(remainder.split("_", 1)[0])
    except (IndexError, ValueError):
        return None


def is_aggregated(df: pd.DataFrame) -> bool:
    return any(f"{metric}_mean" in df.columns for metric in METRICS)


def aggregate_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "status" in df.columns:
        df = df[df["status"].eq("success")]
    if "infer_pr" not in df.columns:
        raise ValueError("CSV must contain an infer_pr column")

    rows = []
    for infer_pr, group in df.groupby("infer_pr", sort=True):
        row = {"infer_pr": infer_pr}
        if "train_pr" in group.columns:
            row["train_pr"] = group["train_pr"].iloc[0]
        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            if values.empty:
                continue
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            half_ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_ci
            row[f"{metric}_ci95_high"] = mean + half_ci
        rows.append(row)
    return pd.DataFrame(rows).sort_values("infer_pr")


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not is_aggregated(df):
        df = aggregate_raw(df)
    return df.sort_values("infer_pr")


def load_legacy_matched_summary(results_dir: Path, agent: str, network: str) -> pd.DataFrame | None:
    """Build PR-aware data from older one-train-pr-per-file inference CSVs."""
    candidates: dict[float, Path] = {}
    patterns = [
        f"inference_{agent}_{network}_from_pr_*_aggregated.csv",
        f"inference_{agent}_{network}_from_pr_*.csv",
    ]
    for pattern in patterns:
        for path in sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime):
            train_pr = infer_train_pr_from_filename(path, agent, network)
            if train_pr is None or not has_usable_metrics(path):
                continue
            candidates[train_pr] = path

    matched_rows = []
    for train_pr, path in sorted(candidates.items()):
        df = load_summary(path)
        matched = df[np.isclose(pd.to_numeric(df["infer_pr"], errors="coerce"), train_pr)]
        if matched.empty:
            continue
        row = matched.iloc[-1].copy()
        row["train_pr"] = train_pr
        matched_rows.append(row)

    if not matched_rows:
        return None
    return pd.DataFrame(matched_rows).sort_values("infer_pr").reset_index(drop=True)


def metric_series(
    df: pd.DataFrame,
    metric: str,
    x_col: str = "infer_pr",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[f"{metric}_mean"], errors="coerce").to_numpy(dtype=float)

    low_col = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"
    if low_col in df.columns and high_col in df.columns:
        low = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=float)
    else:
        low = y.copy()
        high = y.copy()
    return x, y, low, high


def parse_omnet_slot_from_path(path: Path) -> int | None:
    match = OMNET_SLOT_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def omnet_glob_patterns(agent: str, network: str, train_pr: float | None) -> list[str]:
    if train_pr is None:
        return [
            f"inference_{agent}_{network}_from_pr_*_omnet_*_aggregated.csv",
            f"inference_{agent}_{network}_from_pr_*_omnet_*.csv",
            f"inference_matched_{agent}_{network}_omnet_*_aggregated.csv",
            f"inference_matched_{agent}_{network}_omnet_*.csv",
        ]
    tag = f"{train_pr:.2f}"
    return [
        f"inference_{agent}_{network}_from_pr_{tag}_omnet_*_aggregated.csv",
        f"inference_{agent}_{network}_from_pr_{tag}_omnet_*.csv",
        f"inference_matched_{agent}_{network}_omnet_*_aggregated.csv",
        f"inference_matched_{agent}_{network}_omnet_*.csv",
    ]


def load_combined_omnet_results(
    results_dir: Path,
    agent: str,
    network: str,
    train_pr: float | None = None,
    infer_pr: float | None = None,
) -> pd.DataFrame | None:
    """Load one combined OMNeT CSV with all slots, if present."""
    patterns = [
        f"omnet_combined_{agent}_{network}*_aggregated.csv",
        f"omnet_combined_{agent}_{network}*.csv",
    ]
    for pattern in patterns:
        for path in sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            if not has_usable_metrics(path):
                continue
            df = load_summary(path)
            if "omnet_slot" not in df.columns:
                continue
            if train_pr is not None and "train_pr" in df.columns:
                df = df[
                    np.isclose(
                        pd.to_numeric(df["train_pr"], errors="coerce"),
                        train_pr,
                    )
                ]
            if infer_pr is not None and "infer_pr" in df.columns:
                df = df[
                    np.isclose(
                        pd.to_numeric(df["infer_pr"], errors="coerce"),
                        infer_pr,
                    )
                ]
            if not df.empty:
                return df.sort_values("omnet_slot").reset_index(drop=True)
    return None


def load_omnet_slot_results(
    results_dir: Path,
    agent: str,
    network: str,
    train_pr: float | None = None,
    infer_pr: float | None = None,
    use_combined: bool = True,
) -> pd.DataFrame | None:
    """
    Combine per-slot live OMNeT inference CSVs into one table keyed by omnet_slot.

    Each slot normally has its own sweep CSV (filename contains ``_omnet_{slot:02d}_``).
    When ``train_pr`` is None, every slot with an ``_omnet_{NN}_`` CSV is included
    (e.g. slots 1–6 @ train_pr=0.1, 7–12 @ 0.5, 13–18 @ 1.0 in one plot).
    """
    if use_combined:
        combined = load_combined_omnet_results(
            results_dir,
            agent,
            network,
            train_pr=train_pr,
            infer_pr=infer_pr,
        )
        if combined is not None:
            return combined

    chosen: dict[int, Path] = {}
    for pattern in omnet_glob_patterns(agent, network, train_pr):
        for path in sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            slot = parse_omnet_slot_from_path(path)
            if slot is None or not has_usable_metrics(path):
                continue
            existing = chosen.get(slot)
            if existing is None:
                chosen[slot] = path
                continue
            if existing.name.endswith("_aggregated.csv"):
                continue
            if path.name.endswith("_aggregated.csv"):
                chosen[slot] = path

    if not chosen:
        return None

    rows: list[dict[str, float | int]] = []
    for slot in sorted(chosen):
        summary = load_summary(chosen[slot])
        if infer_pr is not None:
            summary = summary[
                np.isclose(
                    pd.to_numeric(summary["infer_pr"], errors="coerce"),
                    infer_pr,
                )
            ]
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row["omnet_slot"] = slot
        if "infer_pr" in summary.columns:
            row["infer_pr"] = float(summary["infer_pr"].iloc[0])
        file_train_pr = infer_train_pr_from_filename(chosen[slot], agent, network)
        if file_train_pr is not None:
            row["train_pr"] = file_train_pr
        elif "train_pr" in summary.columns:
            row["train_pr"] = float(summary["train_pr"].iloc[0])
        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("omnet_slot").reset_index(drop=True)


def discover_omnet_repeat_dirs(results_dir: Path) -> list[Path]:
    """Return sibling folders like experiments/omnet_1_press for per-run plots."""
    parent = results_dir.parent
    return sorted(
        path for path in parent.glob("omnet_*_*")
        if path.is_dir()
    )


def style_ax(ax, ylabel: str, title: str) -> None:
    ax.set_xlabel("Penetration Rate", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def style_sensitivity_ax(ax, title: str) -> None:
    ax.set_xlabel("Penetration Rate", fontsize=LABEL_SIZE)
    ax.set_ylabel("Normalized Sensitivity", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.grid(True, linestyle="-", alpha=0.45, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def plot_line_with_ci(
    ax,
    df: pd.DataFrame,
    metric: str,
    label: str,
    color: str,
    marker: str,
    x_col: str = "infer_pr",
    linestyle: str = "-",
) -> None:
    x, y, low, high = metric_series(df, metric, x_col=x_col)
    ax.plot(
        x,
        y,
        color=color,
        lw=LW,
        marker=marker,
        ms=MS,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )
    ax.fill_between(x, low, high, color=color, alpha=ALPHA_CI, linewidth=0, zorder=2)


def style_omnet_ax(ax, ylabel: str, title: str) -> None:
    ax.set_xlabel("SNR Penalty (dB)", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ticks = list(OMNET_GROUP_SNR_PENALTY.values())
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{snr_penalty}" for snr_penalty in ticks],
        fontsize=max(TICK_SIZE - 2, 1),
    )


def sensitivity_series(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return d(normalized metric) / d(penetration rate)."""
    x, y, _, _ = metric_series(df, metric)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return x, np.zeros_like(x)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    span = np.nanmax(y) - np.nanmin(y)
    if span == 0:
        y_norm = np.zeros_like(y)
    else:
        y_norm = (y - np.nanmin(y)) / span

    return x, np.gradient(y_norm, x)


def plot_sensitivity_metric(
    data: dict[str, dict[str, pd.DataFrame]],
    metric: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for agent_label, frames in data.items():
        for key, prefix in (("baseline", ""), ("matched", "PR-")):
            label = f"{prefix}{agent_label}"
            color, marker, linestyle = SENSITIVITY_STYLES.get(
                label, ("#333333", "o", "-")
            )
            alpha = 0.45 if key == "baseline" else 1.0
            linewidth = LW * 0.9 if key == "baseline" else LW
            x, sensitivity = sensitivity_series(frames[key], metric)
            ax.plot(
                x,
                sensitivity,
                color=color,
                marker=marker,
                linestyle=linestyle,
                lw=linewidth,
                ms=MS,
                alpha=alpha,
                label=label,
            )

    style_sensitivity_ax(ax, title)
    ax.axhline(0, color="0.25", lw=0.9, alpha=0.8)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_metric(
    data: dict[str, dict[str, pd.DataFrame]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for agent_label, frames in data.items():
        for key, prefix in (("baseline", ""), ("matched", "PR-")):
            label = f"{prefix}{agent_label}"
            color, marker, _ = SENSITIVITY_STYLES.get(
                label, ("#333333", "o", "-")
            )
            plot_line_with_ci(ax, frames[key], metric, label, color, marker)

    style_ax(ax, ylabel, title)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined(data: dict[str, dict[str, pd.DataFrame]], output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(data), 3, figsize=(18, 5 * len(data)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    #fig.suptitle("Baseline vs. Penetration-Aware TSC Models", fontsize=15, fontweight="bold", y=1.01)

    for row_idx, (agent_label, frames) in enumerate(data.items()):
        for col_idx, (metric, ylabel) in enumerate(METRIC_SPECS):
            ax = axes[row_idx, col_idx]
            plot_line_with_ci(ax, frames["baseline"], metric, agent_label, C_BASELINE, "o")
            plot_line_with_ci(ax, frames["matched"], metric, f"PR-{agent_label}", C_MATCHED, "s")
            style_ax(ax, ylabel, f"{agent_label} - {ylabel}")
            ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def safe_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def plot_overview_panels(data: dict[str, dict[str, pd.DataFrame]], out_dir: Path, dpi: int) -> None:
    """Save each panel from combined_overview.png as its own figure."""
    for agent_label, frames in data.items():
        for metric, ylabel in METRIC_SPECS:
            fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
            plot_line_with_ci(ax, frames["baseline"], metric, agent_label, C_BASELINE, "o")
            plot_line_with_ci(ax, frames["matched"], metric, f"PR-{agent_label}", C_MATCHED, "s")
            style_ax(ax, ylabel, f"{agent_label} - {ylabel}")
            ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

            output_path = out_dir / f"overview_{safe_name(agent_label)}_{metric}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {output_path}")


def plot_mixed_throughput_travel_time(
    data: dict[str, dict[str, pd.DataFrame]],
    out_dir: Path,
    dpi: int,
) -> None:
    """Save one dual-axis throughput/travel-time figure for each agent."""
    mix_dir = out_dir / "mix plot"
    mix_dir.mkdir(parents=True, exist_ok=True)

    for agent_label, frames in data.items():
        fig, ax_throughput = plt.subplots(figsize=(11, 7), constrained_layout=True)
        ax_travel_time = ax_throughput.twinx()

        plot_line_with_ci(
            ax_throughput,
            frames["baseline"],
            "throughput",
            f"{agent_label} (v)",
            MIX_THROUGHPUT_BASELINE,
            "o",
            linestyle="-",
        )
        plot_line_with_ci(
            ax_throughput,
            frames["matched"],
            "throughput",
            f"PR-{agent_label} (v)",
            MIX_THROUGHPUT_MATCHED,
            "s",
            linestyle="-",
        )
        plot_line_with_ci(
            ax_travel_time,
            frames["baseline"],
            "travel_time",
            f"{agent_label} (s)",
            MIX_TRAVEL_TIME_BASELINE,
            "^",
            linestyle="--",
        )
        plot_line_with_ci(
            ax_travel_time,
            frames["matched"],
            "travel_time",
            f"PR-{agent_label} (s)",
            MIX_TRAVEL_TIME_MATCHED,
            "D",
            linestyle="--",
        )

        ax_throughput.set_xlabel("Penetration Rate", fontsize=LABEL_SIZE)
        ax_throughput.set_ylabel("Throughput (vehicles)", fontsize=LABEL_SIZE)
        ax_travel_time.set_ylabel("Travel Time (s)", fontsize=LABEL_SIZE)
        ax_throughput.set_title(
            agent_label,
            fontsize=TITLE_SIZE,
            fontweight="bold",
            pad=10,
        )
        ax_throughput.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
        ax_throughput.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax_throughput.tick_params(labelsize=TICK_SIZE)
        ax_travel_time.tick_params(labelsize=TICK_SIZE)

        handles_left, labels_left = ax_throughput.get_legend_handles_labels()
        handles_right, labels_right = ax_travel_time.get_legend_handles_labels()
        ax_throughput.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            fontsize=LEGEND_SIZE,
            framealpha=0.9,
        )

        output_path = mix_dir / f"{safe_name(agent_label)}_throughput_travel_time.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


def plot_omnet_metric(
    data: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for agent_label, df in data.items():
        grouped = add_omnet_group_columns(df)
        color, marker, _ = SENSITIVITY_STYLES.get(agent_label, ("#333333", "o", "-"))
        plot_line_with_ci(
            ax, grouped, metric, agent_label, color, marker, x_col="snr_penalty_db",
        )

    style_omnet_ax(ax, ylabel, title)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def omnet_rate_label(rate: float) -> str:
    return f"{rate:.1f}"


def add_omnet_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.copy()
    grouped["omnet_group_run"] = (
        (pd.to_numeric(grouped["omnet_slot"], errors="coerce").astype(int) - 1) % 6
    ) + 1
    grouped["snr_penalty_db"] = grouped["omnet_group_run"].map(OMNET_GROUP_SNR_PENALTY)
    if "train_pr" in grouped.columns:
        grouped["omnet_rate"] = pd.to_numeric(grouped["train_pr"], errors="coerce")
    elif "infer_pr" in grouped.columns:
        grouped["omnet_rate"] = pd.to_numeric(grouped["infer_pr"], errors="coerce")
    else:
        grouped["omnet_rate"] = np.nan
    return grouped.sort_values(["omnet_rate", "snr_penalty_db"]).reset_index(drop=True)


def draw_omnet_grouped_metric(
    ax,
    data: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for agent_label, df in data.items():
        grouped = add_omnet_group_columns(df)
        rates = sorted(rate for rate in grouped["omnet_rate"].dropna().unique())
        for rate in rates:
            rate_df = grouped[np.isclose(grouped["omnet_rate"], rate)]
            color, marker = OMNET_RATE_STYLES.get(round(float(rate), 2), ("#333333", "o"))
            label = omnet_rate_label(float(rate))
            if len(data) > 1:
                label = f"{agent_label} {label}"
            plot_line_with_ci(
                ax,
                rate_df,
                metric,
                label,
                color,
                marker,
                x_col="snr_penalty_db",
            )
    style_omnet_ax(ax, ylabel, title)
    ax.legend(
        title="Training PR",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
        framealpha=0.9,
    )


def plot_omnet_grouped_metric(
    data: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    draw_omnet_grouped_metric(ax, data, metric, ylabel, title)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_omnet_grouped_combined(data: dict[str, pd.DataFrame], output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    for ax, (metric, ylabel) in zip(axes, METRIC_SPECS):
        draw_omnet_grouped_metric(ax, data, metric, ylabel, f"{ylabel} vs. SNR Penalty")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_omnet_grouped_set(
    data: dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int,
    *,
    prefix: str,
    title_suffix: str = "",
) -> None:
    """Save separate and combined grouped OMNeT figures for one data set."""
    plot_omnet_grouped_metric(
        data,
        "travel_time",
        "Travel Time (s)",
        f"Travel Time vs. SNR Penalty{title_suffix}",
        out_dir / f"{prefix}_travel_time.png",
        dpi,
    )
    plot_omnet_grouped_metric(
        data,
        "throughput",
        "Throughput (vehicles)",
        f"Throughput vs. SNR Penalty{title_suffix}",
        out_dir / f"{prefix}_throughput.png",
        dpi,
    )
    plot_omnet_grouped_metric(
        data,
        "queue",
        "Queue (vehicles)",
        f"Queue vs. SNR Penalty{title_suffix}",
        out_dir / f"{prefix}_queue.png",
        dpi,
    )
    plot_omnet_grouped_combined(
        data,
        out_dir / f"{prefix}_by_rate.png",
        dpi,
    )


def plot_omnet_combined(data: dict[str, pd.DataFrame], output_path: Path, dpi: int) -> None:
    n_agents = len(data)
    fig, axes = plt.subplots(n_agents, 3, figsize=(18, 5 * n_agents), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row_idx, (agent_label, df) in enumerate(data.items()):
        grouped = add_omnet_group_columns(df)
        color, marker, _ = SENSITIVITY_STYLES.get(agent_label, ("#333333", "o", "-"))
        for col_idx, (metric, ylabel) in enumerate(METRIC_SPECS):
            ax = axes[row_idx, col_idx]
            plot_line_with_ci(
                ax, grouped, metric, agent_label, color, marker, x_col="snr_penalty_db",
            )
            style_omnet_ax(ax, ylabel, f"{agent_label} - {ylabel} vs. SNR Penalty")
            ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_improvement(data: dict[str, dict[str, pd.DataFrame]], output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(data), 3, figsize=(18, 5 * len(data)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(
        "Penetration-Aware Improvement over Baseline",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=1.01,
    )

    for row_idx, (agent_label, frames) in enumerate(data.items()):
        merged = pd.merge(
            frames["baseline"],
            frames["matched"],
            on="infer_pr",
            suffixes=("_baseline", "_matched"),
        )

        specs = [
            ("travel_time", "Travel Time Reduction (s)", "lower_better"),
            ("throughput", "Throughput Gain (vehicles)", "higher_better"),
            ("queue", "Queue Reduction (vehicles)", "lower_better"),
        ]
        for col_idx, (metric, ylabel, direction) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            base = merged[f"{metric}_mean_baseline"]
            matched = merged[f"{metric}_mean_matched"]
            if direction == "lower_better":
                improvement = base - matched
            else:
                improvement = matched - base

            ax.bar(merged["infer_pr"], improvement, width=0.018, color=C_IMPROV, alpha=0.7)
            ax.axhline(0, color="0.35", lw=0.8, linestyle="--")
            style_ax(ax, ylabel, f"{agent_label} - {ylabel}")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_labels = {
        "presslight": "PressLight",
        "mplight": "MPLight",
    }

    data: dict[str, dict[str, pd.DataFrame]] = {}
    for agent in args.agents:
        baseline_path, matched_path = find_result_files(
            results_dir, agent, args.network, args.baseline_train_pr
        )
        if baseline_path is None:
            print(f"Skipping {agent}: could not find a baseline CSV.")
            print(f"  baseline: {baseline_path}")
            continue

        print(f"{agent_labels[agent]} baseline: {baseline_path}")
        baseline_df = load_summary(baseline_path)
        if matched_path is not None:
            print(f"{agent_labels[agent]} PR-aware: {matched_path}")
            matched_df = load_summary(matched_path)
        else:
            print(f"{agent_labels[agent]} PR-aware: assembling from older per-rate inference CSVs")
            matched_df = load_legacy_matched_summary(results_dir, agent, args.network)
            if matched_df is None:
                print(f"Skipping {agent}: could not find PR-matched CSVs.")
                continue

        data[agent_labels[agent]] = {
            "baseline": baseline_df,
            "matched": matched_df,
        }

    if data:
        plot_metric(
            data,
            "travel_time",
            "Travel Time (s)",
            "Travel Time vs. Penetration Rate",
            out_dir / "travel_time_comparison.png",
            args.dpi,
        )
        plot_metric(
            data,
            "throughput",
            "Throughput (vehicles)",
            "Throughput vs. Penetration Rate",
            out_dir / "throughput_comparison.png",
            args.dpi,
        )
        plot_metric(
            data,
            "queue",
            "Queue (vehicles)",
            "Queue vs. Penetration Rate",
            out_dir / "queue_comparison.png",
            args.dpi,
        )
        plot_combined(data, out_dir / "combined_overview.png", args.dpi)
        plot_overview_panels(data, out_dir, args.dpi)
        plot_mixed_throughput_travel_time(data, out_dir, args.dpi)
        plot_improvement(data, out_dir / "improvement.png", args.dpi)
        plot_sensitivity_metric(
            data,
            "travel_time",
            "Sensitivity of Travel Time to Penetration Rate",
            out_dir / "sensitivity_travel_time_comparison.png",
            args.dpi,
        )
        plot_sensitivity_metric(
            data,
            "throughput",
            "Sensitivity of Throughput to Penetration Rate",
            out_dir / "sensitivity_throughput_comparison.png",
            args.dpi,
        )
        plot_sensitivity_metric(
            data,
            "queue",
            "Sensitivity of Queue to Penetration Rate",
            out_dir / "sensitivity_queue_comparison.png",
            args.dpi,
        )
    else:
        print("No complete baseline + PR-matched result pairs found.")
        if args.skip_omnet:
            raise SystemExit("No plots to generate because --skip_omnet is set.")
        print("Continuing with live OMNeT slot plots only.")

    if not args.skip_omnet:
        omnet_data: dict[str, pd.DataFrame] = {}
        for agent in args.agents:
            omnet_df = load_omnet_slot_results(
                results_dir,
                agent,
                args.network,
                train_pr=args.omnet_train_pr,
                infer_pr=args.omnet_infer_pr,
            )
            if omnet_df is None or omnet_df.empty:
                print(f"No live OMNeT slot CSVs found for {agent}.")
                continue
            label = agent_labels[agent]
            omnet_data[label] = omnet_df
            slots = ", ".join(str(int(s)) for s in omnet_df["omnet_slot"])
            print(f"{label} OMNeT slots ({len(omnet_df)}): {slots}")

        if omnet_data:
            infer_suffix = (
                f" @ infer_pr={args.omnet_infer_pr:.2f}"
                if args.omnet_infer_pr is not None
                else ""
            )
            print(
                f"Plotting grouped live OMNeT runs{infer_suffix} separately by agent: "
                "x=SNR Penalty, lines=training PR."
            )
            for agent_label, agent_df in omnet_data.items():
                plot_omnet_grouped_set(
                    {agent_label: agent_df},
                    out_dir,
                    args.dpi,
                    prefix=f"omnet_grouped_{safe_name(agent_label)}",
                    title_suffix=f"{infer_suffix} ({agent_label})",
                )

        if not args.skip_omnet_repeat_dirs:
            repeat_dirs = discover_omnet_repeat_dirs(results_dir)
            if repeat_dirs:
                print(
                    "Plotting individual OMNeT repeat folders: "
                    + ", ".join(path.name for path in repeat_dirs)
                )
            for repeat_dir in repeat_dirs:
                repeat_data: dict[str, pd.DataFrame] = {}
                for agent in args.agents:
                    repeat_df = load_omnet_slot_results(
                        repeat_dir,
                        agent,
                        args.network,
                        train_pr=args.omnet_train_pr,
                        infer_pr=args.omnet_infer_pr,
                        use_combined=False,
                    )
                    if repeat_df is None or repeat_df.empty:
                        continue
                    repeat_data[agent_labels[agent]] = repeat_df

                if not repeat_data:
                    print(f"No live OMNeT slot CSVs found in {repeat_dir}.")
                    continue

                repeat_prefix = safe_name(repeat_dir.name) + "_grouped"
                plot_omnet_grouped_set(
                    repeat_data,
                    out_dir,
                    args.dpi,
                    prefix=repeat_prefix,
                    title_suffix=f" ({repeat_dir.name})",
                )

    print("\nAll plots saved successfully.")


if __name__ == "__main__":
    main()
