#!/usr/bin/env python3
"""Plot runtime profiling results for deployment discussion.

The input is the CSV written by ``experiments/summarize_runtime_profiles.py``.
The figures focus on SUMO-only controller profiling: real-time factor, DRL
action-selection latency, and approximate per-decision wall-clock breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SUMMARY = Path("experiments/results/runtime_profile_summary.csv")
DEFAULT_OUT_DIR = Path("plots_penetration/runtime_analysis")
CONTROL_INTERVAL_MS = 30_000.0
COLORS = {
    "presslight": "#4C72B0",
    "mplight": "#DD8452",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-ready runtime profiling plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["use_omnet"].astype(str).str.lower().eq("false")].copy()
    if df.empty:
        raise SystemExit(f"No OMNeT-disabled runtime rows found in {path}")

    df["agent_label"] = df["model"].map(
        {
            "presslight": "PressLight",
            "mplight": "MPLight",
        }
    ).fillna(df["model"].astype(str))
    df = df.sort_values("agent_label")
    return df


def _bar_colors(df: pd.DataFrame) -> list[str]:
    return [COLORS.get(str(agent).lower(), "#55A868") for agent in df["model"]]


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def annotate_bars(ax: plt.Axes, bars, values: pd.Series, fmt: str) -> None:
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def save_single_runtime_factor(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bars = ax.bar(
        df["agent_label"],
        df["real_time_factor_mean"],
        yerr=df["real_time_factor_std"],
        capsize=4,
        color=_bar_colors(df),
        edgecolor="black",
        linewidth=0.7,
    )
    annotate_bars(ax, bars, df["real_time_factor_mean"], "{:.1f}x")
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(0.02, 1.03, "Real-time threshold (1x)", transform=ax.get_yaxis_transform(), fontsize=10)
    ax.set_ylabel("Real-Time Factor (x)")
    ax.set_title("SUMO-Only Runtime Speed")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_real_time_factor.png", dpi=dpi)
    plt.close(fig)


def save_single_action_latency(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bars = ax.bar(
        df["agent_label"],
        df["action_select_mean_ms_mean"],
        yerr=df["action_select_mean_ms_std"],
        capsize=4,
        color=_bar_colors(df),
        edgecolor="black",
        linewidth=0.7,
    )
    annotate_bars(ax, bars, df["action_select_mean_ms_mean"], "{:.2f} ms")
    ax.set_ylabel("DRL Action Selection (ms)")
    ax.set_title("Controller Inference Latency")
    ax.text(
        0.5,
        0.95,
        "Control interval = 30,000 ms",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.25"},
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_action_latency.png", dpi=dpi)
    plt.close(fig)


def save_single_decision_breakdown(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    labels = df["agent_label"].to_numpy()
    x = np.arange(len(labels))
    action_ms = df["action_select_mean_ms_mean"].to_numpy()
    decision_ms = df["decision_wall_mean_s_mean"].to_numpy() * 1000.0
    other_ms = np.maximum(decision_ms - action_ms, 0.0)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar(x, action_ms, label="DRL action selection", color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.bar(x, other_ms, bottom=action_ms, label="SUMO step + logging overhead", color="#C7C7C7", edgecolor="black", linewidth=0.6)
    for xpos, total in zip(x, decision_ms):
        ax.annotate(f"{total:.1f} ms", xy=(xpos, total), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Wall Time per Decision (ms)")
    ax.set_title("Per-Decision Runtime Breakdown")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_decision_breakdown.png", dpi=dpi)
    plt.close(fig)


def save_combined_figure(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    colors = _bar_colors(df)

    bars = axes[0].bar(df["agent_label"], df["real_time_factor_mean"], yerr=df["real_time_factor_std"], capsize=4, color=colors, edgecolor="black", linewidth=0.7)
    annotate_bars(axes[0], bars, df["real_time_factor_mean"], "{:.1f}x")
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1.1)
    axes[0].set_title("Runtime Speed")
    axes[0].set_ylabel("Real-Time Factor (x)")

    bars = axes[1].bar(df["agent_label"], df["action_select_mean_ms_mean"], yerr=df["action_select_mean_ms_std"], capsize=4, color=colors, edgecolor="black", linewidth=0.7)
    annotate_bars(axes[1], bars, df["action_select_mean_ms_mean"], "{:.2f} ms")
    axes[1].set_title("DRL Inference")
    axes[1].set_ylabel("Action Selection (ms)")
    axes[1].text(0.5, 0.95, "30 s control interval", transform=axes[1].transAxes, ha="center", va="top", fontsize=10)

    labels = df["agent_label"].to_numpy()
    x = np.arange(len(labels))
    action_ms = df["action_select_mean_ms_mean"].to_numpy()
    decision_ms = df["decision_wall_mean_s_mean"].to_numpy() * 1000.0
    other_ms = np.maximum(decision_ms - action_ms, 0.0)
    axes[2].bar(x, action_ms, label="DRL action", color="#4C72B0", edgecolor="black", linewidth=0.6)
    axes[2].bar(x, other_ms, bottom=action_ms, label="Other loop time", color="#C7C7C7", edgecolor="black", linewidth=0.6)
    for xpos, total in zip(x, decision_ms):
        axes[2].annotate(f"{total:.1f} ms", xy=(xpos, total), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Decision Loop")
    axes[2].set_ylabel("Wall Time per Decision (ms)")
    axes[2].legend(frameon=False, fontsize=9, loc="upper left")

    for ax in axes:
        style_axes(ax)

    fig.suptitle("SUMO-Only Runtime Profiling", fontsize=15, y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_profile_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.summary)

    save_single_runtime_factor(df, args.out_dir, args.dpi)
    save_single_action_latency(df, args.out_dir, args.dpi)
    save_single_decision_breakdown(df, args.out_dir, args.dpi)
    save_combined_figure(df, args.out_dir, args.dpi)

    print(f"Wrote runtime plots to: {args.out_dir}")
    for path in sorted(args.out_dir.glob("runtime_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
