"""
Inference-rate sweep experiment runner.

Two experiment modes:

1. **Baseline** (default): load one model trained at ``--train_pr`` (typically 1.0)
   and evaluate at each ``--rates`` value.  Answers: how does a full-observability
   model degrade under partial connectivity at inference time?

2. **PR-matched** (``--matched_train_pr``): for each rate ``p``, load the model
   trained at ``p`` and evaluate at ``p``.  This is PR-PressLight / PR-MPLight:
   the penetration-indexed policy bank from the paper.

Folder layout (sibling to training runs from run_penetration_sweep.py):

    data/output_data/tsc/{world}_{agent}/{network}/
        omnet_off__pr_1.00/model/          <- training checkpoints
        inference__from_pr_1.00__on_pr_0.10__rep_01/
        inference_matched__pr_0.25__rep_02__seed_102/

Summary CSVs (under experiments/results/):
    inference_{agent}_{network}_from_pr_{train_pr}_{ts}.csv          (raw, baseline)
    inference_matched_{agent}_{network}_{ts}.csv                       (raw, matched)
    *_aggregated.csv                                                   (mean/std/CI per infer_pr)

Usage
-----
    # Baseline PressLight: train @ 100%, test @ all rates (3 seeds)
    python experiments/run_inference_sweep.py \\
        --agent presslight --network hangzhou --train_pr 1.0 \\
        --rates 0.05 0.1 0.25 0.5 1.0 --train_episodes 50 --repeats 3 --seed_base 100

    # PR-PressLight: matched train/test at each rate (top-3 training checkpoints)
    python experiments/run_inference_sweep.py \\
        --agent presslight --network hangzhou --matched_train_pr \\
        --rates 0.05 0.1 0.25 0.5 1.0 --checkpoint_selection topk --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from omnet_parallel import (
    DEFAULT_OMNET_ROOT,
    DEFAULT_TRACI_PORT_BASE,
    DEFAULT_TRACI_PORT_STEP,
    load_port_manifest,
    omnet_env_paths,
)


# ---------------------------------------------------------------------------
# Network shorthand -> SUMO config-file name
# ---------------------------------------------------------------------------
NETWORK_ALIASES: dict[str, str] = {
    "hangzhou": "sumohz4x4",
    "atlanta":  "sumoatl1x5",
}

AGENT_CHOICES = [
    "presslight", "mplight", "dqn", "colight", "frap",
    "maxpressure", "fixedtime", "sotl",
]

METRIC_KEYS = ("travel_time", "loss", "reward", "queue", "delay", "throughput")
CHECKPOINT_SELECTION_CHOICES = ("final", "best", "topk")
SELECTION_METRIC_CHOICES = ("travel_time", "throughput", "reward", "queue", "delay")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_network(name: str) -> str:
    return NETWORK_ALIASES.get(name.lower(), name)


def make_train_prefix(train_pr: float, use_omnet: bool = False) -> str:
    """Must match run_penetration_sweep.make_prefix (2 decimal places)."""
    omnet_tag = "omnet_on" if use_omnet else "omnet_off"
    return f"{omnet_tag}__pr_{train_pr:.2f}"


def get_effective_episodes(args_ns: argparse.Namespace, libsignal_dir: str) -> int | None:
    """CLI --train_episodes, else episodes from configs/tsc/base.yml."""
    if args_ns.train_episodes is not None:
        return args_ns.train_episodes
    base_config = os.path.join(libsignal_dir, "configs", "tsc", "base.yml")
    try:
        with open(base_config, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*episodes:\s*(\d+)\s*$", line)
                if m:
                    return int(m.group(1))
    except OSError:
        return None
    return None


def list_available_episode_checkpoints(model_dir: str) -> list[int]:
    """Return sorted episode indices that have agent-rank-0 checkpoints (e.g. 50_0.pt)."""
    episodes = []
    for path in Path(model_dir).glob("*_0.pt"):
        m = re.match(r"^(\d+)_0\.pt$", path.name)
        if m:
            episodes.append(int(m.group(1)))
    return sorted(set(episodes))


def checkpoint_path(model_dir: str, episodes: int) -> str:
    return os.path.join(model_dir, f"{episodes}_0.pt")


def training_logger_dir(output_root: str, train_prefix: str) -> str:
    return os.path.join(output_root, train_prefix, "logger")


def metric_is_lower_better(metric: str) -> bool:
    """True when smaller values are better for checkpoint ranking."""
    return metric in ("travel_time", "delay", "loss", "queue")


def parse_training_test_metrics(logger_dir: str) -> dict[int, dict[str, float]]:
    """
    Parse per-episode TEST rows from training DTL logs.

    Returns episode index -> {travel_time, reward, queue, delay, throughput, ...}.
    """
    if not os.path.isdir(logger_dir):
        return {}

    metrics_by_episode: dict[int, dict[str, float]] = {}
    dtl_files = sorted(Path(logger_dir).glob("*_DTL.log"))
    for dtl_path in dtl_files:
        with open(dtl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 9 or parts[1] != "TEST":
                    continue
                try:
                    episode = int(parts[2])
                    row = {
                        "travel_time": float(parts[3]),
                        "loss":        float(parts[4]),
                        "reward":      float(parts[5]),
                        "queue":       float(parts[6]),
                        "delay":       float(parts[7]),
                        "throughput":  float(parts[8]),
                    }
                except (IndexError, ValueError):
                    continue
                metrics_by_episode[episode] = row
    return metrics_by_episode


def rank_checkpoint_episodes(
    model_dir: str,
    logger_dir: str,
    *,
    metric: str,
    k: int,
) -> list[dict[str, Any]]:
    """
    Rank saved checkpoints using training-time TEST metrics from DTL logs.

    Only episodes with both ``{ep}_0.pt`` and a TEST row are candidates.
    Returns up to ``k`` dicts sorted best-first:
      {checkpoint_episode, selection_metric, selection_value}.
    """
    if k < 1:
        return []

    available_eps = set(list_available_episode_checkpoints(model_dir))
    test_metrics = parse_training_test_metrics(logger_dir)
    candidates: list[dict[str, Any]] = []

    for ep in sorted(available_eps):
        row = test_metrics.get(ep)
        if row is None or metric not in row:
            continue
        candidates.append({
            "checkpoint_episode": ep,
            "selection_metric":   metric,
            "selection_value":    row[metric],
        })

    if not candidates:
        return []

    reverse = not metric_is_lower_better(metric)
    candidates.sort(key=lambda c: c["selection_value"], reverse=reverse)

    ranked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in candidates:
        ep = item["checkpoint_episode"]
        if ep in seen:
            continue
        seen.add(ep)
        ranked.append(item)
        if len(ranked) >= k:
            break
    return ranked


def resolve_checkpoint_episodes(
    output_root: str,
    train_prefix: str,
    *,
    selection: str,
    k: int,
    metric: str,
    final_episode: int,
    rank: int = 1,
) -> list[dict[str, Any]]:
    """
    Choose which training checkpoint(s) to evaluate for one train_prefix.

    - final: use ``final_episode`` (e.g. 50), repeated ``k`` times if k > 1
    - best: single best checkpoint by training TEST metric
    - topk: best ``k`` distinct checkpoints, starting at ``rank`` (1 = best)
    """
    model_dir = trained_model_dir(output_root, train_prefix)
    logger_dir = training_logger_dir(output_root, train_prefix)
    rank = max(1, int(rank))

    if selection == "final":
        ep = final_episode
        if not os.path.isfile(checkpoint_path(model_dir, ep)):
            available = list_available_episode_checkpoints(model_dir)
            if not available:
                return []
            ep = available[-1]
        return [{
            "checkpoint_episode": ep,
            "selection_metric":   "final",
            "selection_value":    float(ep),
        }] * k

    pick_k = 1 if selection == "best" else k
    ranked = rank_checkpoint_episodes(
        model_dir, logger_dir, metric=metric, k=pick_k + rank - 1,
    )
    if ranked:
        return ranked[rank - 1:rank - 1 + pick_k]

    # Fallback when DTL TEST rows are missing: last saved checkpoint
    available = list_available_episode_checkpoints(model_dir)
    if not available:
        return []
    ep = available[-1]
    fallback = [{
        "checkpoint_episode": ep,
        "selection_metric":   "fallback_last",
        "selection_value":    float(ep),
    }]
    return fallback * pick_k


def verify_checkpoint_file(
    output_root: str,
    train_prefix: str,
    train_pr: float,
    checkpoint_episode: int,
) -> bool:
    """Ensure ``{checkpoint_episode}_0.pt`` exists under the training prefix."""
    model_dir = trained_model_dir(output_root, train_prefix)
    if not os.path.isdir(model_dir):
        print(f"[ERROR] Trained model directory not found:\n  {model_dir}")
        print(f"  (penetration rate {train_pr:.3f} -> folder prefix {train_prefix})")
        return False

    ckpt = checkpoint_path(model_dir, checkpoint_episode)
    if os.path.isfile(ckpt):
        return True

    available = list_available_episode_checkpoints(model_dir)
    print(f"[ERROR] Checkpoint not found:\n  {ckpt}")
    if available:
        print(f"  Available episode checkpoints (rank 0): {available}")
    return False


def preflight_training_runs(
    plan: list[dict[str, Any]],
    output_root: str,
) -> bool:
    """Verify training folders and checkpoint files referenced by the plan."""
    ok = True
    seen: set[tuple[str, int]] = set()

    for spec in plan:
        if spec.get("error"):
            ok = False
            print(
                f"[ERROR] No checkpoints for train_pr={spec['train_pr']:.3f} "
                f"(prefix {spec['train_prefix']})"
            )
            continue
        key = (spec["train_prefix"], spec["checkpoint_episode"])
        if key in seen:
            continue
        seen.add(key)
        if not verify_checkpoint_file(
            output_root,
            spec["train_prefix"],
            spec["train_pr"],
            spec["checkpoint_episode"],
        ):
            ok = False
    return ok


def make_inference_prefix(
    train_pr: float,
    infer_pr: float,
    *,
    matched: bool = False,
    checkpoint_episode: int | None = None,
    repeat_idx: int | None = None,
    seed: int | None = None,
    omnet_slot: int | None = None,
) -> str:
    """Return the output prefix for one inference run."""
    if matched:
        base = f"inference_matched__pr_{infer_pr:.2f}"
    else:
        base = f"inference__from_pr_{train_pr:.2f}__on_pr_{infer_pr:.2f}"
    if checkpoint_episode is not None:
        base += f"__ckpt_{checkpoint_episode:02d}"
    if omnet_slot is not None:
        base += f"__omnet_{omnet_slot:02d}"
    if repeat_idx is not None:
        base += f"__rep_{repeat_idx:02d}"
    if seed is not None:
        base += f"__seed_{seed}"
    return base


def resolve_omnet_slot_config(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return OMNeT/TraCI binding for --live_omnet, or None when disabled."""
    if not args.live_omnet:
        return None
    if args.omnet_slot is None:
        raise ValueError("--live_omnet requires --omnet_slot (1..15)")

    if args.omnet_port_manifest and os.path.isfile(args.omnet_port_manifest):
        manifest = load_port_manifest(args.omnet_port_manifest)
        key = str(args.omnet_slot)
        if key not in manifest:
            raise ValueError(f"Slot {args.omnet_slot} not in manifest {args.omnet_port_manifest}")
        return manifest[key]

    cfg = omnet_env_paths(
        args.omnet_slot,
        args.omnet_root,
        port_base=args.traci_port_base,
        port_step=args.traci_port_step,
    )
    if args.traci_port is not None:
        cfg["traci_port"] = args.traci_port
    return cfg


def resolve_repeat_seeds(args: argparse.Namespace) -> list[int | None]:
    """
    Return one seed per repeat (or None when no seed should be passed to run.py).

    Priority: explicit --seeds > --seed_base + repeat index > None per repeat.
    """
    if args.seeds:
        return list(args.seeds)
    if args.seed_base is not None:
        return [args.seed_base + i for i in range(args.repeats)]
    if args.seed is not None:
        return [args.seed + i for i in range(args.repeats)]
    return [None] * args.repeats


def collect_run_results(
    output_root: str,
    task: str,
    world: str,
    agent: str,
    network: str,
    prefix: str,
) -> dict[str, float]:
    """Parse DTL or BRF logs and return summary metrics."""
    import re

    run_dir = os.path.join(
        output_root, "output_data", task,
        f"{world}_{agent}", network, prefix, "logger",
    )
    if not os.path.isdir(run_dir):
        return {}

    dtl_files = sorted(Path(run_dir).glob("*_DTL.log"))
    if dtl_files:
        last_test_row = None
        with open(dtl_files[-1], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 9 and parts[1] == "TEST":
                    last_test_row = parts
        if last_test_row is not None:
            try:
                return {
                    "travel_time": float(last_test_row[3]),
                    "loss":        float(last_test_row[4]),
                    "reward":      float(last_test_row[5]),
                    "queue":       float(last_test_row[6]),
                    "delay":       float(last_test_row[7]),
                    "throughput":  float(last_test_row[8]),
                }
            except (IndexError, ValueError):
                pass

    brf_files = sorted(Path(run_dir).glob("*_BRF.log"))
    for brf_path in reversed(brf_files):
        with open(brf_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(
                    r"Final Travel Time is\s+([\d.]+),\s*mean rewards:\s*([-\d.]+),\s*"
                    r"queue:\s*([\d.]+),\s*delay:\s*([\d.]+),\s*throughput:\s*(\d+)",
                    line,
                )
                if m:
                    return {
                        "travel_time": float(m.group(1)),
                        "loss":        0.0,
                        "reward":      float(m.group(2)),
                        "queue":       float(m.group(3)),
                        "delay":       float(m.group(4)),
                        "throughput":  float(m.group(5)),
                    }
    return {}


def trained_model_dir(output_root: str, train_prefix: str) -> str:
    return os.path.join(output_root, train_prefix, "model")


def verify_trained_model(
    output_root: str,
    train_prefix: str,
    train_pr: float,
    agent: str,
    network_label: str,
) -> bool:
    model_dir = trained_model_dir(output_root, train_prefix)
    if os.path.isdir(model_dir):
        return True
    print(f"[ERROR] Trained model directory not found:\n  {model_dir}")
    print(
        f"\nTrain {agent} on {network_label} at penetration rate {train_pr:.2f} first:\n"
        f"  python experiments/run_penetration_sweep.py "
        f"--agent {agent} --network {network_label} --rates {train_pr:.3f}"
    )
    return False


def build_inference_command(
    args_ns: argparse.Namespace,
    network_cfg: str,
    train_prefix: str,
    infer_prefix: str,
    infer_pr: float,
    checkpoint_episode: int,
    seed: int | None,
    omnet_cfg: dict[str, Any] | None = None,
) -> list[str]:
    interface = "traci" if omnet_cfg else args_ns.interface
    cmd = [
        sys.executable, "run.py",
        "--task",             args_ns.task,
        "--agent",            args_ns.agent,
        "--world",            args_ns.world,
        "--network",          network_cfg,
        "--dataset",          args_ns.dataset,
        "--interface",        interface,
        "--delay_type",       args_ns.delay_type,
        "--prefix",           infer_prefix,
        "--penetration_rate", str(infer_pr),
        "--train_model",      "false",
        "--test_model",       "true",
        "--load_prefix",      train_prefix,
        "--episodes",         str(checkpoint_episode),
        "--ngpu",             str(args_ns.ngpu),
        "--thread_num",       str(args_ns.thread_num),
    ]
    if omnet_cfg:
        cmd += [
            "--use_omnet",
            "--omnet_csv_path", omnet_cfg["omnet_csv_path"],
            "--traci_port", str(omnet_cfg["traci_port"]),
            "--traci_connect_retries", str(args_ns.traci_connect_retries),
            "--traci_connect_delay", str(args_ns.traci_connect_delay),
        ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if args_ns.debug:
        cmd += ["--debug", "True"]
    return cmd


def build_experiment_plan(
    args: argparse.Namespace,
    output_root: str,
    final_episode: int,
    omnet_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand (rates x checkpoint picks x optional seeds) into run specs."""
    seeds = resolve_repeat_seeds(args)
    plan: list[dict[str, Any]] = []
    omnet_slot = int(omnet_cfg["slot"]) if omnet_cfg else None

    for infer_pr in args.rates:
        train_pr = infer_pr if args.matched_train_pr else args.train_pr
        train_prefix = make_train_prefix(train_pr, use_omnet=args.use_omnet)
        ckpt_specs = resolve_checkpoint_episodes(
            output_root,
            train_prefix,
            selection=args.checkpoint_selection,
            k=args.repeats,
            metric=args.selection_metric,
            final_episode=final_episode,
            rank=args.checkpoint_rank,
        )
        if not ckpt_specs:
            plan.append({
                "train_pr":           train_pr,
                "infer_pr":           infer_pr,
                "train_prefix":       train_prefix,
                "infer_prefix":       None,
                "repeat_idx":         None,
                "seed":               None,
                "checkpoint_episode": None,
                "selection_metric":   None,
                "selection_value":    None,
                "error":              "no_checkpoints",
            })
            continue

        for rep_idx, ckpt in enumerate(ckpt_specs, start=1):
            seed = seeds[rep_idx - 1] if rep_idx - 1 < len(seeds) else None
            ckpt_ep = int(ckpt["checkpoint_episode"])
            infer_prefix = make_inference_prefix(
                train_pr,
                infer_pr,
                matched=args.matched_train_pr,
                checkpoint_episode=ckpt_ep,
                repeat_idx=rep_idx if len(ckpt_specs) > 1 else None,
                seed=seed,
                omnet_slot=omnet_slot,
            )
            entry: dict[str, Any] = {
                "train_pr":           train_pr,
                "infer_pr":           infer_pr,
                "train_prefix":       train_prefix,
                "infer_prefix":       infer_prefix,
                "repeat_idx":         rep_idx,
                "seed":               seed,
                "checkpoint_episode": ckpt_ep,
                "selection_metric":   ckpt.get("selection_metric"),
                "selection_value":    ckpt.get("selection_value"),
            }
            if omnet_cfg:
                entry.update({
                    "omnet_slot":      omnet_cfg["slot"],
                    "traci_port":      omnet_cfg["traci_port"],
                    "omnet_csv_path":  omnet_cfg["omnet_csv_path"],
                    "omnet_workspace": omnet_cfg["workspace"],
                })
            plan.append(entry)
    return plan


def write_parallel_launcher(
    path: str,
    base_args: list[str],
    *,
    num_slots: int = 15,
    omnet_root: str = DEFAULT_OMNET_ROOT,
    manifest_path: str | None = None,
) -> None:
    """
    Write a shell script with one command per OMNeT slot for parallel terminals.

    Terminal k: run OMNeT in workspace slot k, then run the generated LibSignal line.
    """
    lines = [
        "#!/usr/bin/env bash",
        "# Parallel live-OMNeT inference launcher (one slot per terminal).",
        "# 1) Run configure_omnet_ports.py once if INI ports are not assigned yet.",
        "# 2) In terminal k: start OMNeT for slot k, then run the matching COMMAND line.",
        "",
    ]
    for slot in range(1, num_slots + 1):
        if manifest_path and os.path.isfile(manifest_path):
            cfg = load_port_manifest(manifest_path)[str(slot)]
        else:
            cfg = omnet_env_paths(slot, omnet_root)
        cmd = [
            "python3", "experiments/run_inference_sweep.py",
            *base_args,
            "--live_omnet",
            f"--omnet_slot", str(slot),
        ]
        lines.append(f"# --- Slot {slot} | TraCI port {cfg['traci_port']} | {cfg['workspace']} ---")
        lines.append(f"# OMNeT: cd {cfg['workspace_dir']}/simu5G/simulations/NR/cars && opp_run ...")
        lines.append("COMMAND=" + " ".join(f'"{p}"' if " " in p else p for p in cmd))
        lines.append('eval "$COMMAND"')
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


def aggregate_by_infer_pr(run_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute mean, std, and approximate 95% CI per (train_pr, infer_pr)."""
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in run_log:
        if row.get("status") != "success":
            continue
        key = (row["train_pr"], row["infer_pr"])
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (train_pr, infer_pr), rows in sorted(groups.items()):
        out: dict[str, Any] = {
            "agent":       rows[0]["agent"],
            "network":     rows[0]["network"],
            "world":       rows[0]["world"],
            "train_pr":    train_pr,
            "infer_pr":    infer_pr,
            "n_success":   len(rows),
            "n_total":     sum(
                1 for r in run_log
                if r["train_pr"] == train_pr and r["infer_pr"] == infer_pr
            ),
        }
        for metric in METRIC_KEYS:
            values = [r[metric] for r in rows if r.get(metric) is not None]
            if not values:
                out[f"{metric}_mean"] = None
                out[f"{metric}_std"] = None
                out[f"{metric}_ci95_low"] = None
                out[f"{metric}_ci95_high"] = None
                continue
            mean = statistics.mean(values)
            out[f"{metric}_mean"] = mean
            if len(values) == 1:
                out[f"{metric}_std"] = 0.0
                out[f"{metric}_ci95_low"] = mean
                out[f"{metric}_ci95_high"] = mean
            else:
                stdev = statistics.stdev(values)
                half = 1.96 * stdev / sqrt(len(values))
                out[f"{metric}_std"] = stdev
                out[f"{metric}_ci95_low"] = mean - half
                out[f"{metric}_ci95_high"] = mean + half
        aggregated.append(out)
    return aggregated


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_stem(args: argparse.Namespace, network_cfg: str, timestamp: str) -> str:
    omnet_tag = ""
    if getattr(args, "live_omnet", False) and getattr(args, "omnet_slot", None) is not None:
        omnet_tag = f"_omnet_{args.omnet_slot:02d}"
    if args.matched_train_pr:
        return f"inference_matched_{args.agent}_{network_cfg}{omnet_tag}_{timestamp}"
    return f"inference_{args.agent}_{network_cfg}_from_pr_{args.train_pr:.2f}{omnet_tag}_{timestamp}"


def print_results_table(run_log: list[dict[str, Any]]) -> None:
    col = {
        "#": 3, "train_pr": 8, "infer_pr": 8, "ckpt": 5, "rep": 4, "seed": 6,
        "prefix": 36, "status": 8, "travel_time": 12, "reward": 8,
        "queue": 7, "delay": 7, "throughput": 10,
    }
    hdr = (
        f"{'#':>{col['#']}}  {'train_pr':>{col['train_pr']}}  "
        f"{'infer_pr':>{col['infer_pr']}}  {'ckpt':>{col['ckpt']}}  "
        f"{'rep':>{col['rep']}}  {'seed':>{col['seed']}}  "
        f"{'prefix':<{col['prefix']}}  {'status':>{col['status']}}  "
        f"{'travel_time':>{col['travel_time']}}  {'reward':>{col['reward']}}  "
        f"{'queue':>{col['queue']}}  {'delay':>{col['delay']}}  "
        f"{'throughput':>{col['throughput']}}"
    )
    print("\n=== Inference sweep results ===")
    print(hdr)
    print("-" * (sum(col.values()) + 2 * len(col)))
    for e in run_log:
        tt = f"{e['travel_time']:.2f}" if e.get("travel_time") is not None else "—"
        rwd = f"{e['reward']:.3f}" if e.get("reward") is not None else "—"
        q = f"{e['queue']:.3f}" if e.get("queue") is not None else "—"
        d = f"{e['delay']:.3f}" if e.get("delay") is not None else "—"
        tp = f"{e['throughput']:.0f}" if e.get("throughput") is not None else "—"
        seed_s = str(e["seed"]) if e.get("seed") is not None else "—"
        ckpt_s = (
            str(e["checkpoint_episode"])
            if e.get("checkpoint_episode") is not None else "—"
        )
        print(
            f"{e['run_idx']:>{col['#']}}  {e['train_pr']:>{col['train_pr']}.2f}  "
            f"{e['infer_pr']:>{col['infer_pr']}.2f}  "
            f"{ckpt_s:>{col['ckpt']}}  "
            f"{e.get('repeat_idx', 0):>{col['rep']}}  {seed_s:>{col['seed']}}  "
            f"{e.get('infer_prefix', ''):<{col['prefix']}}  {e['status']:>{col['status']}}  "
            f"{tt:>{col['travel_time']}}  {rwd:>{col['reward']}}  "
            f"{q:>{col['queue']}}  {d:>{col['delay']}}  {tp:>{col['throughput']}}"
        )
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate pre-trained agents across inference penetration rates. "
            "Use --matched_train_pr for PR-PressLight / PR-MPLight (train and "
            "test at the same rate). Use --repeats/--seeds for multiple runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--agent", default="presslight", choices=AGENT_CHOICES)
    p.add_argument(
        "--network", default="hangzhou",
        help="Shorthand ('hangzhou', 'atlanta') or full config name",
    )
    p.add_argument(
        "--train_pr", type=float, default=1.0,
        help="Training penetration for checkpoint lookup (ignored with --matched_train_pr)",
    )
    p.add_argument(
        "--rates", type=float, nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help="Inference penetration rates to evaluate",
    )
    p.add_argument(
        "--matched_train_pr", action="store_true",
        help="PR-PressLight/PR-MPLight: load model trained at each rate and test at that rate",
    )
    p.add_argument("--train_episodes", type=int, default=None)
    p.add_argument(
        "--use_omnet", action="store_true",
        help="Look for checkpoints under omnet_on__pr_* prefixes (training folder tag only)",
    )
    p.add_argument(
        "--live_omnet", action="store_true",
        help="Live OMNeT co-sim: use TraCI, --use_omnet at runtime, and per-slot CSV/port",
    )
    p.add_argument(
        "--omnet_slot", type=int, default=None,
        help="OMNeT workspace slot 1..15 (required with --live_omnet)",
    )
    p.add_argument(
        "--omnet_root", type=str, default=DEFAULT_OMNET_ROOT,
        help="Root directory containing gwu-workspace-pedestrians* folders",
    )
    p.add_argument(
        "--omnet_port_manifest", type=str,
        default=os.path.join(os.path.dirname(__file__), "omnet_port_manifest.json"),
        help="JSON slot->port map from configure_omnet_ports.py",
    )
    p.add_argument(
        "--traci_port", type=int, default=None,
        help="Override TraCI port (default: from manifest or slot formula)",
    )
    p.add_argument("--traci_port_base", type=int, default=DEFAULT_TRACI_PORT_BASE)
    p.add_argument("--traci_port_step", type=int, default=DEFAULT_TRACI_PORT_STEP)
    p.add_argument(
        "--traci_connect_retries", type=int, default=120,
        help="Passed to run.py while waiting for OMNeT to connect",
    )
    p.add_argument("--traci_connect_delay", type=float, default=1.0)
    p.add_argument(
        "--export_parallel_launcher", type=str, default=None,
        metavar="PATH",
        help="Write a bash script with one sweep command per OMNeT slot and exit",
    )

    p.add_argument("--task", default="tsc")
    p.add_argument("--world", default="sumo")
    p.add_argument("--dataset", default="onfly")
    p.add_argument(
        "--interface", default="libsumo", choices=["libsumo", "traci"],
        help="SUMO API (overridden to traci when --live_omnet)",
    )
    p.add_argument("--delay_type", default="apx", choices=["apx", "real"])
    p.add_argument(
        "--seed", type=int, default=None,
        help="Deprecated: use --seed_base. Starting seed when --repeats > 1",
    )
    p.add_argument(
        "--seed_base", type=int, default=None,
        help="First seed; run i uses seed_base + i (for --repeats > 1)",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit seed list (overrides --repeats/--seed_base; length sets repeat count)",
    )
    p.add_argument(
        "--repeats", type=int, default=1,
        help="With topk: number of best training checkpoints per rate; "
             "with final: repeat count on the final checkpoint",
    )
    p.add_argument(
        "--checkpoint_selection",
        choices=CHECKPOINT_SELECTION_CHOICES,
        default="topk",
        help="final=last episode checkpoint; best=single best by training TEST; "
             "topk=best --repeats checkpoints per training run",
    )
    p.add_argument(
        "--selection_metric",
        choices=SELECTION_METRIC_CHOICES,
        default="travel_time",
        help="Training TEST metric used to rank checkpoints (lower is better except throughput)",
    )
    p.add_argument(
        "--checkpoint_rank",
        type=int,
        default=1,
        help=(
            "1-based checkpoint rank to start from for best/topk selection. "
            "Use 2 with --checkpoint_selection topk --repeats 1 to evaluate only "
            "the second-best checkpoint."
        ),
    )
    p.add_argument("--ngpu", type=int, default=-1)
    p.add_argument("--thread_num", type=int, default=4)
    p.add_argument("--debug", action="store_true", default=False)

    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--stop_on_failure", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.repeats < 1:
        print("[ERROR] --repeats must be >= 1")
        sys.exit(1)
    if args.seeds:
        args.repeats = len(args.seeds)

    network_cfg = resolve_network(args.network)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libsignal_dir = os.path.dirname(script_dir)

    if args.export_parallel_launcher:
        base = [
            "--agent", args.agent,
            "--network", args.network,
            "--train_pr", str(args.train_pr),
            "--rates", *[str(r) for r in args.rates],
            "--checkpoint_selection", args.checkpoint_selection,
            "--repeats", str(args.repeats),
        ]
        if args.matched_train_pr:
            base.append("--matched_train_pr")
        if args.train_episodes is not None:
            base.extend(["--train_episodes", str(args.train_episodes)])
        write_parallel_launcher(
            args.export_parallel_launcher,
            base,
            omnet_root=args.omnet_root,
            manifest_path=args.omnet_port_manifest,
        )
        print(f"Wrote parallel launcher script:\n  {args.export_parallel_launcher}")
        sys.exit(0)

    try:
        omnet_cfg = resolve_omnet_slot_config(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root = os.path.join(
        libsignal_dir, "data", "output_data", args.task,
        f"{args.world}_{args.agent}", network_cfg,
    )
    data_root = os.path.join(libsignal_dir, "data")

    summary_dir = os.path.join(script_dir, "results")
    os.makedirs(summary_dir, exist_ok=True)
    stem = summary_stem(args, network_cfg, timestamp)
    summary_csv = os.path.join(summary_dir, f"{stem}.csv")
    aggregated_csv = os.path.join(summary_dir, f"{stem}_aggregated.csv")

    effective_episodes = get_effective_episodes(args, libsignal_dir)
    if effective_episodes is None:
        print("[ERROR] Could not determine training episodes. Pass --train_episodes explicitly.")
        sys.exit(1)

    if args.checkpoint_selection == "best" and args.repeats > 1:
        print(
            "[WARN] --checkpoint_selection best uses one checkpoint per rate; "
            f"--repeats {args.repeats} is ignored (use topk for multiple checkpoints)."
        )

    plan = build_experiment_plan(args, output_root, effective_episodes, omnet_cfg)
    mode_label = "PR-matched (PR-PressLight/MPLight)" if args.matched_train_pr else "Baseline"
    seeds = resolve_repeat_seeds(args)

    if not preflight_training_runs(plan, output_root):
        sys.exit(1)

    # Show which training checkpoints were selected per train prefix
    ckpt_summary: dict[str, set[int]] = {}
    for spec in plan:
        if spec.get("error"):
            continue
        ckpt_summary.setdefault(spec["train_prefix"], set()).add(spec["checkpoint_episode"])
    print("Checkpoint selection per training run:")
    for prefix in sorted(ckpt_summary):
        eps = sorted(ckpt_summary[prefix])
        print(f"  {prefix}: {eps}")
    print()

    print("=" * 70)
    print("Inference penetration-rate sweep")
    print(f"  Mode         : {mode_label}")
    print(f"  Agent        : {args.agent}")
    print(f"  Network      : {args.network}  ->  {network_cfg}")
    if args.matched_train_pr:
        print(f"  Train/infer  : matched at each rate in --rates")
    else:
        print(f"  Trained at   : pr={args.train_pr:.2f}")
    print(f"  Infer rates  : {args.rates}")
    if omnet_cfg:
        print(f"  Live OMNeT   : slot={omnet_cfg['slot']}  port={omnet_cfg['traci_port']}")
        print(f"                 workspace={omnet_cfg['workspace']}")
        print(f"                 csv={omnet_cfg['omnet_csv_path']}")
    print(f"  Checkpoints  : {args.checkpoint_selection} "
          f"(metric={args.selection_metric}, k={args.repeats}, rank={args.checkpoint_rank})")
    print(f"  Repeats/k    : {args.repeats}")
    print(f"  Seeds        : {seeds if any(s is not None for s in seeds) else '(not set)'}")
    print(f"  Total runs   : {len(plan)}")
    print(f"  Output root  : {output_root}")
    print(f"  Raw CSV      : {summary_csv}")
    print(f"  Aggregated   : {aggregated_csv}")
    if args.checkpoint_selection == "final":
        print(f"  Final episode: {effective_episodes}  (loads {{ep}}_0.pt)")
    print("=" * 70)
    print()

    run_log: list[dict[str, Any]] = []

    for idx, spec in enumerate(plan, 1):
        if spec.get("error"):
            run_log.append({
                "run_idx": idx,
                "mode": "matched" if args.matched_train_pr else "baseline",
                "agent": args.agent,
                "network": network_cfg,
                "world": args.world,
                "train_pr": spec["train_pr"],
                "infer_pr": spec["infer_pr"],
                "status": "failed",
                "exit_code": None,
            })
            continue

        cmd = build_inference_command(
            args, network_cfg,
            spec["train_prefix"], spec["infer_prefix"],
            spec["infer_pr"], spec["checkpoint_episode"], spec["seed"],
            omnet_cfg=omnet_cfg,
        )

        print("-" * 70)
        print(
            f"[{idx}/{len(plan)}] train_pr={spec['train_pr']:.2f}  "
            f"infer_pr={spec['infer_pr']:.2f}  ckpt={spec['checkpoint_episode']}  "
            f"rep={spec['repeat_idx']}  seed={spec['seed']}  "
            f"prefix={spec['infer_prefix']}"
        )
        print("CMD:", " ".join(cmd))
        print()

        entry: dict[str, Any] = {
            "run_idx":            idx,
            "mode":               "matched" if args.matched_train_pr else "baseline",
            "agent":              args.agent,
            "network":            network_cfg,
            "world":              args.world,
            "train_pr":           spec["train_pr"],
            "infer_pr":           spec["infer_pr"],
            "checkpoint_episode": spec["checkpoint_episode"],
            "selection_metric":   spec.get("selection_metric"),
            "selection_value":    spec.get("selection_value"),
            "omnet_slot":         spec.get("omnet_slot"),
            "traci_port":         spec.get("traci_port"),
            "omnet_csv_path":     spec.get("omnet_csv_path"),
            "omnet_workspace":    spec.get("omnet_workspace"),
            "repeat_idx":         spec["repeat_idx"],
            "seed":               spec["seed"],
            "train_prefix":       spec["train_prefix"],
            "infer_prefix":       spec["infer_prefix"],
            "output_dir":         os.path.join(output_root, spec["infer_prefix"]),
            "status":        "pending",
            "exit_code":     None,
            "travel_time":   None,
            "loss":          None,
            "reward":        None,
            "queue":         None,
            "delay":         None,
            "throughput":    None,
        }

        if args.dry_run:
            entry["status"] = "dry_run"
            run_log.append(entry)
            continue

        proc = subprocess.run(cmd, cwd=libsignal_dir)
        entry["exit_code"] = proc.returncode

        if proc.returncode == 0:
            entry["status"] = "success"
            metrics = collect_run_results(
                data_root, args.task, args.world,
                args.agent, network_cfg, spec["infer_prefix"],
            )
            entry.update(metrics)
            print(f"\n  Run {idx} OK — {metrics}")
        else:
            entry["status"] = "failed"
            print(f"\n  Run {idx} FAILED (exit code {proc.returncode})")
            if args.stop_on_failure:
                print("Stopping sweep (--stop_on_failure).")
                run_log.append(entry)
                break

        run_log.append(entry)
        print()

    if run_log:
        write_csv(summary_csv, run_log)
        print(f"\nRaw summary CSV:\n  {summary_csv}\n")

        aggregated = aggregate_by_infer_pr(run_log)
        if aggregated:
            write_csv(aggregated_csv, aggregated)
            print(f"Aggregated CSV (mean/std/95% CI per infer_pr):\n  {aggregated_csv}\n")

    print_results_table(run_log)


if __name__ == "__main__":
    main()
