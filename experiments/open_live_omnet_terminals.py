#!/usr/bin/env python3
"""
Open terminal windows for parallel live OMNeT + LibSignal inference slots.

By default this opens 18 LibSignal terminals. Each terminal displays the exact
slot command and waits for you to press Enter before running it.

Live co-simulation startup order (same slot, same TraCI port):
  1. Press Enter in the LibSignal terminal (starts SUMO and waits for OMNeT).
  2. Within ~1–2 minutes, press Enter in the matching OMNeT terminal (slot N).
  Do not start OMNeT before LibSignal — Veins connects to SUMO that LibSignal launches.

Examples:
    # Open 18 LibSignal terminals, one per OMNeT slot. By default, slots are
    # split evenly across training PR=0.1, 0.5, and 1.0, while all infer at PR=1.0.
    python3 experiments/open_live_omnet_terminals.py \
      --agent mplight --network hangzhou \
      --slot_train_prs 0.1 0.5 1.0 --rates 1.0 --checkpoint_selection topk \
      --checkpoint_rank 3 --repeats 1

    # Open 18 OMNeT terminals in the matching workspace directories.
    python3 experiments/open_live_omnet_terminals.py --kind omnet

    # Open both LibSignal and OMNeT terminals: 36 windows total.
    python3 experiments/open_live_omnet_terminals.py --kind both
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBSIGNAL_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from omnet_parallel import DEFAULT_OMNET_ROOT, omnet_env_paths

DEFAULT_VENV_ACTIVATE = "/home/exx/percoenv/bin/activate"
DEFAULT_LIVE_SLOTS = 18
DEFAULT_INFER_RATES = [1.0]
DEFAULT_SLOT_TRAIN_PRS = [0.1, 0.5, 1.0]


def quote_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def grouped_value_for_slot(args: argparse.Namespace, slot: int, values: list[float]) -> float:
    """Split values across slot groups and return the value for one slot."""
    if not values:
        raise ValueError("slot group values must contain at least one value")
    slot_idx = slot - args.start_slot
    group_size = max(1, (args.slots + len(values) - 1) // len(values))
    value_idx = min(slot_idx // group_size, len(values) - 1)
    return values[value_idx]


def train_pr_for_slot(args: argparse.Namespace, slot: int) -> float:
    """Return the training PR assigned to one terminal slot."""
    if args.train_pr is not None:
        return float(args.train_pr)
    return grouped_value_for_slot(args, slot, list(args.slot_train_prs))


def rates_for_slot(args: argparse.Namespace, slot: int) -> list[float]:
    """Return the inference PR(s) assigned to one terminal slot."""
    return list(args.rates)


def build_libsignal_command(args: argparse.Namespace, slot: int) -> list[str]:
    train_pr = train_pr_for_slot(args, slot)
    rates = rates_for_slot(args, slot)
    cmd = [
        "python3", "experiments/run_inference_sweep.py",
        "--agent", args.agent,
        "--network", args.network,
        "--train_pr", str(train_pr),
        "--rates", *[str(r) for r in rates],
        "--checkpoint_selection", args.checkpoint_selection,
        "--checkpoint_rank", str(args.checkpoint_rank),
        "--repeats", str(args.repeats),
        "--live_omnet",
        "--omnet_slot", str(slot),
    ]
    if args.matched_train_pr:
        cmd.append("--matched_train_pr")
    if args.train_episodes is not None:
        cmd.extend(["--train_episodes", str(args.train_episodes)])
    if args.selection_metric:
        cmd.extend(["--selection_metric", args.selection_metric])
    return cmd


def terminal_script(
    title: str,
    cwd: str,
    command: str | None,
    auto_run: bool,
    venv_activate: str | None,
    startup_hint: str | None = None,
) -> str:
    lines = [f"cd {shlex.quote(cwd)}"]
    if venv_activate:
        lines.append(f"source {shlex.quote(venv_activate)}")
    lines.extend([
        "clear",
        f"echo {shlex.quote(title)}",
        "echo",
    ])
    if startup_hint:
        lines.extend([
            f"echo {shlex.quote(startup_hint)}",
            "echo",
        ])
    if command:
        lines.extend([
            "echo 'Command:'",
            f"printf '%s\\n' {shlex.quote(command)}",
            "echo",
        ])
        if auto_run:
            lines.append(command)
        else:
            lines.extend([
                "read -p 'Press Enter to start this slot, or Ctrl+C to cancel... '",
                command,
            ])
    else:
        lines.extend([
            "echo 'No command was provided for this terminal.'",
            "echo 'Run your OMNeT command here when the matching LibSignal slot is waiting.'",
        ])
    lines.extend([
        "echo",
        "echo 'Terminal finished. Press Enter to keep shell open.'",
        "read _",
        "exec bash",
    ])
    return "; ".join(lines)


def open_terminal(
    title: str,
    cwd: str,
    command: str | None,
    auto_run: bool,
    dry_run: bool,
    venv_activate: str | None,
    startup_hint: str | None = None,
) -> None:
    script = terminal_script(title, cwd, command, auto_run, venv_activate, startup_hint)
    if dry_run:
        print(f"\n[{title}]")
        print(f"cwd: {cwd}")
        print(command or "(interactive OMNeT terminal)")
        return

    if shutil.which("gnome-terminal"):
        subprocess.Popen([
            "gnome-terminal",
            "--title", title,
            "--",
            "bash", "-lc", script,
        ])
        return

    if shutil.which("x-terminal-emulator"):
        subprocess.Popen([
            "x-terminal-emulator",
            "-T", title,
            "-e", "bash", "-lc", script,
        ])
        return

    raise RuntimeError("No supported terminal launcher found (gnome-terminal or x-terminal-emulator).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Open live-OMNeT parallel run terminals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kind", choices=["libsignal", "omnet", "both"], default="libsignal")
    p.add_argument("--slots", type=int, default=DEFAULT_LIVE_SLOTS)
    p.add_argument("--start_slot", type=int, default=1)
    p.add_argument("--omnet_root", default=DEFAULT_OMNET_ROOT)
    p.add_argument("--auto_run", action="store_true", help="Run commands immediately instead of waiting for Enter")
    p.add_argument("--dry_run", action="store_true", help="Print commands without opening terminals")

    p.add_argument("--agent", default="presslight")
    p.add_argument("--network", default="hangzhou")
    p.add_argument(
        "--train_pr", type=float, default=None,
        help=(
            "Use one fixed training PR for every LibSignal terminal. If omitted, "
            "--slot_train_prs are split across slots."
        ),
    )
    p.add_argument("--matched_train_pr", action="store_true")
    p.add_argument(
        "--rates", type=float, nargs="+", default=DEFAULT_INFER_RATES,
        help=(
            "Inference PR list used in every LibSignal terminal. Default is PR=1.0 for all slots."
        ),
    )
    p.add_argument(
        "--slot_train_prs", type=float, nargs="+", default=DEFAULT_SLOT_TRAIN_PRS,
        help=(
            "Training PR values assigned by slot groups when --train_pr is omitted. "
            "With the default 18 slots, this gives 6 slots each trained at 0.1, 0.5, and 1.0."
        ),
    )
    p.add_argument("--checkpoint_selection", choices=["final", "best", "topk"], default="topk")
    p.add_argument(
        "--checkpoint_rank",
        type=int,
        default=5,
        help=(
            "1-based rank to start checkpoint selection from. "
            "Default 2 runs the second-best checkpoint with "
            "--checkpoint_selection topk --repeats 1."
        ),
    )
    p.add_argument("--selection_metric", default="travel_time")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--train_episodes", type=int, default=None)

    p.add_argument(
        "--omnet_command",
        default=None,
        help="Optional OMNeT command to place in OMNeT terminals. If omitted, terminals open in the cars folder.",
    )
    p.add_argument(
        "--venv_activate",
        default=DEFAULT_VENV_ACTIVATE,
        help="Path to activate script sourced in each terminal before the run command",
    )
    p.add_argument(
        "--no_venv_activate",
        action="store_true",
        help="Do not source a virtualenv in opened terminals",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    last_slot = args.start_slot + args.slots - 1
    venv_activate = None if args.no_venv_activate else args.venv_activate

    for slot in range(args.start_slot, last_slot + 1):
        env = omnet_env_paths(slot, args.omnet_root)

        if args.kind in ("libsignal", "both"):
            assigned_train_pr = train_pr_for_slot(args, slot)
            assigned_rates = rates_for_slot(args, slot)
            infer_rate_label = ",".join(str(r) for r in assigned_rates)
            cmd = quote_cmd(build_libsignal_command(args, slot))
            title = (
                f"LibSignal OMNeT slot {slot} trainPR {assigned_train_pr} "
                f"inferPR {infer_rate_label} port {env['traci_port']}"
            )
            open_terminal(
                title,
                LIBSIGNAL_DIR,
                cmd,
                args.auto_run,
                args.dry_run,
                venv_activate,
                startup_hint=(
                    f"STEP 1 (slot {slot}): Press Enter here first to start SUMO on "
                    f"port {env['traci_port']} with train PR {assigned_train_pr} "
                    f"and infer PR {infer_rate_label}, then start OMNeT slot {slot}."
                ),
            )

        if args.kind in ("omnet", "both"):
            cars_dir = Path(env["workspace_dir"]) / "simu5G" / "simulations" / "NR" / "cars"
            title = f"OMNeT slot {slot} port {env['traci_port']}"
            omnet_cmd = args.omnet_command
            if omnet_cmd is None and args.kind == "both":
                omnet_cmd = (
                    "opp_run -r 0 -m -u Cmdenv -c C-V2X-D2DMulticast "
                    "-n ../../../emulation:../..:../../../src:../../../../inet4.5/examples:"
                    "../../../../inet4.5/showcases:../../../../inet4.5/src:"
                    "../../../../inet4.5/tests/validation:../../../../inet4.5/tests/networks:"
                    "../../../../inet4.5/tutorials:../../../../veins-veins-5.2/examples/veins:"
                    "../../../../veins-veins-5.2/src/veins:../../../../veins_inet/src/veins_inet:"
                    "../../../../veins_inet/examples/veins_inet -x inet.common.selfdoc "
                    "--image-path=../../../images:../../../../inet4.5/images:"
                    "../../../../veins-veins-5.2/images:../../../../veins_inet/images "
                    "-l ../../../src/simu5G -l ../../../../inet4.5/src/INET "
                    "-l ../../../../veins-veins-5.2/src/veins "
                    "-l ../../../../veins_inet/src/veins_inet omnetpp-CV2X.ini"
                )
            open_terminal(
                title,
                str(cars_dir),
                omnet_cmd,
                args.auto_run,
                args.dry_run,
                venv_activate,
                startup_hint=(
                    f"STEP 2 (slot {slot}): Start this only AFTER LibSignal slot {slot} "
                    f"is running (port {env['traci_port']})."
                ),
            )

    print(f"Prepared terminals for slots {args.start_slot}..{last_slot} ({args.kind}).")


if __name__ == "__main__":
    main()
