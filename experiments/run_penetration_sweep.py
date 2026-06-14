"""
Penetration-rate sweep experiment runner.

For each (penetration_rate, omnet_mode) combination this script launches
run.py as a subprocess with an auto-generated --prefix so that every run
lands in its own, fully self-contained folder:

    data/output_data/tsc/{world}_{agent}/{network}/{prefix}/
        logger/   ← BRF / DTL log files
        model/    ← saved .pt checkpoints
        dataset/  ← replay buffer (if any)

Prefix convention:
    omnet_off__pr_1.00   (no OMNeT, full penetration)
    omnet_off__pr_0.50   (no OMNeT, 50 % penetration)
    omnet_on__pr_1.00    (OMNeT mode, full penetration)

Supported networks (use --network shorthand or full config name):
    hangzhou   →  sumohz4x4   (Hangzhou 4×4 grid)
    atlanta    →  sumoatl1x5  (Atlanta 1×5 arterial)

Supported agents:
    presslight, mplight  (and any other registered agent)

Usage
-----
    # From the LibSignal-master directory:
    python experiments/run_penetration_sweep.py \\
        --agent presslight \\
        --network hangzhou \\
        --rates 0.25 0.5 0.75 1.0 \\
        --episodes 50

    python experiments/run_penetration_sweep.py \\
        --agent mplight \\
        --network atlanta \\
        --rates 0.1 0.25 0.5 1.0 \\
        --omnet_modes off

Results
-------
After all runs finish a CSV summary is written to:
    experiments/results/sweep_{agent}_{network}_{timestamp}.csv
"""

import argparse
import concurrent.futures
import csv
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Network shorthand → config-file name mapping
# ---------------------------------------------------------------------------
NETWORK_ALIASES: dict[str, str] = {
    "hangzhou": "sumohz4x4",
    "atlanta":  "sumoatl1x5",
}

AGENT_CHOICES = ["presslight", "mplight", "dqn", "colight", "frap",
                 "maxpressure", "fixedtime", "sotl"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_network(name: str) -> str:
    """Expand shorthand names to their config-file names."""
    return NETWORK_ALIASES.get(name.lower(), name)


def make_prefix(penetration_rate: float, use_omnet: bool) -> str:
    omnet_tag = "omnet_on" if use_omnet else "omnet_off"
    return f"{omnet_tag}__pr_{penetration_rate:.2f}"


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def read_last_lines(path: str, n: int = 20) -> list[str]:
    """Return the last n lines of a text file, used for concise failure output."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def parse_step_metrics(line: str) -> dict:
    """Extract compact metrics from trainer 'step:' log lines."""
    m = re.search(
        r'step:(?P<step>\d+)/(?P<steps>\d+),\s*q_loss:(?P<loss>[-\d.]+),\s*'
        r'rewards:(?P<reward>[-\d.]+),\s*queue:(?P<queue>[-\d.]+),\s*'
        r'delay:(?P<delay>[-\d.]+),\s*throughput:(?P<throughput>[-\d.]+)',
        line
    )
    if not m:
        return {}
    out = m.groupdict()
    for key in ('loss', 'reward', 'queue', 'delay', 'throughput'):
        out[key] = float(out[key])
    out['step'] = int(out['step'])
    out['steps'] = int(out['steps'])
    return out


def parse_episode_progress(line: str) -> dict:
    """Extract episode progress from trainer 'episode:' log lines."""
    m = re.search(
        r'episode:(?P<episode>\d+)/(?P<episodes>\d+),\s*'
        r'real avg travel time:(?P<travel_time>[-\d.]+)',
        line
    )
    if not m:
        return {}
    out = m.groupdict()
    out['episode'] = int(out['episode'])
    out['episodes'] = int(out['episodes'])
    out['travel_time'] = float(out['travel_time'])
    return out


def should_print_episode(episode: int, episodes: int, every: int) -> bool:
    """Episodes in logs are zero-indexed; print human-friendly checkpoints."""
    current = episode + 1
    return current == 1 or current == episodes or current % every == 0


def get_effective_episodes(args_ns, libsignal_dir: str) -> int | None:
    """Return CLI --episodes or the shared TSC config default if it can be read."""
    if args_ns.episodes is not None:
        return args_ns.episodes
    base_config = os.path.join(libsignal_dir, 'configs', 'tsc', 'base.yml')
    try:
        with open(base_config, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(r'\s*episodes:\s*(\d+)\s*$', line)
                if m:
                    return int(m.group(1))
    except OSError:
        return None
    return None


def is_existing_success(output_root: str, prefix: str,
                        effective_episodes: int | None) -> bool:
    """
    A run is considered reusable when it has metrics logs and a final checkpoint.

    The checkpoint guard prevents reusing older 100-episode runs after changing
    the default to 50 episodes.
    """
    metrics = collect_run_results_from_output_dir(output_root, prefix)
    if not metrics:
        return False

    model_dir = os.path.join(output_root, prefix, 'model')
    if effective_episodes is not None:
        return os.path.isfile(os.path.join(model_dir, f'{effective_episodes}_0.pt'))
    return bool(list(Path(model_dir).glob('*.pt')))


def collect_run_results_from_output_dir(output_root: str, prefix: str) -> dict:
    run_dir = os.path.join(output_root, prefix, 'logger')
    return collect_run_results_from_logger(run_dir)


def collect_run_results_from_logger(run_dir: str) -> dict:
    if not os.path.isdir(run_dir):
        return {}

    # --- 1. DTL log (preferred: written every episode for both TRAIN and TEST) ---
    dtl_files = sorted(Path(run_dir).glob('*_DTL.log'))
    if dtl_files:
        last_test_row = None
        with open(dtl_files[-1], 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                # agent_name  mode  step  travel_time  loss  reward  queue  delay  throughput
                if len(parts) >= 9 and parts[1] == 'TEST':
                    last_test_row = parts
        if last_test_row is not None:
            try:
                return {
                    'travel_time': float(last_test_row[3]),
                    'loss':        float(last_test_row[4]),
                    'reward':      float(last_test_row[5]),
                    'queue':       float(last_test_row[6]),
                    'delay':       float(last_test_row[7]),
                    'throughput':  float(last_test_row[8]),
                }
            except (IndexError, ValueError):
                pass

    # --- 2. BRF log fallback (inference-only runs write only a BRF log) ---
    brf_files = sorted(Path(run_dir).glob('*_BRF.log'))
    for brf_path in reversed(brf_files):   # most-recent first
        with open(brf_path, 'r') as f:
            for line in f:
                # "Final Travel Time is 615.7262, mean rewards: -3.8946,
                #  queue: 2.9609, delay: 0.2208, throughput: 2458"
                m = re.search(
                    r'Final Travel Time is\s+([\d.]+),\s*mean rewards:\s*([-\d.]+),\s*'
                    r'queue:\s*([\d.]+),\s*delay:\s*([\d.]+),\s*throughput:\s*(\d+)',
                    line)
                if m:
                    return {
                        'travel_time': float(m.group(1)),
                        'loss':        0.0,
                        'reward':      float(m.group(2)),
                        'queue':       float(m.group(3)),
                        'delay':       float(m.group(4)),
                        'throughput':  float(m.group(5)),
                    }
    return {}


def collect_run_results(output_root: str, task: str, world: str,
                        agent: str, network: str, prefix: str) -> dict:
    """
    Parse log files from a completed run and return summary metrics.

    Tries in order:
    1. *_DTL.log  – tab-separated rows written during train/test episodes.
       Format: agent  mode  step  travel_time  loss  reward  queue  delay  throughput
    2. *_BRF.log  – free-text summary written by pure-test (inference) runs.
       Format: "Final Travel Time is X, mean rewards: Y, queue: Z, delay: W, throughput: T"

    Returns an empty dict when no usable log is found.
    """
    run_dir = os.path.join(output_root, 'output_data', task,
                           f"{world}_{agent}", network, prefix, 'logger')
    return collect_run_results_from_logger(run_dir)


def build_run_command(args_ns, network_cfg: str,
                      penetration_rate: float, use_omnet: bool) -> tuple:
    """Build the subprocess command list for a single run."""
    prefix = make_prefix(penetration_rate, use_omnet)
    cmd = [
        sys.executable, 'run.py',
        '--task',             args_ns.task,
        '--agent',            args_ns.agent,
        '--world',            args_ns.world,
        '--network',          network_cfg,
        '--dataset',          args_ns.dataset,
        '--interface',        args_ns.interface,
        '--delay_type',       args_ns.delay_type,
        '--prefix',           prefix,
        '--penetration_rate', str(penetration_rate),
        '--ngpu',             str(args_ns.ngpu),
        '--thread_num',       str(args_ns.thread_num),
    ]
    if args_ns.episodes is not None:
        cmd += ['--episodes', str(args_ns.episodes)]
    if args_ns.seed is not None:
        cmd += ['--seed', str(args_ns.seed)]
    if use_omnet:
        cmd += ['--use_omnet', '--omnet_csv_path', args_ns.omnet_csv_path]
    if args_ns.debug:
        cmd += ['--debug', 'True']
    return cmd, prefix


def locked_print(print_lock: threading.Lock, *args, **kwargs) -> None:
    with print_lock:
        print(*args, **kwargs)


def run_one_experiment(idx: int, total: int, rate: float, use_omnet: bool,
                       args, network_cfg: str, libsignal_dir: str,
                       output_root: str, console_log_dir: str,
                       effective_episodes: int | None,
                       print_lock: threading.Lock) -> dict:
    cmd, prefix = build_run_command(args, network_cfg, rate, use_omnet)
    omnet_label = "omnet=ON" if use_omnet else "omnet=OFF"
    progress = 100.0 * idx / total
    output_dir = os.path.join(output_root, prefix)
    run_console_log = None if args.verbose else os.path.join(
        console_log_dir, f"{idx:03d}_{prefix}.log"
    )

    entry = {
        'run_idx':          idx,
        'agent':            args.agent,
        'network':          network_cfg,
        'world':            args.world,
        'penetration_rate': rate,
        'use_omnet':        use_omnet,
        'prefix':           prefix,
        'output_dir':       output_dir,
        'status':           'pending',
        'exit_code':        None,
        'travel_time':      None,
        'loss':             None,
        'reward':           None,
        'queue':            None,
        'delay':            None,
        'throughput':       None,
        'console_log':      run_console_log,
    }

    if args.skip_existing and is_existing_success(output_root, prefix, effective_episodes):
        metrics = collect_run_results_from_output_dir(output_root, prefix)
        entry.update(metrics)
        entry['status'] = 'skipped'
        entry['exit_code'] = 0
        locked_print(
            print_lock,
            f"[{idx}/{total} | {progress:5.1f}%] SKIP   "
            f"pr={rate:.3f}  {omnet_label}  prefix={prefix}  "
            f"existing checkpoint/metrics found",
            flush=True,
        )
        return entry

    locked_print(print_lock, "-" * 70, flush=True)
    locked_print(
        print_lock,
        f"[{idx}/{total} | {progress:5.1f}%] START  "
        f"pr={rate:.3f}  {omnet_label}  prefix={prefix}",
        flush=True,
    )
    locked_print(print_lock, f"  output: {output_dir}", flush=True)
    if args.verbose:
        locked_print(print_lock, "  console: streaming live (--verbose)", flush=True)
        locked_print(print_lock, "  cmd:", " ".join(cmd), flush=True)
    else:
        locked_print(print_lock, f"  console log: {run_console_log}", flush=True)

    if args.dry_run:
        entry['status'] = 'dry_run'
        return entry

    start_time = time.monotonic()
    if args.verbose:
        proc = subprocess.run(cmd, cwd=libsignal_dir)
    else:
        last_step_metrics = {}
        child_env = os.environ.copy()
        child_env.setdefault('PYTHONUNBUFFERED', '1')
        with open(run_console_log, 'w', encoding='utf-8') as log_f:
            log_f.write("CMD: " + " ".join(cmd) + "\n\n")
            log_f.flush()
            proc_handle = subprocess.Popen(
                cmd,
                cwd=libsignal_dir,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc_handle.stdout is not None
            for line in proc_handle.stdout:
                log_f.write(line)
                log_f.flush()

                if args.no_progress:
                    continue

                step_metrics = parse_step_metrics(line)
                if step_metrics:
                    last_step_metrics = step_metrics
                    continue

                episode_progress = parse_episode_progress(line)
                if not episode_progress:
                    continue

                episode = episode_progress['episode']
                episodes = episode_progress['episodes']
                if not should_print_episode(
                    episode, episodes, max(1, args.progress_every)
                ):
                    continue

                current = episode + 1
                pct_episode = 100.0 * current / episodes if episodes else 0.0
                msg = (
                    f"  progress [{idx}/{total}] pr={rate:.3f}: "
                    f"episode {current}/{episodes} ({pct_episode:5.1f}%)  "
                    f"elapsed={format_duration(time.monotonic() - start_time)}  "
                    f"travel_time={episode_progress['travel_time']:.2f}"
                )
                if last_step_metrics:
                    msg += (
                        f"  queue={last_step_metrics['queue']:.3f}"
                        f"  delay={last_step_metrics['delay']:.3f}"
                        f"  throughput={last_step_metrics['throughput']:.0f}"
                    )
                locked_print(print_lock, msg, flush=True)

            proc_handle.wait()
            proc = subprocess.CompletedProcess(cmd, proc_handle.returncode)

    elapsed = time.monotonic() - start_time
    entry['exit_code'] = proc.returncode

    if proc.returncode == 0:
        entry['status'] = 'success'
        metrics = collect_run_results(
            os.path.join(libsignal_dir, 'data'),
            args.task, args.world, args.agent, network_cfg, prefix
        )
        entry.update(metrics)
        tt = f"{metrics.get('travel_time'):.2f}" if metrics.get('travel_time') is not None else "n/a"
        tp = f"{metrics.get('throughput'):.0f}" if metrics.get('throughput') is not None else "n/a"
        delay = f"{metrics.get('delay'):.3f}" if metrics.get('delay') is not None else "n/a"
        locked_print(
            print_lock,
            f"[{idx}/{total}] DONE   pr={rate:.3f}  "
            f"elapsed={format_duration(elapsed)}  "
            f"travel_time={tt}  throughput={tp}  delay={delay}",
            flush=True,
        )
    else:
        entry['status'] = 'failed'
        locked_print(
            print_lock,
            f"[{idx}/{total}] FAILED pr={rate:.3f}  "
            f"exit_code={proc.returncode}  elapsed={format_duration(elapsed)}",
            flush=True,
        )
        if not args.verbose:
            locked_print(print_lock, f"  Full console log: {run_console_log}", flush=True)
            tail = read_last_lines(run_console_log, args.log_tail_lines)
            if tail:
                locked_print(print_lock, f"  Last {len(tail)} log lines:", flush=True)
                for line in tail:
                    locked_print(print_lock, "    " + line.rstrip(), flush=True)

    return entry


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep penetration rates for a chosen agent + network "
                    "and organise all results into per-run sub-folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- primary experiment axes ---
    p.add_argument('--agent', default='presslight',
                   choices=AGENT_CHOICES,
                   help="RL agent to train/evaluate")
    p.add_argument('--network', default='hangzhou',
                   help=("network to use; shorthands: 'hangzhou' (sumohz4x4), "
                         "'atlanta' (sumoatl1x5), or any full config name"))
    p.add_argument('--rates', type=float, nargs='+',
                   default=[0.1, 0.25, 0.5, 0.75, 1.0],
                   help="penetration rates to sweep")
    p.add_argument('--omnet_modes', type=str, nargs='+',
                   choices=['on', 'off', 'both'], default=['off'],
                   help="OMNeT modes: 'on', 'off', or 'both'")

    # --- run.py pass-through ---
    p.add_argument('--task',       default='tsc')
    p.add_argument('--world',      default='sumo')
    p.add_argument('--dataset',    default='onfly')
    p.add_argument('--interface',  default='libsumo',
                   choices=['libsumo', 'traci'])
    p.add_argument('--delay_type', default='apx',
                   choices=['apx', 'real'])
    p.add_argument('--episodes',   type=int, default=None,
                   help="override number of training episodes (uses config default if omitted)")
    p.add_argument('--seed',       type=int, default=None)
    p.add_argument('--ngpu',       type=int, default=-1)
    p.add_argument('--thread_num', type=int, default=4)
    p.add_argument('--debug',      action='store_true', default=False)
    p.add_argument('--omnet_csv_path', type=str,
                   default='/home/exx/Desktop/vtc2026/omnet_files/'
                           'gwu-workspace-pedestrians/simu5G/simulations/NR/cars/'
                           'SUMO_output_CV2X.csv',
                   help="OMNeT SUMO_output CSV (only used when --omnet_modes includes 'on')")

    # --- sweep control ---
    p.add_argument('--dry_run', action='store_true', default=False,
                   help="print commands without executing them")
    p.add_argument('--stop_on_failure', action='store_true', default=False,
                   help="abort the sweep if any run exits non-zero")
    p.add_argument('--workers', type=int, default=1,
                   help="number of PR runs to execute in parallel")
    p.add_argument('--skip_existing', action='store_true', default=False,
                   help="skip runs that already have metrics and a final checkpoint")
    p.add_argument('--verbose', action='store_true', default=False,
                   help="stream run.py output live instead of writing it to per-run log files")
    p.add_argument('--progress_every', type=int, default=1,
                   help="print compact per-run progress every N training episodes in compact mode")
    p.add_argument('--no_progress', action='store_true', default=False,
                   help="disable compact per-episode progress printing")
    p.add_argument('--log_tail_lines', type=int, default=20,
                   help="number of subprocess log lines to print when a run fails")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.workers < 1:
        print("[ERROR] --workers must be >= 1")
        sys.exit(1)
    if args.workers > 1 and args.verbose:
        print("[ERROR] --verbose cannot be used with --workers > 1")
        print("Use compact mode so each worker writes to its own console log.")
        sys.exit(1)

    network_cfg = resolve_network(args.network)

    # Expand omnet_modes
    if 'both' in args.omnet_modes:
        omnet_flags = [False, True]
    else:
        omnet_flags = []
        if 'off' in args.omnet_modes:
            omnet_flags.append(False)
        if 'on' in args.omnet_modes:
            omnet_flags.append(True)

    # Full experiment matrix: (penetration_rate, use_omnet)
    experiments = [
        (rate, use_omnet)
        for use_omnet in omnet_flags
        for rate in args.rates
    ]

    # Paths
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    libsignal_dir = os.path.dirname(script_dir)
    timestamp     = datetime.now().strftime('%Y%m%d_%H%M%S')

    summary_dir = os.path.join(script_dir, 'results')
    os.makedirs(summary_dir, exist_ok=True)
    summary_csv = os.path.join(
        summary_dir,
        f"sweep_{args.agent}_{network_cfg}_{timestamp}.csv"
    )

    output_root = os.path.join(libsignal_dir, 'data',
                               'output_data', args.task,
                               f"{args.world}_{args.agent}", network_cfg)
    console_log_dir = os.path.join(
        script_dir, 'run_logs',
        f"sweep_{args.agent}_{network_cfg}_{timestamp}"
    )
    if not args.dry_run and not args.verbose:
        os.makedirs(console_log_dir, exist_ok=True)
    effective_episodes = get_effective_episodes(args, libsignal_dir)

    print("=" * 70)
    print("Penetration-rate sweep")
    print(f"  Agent   : {args.agent}")
    print(f"  Network : {args.network}  →  config: {network_cfg}")
    print(f"  Rates   : {args.rates}")
    print(f"  OMNeT   : {['off' if not f else 'on' for f in omnet_flags]}")
    print(f"  Runs    : {len(experiments)}")
    print(f"  Workers : {args.workers}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Output  : {output_root}/<prefix>/")
    print(f"  Summary : {summary_csv}")
    if args.verbose:
        print("  Console : verbose mode (run.py output streams live)")
    else:
        print(f"  Console : compact mode; run.py logs in {console_log_dir}/")
        if args.no_progress:
            print("  Progress: disabled")
        else:
            print(f"  Progress: training episode updates every {args.progress_every} episode(s)")
    if effective_episodes is not None:
        print(f"  Episodes: {effective_episodes}")
    print("=" * 70)
    print()

    run_log = []
    print_lock = threading.Lock()

    if args.workers == 1:
        for idx, (rate, use_omnet) in enumerate(experiments, 1):
            entry = run_one_experiment(
                idx, len(experiments), rate, use_omnet,
                args, network_cfg, libsignal_dir, output_root,
                console_log_dir, effective_episodes, print_lock,
            )
            run_log.append(entry)
            print()
            if entry['status'] == 'failed' and args.stop_on_failure:
                print("Stopping sweep (--stop_on_failure).")
                break
    else:
        print(
            f"Running up to {args.workers} PR runs in parallel. "
            "Progress lines may interleave by run index.\n",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {}
            for idx, (rate, use_omnet) in enumerate(experiments, 1):
                future = executor.submit(
                    run_one_experiment,
                    idx, len(experiments), rate, use_omnet,
                    args, network_cfg, libsignal_dir, output_root,
                    console_log_dir, effective_episodes, print_lock,
                )
                future_to_idx[future] = idx

            stop_requested = False
            for future in concurrent.futures.as_completed(future_to_idx):
                entry = future.result()
                run_log.append(entry)
                if entry['status'] == 'failed' and args.stop_on_failure:
                    stop_requested = True
                    print("Stopping sweep (--stop_on_failure): cancelling pending runs.")
                    for pending in future_to_idx:
                        if not pending.done():
                            pending.cancel()
                    break

            if stop_requested:
                concurrent.futures.wait(future_to_idx)

    run_log.sort(key=lambda row: row['run_idx'])

    # ------------------------------------------------------------------
    # Write summary CSV
    # ------------------------------------------------------------------
    if run_log:
        fieldnames = list(run_log[0].keys())
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(run_log)
        print(f"\nSummary CSV written to:\n  {summary_csv}\n")

    # ------------------------------------------------------------------
    # ASCII results table
    # ------------------------------------------------------------------
    col = {'#': 3, 'rate': 6, 'omnet': 5, 'prefix': 28,
           'status': 8, 'travel_time': 12, 'reward': 8,
           'queue': 7, 'delay': 7, 'throughput': 10}
    header_fmt  = (f"{'#':>{col['#']}}  {'rate':>{col['rate']}}  "
                   f"{'omnet':>{col['omnet']}}  {'prefix':<{col['prefix']}}  "
                   f"{'status':>{col['status']}}  {'travel_time':>{col['travel_time']}}  "
                   f"{'reward':>{col['reward']}}  {'queue':>{col['queue']}}  "
                   f"{'delay':>{col['delay']}}  {'throughput':>{col['throughput']}}")
    print("\n=== Sweep results ===")
    print(header_fmt)
    print("-" * (sum(col.values()) + 2 * len(col)))
    for e in run_log:
        tt  = f"{e['travel_time']:.2f}" if e['travel_time'] is not None else "—"
        rwd = f"{e['reward']:.3f}"      if e['reward']      is not None else "—"
        q   = f"{e['queue']:.3f}"       if e['queue']       is not None else "—"
        d   = f"{e['delay']:.3f}"       if e['delay']       is not None else "—"
        tp  = f"{e['throughput']:.0f}"  if e['throughput']  is not None else "—"
        print(f"{e['run_idx']:>{col['#']}}  {e['penetration_rate']:>{col['rate']}.2f}  "
              f"{'on' if e['use_omnet'] else 'off':>{col['omnet']}}  "
              f"{e['prefix']:<{col['prefix']}}  {e['status']:>{col['status']}}  "
              f"{tt:>{col['travel_time']}}  {rwd:>{col['reward']}}  "
              f"{q:>{col['queue']}}  {d:>{col['delay']}}  {tp:>{col['throughput']}}")
    print()


if __name__ == '__main__':
    main()
