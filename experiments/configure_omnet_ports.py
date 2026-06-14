#!/usr/bin/env python3
"""
Assign unique TraCI ports across OMNeT workspace folders (1..15 by default).

Updates ``*.veinsManager.port`` in each ``omnetpp-CV2X.ini`` and writes a JSON
manifest for run_inference_sweep.py (--live_omnet --omnet_slot N).

Usage (from LibSignal-master):
    python experiments/configure_omnet_ports.py
    python experiments/configure_omnet_ports.py --num_slots 15 --port_base 9999
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from omnet_parallel import (  # noqa: E402
    DEFAULT_NUM_SLOTS,
    DEFAULT_OMNET_ROOT,
    DEFAULT_TRACI_PORT_BASE,
    DEFAULT_TRACI_PORT_STEP,
    configure_all_omnet_ports,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Configure unique TraCI ports per OMNeT workspace")
    p.add_argument("--omnet_root", default=DEFAULT_OMNET_ROOT)
    p.add_argument("--num_slots", type=int, default=DEFAULT_NUM_SLOTS)
    p.add_argument("--port_base", type=int, default=DEFAULT_TRACI_PORT_BASE)
    p.add_argument("--port_step", type=int, default=DEFAULT_TRACI_PORT_STEP)
    p.add_argument(
        "--manifest",
        default=os.path.join(SCRIPT_DIR, "omnet_port_manifest.json"),
        help="Where to write slot -> port/workspace mapping",
    )
    args = p.parse_args()

    manifest = configure_all_omnet_ports(
        args.omnet_root,
        args.num_slots,
        port_base=args.port_base,
        port_step=args.port_step,
        manifest_path=args.manifest,
    )

    print(f"Configured {len(manifest)} OMNeT workspace(s). Manifest:\n  {args.manifest}\n")
    print(f"{'slot':>4}  {'port':>6}  workspace")
    print("-" * 50)
    for slot in sorted(manifest, key=int):
        e = manifest[slot]
        print(f"{slot:>4}  {e['traci_port']:>6}  {e['workspace']}")


if __name__ == "__main__":
    main()
