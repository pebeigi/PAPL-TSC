"""
Helpers for parallel live OMNeT + SUMO + LibSignal runs.

Each slot (1..N) maps to one OMNeT workspace folder and one TraCI port so
multiple simulations can run concurrently without port collisions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_OMNET_ROOT = "/home/exx/Desktop/vtc2026/omnet_files"
DEFAULT_TRACI_PORT_BASE = 9999
DEFAULT_TRACI_PORT_STEP = 2
DEFAULT_NUM_SLOTS = 15
OMNET_CSV_REL = "simu5G/simulations/NR/cars/SUMO_output_CV2X.csv"
OMNET_INI_REL = "simu5G/simulations/NR/cars/omnetpp-CV2X.ini"


def workspace_name_for_slot(slot: int) -> str:
    """Slot 1 -> gwu-workspace-pedestrians; slot 2..N -> gwu-workspace-pedestrians-N."""
    if slot < 1:
        raise ValueError(f"omnet slot must be >= 1, got {slot}")
    if slot == 1:
        return "gwu-workspace-pedestrians"
    return f"gwu-workspace-pedestrians-{slot}"


def traci_port_for_slot(
    slot: int,
    *,
    port_base: int = DEFAULT_TRACI_PORT_BASE,
    port_step: int = DEFAULT_TRACI_PORT_STEP,
) -> int:
    return port_base + (slot - 1) * port_step


def omnet_env_paths(
    slot: int,
    omnet_root: str = DEFAULT_OMNET_ROOT,
    *,
    port_base: int = DEFAULT_TRACI_PORT_BASE,
    port_step: int = DEFAULT_TRACI_PORT_STEP,
) -> dict[str, Any]:
    """Return workspace folder, TraCI port, CSV path, and INI path for one slot."""
    workspace = workspace_name_for_slot(slot)
    root = Path(omnet_root)
    ws_dir = root / workspace
    return {
        "slot": slot,
        "workspace": workspace,
        "workspace_dir": str(ws_dir),
        "traci_port": traci_port_for_slot(slot, port_base=port_base, port_step=port_step),
        "omnet_csv_path": str(ws_dir / OMNET_CSV_REL),
        "omnet_ini_path": str(ws_dir / OMNET_INI_REL),
    }


def build_port_manifest(
    omnet_root: str = DEFAULT_OMNET_ROOT,
    num_slots: int = DEFAULT_NUM_SLOTS,
    *,
    port_base: int = DEFAULT_TRACI_PORT_BASE,
    port_step: int = DEFAULT_TRACI_PORT_STEP,
) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for slot in range(1, num_slots + 1):
        manifest[str(slot)] = omnet_env_paths(
            slot, omnet_root, port_base=port_base, port_step=port_step,
        )
    return manifest


def write_port_manifest(path: str, manifest: dict[str, dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def load_port_manifest(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def patch_omnet_ini_port(ini_path: str, port: int) -> bool:
    """Set ``*.veinsManager.port`` in an OMNeT INI file. Returns True if file changed."""
    ini = Path(ini_path)
    if not ini.is_file():
        return False
    text = ini.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^(\*\.veinsManager\.port\s*=\s*)\d+\s*$",
        re.MULTILINE,
    )
    replacement = rf"\g<1>{port}"
    new_text, count = pattern.subn(replacement, text)
    if count == 0:
        new_text = text.rstrip() + f"\n*.veinsManager.port = {port}\n"
    if new_text == text:
        return False
    ini.write_text(new_text, encoding="utf-8")
    return True


def configure_all_omnet_ports(
    omnet_root: str = DEFAULT_OMNET_ROOT,
    num_slots: int = DEFAULT_NUM_SLOTS,
    *,
    port_base: int = DEFAULT_TRACI_PORT_BASE,
    port_step: int = DEFAULT_TRACI_PORT_STEP,
    manifest_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Assign unique TraCI ports in each workspace INI and optionally write manifest."""
    manifest = build_port_manifest(
        omnet_root, num_slots, port_base=port_base, port_step=port_step,
    )
    for entry in manifest.values():
        patch_omnet_ini_port(entry["omnet_ini_path"], int(entry["traci_port"]))
    if manifest_path:
        write_port_manifest(manifest_path, manifest)
    return manifest
