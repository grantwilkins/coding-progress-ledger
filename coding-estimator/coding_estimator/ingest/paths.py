"""Resolve paths into the upstream coding-progress-ledger checkout."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LEDGER_ROOT = Path(__file__).resolve().parents[2].parent / "coding-progress-ledger"

SOURCE_ROOT_NAMES: dict[str, str] = {
    "swe_agent_pilot": "swe_agent_pilot",
    "swe_agent_pilot_v3": "swe_agent_pilot_v3",
    "swe_agent_live": "swe_agent_live",
    "swe_agent_live_wallclock": "swe_agent_live_wallclock",
    "hermes_pilot": "hermes_pilot",
    "hermes_pilot_h5": "hermes_pilot_h5",
    "hermes_pilot_h5_v2": "hermes_pilot_h5_v2",
    "tb_live": "tb_live",
}


def ledger_root() -> Path:
    env = os.environ.get("LEDGER_ROOT")
    root = Path(env).resolve() if env else DEFAULT_LEDGER_ROOT.resolve()
    if not root.exists():
        raise FileNotFoundError(f"ledger_root does not exist: {root}")
    if not (root / "ledger_progress").is_dir():
        raise FileNotFoundError(f"ledger_root is not a coding-progress-ledger checkout: {root}")
    return root


def runs_root(source: str) -> Path:
    if source not in SOURCE_ROOT_NAMES:
        raise KeyError(f"unknown source: {source}")
    path = ledger_root() / "runs" / SOURCE_ROOT_NAMES[source]
    if not path.is_dir():
        raise FileNotFoundError(f"runs root for {source!r} not present: {path}")
    return path


def run_dir(source: str, run_id: str) -> Path:
    path = runs_root(source) / run_id
    if not path.is_dir():
        raise FileNotFoundError(f"run dir not present: {path}")
    return path


def list_run_ids(source: str) -> list[str]:
    root = runs_root(source)
    return sorted(p.name for p in root.iterdir() if p.is_dir())
