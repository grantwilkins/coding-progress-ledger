"""Resolve paths into the upstream coding-progress-ledger checkout."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LEDGER_ROOT = Path(__file__).resolve().parents[2].parent / "coding-progress-ledger"
REPO_ROOT = Path(__file__).resolve().parents[2]

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
LOCAL_SOURCE_ROOT_NAMES: dict[str, str] = {
    "tb_live_v2": "tb_live_v2",
    "terminal_bench_pilot": "terminal_bench_pilot",
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
    if source in LOCAL_SOURCE_ROOT_NAMES:
        path = REPO_ROOT / "runs" / LOCAL_SOURCE_ROOT_NAMES[source]
    elif source in SOURCE_ROOT_NAMES:
        path = ledger_root() / "runs" / SOURCE_ROOT_NAMES[source]
    else:
        raise KeyError(f"unknown source: {source}")
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
