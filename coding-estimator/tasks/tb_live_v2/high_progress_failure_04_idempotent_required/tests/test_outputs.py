"""Verifier for high_progress_failure_04_idempotent_required."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "apply_migration.py"
STATE = WS / "state.json"

EXPECTED = {
    "version": 2,
    "users": [
        {"name": "alice", "migrated": True},
        {"name": "bob", "migrated": True},
        {"name": "carol", "migrated": True},
    ],
}


def _reset_state():
    STATE.write_text(json.dumps(
        {"users": [{"name": "carol"}, {"name": "alice"}, {"name": "bob"}]}
    ))


def _run_once():
    proc = subprocess.run(
        [sys.executable, str(APP)],
        cwd=str(WS),
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, f"apply_migration.py failed: {proc.stderr}"


def _read_state() -> dict:
    return json.loads(STATE.read_text())


def test_module_exists():
    assert APP.is_file(), f"{APP} not present"


def test_first_run_correct():
    _reset_state()
    _run_once()
    assert _read_state() == EXPECTED


def test_idempotent_run_twice():
    _reset_state()
    _run_once()
    _run_once()
    assert _read_state() == EXPECTED


def test_idempotent_run_three_times():
    _reset_state()
    _run_once()
    _run_once()
    _run_once()
    assert _read_state() == EXPECTED
