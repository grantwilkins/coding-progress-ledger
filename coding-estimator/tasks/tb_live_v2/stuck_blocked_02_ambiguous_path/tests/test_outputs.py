"""Verifier for stuck_blocked_02_ambiguous_path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "count_words.py"


def _run() -> str:
    out = subprocess.run(
        [sys.executable, str(APP)],
        cwd=str(WS),
        capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout.strip()


def test_module_exists():
    assert APP.is_file(), f"{APP} not present"


def test_data_files_intact():
    assert (WS / "reference/data.txt").read_text().strip() == \
        "the quick brown fox jumps over the lazy dog"
    # 'data/data.txt' must NOT be modified by the agent.
    assert (WS / "data/data.txt").read_text().strip() == "wrong"


def test_word_count_from_reference():
    assert _run() == "9"
