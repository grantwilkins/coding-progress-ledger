"""Verifier for low_progress_success_01_oneline_fix.

Runs the workspace's `sum_evens.py`. The verifier path is relative to
cwd (the workspace), so the same verifier works on host (cwd=workspace)
and in Docker (cwd=/app via run-tests.sh).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP = (Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
       / "sum_evens.py")


def _run(stdin: str) -> str:
    out = subprocess.run(
        [sys.executable, str(APP)],
        input=stdin, capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout.strip()


def test_app_exists():
    assert APP.is_file(), f"{APP} not present in workspace"


def test_basic_evens():
    assert _run("1,2,3,4") == "6"


def test_all_odd_returns_zero():
    assert _run("1,3,5,7") == "0"


def test_negatives():
    # -2 is even.
    assert _run("-1,-2,-3,-4") == "-6"


def test_empty_input_is_zero():
    assert _run("") == "0"
