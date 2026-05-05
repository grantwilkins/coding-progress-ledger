"""Verifier for low_progress_success_02_config_flag_decisive."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "fetch_data.py"


def test_module_exists():
    assert APP.is_file(), f"{APP} not present"


def test_offline_flag_prints_marker():
    out = subprocess.run(
        [sys.executable, str(APP), "--offline"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert out.stdout.strip() == "OFFLINE_MODE"


def test_offline_flag_exit_zero():
    proc = subprocess.run(
        [sys.executable, str(APP), "--offline"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0
