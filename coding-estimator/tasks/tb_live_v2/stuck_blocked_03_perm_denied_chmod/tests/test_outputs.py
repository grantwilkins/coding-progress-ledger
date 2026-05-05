"""Verifier for stuck_blocked_03_perm_denied_chmod."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
SCRIPT = WS / "build.sh"
OUT = WS / "dist" / "output.txt"


def test_script_exists():
    assert SCRIPT.is_file(), f"{SCRIPT} not present"


def test_script_is_executable():
    mode = SCRIPT.stat().st_mode & 0o777
    assert mode & 0o100, f"build.sh must be owner-executable; got mode {oct(mode)}"


def test_dot_slash_run_succeeds_and_writes_output():
    # Remove any prior output so this test really proves the run worked.
    if OUT.is_file():
        OUT.unlink()
    proc = subprocess.run(
        ["./build.sh"], cwd=str(WS),
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, f"./build.sh failed: {proc.stderr}"
    assert OUT.is_file(), f"{OUT} not produced"
    assert OUT.read_text().strip() == "build complete"
