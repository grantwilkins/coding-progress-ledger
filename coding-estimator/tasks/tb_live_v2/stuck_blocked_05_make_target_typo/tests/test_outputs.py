"""Verifier for stuck_blocked_05_make_target_typo."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
MAKEFILE = WS / "Makefile"
OUT = WS / "dist" / "output.txt"


def test_makefile_present():
    assert MAKEFILE.is_file()


def test_make_build_succeeds_and_writes_output():
    if (WS / "dist").exists():
        shutil.rmtree(WS / "dist")
    proc = subprocess.run(
        ["make", "build"], cwd=str(WS),
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, f"make build failed: {proc.stderr}"
    assert OUT.is_file(), f"{OUT} not produced"
    assert OUT.read_text().strip() == "built ok"
