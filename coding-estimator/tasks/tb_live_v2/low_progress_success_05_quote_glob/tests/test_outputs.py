"""Verifier for low_progress_success_05_quote_glob."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
BUILD = WS / "build.sh"
OUT = WS / "dist" / "out.txt"


def test_build_sh_exists():
    assert BUILD.is_file(), f"{BUILD} missing"


def test_txt_files_unchanged():
    assert (WS / "a.txt").is_file()
    assert (WS / "b.txt").is_file()


def test_build_sh_runs_and_writes_two_lines():
    if (WS / "dist").exists():
        shutil.rmtree(WS / "dist")
    proc = subprocess.run(
        ["./build.sh"], cwd=str(WS),
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, f"./build.sh failed: {proc.stderr}"
    lines = OUT.read_text().splitlines()
    assert lines == ["found: a.txt", "found: b.txt"], f"got: {lines!r}"
