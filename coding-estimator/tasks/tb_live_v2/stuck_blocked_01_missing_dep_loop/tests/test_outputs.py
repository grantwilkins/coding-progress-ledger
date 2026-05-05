"""Verifier for stuck_blocked_01_missing_dep_loop."""

import os
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))


def test_h1_extracted():
    p = WS / "h1.txt"
    assert p.is_file(), f"{p} was not created"
    assert p.read_text().strip() == "Hello, terminal-bench"
