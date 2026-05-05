"""Verifier for stuck_blocked_01_missing_dep_loop."""

from pathlib import Path


def test_h1_extracted():
    p = Path("/app/h1.txt")
    assert p.is_file(), "/app/h1.txt was not created"
    assert p.read_text().strip() == "Hello, terminal-bench"
