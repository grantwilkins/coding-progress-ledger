"""Verifier for validation_new_work_04_tz_offset_in_log."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "count_recent.py"


def _run(log_text: str, cutoff: str, tmp_path: Path) -> str:
    p = tmp_path / "log.txt"
    p.write_text(log_text)
    out = subprocess.run(
        [sys.executable, str(APP), str(p), cutoff],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout.strip()


def test_module_exists():
    assert APP.is_file(), f"{APP} not present"


def test_zulu_only(tmp_path):
    log = textwrap.dedent("""\
        2026-03-01T12:00:00Z hello
        2026-03-01T13:00:00Z world
    """)
    # Cutoff at 12:30Z → only the 13:00Z line counts.
    assert _run(log, "2026-03-01T12:30:00Z", tmp_path) == "1"


def test_offset_lines(tmp_path):
    # 14:00+02:00 == 12:00 UTC
    # 15:30-05:00 == 20:30 UTC
    log = textwrap.dedent("""\
        2026-03-01T14:00:00+02:00 europe
        2026-03-01T15:30:00-05:00 newyork
    """)
    # Cutoff at 18:00 UTC → only the newyork line (20:30 UTC) counts.
    assert _run(log, "2026-03-01T18:00:00Z", tmp_path) == "1"


def test_mixed_formats(tmp_path):
    log = textwrap.dedent("""\
        2026-03-01T12:00:00Z          a
        2026-03-01T13:00:00+00:00     b
        2026-03-01T14:00:00+02:00     c
        2026-03-01T15:30:00-05:00     d
    """)
    # Cutoff 12:30 UTC. UTC-equivalent times of each:
    #   a=12:00, b=13:00, c=12:00, d=20:30 → 2 strictly after.
    assert _run(log, "2026-03-01T12:30:00Z", tmp_path) == "2"


def test_malformed_lines_skipped(tmp_path):
    log = (
        "2026-03-01T12:00:00Z keep\n"
        "garbage line no timestamp\n"
        "2026-03-01T13:00:00Z keep\n"
    )
    assert _run(log, "2026-03-01T11:00:00Z", tmp_path) == "2"
