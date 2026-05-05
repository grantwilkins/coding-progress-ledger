"""Verifier for validation_new_work_02_silent_io_format_drift."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "count_users.py"


def _run(jsonl: str, tmp_path: Path) -> str:
    p = tmp_path / "events.jsonl"
    p.write_text(jsonl)
    out = subprocess.run(
        [sys.executable, str(APP), str(p)],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout.strip()


def test_module_exists():
    assert APP.is_file(), f"{APP} not created"


def test_clean_fixture(tmp_path):
    jsonl = textwrap.dedent("""\
        {"user": "alice", "ev": "x"}
        {"user": "bob", "ev": "y"}
        {"user": "alice", "ev": "z"}
    """)
    assert _run(jsonl, tmp_path) == "2"


def test_malformed_line_skipped(tmp_path):
    # Second line is broken JSON; spec says skip silently.
    jsonl = (
        '{"user": "alice"}\n'
        '{not json,}\n'
        '{"user": "bob"}\n'
    )
    assert _run(jsonl, tmp_path) == "2"


def test_missing_user_field_skipped(tmp_path):
    jsonl = (
        '{"user": "alice"}\n'
        '{"actor": "bob"}\n'
        '{"user": "carol"}\n'
    )
    assert _run(jsonl, tmp_path) == "2"


def test_blank_lines_skipped(tmp_path):
    jsonl = (
        '\n'
        '{"user": "alice"}\n'
        '\n'
        '{"user": "alice"}\n'
        '{"user": "bob"}\n'
        '\n'
    )
    assert _run(jsonl, tmp_path) == "2"
