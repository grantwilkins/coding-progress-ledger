"""Verifier for validation_new_work_05_quoted_field_in_tsv."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "tsv_to_json.py"


def _run(tsv_text: str, tmp_path: Path) -> list:
    p = tmp_path / "f.tsv"
    p.write_text(tsv_text)
    out = subprocess.run(
        [sys.executable, str(APP), str(p)],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return json.loads(out.stdout)


def test_module_exists():
    assert APP.is_file(), f"{APP} not created"


def test_basic_two_columns(tmp_path):
    tsv = "name\tage\nAlice\t30\nBob\t25\n"
    assert _run(tsv, tmp_path) == [
        {"name": "Alice", "age": "30"},
        {"name": "Bob", "age": "25"},
    ]


def test_quoted_tab_in_field(tmp_path):
    # Second row's "desc" is a quoted field containing a literal tab.
    # Naive split-on-tab would split it into two and corrupt the row.
    tsv = 'k\tdesc\trev\nx\t"a\tb"\t1\ny\tplain\t2\n'
    got = _run(tsv, tmp_path)
    assert got == [
        {"k": "x", "desc": "a\tb", "rev": "1"},
        {"k": "y", "desc": "plain", "rev": "2"},
    ]


def test_blank_line_skipped(tmp_path):
    tsv = "a\tb\n1\t2\n\n3\t4\n"
    assert _run(tsv, tmp_path) == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]
