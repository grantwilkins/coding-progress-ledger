"""Verifier for progress_drop_01_lint_then_runtime_failure.

Pass criterion: csv_summary.py exists in the workspace, is invokable as
`python csv_summary.py <path>`, and produces the exact spec'd output
on three CSV fixtures including the empty-string and quoted-comma traps.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "csv_summary.py"


def _run(csv_text: str, tmp_path: Path) -> str:
    p = tmp_path / "f.csv"
    p.write_text(csv_text)
    out = subprocess.run(
        [sys.executable, str(APP), str(p)],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout.strip()


def test_module_exists():
    assert APP.is_file(), f"{APP} not created"


def test_uniform_numeric(tmp_path):
    csv_text = textwrap.dedent("""\
        a,b
        1.0,2.5
        3.0,4.5
    """)
    assert _run(csv_text, tmp_path) == "rows=2 cols=2 numeric_cols=2"


def test_empty_string_is_not_numeric(tmp_path):
    csv_text = textwrap.dedent("""\
        a,b
        1.0,
        2.0,3.0
    """)
    assert _run(csv_text, tmp_path) == "rows=2 cols=2 numeric_cols=1"


def test_quoted_comma_is_one_cell(tmp_path):
    csv_text = 'a,b\n"hello, world",1.0\n"goodbye, world",2.0\n'
    assert _run(csv_text, tmp_path) == "rows=2 cols=2 numeric_cols=1"
