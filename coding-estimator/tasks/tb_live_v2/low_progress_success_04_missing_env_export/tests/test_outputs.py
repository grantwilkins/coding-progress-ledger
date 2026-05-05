"""Verifier for low_progress_success_04_missing_env_export."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
ENV = WS / ".env"
APP = WS / "report.py"


def _parse_env() -> dict[str, str]:
    env = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            raise ValueError(f"convention violated: {line!r} uses 'export'")
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if not m:
            raise ValueError(f"convention violated: {line!r} not KEY=VALUE")
        k, v = m.group(1), m.group(2)
        if v.startswith('"') or v.startswith("'"):
            raise ValueError(f"convention violated: {line!r} is quoted")
        env[k] = v
    return env


def test_env_file_exists():
    assert ENV.is_file(), ".env not present"


def test_env_has_data_dir():
    env = _parse_env()
    assert "DATA_DIR" in env, ".env missing DATA_DIR"
    assert env["DATA_DIR"] == "data"


def test_report_unchanged_runs_with_loaded_env():
    env = _parse_env()
    proc = subprocess.run(
        [sys.executable, str(APP)],
        cwd=str(WS),
        env={**os.environ, **env},
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert proc.stdout.strip() == "csv_count=2"
