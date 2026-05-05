"""Verifier for stuck_blocked_04_module_not_found_pythonpath."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
RUN = WS / "run.sh"
WIDGET = WS / "lib" / "widget.py"
RENDER = WS / "tools" / "render.py"
OUT = WS / "out" / "result.txt"


def test_layout_unchanged():
    # Constraints: widget stays in lib/, no __init__.py added.
    assert WIDGET.is_file(), "lib/widget.py was moved or deleted"
    assert RENDER.is_file(), "tools/render.py was moved or deleted"
    assert not (WS / "widget.py").exists(), "widget.py copied to root"
    assert not (WS / "lib" / "__init__.py").exists(), "__init__.py added to lib/"
    assert not (WS / "tools" / "__init__.py").exists(), "__init__.py added to tools/"


def test_run_sh_succeeds_and_writes_output():
    if OUT.is_file():
        OUT.unlink()
    proc = subprocess.run(
        ["./run.sh"], cwd=str(WS),
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, f"./run.sh failed: {proc.stderr}"
    assert OUT.is_file(), f"{OUT} not produced"
    assert OUT.read_text().strip() == "rendered: hello"


def test_no_pip_marker():
    # No requirements.txt or pip install in run.sh.
    assert not (WS / "requirements.txt").is_file(), "requirements.txt should not be present"
    text = RUN.read_text()
    assert "pip install" not in text, "run.sh must not pip-install anything"
