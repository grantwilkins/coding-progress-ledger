"""Verifier for low_progress_success_03_typo_in_string."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "greet.py"


def _load():
    spec = importlib.util.spec_from_file_location("greet", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_exists():
    assert APP.is_file(), f"{APP} not present"


def test_greet_alice():
    assert _load().greet("Alice") == "Hello, terminal-bench, Alice!"


def test_greet_empty_name():
    assert _load().greet("") == "Hello, terminal-bench, !"


def test_greet_bob():
    assert _load().greet("Bob") == "Hello, terminal-bench, Bob!"
