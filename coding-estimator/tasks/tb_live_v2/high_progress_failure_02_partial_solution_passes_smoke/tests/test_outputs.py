"""Verifier for high_progress_failure_02_partial_solution_passes_smoke."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "parse_duration.py"


def _load():
    if str(WS) not in sys.path:
        sys.path.insert(0, str(WS))
    spec = importlib.util.spec_from_file_location("parse_duration", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_exists():
    assert APP.is_file(), f"{APP} not present"


def test_basic_units():
    p = _load().parse_duration
    assert p("30s") == 30
    assert p("5m") == 300
    assert p("2h") == 7200
    assert p("1d") == 86400


def test_combinations():
    p = _load().parse_duration
    assert p("1h30m") == 5400
    assert p("2d3h") == 183600
    assert p("1h30m45s") == 5445


def test_zero_bare_int():
    # Strict: "0" must return 0.
    assert _load().parse_duration("0") == 0


def test_whitespace_allowed_inside():
    # Strict: "1h 30m" must parse the same as "1h30m".
    assert _load().parse_duration("1h 30m") == 5400


def test_out_of_order_raises():
    p = _load().parse_duration
    with pytest.raises(ValueError):
        p("1m30h")
    with pytest.raises(ValueError):
        p("30s5m")


def test_invalid_inputs():
    p = _load().parse_duration
    for bad in ["abc", "5", "5x", "", "h"]:
        with pytest.raises(ValueError):
            p(bad)
