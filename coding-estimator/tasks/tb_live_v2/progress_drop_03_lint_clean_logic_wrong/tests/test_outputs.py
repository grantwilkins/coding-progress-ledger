"""Verifier for progress_drop_03_lint_clean_logic_wrong."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "sliding_mean.py"


def _load():
    if str(WS) not in sys.path:
        sys.path.insert(0, str(WS))
    spec = importlib.util.spec_from_file_location("sliding_mean", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_exists():
    assert APP.is_file(), f"{APP} not created"


def test_basic_window_3():
    f = _load().sliding_mean
    assert f([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]


def test_window_equals_len():
    f = _load().sliding_mean
    got = f([1, 2, 3, 4], 4)
    assert got == [2.5]


def test_window_one():
    f = _load().sliding_mean
    assert f([1, 2, 3], 1) == [1.0, 2.0, 3.0]


def test_window_greater_than_len_returns_empty():
    f = _load().sliding_mean
    assert f([1, 2], 5) == []


def test_empty_input_returns_empty():
    f = _load().sliding_mean
    assert f([], 3) == []


def test_zero_window_raises():
    f = _load().sliding_mean
    with pytest.raises(ValueError):
        f([1, 2, 3], 0)


def test_negative_window_raises():
    f = _load().sliding_mean
    with pytest.raises(ValueError):
        f([1, 2, 3], -1)
