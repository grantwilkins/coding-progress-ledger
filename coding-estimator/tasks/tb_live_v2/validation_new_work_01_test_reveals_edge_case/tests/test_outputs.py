"""Verifier for validation_new_work_01_test_reveals_edge_case."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

APP = Path("/app/days_until.py")


def _load():
    assert APP.is_file(), f"{APP} not created"
    spec = importlib.util.spec_from_file_location("days_until", APP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["days_until"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_basic_week():
    m = _load()
    assert m.days_until("2026-02-01", "2026-02-08") == 7


def test_leap_year_feb_to_march():
    m = _load()
    # 2024 is a leap year; Feb has 29 days
    assert m.days_until("2024-02-01", "2024-03-01") == 29


def test_non_leap_year_feb_to_march():
    m = _load()
    assert m.days_until("2026-02-01", "2026-03-01") == 28


def test_rejects_short_format():
    m = _load()
    with pytest.raises(ValueError):
        m.days_until("2026-02", "2026-03")


def test_rejects_invalid_month():
    m = _load()
    with pytest.raises(ValueError):
        m.days_until("2026-13-01", "2026-12-01")


def test_rejects_invalid_day():
    m = _load()
    with pytest.raises(ValueError):
        m.days_until("2026-02-30", "2026-03-01")
