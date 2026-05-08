"""Verifier for progress_drop_02_currency_format_thousands."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "format_amount.py"


def _load():
    if str(WS) not in sys.path:
        sys.path.insert(0, str(WS))
    spec = importlib.util.spec_from_file_location("format_amount", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_exists():
    assert APP.is_file()


def test_zero():
    assert _load().format_amount(0) == "$0.00"


def test_small_positive():
    assert _load().format_amount(150) == "$1.50"
    assert _load().format_amount(99) == "$0.99"
    assert _load().format_amount(1) == "$0.01"


def test_thousands_separator_required():
    f = _load().format_amount
    assert f(123456) == "$1,234.56"
    assert f(1234567) == "$12,345.67"
    assert f(123456789) == "$1,234,567.89"


def test_negative_sign_before_dollar():
    f = _load().format_amount
    assert f(-150) == "-$1.50"
    assert f(-1234567) == "-$12,345.67"
    assert f(-99) == "-$0.99"


def test_no_float_drift():
    f = _load().format_amount
    # If anyone divided cents/100 as float, this case fails:
    # 0.1 + 0.2 != 0.3 in float, etc.
    assert f(30) == "$0.30"
    assert f(10) == "$0.10"
    assert f(20) == "$0.20"
    # Larger values that would round-trip wrong via float:
    assert f(99999999) == "$999,999.99"
