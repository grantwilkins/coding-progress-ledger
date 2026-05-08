"""Verifier for validation_new_work_03_unicode_normalization."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "search.py"

# U+00E9 is composed é; "é" is decomposed (e + combining acute U+0301).
COMPOSED_E_ACUTE = "é"
DECOMPOSED_E_ACUTE = "é"


def _load():
    if str(WS) not in sys.path:
        sys.path.insert(0, str(WS))
    spec = importlib.util.spec_from_file_location("search", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_exists():
    assert APP.is_file()


def test_basic_ascii_match():
    f = _load().matches
    assert f("hello world", "world") is True
    assert f("HELLO", "hello") is True
    assert f("foo", "bar") is False


def test_composed_haystack_decomposed_needle():
    f = _load().matches
    haystack = "Welcome to Caf" + COMPOSED_E_ACUTE + " Bar"
    needle = "Caf" + DECOMPOSED_E_ACUTE
    assert f(haystack, needle) is True


def test_decomposed_haystack_composed_needle():
    f = _load().matches
    haystack = "Welcome to Caf" + DECOMPOSED_E_ACUTE + " Bar"
    needle = "Caf" + COMPOSED_E_ACUTE
    assert f(haystack, needle) is True


def test_no_match_when_actually_different():
    f = _load().matches
    assert f("Caf" + COMPOSED_E_ACUTE, "Tea") is False


def test_empty_needle_always_matches():
    f = _load().matches
    assert f("anything", "") is True
