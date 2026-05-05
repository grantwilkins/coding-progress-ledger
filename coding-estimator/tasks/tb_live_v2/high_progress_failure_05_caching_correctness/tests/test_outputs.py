"""Verifier for high_progress_failure_05_caching_correctness."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "lookup.py"


def _load():
    if str(WS) not in sys.path:
        sys.path.insert(0, str(WS))
    spec = importlib.util.spec_from_file_location("lookup", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_exists():
    assert APP.is_file(), f"{APP} not present"


def test_basic_hit_and_miss():
    f = _load().lookup
    table = {"alpha": "A", "beta": "B"}
    assert f(table, "alpha") == "A"
    assert f(table, "ALPHA") == "A"  # case-insensitive query
    assert f(table, "gamma") == "NONE"


def test_mutation_in_place_reflected():
    """Strict: mutating the SAME table dict between calls must be observed."""
    f = _load().lookup
    table = {"alpha": "A"}
    assert f(table, "alpha") == "A"
    # Mutate the same dict:
    table["alpha"] = "A2"
    assert f(table, "alpha") == "A2"
    # Add a new key:
    table["beta"] = "B"
    assert f(table, "beta") == "B"
    # Remove a key:
    del table["alpha"]
    assert f(table, "alpha") == "NONE"


def test_different_table_objects_distinguished():
    f = _load().lookup
    t1 = {"x": "X1"}
    t2 = {"x": "X2"}
    # Interleave to make sure no global cache crosses tables.
    assert f(t1, "x") == "X1"
    assert f(t2, "x") == "X2"
    assert f(t1, "x") == "X1"


def test_repeat_calls_same_table_correct():
    f = _load().lookup
    table = {"k": "v"}
    for _ in range(50):
        assert f(table, "k") == "v"
        assert f(table, "missing") == "NONE"


def test_empty_table():
    f = _load().lookup
    assert f({}, "anything") == "NONE"
