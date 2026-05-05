"""Verifier for low_progress_success_01_oneline_fix."""

import subprocess


def _run(stdin: str) -> str:
    out = subprocess.run(
        ["python", "/app/sum_evens.py"],
        input=stdin, capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout.strip()


def test_basic_evens():
    assert _run("1,2,3,4") == "6"


def test_all_odd_returns_zero():
    assert _run("1,3,5,7") == "0"


def test_negatives():
    # -2 is even.
    assert _run("-1,-2,-3,-4") == "-6"


def test_empty_input_is_zero():
    assert _run("") == "0"
