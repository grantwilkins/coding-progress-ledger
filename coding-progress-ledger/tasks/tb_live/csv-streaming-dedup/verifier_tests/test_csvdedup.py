import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

CASES = [
    "basic_dups",
    "no_dups",
    "header_only",
    "quoted_fields",
    "all_same",
    "multicolumn",
]


@pytest.mark.parametrize("name", CASES)
def test_file_arg(name):
    inp = FIXTURES / f"{name}.csv"
    expected = (FIXTURES / f"{name}_expected.csv").read_text()
    result = subprocess.run(
        [sys.executable, "-m", "csvdedup", str(inp)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == expected


def test_stdin_mode():
    inp = (FIXTURES / "basic_dups.csv").read_text()
    expected = (FIXTURES / "basic_dups_expected.csv").read_text()
    result = subprocess.run(
        [sys.executable, "-m", "csvdedup"],
        input=inp,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == expected
