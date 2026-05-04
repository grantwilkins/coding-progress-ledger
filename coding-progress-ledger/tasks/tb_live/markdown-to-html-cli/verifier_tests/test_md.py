import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
NAMES = sorted(p.stem for p in FIXTURES.glob("*.md"))


@pytest.mark.parametrize("name", NAMES)
def test_convert_api(name):
    from md2html import convert
    md = (FIXTURES / f"{name}.md").read_text()
    expected = (FIXTURES / f"{name}.html").read_text()
    assert convert(md).strip() == expected.strip()


def test_cli_reads_file(tmp_path):
    md = "# CLI Heading\n"
    inp = tmp_path / "in.md"
    inp.write_text(md)
    result = subprocess.run(
        [sys.executable, "-m", "md2html", str(inp)],
        capture_output=True, text=True, check=True,
    )
    assert "<h1>CLI Heading</h1>" in result.stdout


def test_cli_reads_stdin():
    result = subprocess.run(
        [sys.executable, "-m", "md2html"],
        input="# Stdin Heading\n", capture_output=True, text=True, check=True,
    )
    assert "<h1>Stdin Heading</h1>" in result.stdout
