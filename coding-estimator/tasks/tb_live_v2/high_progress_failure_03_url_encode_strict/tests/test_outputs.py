"""Verifier for high_progress_failure_03_url_encode_strict."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "url_encode.py"


def _load():
    if str(WS) not in sys.path:
        sys.path.insert(0, str(WS))
    spec = importlib.util.spec_from_file_location("url_encode", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_exists():
    assert APP.is_file()


def test_no_urllib_quote_used():
    text = APP.read_text()
    # Reject any direct `quote`/`quote_plus`/`urlencode` reference.
    forbidden = [
        "urllib.parse.quote",
        "from urllib.parse import quote",
        "quote_plus",
        "urlencode",
        "quote(",
    ]
    for needle in forbidden:
        assert needle not in text, f"forbidden API used: {needle!r}"


def test_basic_unreserved_passthrough():
    f = _load().url_encode
    assert f("hello") == "hello"
    assert f("Hello-World_2024.txt~") == "Hello-World_2024.txt~"


def test_space_encoded_percent_20():
    f = _load().url_encode
    assert f("a b c") == "a%20b%20c"
    # Plus form is wrong — that's quote_plus semantics:
    assert "+" not in f("a b c")


def test_reserved_chars_encoded_uppercase_hex():
    f = _load().url_encode
    # "/" is %2F (reserved gen-delim).
    assert f("a/b") == "a%2Fb"
    # ":" is %3A.
    assert f("a:b") == "a%3Ab"
    # "?" is %3F.
    assert f("a?b") == "a%3Fb"
    # Hex must be UPPERCASE, never lowercase:
    enc = f("a/b")
    assert re.match(r"^[A-Za-z0-9\-._~%A-F0-9]+$", enc)
    # Specifically catch lowercase by looking at all %XX bytes:
    for m in re.finditer(r"%([0-9a-fA-F]{2})", enc):
        assert m.group(1) == m.group(1).upper(), f"lowercase hex: {m.group(0)!r}"


def test_unicode_utf8_encoded():
    f = _load().url_encode
    # é = U+00E9, UTF-8 = c3 a9
    assert f("é") == "%C3%A9"
    # café:  c-a-f then %C3%A9
    assert f("café") == "caf%C3%A9"


def test_empty_string():
    f = _load().url_encode
    assert f("") == ""
