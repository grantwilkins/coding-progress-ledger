#!/usr/bin/env bash
set -euo pipefail
cat > search.py <<'PY'
import unicodedata


def matches(haystack: str, needle: str) -> bool:
    h = unicodedata.normalize("NFC", haystack).lower()
    n = unicodedata.normalize("NFC", needle).lower()
    return n in h
PY
