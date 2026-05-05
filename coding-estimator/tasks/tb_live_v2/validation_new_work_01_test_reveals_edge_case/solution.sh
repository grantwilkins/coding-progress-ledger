#!/usr/bin/env bash
set -euo pipefail
cat > days_until.py <<'PY'
import re
from datetime import date

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _parse(s: str) -> date:
    if not isinstance(s, str) or not ISO_RE.match(s):
        raise ValueError(f"not ISO-8601 YYYY-MM-DD: {s!r}")
    y, m, d = (int(p) for p in s.split("-"))
    return date(y, m, d)

def days_until(start_iso: str, end_iso: str) -> int:
    start = _parse(start_iso)
    end = _parse(end_iso)
    return (end - start).days
PY
