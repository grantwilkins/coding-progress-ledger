#!/usr/bin/env bash
set -euo pipefail
cat > format_amount.py <<'PY'
def format_amount(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    n = abs(cents)
    dollars, c = divmod(n, 100)
    return f"{sign}${dollars:,}.{c:02d}"
PY
