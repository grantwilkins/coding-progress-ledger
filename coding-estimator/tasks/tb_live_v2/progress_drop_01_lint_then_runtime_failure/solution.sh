#!/usr/bin/env bash
# Oracle: write csv_summary.py to cwd. Works on host and in Docker.
set -euo pipefail
cat > csv_summary.py <<'PY'
import csv, sys

def is_number(s: str) -> bool:
    if s == "":
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

def main(path: str) -> int:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print("rows=0 cols=0 numeric_cols=0")
        return 0
    cols = len(rows[0])
    body = rows[1:] if len(rows) > 1 else []
    numeric_cols = sum(
        1 for c in range(cols)
        if body and all(is_number(r[c]) for r in body if c < len(r))
    )
    print(f"rows={len(body)} cols={cols} numeric_cols={numeric_cols}")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
PY
