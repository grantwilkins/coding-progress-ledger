#!/usr/bin/env bash
set -euo pipefail
cat > count_recent.py <<'PY'
import sys
from datetime import datetime


def _parse(ts: str):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def main():
    log_path, cutoff_s = sys.argv[1], sys.argv[2]
    cutoff = _parse(cutoff_s)
    n = 0
    with open(log_path) as f:
        for line in f:
            head = line.split(" ", 1)[0].strip()
            if not head:
                continue
            try:
                t = _parse(head)
            except ValueError:
                continue
            if t > cutoff:
                n += 1
    print(n)


if __name__ == "__main__":
    main()
PY
