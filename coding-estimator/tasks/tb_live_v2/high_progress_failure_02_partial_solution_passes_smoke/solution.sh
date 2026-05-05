#!/usr/bin/env bash
set -euo pipefail
cat > parse_duration.py <<'PY'
import re, sys

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
ORDER = ["d", "h", "m", "s"]


def parse_duration(s):
    s2 = "".join(s.split())
    if s2 == "0":
        return 0
    if not s2:
        raise ValueError(s)
    parts = re.findall(r"(\d+)([smhd])", s2)
    if not parts:
        raise ValueError(s)
    if "".join(n + u for n, u in parts) != s2:
        raise ValueError(s)
    seen = []
    total = 0
    for n, u in parts:
        if u in seen:
            raise ValueError(s)
        if seen and ORDER.index(u) <= ORDER.index(seen[-1]):
            raise ValueError(s)
        seen.append(u)
        total += int(n) * UNITS[u]
    return total


if __name__ == "__main__":
    print(parse_duration(sys.argv[1]))
PY
