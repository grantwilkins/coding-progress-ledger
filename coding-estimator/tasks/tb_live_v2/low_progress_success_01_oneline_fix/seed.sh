#!/usr/bin/env bash
set -euo pipefail
cat > sum_evens.py <<'PY'
import sys
xs = [int(t) for t in sys.stdin.read().strip().split(",") if t]
print(sum(x for x in xs if x % 2 == 1))
PY
