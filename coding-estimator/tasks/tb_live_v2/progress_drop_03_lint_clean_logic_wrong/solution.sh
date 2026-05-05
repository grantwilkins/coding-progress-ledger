#!/usr/bin/env bash
set -euo pipefail
cat > sliding_mean.py <<'PY'
import json, sys


def sliding_mean(xs, window):
    if window <= 0:
        raise ValueError(window)
    n = len(xs)
    if n == 0 or window > n:
        return []
    return [sum(xs[i:i + window]) / window for i in range(n - window + 1)]


if __name__ == "__main__":
    w = int(sys.argv[1])
    xs = [float(x) for x in sys.argv[2:]]
    print(json.dumps(sliding_mean(xs, w)))
PY
