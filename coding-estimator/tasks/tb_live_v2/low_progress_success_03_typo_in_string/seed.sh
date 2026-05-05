#!/usr/bin/env bash
set -euo pipefail
cat > greet.py <<'PY'
def greet(name):
    return f"Hello, terminl-bench, {name}!"
PY
