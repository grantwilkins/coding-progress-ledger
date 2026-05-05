#!/usr/bin/env bash
set -euo pipefail
cat > fetch_data.py <<'PY'
import sys
import urllib.request

if "--offline" in sys.argv:
    print("OFFLINE_MODE")
    sys.exit(0)
print(urllib.request.urlopen("https://example.invalid/data").read().decode())
PY
