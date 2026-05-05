#!/usr/bin/env bash
set -euo pipefail
cat > fetch_data.py <<'PY'
import urllib.request
print(urllib.request.urlopen("https://example.invalid/data").read().decode())
PY
