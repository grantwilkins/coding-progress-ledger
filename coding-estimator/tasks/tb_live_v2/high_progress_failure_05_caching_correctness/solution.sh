#!/usr/bin/env bash
set -euo pipefail
cat > lookup.py <<'PY'
def lookup(table, query):
    return table.get(query.lower(), "NONE")
PY
