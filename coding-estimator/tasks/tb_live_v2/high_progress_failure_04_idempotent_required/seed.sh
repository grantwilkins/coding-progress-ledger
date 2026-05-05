#!/usr/bin/env bash
set -euo pipefail
cat > state.json <<'JSON'
{"users": [{"name": "carol"}, {"name": "alice"}, {"name": "bob"}]}
JSON
