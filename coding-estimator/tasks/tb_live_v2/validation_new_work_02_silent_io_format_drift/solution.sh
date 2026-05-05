#!/usr/bin/env bash
set -euo pipefail
cat > count_users.py <<'PY'
import json, sys

users = set()
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and "user" in obj:
            users.add(obj["user"])
print(len(users))
PY
