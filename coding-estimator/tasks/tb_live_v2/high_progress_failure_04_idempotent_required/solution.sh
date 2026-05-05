#!/usr/bin/env bash
set -euo pipefail
cat > apply_migration.py <<'PY'
import json
from pathlib import Path

p = Path("state.json")
state = json.loads(p.read_text())
state["version"] = 2
state["users"] = sorted(
    [{**u, "migrated": True} for u in state["users"]],
    key=lambda u: u["name"],
)
p.write_text(json.dumps(state))
PY
