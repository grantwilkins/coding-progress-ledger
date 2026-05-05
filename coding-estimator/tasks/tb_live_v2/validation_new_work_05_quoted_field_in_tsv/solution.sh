#!/usr/bin/env bash
set -euo pipefail
cat > tsv_to_json.py <<'PY'
import csv, json, sys
with open(sys.argv[1], newline="") as f:
    r = csv.reader(f, delimiter="\t", quotechar='"')
    rows = [row for row in r if row]
header = rows[0]
out = [dict(zip(header, row)) for row in rows[1:]]
print(json.dumps(out))
PY
