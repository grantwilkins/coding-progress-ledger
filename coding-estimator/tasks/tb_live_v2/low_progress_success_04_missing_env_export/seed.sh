#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
echo "x,y" > data/a.csv
echo "p,q" > data/b.csv
touch data/notes.txt
cat > report.py <<'PY'
import os, pathlib, sys
d = os.environ.get("DATA_DIR")
if not d:
    print("DATA_DIR not set", file=sys.stderr)
    sys.exit(1)
n = sum(1 for p in pathlib.Path(d).iterdir() if p.suffix == ".csv")
print(f"csv_count={n}")
PY
: > .env
