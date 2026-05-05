#!/usr/bin/env bash
set -euo pipefail
mkdir -p data reference
echo "wrong" > data/data.txt
printf "the quick brown fox jumps over the lazy dog\n" > reference/data.txt
cat > count_words.py <<'PY'
from pathlib import Path
text = Path("data.txt").read_text()
print(len(text.split()))
PY
