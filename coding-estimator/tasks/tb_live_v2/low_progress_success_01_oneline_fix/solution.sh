#!/usr/bin/env bash
# Oracle solution. Operates on cwd-relative paths so it works on host
# (cwd = workspace) and in Docker (cwd = /app via WORKDIR).
set -euo pipefail
python -c "import pathlib;p=pathlib.Path('sum_evens.py');p.write_text(p.read_text().replace('% 2 == 1','% 2 == 0'))"
