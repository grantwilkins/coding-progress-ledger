#!/usr/bin/env bash
set -euo pipefail
python3 -c "import pathlib;p=pathlib.Path('greet.py');p.write_text(p.read_text().replace('terminl-bench','terminal-bench'))"
