#!/usr/bin/env bash
set -euo pipefail
python3 -c "import pathlib;p=pathlib.Path('count_words.py');p.write_text(p.read_text().replace('\"data.txt\"','\"reference/data.txt\"'))"
