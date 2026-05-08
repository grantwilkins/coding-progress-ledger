#!/usr/bin/env bash
set -euo pipefail
python3 -c "import pathlib;p=pathlib.Path('build.sh');p.write_text(p.read_text().replace(\"'*.txt'\",'*.txt'))"
./build.sh
