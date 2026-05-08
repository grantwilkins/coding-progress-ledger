#!/usr/bin/env bash
set -euo pipefail
python3 -c "import pathlib;p=pathlib.Path('Makefile');p.write_text(p.read_text().replace('bulid','build'))"
make build
