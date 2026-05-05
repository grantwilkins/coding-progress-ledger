#!/usr/bin/env bash
set -euo pipefail
# Activate venv if present (driver creates one when requirements.txt exists);
# otherwise fall back to system pip into the workspace's user site.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
pip install --no-input --quiet beautifulsoup4
python scrape.py
