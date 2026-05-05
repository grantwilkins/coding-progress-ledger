#!/usr/bin/env bash
set -euo pipefail
pip install --no-input beautifulsoup4
python /app/scrape.py
