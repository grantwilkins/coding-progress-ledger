#!/usr/bin/env bash
set -euo pipefail
sed -i 's/% 2 == 1/% 2 == 0/' /app/sum_evens.py
