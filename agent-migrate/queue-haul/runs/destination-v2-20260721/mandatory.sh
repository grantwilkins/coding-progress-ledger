#!/usr/bin/env bash
set -euo pipefail
uv run python queue-haul/destination_runner.py --plan "$QH_CAMPAIGN_PLAN" --run-root "$QH_RUN_ROOT"
