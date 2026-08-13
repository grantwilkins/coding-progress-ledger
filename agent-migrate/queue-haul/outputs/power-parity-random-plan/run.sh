#!/usr/bin/env bash
set -euo pipefail
: "${QH_POWER_PARITY_RUN_ROOT:?set QH_POWER_PARITY_RUN_ROOT}"
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
uv run python queue-haul/power_parity_experiment.py run --plan "$script_dir/plan.json" --run-root "$QH_POWER_PARITY_RUN_ROOT" --stack-scenarios 28 --fail-fast
uv run python queue-haul/power_parity_experiment.py reduce --run-root "$QH_POWER_PARITY_RUN_ROOT"
