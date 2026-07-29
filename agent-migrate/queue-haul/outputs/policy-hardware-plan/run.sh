#!/usr/bin/env bash
set -euo pipefail
: "${QH_POLICY_RUN_ROOT:?set QH_POLICY_RUN_ROOT}"
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
status=0
uv run python queue-haul/migration_profiler.py run --plan queue-haul/outputs/policy-hardware-plan/plan.json --run-root "$QH_POLICY_RUN_ROOT" || status=$?
uv run python queue-haul/policy_hardware_campaign.py reduce --run-root "$QH_POLICY_RUN_ROOT"
exit "$status"
