#!/usr/bin/env bash
set -euo pipefail
: "${QH_POLICY_RUN_ROOT:?set QH_POLICY_RUN_ROOT}"
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
status=0
run=(uv run python queue-haul/migration_profiler.py run --plan "$script_dir/plan.json" --run-root "$QH_POLICY_RUN_ROOT" --fail-fast --stack-scenarios 30)
[[ -z "${QH_RESUME_FROM_GIT_SHA:-}" ]] || run+=(--resume-from-git-sha "$QH_RESUME_FROM_GIT_SHA")
"${run[@]}" || status=$?
[[ -f "$QH_POLICY_RUN_ROOT/plan.json" ]] || exit "$status"
uv run python queue-haul/policy_hardware_campaign.py reduce   --run-root "$QH_POLICY_RUN_ROOT"
exit "$status"
