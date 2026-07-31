#!/usr/bin/env bash
set -euo pipefail
: "${QH_MECHANISM_RUN_ROOT:?set QH_MECHANISM_RUN_ROOT}"
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
status=0
command=(uv run python queue-haul/migration_profiler.py run --plan "$script_dir/plan.json" --run-root "$QH_MECHANISM_RUN_ROOT" --stack-scenarios 15)
[[ -z "${QH_RESUME_FROM_GIT_SHA:-}" ]] || command+=(--resume-from-git-sha "$QH_RESUME_FROM_GIT_SHA")
"${command[@]}" || status=$?
[[ -f "$QH_MECHANISM_RUN_ROOT/plan.json" ]] || exit "$status"
uv run python queue-haul/mechanism_validation_campaign.py reduce --run-root "$QH_MECHANISM_RUN_ROOT" --out queue-haul/outputs/mechanism-validation
exit "$status"
