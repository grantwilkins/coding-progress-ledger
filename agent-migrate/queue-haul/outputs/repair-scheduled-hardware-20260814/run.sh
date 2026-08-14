#!/usr/bin/env bash
set -euo pipefail
: "${QH_AZURE_SSH_KEY:?set QH_AZURE_SSH_KEY}"
: "${QH_REPAIR_RUN_ROOT:?set QH_REPAIR_RUN_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
uv run python queue-haul/repair_hardware_campaign.py run   --plan "$script_dir/plan.json" --ssh-key "$QH_AZURE_SSH_KEY"   --run-root "$QH_REPAIR_RUN_ROOT"
uv run python queue-haul/repair_hardware_campaign.py validate   --plan "$script_dir/plan.json" --run-root "$QH_REPAIR_RUN_ROOT"
