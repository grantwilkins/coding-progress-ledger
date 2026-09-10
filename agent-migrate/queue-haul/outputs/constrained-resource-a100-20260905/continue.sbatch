#!/bin/bash
#SBATCH --job-name=qh-constrained-v2
#SBATCH --account=ramr
#SBATCH --partition=ramr
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=17
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --dependency=afterany:42104466
#SBATCH --chdir=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul
#SBATCH --output=outputs/constrained-resource-a100-20260905/resume-%j.log
set -euo pipefail
module load gcc/14.2.0 openblas/0.3.28 uv/0.10.8
export QH_LMCACHE_MODE=mp
git diff --quiet c86cf378f027aa5145eec2596682d3e1f096fbe4 -- '*.py'
test -s outputs/constrained-resource-a100-20260905/prepared/plan.json
uv run python constrained_state_campaign.py run --plan outputs/constrained-resource-a100-20260905/prepared/plan.json --run-root outputs/constrained-resource-a100-20260905/run
