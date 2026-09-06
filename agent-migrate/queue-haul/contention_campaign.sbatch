#!/bin/bash
#SBATCH --job-name=qh-contention
#SBATCH --account=ramr
#SBATCH --partition=ramr
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=17
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul
#SBATCH --output=outputs/contention-a100-20260905/job-%j.log
set -euo pipefail
module load gcc/14.2.0 openblas/0.3.28 uv/0.10.8
export QH_LMCACHE_MODE=mp
campaign_root=${1:-outputs/contention-a100-20260905}
uv run python constrained_state_campaign.py run --plan "$campaign_root/prepared/plan.json" --run-root "$campaign_root/run"
uv run python contention_campaign.py summarize --plan "$campaign_root/prepared/plan.json" --run-root "$campaign_root/run"
