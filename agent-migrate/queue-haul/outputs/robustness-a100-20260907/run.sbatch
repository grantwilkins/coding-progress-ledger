#!/bin/bash
#SBATCH --job-name=qh-repeat
#SBATCH --account=ramr
#SBATCH --partition=ramr
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=17
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%1
#SBATCH --chdir=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul
#SBATCH --output=outputs/robustness-a100-20260907/job-%A_%a.log
set -euo pipefail
module load gcc/14.2.0 openblas/0.3.28 uv/0.10.8
export QH_LMCACHE_MODE=mp
for campaign in wan prefill; do
    campaign_root=outputs/robustness-a100-20260907/$SLURM_ARRAY_TASK_ID/$campaign
    uv run python constrained_state_campaign.py run --plan "$campaign_root/prepared/plan.json" --run-root "$campaign_root/run"
done
