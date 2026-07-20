#!/usr/bin/env bash
#SBATCH --job-name=qh-mp-campaign
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1g-%j.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1g-%j.err

set -euo pipefail
REPO=${REPO:-/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate}
RUN_ROOT=${RUN_ROOT:-queue-haul/outputs/mp-campaign-run}
cd "$REPO"
module load gcc/14.2.0 openblas/0.3.28
export QH_LMCACHE_MODE=mp
export QH_APPTAINER_IMAGE=${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129.sandbox}
export QH_PORT_OFFSET=${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}
exec .venv/bin/python queue-haul/stage1g_mp_campaign.py --run-root "$RUN_ROOT"
