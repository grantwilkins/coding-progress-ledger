#!/bin/bash
#SBATCH --job-name=qh-drain-bw
#SBATCH --partition=ramr
#SBATCH --account=ramr
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=2
#SBATCH --time=12:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/capacity-full-drain-block%a-20260807/slurm-%A_%a.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/capacity-full-drain-block%a-20260807/slurm-%A_%a.err

set -euo pipefail

block=${SLURM_ARRAY_TASK_ID:?}
sha=8f9e658
repo=/home/groups/ramr/gfw/coding-progress-ledger
code=/scratch/users/gfw/qh-code-full-drain-${sha}-${SLURM_JOB_ID}
out=${repo}/agent-migrate/queue-haul/outputs/capacity-full-drain-block${block}-20260807
run=/scratch/users/gfw/qh-capacity-full-drain-block${block}-20260807

module load gcc/14.2.0 openblas/0.3.28
git clone --no-checkout "$repo" "$code"
cd "$code"
git checkout "$sha"
cd agent-migrate
export QH_LMCACHE_MODE=mp
export QH_APPTAINER_IMAGE=/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif
exec /home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/.venv/bin/python \
  queue-haul/capacity_sweep_campaign.py full-drain --out "$out" --run-root "$run"
