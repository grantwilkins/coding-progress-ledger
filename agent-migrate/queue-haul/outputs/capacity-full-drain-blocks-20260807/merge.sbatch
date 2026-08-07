#!/bin/bash
#SBATCH --job-name=qh-drain-merge
#SBATCH --partition=ramr
#SBATCH --account=ramr
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/capacity-full-drain-blocks-20260807/merge-%j.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/capacity-full-drain-blocks-20260807/merge-%j.err

set -euo pipefail

sha=8f9e658
repo=/home/groups/ramr/gfw/coding-progress-ledger
code=/scratch/users/gfw/qh-code-full-drain-merge-${sha}-${SLURM_JOB_ID}

module load gcc/14.2.0 openblas/0.3.28
git clone --no-checkout "$repo" "$code"
cd "$code"
git checkout "$sha"
cd agent-migrate
exec /home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/.venv/bin/python \
  queue-haul/capacity_sweep_campaign.py full-drain \
  --merge-run-root \
  /scratch/users/gfw/qh-capacity-full-drain-block0-20260807 \
  /scratch/users/gfw/qh-capacity-full-drain-block1-20260807 \
  /scratch/users/gfw/qh-capacity-full-drain-block2-20260807 \
  /scratch/users/gfw/qh-capacity-full-drain-block3-20260807 \
  /scratch/users/gfw/qh-capacity-full-drain-block4-20260807 \
  --out "$repo"/agent-migrate/queue-haul/outputs/capacity-full-drain-final-20260807
