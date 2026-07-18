#!/usr/bin/env bash
#SBATCH --job-name=qh-catch-up
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1e-%j.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1e-%j.err

export REPO=${REPO:-/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate}
export PLAN=${PLAN:-queue-haul/outputs/append-catch-up-plan.json}
export RUN_ROOT=${RUN_ROOT:-queue-haul/outputs/append-catch-up-run}
export CHECK=check-catch-up
exec "$REPO/queue-haul/stage1_targeted_run.sh"
