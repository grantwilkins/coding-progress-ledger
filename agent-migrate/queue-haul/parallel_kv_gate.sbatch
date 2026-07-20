#!/usr/bin/env bash
#SBATCH --job-name=qh-parallel-gate
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/parallel-kv-gate-%j.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/parallel-kv-gate-%j.err

export REPO=${REPO:-/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate}
export PLAN=${PLAN:-queue-haul/outputs/parallel-kv-gate-plan.json}
export RUN_ROOT=${RUN_ROOT:-queue-haul/outputs/parallel-kv-gate-run}
export CHECK=check-parallel
exec "$REPO/queue-haul/targeted_migration_run.sh"
