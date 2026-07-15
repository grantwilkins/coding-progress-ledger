#!/usr/bin/env bash
#SBATCH --job-name=qh-migration-profile
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1c-%j.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1c-%j.err

set -euo pipefail

REPO=${REPO:-/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate}
PY=${PY:-$REPO/.venv/bin/python}
PLAN=${PLAN:?set PLAN to a reviewed plan JSON file}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new or matching resumable run directory}

cd "$REPO"
module load gcc/14.2.0 openblas/0.3.28
export QH_APPTAINER_IMAGE=${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox}
export QH_APPTAINER_GPU_MODE=${QH_APPTAINER_GPU_MODE:-nv}
export QH_PORT_OFFSET=${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}

$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
RUN_STATUS=0
$PY queue-haul/stage1c_controller.py run --plan "$PLAN" --run-root "$RUN_ROOT" || RUN_STATUS=$?
$PY queue-haul/stage1c_controller.py reduce --run-root "$RUN_ROOT"
exit "$RUN_STATUS"
