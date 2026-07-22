#!/usr/bin/env bash
#SBATCH --job-name=qh-destination
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/destination-%A.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/destination-%A.err

set -euo pipefail
export LC_ALL=C
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4
REPO=${REPO:-/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate}
IMAGE=${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif}
test "$(sha256sum "$IMAGE" | cut -d' ' -f1)" = 50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab
test "$(sha256sum "$QH_JOB_FILE" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$QH_JOB_FILE.sha256")"
export QH_LMCACHE_MODE=mp QH_APPTAINER_IMAGE=$IMAGE
export QH_PORT_OFFSET=${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}
cd "$REPO"
exec bash "$QH_JOB_FILE"
