#!/bin/bash
#SBATCH --job-name=qh-overlap-power
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/gfw/qh-overlap-power-20260803/slurm-%j.out
#SBATCH --error=/scratch/users/gfw/qh-overlap-power-20260803/slurm-%j.err

set -euo pipefail
export LC_ALL=C
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4

REPO=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate
POWERTRACE=/home/groups/ramr/gfw/powertrace-sim
RUN_ROOT=/scratch/users/gfw/qh-overlap-power-20260803/run-${SLURM_JOB_ID}
DRIVER=/scratch/users/gfw/qh-overlap-power-20260803/driver.py
IMAGE=/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif

cd "$REPO"
test "$(git rev-parse HEAD)" = 763920e747640f8d04915a2fbed10c23fb604838
git diff --quiet
git diff --cached --quiet
test "$(git --git-dir="$POWERTRACE/.git" --work-tree="$POWERTRACE" rev-parse HEAD)" = 4f167596c61dc1760f0974522ea18a7f5cfaf836
git --git-dir="$POWERTRACE/.git" --work-tree="$POWERTRACE" diff --quiet
git --git-dir="$POWERTRACE/.git" --work-tree="$POWERTRACE" diff --cached --quiet
test "$(sha256sum "$DRIVER" | cut -d' ' -f1)" = 4636e27465a13b7af5baaa6bf074bd2009fc0b2a9ca565d2c610a39f47c3bad1
test "$(sha256sum "$IMAGE" | cut -d' ' -f1)" = 50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab
test -n "${L_SCRATCH:-}"

export QH_LMCACHE_MODE=mp
export QH_APPTAINER_IMAGE="$IMAGE"
export QH_PORT_OFFSET=$((SLURM_JOB_ID % 40000 + 1000))
export POWERTRACE_TELEMETRY_TMPDIR="$L_SCRATCH/qh-overlap-powertrace-$SLURM_JOB_ID"

mkdir -p "$RUN_ROOT" "$POWERTRACE_TELEMETRY_TMPDIR"
.venv/bin/python "$DRIVER" "$RUN_ROOT"
