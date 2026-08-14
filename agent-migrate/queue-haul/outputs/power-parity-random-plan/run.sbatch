#!/bin/bash
#SBATCH --job-name=qh-power-parity
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=power-parity-%j.out
#SBATCH --error=power-parity-%j.err
set -euo pipefail
export LC_ALL=C
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4
script_path="$(scontrol show job "$SLURM_JOB_ID" -o | awk '{for (i=1;i<=NF;i++) if ($i ~ /^Command=/) {sub(/^Command=/,"",$i); print $i}}')"
script_dir="$(dirname "$script_path")"
export QH_POWER_PARITY_RUN_ROOT="${QH_POWER_PARITY_RUN_ROOT:-/scratch/users/$USER/qh-power-parity-random-20260812}"
export QH_APPTAINER_IMAGE="${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif}"
test "$(sha256sum "$QH_APPTAINER_IMAGE" | cut -d' ' -f1)" = 50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab
export QH_PORT_OFFSET="${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}"
bash "$script_dir/run.sh"
