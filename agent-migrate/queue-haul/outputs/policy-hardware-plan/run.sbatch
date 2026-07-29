#!/bin/bash
#SBATCH --job-name=qh-policy-cdf
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=policy-hardware-%j.out
#SBATCH --error=policy-hardware-%j.err
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QH_POLICY_RUN_ROOT="${QH_POLICY_RUN_ROOT:-/scratch/$USER/qh-policy-run}"
bash "$script_dir/run.sh"
