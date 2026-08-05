#!/bin/bash
# Run the full-shed power trace directly inside an existing 2-GPU allocation.
set -euo pipefail
export LC_ALL=C
source /etc/profile.d/modules.sh
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4

REPO=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate
RUN_ROOT=/scratch/users/gfw/qh-shed-power-20260805/run-${SLURM_JOB_ID}
IMAGE=/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif

test -n "${L_SCRATCH:-}"
test "$(sha256sum "$IMAGE" | cut -d' ' -f1)" = 50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab

export QH_LMCACHE_MODE=mp
export QH_APPTAINER_IMAGE="$IMAGE"
export QH_PORT_OFFSET=$((SLURM_JOB_ID % 40000 + 1000))
export QH_KV_ROLE_SOURCE=kv_both
export QH_KV_ROLE_SINK=kv_both
export QH_LMCACHE_L1_GB=36
export POWERTRACE_TELEMETRY_TMPDIR="$L_SCRATCH/qh-shed-powertrace-$SLURM_JOB_ID"

mkdir -p "$RUN_ROOT" "$POWERTRACE_TELEMETRY_TMPDIR"
cd "$REPO"
exec .venv/bin/python -u queue-haul/outputs/live-power-shed/driver.py "$RUN_ROOT"
