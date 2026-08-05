#!/bin/bash
# Run the full-shed power trace directly inside an existing 2-GPU allocation.
set -euo pipefail
export LC_ALL=C
source "${LMOD_PKG:-/share/software/user/open/lmod/lmod}/init/bash"
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
export QH_LMCACHE_L1_GB=34
# The kv_transfer moves ride on L2, so it must be large enough that eviction is
# rare: an evicted key under an in-flight prefetch leaves the request deferred
# forever. 20 GB against a 12-session foreground and 2.7 GB of migrating KV.
export QH_REDIS_MAXMEMORY_GB=20

# LMCache does not unlink its L1 pool if it is killed; a stale 36 GB segment is
# charged to the cgroup forever. Drop pools whose owning pid is gone.
for pool in /dev/shm/lmcache_l1_pool_*; do
  [ -e "$pool" ] || continue
  kill -0 "${pool##*_}" 2>/dev/null || rm -f "$pool"
done
export POWERTRACE_TELEMETRY_TMPDIR="$L_SCRATCH/qh-shed-powertrace-$SLURM_JOB_ID"

mkdir -p "$RUN_ROOT" "$POWERTRACE_TELEMETRY_TMPDIR"
cd "$REPO"
exec .venv/bin/python -u queue-haul/outputs/live-power-shed/driver.py "$RUN_ROOT"
