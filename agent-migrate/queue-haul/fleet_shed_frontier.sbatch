#!/usr/bin/env bash
# Fleet shed frontier: CPU-only simulation, no GPU and no container.
# Submit prepare, headline and sensitivity arrays, then reduce with afterok
# dependencies. Set FRONTIER_OUT to override the fresh default output directory.
#SBATCH --job-name=qh-fleet-frontier
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/fleet-frontier-%A_%a.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/fleet-frontier-%A_%a.err

set -euo pipefail
export LC_ALL=C
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4
REPO=${REPO:-/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate}
cd "$REPO/queue-haul"
# One core per rank: numpy threads would oversubscribe the array tasks.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN="uv run --project $REPO/.. python fleet_shed_frontier_campaign.py"
OUT=${FRONTIER_OUT:-$REPO/queue-haul/outputs/fleet-shed-frontier-a100-20260820}
case "${PHASE:-shard}" in
  prepare) $RUN prepare --out "$OUT" ;;
  reduce)  $RUN reduce --out "$OUT"
           uv run --project "$REPO/.." python plot_fleet_shed_frontier.py "$OUT" ;;
  shard)   $RUN run-shard --out "$OUT" \
             --shard "${SLURM_ARRAY_TASK_ID:?array job required}" \
             --subset "${SUBSET:-headline}" ;;
  *) echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
esac
