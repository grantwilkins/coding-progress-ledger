#!/usr/bin/env bash
#SBATCH --job-name=qh-stage1c-quick
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1c-quick-%j.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1c-quick-%j.err

set -euo pipefail

REPO=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate
cd "$REPO"

module load gcc/14.2.0 openblas/0.3.28

PY=${PY:-$REPO/.venv/bin/python}
MANIFEST=${MANIFEST:-queue-haul/outputs/stage1c_live-sessions.json}
RUN_ROOT=${RUN_ROOT:-queue-haul/outputs/stage1c_quick}
PROFILE=${PROFILE:-$RUN_ROOT/live_profile.json}
POLICIES=${POLICIES:-greedy,random,all-r,all-s}
DEADLINES=${DEADLINES:-120}
TARGET_FRACS=${TARGET_FRACS:-0.45}
MBPS=${MBPS:-1000}
NVSMI_MS=${NVSMI_MS:-250}
BASELINE_S=${BASELINE_S:-30}
SETTLE_S=${SETTLE_S:-30}
SEED=${SEED:-0}
REPLAY_CONCURRENCY=${REPLAY_CONCURRENCY:-1}
KV_CONCURRENCY=${KV_CONCURRENCY:-0}

mkdir -p "$RUN_ROOT"

$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
$PY queue-haul/stage1c_controller.py live-grid   --manifest "$MANIFEST"   --run-root "$RUN_ROOT"   --policies "$POLICIES"   --deadlines "$DEADLINES"   --target-fracs "$TARGET_FRACS"   --mbps "$MBPS"   --nvsmi-ms "$NVSMI_MS"   --baseline-s "$BASELINE_S"   --settle-s "$SETTLE_S"   --seed "$SEED"   --profile "$PROFILE"   --replay-concurrency "$REPLAY_CONCURRENCY"   --kv-concurrency "$KV_CONCURRENCY"
