#!/usr/bin/env bash
#SBATCH --job-name=qh-stage1c-grid
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1c-grid-%j.out
#SBATCH --error=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate/queue-haul/outputs/stage1c-grid-%j.err

set -euo pipefail

REPO=/home/groups/ramr/gfw/coding-progress-ledger/agent-migrate
cd "$REPO"

module load gcc/14.2.0 openblas/0.3.28

PY=${PY:-$REPO/.venv/bin/python}
MANIFEST=${MANIFEST:-queue-haul/outputs/stage1c_live-sessions.json}
MANIFESTS=${MANIFESTS:-}
TRACE=${TRACE:-/scratch/users/gfw/ptsim/tracelab/syfi_coding_trace.jsonl.gz}
WORKLOADS=${WORKLOADS:-}
RUN_ROOT=${RUN_ROOT:-queue-haul/outputs/stage1c_grid}
PROFILE=${PROFILE:-$RUN_ROOT/live_profile.json}
POLICIES=${POLICIES:-greedy,random,lp}
DEADLINES=${DEADLINES:-10,30,120}
TARGET_FRACS=${TARGET_FRACS:-0.25,0.45,0.65}
MBPS=${MBPS:-1000}
NVSMI_MS=${NVSMI_MS:-250}
BASELINE_S=${BASELINE_S:-120}
SETTLE_S=${SETTLE_S:-120}
SEED=${SEED:-0}
REPLAY_CONCURRENCY=${REPLAY_CONCURRENCY:-1}
KV_CONCURRENCY=${KV_CONCURRENCY:-0}
SMALL_SESSIONS=${SMALL_SESSIONS:-16}
MIXED_SESSIONS=${MIXED_SESSIONS:-8}
LARGE_SESSIONS=${LARGE_SESSIONS:-8}
SMALL_MAX_MODEL_LEN=${SMALL_MAX_MODEL_LEN:-8192}
MIXED_MAX_MODEL_LEN=${MIXED_MAX_MODEL_LEN:-32768}
LARGE_MAX_MODEL_LEN=${LARGE_MAX_MODEL_LEN:-32768}
SMALL_MIN_CONTEXT_TOKENS=${SMALL_MIN_CONTEXT_TOKENS:-1024}
MIXED_MIN_CONTEXT_TOKENS=${MIXED_MIN_CONTEXT_TOKENS:-2048}
LARGE_MIN_CONTEXT_TOKENS=${LARGE_MIN_CONTEXT_TOKENS:-2048}

mkdir -p "$RUN_ROOT"

if [[ -z "$MANIFESTS" && -n "$WORKLOADS" ]]; then
  mkdir -p "$RUN_ROOT/manifests"
  IFS=',' read -r -a WLS <<< "$WORKLOADS"
  for WL in "${WLS[@]}"; do
    case "$WL" in
      small) N=$SMALL_SESSIONS; MAX_LEN=$SMALL_MAX_MODEL_LEN; MIN_CTX=$SMALL_MIN_CONTEXT_TOKENS ;;
      mixed) N=$MIXED_SESSIONS; MAX_LEN=$MIXED_MAX_MODEL_LEN; MIN_CTX=$MIXED_MIN_CONTEXT_TOKENS ;;
      large) N=$LARGE_SESSIONS; MAX_LEN=$LARGE_MAX_MODEL_LEN; MIN_CTX=$LARGE_MIN_CONTEXT_TOKENS ;;
      *) echo "unknown workload: $WL" >&2; exit 2 ;;
    esac
    OUT="$RUN_ROOT/manifests/$WL.json"
    $PY queue-haul/stage1c_controller.py make-manifest --source tracelab --input "$TRACE" --out "$OUT" --workload "$WL" --sessions "$N" --max-model-len "$MAX_LEN" --min-context-tokens "$MIN_CTX" --seed "$SEED"
    MANIFESTS="${MANIFESTS:+$MANIFESTS,}$OUT"
  done
fi

MANIFEST_ARG=(--manifest "$MANIFEST")
if [[ -n "$MANIFESTS" ]]; then
  MANIFEST_ARG=(--manifests "$MANIFESTS")
fi

$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
$PY queue-haul/stage1c_controller.py live-grid \
  "${MANIFEST_ARG[@]}" \
  --run-root "$RUN_ROOT" \
  --policies "$POLICIES" \
  --deadlines "$DEADLINES" \
  --target-fracs "$TARGET_FRACS" \
  --mbps "$MBPS" \
  --nvsmi-ms "$NVSMI_MS" \
  --baseline-s "$BASELINE_S" \
  --settle-s "$SETTLE_S" \
  --seed "$SEED" \
  --profile "$PROFILE" \
  --replay-concurrency "$REPLAY_CONCURRENCY" \
  --kv-concurrency "$KV_CONCURRENCY"
