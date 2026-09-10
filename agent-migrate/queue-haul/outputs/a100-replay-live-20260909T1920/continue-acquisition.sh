#!/usr/bin/env bash
set -euo pipefail
export QH_RUNTIME=native QH_LMCACHE_MODE=mp QH_PREFIX_CACHING=on HF_HOME=/datadrive QH_CACHE_ROOT=/tmp/qh-replay-cache CUDA_VISIBLE_DEVICES=0
for stage in scout episodes followups; do
  /tmp/qh-replay-runtime/bin/python pool_replay_resident.py "$stage" --out outputs/a100-replay-live-20260909T1920 --plan outputs/a100-replay-live-20260909T1920/live-plan.json > "outputs/a100-replay-live-20260909T1920/${stage}-recovered.log" 2>&1
done
