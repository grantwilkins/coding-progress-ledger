#!/bin/bash
set -euo pipefail
while systemctl is-active --quiet qh-shared-recovery-20260914.service; do sleep 10; done
export QH_MODEL_PROFILE=profiles/network-qwen-h100.json
exec /datadrive/qh0912/.venv/bin/python shared_load_controls.py --model Qwen/Qwen3.8-27B --cluster /datadrive/shared-controls-cluster.json --calibration /datadrive/c12.json --resident-rps 0.05 --state-reference /datadrive/qf20/full_state_control.json --arms none --seconds 240 --warmup-s 60 --run-root /datadrive/shared-qwen-baseline005-20260914
