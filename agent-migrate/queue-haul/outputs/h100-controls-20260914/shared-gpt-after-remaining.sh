#!/bin/bash
set -euo pipefail
while systemctl is-active --quiet qh-shared-remaining-20260914.service; do sleep 10; done
export QH_MODEL_PROFILE=profiles/network-gpt-h100.json
exec /datadrive/qh0912/.venv/bin/python shared_load_controls.py --model openai/gpt-oss-20b --cluster /datadrive/shared-controls-cluster.json --calibration /datadrive/c12.json --resident-rps 0.25 --run-root /datadrive/shared-gpt-16k-20260914
