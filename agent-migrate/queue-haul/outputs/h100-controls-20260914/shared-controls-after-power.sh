#!/bin/bash
set -euo pipefail
while systemctl is-active --quiet qh-power-gemma-20260914.service; do sleep 10; done
test -f /datadrive/power-gemma-20260914/complete.json
while ssh -i /home/azureuser/.ssh/azrs -o BatchMode=yes -o ConnectTimeout=10 azureuser@10.13.0.4 'systemctl is-active --quiet qh-power-qwen-20260914.service'; do sleep 10; done
ssh -i /home/azureuser/.ssh/azrs -o BatchMode=yes -o ConnectTimeout=10 azureuser@10.13.0.4 'test -f /datadrive/power-qwen-20260914/complete.json'
export QH_MODEL_PROFILE=profiles/network-gpt-h100.json
/datadrive/qh0912/.venv/bin/python shared_load_controls.py --model openai/gpt-oss-20b --cluster /datadrive/shared-controls-cluster.json --calibration /datadrive/c12.json --resident-rps 0.25 --run-root /datadrive/shared-gpt-20260914
export QH_MODEL_PROFILE=profiles/network-qwen-h100.json
/datadrive/qh0912/.venv/bin/python shared_load_controls.py --model Qwen/Qwen3.8-27B --cluster /datadrive/shared-controls-cluster.json --calibration /datadrive/c12.json --resident-rps 0.25 --state-reference /datadrive/qf20/full_state_control.json --run-root /datadrive/shared-qwen-20260914
export QH_MODEL_PROFILE=profiles/network-gemma-h100.json
/datadrive/qh0912/.venv/bin/python shared_load_controls.py --model google/gemma-4-26B-A4B-it --cluster /datadrive/shared-controls-cluster.json --calibration /datadrive/c12.json --resident-rps 0.25 --run-root /datadrive/shared-gemma-20260914
