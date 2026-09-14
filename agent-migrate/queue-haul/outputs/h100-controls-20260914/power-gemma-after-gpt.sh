#!/bin/bash
set -euo pipefail
while systemctl is-active --quiet qh-power-gpt-20260914.service; do sleep 10; done
test -f /datadrive/power-gpt-20260914/complete.json
exec /datadrive/qh0912/.venv/bin/python power_transition_controls.py --model google/gemma-4-26B-A4B-it --manifest /datadrive/power-controls-manifest.json --run-root /datadrive/power-gemma-20260914 --repeats 1
