set -eu
root=/tmp/qh-replay-completion-20260910
venv=/home/azureuser/coding-progress-ledger/agent-migrate/.venv
mkdir -p "$root/overlay" "$root/cache" "$root/rpc"
cp -as "$venv/lib/python3.12/site-packages/vllm" "$root/overlay/vllm"
sha256sum "$venv/lib/python3.12/site-packages/vllm/v1/engine/output_processor.py" > "$root/original-runtime-sha256.txt"
rm "$root/overlay/vllm/v1/engine/output_processor.py"
cp "$root/output_processor.patched.py" "$root/overlay/vllm/v1/engine/output_processor.py"
sha256sum "$root/overlay/vllm/v1/engine/output_processor.py" > "$root/patched-runtime-sha256.txt"
QH_LMCACHE_MODE=mp PYTHONPATH="$root/overlay:$root/lmcache_compat" VIRTUAL_ENV="$venv" /datadrive/queue-haul-tools/bin/uv run --active --no-project --no-sync pytest -q "$root/test_stream_patch.py" > "$root/overlay-test.log" 2>&1
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version --format=csv > "$root/gpu-identity.csv"
