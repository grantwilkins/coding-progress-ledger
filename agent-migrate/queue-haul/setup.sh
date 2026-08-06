#!/usr/bin/env bash
set -euo pipefail

[[ $EUID -ne 0 ]] || { echo "run setup.sh as the login user, not root" >&2; exit 1; }
command -v dnf >/dev/null || { echo "setup.sh requires dnf" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == *A100* ]] || { echo "no A100 GPU found" >&2; exit 1; }
mountpoint -q /datadrive || { echo "/datadrive is not mounted" >&2; exit 1; }

queue_haul_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(dirname "$queue_haul_dir")
[[ -f "$repo_dir/pyproject.toml" && -f "$repo_dir/uv.lock" ]] || {
  echo "setup.sh requires the complete repository checkout" >&2
  exit 1
}

sudo dnf install -y gcc gcc-c++ make pkgconf-pkg-config ca-certificates curl valkey chrony iperf3
sudo chown "$(id -un):$(id -gn)" /datadrive
chmod u+rwx /datadrive

[[ -e /dev/ptp_hyperv ]] || { echo "/dev/ptp_hyperv not found" >&2; exit 1; }
ptp_line='refclock PHC /dev/ptp_hyperv poll 3 dpoll -2 offset 0 stratum 2'
grep -Fqx "$ptp_line" /etc/chrony.conf || printf '%s\n' "$ptp_line" | sudo tee -a /etc/chrony.conf >/dev/null
sudo systemctl enable --now chronyd
sudo systemctl restart chronyd
chronyc waitsync 60 0.002

curl -LsSf https://astral.sh/uv/0.11.32/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
rustup toolchain install 1.96.0 --profile minimal

export QH_RUNTIME=native QH_LMCACHE_MODE=mp QH_CACHE_ROOT=/datadrive/queue-haul-cache
export HF_HOME=/datadrive HF_HUB_CACHE=/datadrive/hub HF_CACHE=/datadrive
mkdir -p "$HF_HUB_CACHE" "$QH_CACHE_ROOT"

profile="$HOME/.bashrc"
profile_tmp=$(mktemp)
touch "$profile"
sed '/# BEGIN QUEUE-HAUL/,/# END QUEUE-HAUL/d' "$profile" > "$profile_tmp"
cat >> "$profile_tmp" <<'EOF'
# BEGIN QUEUE-HAUL
export QH_RUNTIME=native
export QH_LMCACHE_MODE=mp
export QH_CACHE_ROOT=/datadrive/queue-haul-cache
export HF_HOME=/datadrive
export HF_HUB_CACHE=/datadrive/hub
export HF_CACHE=/datadrive
# END QUEUE-HAUL
EOF
mv "$profile_tmp" "$profile"

cd "$repo_dir"
uv python install 3.12
uv sync --frozen --inexact --python 3.12
uv pip install --python .venv/bin/python \
  'vllm==0.22.0' 'lmcache==0.5.1' \
  --torch-backend=cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v0.5.1-cu129 \
  --index-strategy unsafe-best-match

.venv/bin/hf download openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --exclude 'original/*' --exclude 'metal/*'

.venv/bin/python - <<'PY'
from importlib.metadata import version
from pathlib import Path

import torch
import lmcache.c_ops
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

assert version("vllm") == "0.22.0", version("vllm")
assert version("lmcache") == "0.5.1", version("lmcache")
assert torch.version.cuda == "12.9", torch.version.cuda
snapshot = Path("/datadrive/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee")
assert snapshot.is_dir(), snapshot
assert len(list(snapshot.glob("model-*.safetensors"))) == 3, snapshot
assert all((snapshot / name).is_file() for name in
           ("config.json", "model.safetensors.index.json", "tokenizer.json"))
PY
valkey-server --version

echo "Queue-Haul Azure setup complete. Run: source ~/.bashrc"
