#!/usr/bin/env bash
set -euo pipefail
cat > run.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=lib python3 tools/render.py
SH
chmod +x run.sh
./run.sh
