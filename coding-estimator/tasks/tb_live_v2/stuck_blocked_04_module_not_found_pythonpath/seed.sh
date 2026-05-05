#!/usr/bin/env bash
set -euo pipefail
mkdir -p lib tools out
cat > lib/widget.py <<'PY'
def render(name):
    return f"rendered: {name}"
PY
cat > tools/render.py <<'PY'
import os, pathlib
import widget
out = pathlib.Path("out/result.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(widget.render("hello") + "\n")
PY
cat > run.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
python3 tools/render.py
SH
chmod +x run.sh
