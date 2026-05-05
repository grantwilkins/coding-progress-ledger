#!/usr/bin/env bash
set -euo pipefail
cat > build.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p dist
echo "build complete" > dist/output.txt
SH
chmod 644 build.sh
