#!/usr/bin/env bash
set -euo pipefail
cat > a.txt <<'TXT'
hello
TXT
cat > b.txt <<'TXT'
world
TXT
cat > build.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p dist
: > dist/out.txt
for f in '*.txt'; do
  echo "found: $f" >> dist/out.txt
done
SH
chmod +x build.sh
