#!/usr/bin/env bash
set -euo pipefail
cat > page.html <<'HTML'
<html><body>
  <h1>Hello, terminal-bench</h1>
  <p>Filler.</p>
</body></html>
HTML
cat > scrape.py <<'PY'
from bs4 import BeautifulSoup
from pathlib import Path
html = Path("page.html").read_text()
soup = BeautifulSoup(html, "html.parser")
Path("h1.txt").write_text(soup.find("h1").get_text())
PY
