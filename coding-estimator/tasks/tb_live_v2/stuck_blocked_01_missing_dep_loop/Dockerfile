FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux asciinema && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN cat > /app/page.html <<'HTML'
<html><body>
  <h1>Hello, terminal-bench</h1>
  <p>Filler.</p>
</body></html>
HTML
RUN cat > /app/scrape.py <<'PY'
from bs4 import BeautifulSoup
from pathlib import Path
html = Path("/app/page.html").read_text()
soup = BeautifulSoup(html, "html.parser")
Path("/app/h1.txt").write_text(soup.find("h1").get_text())
PY
