#!/usr/bin/env bash
set -euo pipefail
cat > server.py <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ITEMS = Path("items.txt")

def _read_items():
    if not ITEMS.is_file():
        return []
    return sorted(
        line.strip() for line in ITEMS.read_text().splitlines()
        if line.strip()
    )

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/version":
            self._send(200, {"version": "1.0.0"})
        elif self.path == "/items":
            self._send(200, {"items": _read_items()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/items":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        name = body.get("name", "")
        if not isinstance(name, str) or not name.strip():
            self._send(400, {"error": "name required"})
            return
        with ITEMS.open("a") as f:
            f.write(name + "\n")
        self._send(201, {"name": name})

    def log_message(self, *_): pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
