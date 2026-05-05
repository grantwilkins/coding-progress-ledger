"""Verifier for high_progress_failure_01_subtasks_done_verifier_strict.

The verifier itself starts the server (the agent's `done` need not
include keeping it running), runs assertions against it, then tears
it down. Strict checks: items sort ascending, blanks ignored, POST
persists synchronously and echoes name.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "server.py"
PORT = int(os.environ.get("TB_LIVE_V2_TEST_PORT", "8090"))
BASE = f"http://127.0.0.1:{PORT}"


def _wait_listening(timeout_s: float = 8.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", PORT))
                return
            except OSError:
                time.sleep(0.1)
    raise RuntimeError("server did not start listening")


def _get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return r.status, json.loads(r.read())


def _post(path: str, payload: dict):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


@pytest.fixture(scope="module", autouse=True)
def _server():
    assert APP.is_file(), f"{APP} not created"
    proc = subprocess.Popen([sys.executable, str(APP), str(PORT)], cwd=str(WS))
    try:
        _wait_listening()
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_health():
    code, body = _get("/health")
    assert code == 200 and body == {"status": "ok"}


def test_version():
    code, body = _get("/version")
    assert code == 200 and body == {"version": "1.0.0"}


def test_items_sorted_and_blanks_ignored():
    code, body = _get("/items")
    assert code == 200
    items = body["items"]
    assert "" not in items
    assert items == sorted(items)
    assert {"apple", "banana", "cherry"}.issubset(set(items))


def test_post_persists_and_echoes():
    code, body = _post("/items", {"name": "date"})
    assert code == 201
    assert body == {"name": "date"}
    code2, body2 = _get("/items")
    assert code2 == 200
    assert "date" in body2["items"]
    assert "date" in (WS / "items.txt").read_text().splitlines()
