from __future__ import annotations

import json
import threading
import time
import socket

from ledger_progress import LedgerSession, SubtaskCategory
from ledger_progress.run_manager import main as run_manager_main
from ledger_progress.serialization import event_to_dict


def _request(host, port, raw):
    s = socket.create_connection((host, port), timeout=5)
    s.sendall(raw)
    chunks = []
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
    s.close()
    response = b"".join(chunks)
    body = response.split(b"\r\n\r\n", 1)[1]
    return json.loads(body)


def _post(host, port, path, data):
    body = json.dumps(data).encode()
    raw = (f"POST {path} HTTP/1.0\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
    return _request(host, port, raw)


def _get(host, port, path):
    raw = f"GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n".encode()
    return _request(host, port, raw)


def test_serve_post_events_query_progress(tmp_path):
    run_dir = tmp_path / "run_serve"
    run_dir.mkdir()

    session = LedgerSession("Serve test")
    a = session.add("Code A", step=1, category=SubtaskCategory.PRODUCT)
    v = session.add("Validate", step=1, category=SubtaskCategory.VALIDATION)
    session.add("Code B", step=1, category=SubtaskCategory.PRODUCT)
    session.block(a, step=2, reason="env")
    session.complete(v, "passed", step=3)
    all_events = [event_to_dict(e) for e in session.ledger.events]
    early = all_events[:4]
    late = all_events[4:]

    def run_server():
        rc = run_manager_main(["serve", str(run_dir), "--port", "0", "--exit-after-events", str(len(all_events))])
        assert rc == 0

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    addr_path = run_dir / "serve_address.json"
    deadline = time.time() + 5
    while time.time() < deadline and not addr_path.exists():
        time.sleep(0.05)
    assert addr_path.exists()
    host_port = json.loads(addr_path.read_text())
    host, port = host_port["host"], host_port["port"]

    for event in early:
        _post(host, port, "/events", event)
    progress_early = _get(host, port, "/progress")
    blocked_early = _get(host, port, "/blocked")
    assert progress_early["current_step"] == 1
    assert progress_early["validation_progress"] == 0.0
    assert blocked_early["active_blocked_leaves"] == []

    for event in late[:-1]:
        _post(host, port, "/events", event)
    progress_mid = _get(host, port, "/progress")
    blocked_mid = _get(host, port, "/blocked")
    stalled_mid = _get(host, port, "/stalled?threshold=0")
    assert progress_mid["current_step"] == 2
    assert [s["id"] for s in blocked_mid["active_blocked_leaves"]] == [a]
    assert stalled_mid["stalled_for_blocked"] == 0
    assert stalled_mid["meets_threshold"] is True

    _post(host, port, "/events", late[-1])
    thread.join(timeout=5)
    assert not thread.is_alive()

    appended = (run_dir / "ledger.jsonl").read_text().splitlines()
    assert len(appended) == len(all_events)
