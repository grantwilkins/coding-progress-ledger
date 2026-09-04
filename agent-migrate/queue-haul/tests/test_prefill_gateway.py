"""Claim: the live gateway measures and caps uncached prefill work."""

from __future__ import annotations

import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prefill_gateway import PrefillCompletionLimiter, PrefillGateway, _usage


def test_gateway_extracts_uncached_prompt_usage_from_stream():
    payload = (
        b'data: {"usage":{"prompt_tokens":10,'
        b'"prompt_tokens_details":{"cached_tokens":4}}}\n\n'
        b"data: [DONE]\n\n"
    )

    assert _usage(payload) == (10, 4)


def test_gateway_live_control_reserves_aggregate_completion_time(monkeypatch):
    clock = iter((0.0, 0.0, 0.0, 0.0, 0.0))
    slept = []
    monkeypatch.setattr("prefill_gateway.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("prefill_gateway.time.sleep", slept.append)
    limiter = PrefillCompletionLimiter()

    assert limiter.update(100) == {"tokens_per_s": 100}
    assert limiter.wait(10) == pytest.approx(.1)
    assert limiter.wait(10) == pytest.approx(.2)
    assert slept == pytest.approx([.1, .2])

    assert limiter.update(None) == {"tokens_per_s": None}
    assert limiter.wait(10) == 0


def test_unthrottled_gateway_forwards_sse_before_upstream_eof(tmp_path):
    first_sent = threading.Event()
    release = threading.Event()

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for payload in (b"data: first\n\n", b"data: second\n\n"):
                self.wfile.write(f"{len(payload):x}\r\n".encode())
                self.wfile.write(payload + b"\r\n")
                self.wfile.flush()
                if not first_sent.is_set():
                    first_sent.set()
                    if not release.wait(5):
                        return
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    gateway = PrefillGateway(
        "127.0.0.1", 0, "127.0.0.1", upstream.server_port,
        tmp_path / "gateway.jsonl")
    gateway.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", gateway.server.server_port, timeout=5)
    received = threading.Event()
    observed = {}
    reader = None
    try:
        connection.request("GET", "/stream")
        response = connection.getresponse()

        def read_first():
            observed["line"] = response.readline()
            received.set()

        reader = threading.Thread(target=read_first, daemon=True)
        reader.start()
        assert first_sent.wait(2)
        assert received.wait(2), "gateway buffered the SSE stream until EOF"
        assert observed["line"] == b"data: first\n"
    finally:
        release.set()
        if reader:
            reader.join(5)
        connection.close()
        gateway.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(5)
