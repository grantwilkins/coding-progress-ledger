"""Private HTTP gateway with a live aggregate uncached-prefill completion cap."""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


CONTROL_PATH = "/qh/prefill-control"
THROTTLED_CLASSES = {"background", "replay"}


def _usage(body: bytes) -> tuple[int, int]:
    prompt = cached = 0
    for line in body.splitlines():
        if not line.strip().startswith(b"data:"):
            continue
        value = line.strip()[5:].strip()
        if value == b"[DONE]":
            continue
        try:
            usage = (json.loads(value).get("usage") or {})
        except json.JSONDecodeError:
            continue
        prompt = int(usage.get("prompt_tokens", prompt))
        cached = int((usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", cached))
    return prompt, cached


class PrefillCompletionLimiter:
    def __init__(self):
        self.tokens_per_s: float | None = None
        self.next_completion = time.monotonic()
        self.lock = threading.Lock()

    def update(self, tokens_per_s: float | None) -> dict:
        if tokens_per_s is not None and tokens_per_s <= 0:
            raise ValueError("prefill capacity must be positive")
        with self.lock:
            self.tokens_per_s = tokens_per_s
            self.next_completion = time.monotonic()
            return self.snapshot()

    def snapshot(self) -> dict:
        return {"tokens_per_s": self.tokens_per_s}

    def wait(self, tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        with self.lock:
            rate = self.tokens_per_s
            if rate is None:
                return 0.0
            now = time.monotonic()
            start = max(now, self.next_completion)
            self.next_completion = start + tokens / rate
            delay = self.next_completion - now
        if delay > 0:
            time.sleep(delay)
        return delay


class PrefillGateway:
    def __init__(self, bind_host: str, bind_port: int, upstream_host: str,
                 upstream_port: int, log: Path):
        self.limiter = PrefillCompletionLimiter()
        self.upstream = upstream_host, upstream_port
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log.open("w", buffering=1)
        self.log_lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                return

            def _body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length", "0")))

            def _write(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def _control(self, body: bytes) -> None:
                try:
                    raw = json.loads(body or b"{}")
                    value = raw.get("tokens_per_s")
                    state = owner.limiter.update(
                        None if value is None else float(value))
                    result = {"ok": True, **state}
                    owner.record("control", **state)
                    status = 200
                except Exception as exc:
                    result = {"ok": False,
                              "error": f"{type(exc).__name__}: {exc}"}
                    status = 400
                self._write(status, json.dumps(result).encode(), "application/json")

            def _proxy(self, body: bytes) -> None:
                started = time.monotonic_ns()
                headers = {
                    key: value for key, value in self.headers.items()
                    if key.lower() not in {
                        "host", "content-length", "connection",
                        "transfer-encoding",
                    }
                }
                headers["Content-Length"] = str(len(body))
                connection = http.client.HTTPConnection(
                    *owner.upstream, timeout=900)
                connection.request(self.command, self.path, body, headers)
                response = connection.getresponse()
                response_headers = response.getheaders()
                request_class = self.headers.get("X-QH-Prefill-Class", "")
                if request_class not in THROTTLED_CLASSES:
                    self.send_response(response.status)
                    for key, value in response_headers:
                        if key.lower() not in {"content-length", "connection",
                                               "transfer-encoding"}:
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    transferred = 0
                    # HTTPResponse.read(n) waits for all n bytes (or EOF),
                    # which collapses a short SSE response into one burst.
                    # read1 performs at most one underlying buffered read, so
                    # token events are forwarded as the upstream emits them.
                    while chunk := response.read1(64 * 1024):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        transferred += len(chunk)
                    connection.close()
                    owner.record(
                        "request", request_class=request_class,
                        response_bytes=transferred, throttle_delay_s=0.0,
                        started_ns=started, ended_ns=time.monotonic_ns(),
                        status=response.status,
                    )
                    return
                payload = response.read()
                connection.close()
                prompt, cached = _usage(payload)
                uncached = max(0, prompt - cached)
                delay = owner.limiter.wait(uncached)
                self.send_response(response.status)
                for key, value in response_headers:
                    if key.lower() not in {"content-length", "connection",
                                           "transfer-encoding"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                owner.record(
                    "request", request_class=request_class,
                    prompt_tokens=prompt, cached_tokens=cached,
                    uncached_tokens=uncached, throttle_delay_s=delay,
                    started_ns=started, ended_ns=time.monotonic_ns(),
                    status=response.status,
                )

            def do_GET(self):
                self._proxy(b"")

            def do_POST(self):
                body = self._body()
                if self.path == CONTROL_PATH:
                    self._control(body)
                else:
                    self._proxy(body)

        self.server = ThreadingHTTPServer((bind_host, bind_port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True,
            name="qh-prefill-gateway")

    def record(self, event: str, **fields) -> None:
        with self.log_lock:
            self.log.write(json.dumps({
                "event": event, "monotonic_ns": time.monotonic_ns(),
                "wall_ns": time.time_ns(), **fields,
            }, separators=(",", ":")) + "\n")

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(30)
        self.log.close()
