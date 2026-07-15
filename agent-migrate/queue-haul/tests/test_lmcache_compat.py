from __future__ import annotations

import asyncio
import builtins

import pytest

from lmcache_compat.connector_patch import bypass_lmcache, patch_on_import, recv_exact, transaction


class FragmentedSocket:
    def __init__(self, responses):
        self.responses = responses
        self.pending = bytearray()
        self.sent = []

    def sendall(self, request):
        assert not self.pending
        self.sent.append(request)
        self.pending.extend(self.responses[request])

    def recv_into(self, view):
        count = min(len(view), len(self.pending), 2)
        view[:count] = self.pending[:count]
        del self.pending[:count]
        return count


def test_replay_bypass_is_explicit():
    request = type("Request", (), {"kv_transfer_params": {"qh_bypass_lmcache": True}})()
    assert bypass_lmcache(request)
    assert not bypass_lmcache(type("Request", (), {"kv_transfer_params": None})())


def test_adapter_patch_is_deferred_until_import():
    original = builtins.__import__
    calls = []
    try:
        patch_on_import("json", lambda: calls.append(True))
        assert not calls
        __import__("json")
        assert calls and builtins.__import__ is original
    finally:
        builtins.__import__ = original


@pytest.mark.parametrize("calls", [((b"c", False), (b"x", False)), ((b"g", True), (b"h", True)), ((b"c", False), (b"g", True), (b"x", False), (b"h", True))])
def test_fragmented_metadata_and_concurrent_operations_are_serialized(calls):
    async def run():
        sock = FragmentedSocket({b"c": b"YES", b"g": b"004data", b"x": b"NO!", b"h": b"003end"})
        lock = asyncio.Lock()
        recoveries = []

        async def call(request, body):
            return await transaction(
                lock, lambda: sock, lambda: recoveries.append(request), request, 3, bytes,
                lambda meta: (meta, recv_exact(sock, int(meta))) if body else meta,
            )

        results = await asyncio.gather(*(call(*item) for item in calls))
        expected = {b"c": b"YES", b"g": (b"004", b"data"), b"x": b"NO!", b"h": (b"003", b"end")}
        assert results == [expected[request] for request, _body in calls]
        assert sock.sent == [request for request, _body in calls]
        assert not recoveries

    asyncio.run(run())


def test_protocol_eof_reconnects_and_retries_the_operation():
    async def run():
        sockets = [FragmentedSocket({b"get": b"x"}), FragmentedSocket({b"get": b"YES"})]
        current = 0

        def recover():
            nonlocal current
            current += 1

        assert await transaction(asyncio.Lock(), lambda: sockets[current], recover, b"get", 3, bytes) == b"YES"
        assert current == 1

    asyncio.run(run())
