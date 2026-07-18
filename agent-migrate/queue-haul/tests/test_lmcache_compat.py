from __future__ import annotations

import asyncio
import builtins
import threading

from lmcache_compat.connector_patch import bypass_lmcache, independent_transaction, patch_on_import


class FragmentedSocket:
    def __init__(self, responses):
        self.responses = responses
        self.pending = bytearray()

    def sendall(self, request):
        assert not self.pending
        self.pending.extend(self.responses[request])

    def recv_into(self, view):
        count = min(len(view), len(self.pending), 2)
        view[:count] = self.pending[:count]
        del self.pending[:count]
        return count

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


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


def test_protocol_eof_reconnects_and_retries_the_operation(monkeypatch):
    sockets = [
        FragmentedSocket({b"get": b"x"}),
        FragmentedSocket({b"get": b"YES"}),
    ]
    monkeypatch.setattr(
        "lmcache_compat.connector_patch.socket.create_connection",
        lambda _address: sockets.pop(0),
    )

    assert asyncio.run(
        independent_transaction(("host", 1), b"get", 3, bytes)
    ) == b"YES"


def test_independent_transactions_use_parallel_connections(monkeypatch):
    barrier = threading.Barrier(2)

    class ParallelSocket(FragmentedSocket):
        def recv_into(self, view):
            barrier.wait()
            return super().recv_into(view)

    sockets = [
        ParallelSocket({b"a": b"YES", b"b": b"YES"}),
        ParallelSocket({b"a": b"YES", b"b": b"YES"}),
    ]
    monkeypatch.setattr(
        "lmcache_compat.connector_patch.socket.create_connection",
        lambda _address: sockets.pop(),
    )

    async def run():
        return await asyncio.wait_for(asyncio.gather(*[
            independent_transaction(("host", 1), request, 3, bytes)
            for request in (b"a", b"b")
        ]), 1)

    assert asyncio.run(run()) == [b"YES", b"YES"]
