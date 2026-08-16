from __future__ import annotations

import asyncio
import builtins
import threading
from types import SimpleNamespace

import pytest
import torch

from lmcache_compat.connector_patch import (
    bypass_lmcache,
    independent_transaction,
    kv_first_attention_page_view,
    patch_on_import,
)


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

    mp_request = type("Request", (), {
        "sampling_params": type("Sampling", (), {"extra_args": {
            "kv_transfer_params": {"qh_bypass_lmcache": True},
        }})(),
    })()
    assert bypass_lmcache(mp_request)
    mp_request.sampling_params.extra_args = {"qh_bypass_lmcache": 1}
    assert bypass_lmcache(mp_request)


def test_kv_first_attention_pages_merge_without_copy_or_group_flattening():
    cache = torch.arange(2 * 6 * 2 * 2, dtype=torch.float32).view(2, 6, 2, 1, 2)
    spec = SimpleNamespace(block_size=4, page_size_bytes=64)

    edited = kv_first_attention_page_view(spec, cache)

    assert edited.shape == (2, 3, 4, 1, 2)
    assert edited.data_ptr() == cache.data_ptr()
    assert torch.equal(edited[0, 1].flatten(), cache[0, 2:4].flatten())
    assert torch.equal(edited[1, 2].flatten(), cache[1, 4:6].flatten())
    with pytest.raises(ValueError, match="tile"):
        kv_first_attention_page_view(
            SimpleNamespace(block_size=4, page_size_bytes=128), cache,
        )
    with pytest.raises(ValueError, match="contiguous"):
        kv_first_attention_page_view(spec, torch.empty(2, 6, 4, 1, 2)[:, :, ::2])



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
