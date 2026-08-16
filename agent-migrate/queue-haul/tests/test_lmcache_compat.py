from __future__ import annotations

import asyncio
import builtins
import threading
import pytest
import torch

from lmcache_compat.connector_patch import (
    bypass_lmcache,
    independent_transaction,
    kv_first_attention_block_view,
    patch_attention_kv_layout,
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


def test_kv_first_attention_view_restores_block_major_bytes_without_copy():
    backing = torch.arange(3 + 4 * 2 * 16 * 2 * 3, dtype=torch.float32)
    block_major = torch.as_strided(backing, (4, 2, 16, 2, 3),
                                   (192, 96, 6, 3, 1), 3)
    cache = block_major.as_strided((2, 4, 16, 2, 3),
                                   (96, 192, 6, 3, 1), 3)

    edited = kv_first_attention_block_view(cache)

    assert edited.shape == (4, 2, 16, 2, 3)
    assert edited.stride() == (192, 96, 6, 3, 1)
    assert edited.storage_offset() == cache.storage_offset() == 3
    assert edited.data_ptr() == cache.data_ptr()
    assert torch.equal(edited, block_major)
    assert edited[2, 1, 7, 1, 2] == cache[1, 2, 7, 1, 2]


@pytest.mark.parametrize("cache,error", [
    (torch.empty(2, 4, 8, 2, 3), r"\[2, NB, 16, NH, HS\]"),
    (torch.empty(1, 4, 16, 2, 3), r"\[2, NB, 16, NH, HS\]"),
    (torch.empty(2, 4, 16, 2), r"\[2, NB, 16, NH, HS\]"),
    (torch.empty(2, 0, 16, 2, 3), r"\[2, NB, 16, NH, HS\]"),
    (torch.empty(2, 4, 16, 2, 3), "stride"),
])
def test_kv_first_attention_view_rejects_non_vllm_layout(cache, error):
    with pytest.raises(ValueError, match=error):
        kv_first_attention_block_view(cache)


def test_kv_first_patch_delegates_block_major_view(monkeypatch):
    from lmcache.integration.vllm.kv_cache_group_edits import (
        _SubpagedAttentionViewEdit,
    )

    seen = []
    monkeypatch.delattr(_SubpagedAttentionViewEdit, "_qh_kv_first_patched",
                        raising=False)
    monkeypatch.setattr(_SubpagedAttentionViewEdit, "apply",
                        lambda _self, spec, cache: seen.append((spec, cache)))
    patch_attention_kv_layout()
    block_major = torch.arange(4 * 2 * 16 * 2 * 3).reshape(4, 2, 16, 2, 3)
    cache = block_major.as_strided((2, 4, 16, 2, 3),
                                   (96, 192, 6, 3, 1))
    spec = object()

    assert _SubpagedAttentionViewEdit().apply(spec, cache) is None
    assert seen[0][0] is spec
    assert seen[0][1].data_ptr() == cache.data_ptr()
    assert torch.equal(seen[0][1], block_major)


def test_kv_first_attention_view_rejects_truncated_storage():
    cache = torch.empty(4, 2, 16, 2, 3).as_strided(
        (2, 4, 16, 2, 3), (96, 192, 6, 3, 1),
    )
    cache.untyped_storage().resize_(cache.untyped_storage().nbytes() - 4)
    with pytest.raises(ValueError, match="backing storage"):
        kv_first_attention_block_view(cache)



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
