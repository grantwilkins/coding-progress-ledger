from __future__ import annotations

import asyncio
import builtins
import threading
from types import SimpleNamespace

import torch

from lmcache_compat.connector_patch import (
    bypass_lmcache,
    independent_transaction,
    kv_major_attention_view,
    needs_ipc_safe_kv_allocator,
    patch_on_import,
    restore_page_major_attention,
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


def test_qwen_kv_major_attention_is_reviewed_without_copy():
    allocation = torch.arange(
        2 * 99 * 16 * 2 * 4, dtype=torch.bfloat16).reshape(2, 99, 16, 2, 4)
    kernel = allocation[:, :98]
    assert not kernel.is_contiguous()
    assert kernel[0].is_contiguous() and kernel[1].is_contiguous()
    spec = SimpleNamespace(
        block_size=784,
        page_size_bytes=2 * 784 * 2 * 4 * kernel.element_size(),
    )

    viewed = kv_major_attention_view(spec, kernel)

    assert viewed.shape == (2, 2, 784, 2, 4)
    assert viewed.untyped_storage().data_ptr() == kernel.untyped_storage().data_ptr()
    assert viewed[:, 0].is_contiguous() and viewed[:, 1].is_contiguous()
    assert torch.equal(viewed[1, 0, 0], kernel[0, 49, 0])
    assert torch.equal(viewed[1, 1, -1], kernel[1, 97, -1])


def test_qwen_transposed_page_major_attention_is_restored_without_copy():
    page_major = torch.arange(
        98 * 2 * 16 * 4 * 256, dtype=torch.bfloat16,
    ).reshape(98, 2, 16, 4, 256)
    kv_major = page_major.permute(1, 0, 2, 3, 4)

    assert kv_major.shape == (2, 98, 16, 4, 256)
    assert kv_major.stride() == (16384, 32768, 1024, 256, 1)
    assert not kv_major.is_contiguous()
    assert not kv_major[0].is_contiguous()

    restored = restore_page_major_attention(kv_major)

    assert restored is not None and restored.is_contiguous()
    assert restored.shape == page_major.shape
    assert restored.untyped_storage().data_ptr() == kv_major.untyped_storage().data_ptr()
    assert torch.equal(restored, page_major)


def test_only_sleeping_lmcache_driven_mp_uses_ipc_safe_kv_allocator():
    def config(*, sleep=True, cumem=True, connector="LMCacheMPConnector",
               mode="lmcache_driven"):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                enable_sleep_mode=sleep,
                enable_cumem_allocator=cumem,
            ),
            kv_transfer_config=SimpleNamespace(
                kv_connector=connector,
                kv_connector_extra_config={
                    "lmcache.mp.mp_transfer_mode": mode,
                },
            ),
        )

    assert needs_ipc_safe_kv_allocator(config())
    assert not needs_ipc_safe_kv_allocator(config(sleep=False))
    assert not needs_ipc_safe_kv_allocator(config(cumem=False))
    assert not needs_ipc_safe_kv_allocator(config(mode="engine_driven"))
    assert not needs_ipc_safe_kv_allocator(config(connector="OtherConnector"))
