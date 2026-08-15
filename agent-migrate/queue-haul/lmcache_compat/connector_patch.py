from __future__ import annotations

import asyncio
import builtins
import os
import socket
import threading


def bypass_lmcache(request) -> bool:
    direct = getattr(request, "kv_transfer_params", None)
    extra = getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
    return bool(extra.get("qh_bypass_lmcache") or
                (direct or extra.get("kv_transfer_params") or {}).get("qh_bypass_lmcache"))



def patch_on_import(module: str, patch) -> None:
    original = builtins.__import__

    def import_(name, *args, **kwargs):
        result = original(name, *args, **kwargs)
        if name == module:
            builtins.__import__ = original
            patch()
        return result

    builtins.__import__ = import_


def recv_exact(sock, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    while view:
        count = sock.recv_into(view)
        if not count:
            raise ConnectionError(f"connection closed with {len(view)} of {size} bytes missing")
        view = view[count:]
    return bytes(data)


def exchange(address, request: bytes, header_size: int, parse,
             receive=lambda _sock, meta: meta):
    for attempt in range(2):
        try:
            with socket.create_connection(address) as sock:
                sock.sendall(request)
                return receive(sock, parse(recv_exact(sock, header_size)))
        except Exception:
            if attempt:
                raise


async def independent_transaction(address, request: bytes, header_size: int,
                                  parse, receive=lambda _sock, meta: meta):
    return await asyncio.to_thread(
        exchange, address, request, header_size, parse, receive,
    )


def patch_lmcache() -> None:
    import torch
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.protocol import ClientMetaMessage, Constants, ServerMetaMessage
    from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector

    if getattr(LMCServerConnector, "_qh_patched", False):
        return
    original_init = LMCServerConnector.__init__

    def initialize(self, host, port, loop, local_cpu_backend):
        original_init(self, host, port, loop, local_cpu_backend)
        self._qh_address = host, port
        self._qh_allocate_lock = threading.Lock()

    def request(command, key):
        return ClientMetaMessage(
            command,
            key,
            0,
            MemoryFormat(1),
            torch.float16,
            torch.Size([0, 0, 0, 0]),
        ).serialize()

    async def exists(self, key):
        meta = await independent_transaction(
            self._qh_address,
            request(Constants.CLIENT_EXIST, key),
            ServerMetaMessage.packlength(),
            ServerMetaMessage.deserialize,
        )
        return meta.code == Constants.SERVER_SUCCESS

    async def get(self, key):
        def receive(sock, meta):
            if meta.code != Constants.SERVER_SUCCESS:
                return None
            with self._qh_allocate_lock:
                memory = self.local_cpu_backend.allocate(
                    meta.shape, meta.dtype, meta.fmt,
                )
            if memory is None:
                return None
            view = memoryview(memory.byte_array)
            while view:
                count = sock.recv_into(view)
                if not count:
                    raise ConnectionError(
                        f"connection closed with {len(view)} body bytes missing"
                    )
                view = view[count:]
            return memory

        return await independent_transaction(
            self._qh_address,
            request(Constants.CLIENT_GET, key),
            ServerMetaMessage.packlength(),
            ServerMetaMessage.deserialize,
            receive,
        )

    LMCServerConnector.__init__ = initialize
    LMCServerConnector.exists = exists
    LMCServerConnector.get = get
    LMCServerConnector._qh_patched = True


def patch_adapter() -> None:
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl, logger

    if getattr(LMCacheConnectorV1Impl, "_qh_bypass_patched", False):
        return
    original_lookup = LMCacheConnectorV1Impl.get_num_new_matched_tokens

    def lookup(self, request, num_computed_tokens):
        if not bypass_lmcache(request):
            return original_lookup(self, request, num_computed_tokens)
        logger.info(
            "Reqid: %s, Total tokens %d, LMCache hit tokens: 0, need to load: %d",
            request.request_id,
            request.num_tokens,
            -num_computed_tokens,
        )
        return 0

    LMCacheConnectorV1Impl.get_num_new_matched_tokens = lookup
    LMCacheConnectorV1Impl._qh_bypass_patched = True


def patch_mp_connector() -> None:
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector, logger

    if getattr(LMCacheMPConnector, "_qh_bypass_patched", False):
        return
    original_lookup = LMCacheMPConnector.get_num_new_matched_tokens

    def lookup(self, request, num_computed_tokens):
        if not bypass_lmcache(request):
            return original_lookup(self, request, num_computed_tokens)
        tracker = self._get_or_create_request_tracker(request)
        tracker.num_stored_tokens = 2**63
        logger.info("Reqid: %s, Total tokens %d, LMCache hit tokens: 0",
                    request.request_id, request.num_tokens)
        return 0, False

    LMCacheMPConnector.get_num_new_matched_tokens = lookup
    LMCacheMPConnector._qh_bypass_patched = True


def kv_major_attention_view(spec, kv_cache):
    """Re-view a K/V-major paged tensor at its logical block size."""
    import torch

    if not (isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim == 5 and kv_cache.shape[0] == 2
            and kv_cache.shape[1] != 2):
        raise ValueError("expected a K/V-major five-dimensional attention tensor")
    logical_block_size = spec.block_size
    kernel_pages = kv_cache.shape[1]
    kernel_block_size = kv_cache.shape[2]
    if logical_block_size % kernel_block_size:
        raise ValueError(
            f"logical block size {logical_block_size} is not a multiple "
            f"of kernel block size {kernel_block_size}")
    ratio = logical_block_size // kernel_block_size
    if kernel_pages % ratio:
        raise ValueError(
            f"kernel page count {kernel_pages} is not a multiple of the "
            f"logical/kernel block ratio {ratio}")
    kernel_page_bytes = (2 * kv_cache[0, 0].numel()
                         * kv_cache.element_size())
    if kernel_page_bytes * ratio != spec.page_size_bytes:
        raise ValueError(
            f"{ratio} K/V-major kernel pages "
            f"({kernel_page_bytes * ratio} bytes) do not tile the "
            f"logical page ({spec.page_size_bytes} bytes)")
    if not kv_cache[0].is_contiguous() or not kv_cache[1].is_contiguous():
        raise ValueError(
            "K/V-major attention K and V planes must each be contiguous; "
            f"shape={tuple(kv_cache.shape)}, strides={kv_cache.stride()}, "
            f"K_contiguous={kv_cache[0].is_contiguous()}, "
            f"V_contiguous={kv_cache[1].is_contiguous()}")
    logical_pages = kernel_pages // ratio
    shape = (logical_pages, 2, logical_block_size,
             kv_cache.shape[3], kv_cache.shape[4])
    strides = (ratio * kv_cache.stride(1), kv_cache.stride(0),
               kv_cache.stride(2), kv_cache.stride(3),
               kv_cache.stride(4))
    viewed = kv_cache.as_strided(shape, strides)
    if not viewed[:, 0].is_contiguous() or not viewed[:, 1].is_contiguous():
        raise ValueError("K/V-major logical K and V views are not contiguous")
    return viewed


def restore_page_major_attention(kv_cache):
    """Undo a zero-copy ``(K/V, pages, ...)`` transpose when possible."""
    import torch

    if not (isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim == 5 and kv_cache.shape[0] == 2
            and kv_cache.shape[1] != 2):
        raise ValueError("expected a K/V-major five-dimensional attention tensor")
    page_major = kv_cache.permute(1, 0, 2, 3, 4)
    return page_major if page_major.is_contiguous() else None


def patch_kv_major_attention_groups() -> None:
    """Support vLLM's ``(K/V, pages, block, heads, dim)`` Qwen layout."""
    import torch
    from lmcache.integration.vllm import kv_cache_group_edits as edits

    cls = edits._SubpagedAttentionViewEdit
    if getattr(cls, "_qh_kv_major_patched", False):
        return
    original_apply = cls.apply

    def apply(self, spec, kv_cache):
        if not (isinstance(kv_cache, torch.Tensor)
                and kv_cache.ndim == 5 and kv_cache.shape[0] == 2
                and kv_cache.shape[1] != 2):
            return original_apply(self, spec, kv_cache)
        page_major = restore_page_major_attention(kv_cache)
        if page_major is not None:
            return original_apply(self, spec, page_major)
        return kv_major_attention_view(spec, kv_cache)

    cls.apply = apply
    cls._qh_kv_major_patched = True


if os.environ.get("QH_LMCACHE_MODE") == "mp":
    patch_kv_major_attention_groups()
    patch_mp_connector()
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
