from __future__ import annotations

import asyncio
import builtins
import socket
import threading


def bypass_lmcache(request) -> bool:
    return bool((getattr(request, "kv_transfer_params", None) or {}).get("qh_bypass_lmcache"))


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
