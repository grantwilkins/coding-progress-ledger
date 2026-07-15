from __future__ import annotations

import builtins
import socket


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


async def transaction(lock, socket_getter, recover, request: bytes, header_size: int, parse, receive=lambda meta: meta):
    async with lock:
        for attempt in range(2):
            try:
                sock = socket_getter()
                sock.sendall(request)
                return receive(parse(recv_exact(sock, header_size)))
            except Exception:
                recover()
                if attempt:
                    raise


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

    def request(command, key):
        return ClientMetaMessage(
            command,
            key,
            0,
            MemoryFormat(1),
            torch.float16,
            torch.Size([0, 0, 0, 0]),
        ).serialize()

    def reconnect(self):
        self.client_socket.close()
        self.client_socket = socket.create_connection(self._qh_address)

    async def exists(self, key):
        meta = await transaction(
            self.async_socket_lock,
            lambda: self.client_socket,
            lambda: reconnect(self),
            request(Constants.CLIENT_EXIST, key),
            ServerMetaMessage.packlength(),
            ServerMetaMessage.deserialize,
        )
        return meta.code == Constants.SERVER_SUCCESS

    async def get(self, key):
        def receive(meta):
            if meta.code != Constants.SERVER_SUCCESS:
                return None
            memory = self.receive_all(meta)
            if memory is None:
                raise ConnectionError("connection closed during LMCache body")
            return memory

        return await transaction(
            self.async_socket_lock,
            lambda: self.client_socket,
            lambda: reconnect(self),
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
