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


def kv_first_attention_block_view(kv_cache):
    shape = tuple(kv_cache.shape)
    if kv_cache.ndim != 5 or shape[0] != 2 or shape[2] != 16 \
            or any(size <= 0 for size in shape):
        raise ValueError(f"expected K/V-first attention KV [2, NB, 16, NH, HS], got {shape}")
    _, blocks, kernel, heads, head_size = shape
    inner = heads * head_size
    hidden = kernel * inner
    expected = (hidden, 2 * hidden, inner, head_size, 1)
    if tuple(kv_cache.stride()) != expected:
        raise ValueError(f"expected vLLM K/V-first stride {expected}, got {kv_cache.stride()}")
    offset = kv_cache.storage_offset()
    storage_bytes = kv_cache.untyped_storage().nbytes()
    if storage_bytes % kv_cache.element_size() \
            or offset < 0 or offset + 2 * blocks * hidden > storage_bytes // kv_cache.element_size():
        raise ValueError("K/V-first attention KV exceeds its backing storage")
    return kv_cache.as_strided(
        (blocks, 2, kernel, heads, head_size),
        (2 * hidden, hidden, inner, head_size, 1),
        offset,
    )


def verify_kv_cache_dtypes(kv_caches):
    import torch

    if not kv_caches:
        raise RuntimeError("no KV caches were registered")
    attention, recurrent = 0, {}
    for name, cache in kv_caches.items():
        if isinstance(cache, torch.Tensor):
            if cache.dtype != torch.bfloat16:
                raise RuntimeError(f"attention KV cache {name} is {cache.dtype}, not torch.bfloat16")
            attention += 1
            continue
        if not isinstance(cache, list) or not cache:
            raise RuntimeError(f"KV cache {name} has an unsupported tensor group")
        dtypes = []
        for index, tensor in enumerate(cache):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                dtype = getattr(tensor, "dtype", type(tensor).__name__)
                raise RuntimeError(f"recurrent KV cache {name}[{index}] is unsupported: {dtype}")
            dtypes.append(str(tensor.dtype))
        signature = "+".join(dtypes)
        recurrent[signature] = recurrent.get(signature, 0) + 1
    if not attention:
        raise RuntimeError("no attention KV caches were registered")
    return attention, ",".join(
        f"{signature}:{count}" for signature, count in sorted(recurrent.items())
    ) or "none"


def register_verified_kv_caches(register, kv_caches, logger):
    attention, recurrent = verify_kv_cache_dtypes(kv_caches)
    result = register(kv_caches)
    logger.info("QH_KV_CACHE_DTYPES_VERIFIED attention=torch.bfloat16:%d recurrent=%s",
                attention, recurrent)
    return result


def gemma4_layer_configs(config):
    layers = tuple(getattr(config, "per_layer_config", ()))
    expected = tuple(
        ("full_attention", 512, 2, 16) if (index + 1) % 6 == 0
        else ("sliding_attention", 256, 8, 16)
        for index in range(30)
    )
    observed = tuple(
        (layer.layer_types[index], layer.head_dim,
         layer.num_key_value_heads, layer.num_attention_heads)
        for index, layer in enumerate(layers)
    )
    if observed != expected:
        raise RuntimeError(f"unexpected Gemma4 per-layer KV geometry: {observed}")
    return layers


def patch_gemma4_decoder() -> None:
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
    from vllm.model_executor.models.utils import extract_layer_index

    if getattr(Gemma4DecoderLayer, "_qh_heterogeneous_patched", False):
        return
    original_init = Gemma4DecoderLayer.__init__

    def initialize(self, config, cache_config=None, quant_config=None, prefix=""):
        layers = gemma4_layer_configs(config)
        return original_init(self, layers[extract_layer_index(prefix)], cache_config,
                             quant_config, prefix)

    Gemma4DecoderLayer.__init__ = initialize
    Gemma4DecoderLayer._qh_heterogeneous_patched = True


def patch_gemma4_config(logger) -> None:
    from vllm.config.model import ModelConfig
    from vllm.model_executor.models.config import Gemma4Config
    from vllm.transformers_utils.model_arch_config_convertor import (
        Gemma4ModelArchConfigConvertor,
    )

    if getattr(Gemma4ModelArchConfigConvertor, "_qh_heterogeneous_patched", False):
        return
    original_backend = ModelConfig._get_transformers_backend_cls
    def geometry(self):
        layers = gemma4_layer_configs(self.hf_text_config)
        if not getattr(self, "_qh_geometry_logged", False):
            logger.info("QH_GEMMA4_GEOMETRY_VERIFIED "
                        "sliding=25x(head_dim=256,kv_heads=8) "
                        "full=5x(head_dim=512,kv_heads=2)")
            self._qh_geometry_logged = True
        return layers

    def verify(vllm_config):
        gemma4_layer_configs(vllm_config.model_config.hf_text_config)
        if vllm_config.attention_config.backend is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info("Gemma4 exact heterogeneous geometry requires TRITON_ATTN")

    def transformers_backend(self):
        root, text = self.hf_config, self.hf_text_config
        model_type = getattr(root, "model_type", None)
        if model_type not in {"gemma4", "gemma4_text"}:
            return original_backend(self)
        gemma4_layer_configs(text)
        multimodal = model_type == "gemma4"
        expected_arch = "Gemma4ForConditionalGeneration" if multimodal \
            else "Gemma4ForCausalLM"
        if (getattr(root, "text_config", None) is not text if multimodal
                else root is not text) \
                or getattr(text, "model_type", None) != "gemma4_text" \
                or self.architectures != [expected_arch] \
                or self.runner not in {"auto", "generate"}:
            raise RuntimeError("unexpected Gemma4 multimodal model structure")
        return f"Transformers{'MultiModal' if multimodal else ''}" \
               f"{'MoE' if self.is_moe else ''}ForCausalLM"

    Gemma4ModelArchConfigConvertor.get_head_size = \
        lambda self: max(layer.head_dim for layer in geometry(self))
    Gemma4ModelArchConfigConvertor.get_total_num_kv_heads = \
        lambda self: max(layer.num_key_value_heads for layer in geometry(self))
    Gemma4ModelArchConfigConvertor.get_total_num_attention_heads = \
        lambda self: max(layer.num_attention_heads for layer in geometry(self))
    Gemma4Config.verify_and_update_config = staticmethod(verify)
    ModelConfig._get_transformers_backend_cls = transformers_backend
    Gemma4ModelArchConfigConvertor._qh_heterogeneous_patched = True
    Gemma4Config._qh_heterogeneous_patched = True
    ModelConfig._qh_gemma4_backend_patched = True


def patch_attention_kv_layout() -> None:
    from lmcache.integration.vllm.kv_cache_group_edits import (
        _SubpagedAttentionViewEdit,
    )

    if getattr(_SubpagedAttentionViewEdit, "_qh_kv_first_patched", False):
        return
    original = _SubpagedAttentionViewEdit.apply

    def apply(self, spec, kv_cache):
        if getattr(kv_cache, "ndim", 0) == 5 and kv_cache.shape[0] == 2:
            return original(self, spec, kv_first_attention_block_view(kv_cache))
        return original(self, spec, kv_cache)

    _SubpagedAttentionViewEdit.apply = apply
    _SubpagedAttentionViewEdit._qh_kv_first_patched = True



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

    if os.environ.get("QH_MODEL") == "google/gemma-4-26B-A4B-it":
        patch_gemma4_config(logger)
        patch_gemma4_decoder()
    patch_attention_kv_layout()
    if getattr(LMCacheMPConnector, "_qh_bypass_patched", False):
        return
    original_lookup = LMCacheMPConnector.get_num_new_matched_tokens
    original_register = LMCacheMPConnector.register_kv_caches

    def lookup(self, request, num_computed_tokens):
        if not bypass_lmcache(request):
            return original_lookup(self, request, num_computed_tokens)
        tracker = self._get_or_create_request_tracker(request)
        tracker.num_stored_tokens = 2**63
        logger.info("Reqid: %s, Total tokens %d, LMCache hit tokens: 0",
                    request.request_id, request.num_tokens)
        return 0, False

    def register(self, kv_caches):
        return register_verified_kv_caches(
            lambda caches: original_register(self, caches), kv_caches, logger,
        )

    LMCacheMPConnector.get_num_new_matched_tokens = lookup
    LMCacheMPConnector.register_kv_caches = register
    LMCacheMPConnector._qh_bypass_patched = True
    LMCacheMPConnector._qh_kv_dtype_registration_patched = True


if os.environ.get("QH_LMCACHE_MODE") == "mp":
    patch_mp_connector()
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
