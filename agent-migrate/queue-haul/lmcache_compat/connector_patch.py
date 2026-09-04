from __future__ import annotations

import asyncio
import builtins
from contextlib import nullcontext
import json
import os
import socket
import threading
import time


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


def kv_geometry_registration(manager, chunk_tokens: int) -> dict:
    """Describe bytes from live, registered LMCache kernel groups.

    The manager was built from the actual device tensors after LMCache's
    zero-copy group edits. Consequently this records physical slots and
    object payload sizes instead of reconstructing them from a model config.
    """
    if chunk_tokens <= 0 or not manager.kernel_groups:
        raise ValueError("KV geometry needs a positive chunk and live groups")
    object_by_kernel = {
        kernel: object_index
        for object_index, group in enumerate(manager.object_groups)
        for kernel in group.kernel_group_indices
    }
    if set(object_by_kernel) != set(range(len(manager.kernel_groups))):
        raise ValueError("each KV kernel group must belong to one object group")
    groups = []
    for index, group in enumerate(manager.kernel_groups):
        shape = group.shape_desc
        scalar = int(shape.kv_size) * int(group.num_layers) \
            * int(group.hidden_dim_size) * int(shape.element_size)
        block_bytes = scalar * int(shape.bs)
        transfer_slots = int(manager.get_slots_per_chunk_in_sw(index))
        groups.append({
            "group": f"kernel-{index}:engine-{int(group.engine_group_idx)}",
            "kernel_group": index,
            "engine_group": int(group.engine_group_idx),
            "object_group": object_by_kernel[index],
            "layer_indices": list(map(int, group.layer_indices)),
            "tokens_per_block": int(group.tokens_per_block),
            "slots_per_block": int(group.slots_per_block),
            "num_blocks": int(shape.nb),
            "block_bytes": block_bytes,
            "capacity_bytes": int(shape.nb) * block_bytes,
            "chunk_bytes": transfer_slots * scalar,
        })
    objects = []
    for index, group in enumerate(manager.object_groups):
        kernels = list(map(int, group.kernel_group_indices))
        objects.append({
            "object_group": index,
            "kernel_groups": kernels,
            "sw_size_chunks": int(group.sw_size_chunks),
            "chunk_bytes": sum(groups[kernel]["chunk_bytes"]
                               for kernel in kernels),
        })
    return {
        "schema": "queue-haul-live-kv-geometry-v1",
        "chunk_tokens": int(chunk_tokens),
        "groups": groups,
        "object_groups": objects,
    }


def registered_kv_geometry(connector) -> dict:
    """Rebuild the server's metadata-only grouping over registered views."""
    from lmcache.integration.vllm.utils import vllm_layout_hints
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import (
        normalize_and_discover_per_layer_formats,
    )
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
    from lmcache.v1.multiprocess.group_view import engine_group_layer_indices

    adapter = connector.worker_adapter
    infos = adapter.engine_group_infos
    caches = list(adapter.kv_caches.values())
    normalized, formats = normalize_and_discover_per_layer_formats(
        caches, engine_group_layer_indices(infos), EngineType.VLLM,
        vllm_layout_hints(),
    )
    chunk_tokens = int(adapter.lmcache_tokens_per_chunk)
    manager = KVLayerGroupsManager(
        normalized, formats, infos, chunk_tokens,
        separate_object_groups=(
            os.environ.get("QH_LMCACHE_SEPARATE_OBJECT_GROUPS") == "1"
        ),
    )
    return kv_geometry_registration(manager, chunk_tokens)


def needs_ipc_safe_kv_allocator(vllm_config) -> bool:
    """Whether LMCache must export a sleep-enabled engine's KV tensors.

    vLLM's CUDA virtual-memory allocator is required for sleepable model
    weights, but neither PyTorch CUDA IPC nor ``cudaIpcGetMemHandle`` can
    export allocations from that pool.  LMCache's ``lmcache_driven`` mode
    needs those IPC handles because its server operates directly on the
    engine KV tensors.
    """
    model = getattr(vllm_config, "model_config", None)
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    extra = getattr(transfer, "kv_connector_extra_config", {}) or {}
    return bool(
        getattr(model, "enable_sleep_mode", False)
        and getattr(model, "enable_cumem_allocator", False)
        and getattr(transfer, "kv_connector", None) == "LMCacheMPConnector"
        and extra.get("lmcache.mp.mp_transfer_mode") == "lmcache_driven"
    )


def patch_sleep_compatible_kv_allocator(logger) -> None:
    """Keep LMCache-visible KV tensors IPC-exportable under sleep mode.

    Only the KV allocation context is changed.  Model weights continue to
    use vLLM's tagged CuMem pool and therefore retain level-1 sleep/wake
    behavior.  The scheduler still clears its logical KV state on sleep;
    the standard-allocator KV backing remains resident so the LMCache server's
    CUDA IPC mapping stays valid across wake-up.
    """
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, "_qh_ipc_safe_kv_allocator_patched", False):
        return
    original = Worker._maybe_get_memory_pool_context

    def memory_pool(self, tag):
        if tag == "kv_cache" and needs_ipc_safe_kv_allocator(self.vllm_config):
            logger.info(
                "QH_IPC_SAFE_KV_ALLOCATOR standard PyTorch KV allocation; "
                "vLLM CuMem remains enabled for sleepable weights"
            )
            return nullcontext()
        return original(self, tag)

    Worker._maybe_get_memory_pool_context = memory_pool
    Worker._qh_ipc_safe_kv_allocator_patched = True


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
        if not multimodal:
            gemma4_layer_configs(root)
        same_text = multimodal or all(
            getattr(root, name) == getattr(text, name)
            for name in ("hidden_size", "num_hidden_layers", "vocab_size")
        )
        if (getattr(root, "text_config", None) is not text if multimodal
                else not same_text) \
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

    patch_sleep_compatible_kv_allocator(logger)
    if os.environ.get("QH_MODEL") == "google/gemma-4-26B-A4B-it":
        patch_gemma4_config(logger)
        patch_gemma4_decoder()
    patch_attention_kv_layout()
    if getattr(LMCacheMPConnector, "_qh_bypass_patched", False):
        return
    original_lookup = LMCacheMPConnector.get_num_new_matched_tokens
    original_register = LMCacheMPConnector.register_kv_caches
    original_update = LMCacheMPConnector.update_state_after_alloc

    def lookup(self, request, num_computed_tokens):
        if not bypass_lmcache(request):
            return original_lookup(self, request, num_computed_tokens)
        tracker = self._get_or_create_request_tracker(request)
        tracker.num_stored_tokens = 2**63
        logger.info("Reqid: %s, Total tokens %d, LMCache hit tokens: 0",
                    request.request_id, request.num_tokens)
        return 0, False

    def register(self, kv_caches):
        result = register_verified_kv_caches(
            lambda caches: original_register(self, caches), kv_caches, logger,
        )
        if os.environ.get("QH_KV_GEOMETRY_EVIDENCE") == "1":
            logger.info("QH_KV_GEOMETRY %s", json.dumps(
                registered_kv_geometry(self), sort_keys=True,
                separators=(",", ":"),
            ))
        return result

    def update(self, request, blocks, num_external_tokens):
        result = original_update(self, request, blocks, num_external_tokens)
        if os.environ.get("QH_KV_GEOMETRY_EVIDENCE") == "1":
            tracker = self._get_request_tracker(request.request_id)
            logger.info("QH_KV_ALLOCATION %s", json.dumps({
                "schema": "queue-haul-live-kv-allocation-v1",
                "monotonic_ns": time.monotonic_ns(),
                "request_id": request.request_id,
                "prompt_tokens": int(request.num_prompt_tokens),
                "tokens": int(request.num_tokens),
                "output_tokens": int(request.num_output_tokens),
                "external_tokens": int(num_external_tokens),
                "blocks": {str(group): int(count) for group, count in
                           sorted(tracker.num_allocated_blocks().items())},
            }, sort_keys=True, separators=(",", ":")))
        return result

    LMCacheMPConnector.get_num_new_matched_tokens = lookup
    LMCacheMPConnector.register_kv_caches = register
    LMCacheMPConnector.update_state_after_alloc = update
    LMCacheMPConnector._qh_bypass_patched = True
    LMCacheMPConnector._qh_kv_dtype_registration_patched = True


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
