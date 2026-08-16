from __future__ import annotations

import asyncio
import builtins
import threading
from types import SimpleNamespace
import pytest
import torch

from lmcache_compat.connector_patch import (
    bypass_lmcache,
    gemma4_layer_configs,
    independent_transaction,
    kv_first_attention_block_view,
    patch_attention_kv_layout,
    patch_gemma4_config,
    patch_on_import,
    register_verified_kv_caches,
    verify_kv_cache_dtypes,
)


def gemma_layers(change=None):
    types = ["full_attention" if (index + 1) % 6 == 0
             else "sliding_attention" for index in range(30)]
    layers = [type("Layer", (), {
        "layer_types": types,
        "head_dim": 512 if kind == "full_attention" else 256,
        "num_key_value_heads": 2 if kind == "full_attention" else 8,
        "num_attention_heads": 16,
    })() for kind in types]
    if change:
        setattr(layers[change[0]], change[1], change[2])
    return type("Config", (), {"per_layer_config": layers})()


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


def test_dtype_proof_requires_bf16_attention_and_records_recurrent_signature():
    caches = {
        "attention": torch.empty(2, dtype=torch.bfloat16),
        "mamba": [torch.empty(3, dtype=torch.bfloat16),
                  torch.empty(4, dtype=torch.float32)],
        "mamba_2": [torch.empty(3, dtype=torch.bfloat16),
                    torch.empty(4, dtype=torch.float32)],
    }
    assert verify_kv_cache_dtypes(caches) == (
        1, "torch.bfloat16+torch.float32:2",
    )
    caches["attention"] = torch.empty(2, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="attention.*not torch.bfloat16"):
        verify_kv_cache_dtypes(caches)


@pytest.mark.parametrize("caches", [
    {},
    {"mamba": []},
    {"mamba": (object(),)},
    {"mamba": [torch.empty(1, dtype=torch.int32)]},
])
def test_dtype_proof_rejects_missing_or_unknown_tensor_groups(caches):
    with pytest.raises(RuntimeError):
        verify_kv_cache_dtypes(caches)


def test_dtype_registration_marker_follows_successful_registration_only():
    events = []
    logger = type("Logger", (), {"info": lambda _self, message, *args:
                  events.append(message % args)})()
    caches = {
        "attention": torch.empty(1, dtype=torch.bfloat16),
        "mamba": [torch.empty(1, dtype=torch.bfloat16),
                  torch.empty(1, dtype=torch.float32)],
    }

    assert register_verified_kv_caches(
        lambda values: events.append("registered") or values,
        caches, logger,
    ) is caches
    assert events == ["registered", "QH_KV_CACHE_DTYPES_VERIFIED "
                      "attention=torch.bfloat16:1 "
                      "recurrent=torch.bfloat16+torch.float32:1"]
    events.clear()
    with pytest.raises(RuntimeError):
        register_verified_kv_caches(
            lambda _values: (_ for _ in ()).throw(RuntimeError("register failed")),
            caches, logger,
        )
    assert not events


def test_gemma4_geometry_uses_only_exact_per_layer_configs():
    layers = gemma4_layer_configs(gemma_layers())

    assert len(layers) == 30
    assert [(layers[index].head_dim, layers[index].num_key_value_heads)
            for index in (0, 5)] == [(256, 8), (512, 2)]


def test_gemma4_config_uses_validated_layers_for_summary_and_backend():
    events = []
    logger = SimpleNamespace(info=lambda message, *args:
                             events.append(message % args))
    patch_gemma4_config(logger)
    from vllm.config.model import ModelConfig
    from vllm.model_executor.models.config import Gemma4Config
    from vllm.transformers_utils.model_arch_config_convertor import (
        Gemma4ModelArchConfigConvertor,
    )
    converter = object.__new__(Gemma4ModelArchConfigConvertor)
    converter.hf_text_config = gemma_layers()
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=gemma_layers()),
        attention_config=SimpleNamespace(backend=None),
    )
    text = gemma_layers()
    text.model_type = "gemma4_text"
    root = SimpleNamespace(model_type="gemma4", text_config=text)
    model = SimpleNamespace(
        hf_config=root, hf_text_config=text,
        architectures=["Gemma4ForConditionalGeneration"],
        runner="auto", is_moe=True,
    )

    assert converter.get_head_size() == 512
    assert converter.get_total_num_kv_heads() == 8
    assert converter.get_total_num_attention_heads() == 16
    Gemma4Config.verify_and_update_config(config)
    assert config.attention_config.backend.name == "TRITON_ATTN"
    assert ModelConfig._get_transformers_backend_cls(model) \
        == "TransformersMultiModalMoEForCausalLM"
    model.hf_config = SimpleNamespace(model_type="gemma4", text_config=object())
    with pytest.raises(RuntimeError, match="multimodal model structure"):
        ModelConfig._get_transformers_backend_cls(model)
    assert events.count("QH_GEMMA4_GEOMETRY_VERIFIED "
                        "sliding=25x(head_dim=256,kv_heads=8) "
                        "full=5x(head_dim=512,kv_heads=2)") == 1


@pytest.mark.parametrize("change", [
    (0, "head_dim", 512),
    (5, "num_key_value_heads", 8),
    (29, "num_attention_heads", 8),
])
def test_gemma4_geometry_rejects_any_layer_drift(change):
    with pytest.raises(RuntimeError, match="per-layer KV geometry"):
        gemma4_layer_configs(gemma_layers(change))



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
