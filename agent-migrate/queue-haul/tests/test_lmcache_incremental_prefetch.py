from __future__ import annotations

import lmcache_incremental_prefetch as m


def stage(source: set[str], new: set[str], wan: set[str], resident: set[str]) -> dict:
    return {
        "source_block_keys": sorted(source),
        "new_source_block_keys": sorted(new),
        "wan_get_block_keys": sorted(wan),
        "target_before_keys": sorted(resident),
        "target_after_keys": sorted(resident | wan),
        "destination_l1_hit_keys": sorted(source),
        "destination_l2_prefetched_block_keys": sorted(wan),
        "wan_payload_bytes": len(wan) * 8,
        "measured_block_bytes": 8,
        "vllm_local_cached_tokens": len(source) * 256,
        "lmcache_retrieved_tokens": 0,
        "reported_cached_tokens": len(source) * 256,
        "complete_cacheable_source_prefix": len(source) * 256,
        "continuation_ok": True,
    }


def test_incremental_acceptance_requires_exact_key_conservation():
    a = {f"k{i}" for i in range(48)}
    b, c, d = ({f"k{i}" for i in range(n)} for n in (53, 58, 64))
    stages = [
        stage(a, a, a, set()),
        stage(b, b - a, b - a, a),
        stage(c, c - b, c - b, b),
        stage(d, d - c, d - c, c),
    ]
    result = m.acceptance(stages)
    assert result["ok"]
    assert result["observed_wan_blocks"] == [48, 5, 5, 6]


def test_incremental_acceptance_rejects_full_prefix_refetch():
    a = {f"k{i}" for i in range(48)}
    b, c, d = ({f"k{i}" for i in range(n)} for n in (53, 58, 64))
    stages = [
        stage(a, a, a, set()),
        stage(b, b - a, b, a),
        stage(c, c - b, c, b),
        stage(d, d - c, d, c),
    ]
    result = m.acceptance(stages)
    assert not result["ok"]
    assert not result["gates"]["incremental_wire_transfer"]
    assert not result["gates"]["no_duplicate_prefix_traffic"]


def test_chat_tokens_matches_reasoning_request(monkeypatch):
    seen = {}

    def fake(*args):
        seen["payload"] = args[4]
        return {"tokens": [1, 2], "count": 2}

    monkeypatch.setattr(m, "http_json", fake)
    cfg = type("Config", (), {"host": "h", "src_port": 1, "model": "m"})()
    assert m.chat_tokens(cfg, "p") == [1, 2]
    assert seen["payload"]["chat_template_kwargs"] == {
        "reasoning_effort": "low",
        "enable_thinking": True,
    }
