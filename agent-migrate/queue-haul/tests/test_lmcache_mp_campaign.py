from __future__ import annotations

import lmcache_mp_campaign as m


def test_max_distinct_overlap_attributes_sessions():
    rows = [
        {"key_hashes": "a", "start_ns": "0", "end_ns": "10", "payload_bytes": "8"},
        {"key_hashes": "b", "start_ns": "5", "end_ns": "15", "payload_bytes": "8"},
        {"key_hashes": "c", "start_ns": "7", "end_ns": "8", "payload_bytes": "8"},
    ]
    assert m.max_distinct_overlap(rows, {"a": "s0", "b": "s1", "c": "s0"}) == 2


def test_reduce_transfer_accounts_only_remote_get_bodies():
    rows = [
        {"command": "SET", "key_hashes": "a", "start_ns": "0", "end_ns": "1", "response_wire_bytes": "5", "payload_bytes": "2"},
        {"command": "GET", "key_hashes": "a", "start_ns": "2", "end_ns": "3", "response_wire_bytes": "14", "payload_bytes": "8"},
        {"command": "GET", "key_hashes": "b", "start_ns": "4", "end_ns": "5", "response_wire_bytes": "5", "payload_bytes": "0"},
    ]
    out = m.reduce_transfers(rows, {"a": "s0"})
    assert out == {
        "remote_blocks": 1,
        "remote_payload_bytes": 8,
        "remote_wire_bytes": 14,
        "block_payload_bytes": 8,
        "distinct_session_overlap": 1,
    }


def test_acceptance_requires_parallel_gain_and_all_gates():
    rows = []
    for repeat, concurrency, throughput, overlap in [
        (0, 1, 10.0, 1), (1, 1, 11.0, 1), (2, 1, 9.0, 1),
        (0, 2, 12.0, 2), (1, 2, 13.0, 2), (2, 2, 12.0, 2),
        (0, 4, 14.0, 4), (1, 4, 15.0, 4), (2, 4, 14.0, 4),
    ]:
        rows.append({"repeat": repeat, "concurrency": concurrency, "throughput_bps": throughput,
                     "distinct_session_overlap": overlap, "accounting_ok": True,
                     "wire_ok": True, "continuations_ok": True})
    out = m.acceptance(rows, [{"accounting_ok": True, "wire_ok": True, "continuations_ok": True}] * 3)
    assert out["ok"]
    assert out["serialized_ceiling_bps"] == 11.0
    assert out["best_parallel_median_bps"] == 14.0


def test_cache_report_separates_l2_prefetch_from_engine_retrieval(tmp_path):
    log = tmp_path / "sink.log"
    log.write_text(
        "Prefetch request completed (L1+L2): 4/4 retained keys (0 L1, 4 L2) "
        "in 1 ms (external_request_id=req-1-worker, prefetch_request_id=0)\n"
        "Retrieved 512 tokens in 0.1 seconds\n"
    )
    out = m.cache_report(log, 0, {"req-1"})
    assert out["prefetched_tokens"] == 1024
    assert out["loaded_tokens"] == 512
    assert out["l2_blocks"] == 4


def test_reported_cached_tokens_are_required():
    assert m.cached_tokens({"usage": {"prompt_tokens_details": {"cached_tokens": 768}}}) == 768
