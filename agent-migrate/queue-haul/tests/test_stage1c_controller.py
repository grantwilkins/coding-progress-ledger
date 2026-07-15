"""
Claim: a deterministic trace manifest expands into matched, single-method
profiling scenarios, and reduction reports measured copy, pause, cache, stream,
and continuation quantities without power or deadline acceptance rules.

Plausible wrong implementations:
- Regenerate different conversations for repeats or split a trace turn.
- Change the selected session set when concurrency changes.
- Pool replay and KV bytes or omit matched no-migration controls.
- Count response chunks as tokens or accept partial KV chunk hits.
- Reduce incomplete, stale, or old-schema runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import stage1c_controller as c


def write_trace(path: Path) -> None:
    rows = []
    for session, base, tools, human in (("a", 1024, [], 0), ("b", 2048, [], 0), ("c", 4096, [], 0)):
        for turn in range(3):
            rows.append({"session_id": session, "timestamp": turn * (100 if session == "a" else 1), "input_tokens_total": base + 256 * turn, "prefix_tokens": base, "newly_append_tokens": 256, "output_tokens": 16, "tools": tools, "current_user_message_count": human})
    rows.append({"session_id": "a"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_manifest_is_deterministic_and_uses_complete_trace_boundaries(tmp_path):
    trace = tmp_path / "trace.jsonl"; write_trace(trace)
    assert c.trace_time({"_line": 1, "timing_events": [{"timestamp": "1970-01-01T00:00:01Z"}]}) == 1

    first = c.make_manifest(trace, "coding", 3, 7)
    second = c.make_manifest(trace, "coding", 3, 7)

    assert c.PROBE_MAX_TOKENS == 128
    assert first == second
    assert first["source"]["sha256"] == c.file_hash(trace)
    assert len({row["state_code"] for row in first["sessions"]}) == 3
    session = next(row for row in first["sessions"] if row["id"] == "b")
    index = c.nearest_turn(session, 2300)
    messages = c.session_messages(session, index)
    assert index == 1
    assert messages[-1]["role"] == "user"
    assert session["state_code"] in messages[0]["content"]


def test_plan_keeps_same_order_across_concurrency_and_adds_controls(tmp_path):
    trace, manifest_path = tmp_path / "trace.jsonl", tmp_path / "manifest.json"
    write_trace(trace); c.write_json(manifest_path, c.make_manifest(trace, "coding", 3, 1))

    plan = c.make_plan(manifest_path, [2048], [1, 2], [1000], ["replay", "kv_transfer"], ["none"], 1, 9)

    assert {row["kind"] for row in plan["scenarios"]} == {"migration", "control"}
    for method in c.METHODS:
        migration = [row for row in plan["scenarios"] if row["kind"] == "migration" and row["method"] == method]
        assert len(migration) == 2
        assert migration[0]["sessions"] == migration[1]["sessions"]
        assert all({move["method"] for move in row["moves"]} == {method} for row in migration)
    assert all(sum(row["match_id"] == other["match_id"] for other in plan["scenarios"]) == 2 for row in plan["scenarios"])


def test_plan_rejects_old_schema_and_too_few_sessions(tmp_path):
    with pytest.raises(ValueError, match="unsupported manifest"):
        c.validate_manifest({"schema": "old"})
    trace, manifest_path = tmp_path / "trace.jsonl", tmp_path / "manifest.json"
    write_trace(trace); c.write_json(manifest_path, c.make_manifest(trace, "coding", 2, 1))
    with pytest.raises(ValueError, match="at least 3"):
        c.make_plan(manifest_path, [1024], [3], [1000], ["replay"], ["none"], 1, 0)
    with pytest.raises(ValueError, match="workload must"):
        c.make_manifest(trace, "mixed", 2, 1)


def test_summary_only_adds_tail_and_bootstrap_statistics_when_supported():
    assert set(c.summary(list(range(9)))) == {"n", "median", "q25", "q75"}
    assert "median_ci_low" in c.summary(list(range(10)))
    assert "p95" in c.summary(list(range(20)))


def test_kv_bytes_come_from_logged_full_chunk_layout(tmp_path):
    log = tmp_path / "cache.log"
    rows = [
        {"monotonic_ns": 1, "operation": "source_write", "bytes": 12_582_912, "dtype": 6, "shape": [24, 2, 256, 4096]},
        {"monotonic_ns": 2, "operation": "source_write", "bytes": 6_291_456, "dtype": 6, "shape": [24, 2, 128, 4096]},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))

    layout = c.kv_layout(log, 3)

    assert layout["chunk_bytes"] == 12_582_912
    assert layout["bytes_per_token"] == 49_152
    assert layout["chunk_tokens"] == 256


def test_cli_only_exposes_new_commands():
    for command in ("make-manifest", "make-plan", "run", "reduce"):
        with pytest.raises(SystemExit) as error:
            c.parse_args([command, "--help"])
        assert error.value.code == 0
    with pytest.raises(SystemExit):
        c.parse_args(["live-grid"])


def fake_result(scenario: dict, migration: bool) -> dict:
    row = {
        "move": {"session_id": "a", "method": scenario["method"], "order": 0}, "queued_ns": 0,
        "initial_start_ns": 1_000_000_000, "initial_end_ns": 2_000_000_000,
        "pause_start_ns": 2_000_000_000, "idle_ns": 2_100_000_000,
        "catch_up_start_ns": None, "catch_up_end_ns": None,
        "switch_start_ns": 2_100_000_000, "switch_end_ns": 2_200_000_000,
        "initial": {"processed_tokens": 100, "logical_kv_chunks": 0, "logical_kv_bytes": 0}, "catch_up": None,
        "committed_state": {}, "error": None,
    }
    continuation = {"session_id": "a", "route_port": 1, "committed_context_hash": "h", "request_id": "r", "status_code": 200, "context_hash": "h", "start_ns": 3_000_000_000, "end_ns": 3_200_000_000, "first_byte_ns": 3_100_000_000, "prompt_tokens": 1, "output_tokens": 1, "processed_tokens": 0, "logical_kv_chunks": 0, "logical_kv_bytes": 0, "wire_bytes": 0, "stream_chunks": []}
    return {"schema": c.RESULT_SCHEMA, "scenario_id": scenario["scenario_id"], "status": "complete", "elapsed_s": 3, "deadline_s": 10, "deadline_met": True, "migrations": [row] if migration else [], "continuations": [continuation], "wire_bytes": {"api/client_to_target": 5}}


def test_reduce_validates_and_writes_interpretable_tables_and_plots(tmp_path, monkeypatch):
    plan = {"schema": c.PLAN_SCHEMA, "scenarios": []}
    base = {"match_id": "pair", "method": "replay", "context_size": 1024, "concurrency": 1, "bandwidth_mbps": 1000, "activity": "none", "repeat": 0, "deadline_s": 10, "sessions": [{"session_id": "a", "turn_index": 0, "order": 0}]}
    for kind in ("migration", "control"):
        plan["scenarios"].append({**base, "scenario_id": kind, "kind": kind, "moves": [{"session_id": "a", "turn_index": 0, "order": 0, "method": "replay"}] if kind == "migration" else []})
    c.write_json(tmp_path / "run_metadata.json", {"schema": c.RUN_SCHEMA})
    c.write_json(tmp_path / "plan.json", plan)
    monkeypatch.setattr(c, "plot_resource", lambda *_args: None)
    for scenario in plan["scenarios"]:
        root = tmp_path / "scenarios" / scenario["scenario_id"]; root.mkdir(parents=True)
        c.write_json(root / "result.json", fake_result(scenario, scenario["kind"] == "migration"))

    c.reduce_run(tmp_path)

    for name in ("migrations.csv", "scenarios.csv", "benchmark_summary.csv", "copy_time.png", "copy_time.pdf", "concurrency_scaling.png", "concurrency_scaling.pdf"):
        assert (tmp_path / name).exists()
    assert "continuation_difference_s" in (tmp_path / "scenarios.csv").read_text()
