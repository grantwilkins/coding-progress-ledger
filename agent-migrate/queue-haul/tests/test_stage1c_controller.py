"""
Claim: a deterministic trace manifest expands into matched, single-method
profiling scenarios, and reduction reports measured copy, pause, cache, stream,
and continuation quantities without power or deadline acceptance rules.

Plausible wrong implementations:
- Regenerate different conversations for repeats or split a trace turn.
- Change the selected session set when concurrency changes.
- Add catch-up cache hits to KV bytes transferred over the network.
- Label requested context sizes as measured prompt tokens.
- Count API or unbilled traffic as transferred KV, or compare raw rather than
  baseline-adjusted power.
- Include response generation in time to first response.
- Reduce incomplete, stale, or old-schema runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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

    assert c.PROBE_MAX_TOKENS == 512
    assert first == second
    assert first["source"]["sha256"] == c.file_hash(trace)
    assert len({row["state_code"] for row in first["sessions"]}) == 3
    session = next(row for row in first["sessions"] if row["id"] == "b")
    index = c.nearest_turn(session, 2300)
    messages = c.session_messages(session, index)
    assert index == 1
    assert messages[-1]["role"] == "user"
    assert session["state_code"] in messages[0]["content"]


def test_context_drop_starts_a_new_synthetic_session_segment(tmp_path):
    trace = tmp_path / "trace.jsonl"; write_trace(trace)
    with trace.open("a") as handle:
        for turn, total in enumerate((5000, 1000, 1500), 3):
            handle.write(json.dumps({"session_id": "b", "timestamp": turn, "input_tokens_total": total, "newly_append_tokens": 500, "output_tokens": 16}) + "\n")
    session = next(row for row in c.make_manifest(trace, "coding", 3, 7)["sessions"] if row["id"] == "b")

    assert session["turns"][4]["reset"]
    messages = c.session_messages(session, 5)
    assert not any("turn 3" in row["content"] for row in messages)
    assert any("turn 4" in row["content"] for row in messages)


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
    manifest = c.make_manifest(trace, "coding", 2, 1)
    for session in manifest["sessions"]:
        session["turns"][0]["input_tokens"] = c.MAX_MODEL_TOKENS
    c.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="prompt estimate"):
        c.make_plan(manifest_path, [1024], [1], [1000], ["replay"], ["none"], 1, 0)


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
    assert c.kv_metrics(11_047, layout) == (44, 11_047 * 49_152)
    assert c.expected_hits("kv_transfer", "initial", 11_047) == 11_047
    assert c.expected_hits("kv_transfer", "catch_up", 11_082, 10_999) == 10_752
    assert c.expected_hits("replay", "initial", 11_047) == 0
    with pytest.raises(ValueError, match="measured source"):
        c.expected_hits("kv_transfer", "catch_up", 11_082)


def test_session_probe_appends_one_final_instruction():
    session = c.LiveSession.__new__(c.LiveSession)
    session.state_code = "CODE"
    base = [{"role": "system", "content": "state"}]

    assert session.probe(base) == base + [{"role": "user", "content": "Reply with session state code CODE."}]
    assert session.probe(base, "Continue CODE") == base + [{"role": "user", "content": "Continue CODE"}]
    assert c.chat_payload(c.b.Config(), base, 1, True)["kv_transfer_params"] == {"qh_bypass_lmcache": True}


def test_cli_only_exposes_new_commands():
    for command in ("make-manifest", "make-plan", "run", "reduce"):
        with pytest.raises(SystemExit) as error:
            c.parse_args([command, "--help"])
        assert error.value.code == 0
    with pytest.raises(SystemExit):
        c.parse_args(["live-grid"])


def test_resume_records_explicit_code_change():
    old = {"plan": "p", "git_sha": "old", "git_shas": ["older", "old"]}
    new = {"plan": "p", "git_sha": "new", "git_shas": ["new"]}

    with pytest.raises(RuntimeError, match="--resume-from-git-sha old"):
        c.merge_run_metadata(new.copy(), old, None)
    assert c.merge_run_metadata(new.copy(), old, "old")["git_shas"] == ["older", "old", "new"]
    with pytest.raises(RuntimeError, match="same plan"):
        c.merge_run_metadata({**new, "plan": "other"}, old, "old")


def fake_result(scenario: dict, migration: bool) -> dict:
    row = {
        "move": {"session_id": "a", "method": scenario["method"], "order": 0}, "queued_ns": 0,
        "initial_start_ns": 1_000_000_000, "initial_end_ns": 2_000_000_000,
        "pause_start_ns": 2_000_000_000, "idle_ns": 2_100_000_000,
        "catch_up_start_ns": None, "catch_up_end_ns": None,
        "switch_start_ns": 2_100_000_000, "switch_end_ns": 2_200_000_000,
        "initial": {"start_ns": 1_000_000_000, "end_ns": 2_000_000_000,
                    "first_byte_ns": 1_700_000_000, "stream_chunks": [{"monotonic_ns": 1_600_000_000}],
                    "prompt_tokens": 120, "output_tokens": 5, "processed_tokens": 100,
                    "logical_kv_chunks": 0, "logical_kv_bytes": 0},
        "catch_up": None,
        "committed_state": {}, "error": None,
    }
    continuation = {"session_id": "a", "route_port": 1, "committed_context_hash": "h", "request_id": "r", "status_code": 200, "context_hash": "h", "start_ns": 3_000_000_000, "end_ns": 3_200_000_000, "first_byte_ns": 3_100_000_000, "prompt_tokens": 1, "output_tokens": 1, "processed_tokens": 0, "logical_kv_chunks": 0, "logical_kv_bytes": 0, "wire_bytes": 0, "stream_chunks": []}
    return {"schema": c.RESULT_SCHEMA, "scenario_id": scenario["scenario_id"], "status": "complete", "elapsed_s": 3, "deadline_s": 10, "deadline_met": True, "migrations": [row] if migration else [], "continuations": [continuation], "wire_bytes": {"api/client_to_target": 5}}


def test_reduction_separates_transferred_kv_from_catch_up_cache_hits():
    scenario = {"sessions": [{"session_id": "a", "job_class": "coding"}],
                "scenario_id": "m", "match_id": "p", "method": "kv_transfer",
                "concurrency": 1, "bandwidth_mbps": 1000, "activity": "one_turn", "repeat": 0}
    result = fake_result(scenario, True)
    move = result["migrations"][0]
    move["initial"].update({"processed_tokens": 0, "logical_kv_chunks": 2,
                            "logical_kv_bytes": 2000})
    move["catch_up_start_ns"], move["catch_up_end_ns"] = 2_100_000_000, 2_500_000_000
    move["switch_start_ns"], move["switch_end_ns"] = 2_500_000_000, 2_600_000_000
    move["catch_up"] = {
        "start_ns": 2_100_000_000, "end_ns": 2_500_000_000,
        "first_byte_ns": 2_350_000_000, "stream_chunks": [{"monotonic_ns": 2_300_000_000}],
        "prompt_tokens": 130, "output_tokens": 5, "processed_tokens": 10,
        "logical_kv_chunks": 2, "logical_kv_bytes": 2000,
    }

    row = c.flatten_migration(scenario, result, move)

    assert row["measured_kv_bytes"] == 2000
    assert row["catch_up_cache_hit_bytes"] == 2000
    assert row["measured_prompt_tokens"] == 120
    assert row["measured_processed_tokens"] == 0
    assert row["catch_up_new_tokens"] == 10
    assert row["initial_time_to_first_response_s"] == pytest.approx(.6)
    assert row["initial_response_s"] == pytest.approx(.4)


def test_network_and_power_measurements_use_measured_scopes(tmp_path):
    proxy = tmp_path / "proxy.csv"
    proxy.write_text(
        "monotonic_ns,wall_ns,interval_ns,route,direction,bytes,billed\n"
        "1000000000,0,250000000,kv,target_to_client,1000000,1\n"
        "1250000000,0,250000000,kv,target_to_client,2000000,1\n"
        "1250000000,0,250000000,api,client_to_target,9000000,1\n"
        "1250000000,0,250000000,kv,target_to_client,7000000,0\n"
    )
    network = c.network_measurements(proxy)
    assert network == {"measured_kv_wire_bytes": 3_000_000,
                       "kv_network_window_s": .5,
                       "measured_kv_throughput_mbps": 48.0}

    power = tmp_path / "power.csv"
    power.write_text(
        "monotonic_ns,wall_ns,gpu,power_w,utilization_pct,memory_mib,valid\n"
        "0,0,0,80,0,0,1\n0,0,1,100,0,0,1\n"
        "1000000000,0,0,90,0,0,1\n1000000000,0,1,130,0,0,1\n"
        "2000000000,0,0,110,0,0,1\n2000000000,0,1,150,0,0,1\n"
        "2000000000,0,1,999,0,0,0\n"
    )
    measured = c.power_measurements(power, 1_000_000_000, 2_000_000_000)
    assert measured["source_baseline_power_w"] == 80
    assert measured["destination_baseline_power_w"] == 100
    assert measured["source_mean_power_w"] == 100
    assert measured["destination_mean_power_w"] == 140
    assert measured["total_added_energy_j"] == 60


def test_model_check_uses_measured_work_and_stays_in_its_valid_range():
    case = SimpleNamespace(
        kv_transfer=SimpleNamespace(setup_s=.2, block_processing_s=.5, sync_s=.3),
        replay=SimpleNamespace(rate=lambda _tokens, _concurrency: 500),
    )
    source = SimpleNamespace(valid_range=(100, 2000))
    profile = SimpleNamespace(sources={"kv_transfer": source, "replay": source}, case=lambda: case)
    base = {"concurrency": 1, "activity": "none", "measured_prompt_tokens": 1000,
            "bandwidth_mbps": 8, "measured_kv_bytes": 1_000_000,
            "measured_kv_chunks": 2, "measured_processed_tokens": 1000}

    assert c.current_model_time({**base, "method": "kv_transfer"}, profile) == pytest.approx(2.5)
    assert c.current_model_time({**base, "method": "replay"}, profile) == pytest.approx(2)
    assert c.current_model_time({**base, "method": "replay", "concurrency": 2}, profile) is None
    assert c.current_model_time({**base, "method": "replay", "measured_prompt_tokens": 99}, profile) is None


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
        if scenario["kind"] == "migration":
            (root / "proxy_bytes.csv").write_text(
                "monotonic_ns,wall_ns,interval_ns,route,direction,bytes,billed\n"
                "1000000000,0,250000000,api,client_to_target,5,1\n")
            (root / "power.csv").write_text(
                "monotonic_ns,wall_ns,gpu,power_w,utilization_pct,memory_mib,valid\n"
                "0,0,0,80,0,0,1\n0,0,1,100,0,0,1\n"
                "1000000000,0,0,90,0,0,1\n1000000000,0,1,110,0,0,1\n"
                "2200000000,0,0,90,0,0,1\n2200000000,0,1,110,0,0,1\n")

    c.reduce_run(tmp_path)

    for name in ("migrations.csv", "scenarios.csv", "benchmark_summary.csv",
                 "initial_time.png", "throughput.png", "concurrency_scaling.png",
                 "service_effects.png", "power_energy.png"):
        assert (tmp_path / name).exists()
    table = (tmp_path / "scenarios.csv").read_text()
    assert "continuation_difference_s" in table
    assert "measured_prompt_tokens" in table
    assert "context_size" not in table
