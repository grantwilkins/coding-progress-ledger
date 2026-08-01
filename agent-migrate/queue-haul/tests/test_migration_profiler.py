"""
Claim: a deterministic trace manifest expands into matched, single-method
profiling scenarios, and reduction reports measured copy, pause, cache, stream,
and continuation quantities without power or deadline acceptance rules.

Plausible wrong implementations:
- Regenerate different conversations for repeats or split a trace turn.
- Change the selected session set across concurrency, methods, or bandwidths.
- Rerun identical controls for every method and bandwidth.
- Couple migration concurrency to serving concurrency or sleep an awake drain.
- Serialize append turns instead of overlapping generation with the next copy.
- Omit or duplicate a factorial cell, leak repeat 2 into fitting, or run sleep.
- Claim parallel KV from aggregate bytes without independent overlapping links.
- Add catch-up cache hits to KV bytes transferred over the network.
- Mix appended prompt tokens with decoded output or infer growth from requested tokens.
- Label requested context sizes as measured prompt tokens.
- Count API or unbilled traffic as transferred KV, or compare raw rather than
  baseline-adjusted power.
- Mix transition samples into steady awake/sleep windows or average irregular
  samples without time weighting.
- Compare the sleeping source GPU with the idle destination GPU.
- Include response generation in time to first response.
- Reduce incomplete, stale, or old-schema runs.
- Drain future scheduled arrivals on the source after quiescence.
- Drop paused arrivals or resume them through the old source route.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import migration_profiler as c


def write_trace(path: Path) -> None:
    rows = []
    for session, base, tools, human in (("a", 1024, [], 0), ("b", 2048, [], 0), ("c", 4096, [], 0)):
        for turn in range(3):
            rows.append({"session_id": session, "timestamp": turn * (100 if session == "a" else 1), "input_tokens_total": base + 256 * turn, "prefix_tokens": base, "newly_append_tokens": 256, "output_tokens": 16, "tools": tools, "current_user_message_count": human})
    rows.append({"session_id": "a"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_mp_scenario_rejects_proxy_restart_and_mismatched_bandwidth(monkeypatch, tmp_path):
    monkeypatch.setattr(c.b, "lmcache_mode", lambda: "mp")
    scenario = {"bandwidth_mbps": 10000}
    stack = SimpleNamespace(run_root=tmp_path, bandwidth_mbps=10000)

    with pytest.raises(ValueError, match="bandwidth-pinned"):
        c.run_scenario(stack, SimpleNamespace(), {}, scenario, tmp_path, "run")
    stack.bandwidth_mbps = 5000
    with pytest.raises(ValueError, match="does not match"):
        c.run_scenario(stack, SimpleNamespace(), {}, scenario,
                       tmp_path, "run", configure_proxy=False)


def test_shared_mp_csv_slice_contains_only_new_scenario_rows(tmp_path):
    source, destination = tmp_path / "shared.csv", tmp_path / "scenario.csv"
    source.write_text("time,value\n1,old\n")
    offset = source.stat().st_size
    with source.open("a") as handle:
        handle.write("2,new\n3,newer\n")

    c.write_csv_tail(source, destination, offset)

    assert destination.read_text() == "time,value\n2,new\n3,newer\n"




def test_mp_scenarios_preserve_timeline_inputs():
    assert c.MP_SCENARIO_CSVS == (
        "proxy_bytes.csv", "proxy_connections.csv", "resp_transfers.csv",
    )
def test_mp_plan_reuses_stack_and_restarts_only_for_bandwidth(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "manifest": {"path": str(manifest), "sha256": c.file_hash(manifest)},
        "scenarios": [
            {"scenario_id": "a", "bandwidth_mbps": 1000},
            {"scenario_id": "b", "bandwidth_mbps": 1000},
            {"scenario_id": "c", "bandwidth_mbps": 1000},
            {"scenario_id": "d", "bandwidth_mbps": 5000},
        ],
    }))
    starts, runs, stops = [], [], []

    def start(_cfg, root, bandwidth, _extra):
        starts.append(bandwidth)
        return SimpleNamespace(run_root=root, bandwidth_mbps=bandwidth)

    monkeypatch.setattr(c, "validate_plan", lambda *_: None)
    monkeypatch.setattr(c, "git_state", lambda _: ("sha", False))
    monkeypatch.setattr(c, "config_record", lambda _: {})
    monkeypatch.setattr(c.b, "lmcache_mode", lambda: "mp")
    monkeypatch.setattr(c.b, "start_stack", start)
    monkeypatch.setattr(c.b, "start_sink", lambda *_: None)
    monkeypatch.setattr(
        c.b, "stop_stack",
        lambda stack: stops.append(stack.bandwidth_mbps),
    )
    monkeypatch.setattr(
        c, "run_scenario",
        lambda stack, _cfg, _manifest, scenario, *_args, **kwargs:
            runs.append((scenario["scenario_id"], stack.bandwidth_mbps,
                         kwargs["configure_proxy"])),
    )

    c.run_plan(
        plan, tmp_path / "run", SimpleNamespace(), False, [],
        stack_scenarios=2,
    )

    assert starts == [1000, 1000, 5000]
    assert runs == [("a", 1000, False), ("b", 1000, False),
                    ("c", 1000, False), ("d", 5000, False)]
    assert stops == [1000, 1000, 5000]


def test_crossover_plan_is_exact_paired_and_bandwidth_blocked(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": c.MANIFEST_SCHEMA, "workload": "coding",
        "sessions": [{
            "id": f"s{index}", "job_class": "coding", "rank": index,
            "state_code": f"C{index}", "turns": [{
                "time_s": 0, "input_tokens": 4096, "append_tokens": 32,
                "output_tokens": 1, "reset": False,
            }],
        } for index in range(2)],
    }))
    plan = c.make_crossover_plan(
        manifest, [2048, 4096], [1000, 2500], 3, seed=7,
    )

    assert len(plan["scenarios"]) == 24
    assert plan["scenarios"][0]["smoke"]
    assert plan["scenarios"][0]["context_size"] == 4096
    assert {
        (row["context_size"], row["bandwidth_mbps"],
         row["repeat"], row["method"])
        for row in plan["scenarios"]
    } == {
        (size, bandwidth, repeat, method)
        for size in (2048, 4096) for bandwidth in (1000, 2500)
        for repeat in range(3) for method in c.METHODS
    }
    samples = {}
    for row in plan["scenarios"]:
        assert row["sessions"][0]["initial_tokens"] \
            == row["context_size"] - c.CROSSOVER_PROMPT_HEADROOM_TOKENS
        samples.setdefault(
            (row["context_size"], row["repeat"]), set()
        ).add((row["sample_id"], row["sessions"][0]["session_id"]))
    assert all(len(rows) == 1 for rows in samples.values())
    links = [row["bandwidth_mbps"] for row in plan["scenarios"]]
    assert sum(
        index == 0 or link != links[index - 1]
        for index, link in enumerate(links)
    ) == 2


def test_fail_fast_records_first_failure_and_stops(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "manifest": {"path": str(manifest), "sha256": c.file_hash(manifest)},
        "scenarios": [
            {"scenario_id": "a", "bandwidth_mbps": 1000},
            {"scenario_id": "b", "bandwidth_mbps": 1000},
        ],
    }))
    runs = []
    stack = SimpleNamespace(run_root=tmp_path, bandwidth_mbps=1000)
    monkeypatch.setattr(c, "validate_plan", lambda *_: None)
    monkeypatch.setattr(c, "git_state", lambda _: ("sha", False))
    monkeypatch.setattr(c, "config_record", lambda _: {})
    monkeypatch.setattr(c.b, "lmcache_mode", lambda: "mp")
    monkeypatch.setattr(c.b, "start_stack", lambda *_: stack)
    monkeypatch.setattr(c.b, "start_sink", lambda *_: None)
    monkeypatch.setattr(c.b, "stop_stack", lambda *_: None)

    def fail(_stack, _cfg, _manifest, scenario, *_args, **_kwargs):
        runs.append(scenario["scenario_id"])
        raise RuntimeError("broken hardware")

    monkeypatch.setattr(c, "run_scenario", fail)
    with pytest.raises(RuntimeError, match="scenario failed"):
        c.run_plan(
            plan, tmp_path / "run", SimpleNamespace(), False, [],
            fail_fast=True,
        )

    assert runs == ["a"]
    assert json.loads(
        (tmp_path / "run/scenarios/a/result.json").read_text()
    )["status"] == "failed"


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


def test_plan_separates_concurrency_and_reuses_compatible_controls(tmp_path):
    trace, manifest_path = tmp_path / "trace.jsonl", tmp_path / "manifest.json"
    write_trace(trace); c.write_json(manifest_path, c.make_manifest(trace, "coding", 3, 1))

    plan = c.make_plan(
        manifest_path, [2048], [1, 2], [1000, 10000],
        ["replay", "kv_transfer"], ["none"], 1, 9,
        serving_concurrency=[1, 3],
    )

    assert {row["kind"] for row in plan["scenarios"]} == {"migration", "control"}
    assert len([row for row in plan["scenarios"] if row["kind"] == "control"]) == 2
    for method in c.METHODS:
        migration = [row for row in plan["scenarios"] if row["kind"] == "migration" and row["method"] == method]
        assert len(migration) == 8
        assert all(row["sessions"] == migration[0]["sessions"] for row in migration)
        assert all({move["method"] for move in row["moves"]} == {method} for row in migration)
        assert {(row["move_concurrency"], row["serving_concurrency"])
                for row in migration} == {(1, 1), (1, 3), (2, 1), (2, 3)}
    for match_id in {row["match_id"] for row in plan["scenarios"]}:
        matched = [row for row in plan["scenarios"] if row["match_id"] == match_id]
        assert sum(row["kind"] == "control" for row in matched) == 1
        assert len(matched) == 9
        assert all(row["sessions"] == matched[0]["sessions"] for row in matched)
    assert all(row["final_state"] == "awake" for row in plan["scenarios"])


def test_plan_can_pin_the_same_sessions_across_repeats(tmp_path):
    trace, manifest_path = tmp_path / "trace.jsonl", tmp_path / "manifest.json"
    write_trace(trace); c.write_json(manifest_path, c.make_manifest(trace, "coding", 3, 1))
    session_ids = ["a", "c"]

    plan = c.make_plan(
        manifest_path, [2048], [1, 2], [1000], ["replay"], ["none"], 2, 9,
        session_ids=session_ids,
    )

    assert all([row["session_id"] for row in scenario["sessions"]] == session_ids
               for scenario in plan["scenarios"])
    with pytest.raises(ValueError, match="exactly 2"):
        c.make_plan(
            manifest_path, [2048], [2], [1000], ["replay"], ["none"], 1, 9,
            session_ids=["a"],
        )


def test_bounded_campaign_has_exact_surface_stages_and_split():
    manifest = Path(c.__file__).with_name("outputs") / "coding-manifest.json"

    plan = c.make_campaign(manifest, 7)

    assert len(plan["scenarios"]) == 105
    assert sum(row["kind"] == "migration" for row in plan["scenarios"]) == 90
    assert sum(row["kind"] == "control" for row in plan["scenarios"]) == 15
    assert all(row["final_state"] == "awake" for row in plan["scenarios"])
    smoke = plan["scenarios"][0]
    assert smoke["smoke"]
    assert (
        smoke["campaign"], smoke["context_size"], smoke["bandwidth_mbps"],
        smoke["move_concurrency"], smoke["repeat"],
    ) == ("parallel_surface", 4096, 1000, 4, 0)
    assert all(
        row["split"] == ("validation" if row["repeat"] == 2 else "train")
        for row in plan["scenarios"]
    )
    staged = [
        row for row in plan["scenarios"]
        if row["campaign"] == "staged_append"
        and row["kind"] == "migration"
        and row["copy_policy"] == "after_each_request"
    ]
    assert len(staged) == 12
    assert all(len(row["request_schedule"]) == 4 for row in staged)
    broken = {**plan, "scenarios": plan["scenarios"][:-1]}
    with pytest.raises(ValueError, match="105 scenarios"):
        c.validate_campaign_plan(broken)


def test_catch_up_plan_pairs_each_measured_append_size(tmp_path):
    trace, manifest_path = tmp_path / "trace.jsonl", tmp_path / "manifest.json"
    write_trace(trace); c.write_json(manifest_path, c.make_manifest(trace, "coding", 3, 1))

    plan = c.make_plan(
        manifest_path, [2048], [1], [1000, 10000], ["kv_transfer"],
        ["one_turn"], 1, 9, activity_tokens=[32, 128],
    )

    assert len(plan["scenarios"]) == 6
    assert {row["activity_tokens"] for row in plan["scenarios"]} == {32, 128}
    for tokens in (32, 128):
        rows = [row for row in plan["scenarios"]
                if row["activity_tokens"] == tokens]
        assert sum(row["kind"] == "control" for row in rows) == 1
        assert {row["bandwidth_mbps"] for row in rows
                if row["kind"] == "migration"} == {1000, 10000}


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
    with pytest.raises(ValueError, match="final state"):
        c.make_plan(
            manifest_path, [1024], [1], [1000], ["replay"], ["none"], 1, 0,
            final_state="off",
        )
    c.write_json(manifest_path, c.make_manifest(trace, "coding", 2, 1))
    with pytest.raises(ValueError, match="sleep request"):
        c.make_plan(
            manifest_path, [1024], [1], [1000], ["replay"], ["none"], 1, 0,
            final_state="sleep",
        )
    plan = c.make_plan(
        manifest_path, [1024], [1], [1000], ["replay"], ["none"], 1, 0,
    )
    plan["scenarios"][0]["request_schedule"] = [
        {"at_s": 0, "append_tokens": c.MAX_MODEL_TOKENS}
    ]
    with pytest.raises(ValueError, match="prompt estimate"):
        c.validate_plan(plan, json.loads(manifest_path.read_text()))


def test_awake_drain_never_requests_sleep():
    assert not c.should_sleep({"final_state": "awake"}, True)
    assert c.should_sleep({"final_state": "sleep"}, True)
    assert not c.should_sleep({"final_state": "sleep"}, False)


def test_destination_load_gates_action_and_always_closes():
    events = []
    class Load:
        def start(self): events.append("start")
        def wait_ready(self): events.append("ready")
        def close(self): events.append("close")
    assert c.with_destination_load(Load(), lambda: events.append("action") or 7) == 7
    assert events == ["start", "ready", "action", "close"]
    with pytest.raises(RuntimeError):
        c.with_destination_load(Load(), lambda: (_ for _ in ()).throw(RuntimeError()))
    assert events[-1] == "close"
    class FailedLoad(Load):
        def wait_ready(self): raise TimeoutError
    with pytest.raises(TimeoutError):
        c.with_destination_load(FailedLoad(), lambda: None)
    assert events[-1] == "close"


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
    assert c.expected_hits("kv_transfer", "initial", 4_235, 4_235) == 4_235
    assert c.expected_hits("kv_transfer", "initial", 16_523, 16_384) == 16_384
    assert c.expected_hits("kv_transfer", "catch_up", 11_082, 10_999) == 10_752
    assert c.expected_hits("replay", "initial", 11_047) == 0
    with pytest.raises(ValueError, match="measured source"):
        c.expected_hits("kv_transfer", "initial", 11_082)


def test_stored_tokens_uses_final_report_for_request(tmp_path):
    log = tmp_path / "source.log"
    log.write_text(
        "Reqid: first, Total tokens 16523\n"
        "Stored 8192 out of total 8192 tokens\n"
        "Stored 16384 out of total 16384 tokens\n"
        "Reqid: second, Total tokens 4235\n"
        "Stored 4235 out of total 4235 tokens\n"
    )

    assert c.stored_tokens(log, "first") == 16_384
    assert c.stored_tokens(log, "second") == 4_235


def test_session_probe_appends_one_final_instruction():
    session = c.LiveSession.__new__(c.LiveSession)
    session.state_code = "CODE"
    base = [{"role": "system", "content": "state"}]

    assert session.probe(base) == base + [{"role": "user", "content": "Reply with session state code CODE."}]
    assert session.probe(base, "Continue CODE") == base + [{"role": "user", "content": "Continue CODE"}]
    assert c.chat_payload(c.b.Config(), base, 1)["reasoning_effort"] == "low"
    assert c.chat_payload(c.b.Config(), base, 1, True)["kv_transfer_params"] == {"qh_bypass_lmcache": True}


def test_session_request_requires_http_success_not_exact_model_text(monkeypatch):
    session = c.LiveSession.__new__(c.LiveSession)
    session.cfg, session.state_code = c.b.Config(), "CODE"
    session.session_id, session.timeout_s = "session", 1
    session.event_log = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    result = c.RequestResult("request", 200, "hash", 1, 2)
    monkeypatch.setattr(c, "stream_chat", lambda *_args: (result, "valid reply"))

    assert session.request(1, [], "probe") == (result, "valid reply")
    failed = c.RequestResult("request", 500, "hash", 1, 2)
    monkeypatch.setattr(c, "stream_chat", lambda *_args: (failed, "CODE"))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        session.request(1, [], "probe")


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
        "1750000000,0,250000000,kv,target_to_client,5000000,1\n"
    )
    network = c.network_measurements(proxy, 1_000_000_000, 1_500_000_000)
    assert network == {"measured_kv_wire_bytes": 3_000_000,
                       "kv_network_window_s": .5,
                       "measured_kv_throughput_mbps": 48.0}

    power = tmp_path / "power.csv"
    power.write_text(
        "monotonic_ns,wall_ns,gpu,power_w,utilization_pct,memory_mib,valid\n"
        "0,0,0,80,0,0,1\n0,0,1,100,0,0,1\n"
        "1000000000,0,0,90,0,0,1\n1000000000,0,1,130,0,0,1\n"
        "1500000000,0,0,110,0,0,1\n1500000000,0,1,150,0,0,1\n"
        "2000000000,0,0,110,0,0,1\n2000000000,0,1,150,0,0,1\n"
        "2000000000,0,1,999,0,0,0\n"
    )
    measured = c.power_measurements(power, 1_000_000_000, 2_000_000_000)
    assert measured["source_baseline_power_w"] == 80
    assert measured["destination_baseline_power_w"] == 100
    assert measured["source_mean_power_w"] == 100
    assert measured["destination_mean_power_w"] == 140
    assert measured["total_added_energy_j"] == 60


def test_parallel_gate_requires_independent_overlapping_positive_byte_windows(tmp_path):
    proxy = tmp_path / "proxy_connections.csv"
    proxy.write_text(
        "connection_id,route,key_hash,start_ns,end_ns,client_to_target_bytes,target_to_client_bytes\n"
        "a,kv,ka,0,500000000,186,1000036\n"
        "b,kv,kb,0,500000000,186,1000036\n"
        "z,api,,0,500000000,999,0\n"
    )

    measured = c.parallel_connection_measurements(
        proxy, 0, 500_000_000, required=2,
        session_keys={"sa": {"ka"}, "sb": {"kb"}},
    )

    assert measured == {
        "connection_count": 2,
        "session_count": 2,
        "max_parallel_sessions": 2,
        "overlap_windows": 1,
        "wire_bytes": 2000072,
        "kv_body_bytes": 2000000,
        "session_kv_body_bytes": {"sa": 1000000, "sb": 1000000},
    }
    with pytest.raises(RuntimeError, match="maps to 2 sessions"):
        c.parallel_connection_measurements(
            proxy, 0, 500_000_000, 2,
            {"sa": {"ka", "kb"}, "sb": {"kb"}},
        )


def test_parallel_gate_rejects_sequential_connections_with_same_total_bytes(tmp_path):
    proxy = tmp_path / "proxy_connections.csv"
    proxy.write_text(
        "connection_id,route,key_hash,start_ns,end_ns,client_to_target_bytes,target_to_client_bytes\n"
        "a,kv,ka,0,500000000,186,1000036\n"
        "b,kv,kb,500000000,1000000000,186,1000036\n"
    )

    with pytest.raises(RuntimeError, match="overlapping"):
        c.parallel_connection_measurements(
            proxy, 0, 1_000_000_000, 2,
            {"sa": {"ka"}, "sb": {"kb"}},
        )


def test_campaign_allows_ambiguous_shared_prefix_bytes(tmp_path):
    proxy = tmp_path / "proxy_connections.csv"
    proxy.write_text(
        "connection_id,route,key_hash,start_ns,end_ns,client_to_target_bytes,target_to_client_bytes\n"
        "a,kv,shared,0,500000000,186,1000036\n"
        "b,kv,unique,0,500000000,186,1000036\n"
    )

    measured = c.parallel_connection_measurements(
        proxy, 0, 500_000_000, 2,
        {"sa": {"shared", "unique"}, "sb": {"shared"}}, strict=False,
    )
    stage = {
        "copied_blocks_before": 111, "copied_blocks_after": 114,
        "logical_body_bytes": 3, "wire_body_bytes": 2,
    }

    assert measured["kv_body_bytes"] == 2_000_000
    assert c.max_overlap([(0, 2), (1, 3)]) == 2
    assert not c.valid_append_stage(stage)
    assert c.valid_append_stage({**stage, "wire_body_bytes": 3})
    assert not c.valid_append_stage({**stage, "wire_body_bytes": 4})


def test_live_runtime_pipelines_four_append_stages(monkeypatch, tmp_path):
    session = c.LiveSession(
        SimpleNamespace(src_port=1),
        {"id": "s", "state_code": "CODE", "turns": [
            {"input_tokens": 100, "append_tokens": 1, "output_tokens": 1}
        ]},
        0, SimpleNamespace(write=lambda *_args, **_kwargs: None),
        tmp_path / "source.log", tmp_path / "cache.log", 10,
    )
    session.messages = [{"role": "user", "content": "base"}]
    session.warm_prompt_tokens = 100
    session.prompt_tokens_by_hash[c.messages_hash(session.messages)] = 100
    calls = 0

    def request(_port, _messages, label, _prompt=None, **_kwargs):
        nonlocal calls
        time.sleep(.02)
        calls += 1
        now = time.monotonic_ns()
        return c.RequestResult(
            label, 200, "", now, now, prompt_tokens=100 + 256 * calls,
        ), "CODE"

    session.request = request
    runtime = c.LiveRuntime(
        {"s": session}, SimpleNamespace(), "one_turn",
        tmp_path / "sink.log", tmp_path / "cache.log",
        SimpleNamespace(write=lambda *_args, **_kwargs: None),
        tmp_path / "requests.jsonl",
        [{"at_s": 0, "append_tokens": 32}] * 4,
        "after_each_request", 1, time.monotonic_ns(),
    )
    overlap = []

    def prepare(_move, state, _phase):
        overlap.append(session.activity_thread is not None)
        now = time.monotonic_ns()
        return c.RequestResult(
            "copy", 200, state.context_hash, now, now,
            prompt_tokens=session.prompt_tokens_by_hash[state.context_hash],
        )

    runtime.prepare = prepare
    monkeypatch.setattr(
        c, "kv_layout",
        lambda *_args: {"chunk_tokens": 256, "chunk_bytes": 10},
    )
    state = session.snapshot()
    runtime._start_next(session)
    stages = runtime.background(c.Move("s", "kv_transfer", 0), state)
    runtime.close()

    assert [row.stage_index for row in stages] == list(range(4))
    assert overlap == [True, True, True, False]
    assert [row.logical_body_bytes for row in stages] == [10] * 4


def test_mp_prepare_accepts_concurrent_l1_fill_and_advances_key_watermark(
        monkeypatch, tmp_path):
    monkeypatch.setattr(c.b, "lmcache_mode", lambda: "mp")
    sink = tmp_path / "lmcache-sink.log"
    sink.write_text(
        "Registered non-GPU context model=model, world_size=1\n"
    )
    transfers = tmp_path / "resp_transfers.csv"
    transfers.write_text(
        "connection_id,command,key_hashes,start_ns,end_ns,"
        "request_wire_bytes,response_wire_bytes,request_body_bytes,payload_bytes\n"
        "a,GET,k1,1,2,1,11,1,10\n"
    )
    calls = []
    session = SimpleNamespace(
        cache_keys={"stale", "k1", "k2"},
        copied_keys={"stale", "k1"}, copied_token_ids=[0] * 256,
        warm_cached_tokens=512,
        prompt_tokens_by_hash={"h": 520},
        probe=lambda messages: messages + [{"role": "user", "content": "probe"}],
    )

    def request(*_args, **_kwargs):
        calls.append("request")
        return c.RequestResult(
            "r", 200, "h", 3, 4, prompt_tokens=520,
            cached_tokens=512,
        ), "CODE"

    session.request = request
    monkeypatch.setattr(
        c.b, "mp_chat_tokens",
        lambda _cfg, _messages: calls.append("tokenize") or [0] * 512,
    )
    monkeypatch.setattr(
        c.b, "mp_warm_prefetch",
        lambda *_args: calls.append("prefetch")
        or {"total_keys": 2, "found_keys": 0},
    )
    monkeypatch.setattr(c.b, "mp_request_hit", lambda *_args: 512)
    runtime = c.LiveRuntime(
        {"s": session},
        SimpleNamespace(api_proxy_port=1),
        "none", sink, transfers,
        SimpleNamespace(write=lambda *_args, **_kwargs: None),
        tmp_path / "requests.jsonl",
    )

    result = runtime.prepare(c.Move("s", "kv_transfer", 0),
                             c.SessionState("s", 0, (), "h"), "initial")
    runtime.close()

    assert calls == ["tokenize", "prefetch", "request"]
    assert session.copied_keys == {"stale", "k1", "k2"}
    assert session.copied_token_ids == [0] * 512
    assert (result.logical_kv_chunks, result.logical_kv_bytes,
            result.processed_tokens) == (2, 20, 8)


def test_request_schedule_is_relative_to_post_warm_epoch(tmp_path):
    calls = []
    session = SimpleNamespace(
        session_id="s",
        start_activity=lambda tokens, at_ns, stage: calls.append(
            (tokens, at_ns, stage)
        ),
    )
    runtime = c.LiveRuntime(
        {"s": session}, SimpleNamespace(), "one_turn",
        tmp_path / "sink.log", tmp_path / "cache.log",
        SimpleNamespace(), tmp_path / "requests.jsonl",
        [{"at_s": 2.5, "append_tokens": 32}],
        scenario_start_ns=10_000_000_000,
    )

    runtime._start_next(session)
    runtime.close()

    assert calls == [(32, 12_500_000_000, 0)]


def test_quiescence_drains_admitted_request_and_routes_queue_to_destination(
        monkeypatch, tmp_path):
    monkeypatch.setattr(c.b, "lmcache_mode", lambda: "mp")
    monkeypatch.setattr(c.b, "mp_model_layout", lambda _path: ("model", 1))
    waited = []
    monkeypatch.setattr(c.b, "mp_wait_source_keys", lambda *args: waited.append(args[4]) or {"k"})
    source, cache = tmp_path / "source.log", tmp_path / "cache.log"
    source.write_text(""); cache.write_text("")
    session = c.LiveSession(
        SimpleNamespace(src_port=1, api_proxy_port=2),
        {"id": "s", "state_code": "CODE", "turns": [{
            "input_tokens": 100, "append_tokens": 1, "output_tokens": 1,
        }]}, 0, SimpleNamespace(write=lambda *_args, **_kwargs: None),
        source, cache, 2,
    )
    session.messages = [{"role": "user", "content": "base"}]
    session.warm_prompt_tokens = 100
    started, release, calls = threading.Event(), threading.Event(), []

    def request(port, _messages, label, _prompt=None, **_kwargs):
        calls.append(port)
        if len(calls) == 1:
            started.set()
            assert release.wait(1)
        now = time.monotonic_ns()
        return c.RequestResult(
            label, 200, "", now, now, first_byte_ns=now,
            prompt_tokens=768, output_tokens=1, cached_tokens=512,
        ), "CODE"

    session.request = request
    runtime = c.LiveRuntime(
        {"s": session}, SimpleNamespace(api_proxy_port=2), "one_turn",
        tmp_path / "sink.log", cache,
        SimpleNamespace(write=lambda *_args, **_kwargs: None),
        tmp_path / "requests.jsonl",
        [{"at_s": 0, "append_tokens": 1}] * 2,
        scenario_start_ns=time.monotonic_ns(),
    )
    worker = threading.Thread(target=runtime.run_activities, args=("s",))
    worker.start()
    assert started.wait(1)
    runtime.pause("s")
    release.set()
    state = runtime.wait_idle("s")
    assert calls == [1]

    runtime.commit(c.Move("s", "replay", 0), state)
    worker.join(1)
    runtime.close()

    assert not worker.is_alive()
    assert calls == [1, 2]
    assert [row["location"] for row in session.activity_records] == [
        "source", "destination",
    ]


    assert waited == [256]
def test_connection_attribution_conserves_duplicate_bodies(tmp_path):
    path = tmp_path / "proxy_connections.csv"
    path.write_text(
        "connection_id,route,key_hash,start_ns,end_ns,client_to_target_bytes,target_to_client_bytes\n"
        "a,kv,k,0,1,0,46\n"
        "b,kv,k,1,2,0,56\n"
        "c,kv,z,1,2,0,76\n"
    )

    measured = c.attributed_connections(path, 0, 3, {"k", "z"})

    assert measured["wire_body_bytes"] == 70
    assert measured["protocol_bytes"] == 108
    assert measured["key_body_bytes"] == {"k": 30, "z": 40}


def test_control_service_rows_preserve_schedule_and_growth():
    rows = c.service_request_rows(
        {
            "scenario_id": "control", "campaign": "staged_append",
            "split": "train", "kind": "control", "concurrency": 1,
            "serving_concurrency": 4,
        },
        {"activities": [{
            "session_id": "s", "stage_index": 0,
            "scheduled_ns": 1_000_000_000, "start_ns": 1_500_000_000,
            "first_byte_ns": 1_750_000_000, "end_ns": 2_000_000_000,
            "prompt_tokens": 120, "output_tokens": 5,
            "measured_append_tokens": 20, "status_code": 200,
        }]},
    )

    assert rows[0]["retained_growth_tokens"] == 20
    assert rows[0]["schedule_delay_s"] == .5
    assert rows[0]["ttft_s"] == .25
    assert rows[0]["service_s"] == .5
    assert rows[0]["serving_concurrency"] == 4
    assert rows[0]["success"]


def test_catch_up_profile_separates_prompt_output_and_uses_strict_convergence():
    row = {
        "measured_kv_bytes": 1000 * 49_152,
        "measured_prompt_tokens": 1000,
        "initial_time_to_first_response_s": 2.0,
        "initial_kv_wire_bytes": 49_153_000,
        "catch_up_kv_wire_bytes": 2_000_000,
        "catch_up_new_tokens": 40,
        "measured_activity_append_tokens": 32,
        "activity_output_tokens": 8,
        "activity_s": .1,
        "service_pause_s": .5,
        "activity_overlapped_initial_copy": True,
    }

    measured = c.catch_up_profile(row)

    assert measured["bytes_per_token"] == 49_152
    assert measured["appended_prompt_tokens"] == 32
    assert measured["decoded_output_tokens"] == 8
    assert measured["state_growth_bytes"] == 40 * 49_152
    assert measured["effective_copy_bytes_per_s"] == 1000 * 49_152 / 2
    assert measured["kv_growth_bytes_per_s"] == 40 * 49_152 / .1
    assert measured["converges"]
    assert not c.catch_up_profile({**row, "activity_s": .08})["converges"]


def test_power_state_summary_uses_time_weighting_and_same_gpu(tmp_path):
    power = tmp_path / "power.csv"
    power.write_text(
        "monotonic_ns,wall_ns,gpu,power_w,utilization_pct,memory_mib,valid\n"
        "0,0,0,80,0,1000,1\n0,0,1,100,0,2000,1\n"
        "1000000000,0,0,100,0,1000,1\n1000000000,0,1,100,0,2000,1\n"
        "3000000000,0,0,50,0,100,1\n3000000000,0,1,100,0,2000,1\n"
        "4000000000,0,0,50,0,100,1\n4000000000,0,1,100,0,2000,1\n"
    )

    node = tmp_path / "node.csv"
    node.write_text(
        "monotonic_ns,wall_ns,node,current_watts\n"
        "0,0,n0,200\n3000000000,0,n0,170\n4000000000,0,n0,170\n"
    )
    rows = c.power_state_summary(
        power, node,
        [{"awake_ns": [0, 3_000_000_000], "sleep_ns": [3_000_000_000, 4_000_000_000]}],
    )

    source_awake = next(row for row in rows
                        if row["device"] == "source" and row["state"] == "awake")
    source_sleep = next(row for row in rows
                        if row["device"] == "source" and row["state"] == "sleep")
    destination = [row for row in rows if row["device"] == "destination"]
    assert source_awake["mean_power_w"] == pytest.approx((80 + 2 * 100) / 3)
    assert source_sleep["mean_power_w"] == 50
    assert source_sleep["mean_memory_mib"] == 100
    assert {row["mean_power_w"] for row in destination} == {100}
    node_rows = [row for row in rows if row["scope"] == "node"]
    assert [row["mean_power_w"] for row in node_rows] == [200, 170]


def test_node_power_parsing_hard_fails_missing_or_disabled_power(monkeypatch):
    assert c.parse_node_power("CurrentWatts=210") == 210
    for text in ("CurrentWatts=n/s", "CurrentWatts=0", "AveWatts=10"):
        with pytest.raises(RuntimeError, match="Slurm node power"):
            c.parse_node_power(text)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    with pytest.raises(RuntimeError, match="SLURMD_NODENAME"):
        c.node_power_reading()


def test_power_sampler_uses_allocated_gpu_order(tmp_path, monkeypatch):
    path = tmp_path / "power.csv"
    sampler = c.PowerSampler(path)
    calls, outputs = [], iter(["80,0,10\n", "90,0,20\n"])
    monkeypatch.setattr(c.b, "allocated_gpu_ids", lambda: ["2", "3"])

    def check_output(command, text):
        calls.append(command)
        if len(calls) == 2:
            sampler.stop.set()
        return next(outputs)

    monkeypatch.setattr(c.subprocess, "check_output", check_output)
    sampler._run()

    assert [command[2] for command in calls] == ["2", "3"]
    assert [line.split(",")[2] for line in path.read_text().splitlines()[1:]] == ["0", "1"]


def test_power_profile_orders_steady_windows_and_verified_wake(tmp_path, monkeypatch):
    calls = []

    class Sampler:
        def __init__(self, path):
            calls.append(("sampler", path.name))

        def start(self):
            calls.append(("sampler", "start"))

        def close(self):
            calls.append(("sampler", "close"))

    monkeypatch.setattr(c, "PowerSampler", Sampler)
    monkeypatch.setattr(c.b, "set_source_sleep",
                        lambda _cfg, state: calls.append(("sleep", state)))
    monkeypatch.setattr(c.b, "reset_vllm_caches",
                        lambda *_args: calls.append(("reset",)))
    monkeypatch.setattr(c.time, "sleep",
                        lambda seconds: calls.append(("wait", seconds)))
    times = iter(range(1, 9))
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: next(times))
    probe = c.RequestResult("wake", 200, "h", 7, 8, first_byte_ns=8)
    monkeypatch.setattr(
        c, "stream_chat",
        lambda *args: calls.append(("probe", args[3])) or (probe, "ready"),
    )
    monkeypatch.setattr(c, "power_state_summary", lambda *_args: [])
    monkeypatch.setattr(c, "write_csv", lambda *_args: None)
    monkeypatch.setattr(c, "write_json", lambda *_args: None)
    stack = SimpleNamespace(run_root=tmp_path)

    c.profile_power_states(stack, c.b.Config(), tmp_path / "power", 1, 60, False)

    assert calls == [
        ("sleep", False), ("sampler", "gpu_power.csv"), ("sampler", "start"),
        ("reset",), ("wait", 10), ("wait", 60), ("sleep", True),
        ("wait", 10), ("wait", 60), ("sleep", False),
        ("probe", c.PROBE_MAX_TOKENS),
        ("sampler", "close"),
    ]


def test_batch_power_measurement_requests_two_gpus():
    script = Path(c.__file__).with_name("migration_profile.sbatch").read_text()

    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --exclusive" not in script
    assert "--power-state-cycles" in script
    assert "--node-power" not in script


def test_targeted_jobs_use_reviewed_plans_and_hard_gates():
    root = Path(c.__file__).parent
    runner = (root / "targeted_migration_run.sh").read_text()
    jobs = {
        "parallel_kv_gate.sbatch":
            ("outputs/parallel-kv-gate-plan.json", "check-parallel"),
        "append_catch_up.sbatch":
            ("outputs/append-catch-up-plan.json", "check-catch-up"),
    }

    assert "migration_testbed.py preflight --required-gpus 2" in runner
    assert "migration_profiler.py run" in runner
    assert "migration_profiler.py reduce" in runner
    assert '"$CHECK" --run-root' in runner
    assert "resume=()" not in runner and '"$@"' in runner
    assert "src_port" in runner and "- 8100" in runner
    for name, required in jobs.items():
        script = (root / name).read_text()
        assert "#SBATCH --gres=gpu:2" in script
        assert all(value in script for value in required)
        assert "targeted_migration_run.sh" in script


def test_model_check_uses_measured_work_and_stays_in_its_valid_range():
    case = SimpleNamespace(
        kv_transfer=SimpleNamespace(
            setup_s=.2, destination_bytes_per_s=1_000_000,
            initial_completion_s=.3,
        ),
        replay=SimpleNamespace(rate=lambda _tokens, _concurrency: 500),
        replay_completion_s=0,
    )
    profile = SimpleNamespace(
        sources={"kv_transfer": SimpleNamespace(valid_range=(500_000, 2_000_000)),
                 "replay": SimpleNamespace(valid_range=(100, 2000))},
        case=lambda: case,
    )
    base = {"concurrency": 1, "activity": "none", "measured_prompt_tokens": 1000,
            "bandwidth_mbps": 8, "measured_kv_bytes": 1_000_000,
            "measured_kv_chunks": 2, "measured_processed_tokens": 1000}

    assert c.current_model_time({**base, "method": "kv_transfer"}, profile) == pytest.approx(1.5)
    assert c.current_model_time({**base, "method": "replay"}, profile) == pytest.approx(2)
    assert c.current_model_time({**base, "method": "replay", "concurrency": 2}, profile) is None
    assert c.current_model_time({**base, "method": "replay", "measured_prompt_tokens": 99}, profile) is None


@pytest.mark.parametrize(
    ("plan_schema", "has_stage_table"),
    ((c.PLAN_SCHEMA, True), ("queue-haul-migration-plan-v2", False)),
)
def test_reduce_validates_and_writes_interpretable_tables_and_plots(
    tmp_path, monkeypatch, plan_schema, has_stage_table,
):
    plan = {"schema": plan_schema, "scenarios": []}
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

    for name in ("migrations.csv", "service_requests.csv", "scenarios.csv",
                 "benchmark_summary.csv",
                 "initial_time.png", "throughput.png", "concurrency_scaling.png",
                 "service_effects.png", "power_energy.png"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "migration_stages.csv").exists() == has_stage_table
    table = (tmp_path / "scenarios.csv").read_text()
    assert "continuation_difference_s" in table
    assert "measured_prompt_tokens" in table
    assert "context_size" not in table
