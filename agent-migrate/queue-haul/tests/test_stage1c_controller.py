"""
Claim: a deterministic trace manifest expands into matched, single-method
profiling scenarios, and reduction reports measured copy, pause, cache, stream,
and continuation quantities without power or deadline acceptance rules.

Plausible wrong implementations:
- Regenerate different conversations for repeats or split a trace turn.
- Change the selected session set across concurrency, methods, or bandwidths.
- Rerun identical controls for every method and bandwidth.
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

    plan = c.make_plan(
        manifest_path, [2048], [1, 2], [1000, 10000],
        ["replay", "kv_transfer"], ["none"], 1, 9,
    )

    assert {row["kind"] for row in plan["scenarios"]} == {"migration", "control"}
    assert len([row for row in plan["scenarios"] if row["kind"] == "control"]) == 2
    for method in c.METHODS:
        migration = [row for row in plan["scenarios"] if row["kind"] == "migration" and row["method"] == method]
        assert len(migration) == 4
        assert all(row["sessions"] == migration[0]["sessions"] for row in migration)
        assert all({move["method"] for move in row["moves"]} == {method} for row in migration)
    for match_id in {row["match_id"] for row in plan["scenarios"]}:
        matched = [row for row in plan["scenarios"] if row["match_id"] == match_id]
        assert sum(row["kind"] == "control" for row in matched) == 1
        assert len(matched) == 5
        assert all(row["sessions"] == matched[0]["sessions"] for row in matched)


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
    proxy = tmp_path / "proxy.csv"
    proxy.write_text(
        "monotonic_ns,wall_ns,interval_ns,connection_id,route,direction,bytes,billed\n"
        "0,0,250000000,a,kv,target_to_client,1000000,1\n"
        "0,0,250000000,b,kv,target_to_client,1000000,1\n"
        "250000000,0,250000000,a,kv,target_to_client,1000000,1\n"
        "250000000,0,250000000,b,kv,target_to_client,1000000,1\n"
        "0,0,250000000,z,api,client_to_target,999,1\n"
        "0,0,250000000,z,kv,target_to_client,999,0\n"
    )

    measured = c.parallel_connection_measurements(
        proxy, 0, 500_000_000, required=2,
    )

    assert measured == {
        "connection_count": 2,
        "max_parallel_connections": 2,
        "overlap_buckets": 2,
        "wire_bytes": 4000000,
    }
    proxy.write_text(proxy.read_text().replace(",b,kv", ",a,kv"))
    with pytest.raises(RuntimeError, match="independent"):
        c.parallel_connection_measurements(proxy, 0, 500_000_000, required=2)


def test_parallel_gate_rejects_sequential_connections_with_same_total_bytes(tmp_path):
    proxy = tmp_path / "proxy.csv"
    proxy.write_text(
        "monotonic_ns,wall_ns,interval_ns,connection_id,route,direction,bytes,billed\n"
        "0,0,250000000,a,kv,target_to_client,1000000,1\n"
        "250000000,0,250000000,a,kv,target_to_client,1000000,1\n"
        "500000000,0,250000000,b,kv,target_to_client,1000000,1\n"
        "750000000,0,250000000,b,kv,target_to_client,1000000,1\n"
    )

    with pytest.raises(RuntimeError, match="overlapping"):
        c.parallel_connection_measurements(proxy, 0, 1_000_000_000, required=2)


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
    script = Path(c.__file__).with_name("stage1c_benchmark.sbatch").read_text()

    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --exclusive" not in script
    assert "--power-state-cycles" in script
    assert "--node-power" not in script


def test_targeted_jobs_use_reviewed_plans_and_hard_gates():
    root = Path(c.__file__).parent
    runner = (root / "stage1_targeted_run.sh").read_text()
    jobs = {
        "stage1d_parallel_gate.sbatch":
            ("outputs/parallel-kv-gate-plan.json", "check-parallel"),
        "stage1e_catch_up.sbatch":
            ("outputs/append-catch-up-plan.json", "check-catch-up"),
    }

    assert "stage1b_drain_sink.py preflight --required-gpus 2" in runner
    assert "stage1c_controller.py run" in runner
    assert "stage1c_controller.py reduce" in runner
    assert '"$CHECK" --run-root' in runner
    assert "resume=()" not in runner and '"$@"' in runner
    for name, required in jobs.items():
        script = (root / name).read_text()
        assert "#SBATCH --gres=gpu:2" in script
        assert all(value in script for value in required)
        assert "stage1_targeted_run.sh" in script


def test_model_check_uses_measured_work_and_stays_in_its_valid_range():
    case = SimpleNamespace(
        kv_transfer=SimpleNamespace(setup_s=.2, destination_bytes_per_s=1_000_000, sync_s=.3),
        replay=SimpleNamespace(rate=lambda _tokens, _concurrency: 500),
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
