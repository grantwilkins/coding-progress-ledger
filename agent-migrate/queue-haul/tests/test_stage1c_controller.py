"""
Claim:
Stage 1c is a minimal controller proof: a tiny active-session fixture is mapped
through the existing Queue-Haul solver, produces both replay and KV actions, and
serializes those decisions into a manifest with hard acceptance checks.

Plausible wrong implementations:
- Bypass dispatch.solve or call it with the wrong signature.
- Let idle/cold sessions produce zero-cost moves.
- Produce a single-action proof fixture.
- Accept manifests without route-specific replay/KV evidence.
"""

from __future__ import annotations

import csv
import gzip
import json
import threading
import time
from pathlib import Path

import pytest

import stage1c_controller as c


def test_default_fixture_solves_to_replay_and_kv_mix():
    fixture = c.default_fixture()

    summary = c.plan_summary(fixture)
    actions = [s["action"] for s in summary["sessions"]]

    assert summary["solver"]["feasible"] is True
    assert summary["solver"]["shortfall_w"] == 0
    assert set(actions) == {"R", "S"}
    assert [s["dispatch_rank"] for s in summary["sessions"]] == list(range(len(actions)))
    assert summary["movement"]["lambda_src_bytes_per_s"] == 125_000_000.0


def test_population_hard_fails_non_active_sessions():
    fixture = c.default_fixture()
    fixture["sessions"][0]["state"] = "idle"

    with pytest.raises(ValueError, match="active"):
        c.build_population(fixture)


def test_plan_uses_fixture_costs_for_action_choice():
    fixture = c.default_fixture()

    rows = c.planned_sessions(fixture)
    by_id = {row["id"]: row["action"] for row in rows}

    assert by_id["r0"] == "R"
    assert by_id["k0"] == "S"


def test_manifest_check_requires_solver_mix_deadline_and_route_evidence():
    manifest = {
        "schema": c.SCHEMA,
        "solver": {"feasible": True, "shortfall_w": 0},
        "smoke2": {"acceptance": {"ok": True}},
        "sessions": [
            {"id": "r", "action": "R", "dispatch_rank": 0, "actual_start_s": 0.0, "actual_end_s": 1.0, "http_status": 200, "deadline_met": True, "proxy_delta": {"api/client_to_target": 123}},
            {"id": "s", "action": "S", "dispatch_rank": 1, "actual_start_s": 1.0, "actual_end_s": 2.0, "http_status": 200, "deadline_met": True, "proxy_delta": {"kv/target_to_client": 456}},
        ],
    }

    c.check_manifest(manifest)
    manifest["sessions"][1]["proxy_delta"] = {}
    with pytest.raises(ValueError, match="KV action"):
        c.check_manifest(manifest)
    manifest["sessions"][1]["proxy_delta"] = {"kv/target_to_client": 456}
    manifest["sessions"][1]["actual_start_s"] = 0.5
    with pytest.raises(ValueError, match="serial"):
        c.check_manifest(manifest)


def test_link_stack_logs_uses_relative_symlinks(tmp_path: Path):
    run_root = tmp_path / "run"
    stack_root = run_root / "stack"
    stack_root.mkdir(parents=True)
    (stack_root / "source.log").write_text("ok")

    c.link_stack_logs(run_root, stack_root)

    link = run_root / "source.log"
    assert link.is_symlink()
    assert link.readlink() == Path("stack/source.log")
    assert link.read_text() == "ok"




def _write_tracelab(path: Path) -> None:
    rows = []
    for sid, base in (("s0", 4096), ("s1", 8192)):
        for i in range(3):
            rows.append({
                "session_id": sid,
                "timestamp": 1000 + i * 60 + (0 if sid == "s0" else 10),
                "input_tokens_total": base + i * 256,
                "prefix_tokens": base - 128,
                "newly_append_tokens": 128 + i,
                "output_tokens": 32 + i,
            })
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_tracelab_manifest_groups_clamps_and_preserves_turns(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    _write_tracelab(trace)

    manifest = c.tracelab_manifest(trace, n_sessions=1, seed=0, max_model_len=4096, decode_margin=256, min_context_tokens=1024)
    session = manifest["sessions"][0]

    assert manifest["schema"] == c.MANIFEST_SCHEMA
    assert manifest["warmup_s"] == 30
    assert manifest["source"]["type"] == "tracelab"
    assert session["original_T"] > session["served_T"]
    assert session["served_T"] == 3840
    assert session["context_limit"] == 3840
    assert session["turn_rate_hz"] == pytest.approx(2 / 120)
    assert session["ell_pre"] == pytest.approx((2 / 120) * 129.5 / c.LIVE_A100_F_PREFILL_TPS)
    assert session["ell_dec"] == pytest.approx((2 / 120) * 33 / c.LIVE_A100_G_DECODE_TPS)
    assert [t["round"] for t in session["turns"]] == [0, 1, 2]
    assert session["turns"][1]["gap_s"] == 60


def test_tracelab_manifest_workload_selects_small_and_large(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    with gzip.open(trace, "wt") as f:
        for sid, base in (("tiny", 1200), ("mid", 4096), ("big", 12000)):
            for i in range(3):
                f.write(json.dumps({
                    "session_id": sid,
                    "timestamp": i + 10 * base,
                    "input_tokens_total": base + i * 100,
                    "prefix_tokens": max(0, base - 100),
                    "newly_append_tokens": 100,
                    "output_tokens": 16,
                }) + "\n")

    small = c.tracelab_manifest(trace, 2, 0, min_context_tokens=1024, workload="small")
    large = c.tracelab_manifest(trace, 2, 0, min_context_tokens=1024, workload="large")

    assert small["workload"]["name"] == "small"
    assert [s["id"] for s in small["sessions"]] == ["tiny", "mid"]
    assert [s["id"] for s in large["sessions"]] == ["big", "mid"]


def test_tracelab_workload_partitions_are_disjoint_and_pinned(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    with gzip.open(trace, "wt") as f:
        for group, gap in (("slow", 100), ("mid", 10), ("fast", 1)):
            for session_index in range(4):
                for turn in range(3):
                    f.write(json.dumps({
                        "session_id": f"{group}-{session_index}",
                        "timestamp": turn * gap,
                        "input_tokens_total": 2048 + turn * 100,
                        "prefix_tokens": 1900,
                        "newly_append_tokens": 100,
                        "output_tokens": 16,
                        "current_user_message_count": int(group == "slow"),
                        "tools": [{"tool_name": "Bash"}] if group == "fast" else [],
                    }) + "\n")

    manifests = {
        name: c.tracelab_manifest(trace, 2, 7, min_context_tokens=1024, workload=name)
        for name in ("interactive_coding", "coding", "agentic_tool_loop")
    }
    selected = [{s["id"] for s in manifest["sessions"]} for manifest in manifests.values()]

    assert not (selected[0] & selected[1] or selected[0] & selected[2] or selected[1] & selected[2])
    assert all(s["workload_class"] == name for name, manifest in manifests.items() for s in manifest["sessions"])
    assert {m["source"]["sha256"] for m in manifests.values()} == {c._file_sha256(trace)}
    assert all(m["workload"]["selection"] == "trace_partition" for m in manifests.values())


def test_tracelab_manifest_hard_fails_without_enough_long_sessions(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    _write_tracelab(trace)

    with pytest.raises(ValueError, match="after filtering"):
        c.tracelab_manifest(trace, n_sessions=3, seed=0, min_context_tokens=1024)


def test_session_prompt_rolls_followups_and_cache_busts(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    _write_tracelab(trace)
    session = c.tracelab_manifest(trace, 1, 0, max_model_len=4096, decode_margin=256, min_context_tokens=1024)["sessions"][0]
    session["scenario_nonce"] = "scenario-a"

    prompt = c.session_prompt(session, 2, replay_nonce="abc")

    assert "Scenario nonce scenario-a" in prompt
    assert "Replay nonce abc" in prompt
    assert "User follow-up 0" in prompt
    assert "User follow-up 2" in prompt
    assert session["id"] in prompt


def test_session_followup_does_not_replay_initial_turn(monkeypatch):
    session = {"id": "s", "turns": [{"append_tokens": 1000}, {"append_tokens": 2}, {"append_tokens": 3}]}
    monkeypatch.setattr(c, "_words", lambda tag, tokens: f"{tag}:{tokens}")

    assert "turn_s_1:2" in c.session_followup(session, 1)
    assert "turn_s_2:3" in c.session_followup(session, 2)
    assert "turn_s_3:2" in c.session_followup(session, 3)


def test_live_manifest_validation_requires_controller_fields():
    manifest = {"schema": c.MANIFEST_SCHEMA, "sessions": [{"id": "s"}]}

    with pytest.raises(ValueError, match="missing served_T"):
        c.live_sessions(manifest)


def test_prompt_tokens_counts_batch_encoding_input_ids(monkeypatch):
    class Tok:
        def apply_chat_template(self, msg, tokenize, add_generation_prompt):
            assert tokenize and add_generation_prompt
            assert msg[0]["content"] == "x"
            return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}

    monkeypatch.setattr(c, "_tokenizer", lambda _cfg: Tok())

    assert c.prompt_tokens(type("Cfg", (), {})(), "x") == 4


def test_tokenizer_initialization_is_thread_safe(tmp_path: Path, monkeypatch):
    root = tmp_path / "model"
    (root / "snapshot").mkdir(parents=True)
    calls, tokens = [], []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            calls.append(1)
            time.sleep(0.02)
            return object()

    monkeypatch.setattr(c.b, "model_snapshot_dir", lambda *_args: root)
    monkeypatch.setitem(__import__("sys").modules, "transformers", type("M", (), {"AutoTokenizer": AutoTokenizer}))
    c._TOKENIZER_CACHE.clear()
    threads = [threading.Thread(target=lambda: tokens.append(c._tokenizer(type("Cfg", (), {"hf_home": tmp_path, "model": "m"})()))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert len({id(token) for token in tokens}) == 1
    c._TOKENIZER_CACHE.clear()


def test_session_worker_advances_canonical_transcript_with_actual_output(tmp_path: Path, monkeypatch):
    class Log:
        def write(self, *_args, **_kwargs):
            pass

    session = {"id": "s", "served_T": 1024, "decode_tokens": 8, "turn_rate_hz": 1.0, "turns": [{"append_tokens": 4}], "messages": [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}], "generation": 1}
    cfg = type("Cfg", (), {"src_port": 8100, "api_proxy_port": 8400})()
    now = time.time()
    monkeypatch.setattr(c, "prompt_tokens", lambda *_args: 10)
    monkeypatch.setattr(c, "wait_session_kv_bytes", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(c, "stream_chat", lambda *_args: {"status": 200, "content": "actual-a1", "first_token_ts": now, "start_ts": now, "end_ts": now, "prompt_sha256": "x", "request_id": "req", "response_text": ""})
    worker = c.SessionWorker(cfg, session, cfg.src_port, Log(), 0, tmp_path)
    thread = threading.Thread(target=worker._serve)
    worker.threads = [thread]
    thread.start()
    with worker.cond:
        worker.pending = 1
        worker.cond.notify_all()
        assert worker.cond.wait_for(lambda: worker.generation == 2, 2)
    worker.stop()

    snapshot = worker.snapshot()
    assert snapshot["generation"] == 2
    assert [m["role"] for m in snapshot["messages"]] == ["user", "assistant", "user", "assistant"]
    assert snapshot["messages"][-1]["content"] == "actual-a1"


def test_session_worker_hard_fails_if_migration_would_trim_prefix(tmp_path: Path, monkeypatch):
    session = {"id": "s", "served_T": 5, "context_limit": 10, "decode_tokens": 1, "turn_rate_hz": 1, "turns": [{"append_tokens": 1}], "messages": [{"role": "user", "content": "u"}], "generation": 1}
    worker = c.SessionWorker(type("Cfg", (), {})(), session, 1, type("Log", (), {"write": lambda *_a, **_k: None})(), 0, tmp_path)
    monkeypatch.setattr(c, "prompt_tokens", lambda *_args: 11)

    worker.begin_migration()
    with pytest.raises(RuntimeError, match="append-only migration limit"):
        worker._bounded(worker.messages)


def test_source_request_p95_uses_serving_telemetry():
    workers = {"a": type("W", (), {"request_s": [1.0, 2.0]})(), "b": type("W", (), {"request_s": [3.0, 4.0]})()}

    assert c.source_request_p95_s(workers) == pytest.approx(3.85)
    with pytest.raises(RuntimeError, match="no source request"):
        c.source_request_p95_s({"a": type("W", (), {"request_s": []})()})


def test_begin_migration_keeps_serving_and_snapshots_inflight_prompt(tmp_path: Path):
    session = {"id": "s", "served_T": 10, "decode_tokens": 1, "turn_rate_hz": 1, "turns": [{"append_tokens": 1}], "messages": [{"role": "user", "content": "old"}], "generation": 1}
    worker = c.SessionWorker(type("Cfg", (), {})(), session, 1, type("Log", (), {"write": lambda *_a, **_k: None})(), 0, tmp_path)
    worker.in_flight = True
    worker.in_flight_messages = worker.messages + [{"role": "user", "content": "pending"}]

    snapshot = worker.begin_migration()

    assert not worker.paused and worker.migrating
    assert snapshot["messages"][-1]["content"] == "pending"


def test_session_worker_queues_turns_and_hard_fails_on_overflow(tmp_path: Path):
    class Log:
        def __init__(self):
            self.kinds = []
        def write(self, kind, **_kwargs):
            self.kinds.append(kind)

    session = {"id": "s", "served_T": 10, "decode_tokens": 1, "turn_rate_hz": 1, "turns": [{"append_tokens": 1}], "messages": [{"role": "user", "content": "old"}], "generation": 1, "max_pending_turns": 2}
    log = Log()
    worker = c.SessionWorker(type("Cfg", (), {})(), session, 1, log, 0, tmp_path)

    assert worker.queue_arrival(1.0)
    assert worker.queue_arrival(2.0)
    assert not worker.queue_arrival(3.0)
    assert worker.pending == 2
    assert "turn_drop" not in log.kinds
    with pytest.raises(RuntimeError, match="exceeded 2 queued turns"):
        worker.snapshot()


def test_session_worker_surfaces_thread_failure(tmp_path: Path, monkeypatch):
    session = {"id": "s", "served_T": 10, "decode_tokens": 1, "turn_rate_hz": 1, "turns": [{"append_tokens": 1}], "messages": [{"role": "user", "content": "old"}], "generation": 1}
    worker = c.SessionWorker(type("Cfg", (), {})(), session, 1, type("Log", (), {"write": lambda *_a, **_k: None})(), 0, tmp_path)
    monkeypatch.setattr(c, "prompt_tokens", lambda *_args: (_ for _ in ()).throw(ValueError("broken tokenizer")))
    thread = threading.Thread(target=worker._serve)
    thread.start()
    with worker.cond:
        worker.pending = 1
        worker.cond.notify_all()
        assert worker.cond.wait_for(lambda: worker.error is not None, 1)
    thread.join()

    with pytest.raises(RuntimeError, match="broken tokenizer"):
        worker.snapshot()


def test_live_plan_uses_calibrated_power_curve_and_dispatches_all(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    _write_tracelab(trace)
    manifest = c.tracelab_manifest(trace, 2, 0, max_model_len=4096, decode_margin=256, min_context_tokens=1024)
    manifest["deadline_s"] = 1e9
    manifest["constants"]["lambda_src_bytes_per_s"] = 1e18
    manifest["constants"]["mu_bytes_per_s"] = 1e18

    summary = c.live_plan_summary(manifest)
    one_session = c.live_plan_summary({**manifest, "sessions": manifest["sessions"][:1]})
    mechanism = c.live_plan_summary({**manifest, "dest_load_budget_ell": 2.0})

    assert summary["power_curve"]["name"] == "saturating"
    assert summary["power_curve"]["p_idle_w"] == pytest.approx(c.LIVE_A100_P_IDLE_W)
    assert summary["power_curve"]["p_busy_w"] == pytest.approx(c.LIVE_A100_P_BUSY_W)
    assert summary["power_curve"]["power_knee"] == pytest.approx(c.LIVE_A100_POWER_KNEE)
    assert summary["power_curve"]["rho_star"] == pytest.approx(c.LIVE_A100_RHO_STAR)
    assert one_session["power_curve"]["rho_star"] == summary["power_curve"]["rho_star"]
    assert summary["destination_load_budget_ell"] == summary["destination_admission_limit_ell"]
    assert summary["experiment_mode"] == "admission"
    assert mechanism["destination_load_budget_ell"] == 2.0
    assert mechanism["destination_admission_limit_ell"] == summary["destination_admission_limit_ell"]
    assert mechanism["experiment_mode"] == "mechanism_only"
    assert mechanism["selected_destination_load_ell"] == pytest.approx(summary["selected_destination_load_ell"])
    assert summary["full_source_drop_w"] <= c.LIVE_A100_P_BUSY_W - c.LIVE_A100_P_IDLE_W + 1e-6
    assert len(summary["sessions"]) == 2
    assert [s["dispatch_rank"] for s in summary["sessions"]] == [0, 1]
    assert summary["sessions"][-1]["predicted_cumulative_source_drop_w"] == pytest.approx(summary["full_source_drop_w"])


def test_run_live_moves_stages_one_token_with_bounded_replay_and_overlapping_kv(tmp_path: Path, monkeypatch):
    dispatches = []

    class Log:
        def write(self, kind, **row):
            if kind == "handoff_stage_start" and row["phase"] == "initial":
                dispatches.append((row["action"], row["session_id"]))

    class Worker:
        def __init__(self, sid):
            self.sid, self.generation = sid, 1
            self.session = {"scenario_nonce": "test"}
            self.messages = [{"role": "user", "content": f"prompt-{sid}"}]
            self.pause_times = []
            self.resumed = False
            self.port = None
            self.log = Log()
            self.sink_log = tmp_path / "sink.log"

        def snapshot(self):
            return {"messages": self.messages, "generation": self.generation, "context_sha256": c.messages_sha256(self.messages), "last_request_id": None, "session_kv_bytes": 1}

        def pause_boundary(self):
            self.pause_times.append(time.time())

        def commit_switch(self, expected_generation, port, messages):
            assert expected_generation == self.generation
            self.port, self.messages = port, messages
            return True

        def resume(self):
            self.resumed = True

    cfg = type("Cfg", (), {"api_proxy_port": 8400})()
    workers = {sid: Worker(sid) for sid in ("a", "b", "c", "d", "e")}
    rows = [
        {"id": "a", "action": "R", "fixture_index": 0, "dispatch_rank": 0, "deadline_s": 2.0},
        {"id": "b", "action": "R", "fixture_index": 1, "dispatch_rank": 1, "deadline_s": 2.0},
        {"id": "c", "action": "S", "fixture_index": 2, "dispatch_rank": 2, "deadline_s": 2.0},
        {"id": "d", "action": "S", "fixture_index": 3, "dispatch_rank": 3, "deadline_s": 2.0},
        {"id": "e", "action": "S", "fixture_index": 4, "dispatch_rank": 4, "deadline_s": 2.0},
    ]
    starts, ends = {"R": [], "S": []}, {}

    def fake_stream_chat(_cfg, _port, messages, max_tokens):
        assert max_tokens == 1
        action = "R" if messages[0]["role"] == "system" else "S"
        sid = messages[0]["content"].split()[3] if action == "R" else messages[0]["content"].rsplit("-", 1)[1]
        start = time.time()
        starts[action].append(start)
        time.sleep(0.2)
        end = time.time()
        ends[sid] = end
        return {"status": 200, "content": "x", "first_token_ts": start + 0.01, "start_ts": start, "end_ts": end, "prompt_sha256": c.messages_sha256(messages), "request_id": f"req-{sid}", "response_text": ""}

    monkeypatch.setattr(c, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(c, "wait_lmcache_lookup_tokens", lambda _path, request_id, timeout_s=10: (10, 10) if request_id in {"req-c", "req-d", "req-e"} else (10, 0))
    t0 = time.time()
    out = c.run_live_moves(cfg, tmp_path, workers, rows, replay_concurrency=1, kv_concurrency=2)

    assert time.time() - t0 < 0.55
    assert starts["R"][1] - starts["R"][0] >= 0.18
    assert starts["S"][0] - starts["R"][0] < 0.1
    assert [sid for action, sid in dispatches if action == "S"] == ["c", "d", "e"]
    assert starts["S"][2] - min(starts["S"][:2]) >= 0.18
    assert [r["dispatch_rank"] for r in out] == [0, 1, 2, 3, 4]
    assert all(workers[sid].pause_times[0] >= ends[sid] for sid in workers)
    assert all(w.resumed and w.port == 8400 for w in workers.values())
    assert all(r["commit_result"] == "committed" and r["staging_attempts"] == 1 for r in out)
    assert all(r["staging_s"] < r["completion_s"] and r["switch_downtime_s"] >= 0 for r in out)


def test_run_live_moves_reconciles_stale_generation_with_final_delta(tmp_path: Path, monkeypatch):
    class Log:
        def write(self, *_args, **_kwargs):
            pass

    class Worker:
        log = Log()
        sink_log = tmp_path / "sink.log"

        def __init__(self):
            self.generation, self.port = 1, 8100
            self.session = {"scenario_nonce": "test"}
            self.messages = [{"role": "user", "content": "prompt"}]

        def snapshot(self):
            return {"messages": self.messages, "generation": self.generation, "context_sha256": c.messages_sha256(self.messages), "last_request_id": None, "session_kv_bytes": 1}

        def pause_boundary(self):
            self.generation += 1
            self.messages = self.messages + [{"role": "assistant", "content": "new"}]

        def commit_switch(self, expected_generation, port, messages):
            return self.generation == expected_generation

        def resume(self):
            pass

    now, actions = time.time(), []

    def fake_stream(_cfg, _port, messages, _max_tokens):
        action = "R" if messages[0]["role"] == "system" else "S"
        actions.append(action)
        return {"status": 200, "content": "x", "first_token_ts": now, "start_ts": now, "end_ts": now, "prompt_sha256": c.messages_sha256(messages), "request_id": f"req-{action}", "response_text": ""}

    monkeypatch.setattr(c, "stream_chat", fake_stream)
    monkeypatch.setattr(c, "wait_lmcache_lookup_tokens", lambda _path, request_id, timeout_s=10: (10, 10 if request_id == "req-S" else 0))
    row = {"id": "a", "action": "S", "fixture_index": 0, "dispatch_rank": 0, "deadline_s": 2.0}

    out = c.run_live_moves(type("Cfg", (), {"api_proxy_port": 8400})(), tmp_path, {"a": Worker()}, [row])

    assert actions == ["S", "S"]
    assert out[0]["commit_result"] == "committed"
    assert out[0]["generation_delta"] == 1
    assert out[0]["staging_attempts"] == 2
    assert out[0]["effective_action"] == "S+delta"
    assert out[0]["final_delta_action"] == "S"
    assert out[0]["append_only_context"] is True
    assert out[0]["committed_context_sha256"] == out[0]["staged_context_sha256"]


def test_run_live_moves_releases_stage_slot_while_waiting_for_source(tmp_path: Path, monkeypatch):
    second_started = threading.Event()
    overlapped = []

    class Worker:
        log = type("Log", (), {"write": lambda *_a, **_k: None})()
        sink_log = tmp_path / "sink.log"
        session = {"scenario_nonce": "test"}

        def __init__(self, sid):
            self.sid, self.generation = sid, 1
            self.messages = [{"role": "user", "content": sid}]

        def snapshot(self):
            return {"messages": self.messages, "generation": 1, "context_sha256": c.messages_sha256(self.messages), "last_request_id": None, "session_kv_bytes": 1}

        def pause_boundary(self):
            if self.sid == "a":
                overlapped.append(second_started.wait(0.5))

        def commit_switch(self, *_args):
            return True

        def resume(self):
            pass

    def stream(_cfg, _port, messages, _max_tokens):
        sid = messages[0]["content"].split()[3]
        if sid == "b":
            second_started.set()
        now = time.time()
        return {"status": 200, "content": "x", "first_token_ts": now, "start_ts": now, "end_ts": now, "prompt_sha256": c.messages_sha256(messages), "request_id": sid, "response_text": ""}

    monkeypatch.setattr(c, "stream_chat", stream)
    monkeypatch.setattr(c, "wait_lmcache_lookup_tokens", lambda *_args, **_kwargs: (10, 0))
    rows = [{"id": sid, "action": "R", "fixture_index": i, "dispatch_rank": i, "deadline_s": 2} for i, sid in enumerate(("a", "b"))]

    c.run_live_moves(type("Cfg", (), {"api_proxy_port": 8400})(), tmp_path, {sid: Worker(sid) for sid in ("a", "b")}, rows, replay_concurrency=1)

    assert overlapped == [True]


def test_run_live_moves_rejects_rewritten_prefix(tmp_path: Path, monkeypatch):
    worker = type("Worker", (), {})()
    worker.log = type("Log", (), {"write": lambda *_a, **_k: None})()
    worker.sink_log = tmp_path / "sink.log"
    worker.session = {"scenario_nonce": "test"}
    worker.generation = 1
    worker.messages = [{"role": "user", "content": "old"}]
    worker.snapshot = lambda: {"messages": worker.messages, "generation": worker.generation, "context_sha256": c.messages_sha256(worker.messages), "last_request_id": None, "session_kv_bytes": 1}
    worker.pause_boundary = lambda: (setattr(worker, "generation", 2), setattr(worker, "messages", [{"role": "user", "content": "new"}]))
    worker.resume = lambda: None
    now = time.time()
    monkeypatch.setattr(c, "stream_chat", lambda _cfg, _port, messages, _max: {"status": 200, "content": "x", "first_token_ts": now, "start_ts": now, "end_ts": now, "prompt_sha256": c.messages_sha256(messages), "request_id": "r", "response_text": ""})
    monkeypatch.setattr(c, "wait_lmcache_lookup_tokens", lambda *_args, **_kwargs: (10, 0))

    with pytest.raises(RuntimeError, match="changed a transferred prefix"):
        c.run_live_moves(type("Cfg", (), {"api_proxy_port": 8400})(), tmp_path, {"a": worker}, [{"id": "a", "action": "R", "fixture_index": 0, "dispatch_rank": 0, "deadline_s": 2}])


def test_nvsmi_command_uses_250ms_sampling():
    assert c.nvsmi_cmd(250)[-2:] == ["-lms", "250"]


def test_memlog_command_records_cgroup_and_top_rss():
    cmd = " ".join(c.memlog_cmd(0.5))

    assert "memory.current" in cmd
    assert "memory.events" in cmd
    assert "ps -u $USER" in cmd
    assert "sleep 0.5" in cmd


def test_power_summary_rows_uses_named_windows(tmp_path: Path):
    path = tmp_path / "gpu_power.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["timestamp", "index", "power_w", "util_gpu", "memory_mib"])
        writer.writeheader()
        writer.writerows([
            {"timestamp": 0, "index": 0, "power_w": 100, "util_gpu": 0, "memory_mib": 1},
            {"timestamp": 1, "index": 0, "power_w": 200, "util_gpu": 0, "memory_mib": 1},
            {"timestamp": 0, "index": 1, "power_w": 50, "util_gpu": 0, "memory_mib": 1},
        ])

    rows = c.power_summary_rows(path, {"baseline": (0, 1)})

    by_gpu = {r["gpu"]: r for r in rows}
    assert by_gpu[0]["power_mean_w"] == pytest.approx(150)
    assert by_gpu[1]["power_mean_w"] == pytest.approx(50)


def test_power_summary_rows_skips_nvidia_smi_na_samples(tmp_path: Path):
    path = tmp_path / "gpu_power.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["timestamp", "index", "power_w", "util_gpu", "memory_mib"])
        writer.writeheader()
        writer.writerows([
            {"timestamp": 0, "index": 0, "power_w": 100, "util_gpu": 0, "memory_mib": 1},
            {"timestamp": 0.5, "index": 0, "power_w": " [N/A]", "util_gpu": 0, "memory_mib": 1},
            {"timestamp": 1, "index": 0, "power_w": 300, "util_gpu": 0, "memory_mib": 1},
        ])

    rows = c.power_summary_rows(path, {"baseline": (0, 1)})

    assert rows == [{"phase": "baseline", "gpu": 0, "samples": 2, "total_samples": 3, "coverage": pytest.approx(2 / 3), "power_mean_w": 200.0}]
    with pytest.raises(RuntimeError, match="coverage"):
        c.require_power_coverage(rows)
    c.require_power_coverage(rows, 0.5)


def test_source_power_drop_uses_measured_windows():
    rows = [{"phase": phase, "gpu": 0, "power_mean_w": power} for phase, power in (("baseline", 250), ("post", 170))]

    assert c.source_power_drop_w(rows) == 80


def test_check_live_manifest_requires_files_and_route_evidence(tmp_path: Path):
    for name in c.LIVE_ARTIFACTS:
        (tmp_path / name).write_text("x")
    common = {"http_status": 200, "first_token_s": 0.1, "prompt_sha256": "hash", "staged_context_sha256": "hash", "committed_context_sha256": "hash", "commit_result": "committed", "deadline_met": True, "stage_lmcache_total_tokens": 10}
    manifest = {
        "schema": c.LIVE_SCHEMA,
        "sessions": [
            {**common, "id": "r", "action": "R", "dispatch_rank": 0, "stage_lmcache_hit_tokens": 0, "proxy_delta": {"api/client_to_target": 10}},
            {**common, "id": "s", "action": "S", "dispatch_rank": 1, "stage_lmcache_hit_tokens": 10, "proxy_delta": {"kv/target_to_client": 10}},
        ],
    }

    c.check_live_manifest(manifest, tmp_path)
    manifest["sessions"][1]["stage_lmcache_hit_tokens"] = 0
    with pytest.raises(ValueError, match="KV action"):
        c.check_live_manifest(manifest, tmp_path)



def test_ell_power5s_rows_joins_power_to_live_ell(tmp_path: Path):
    path = tmp_path / "gpu_power.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["timestamp", "index", "power_w", "util_gpu", "memory_mib"])
        writer.writeheader()
        for ts, p0, p1 in ((0, 300, 80), (1, 320, 90), (6, 150, 200), (7, 170, 220)):
            writer.writerow({"timestamp": ts, "index": 0, "power_w": p0, "util_gpu": 0, "memory_mib": 1})
            writer.writerow({"timestamp": ts, "index": 1, "power_w": p1, "util_gpu": 0, "memory_mib": 1})
    manifest = {
        "input_manifest": {"sessions": [
            {"id": "s0", "ell_pre": 0.1, "ell_dec": 0.05},
            {"id": "s1", "ell_pre": 0.2, "ell_dec": 0.05},
        ]},
        "sessions": [{"id": "s0", "move_start_ts": 6.0}],
        "windows": {"baseline": [0, 5], "drain": [5, 10]},
    }

    rows = c.ell_power5s_rows(path, manifest, bucket_s=5)

    by_key = {(r["bucket"], r["gpu"]): r for r in rows}
    assert by_key[(0, 0)]["ell"] == pytest.approx(0.4)
    assert by_key[(0, 1)]["ell"] == pytest.approx(0.0)
    assert by_key[(1, 0)]["ell"] == pytest.approx(0.25)
    assert by_key[(1, 1)]["ell"] == pytest.approx(0.15)
    assert by_key[(0, 0)]["power_mean_w"] == pytest.approx(310)
    assert by_key[(1, 1)]["power_mean_w"] == pytest.approx(210)


def test_write_ell_power5s_writes_csv_and_plot(tmp_path: Path):
    path = tmp_path / "gpu_power.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["timestamp", "index", "power_w", "util_gpu", "memory_mib"])
        writer.writeheader()
        writer.writerows([
            {"timestamp": 0, "index": 0, "power_w": 100, "util_gpu": 0, "memory_mib": 1},
            {"timestamp": 0, "index": 1, "power_w": 50, "util_gpu": 0, "memory_mib": 1},
        ])
    manifest = {"input_manifest": {"sessions": [{"id": "s", "ell_pre": 0.1, "ell_dec": 0.1}]}, "sessions": [], "windows": {"baseline": [0, 5]}}

    rows = c.write_ell_power5s(path, manifest, tmp_path / "ell_power5s.csv", tmp_path / "ell_power5s.png")

    assert rows[0]["ell"] == pytest.approx(0.2)
    assert (tmp_path / "ell_power5s.csv").read_text().splitlines()[0] == "bucket,bucket_start_s,bucket_end_s,gpu,node,ell,power_mean_w,samples"
    assert (tmp_path / "ell_power5s.png").exists()


def test_delay_summary_writes_total_delay_csv_and_plot(tmp_path: Path):
    rows = [
        {"dispatch_rank": 0, "id": "s0", "action": "R", "first_token_s": 1.5, "completion_s": 2.5},
        {"dispatch_rank": 1, "id": "s1", "action": "S", "first_token_s": 3.0, "completion_s": 4.0},
    ]

    delays = c.write_delay_summary(rows, tmp_path / "delay_summary.csv", tmp_path / "delay_summary.png")

    assert sum(d["first_token_s"] for d in delays) == pytest.approx(4.5)
    assert (tmp_path / "delay_summary.csv").read_text().splitlines()[0] == "dispatch_rank,id,action,commit_result,selection_to_stage_start_s,initial_staging_s,final_delta_s,staging_s,first_token_s,sink_warm_s,completion_s,source_boundary_wait_s,switch_downtime_s"
    assert (tmp_path / "delay_summary.png").exists()



def _manifest_for_live_policy_tests(tmp_path: Path) -> dict:
    trace = tmp_path / "trace.jsonl.gz"
    _write_tracelab(trace)
    manifest = c.tracelab_manifest(trace, 2, 0, max_model_len=4096, decode_margin=256, min_context_tokens=1024)
    manifest["constants"]["lambda_src_bytes_per_s"] = 1e18
    manifest["constants"]["mu_bytes_per_s"] = 1e18
    return manifest


def test_live_plan_supports_policy_deadline_and_target_fraction(tmp_path: Path):
    manifest = _manifest_for_live_policy_tests(tmp_path)

    policies = ("lp", "milp", "power-unaware", "random", "greedy")
    summaries = [c.live_plan_summary(manifest, policy=p, deadline_s=30, target_frac=0.25, seed=3) for p in policies]

    assert {s["policy"] for s in summaries} == set(policies)
    for summary in summaries:
        assert summary["deadline_s"] == 30
        assert summary["target_frac"] == pytest.approx(0.25)
        assert summary["target_w"] == pytest.approx(0.25 * summary["full_source_drop_w"])
        assert summary["sessions"]
        assert [r["dispatch_rank"] for r in summary["sessions"]] == list(range(len(summary["sessions"])))
    greedy = next(s for s in summaries if s["policy"] == "greedy")
    assert greedy["solver"]["method"] == "node_aware_greedy"
    assert next(s for s in summaries if s["policy"] == "lp")["solver"]["method"] == "power_function_lp_rounded"
    assert next(s for s in summaries if s["policy"] == "milp")["solver"]["method"] == "single_source_milp"


def test_counterfactual_live_plans_force_single_action(tmp_path: Path):
    manifest = _manifest_for_live_policy_tests(tmp_path)

    all_r = c.live_plan_summary(manifest, policy="all-r", target_frac=1.0)
    all_s = c.live_plan_summary(manifest, policy="all-s", target_frac=1.0)

    assert {r["action"] for r in all_r["sessions"]} == {"R"}
    assert {r["action"] for r in all_s["sessions"]} == {"S"}
    assert all_r["solver"]["method"] == "all-r"
    assert all_s["solver"]["method"] == "all-s"


def test_random_live_plan_is_seed_reproducible(tmp_path: Path):
    manifest = _manifest_for_live_policy_tests(tmp_path)

    a = c.live_plan_summary(manifest, policy="random", target_frac=1.0, seed=7)
    b = c.live_plan_summary(manifest, policy="random", target_frac=1.0, seed=7)

    assert [(r["id"], r["action"]) for r in a["sessions"]] == [(r["id"], r["action"]) for r in b["sessions"]]


def test_live_plan_records_target_miss_without_raising(tmp_path: Path):
    manifest = _manifest_for_live_policy_tests(tmp_path)

    summary = c.live_plan_summary(manifest, policy="greedy", target_frac=1.1)

    assert summary["planned_hit"] is False
    assert summary["planned_shortfall_w"] > 0




def test_live_grid_multi_manifest_uses_workload_dirs_profiles_and_one_stack(tmp_path: Path, monkeypatch):
    calls, stacks, stopped = [], [], []

    def fake_start_stack(_cfg, root, _mbps, _extra):
        stack = type("Stack", (), {"run_root": root})()
        stacks.append(root)
        return stack

    def fake_live_drain(_cfg, dst, manifest, _mbps, _nvsmi_ms, _extra, policy, _seed, D, frac, profile, _rc, _kc, stack):
        calls.append((dst, manifest["workload"]["name"], policy, D, frac, profile, stack.run_root))

    monkeypatch.setattr(c.b, "start_stack", fake_start_stack)
    monkeypatch.setattr(c.b, "start_sink", lambda *_args: None)
    monkeypatch.setattr(c.b, "stop_stack", lambda stack: stopped.append(stack.run_root))
    monkeypatch.setattr(c, "live_drain", fake_live_drain)
    monkeypatch.setattr(c, "write_grid_summary", lambda roots, *_args: roots)
    manifests = [
        {"schema": c.MANIFEST_SCHEMA, "workload": {"name": "small"}, "sessions": [{"served_T": 1024}]},
        {"schema": c.MANIFEST_SCHEMA, "workload": {"name": "large"}, "sessions": [{"served_T": 32768}]},
    ]

    c.live_grid(type("Cfg", (), {})(), tmp_path, manifests, ["greedy"], [30.0], [0.45], 1000.0, 250, 1.0, 1.0, 0, [], tmp_path / "profile.json")

    assert stacks == [tmp_path / "stack"]
    assert stopped == stacks
    assert calls == [
        (tmp_path / "small" / "greedy_D30_T0p45", "small", "greedy", 30.0, 0.45, tmp_path / "profile_small.json", tmp_path / "stack"),
        (tmp_path / "large" / "greedy_D30_T0p45", "large", "greedy", 30.0, 0.45, tmp_path / "profile_large.json", tmp_path / "stack"),
    ]


def test_smart_jobs_use_profile_deadlines_and_only_repeat_random_and_greedy(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(c, "apply_live_profile", lambda manifest, _profile: manifest)
    monkeypatch.setattr(c, "live_plan_summary", lambda _manifest, _policy, **kwargs: {
        "planned_completion_s": 100 * kwargs["target_frac"]
    })

    jobs = c._smart_jobs(tmp_path, {}, {}, ["greedy", "random", "all-r", "all-s"],
                         [0.25, 0.65], [0.75, 1.0, 1.5], [0, 1, 2], 1, None)

    assert len(jobs) == 16
    assert [job[2] for job in jobs].count("greedy") == 6
    assert [job[2] for job in jobs].count("random") == 6
    assert [job[2] for job in jobs].count("all-r") == 2
    assert [job[2] for job in jobs].count("all-s") == 2
    assert {job[3] for job in jobs if job[2] == "random"} == {0, 1, 2}
    assert {job[4] for job in jobs if job[5] == 0.25 and job[2] == "greedy"} == {18.75, 25.0, 37.5}
    assert all(job[1]["scenario"]["reference_deadline_s"] == 25.0
               for job in jobs if job[5] == 0.25)
    assert len({job[0] for job in jobs}) == len(jobs)

    with pytest.raises(ValueError, match="offline"):
        c._smart_jobs(tmp_path, {}, {}, ["lp"], [0.25], [1.0], [0], 1, None)


def test_write_vllm_metrics_creates_fresh_role_snapshots(tmp_path: Path, monkeypatch):
    cfg = type("Cfg", (), {"host": "127.0.0.1", "src_port": 1, "sink_port": 2})()
    monkeypatch.setattr(c.b, "http_text", lambda _host, port, _method, _path: f"port {port}\n")

    c.write_vllm_metrics(cfg, tmp_path, "before")

    assert (tmp_path / "source_metrics_before.prom").read_text() == "port 1\n"
    assert (tmp_path / "sink_metrics_before.prom").read_text() == "port 2\n"


def test_live_grid_skips_completed_scenarios(tmp_path: Path, monkeypatch):
    dst = tmp_path / "greedy_D30_T0p45"
    dst.mkdir()
    (dst / "controller_manifest.json").write_text("{}")
    monkeypatch.setattr(c.b, "start_stack", lambda *_args: pytest.fail("started stack for completed scenario"))
    monkeypatch.setattr(c, "live_drain", lambda *_args: pytest.fail("reran completed scenario"))
    monkeypatch.setattr(c, "write_grid_summary", lambda roots, *_args: roots)

    c.live_grid(type("Cfg", (), {})(), tmp_path, {"schema": c.MANIFEST_SCHEMA, "workload": {"name": "large"}, "sessions": [{"served_T": 32768}]}, ["greedy"], [30.0], [0.45], 1000.0, 250, 1.0, 1.0, 0, [], tmp_path / "profile.json")


def test_live_profile_costs_override_runtime_model(tmp_path: Path):
    manifest = _manifest_for_live_policy_tests(tmp_path)
    profile = {
        "schema": c.PROFILE_SCHEMA,
        "mbps": 1000.0,
        "points": [
            {"action": "R", "tokens": 3000, "completion_s": 30.0},
            {"action": "R", "tokens": 5000, "completion_s": 50.0},
            {"action": "S", "tokens": 3000, "kv_bytes": 1000, "completion_s": 10.0},
            {"action": "S", "tokens": 5000, "kv_bytes": 3000, "completion_s": 20.0},
        ],
    }

    for i, session in enumerate(manifest["sessions"]):
        session["session_kv_bytes"] = 1000 + 2000 * i
    patched = c.apply_live_profile(manifest, profile)
    sessions, _pop, _pool, _move, imp, *_ = c._live_model(patched)
    summary = c.live_plan_summary(patched, target_frac=0.25)

    for i, session in enumerate(sessions):
        assert imp.c_replay[i] == pytest.approx(session["c_replay_s"])
        assert imp.c_transfer[i] == pytest.approx(session["c_transfer_s"])
    assert sessions[0]["profile_transfer_bytes"] == manifest["sessions"][0]["session_kv_bytes"]
    assert all(s["profile_transfer_bytes_source"] == "measured_session" for s in sessions)
    for session in sessions:
        assert session["c_transfer_s"] == pytest.approx(session["profile_transfer_stage_s"])

    assert summary["profile"]["schema"] == c.PROFILE_SCHEMA


def test_planned_wall_counts_parallel_source_boundary_once():
    rows = [{"action": "R", "planned_finish_s": 1, "fixture_index": i} for i in range(2)]

    assert c._planned_wall_s(rows, [{}, {}], {"source_boundary_s": 5}, replay_concurrency=1) == 7


def test_live_profile_uses_profiled_transfer_bytes_and_rate(tmp_path: Path):
    manifest = _manifest_for_live_policy_tests(tmp_path)
    for i, session in enumerate(manifest["sessions"]):
        session["session_kv_bytes"] = 1 if i == 0 else 50_000
    profile = {
        "schema": c.PROFILE_SCHEMA,
        "mbps": 1000.0,
        "points": [
            {"action": "R", "tokens": 3000, "completion_s": 3.0},
            {"action": "R", "tokens": 5000, "completion_s": 5.0},
            {"action": "S", "tokens": 3000, "kv_bytes": 30_000, "completion_s": 30.0, "source_elapsed_s": 2.0},
            {"action": "S", "tokens": 5000, "kv_bytes": 50_000, "completion_s": 50.0, "source_elapsed_s": 2.0},
        ],
    }

    patched = c.apply_live_profile(manifest, profile)
    sessions, _pop, _pool, move, imp, *_ = c._live_model(patched)

    assert move.lambda_src == pytest.approx(1000.0)
    assert sessions[0]["profile_transfer_bytes"] == pytest.approx(1.0)
    assert sessions[1]["profile_transfer_bytes"] == pytest.approx(50_000.0)
    assert imp.b_transfer.tolist() == pytest.approx([s["profile_transfer_bytes"] for s in sessions])


def test_live_plan_hit_requires_profiled_aggregate_deadline(tmp_path: Path):
    manifest = _manifest_for_live_policy_tests(tmp_path)
    manifest["deadline_s"] = 10.0
    for session in manifest["sessions"]:
        session["session_kv_bytes"] = 1
    profile = {
        "schema": c.PROFILE_SCHEMA,
        "mbps": 1000.0,
        "points": [
            {"action": "R", "tokens": 3000, "completion_s": 1.0},
            {"action": "R", "tokens": 5000, "completion_s": 1.0},
            {"action": "S", "tokens": 3000, "kv_bytes": 30_000, "completion_s": 30.0, "source_elapsed_s": 2.0},
            {"action": "S", "tokens": 5000, "kv_bytes": 50_000, "completion_s": 50.0, "source_elapsed_s": 2.0},
        ],
    }

    summary = c.live_plan_summary(c.apply_live_profile(manifest, profile), policy="all-s", target_frac=1.0)

    assert summary["planned_power_hit"] is True
    assert summary["planned_deadline_hit"] is False
    assert summary["planned_hit"] is False
    assert summary["planned_completion_s"] > summary["deadline_s"]


def test_grid_sbatch_defaults_to_old_runtime():
    text = Path("queue-haul/stage1c_grid.sbatch").read_text()

    assert "QH_APPTAINER_IMAGE=${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox}" in text
    assert "QH_APPTAINER_GPU_MODE=${QH_APPTAINER_GPU_MODE:-nv}" in text
    assert "QH_PORT_OFFSET=${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}" in text
    assert 'SMART_ARGS=(--deadline-scales "$DEADLINE_SCALES" --random-seeds "$RANDOM_SEEDS")' in text
    assert "KV_CONCURRENCY=${KV_CONCURRENCY:-2}" in text
    assert "DEST_LOAD_BUDGET_ELL=${DEST_LOAD_BUDGET_ELL:-}" in text
    assert 'SMART_ARGS+=(--dest-load-budget-ell "$DEST_LOAD_BUDGET_ELL")' in text
    assert "DEST_LOAD_ARGS" not in text
    assert c.parse_args(["live-drain", "--manifest", "sessions.json"]).kv_concurrency == 2
    assert c.parse_args(["live-grid", "--manifest", "sessions.json", "--dest-load-budget-ell", "2"]).dest_load_budget_ell == 2


def test_profile_prompt_has_cache_namespace(monkeypatch):
    monkeypatch.setattr(c, "prompt_tokens", lambda _cfg, _prompt: 1024)

    prompt, _tokens = c.profile_prompt(type("Cfg", (), {})(), 1024, "profile-a")

    assert "calibration session profile-a" in prompt


def test_live_profile_recalibrates_on_lmcache_runtime_change(tmp_path: Path, monkeypatch):
    manifest = _manifest_for_live_policy_tests(tmp_path)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema": c.PROFILE_SCHEMA, "lmcache_max_local_cpu_gb": "0.25", "points": []}))
    calls = []

    def fake_calibrate(_cfg, _run_root, sessions, mbps, namespace):
        calls.append((len(sessions), mbps, namespace))
        return {"schema": c.PROFILE_SCHEMA, "lmcache_max_local_cpu_gb": c.b.LMCACHE_MAX_LOCAL_CPU_GB, "mbps": mbps, "points": []}

    monkeypatch.setattr(c, "calibrate_live_profile", fake_calibrate)
    profile, used = c.ensure_live_profile(type("Cfg", (), {})(), tmp_path, path, manifest, 1000.0)

    assert used == path
    assert calls == [(2, 1000.0, str(path))]
    assert profile["lmcache_max_local_cpu_gb"] == "4"



def test_stored_session_kv_bytes_uses_largest_session_snapshot(tmp_path: Path):
    log = tmp_path / "source.log"
    log.write_text("\n".join([
        "Storing KV cache for 8192 out of 8192 tokens for request req-a",
        "Stored 5376 out of total 8192 tokens. size: 0.2461 gb, cost 1 ms",
        "Storing KV cache for 16384 out of 16384 tokens for request req-a",
        "Stored 5376 out of total 16384 tokens. size: 0.2461 gb, cost 1 ms",
        "Storing KV cache for 8192 out of 8192 tokens for request req-b",
        "Stored 5376 out of total 8192 tokens. size: 0.1111 gb, cost 1 ms",
    ]))

    assert c.stored_session_kv_bytes(log, "req-a") == 246_100_000



def test_lmcache_lookup_tokens_matches_exact_request(tmp_path: Path):
    log = tmp_path / "sink.log"
    log.write_text("\n".join([
        "Reqid: req-a, Total tokens 100, LMCache hit tokens: 96, need to load: 96",
        "Reqid: req-b, Total tokens 100, LMCache hit tokens: 0, need to load: 0",
    ]))

    assert c.lmcache_lookup_tokens(log, "req-a") == (100, 96)
    assert c.lmcache_lookup_tokens(log, "req-b") == (100, 0)
    assert c.lmcache_lookup_tokens(log, "req") is None


def test_write_proxy_slice_filters_to_scenario_windows(tmp_path: Path):
    src = tmp_path / "all_proxy.csv"
    with src.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["ts", "route", "direction", "bytes", "billed"])
        writer.writeheader()
        writer.writerows([
            {"ts": 1.0, "route": "kv", "direction": "target_to_client", "bytes": 1, "billed": 1},
            {"ts": 5.0, "route": "kv", "direction": "target_to_client", "bytes": 2, "billed": 1},
            {"ts": 9.0, "route": "kv", "direction": "target_to_client", "bytes": 3, "billed": 1},
        ])

    c.write_proxy_slice(src, tmp_path / "proxy_bytes.csv", {"drain": (4.0, 6.0)})

    with (tmp_path / "proxy_bytes.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["bytes"]) for r in rows] == [2]

def test_proxy_audit_reports_user_space_link_rate(tmp_path: Path):
    proxy = tmp_path / "proxy.csv"
    with proxy.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["ts", "route", "direction", "bytes", "billed"])
        writer.writeheader()
        writer.writerows([
            {"ts": 0.0, "route": "kv", "direction": "target_to_client", "bytes": 125_000_000, "billed": 1},
            {"ts": 2.0, "route": "kv", "direction": "target_to_client", "bytes": 125_000_000, "billed": 1},
            {"ts": 1.0, "route": "api", "direction": "client_to_target", "bytes": 1_000, "billed": 1},
        ])

    rows = c.write_proxy_audit(proxy, tmp_path / "proxy_audit.csv", {"drain": (0, 2)}, 1000.0)

    kv = next(r for r in rows if r["route"] == "kv")
    assert kv["bytes"] == 250_000_000
    assert kv["target_bytes_per_s"] == pytest.approx(125_000_000)
    assert kv["target_ratio"] == pytest.approx(1.0)
    assert kv["ok"] == 1
    assert (tmp_path / "proxy_audit.csv").exists()

def test_request_count_rows_groups_request_starts_by_phase_and_port(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join([
        json.dumps({"ts": 0.5, "kind": "request_start", "port": 8100}),
        json.dumps({"ts": 1.5, "kind": "request_start", "port": 8400}),
        json.dumps({"ts": 2.0, "kind": "request_end", "port": 8400}),
    ]) + "\n")

    rows = c.request_count_rows(path, {"baseline": (0, 1), "post": (1, 3)})

    assert rows == [
        {"phase": "baseline", "port": 8100, "requests": 1},
        {"phase": "post", "port": 8400, "requests": 1},
    ]


def test_grid_summary_row_aggregates_power_and_delay(tmp_path: Path):
    run = tmp_path / "r"
    run.mkdir()
    with (run / "power_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["phase", "gpu", "samples", "power_mean_w"])
        writer.writeheader()
        writer.writerows([
            {"phase": "baseline", "gpu": 0, "samples": 1, "power_mean_w": 300},
            {"phase": "baseline", "gpu": 1, "samples": 1, "power_mean_w": 80},
            {"phase": "post", "gpu": 0, "samples": 1, "power_mean_w": 100},
            {"phase": "post", "gpu": 1, "samples": 1, "power_mean_w": 220},
        ])
    (run / "controller_manifest.json").write_text(json.dumps({
        "schema": c.LIVE_SCHEMA,
        "policy": "lp",
        "deadline_s": 30,
        "target_frac": 0.45,
        "target_w": 1000,
        "full_source_drop_w": 2000,
        "planned_source_drop_w": 900,
        "planned_shortfall_w": 100,
        "planned_hit": False,
        "sessions": [{"first_token_s": 1.5, "completion_s": 2.5}],
    }))

    row = c.grid_summary_row(run)

    assert row["measured_source_drop_w"] == pytest.approx(200)
    assert row["measured_sink_rise_w"] == pytest.approx(140)
    assert row["total_completion_s"] == pytest.approx(2.5)
    assert row["planned_hit"] is False


def test_grid_summary_row_uses_wall_clock_parallel_delay(tmp_path: Path):
    run = tmp_path / "r"
    run.mkdir()
    with (run / "power_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["phase", "gpu", "samples", "power_mean_w"])
        writer.writeheader()
        writer.writerows([
            {"phase": "baseline", "gpu": 0, "samples": 1, "power_mean_w": 300},
            {"phase": "baseline", "gpu": 1, "samples": 1, "power_mean_w": 80},
            {"phase": "post", "gpu": 0, "samples": 1, "power_mean_w": 100},
            {"phase": "post", "gpu": 1, "samples": 1, "power_mean_w": 220},
        ])
    (run / "controller_manifest.json").write_text(json.dumps({
        "schema": c.LIVE_SCHEMA,
        "policy": "lp",
        "deadline_s": 30,
        "target_frac": 0.45,
        "target_w": 100,
        "full_source_drop_w": 200,
        "planned_source_drop_w": 100,
        "planned_shortfall_w": 0,
        "planned_hit": True,
        "acceptance": {"ok": False, "deadline_hit": True},
        "sessions": [
            {"first_token_s": 1.0, "completion_s": 5.0, "move_start_ts": 10.0, "move_end_ts": 15.0, "deadline_met": True},
            {"first_token_s": 1.0, "completion_s": 3.0, "move_start_ts": 11.0, "move_end_ts": 14.0, "deadline_met": True},
        ],
    }))

    row = c.grid_summary_row(run)

    assert row["total_completion_s"] == pytest.approx(5.0)
    assert row["total_first_token_s"] == pytest.approx(2.0)
    assert row["deadline_hit"] is True
    assert row["acceptance_ok"] is False


def test_grid_summary_deadline_hit_uses_wall_clock_completion(tmp_path: Path):
    run = tmp_path / "r"
    run.mkdir()
    with (run / "power_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["phase", "gpu", "samples", "power_mean_w"])
        writer.writeheader()
        writer.writerows([
            {"phase": "baseline", "gpu": 0, "samples": 1, "power_mean_w": 300},
            {"phase": "baseline", "gpu": 1, "samples": 1, "power_mean_w": 80},
            {"phase": "post", "gpu": 0, "samples": 1, "power_mean_w": 100},
            {"phase": "post", "gpu": 1, "samples": 1, "power_mean_w": 220},
        ])
    (run / "controller_manifest.json").write_text(json.dumps({
        "schema": c.LIVE_SCHEMA,
        "policy": "greedy",
        "deadline_s": 4,
        "target_frac": 0.45,
        "target_w": 100,
        "full_source_drop_w": 200,
        "planned_source_drop_w": 100,
        "planned_shortfall_w": 0,
        "planned_hit": True,
        "input_manifest": {"workload": {"name": "small"}, "sessions": [{"served_T": 1024}, {"served_T": 2048}]},
        "sessions": [
            {"first_token_s": 1.0, "completion_s": 3.0, "move_start_ts": 10.0, "move_end_ts": 13.0, "deadline_met": True},
            {"first_token_s": 1.0, "completion_s": 3.0, "move_start_ts": 13.0, "move_end_ts": 16.0, "deadline_met": True},
        ],
    }))

    row = c.grid_summary_row(run)

    assert row["workload"] == "small"
    assert row["input_sessions"] == 2
    assert row["median_served_T"] == pytest.approx(1536)
    assert row["total_completion_s"] == pytest.approx(6)
    assert row["deadline_hit"] is False


def test_write_grid_summary_writes_csv_and_plots(tmp_path: Path):
    run = tmp_path / "r"
    run.mkdir()
    with (run / "power_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["phase", "gpu", "samples", "power_mean_w"])
        writer.writeheader()
        writer.writerows([
            {"phase": "baseline", "gpu": 0, "samples": 1, "power_mean_w": 300},
            {"phase": "baseline", "gpu": 1, "samples": 1, "power_mean_w": 80},
            {"phase": "post", "gpu": 0, "samples": 1, "power_mean_w": 100},
            {"phase": "post", "gpu": 1, "samples": 1, "power_mean_w": 220},
        ])
    (run / "controller_manifest.json").write_text(json.dumps({
        "schema": c.LIVE_SCHEMA,
        "policy": "greedy",
        "deadline_s": 120,
        "target_frac": 0.65,
        "target_w": 1000,
        "full_source_drop_w": 2000,
        "planned_source_drop_w": 1000,
        "planned_shortfall_w": 0,
        "planned_hit": True,
        "sessions": [{"first_token_s": 1.0, "completion_s": 2.0}],
    }))

    rows = c.write_grid_summary([run], tmp_path / "scenario_summary.csv", tmp_path / "grid_power_drop.png", tmp_path / "grid_delay.png")

    assert rows[0]["policy"] == "greedy"
    assert (tmp_path / "scenario_summary.csv").exists()
    assert (tmp_path / "grid_power_drop.png").exists()
    assert (tmp_path / "grid_delay.png").exists()


def test_tracelab_manifest_skips_bad_token_rows(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    rows = [
        {"session_id": "s", "timestamp": 0, "input_tokens_total": 4096, "prefix_tokens": 0, "newly_append_tokens": 100, "output_tokens": 10},
        {"session_id": "s", "timestamp": 1, "input_tokens_total": 0, "prefix_tokens": 0, "newly_append_tokens": 100, "output_tokens": 10},
        {"session_id": "s", "timestamp": 2, "input_tokens_total": 4200, "prefix_tokens": 4096, "newly_append_tokens": 100, "output_tokens": 10},
        {"session_id": "s", "timestamp": 3, "input_tokens_total": 4300, "prefix_tokens": 4200, "newly_append_tokens": 100, "output_tokens": 10},
    ]
    with gzip.open(trace, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    manifest = c.tracelab_manifest(trace, 1, 0, max_model_len=8192, min_context_tokens=1024)

    assert len(manifest["sessions"][0]["turns"]) == 3
