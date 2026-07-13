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
    assert manifest["source"]["type"] == "tracelab"
    assert session["original_T"] > session["served_T"]
    assert session["served_T"] == 3840
    assert session["turn_rate_hz"] == pytest.approx(2 / 120)
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


def test_live_plan_uses_log_power_curve_and_dispatches_all(tmp_path: Path):
    trace = tmp_path / "trace.jsonl.gz"
    _write_tracelab(trace)
    manifest = c.tracelab_manifest(trace, 2, 0, max_model_len=4096, decode_margin=256, min_context_tokens=1024)
    manifest["deadline_s"] = 1e9
    manifest["constants"]["lambda_src_bytes_per_s"] = 1e18
    manifest["constants"]["mu_bytes_per_s"] = 1e18

    summary = c.live_plan_summary(manifest)

    assert summary["power_curve"]["name"] == "log"
    assert summary["power_curve"]["p_idle_w"] == pytest.approx(c.LIVE_A100_P_IDLE_W)
    assert summary["power_curve"]["p_busy_w"] == pytest.approx(c.LIVE_A100_P_BUSY_W)
    assert summary["full_source_drop_w"] <= c.LIVE_A100_P_BUSY_W - c.LIVE_A100_P_IDLE_W + 1e-6
    assert len(summary["sessions"]) == 2
    assert [s["dispatch_rank"] for s in summary["sessions"]] == [0, 1]
    assert summary["sessions"][-1]["predicted_cumulative_source_drop_w"] == pytest.approx(summary["full_source_drop_w"])


def test_run_live_moves_warms_sink_with_bounded_replay_and_overlapping_kv(tmp_path: Path, monkeypatch):
    class Worker:
        def __init__(self, sid):
            self.last_prompt = f"prompt-{sid}"
            self.pause_times = []
            self.resumed = False
            self.port = None
            self.cache_busted = False

        def pause_boundary(self):
            self.pause_times.append(time.time())

        def switch_to(self, port):
            self.port = port

        def resume(self):
            self.resumed = True

        def cache_bust_on_sink(self):
            self.cache_busted = True

    cfg = type("Cfg", (), {"api_proxy_port": 8400})()
    workers = {sid: Worker(sid) for sid in ("a", "b", "c")}
    sessions = [{"decode_tokens": 1}, {"decode_tokens": 1}, {"decode_tokens": 1}]
    rows = [
        {"id": "a", "action": "R", "fixture_index": 0, "dispatch_rank": 0, "deadline_s": 2.0, "cache_bust_after_switch": True},
        {"id": "b", "action": "R", "fixture_index": 1, "dispatch_rank": 1, "deadline_s": 2.0},
        {"id": "c", "action": "S", "fixture_index": 2, "dispatch_rank": 2, "deadline_s": 2.0},
    ]
    starts, ends = {"R": [], "S": []}, {}

    def fake_stream_chat(_cfg, _port, prompt, _max_tokens):
        action = "R" if prompt.startswith("Replay cache bust") else "S"
        sid = prompt.split()[3] if action == "R" else prompt.rsplit("-", 1)[1]
        start = time.time()
        starts[action].append(start)
        time.sleep(0.2)
        end = time.time()
        ends[sid] = end
        return {"status": 200, "first_token_ts": start + 0.01, "start_ts": start, "end_ts": end, "prompt_sha256": "x", "request_id": f"req-{sid}", "response_text": ""}

    monkeypatch.setattr(c, "stream_chat", fake_stream_chat)
    t0 = time.time()
    out = c.run_live_moves(cfg, tmp_path, sessions, workers, rows, settle_s=0.0, replay_concurrency=1, kv_concurrency=2)

    assert time.time() - t0 < 0.55
    assert starts["R"][1] - starts["R"][0] >= 0.18
    assert starts["S"][0] - starts["R"][0] < 0.1
    assert [r["dispatch_rank"] for r in out] == [0, 1, 2]
    assert all(workers[sid].pause_times[0] >= ends[sid] for sid in workers)
    assert all(w.resumed and w.port == 8400 for w in workers.values())
    assert workers["a"].cache_busted and not workers["b"].cache_busted
    assert all(r["warm_move"] and r["deadline_met"] and r["switch_downtime_s"] >= 0 for r in out)
    assert all(r["sink_warm_s"] <= r["completion_s"] for r in out)


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

    assert rows == [{"phase": "baseline", "gpu": 0, "samples": 2, "power_mean_w": 200.0}]


def test_check_live_manifest_requires_files_and_route_evidence(tmp_path: Path):
    for name in c.LIVE_ARTIFACTS:
        (tmp_path / name).write_text("x")
    manifest = {
        "schema": c.LIVE_SCHEMA,
        "sessions": [
            {"id": "r", "action": "R", "dispatch_rank": 0, "http_status": 200, "first_token_s": 0.1, "proxy_delta": {"api/client_to_target": 10}},
            {"id": "s", "action": "S", "dispatch_rank": 1, "http_status": 200, "first_token_s": 0.2, "proxy_delta": {"kv/target_to_client": 10}},
        ],
    }

    c.check_live_manifest(manifest, tmp_path)
    manifest["sessions"][1]["proxy_delta"] = {}
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
    assert (tmp_path / "delay_summary.csv").read_text().splitlines()[0] == "dispatch_rank,id,action,first_token_s,sink_warm_s,completion_s,source_boundary_wait_s,switch_downtime_s"
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

    summaries = [c.live_plan_summary(manifest, policy=p, deadline_s=30, target_frac=0.25, seed=3) for p in ("lp", "random", "greedy")]

    assert {s["policy"] for s in summaries} == {"lp", "random", "greedy"}
    for summary in summaries:
        assert summary["deadline_s"] == 30
        assert summary["target_frac"] == pytest.approx(0.25)
        assert summary["target_w"] == pytest.approx(0.25 * summary["full_source_drop_w"])
        assert summary["sessions"]
        assert [r["dispatch_rank"] for r in summary["sessions"]] == list(range(len(summary["sessions"])))
    greedy = next(s for s in summaries if s["policy"] == "greedy")
    assert greedy["solver"]["method"] == "node_aware_greedy"


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




def test_live_grid_multi_manifest_uses_workload_dirs_profiles_and_one_stack_each(tmp_path: Path, monkeypatch):
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

    assert stacks == [tmp_path / "small" / "stack", tmp_path / "large" / "stack"]
    assert stopped == stacks
    assert calls == [
        (tmp_path / "small" / "greedy_D30_T0p45", "small", "greedy", 30.0, 0.45, tmp_path / "profile_small.json", tmp_path / "small" / "stack"),
        (tmp_path / "large" / "greedy_D30_T0p45", "large", "greedy", 30.0, 0.45, tmp_path / "profile_large.json", tmp_path / "large" / "stack"),
    ]


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
    assert sessions[0]["profile_transfer_bytes"] > manifest["sessions"][0]["session_kv_bytes"]
    assert sessions[0]["c_transfer_s"] > 10.0
    assert sessions[1]["c_transfer_s"] == pytest.approx(20.0)
    assert summary["profile"]["schema"] == c.PROFILE_SCHEMA


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
    assert sessions[0]["profile_transfer_bytes"] == pytest.approx(10.0 * sessions[0]["served_T"])
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


def test_live_profile_recalibrates_on_lmcache_runtime_change(tmp_path: Path, monkeypatch):
    manifest = _manifest_for_live_policy_tests(tmp_path)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema": c.PROFILE_SCHEMA, "lmcache_max_local_cpu_gb": "0.25", "points": []}))
    calls = []

    def fake_calibrate(_cfg, _run_root, sessions, mbps):
        calls.append((len(sessions), mbps))
        return {"schema": c.PROFILE_SCHEMA, "lmcache_max_local_cpu_gb": c.b.LMCACHE_MAX_LOCAL_CPU_GB, "mbps": mbps, "points": []}

    monkeypatch.setattr(c, "calibrate_live_profile", fake_calibrate)
    profile, used = c.ensure_live_profile(type("Cfg", (), {})(), tmp_path, path, manifest, 1000.0)

    assert used == path
    assert calls == [(2, 1000.0)]
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
        "sessions": [
            {"first_token_s": 1.0, "completion_s": 5.0, "move_start_ts": 10.0, "move_end_ts": 15.0, "deadline_met": True},
            {"first_token_s": 1.0, "completion_s": 3.0, "move_start_ts": 11.0, "move_end_ts": 14.0, "deadline_met": True},
        ],
    }))

    row = c.grid_summary_row(run)

    assert row["total_completion_s"] == pytest.approx(5.0)
    assert row["total_first_token_s"] == pytest.approx(2.0)
    assert row["deadline_hit"] is True


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
