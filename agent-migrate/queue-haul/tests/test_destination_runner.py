"""
Claim:
Destination load uses scheduled work, exact session identity, and conservative
normal/emergency/stability classifications.

Plausible wrong implementations:
- Use achieved completions instead of offered tokens.
- Accept requested destination load when the measured load missed it.
- Reuse one prefix across nominally distinct sessions.
- Accept a status-200 prewarm with missing prompt or completion work.
- Select forced tokens outside the empirically working token range.
- Omit the cache-block contract from the generated compatibility identity.
- Treat an SLO boundary as infeasible or a just-outside value as feasible.
- Declare a growing destination queue stable.
"""

import pytest
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import destination_campaign as campaign
import destination_runner as runner


def request(ttft=.1, tpot=.01, error=""):
    return {"status": 200, "error": error, "output_tokens": 2,
            "planned_output_tokens": 2, "ttft_s": ttft, "mean_tpot_s": tpot,
            "input_tokens": 10}


def metrics(slope=0):
    return [{"monotonic_ns": i * 10**9,
             "vllm:num_requests_waiting": 1 + slope * i} for i in range(100)]


def test_schedule_and_session_tokens_are_deterministic_but_isolated():
    assert runner.poisson_schedule(2, 4, 7) == runner.poisson_schedule(2, 4, 7)
    window = runner.poisson_window(2, 4, 7)
    assert window == runner.poisson_window(2, 4, 7) and all(t <= 4 for t in window)
    assert runner.uniform_schedule(2, 4, 7) == (0, .5, 1, 1.5)
    assert runner.anchor_rate(100, 20) == 5
    a = runner.Session("a", 4, 2, 3, 100, 7)
    b = runner.Session("b", 4, 2, 3, 100, 7)
    first, _ = a.prompt(0)
    assert a.prompt(1)[0][:4] == first[:4]
    assert len(a.prompt(1)[0]) == 6
    assert b.prompt(0)[0][:4] != first[:4]


def test_session_forced_tokens_use_the_observed_safe_range(monkeypatch):
    vocabularies = []

    def tokens(label, count, vocabulary, seed):
        vocabularies.append((label, vocabulary))
        return [16] * count

    monkeypatch.setattr(runner, "deterministic_tokens", tokens)
    runner.Session("s", 4, 2, 3, 201088, 0).prompt(0)

    assert vocabularies[-1] == ("s:0:output", 200000)


def test_prewarm_rejects_status_200_without_token_work(monkeypatch):
    monkeypatch.setattr(runner, "_completion", lambda *_: {
        "status": 200, "error": "", "prompt_tokens": 0, "output_tokens": 0,
        "done": True,
    })

    with pytest.raises(RuntimeError, match="failed to prewarm"):
        runner.prewarm("h", 1, "m", [runner.Session("s", 4, 2, 3, 100, 0)])


def test_agentic_chat_turn_is_unique_and_compute_heavy():
    session = runner.Session("s", 4, 2048, 32, 100, 0)

    first = runner.agentic_messages(session, 1)
    second = runner.agentic_messages(session, 2)

    assert first != second and first != runner.agentic_messages(session, 33)
    assert first[0]["role"] == "system"
    assert first[1]["content"].count(" x") == 2048


def test_service_reset_clears_remote_and_local_caches(monkeypatch, tmp_path):
    calls = []
    stack = SimpleNamespace(run_root=tmp_path)
    cfg = SimpleNamespace()
    monkeypatch.setattr(runner.testbed, "flush_lmcache",
                        lambda actual_stack, actual_cfg: calls.append("remote"))
    monkeypatch.setattr(runner.testbed, "reset_vllm_caches",
                        lambda actual_cfg, logs: calls.append(logs))

    runner.reset_service_cache(stack, cfg)

    assert calls == [
        "remote", (tmp_path / "source.log", tmp_path / "sink.log"),
    ]


def test_offered_work_does_not_depend_on_completion():
    rows = [request(), request(error="failed")]
    assert runner.offered_work(rows, 2) == (10, 2)


def test_slo_boundary_is_inclusive_and_queue_growth_is_not_stable():
    slos = {"normal": {"p90_ttft_s": 2, "p90_mean_tpot_s": .1},
            "emergency": {"p90_ttft_s": 10, "p90_mean_tpot_s": .25}}
    exact = runner.classify([request(2, .1)], metrics(), True, slos)
    outside = runner.classify([request(2.001, .1)], metrics(), True, slos)
    growing = runner.classify([request()], metrics(.1), True, slos)
    assert exact == {"normal": True, "emergency": True, "stable": True}
    assert not outside["normal"] and outside["emergency"]
    assert not growing["stable"]


def test_queue_drift_requires_real_samples():
    with pytest.raises(ValueError, match="sampled"):
        runner.queue_drift_upper(metrics()[:1])


def test_client_side_backlog_is_not_classified_stable():
    requests = [{**request(), "scheduled_ns": i * 10**9, "start_ns": 2 * i * 10**9}
                for i in range(1, 100)]
    assert not runner.classify(requests, metrics(), True, {})["stable"]


def test_anchor_gate_accepts_improvement_and_fifteen_percent_underdelivery():
    expected = {("prefill", 4096): 100, ("decode", 4096): 50}
    assert runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": 85},
        {"metric": "decode", "context_tokens": 4096, "tokens_per_s": 57.5},
    ], expected)["within_limit"]
    assert runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": 200},
        {"metric": "decode", "context_tokens": 4096, "tokens_per_s": 100},
    ], expected)["within_limit"]
    report = runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": 84.9},
        {"metric": "decode", "context_tokens": 4096, "tokens_per_s": 50},
    ], expected)
    assert not report["within_limit"]
    with pytest.raises(ValueError, match="incomplete"):
        runner.anchor_gate([{"metric": "prefill", "context_tokens": 4096,
                             "tokens_per_s": 100}], expected)


def test_anchor_gate_uses_independent_run_median_not_last_request():
    assert runner.anchor_gate([
        {"metric": "prefill", "context_tokens": 4096, "tokens_per_s": value}
        for value in (100, 100, 1)
    ], {("prefill", 4096): 100})["within_limit"]


def test_anchor_mismatch_recalibrates_central_profile():
    profile = {"cases": {"central": {"prefill_tps": {"1": [[1, 10], [20, 30]]}}}}
    report = {"anchors": [{"metric": "prefill", "context_tokens": 10,
                            "observed_tokens_per_s": 25}]}
    runner.apply_anchor_rates(profile, report)
    assert profile["cases"]["central"]["prefill_tps"]["1"] == [[1, 10], [10, 25], [20, 30]]


def test_profile_rate_interpolates_only_inside_measured_domain():
    profile = {"cases": {"central": {"prefill_tps": {"1": [[10, 100], [20, 200]]}}}}
    assert runner.profile_rate(profile, "prefill", 15) == 150
    with pytest.raises(ValueError, match="outside"):
        runner.profile_rate(profile, "prefill", 9)


def test_repaired_baseline_passes_its_independent_anchor_gate():
    root = Path(runner.__file__).parent
    profile = json.loads((root / "profiles/gpt_oss_20b_a100_tp1.json").read_text())
    rows = json.loads((root / "outputs/destination-anchor-baseline-20260722.json").read_text())["anchors"]
    expected = {(metric, context): runner.profile_rate(profile, metric, context)
                for metric in ("prefill", "decode") for context in (4096, 16384, 24576)}
    assert runner.anchor_gate(rows, expected)["within_limit"]


def test_integrity_preflight_requires_same_but_not_cross_session_cache(monkeypatch):
    cached = iter((0, 4000, 0))
    monkeypatch.setattr(runner, "_completion", lambda *_: {"cached_tokens": next(cached)})
    monkeypatch.setattr(runner.testbed, "http_json", lambda *_: {"count": 2})
    report = runner.integrity_preflight(SimpleNamespace(
        host="h", sink_port=1, model="m"),
        {"image_sha256": campaign.IMAGE_SHA256}, {"acceptance": {"ok": True}}, 100)
    assert report["same_session_cache_hit"] and report["cross_session_cache_hits"] == 0


def test_runtime_identity_changes_with_cache_block_contract(monkeypatch, tmp_path):
    reference = tmp_path / "hub/models--m/refs/main"
    reference.parent.mkdir(parents=True)
    reference.write_text("revision")
    cfg = SimpleNamespace(hf_home=tmp_path, model="m", max_model_len=10,
                          max_num_seqs=2, max_num_batched_tokens=4)
    plan = {"image_sha256": "image", "service": {
        "directions": ["coding"], "cache_block_tokens": 16,
    }}
    monkeypatch.setattr(runner, "manifest_sessions",
                        lambda *_: [runner.Session("s", 4, 2, 3, 100, 0)])
    monkeypatch.setattr(runner, "profile_rate", lambda *_: 10)
    monkeypatch.setattr(runner.testbed, "lmcache_mode", lambda: "mp")

    first = runner.runtime_identity(cfg, plan, {}, {"kv_capacity_tokens": 1}, "p")
    plan["service"]["cache_block_tokens"] = 32
    second = runner.runtime_identity(cfg, plan, {}, {"kv_capacity_tokens": 1}, "p")

    assert first["compatibility"]["kv_abi"] != second["compatibility"]["kv_abi"]


def test_launch_inputs_are_relative_and_checksum_pinned(tmp_path):
    splits = {job: {split: [f"{job}-{split}-{i}" for i in range(n)]
                    for split, n in (("fit", 12), ("tune", 6), ("validation", 6))}
              for job in campaign.JOB_CLASSES}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest": {"schema": campaign.MANIFEST_SCHEMA,
                                                  "splits": splits},
                                    "traces": [{"session_id": sid, "input_tokens_total": 256}
                                               for values in splits.values()
                                               for ids in values.values() for sid in ids]}))
    bundle = tmp_path / "bundle"; campaign.prepare(manifest, bundle)
    plan, loaded, profile = runner.load_inputs(bundle / "plan.json")
    assert plan["schema"] == campaign.SCHEMA and loaded["manifest"]["splits"] == splits
    assert profile["model"] == "openai/gpt-oss-20b"
    (bundle / "baseline-profile.json").write_text("{}")
    with pytest.raises(RuntimeError, match="changed"):
        runner.load_inputs(bundle / "plan.json")


def test_run_root_requires_explicit_commit_resume(tmp_path):
    path = tmp_path / "run.json"
    runner.write_run_metadata(path, {"git_sha": "one"})
    runner.write_run_metadata(path, {"git_sha": "one"})
    with pytest.raises(RuntimeError, match="different campaign or commit"):
        runner.write_run_metadata(path, {"git_sha": "two"})
    runner.write_run_metadata(path, {"git_sha": "two"}, "one")
    assert json.loads(path.read_text())["git_history"] == ["one", "two"]
    runner.write_run_metadata(path, {"git_sha": "two"})


def test_resume_commit_defaults_from_environment(monkeypatch):
    monkeypatch.setenv("QH_RESUME_FROM_GIT_SHA", "one")
    assert runner.parse_args(["--plan", "p", "--run-root", "r"]).resume_from_git_sha == "one"


def test_loaded_scenario_has_one_session_and_one_method():
    session = {"id": "s", "job_class": "coding", "state_code": "CODE",
               "turns": [{"input_tokens": 1024, "append_tokens": 1,
                           "output_tokens": 1}]}
    manifest = {"schema": runner.profiler.MANIFEST_SCHEMA,
                "workload": "coding", "sessions": [session]}
    scenario = runner.migration_scenario(session,
                                         "replay", 16384, 10000, 2)
    assert {"scenario_id", "kind", "method", "activity", "request_schedule",
            "repeat", "deadline_s", "sessions", "moves", "serving_concurrency",
            "concurrency", "move_concurrency", "copy_policy", "final_state",
            "bandwidth_mbps"} <= scenario.keys()
    assert scenario["concurrency"] == scenario["move_concurrency"] == 1
    assert scenario["activity"] == "none" and scenario["request_schedule"] == []
    assert scenario["moves"] == [{**scenario["sessions"][0], "method": "replay"}]
    runner.validate_loaded_scenario(manifest, scenario)
    del scenario["activity"]
    with pytest.raises(ValueError, match="invalid loaded"):
        runner.validate_loaded_scenario(manifest, scenario)


def test_loaded_stack_preserves_proxy_until_cell_finishes(monkeypatch, tmp_path):
    cfg, stack, events = object(), object(), []
    monkeypatch.setattr(runner.testbed, "start_stack",
                        lambda *args: events.append(("start", args)) or stack)
    monkeypatch.setattr(runner.testbed, "start_sink",
                        lambda *args: events.append(("sink", args)))
    monkeypatch.setattr(runner.testbed, "stop_stack",
                        lambda *args: events.append(("stop", args)))
    with pytest.raises(RuntimeError, match="body"):
        with runner.loaded_stack(cfg, tmp_path, 10000, ["x"]) as actual:
            assert actual is stack
            raise RuntimeError("body")
    assert events == [
        ("start", (cfg, tmp_path, 10000, ["x"])),
        ("sink", (stack, cfg, ["x"])),
        ("stop", (stack,)),
    ]


def test_loaded_stack_stops_when_sink_start_fails(monkeypatch, tmp_path):
    stack, stopped = object(), []
    monkeypatch.setattr(runner.testbed, "start_stack", lambda *_: stack)
    monkeypatch.setattr(runner.testbed, "start_sink",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("sink")))
    monkeypatch.setattr(runner.testbed, "stop_stack", stopped.append)
    with pytest.raises(RuntimeError, match="sink"):
        with runner.loaded_stack(object(), tmp_path, 10000, []):
            pass
    assert stopped == [stack]


def loaded_inputs(monkeypatch):
    monkeypatch.setattr(runner, "migration_manifest", lambda _: {
        "schema": runner.profiler.MANIFEST_SCHEMA, "workload": "coding",
        "sessions": [{"id": "s", "job_class": "coding", "state_code": "CODE",
                      "turns": [{"input_tokens": 1024, "append_tokens": 1,
                                  "output_tokens": 1}]}],
    })
    monkeypatch.setattr(runner, "manifest_sessions", lambda *_: [])
    monkeypatch.setattr(runner, "profile_rate", lambda *_: 1)
    plan = {
        "service": {"directions": []},
        "migration": {"rho": [.8, "emergency_inside"],
                      "emergency_inside_fraction": .5,
                      "context_tokens": 16384, "bandwidth_gbps": 10,
                      "heldout_context_tokens": 24576,
                      "heldout_bandwidth_gbps": 5, "repeats": 1,
                      "methods": ["replay", "kv_transfer"]},
    }
    return plan, {"normal": 1, "emergency": 2}


def checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "complete", **payload}))


def test_loaded_rehearsal_uses_one_isolated_stack_per_incomplete_method(
        monkeypatch, tmp_path):
    plan, bounds = loaded_inputs(monkeypatch)
    base = tmp_path / "rho0.800000-t16384-b10000-r0"
    checkpoint(base / "control/result.json",
               {"destination_load": {"achieved_rho": .8}})
    stacks, calls = [], []

    @contextmanager
    def stack(_cfg, root, bandwidth, extra):
        stacks.append((root, bandwidth, extra))
        yield SimpleNamespace(run_root=root, bandwidth_mbps=bandwidth)

    def run(actual, _cfg, _manifest, scenario, root, _run_id, load,
            configure_proxy=True):
        calls.append((actual, scenario, root, load, configure_proxy))
        checkpoint(root / "result.json", {
            "migrations": [{}], "destination_load": {"achieved_rho": .8},
        })

    monkeypatch.setattr(runner, "loaded_stack", stack)
    monkeypatch.setattr(runner.profiler, "run_scenario", run)
    monkeypatch.setattr(runner, "DestinationLoad", lambda *_args, **_kwargs: object())
    cfg = SimpleNamespace(host="h", sink_port=1, model="m")
    rows = runner.measure_loaded(plan, {}, {}, cfg, bounds, tmp_path, ["x"],
                                 rehearsal=True)

    assert [root.name for root, _, _ in stacks] == ["replay", "kv_transfer"]
    assert all(bandwidth == 10000 and extra == ["x"]
               for _, bandwidth, extra in stacks)
    assert len(rows) == len(calls) == 2
    assert all(stack.run_root == root and stack.bandwidth_mbps == 10000
               and not configure for stack, _, root, _, configure in calls)


def test_unloaded_calibration_does_not_start_destination_traffic(
        monkeypatch, tmp_path):
    plan, bounds = loaded_inputs(monkeypatch)
    plan["migration"]["rho"] = [0]

    @contextmanager
    def stack(_cfg, root, _bandwidth, _extra):
        yield SimpleNamespace(run_root=root)

    def run(_stack, _cfg, _manifest, _scenario, root, _run_id, load,
            configure_proxy=True):
        assert load is None and not configure_proxy
        checkpoint(root / "result.json", {
            "migrations": [{}], "destination_load": None,
        })

    monkeypatch.setattr(runner, "loaded_stack", stack)
    monkeypatch.setattr(runner.profiler, "run_scenario", run)
    monkeypatch.setattr(
        runner, "DestinationLoad",
        lambda *_args, **_kwargs: pytest.fail("unloaded calibration started traffic"),
    )
    rows = runner.measure_loaded(
        plan, {}, {}, SimpleNamespace(host="h", sink_port=1, model="m"),
        bounds, tmp_path, [], rehearsal=True,
    )
    assert {row["rho"] for row in rows} == {0}
    assert not list(tmp_path.glob("rho*/control/result.json"))


def test_loaded_rehearsal_skips_all_complete_checkpoints(monkeypatch, tmp_path):
    plan, bounds = loaded_inputs(monkeypatch)
    base = tmp_path / "rho0.800000-t16384-b10000-r0"
    checkpoint(base / "control/result.json",
               {"destination_load": {"achieved_rho": .8}})
    for method in plan["migration"]["methods"]:
        checkpoint(base / method / "result.json", {
            "migrations": [{}], "destination_load": {"achieved_rho": .8},
        })
    monkeypatch.setattr(runner, "loaded_stack",
                        lambda *_: (_ for _ in ()).throw(AssertionError("stack")))
    rows = runner.measure_loaded(
        plan, {}, {}, SimpleNamespace(), bounds, tmp_path, [], rehearsal=True)
    assert len(rows) == 2


def test_loaded_control_uses_its_own_stack(monkeypatch, tmp_path):
    plan, bounds = loaded_inputs(monkeypatch)
    base = tmp_path / "rho0.800000-t16384-b10000-r0"
    for method in plan["migration"]["methods"]:
        checkpoint(base / method / "result.json", {
            "migrations": [{}], "destination_load": {"achieved_rho": .8},
        })
    used, events = [], []

    @contextmanager
    def stack(_cfg, root, bandwidth, extra):
        used.append((root, bandwidth, extra))
        root.mkdir(parents=True)
        yield object()

    class Load:
        def __init__(self, *_args, **_kwargs): pass
        def start(self): events.append("start")
        def wait_ready(self): events.append("ready")
        def close(self): events.append("close")
        def summary(self): return {"achieved_rho": .8}

    monkeypatch.setattr(runner, "loaded_stack", stack)
    monkeypatch.setattr(runner, "DestinationLoad", Load)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    runner.measure_loaded(plan, {}, {}, SimpleNamespace(host="h", sink_port=1, model="m"), bounds, tmp_path,
                          ["x"], rehearsal=True)
    assert used == [(base / "control/testbed", 10000, ["x"])]
    assert events == ["start", "ready", "close"]


def test_adaptive_search_brackets_each_nested_boundary():
    boundaries = {"normal": 1, "emergency": 2, "stable": 3}
    found = runner.find_boundaries(
        lambda radius: {mode: radius <= bound for mode, bound in boundaries.items()}
    )
    for mode, bound in boundaries.items():
        assert found[mode][0] <= bound <= found[mode][1]
        assert found[mode][1] - found[mode][0] <= .05 * found[mode][1]


def test_adaptive_search_censors_below_minimum_without_running_zero():
    calls = []
    bounds = runner.find_boundaries(lambda radius: calls.append(radius) or
                                    {mode: False for mode in runner.MODES})
    assert 0 not in calls
    assert set(bounds.values()) == {(.025, .025)}


def test_adaptive_search_censors_above_maximum():
    assert set(runner.find_boundaries(
        lambda radius: {mode: True for mode in runner.MODES}
    ).values()) == {(4, 4)}


def test_nest_bounds_only_shrinks_inverted_envelopes():
    assert runner.nest_bounds({"normal": 3, "emergency": 2, "stable": 1}) == {
        "normal": 1, "emergency": 1, "stable": 1,
    }


def test_destination_load_rejects_rho_miss(monkeypatch):
    load = runner.DestinationLoad.__new__(runner.DestinationLoad)
    load.failure = None
    load.sampler = SimpleNamespace(rows=[{"monotonic_ns": 0}, {"monotonic_ns": 30e9}])
    load.target = load.prefill_rate = load.decode_rate = load.normal_bound = 1
    monkeypatch.setattr(runner, "measured_rho", lambda *args: .4)
    clock = iter((0, 0, 91))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="misses target"):
        load.wait_ready()
    assert load.achieved == .4


def test_loaded_reduction_rejects_target_miss(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "status": "complete",
        "destination_load": {"achieved_rho": .4},
        "migrations": [{"initial_start_ns": 0, "switch_end_ns": 1_000_000_000}],
    }))
    with pytest.raises(RuntimeError, match="misses target"):
        runner.reduce_loaded_results({}, [{
            "path": result, "rho": .8, "method": "replay", "repeat": 0,
            "context_tokens": 16384, "bandwidth_mbps": 10000,
        }], tmp_path)


def test_loaded_reduction_separates_runtime_baseline_from_load(tmp_path, monkeypatch):
    index = []
    for method in ("replay", "kv_transfer"):
        for rho, observed in ((0, 5), (.8, 7.5)):
            result = tmp_path / f"{method}-{rho}.json"
            result.write_text(json.dumps({
                "status": "complete",
                "destination_load": None if not rho else {"achieved_rho": rho},
                "migrations": [{
                    "initial_start_ns": 0,
                    "switch_end_ns": int(observed * 1e9),
                }],
            }))
            index.append({
                "path": result, "rho": rho, "method": method, "repeat": 0,
                "context_tokens": 16384, "bandwidth_mbps": 10000,
            })
    monkeypatch.setattr(runner, "unloaded_duration", lambda *_: 10)
    rows, validation = runner.reduce_loaded_results({}, index, tmp_path)
    by_cell = {(r["method"], r["rho"]): r["duration_factor"] for r in rows}
    assert by_cell == {
        ("replay", 0): .5, ("replay", .8): .75,
        ("kv_transfer", 0): .5, ("kv_transfer", .8): .75,
    }
    assert validation == []


def test_drive_stop_cancels_scheduled_requests(monkeypatch):
    scheduled, stop, result = threading.Event(), threading.Event(), []
    def scheduler(*_):
        scheduled.set()
        return (60,)
    monkeypatch.setattr(runner, "_completion",
                        lambda *_: pytest.fail("cancelled request was launched"))
    thread = threading.Thread(target=lambda: result.extend(runner.drive(
        "h", 1, "m", [runner.Session("s", 1, 1, 1, 100, 0)], 1, 1, 0,
        scheduler=scheduler, stop=stop)))
    thread.start(); assert scheduled.wait(1); stop.set(); thread.join(1)
    assert not thread.is_alive() and result == []


def test_destination_load_close_uses_request_timeout(tmp_path):
    joined = []
    load = runner.DestinationLoad.__new__(runner.DestinationLoad)
    load.stop, load.failure, load.rows = threading.Event(), None, []
    load.chunk_s, load.timeout_s, load.root = 15, 720, tmp_path
    load.thread = SimpleNamespace(join=joined.append, is_alive=lambda: False, ident=1)
    load.sampler = SimpleNamespace(close=lambda: None)
    load.close()
    assert joined == [745]


def test_destination_load_close_hard_fails_if_request_outlives_timeout(tmp_path):
    load = runner.DestinationLoad.__new__(runner.DestinationLoad)
    load.stop, load.failure, load.rows = threading.Event(), None, []
    load.chunk_s, load.timeout_s, load.root = 0, 0, tmp_path
    load.thread = SimpleNamespace(join=lambda _: None, is_alive=lambda: True, ident=1)
    load.sampler = SimpleNamespace(close=lambda: None)
    with pytest.raises(RuntimeError, match="foreground failed"):
        load.close()


def test_retry_call_records_and_recovers(tmp_path):
    calls = []
    def action():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("cold start")
        return "ok"
    assert runner.retry_call(action, tmp_path / "retries.jsonl", 3, 0) == "ok"
    assert len((tmp_path / "retries.jsonl").read_text().splitlines()) == 2


def test_invalid_checkpoint_is_archived(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"status":"complete"}')
    assert runner.read_checkpoint(path, ("classification",)) is None and not path.exists()
    assert len(list(tmp_path.glob("result.invalid-*.json"))) == 1
    path.write_text('{"status":"complete","migrations":[]}')
    assert runner.read_checkpoint(path, ("migrations",),
                                  lambda row: len(row["migrations"]) == 1) is None


def test_incomplete_anchor_checkpoint_is_archived(tmp_path):
    path = tmp_path / "anchors.json"
    path.write_text('[{"metric":"prefill","context_tokens":1,"run_id":0}]')
    assert runner.read_anchor_checkpoint(path, {("prefill", 1): 1}, 3) is None
    assert len(list(tmp_path.glob("anchors.invalid-*.json"))) == 1


def test_frontier_searches_once_then_repeats_only_boundary_cells(monkeypatch, tmp_path):
    calls = []
    thresholds = {"normal": 1, "emergency": 2, "stable": 3}
    monkeypatch.setattr(runner, "manifest_sessions", lambda *_: [runner.Session("s", 10, 2, 3, 100, 0)])
    monkeypatch.setattr(runner, "profile_rate", lambda *_: 100)
    monkeypatch.setattr(runner, "reset_service_cache", lambda *args: None)
    def probe(*args, **kwargs):
        radius = args[4]; calls.append(radius)
        return {"classification": {mode: radius <= value for mode, value in thresholds.items()}}
    monkeypatch.setattr(runner, "service_probe", probe)
    plan = {"service": {"directions": ["coding"], "initial_repeats": 3,
                        "disagreement_repeats": 5, "radial_resolution": .05,
                        "hold_min_s": 1, "block_bootstrap_s": 30,
                        "bootstrap_samples": 10, "cache_block_tokens": 16,
                        "slos": {}}}
    rows, bounds = runner.measure_frontier(plan, {}, {}, SimpleNamespace(
        host="h", sink_port=1, model="m"), object(), tmp_path)
    assert len(rows) == 9 and all(sum(r["mode"] == mode for r in rows) == 3 for mode in thresholds)
    assert bounds["normal"] <= bounds["emergency"] <= bounds["stable"]
    assert (tmp_path / "validation.jsonl").is_file() and len(calls) < 60


def test_frontier_reruns_only_disagreements_and_accepts_four_of_five(monkeypatch, tmp_path):
    calls, thresholds = [], {"normal": 1, "emergency": 2, "stable": 3}
    monkeypatch.setattr(runner, "manifest_sessions", lambda *_: [runner.Session("s", 10, 2, 3, 100, 0)])
    monkeypatch.setattr(runner, "profile_rate", lambda *_: 100)
    monkeypatch.setattr(runner, "reset_service_cache", lambda *args: None)
    def probe(*args, **kwargs):
        radius, root, seed = args[4], args[9], args[10]
        calls.append((root.name, radius, seed))
        labels = {mode: radius <= bound for mode, bound in thresholds.items()}
        if root.name.startswith("normal-r") and labels["normal"] and seed == 2:
            labels["normal"] = False
        return {"classification": labels}
    monkeypatch.setattr(runner, "service_probe", probe)
    plan = {"service": {"directions": ["coding"], "initial_repeats": 3,
                        "disagreement_repeats": 5, "radial_resolution": .05,
                        "hold_min_s": 1, "block_bootstrap_s": 30,
                        "bootstrap_samples": 10, "cache_block_tokens": 16,
                        "slos": {}}}
    runner.measure_frontier(plan, {}, {}, SimpleNamespace(
        host="h", sink_port=1, model="m"), object(), tmp_path)
    counts = {}
    for name, radius, _ in calls:
        if name.startswith("normal-r"):
            counts[radius] = counts.get(radius, 0) + 1
    assert sorted(counts.values()) == [3, 5]


def test_frontier_records_three_of_five_boundary(monkeypatch, tmp_path):
    thresholds = {"normal": 1, "emergency": 2, "stable": 3}
    monkeypatch.setattr(runner, "manifest_sessions", lambda *_: [runner.Session("s", 10, 2, 3, 100, 0)])
    monkeypatch.setattr(runner, "profile_rate", lambda *_: 100)
    monkeypatch.setattr(runner, "reset_service_cache", lambda *args: None)
    def probe(*args, **kwargs):
        radius, root, seed = args[4], args[9], args[10]
        labels = {mode: radius <= bound for mode, bound in thresholds.items()}
        if root.name.startswith("normal-r") and not labels["normal"] and seed in (2, 3):
            labels["normal"] = True
        return {"classification": labels}
    monkeypatch.setattr(runner, "service_probe", probe)
    plan = {"service": {"directions": ["coding"], "initial_repeats": 3,
                        "disagreement_repeats": 5, "radial_resolution": .05,
                        "hold_min_s": 1, "block_bootstrap_s": 30,
                        "bootstrap_samples": 10, "cache_block_tokens": 16,
                        "slos": {}}}
    rows, _ = runner.measure_frontier(plan, {}, {}, SimpleNamespace(
        host="h", sink_port=1, model="m"), object(), tmp_path)
    normal = next(row for row in rows if row["mode"] == "normal")
    assert normal["outside_feasible_votes"] == 2
    assert normal["outside_repeats"] == 5


def test_rho_uses_token_counter_differences_and_requires_thirty_seconds():
    rows = [
        {"monotonic_ns": 0, "vllm:prompt_tokens_total": 10,
         "vllm:generation_tokens_total": 20},
        {"monotonic_ns": 40_000_000_000, "vllm:prompt_tokens_total": 50,
         "vllm:generation_tokens_total": 60},
    ]
    assert runner.measured_rho(rows, 2, 2) == 1
    assert runner.measured_rho(rows, 2, 2, 2) == .5
    assert runner.require_rho(rows, 1.04, 2, 2) == 1
    with pytest.raises(RuntimeError, match="misses"):
        runner.require_rho(rows, 1.051, 2, 2)
    with pytest.raises(ValueError, match="thirty"):
        runner.measured_rho([rows[0], {**rows[1], "monotonic_ns": 29_000_000_000}], 2, 2)


def open_load(tmp_path, rps, max_inflight, seed=0):
    load = runner.DestinationLoad.__new__(runner.DestinationLoad)
    load.host, load.port, load.model = "h", 1, "m"
    load.sessions = [runner.Session("s", 4, 2, 3, 100, 0)]
    load.rate, load.max_inflight, load.seed = rps, max_inflight, seed
    load.timeout_s, load.rows = 720, []
    load.stop, load.admit = threading.Event(), threading.Event()
    load.admit.set()
    load.blocked_arrivals = 0
    load.root = tmp_path
    return load


@contextmanager
def running(load, release=None):
    """Always tear the arrival thread down, so a failure cannot leak into the next test."""
    thread = threading.Thread(target=load._run_open, daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        load.stop.set()
        if release is not None:
            release.set()
        thread.join(10)
        assert not thread.is_alive(), "arrival thread outlived its stop signal"


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.01)
    return predicate()


def test_open_loop_arrivals_do_not_wait_on_completions(monkeypatch, tmp_path):
    """A slow server must not throttle the offered arrival process."""
    released, launched = threading.Event(), []
    def slow(*_args):
        launched.append(1)
        released.wait(10)
        return {"status": 200}
    monkeypatch.setattr(runner, "issue", slow)
    load = open_load(tmp_path, rps=200, max_inflight=8)
    with running(load, released):
        assert wait_until(lambda: len(launched) == 8), \
            f"arrivals stalled at {len(launched)} before the in-flight cap"


def test_open_loop_caps_in_flight_and_counts_blocked_arrivals(monkeypatch, tmp_path):
    released, inflight, peak, lock = threading.Event(), [], [], threading.Lock()
    def held(*_args):
        with lock:
            inflight.append(1)
            peak.append(len(inflight))
        released.wait(10)
        with lock:
            inflight.pop()
        return {"status": 200}
    monkeypatch.setattr(runner, "issue", held)
    load = open_load(tmp_path, rps=200, max_inflight=3)
    with running(load, released):
        assert wait_until(lambda: load.blocked_arrivals > 0), \
            "an arrival blocked by the cap was never recorded"
        assert max(peak) <= 3, f"in-flight reached {max(peak)}, above the cap"


def test_pause_stops_arrivals_and_resume_restarts_them(monkeypatch, tmp_path):
    launched = []
    monkeypatch.setattr(runner, "issue", lambda *a: launched.append(1) or {"status": 200})
    load = open_load(tmp_path, rps=200, max_inflight=8)
    load.pause()
    with running(load):
        time.sleep(.3)
        assert launched == [], f"paused load issued {len(launched)} requests"
        load.resume()
        assert wait_until(lambda: bool(launched)), "resumed load never issued a request"


def test_open_loop_mode_requires_both_rate_and_cap():
    sessions = [runner.Session("s", 4, 2, 3, 100, 0)]
    with pytest.raises(ValueError, match="open-loop mode"):
        runner.DestinationLoad("h", 1, "m", sessions, 1, 1, 1, Path("/tmp"), 0, rps=4)
    with pytest.raises(ValueError, match="open-loop mode"):
        runner.DestinationLoad("h", 1, "m", sessions, 1, 1, 1, Path("/tmp"), 0,
                               max_inflight=8)


def test_open_loop_rate_overrides_the_rho_derived_rate(tmp_path):
    sessions = [runner.Session("s", 4, 2, 3, 100, 0)]
    load = runner.DestinationLoad("h", 1, "m", sessions, 16.5, 10, 10, tmp_path, 0,
                                  rps=4, max_inflight=32)
    assert load.rate == 4 and load.summary()["offered_rps"] == 4
    closed = runner.DestinationLoad("h", 1, "m", sessions, 16.5, 10, 10, tmp_path, 0)
    assert closed.rate == pytest.approx(16.5 / closed.work)


def test_deterministic_trace_reports_scheduled_prefill_and_decode_rho(tmp_path):
    sessions = [runner.Session("s", 4, 2, 3, 100, 0)]
    load = runner.DestinationLoad(
        "h", 1, "m", sessions, .04, 10, 5, tmp_path, 0,
        rps=2.5, max_inflight=8, arrival_schedule=(0, 30, 40, 50),
        warmup_s=30, measurement_s=30,
    )
    summary = load.summary()
    assert summary["offered_rho_prefill"] == pytest.approx(.02)
    assert summary["offered_rho_decode"] == pytest.approx(.06)
    assert summary["offered_rho"] == pytest.approx(.08)

def test_close_is_safe_when_start_failed_before_the_thread_ran(tmp_path):
    """A prewarm that raises must not mask itself with a join error."""
    sessions = [runner.Session("s", 4, 2, 3, 100, 0)]
    load = runner.DestinationLoad("h", 1, "m", sessions, 1, 1, 1, tmp_path, 0)
    assert load.thread.ident is None
    load.close()


def test_prewarm_uses_its_own_shorter_timeout(tmp_path, monkeypatch):
    """A wedged engine should fail in minutes, not one request timeout per session."""
    seen = []
    monkeypatch.setattr(runner, "prewarm",
                        lambda host, port, model, sessions, timeout: seen.append(timeout))
    monkeypatch.setattr(runner.MetricsSampler, "start", lambda self: None)
    sessions = [runner.Session("s", 4, 2, 3, 100, 0)]
    load = runner.DestinationLoad("h", 1, "m", sessions, 1, 1, 1, tmp_path, 0,
                                  timeout_s=720, prewarm_timeout_s=300)
    load.start()
    load.stop.set(); load.thread.join(5)
    assert seen == [300]
