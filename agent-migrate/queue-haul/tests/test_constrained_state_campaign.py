"""Tests for the small two-A100 constrained-resource campaign."""

import json

import pytest

import constrained_state_campaign as campaign


def pack(name="pack"):
    sessions = [{"session_id": f"{name}-{i}", "initial_tokens": 2048,
                 "turn_index": 0} for i in range(8)]
    return {"pack_id": name, "sessions": sessions,
            "power_gains": [float(mask.bit_count()) for mask in range(256)]}


def demands():
    return [{"pack_id": "pack", "session_id": f"pack-{i}", "action": action,
             "duration_s": 2 if action == "kv_transfer" else 1,
             **{f"d_{resource}": 1 for resource in campaign.RESOURCES}}
            for i in range(8) for action in campaign.ACTIONS]


def inputs():
    packs = [pack(f"pack-{i}") for i in range(10)]
    rows = [{"pack_id": item["pack_id"],
             "session_id": session["session_id"], "action": action,
             "duration_s": 1,
             **{f"d_{resource}": 1 for resource in campaign.RESOURCES}}
            for item in packs for session in item["sessions"]
            for action in campaign.ACTIONS]
    return {"packs": packs, "profile": {"model": "gpt-oss", "gpu": "A100"},
            "action_demands": rows, "target": 8,
            "wan_mbps": [1000, 2500, 5000, 10000],
            "background_manifest": {"seed": 3}}


def capacity(value=8):
    return {"resources": dict.fromkeys(campaign.RESOURCES, value)}


def test_freeze_keeps_only_the_fixed_contract():
    first = campaign.freeze_inputs(inputs(), 7)
    second = campaign.freeze_inputs(inputs(), 7)

    assert first == second
    assert first["constants"] == {"deadline_s": 30, "power_window_s": 5,
                                  "repeats": 2}
    assert len(first["sha256"]) == 64
    with pytest.raises(ValueError, match="ten eight-session packs"):
        campaign.freeze_inputs({**inputs(), "packs": inputs()["packs"][:-1]}, 7)


def test_default_inputs_reuse_ten_existing_packs():
    result = campaign.default_inputs()

    assert len(result["packs"]) == 10
    assert {row["pack_id"].split("-r")[0] for row in result["packs"]} == {
        "tiny", "small", "medium", "mixed", "large"}
    campaign.freeze_inputs(result, 1)


def test_discovery_and_state_grid_are_exact_and_deduplicated():
    rows = [{"axis": axis, "count": count,
             "operational": count <= limit, "telemetry": {"count": count}}
            for axis, limit in (("prefill", 2), ("serving", 1), ("hbm", 2))
            for count in range(limit + 2)]

    limits, retained = campaign.discovery_limits(rows)
    states = campaign.state_grid(limits, [1000, 5000, 10000])

    assert limits == {"prefill": 2, "serving": 1, "hbm": 2}
    assert len(retained) == 3 + 2 + 3
    assert len(states) == 3 * 3 + 1 + 2
    assert len({row["state_id"] for row in states}) == len(states)
    assert {(row["wan_mbps"], row["n_prefill"])
            for row in states if row["family"] == "wan_prefill"} == {
                (wan, count) for wan in (1000, 5000, 10000)
                for count in range(3)}
    assert all(row["wan_mbps"] == 10000
               for row in states if row["family"] != "wan_prefill")


def test_live_discovery_rejects_an_operational_safety_cap(tmp_path):
    raw = campaign.default_inputs()
    raw["background_manifest"]["live"]["max_units"] = dict.fromkeys(
        campaign.AXES, 0)
    frozen = campaign.freeze_inputs(raw, 1)

    def measure(_inputs, state, _root):
        return {**state, "operational": True,
                "capacity_inputs": capacity()}

    with pytest.raises(campaign.BackgroundLimit,
                       match="prefill discovery reached max_units=0"):
        campaign.discover_live(frozen, tmp_path, measure)


def test_resident_telemetry_requires_visible_hbm(monkeypatch):
    metrics = {"vllm:gpu_cache_usage_perc": .25,
               "vllm:num_requests_running": 0,
               "vllm:num_requests_waiting": 0}
    monkeypatch.setattr(campaign, "_metrics", lambda *_args: metrics)

    assert campaign._resident_telemetry(
        object(), object(), object(), .24, .01) == metrics

    metrics["vllm:gpu_cache_usage_perc"] = .2
    with pytest.raises(campaign.BackgroundLimit, match="HBM use decreased"):
        campaign._resident_telemetry(
            object(), object(), object(), .24, .01)


def test_serving_background_uses_an_operational_hold(monkeypatch):
    class Load:
        failure = None
        blocked_arrivals = 0
        rows = [{"ok": True}]

        def wait_ready(self):
            raise AssertionError("serving must not use the normalized-load gate")

    class Destination:
        @staticmethod
        def service_completion(row):
            return row["ok"]

    slept = []
    monkeypatch.setattr(campaign.time, "sleep", slept.append)

    campaign._wait_background(Load(), "serving", 30, Destination)

    assert slept == [30]
    bad = Load()
    bad.blocked_arrivals = 1
    with pytest.raises(RuntimeError, match="serving background was not maintained"):
        campaign._wait_background(bad, "serving", 30, Destination)


def test_serving_stability_rejects_only_clear_backlog_growth():
    def rows(levels):
        return [{"monotonic_ns": second * 1_000_000_000,
                 "vllm:num_requests_running": level,
                 "vllm:num_requests_waiting": 0}
                for second, level in enumerate(levels)]

    stable = campaign._backlog_stability(
        rows([10 + second % 2 for second in range(30)]), 0, 30_000_000_000)
    growing = campaign._backlog_stability(
        rows(range(30)), 0, 30_000_000_000)

    assert stable["slope_lower_95_per_s"] <= 0
    assert growing["slope_lower_95_per_s"] > 0


def test_oracle_emits_only_the_four_requested_booleans():
    item, rows = pack(), demands()
    for row in rows:
        row["d_wan"] = 1 if row["action"] == "kv_transfer" else 0
        row["d_prefill"] = 1 if row["action"] == "replay" else 0
        row["d_service"] = row["d_hbm"] = 0

    result = campaign.oracle_case(
        item, rows, {"wan": 4, "prefill": 4, "service": 0, "hbm": 0}, 8)

    assert result == {"any_full": True, "kv_full": False,
                      "replay_full": False, "mixed_full": True}


def test_per_session_greedy_uses_current_state_and_dispatches_all():
    item, rows = pack(), demands()

    def timing(_pack, moves, state):
        action = moves[0]["action"]
        duration = (10 / state["wan"] if action == "kv_transfer"
                    else 10 / state["prefill"])
        return {"makespan_s": duration}

    slow_wan = campaign.per_session_greedy(
        item, rows, {"wan": 1, "service": 1, "prefill": 10, "hbm": 8}, timing)
    slow_prefill = campaign.per_session_greedy(
        item, rows, {"wan": 10, "service": 1, "prefill": 1, "hbm": 8}, timing)

    assert {row["action"] for row in slow_wan} == {"replay"}
    assert {row["action"] for row in slow_prefill} == {"kv_transfer"}
    assert all(row["dispatch"] for row in slow_wan + slow_prefill)


def test_existing_planner_receives_current_wan_and_background():
    from profiles import ModelProfile

    frozen = campaign.default_inputs()
    profile = ModelProfile.load(campaign.DEFAULT_PROFILE)
    item = next(row for row in frozen["packs"] if row["pack_id"] == "mixed-r0")
    capacity_inputs = {"baseline_work": [0, 0], "baseline_kv_tokens": 0}

    slow = campaign._planner_decisions(
        frozen, {"wan_mbps": 1000}, item, "queue_haul", profile,
        capacity_inputs)
    fast = campaign._planner_decisions(
        frozen, {"wan_mbps": 10000}, item, "queue_haul", profile,
        capacity_inputs)

    assert len(slow) == len(fast) == 8
    assert sum(row["action"] == "kv_transfer" for row in slow) \
        < sum(row["action"] == "kv_transfer" for row in fast)


def test_schedule_executes_every_state_pack_policy_and_repeat():
    states = campaign.state_grid(
        {"prefill": 1, "serving": 1, "hbm": 1}, [1000, 10000])
    packs = [pack("a"), pack("b")]

    first = campaign.execution_schedule(states, packs, 11)
    second = campaign.execution_schedule(states, packs, 11)

    assert first == second
    assert len(first) == len(states) * 2 * len(campaign.POLICIES) * 2
    for key in {(row["state_id"], row["pack_id"], row["repeat"])
                for row in first}:
        block = [row for row in first
                 if (row["state_id"], row["pack_id"], row["repeat"]) == key]
        assert {row["policy"] for row in block} == set(campaign.POLICIES)
        assert sorted(row["policy_order"] for row in block) == list(range(5))


def test_reducer_keeps_nonattainment_and_fills_not_moved():
    item = pack()
    state = {**campaign.state_grid(
        {"prefill": 0, "serving": 0, "hbm": 0}, [10000])[0],
        "capacity_inputs": capacity()}
    job = next(row for row in campaign.execution_schedule(
        [state], [item], 1, repeats=1) if row["policy"] == "queue_haul")
    raw = [{"episode_id": job["episode_id"],
            "capacity_inputs": state["capacity_inputs"],
            "decisions": [
                {"session_id": "pack-0", "action": "replay", "completion_s": 10},
                {"session_id": "pack-1", "action": "kv_transfer", "completion_s": 40},
            ]}]

    result = campaign.normalize_episodes(raw, [job], [item], 8)[0]

    assert result["target_attained"] is False
    assert result["achieved_relief_w"] == 1
    assert (result["replay_count"], result["kv_count"],
            result["not_moved_count"]) == (1, 1, 6)
    assert result["target_time_s"] is None


def test_compile_reduce_and_two_figures(tmp_path):
    frozen = campaign.freeze_inputs(inputs(), 7)
    discovery = [{"axis": axis, "count": 0, "operational": True,
                  "telemetry": {"ok": True}} for axis in campaign.AXES]
    states = [{**row, "operational": True, "capacity_inputs": capacity()}
              for row in campaign.state_grid(dict.fromkeys(campaign.AXES, 0),
                                             frozen["inputs"]["wan_mbps"])]
    compiled = campaign.compile_campaign(frozen, discovery, states,
                                         tmp_path / "compiled")
    raw = []
    packs = {row["pack_id"]: row for row in frozen["inputs"]["packs"]}
    for job in compiled["schedule"]:
        action = "kv_transfer" if job["policy"] == "kv_only" else "replay"
        decisions = [{"session_id": session["session_id"],
                      "action": action, "completion_s": 10}
                     for session in packs[job["pack_id"]]["sessions"]]
        raw.append({"episode_id": job["episode_id"],
                    "capacity_inputs": capacity(), "decisions": decisions})

    out = tmp_path / "reduced"
    campaign.reduce_campaign(compiled, raw, out)

    assert set(path.name for path in (tmp_path / "compiled").iterdir()) == {
        "plan.json", "background_discovery.csv", "background_states.csv",
        "oracle_annotations.csv", "execution_schedule.csv"}
    assert (out / "episodes.csv").is_file()
    assert (out / "target_attainment.png").is_file()
    assert (out / "action_composition.png").is_file()


def test_live_runner_uses_schedule_order_and_persists_raw_rows(tmp_path):
    frozen = campaign.freeze_inputs(inputs(), 7)
    state = {**campaign.state_grid(dict.fromkeys(campaign.AXES, 0), [10000])[0],
             "capacity_inputs": capacity()}
    schedule = campaign.execution_schedule(
        [state], frozen["inputs"]["packs"][:1], 7, repeats=1)[:2]
    plan = {"frozen": frozen, "states": [state], "annotations": [],
            "schedule": schedule}
    called, failed, roots = [], set(), []

    def runner(inputs_, state_, pack_, job_, root_):
        called.append(job_["episode_id"])
        roots.append(root_)
        if job_["episode_id"] == schedule[1]["episode_id"] and not failed:
            failed.add(job_["episode_id"])
            root_.mkdir(parents=True)
            raise RuntimeError("allocation ended")
        action = "kv_transfer" if job_["policy"] == "kv_only" else "replay"
        return {"capacity_inputs": state_["capacity_inputs"],
                "decisions": [{"session_id": row["session_id"],
                               "action": action, "completion_s": 1}
                              for row in pack_["sessions"]]}

    with pytest.raises(RuntimeError, match="allocation ended"):
        campaign.run_live(plan, tmp_path, runner)
    raw = campaign.run_live(plan, tmp_path, runner)

    assert called == [schedule[0]["episode_id"], schedule[1]["episode_id"],
                      schedule[1]["episode_id"]]
    assert roots[-1].name == f"{schedule[1]['episode_id']}-attempt-1"
    assert len(raw) == 2
    assert len((tmp_path / "raw_episodes.jsonl").read_text().splitlines()) == 2
    (tmp_path / "plan.sha256").write_text("changed\n")
    with pytest.raises(ValueError, match="run plan hash changed"):
        campaign.run_live(plan, tmp_path, runner)
