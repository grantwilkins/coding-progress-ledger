"""Tests for the small two-A100 constrained-resource campaign."""

import json
from contextlib import contextmanager

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


@pytest.mark.parametrize("instrumentation_error", (False, True))
def test_discovery_reuses_only_matching_stacks_and_closes_failed_loads(
        monkeypatch, tmp_path, instrumentation_error):
    raw = campaign.default_inputs()
    raw["background_manifest"]["live"]["max_units"] = dict.fromkeys(campaign.AXES, 3)
    frozen = campaign.freeze_inputs(raw, 1)
    opened, closed, active, measured = [], [], [], []

    @contextmanager
    def stack(inputs_, wan_mbps, root, n_hbm=0):
        assert not active
        shared = {"key": (wan_mbps, n_hbm), "id": len(opened)}
        opened.append(shared)
        active.append(shared)
        try:
            if n_hbm:
                raise campaign.BackgroundLimit("allocation cannot start")
            yield shared
        finally:
            closed.append(active.pop())

    def measure(inputs_, state, root, shared=None):
        assert active == [shared]
        assert shared["key"] == (state["wan_mbps"], state["n_hbm"])
        measured.append((state, shared))
        if instrumentation_error and state["family"] == "wan_prefill":
            raise ValueError("missing telemetry")
        if max(state["n_prefill"], state["n_serving"]) > 1:
            raise campaign.BackgroundLimit("background cannot be maintained")
        return {**state, "operational": True, "capacity_inputs": capacity()}

    monkeypatch.setattr(campaign, "_a100_stack", stack)
    monkeypatch.setattr(campaign, "measure_a100_background", measure)
    if instrumentation_error:
        with pytest.raises(ValueError, match="missing telemetry"):
            campaign.discover_live(frozen, tmp_path)
        assert not active and closed == opened
        return
    discovery, states = campaign.discover_live(frozen, tmp_path)

    assert len(discovery) == 8 and len(states) == 9
    assert measured[0][1] is measured[1][1]
    grid = [(state, shared) for state, shared in measured if state["family"] == "wan_prefill"]
    for wan in raw["wan_mbps"]:
        assert len({shared["id"] for state, shared in grid if state["wan_mbps"] == wan}) == 1
    assert not active and closed == opened
    assert len(opened) < len(measured)


def test_hbm_reservation_removes_physical_blocks_and_stops_before_model_no_longer_fits():
    from dataclasses import replace
    from profiles import ModelProfile

    profile = ModelProfile.load(campaign.DEFAULT_PROFILE)
    unit = 4 * 1024**3
    blocks = campaign._hbm_blocks(profile, unit, 32768)
    reserved = (profile.kv_capacity_tokens - blocks * 16) * 49152
    assert unit <= reserved < unit + 16 * 49152
    assert campaign._hbm_blocks(profile, 10 * unit, 32768) * 16 >= 32768
    with pytest.raises(campaign.BackgroundLimit, match="maximum-length"):
        campaign._hbm_blocks(profile, 11 * unit, 32768)
    with pytest.raises(ValueError, match="geometry"):
        campaign._hbm_blocks(replace(profile, model="other"), unit, 32768)


def test_hbm_worker_holds_exact_bytes_in_existing_cuda_context(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.delenv("QH_LMCACHE_MODE", raising=False)
    from lmcache_compat.connector_patch import ConstrainedHBM

    allocated, calls = [1024], []
    def zeros(size, dtype, device):
        calls.append((size, dtype, device))
        allocated[0] += size
        return SimpleNamespace(numel=lambda: size)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        zeros=zeros, uint8="uint8", cuda=SimpleNamespace(
            empty_cache=lambda: None, synchronize=lambda device: None,
            memory_allocated=lambda device: allocated[0])))
    worker = ConstrainedHBM()
    worker.device = "cuda:0"
    before = worker.qh_hbm()
    held = worker.qh_hbm(str(4 * 1024**3))
    assert held["allocated_bytes"] == 4 * 1024**3
    assert held["torch_allocated_bytes"] - before["torch_allocated_bytes"] == 4 * 1024**3
    assert worker.qh_hbm() == held
    assert calls == [(4 * 1024**3, "uint8", "cuda:0")]
    with pytest.raises(ValueError, match="already allocated"):
        worker.qh_hbm("1")


def test_hbm_rpc_verifies_destination_allocation_and_stable_worker(monkeypatch):
    from types import SimpleNamespace
    import migration_testbed as testbed

    cfg = SimpleNamespace(host="localhost", sink_port=8200)
    held, calls, copies = [0], [], [1]
    def rpc(host, port, method, path, payload):
        assert (host, port, method, path) == ("localhost", 8200, "POST", "/collective_rpc")
        assert payload["method"] == "qh_hbm"
        calls.append(payload["args"])
        if payload["args"]:
            held[0] = int(payload["args"][0])
        return {"results": [{"pid": 17, "allocated_bytes": held[0],
                             "torch_allocated_bytes": 1024 + held[0]}] * copies[0]}
    monkeypatch.setattr(testbed, "http_json", rpc)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *args, **kw:
                        f"17, {1000 + held[0] // 1024**2}\n99, 70000\n")
    holder = campaign._hbm_allocation(cfg, 4 * 1024**3)
    assert calls == [[], ["4294967296"]]
    assert holder["before"]["process_gpu_memory_bytes"] == 1000 * 1024**2
    assert campaign._hbm_telemetry(cfg, holder)["allocated_bytes"] == 4 * 1024**3
    held[0] = 0
    with pytest.raises(campaign.BackgroundLimit, match="changed"):
        campaign._hbm_telemetry(cfg, holder)
    copies[0] = 2
    with pytest.raises(ValueError, match="one destination worker"):
        campaign._hbm_telemetry(cfg, holder)


@pytest.mark.parametrize("field", ("torch_allocated_bytes", "process_gpu_memory_bytes"))
def test_hbm_allocation_requires_visible_memory_increase(monkeypatch, field):
    unit = 4 * 1024**3
    before = {"pid": 17, "allocated_bytes": 0, "torch_allocated_bytes": 1024,
              "process_gpu_memory_bytes": 5000 * 1024**2}
    after = {**before, "allocated_bytes": unit,
             **{key: before[key] + unit for key in (
                 "torch_allocated_bytes", "process_gpu_memory_bytes")}}
    after[field] = before[field]
    monkeypatch.setattr(campaign, "_hbm_rpc",
                        lambda cfg, allocated=None: before if allocated is None else after)
    with pytest.raises(campaign.BackgroundLimit, match="visibly resident"):
        campaign._hbm_allocation(object(), unit)


@pytest.mark.parametrize("kind", ("prefill", "serving"))
@pytest.mark.parametrize("measure", (False, True))
def test_background_uses_operational_hold_and_measured_capacity(monkeypatch, kind, measure):
    from types import SimpleNamespace

    class Load:
        failure = None
        blocked_arrivals = 0
        rows = [{"ok": True, "end_ns": 10_000_000_000},
                {"ok": True, "end_ns": 40_000_000_000}]
        prefill_rate, decode_rate, normal_bound, rate = 10, 20, 1, .25
        sampler = SimpleNamespace(error=None, rows=[
            {"monotonic_ns": second * 1_000_000_000,
             "vllm:num_requests_running": 3,
             "vllm:num_requests_waiting": 0} for second in range(61)])

        def wait_ready(self):
            raise AssertionError("background must not use the normalized-load gate")

    class Destination:
        @staticmethod
        def service_completion(row):
            return row["ok"]

        @staticmethod
        def measured_rho(rows, prefill_rate, decode_rate, normal_bound):
            assert (prefill_rate, decode_rate, normal_bound) == (10, 20, 1)
            return 1.2

    slept = []
    monkeypatch.setattr(campaign.time, "sleep", slept.append)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: sum(slept))
    monkeypatch.setattr(campaign.time, "monotonic_ns", lambda: int(sum(slept) * 1e9))

    load = Load()
    report = campaign._wait_background(load, kind, 30, Destination, measure)

    assert slept == [30] * (2 if measure else 1)
    assert load.achieved == 1.2
    assert report["completed_requests"] == 1
    assert report["configured_rps"] == .25
    assert report["slope_lower_95_per_s"] == 0
    bad = Load()
    bad.blocked_arrivals = 1
    with pytest.raises(RuntimeError, match="serving background was not maintained"):
        campaign._wait_background(bad, kind, 30, Destination)


def test_total_backlog_detects_waiting_and_running_growth():
    def rows(levels, running=False):
        return [{"monotonic_ns": second * 1_000_000_000,
                 "vllm:num_requests_running": level if running else 0,
                 "vllm:num_requests_waiting": 0 if running else level}
                for second, level in enumerate(levels)]

    stable = campaign._backlog_stability(
        rows([10 + second % 2 for second in range(30)]), 0, 30_000_000_000)
    growing = campaign._backlog_stability(
        rows(range(30)), 0, 30_000_000_000)
    busy = campaign._backlog_stability(
        rows(range(30), running=True), 0, 30_000_000_000)

    assert stable["slope_lower_95_per_s"] <= 0
    assert growing["slope_lower_95_per_s"] > 0
    assert busy["slope_lower_95_per_s"] > 0


@pytest.mark.parametrize("kind", ("prefill", "serving"))
@pytest.mark.parametrize("growth", (1, 20))
def test_background_distinguishes_count_noise_from_in_service_backlog(monkeypatch, kind, growth):
    from types import SimpleNamespace

    slept = []
    monkeypatch.setattr(campaign.time, "sleep", slept.append)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: sum(slept))
    monkeypatch.setattr(campaign.time, "monotonic_ns", lambda: int(sum(slept) * 1e9))
    load = SimpleNamespace(
        failure=None, blocked_arrivals=0, rows=[{"end_ns": 10_000_000_000}],
        prefill_rate=10, decode_rate=20, normal_bound=1, rate=.25,
        sampler=SimpleNamespace(error=None, rows=[
            {"monotonic_ns": second * 1_000_000_000,
             "vllm:num_requests_running": growth * second / 30,
             "vllm:num_requests_waiting": 0} for second in range(31)]))
    destination = SimpleNamespace(service_completion=lambda row: True,
                                  measured_rho=lambda *args: 1.2)
    if growth == 20:
        with pytest.raises(RuntimeError, match="background backlog grew"):
            campaign._wait_background(load, kind, 30, destination)
    else:
        report = campaign._wait_background(load, kind, 30, destination)
        assert report["slope_lower_95_per_s"] > 0
        assert report["completed_requests"] == 1


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
    capacity_inputs = {"baseline_work": [0, 0], "baseline_kv_tokens": 0,
                       "resources": {"prefill": campaign.D_S}}

    slow = campaign._planner_decisions(
        frozen, {"wan_mbps": 1000}, item, "queue_haul", profile,
        capacity_inputs)
    fast = campaign._planner_decisions(
        frozen, {"wan_mbps": 10000}, item, "queue_haul", profile,
        capacity_inputs)

    assert len(slow) == len(fast) == 8
    assert sum(row["action"] == "kv_transfer" for row in slow) \
        < sum(row["action"] == "kv_transfer" for row in fast)


@pytest.mark.parametrize("policy", ("queue_haul", "greedy", "kv_only", "replay_only"))
def test_live_planner_uses_actual_destination_kv_capacity(policy):
    from profiles import ModelProfile

    frozen = campaign.default_inputs()
    profile = ModelProfile.load(campaign.DEFAULT_PROFILE)
    original_capacity = profile.kv_capacity_tokens
    item = next(row for row in frozen["packs"] if row["pack_id"] == "large-r0")
    tokens = {row["initial_tokens"] for row in item["sessions"]}
    assert len(tokens) == 1
    result = campaign._planner_decisions(
        frozen, {"wan_mbps": 10000}, item, policy, profile,
        {"baseline_work": [0, 0], "baseline_kv_tokens": 0,
         "kv_capacity_tokens": 2 * min(tokens),
         "resources": {"prefill": campaign.D_S}})

    assert len(result) == 2
    assert profile.kv_capacity_tokens == original_capacity


@pytest.mark.parametrize("policy", campaign.POLICIES)
def test_operational_saturated_background_keeps_valid_policy_nonattainment(policy):
    from profiles import ModelProfile

    frozen = campaign.default_inputs()
    profile = ModelProfile.load(campaign.DEFAULT_PROFILE)
    capacity_inputs = {"baseline_work": [1.15, .04], "baseline_kv_tokens": 0,
                       "resources": {"prefill": 0}}
    result = campaign._planner_decisions(
        frozen, {"wan_mbps": 1000}, frozen["packs"][0], policy,
        profile, capacity_inputs)

    assert len(result) == (8 if policy == "per_session_greedy" else 0)
    assert all(row["action"] == "kv_transfer" for row in result)
    assert capacity_inputs["baseline_work"] == [1.15, .04]


def test_live_per_session_timing_responds_to_wan_and_prefill_without_admission(monkeypatch):
    import planner
    from profiles import ModelProfile

    def no_admission(*args, **kwargs):
        pytest.fail("independent action choice called aggregate admission")

    monkeypatch.setattr(planner, "plan", no_admission)
    frozen = campaign.default_inputs()
    profile = ModelProfile.load(campaign.DEFAULT_PROFILE)
    item = next(row for row in frozen["packs"] if row["pack_id"] == "mixed-r0")
    def choices(wan, prefill):
        return campaign._planner_decisions(
            frozen, {"wan_mbps": wan}, item, "per_session_greedy", profile,
            {"baseline_work": [1 - prefill, 1],
             "baseline_kv_tokens": profile.kv_capacity_tokens,
             "resources": {"prefill": prefill * campaign.D_S}})

    slow, fast, loaded = choices(1000, 1), choices(10000, 1), choices(1000, .05)
    kv = lambda rows: sum(row["action"] == "kv_transfer" for row in rows)
    assert kv(slow) < kv(fast)
    assert kv(slow) < kv(loaded)
    assert all(len(rows) == 8 for rows in (slow, fast, loaded, choices(.01, 1)))


@pytest.mark.parametrize("policy", ("queue_haul", "greedy", "kv_only", "replay_only"))
def test_live_planner_receives_residual_prefill_budget(monkeypatch, policy):
    from types import SimpleNamespace
    import planner
    from profiles import ModelProfile

    seen = []
    def capture(*args, destination, **kwargs):
        seen.append(destination.pools[0])
        return SimpleNamespace(moves=())

    monkeypatch.setattr(planner, "plan", capture)
    frozen = campaign.default_inputs()
    profile = ModelProfile.load(campaign.DEFAULT_PROFILE)
    for residual in (.2, 0):
        campaign._planner_decisions(
            frozen, {"wan_mbps": 1000}, frozen["packs"][0], policy, profile,
            {"baseline_work": [1 - residual, 0], "baseline_kv_tokens": 0,
             "resources": {"prefill": residual * campaign.D_S}})
    assert seen[0].migration_headroom == {"replay": .2}
    assert seen[1].methods == ("kv_transfer",)


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


@pytest.mark.parametrize("deadline, attained, relief", ((14, False, 3.2), (30, True, 8)))
def test_state_deadline_controls_trailing_window_attainment(deadline, attained, relief):
    item = pack()
    state = {**campaign.state_grid(dict.fromkeys(campaign.AXES, 0), [10000])[0],
             "deadline_s": deadline}
    job = next(row for row in campaign.execution_schedule([state], [item], 1, repeats=1)
               if row["policy"] == "queue_haul")
    raw = [{"episode_id": job["episode_id"], "capacity_inputs": capacity(),
            "decisions": [{"session_id": row["session_id"], "action": "replay",
                           "completion_s": 12} for row in item["sessions"]]}]
    result = campaign.normalize_episodes(raw, [job], [item], 8)[0]
    assert result["deadline_s"] == deadline
    assert result["target_attained"] is attained
    assert result["achieved_relief_w"] == 8
    assert result["window_relief_w"] == pytest.approx(relief)
    assert result["target_time_s"] == (17 if attained else None)


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


def test_live_runner_retries_incomplete_stream_episode(tmp_path):
    frozen = campaign.freeze_inputs(inputs(), 7)
    state = {**campaign.state_grid(dict.fromkeys(campaign.AXES, 0), [10000])[0],
             "capacity_inputs": capacity()}
    schedule = campaign.execution_schedule(
        [state], frozen["inputs"]["packs"][:1], 7, repeats=1)[:1]
    plan = {"frozen": frozen, "states": [state], "annotations": [],
            "schedule": schedule}
    roots = []

    def runner(_inputs, state_, pack_, job_, root):
        roots.append(root)
        root.mkdir(parents=True)
        if len(roots) == 1:
            raise campaign.RetryableEpisode("incomplete stream")
        action = "kv_transfer" if job_["policy"] == "kv_only" else "replay"
        return {"capacity_inputs": state_["capacity_inputs"],
                "decisions": [{"session_id": row["session_id"],
                               "action": action, "completion_s": 1}
                              for row in pack_["sessions"]]}

    assert len(campaign.run_live(plan, tmp_path, runner)) == 1
    assert [root.name for root in roots] == [
        schedule[0]["episode_id"], f"{schedule[0]['episode_id']}-attempt-1"]


def test_live_runner_bounds_stream_retries(tmp_path):
    frozen = campaign.freeze_inputs(inputs(), 7)
    state = {**campaign.state_grid(dict.fromkeys(campaign.AXES, 0), [10000])[0],
             "capacity_inputs": capacity()}
    schedule = campaign.execution_schedule(
        [state], frozen["inputs"]["packs"][:1], 7, repeats=1)[:1]
    plan = {"frozen": frozen, "states": [state], "annotations": [],
            "schedule": schedule}
    attempts = []

    def runner(*args):
        attempts.append(args[-1])
        raise campaign.RetryableEpisode("incomplete stream")

    with pytest.raises(campaign.RetryableEpisode):
        campaign.run_live(plan, tmp_path, runner)
    assert len(attempts) == campaign.MAX_EPISODE_ATTEMPTS


def test_live_runner_reuses_stack_across_blocks_and_restarts_after_failure(monkeypatch, tmp_path):
    frozen = campaign.freeze_inputs(inputs(), 7)
    states = [{**campaign.state_grid(dict.fromkeys(campaign.AXES, 0),
                                    [10000])[0],
               "operational": True, "capacity_inputs": capacity()}]
    schedule = campaign.execution_schedule(
        states, frozen["inputs"]["packs"][:2], 7, repeats=1)
    plan = {"frozen": frozen, "states": states, "annotations": [],
            "schedule": schedule}
    starts, stops, calls = [], [], []

    @contextmanager
    def stack(_inputs, wan_mbps, root, n_hbm=0):
        root.mkdir(parents=True)
        shared = {"wan_mbps": wan_mbps, "root": root}
        starts.append(shared)
        try:
            yield shared
        finally:
            stops.append(shared)

    def runner(_inputs, state, pack_, job, root, shared):
        calls.append((job["episode_id"], shared))
        assert shared["wan_mbps"] == state["wan_mbps"]
        if len(calls) == 1:
            root.mkdir(parents=True)
            raise campaign.RetryableEpisode("incomplete stream")
        action = "kv_transfer" if job["policy"] == "kv_only" else "replay"
        return {"capacity_inputs": state["capacity_inputs"],
                "decisions": [{"session_id": row["session_id"],
                               "action": action, "completion_s": 1}
                              for row in pack_["sessions"]]}

    monkeypatch.setattr(campaign, "_a100_stack", stack)
    monkeypatch.setattr(campaign, "a100_episode", runner)
    raw = campaign.run_live(plan, tmp_path)

    assert len(raw) == 10
    assert [row["wan_mbps"] for row in starts] == [10000, 10000]
    assert stops == starts
    assert [row["root"].name for row in starts] == [
        schedule[0]["block_id"], f"{schedule[0]['block_id']}-attempt-1"]
    assert calls[0][1] is not calls[1][1]
    assert len({id(shared) for _, shared in calls[1:]}) == 1
