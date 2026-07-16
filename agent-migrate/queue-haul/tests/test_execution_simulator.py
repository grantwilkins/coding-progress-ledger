"""
Claim:
Execution follows background copy, pause, catch-up, route switch, and final node
state while overlapping transfers share every bottleneck.

Plausible wrong implementations:
- Credit source power when bytes finish instead of when the route switches.
- Serialize simultaneous transfers or exceed a shared link.
- Recalculate unrelated transfers or miss a transfer connected through a second link.
- Stall when shared rates leave a small floating-point byte remainder.
- Change or retain the wrong GPU slot load when one session moves.
- Enter sleep/off before the final source session commits.
- Skip catch-up when a request changes state during background preparation.
- Transfer nonexistent cold KV or defer replay for GPU-resident active KV.
"""

import json

import pytest

import simulate
from profiles import ModelProfile
from simulate import (ExecutionScenario, ExecutionSimulator, NetworkLink, PlannedMove,
                      PowerNode, ServingInstance, SimRequest, SimSession, execute,
                      fair_link_rates, predict, step_average)


def model(tmp_path, switch=1, block_s=0, shutdown=2, setup=0, tp=2, replay_rate=None):
    source = {"kind": "measured", "reference": "hand", "valid_range": [1, 1000], "relative_error": 0}
    rate = {"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]}
    raw = {
        "schema": "queue-haul-model-profile-v1", "profile_id": "hand", "status": "fitted",
        "model": "m", "hardware": "h", "precision": "bf16", "tensor_parallel": tp,
        "gpus_per_node": 2, "power_scope": "gpu", "power_window_s": 1,
        "max_ell": 1, "max_parallel_moves": 2,
        "max_parallel_replay": 1, "max_parallel_kv": 1,
        "sources": {k: source for k in ("power", "service", "replay", "kv_transfer", "transitions")},
        "cases": {"central": {
            "F": 100, "G": 100, "power_curve": [[0, 10], [0.5, 30], [1, 40]],
            "prefill_tps": rate, "decode_tps": rate, "replay_tps": replay_rate or rate,
            "kv_transfer": {"block_tokens": 10, "block_bytes": 100, "setup_s": setup,
                            "block_processing_s": block_s, "sync_s": 0},
            "switch_s": switch, "sleep_power_w": 2, "sleep_s": 1, "shutdown_s": shutdown,
            "action_power_w": {"replay": [0, 0], "kv_transfer": [0, 0],
                               "replay_on_request": [0, 0], "catch_up": [0, 0],
                               "sleep": [0, 0], "off": [0, 0]},
        }},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    return ModelProfile.load(path)


def scenario(sessions, deadline=20, end=30, final="awake", tp=2, links=None,
             controller_delay=0):
    return ExecutionScenario(
        deadline, end, 0, final, controller_delay,
        (PowerNode("src", 2, True), PowerNode("dst", 2, False)),
        (ServingInstance("source", ("src",) * tp), ServingInstance("dest", ("dst",) * tp)),
        tuple(sessions), tuple(links or (NetworkLink("wan", 100),)),
    )


def test_fair_rates_redistribute_capacity_across_two_bottlenecks():
    rates = fair_link_rates({0: ("a",), 1: ("a", "b"), 2: ("b",)}, {"a": 100, "b": 20})
    assert rates == pytest.approx({0: 90, 1: 10, 2: 10})
    assert rates[0] + rates[1] == pytest.approx(100)
    assert rates[1] + rates[2] == pytest.approx(20)


def test_rate_changes_stay_within_connected_links(tmp_path, monkeypatch):
    paths = (("a",), ("a", "b"), ("b",), ("c",), ("c",))
    contexts = (40, 10, 40, 10, 20)
    sessions = tuple(
        SimSession(str(i), f"source-{i}", context, 0, 0, 1)
        for i, context in enumerate(contexts)
    )
    topology = ExecutionScenario(
        20, 30, 0, "awake", 0,
        (PowerNode("src", 5, True), PowerNode("dst", 5, False)),
        tuple(ServingInstance(f"source-{i}", ("src",)) for i in range(5))
        + tuple(ServingInstance(f"dest-{i}", ("dst",)) for i in range(5)),
        sessions,
        (NetworkLink("a", 100), NetworkLink("b", 100), NetworkLink("c", 200)),
    )
    moves = tuple(
        PlannedMove(str(i), f"dest-{i}", "kv_transfer", i, path)
        for i, path in enumerate(paths)
    )
    calls = []
    original = simulate.fair_link_rates

    def record(active, links):
        calls.append((tuple(sorted(active.values())), tuple(sorted(links))))
        return original(active, links)

    monkeypatch.setattr(simulate, "fair_link_rates", record)
    result = execute(topology, model(tmp_path, tp=1), moves)
    finished = {row.session_id: row.end_s for row in result.network}

    assert ((("c",),), ("c",)) in calls
    assert finished == pytest.approx({"0": 5, "1": 2, "2": 5, "3": 1, "4": 1.5})


def test_deadline_power_is_an_exact_trailing_window_average():
    points = ((0, 100, 0), (4, 20, 0))
    assert step_average(points, 5, 5) == pytest.approx(84)
    assert step_average(points, 5, 1) == pytest.approx(20)


def test_parallel_transfer_and_power_credit_at_commit(tmp_path):
    sessions = [SimSession(str(i), "source", 10, 25, 0, 40) for i in range(2)]
    moves = tuple(PlannedMove(str(i), "dest", "kv_transfer", i, ("wan",)) for i in range(2))
    result = execute(scenario(sessions), model(tmp_path), moves)

    rows = {row.session_id: row for row in result.sessions}
    assert rows["0"].initial_ready_s == pytest.approx(2)  # 100 B each over one 100 B/s link
    assert rows["1"].initial_ready_s == pytest.approx(2)
    assert rows["0"].committed_s == pytest.approx(3)
    assert result.power[-1][0] == 30
    assert len(result.network) == 2
    assert {row.session_id for row in result.network} == {"0", "1"}
    assert {(row.bytes, row.start_s, row.end_s) for row in result.network} == {
        (100, 0, 2)
    }
    before = [p for p in result.power if p[0] < 3]
    after = [p for p in result.power if p[0] >= 3]
    assert len({p[1] for p in before}) == 1
    assert after[-1][1] < before[0][1]


def test_event_loop_hard_fails_if_time_does_not_advance(tmp_path, monkeypatch):
    session = SimSession("stuck", "source", 10, 0, 0, 1)
    simulator = ExecutionSimulator(
        scenario((session,)), model(tmp_path),
        (PlannedMove("stuck", "dest", "kv_transfer", 0, ("wan",)),),
    )
    monkeypatch.setattr(simulator, "_advance", lambda _target, _rates: None)

    with pytest.raises(RuntimeError, match="failed to advance"):
        simulator.run()


def test_prediction_preserves_results_without_audit_records(tmp_path):
    sessions = tuple(SimSession(str(i), "source", 10, 25, 0, 40) for i in range(2))
    moves = tuple(
        PlannedMove(str(i), "dest", "kv_transfer", i, ("wan",)) for i in range(2)
    )
    topology, profile = scenario(sessions), model(tmp_path)
    detailed = execute(topology, profile, moves)
    summary = predict(topology, profile, moves)

    assert summary.events == summary.requests == summary.network == ()
    assert summary.sessions == detailed.sessions
    assert summary.power == detailed.power
    assert summary.modeled_source_power_at_deadline_w \
        == detailed.modeled_source_power_at_deadline_w
    assert summary.deadline_met == detailed.deadline_met


def test_catch_up_and_off_wait_for_last_session(tmp_path):
    sessions = (
        SimSession("active", "source", 10, 25, 0, 40, requests=(SimRequest(0, 10, 0),)),
        SimSession("quiet", "source", 10, 25, 0, 40),
    )
    moves = (
        PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),
        PlannedMove("quiet", "dest", "kv_transfer", 1, ("wan",)),
    )
    result = execute(scenario(sessions, final="off"), model(tmp_path, shutdown=2), moves)
    active = next(row for row in result.sessions if row.session_id == "active")
    commits = [row.committed_s for row in result.sessions]
    off_start = next(e.time_s for e in result.events if e.event == "off_start")
    off_done = next(e.time_s for e in result.events if e.event == "off_done")

    assert active.catch_up_start_s is not None
    assert off_start == max(commits)
    assert off_done == pytest.approx(off_start + 2)


def test_shared_source_node_stays_awake_until_every_instance_moves(tmp_path):
    sessions = (
        SimSession("short", "source-a", 10, 25, 0, 40),
        SimSession("long", "source-b", 20, 25, 0, 40),
    )
    topology = ExecutionScenario(
        20, 30, 0, "off", 0,
        (PowerNode("src", 2, True), PowerNode("dst", 2, False)),
        (
            ServingInstance("source-a", ("src",)),
            ServingInstance("source-b", ("src",)),
            ServingInstance("dest-a", ("dst",)),
            ServingInstance("dest-b", ("dst",)),
        ),
        sessions, (NetworkLink("wan", 100),),
    )
    moves = (
        PlannedMove("short", "dest-a", "kv_transfer", 0, ("wan",)),
        PlannedMove("long", "dest-b", "kv_transfer", 1, ("wan",)),
    )
    result = execute(topology, model(tmp_path, tp=1), moves)
    commits = {row.session_id: row.committed_s for row in result.sessions}
    off_start = next(event.time_s for event in result.events if event.event == "off_start")

    assert commits["short"] < commits["long"]
    assert off_start == commits["long"]
    assert next(power for time, power, _ in result.power if time == commits["short"]) > 0


def test_moving_one_session_updates_only_its_gpu_power(tmp_path):
    sessions = (
        SimSession("light", "source-a", 10, 25, 0, 40),
        SimSession("heavy", "source-b", 10, 50, 0, 40),
    )
    topology = ExecutionScenario(
        20, 30, 0, "awake", 0,
        (PowerNode("src", 2, True), PowerNode("dst", 2, False)),
        (
            ServingInstance("source-a", ("src",)),
            ServingInstance("source-b", ("src",)),
            ServingInstance("dest-a", ("dst",)),
            ServingInstance("dest-b", ("dst",)),
        ),
        sessions, (NetworkLink("wan", 100),),
    )
    result = execute(
        topology, model(tmp_path, tp=1),
        (PlannedMove("light", "dest-a", "kv_transfer", 0, ("wan",)),),
    )
    committed = result.sessions[0].committed_s

    assert result.power[0][1] == pytest.approx(50)  # P(.25) + P(.5) = 20 + 30
    assert next(power for time, power, _ in result.power if time == committed) \
        == pytest.approx(40)  # P(0) + P(.5) = 10 + 30


def test_deferred_replay_copies_only_source_local_log(tmp_path):
    sessions = (
        SimSession("external", "source", 10, 0, 0, 100, True, state="cold"),
        SimSession("local", "source", 10, 0, 0, 100, False, state="cold"),
    )
    moves = (
        PlannedMove("external", "dest", "replay_on_request", 0, ("wan",), ("wan",)),
        PlannedMove("local", "dest", "replay_on_request", 1, ("wan",)),
    )
    result = execute(scenario(sessions), model(tmp_path), moves)
    rows = {row.session_id: row for row in result.sessions}
    assert rows["external"].initial_ready_s == 0
    assert rows["local"].initial_ready_s == pytest.approx(1)


def test_deferred_replay_waits_for_an_observed_request(tmp_path):
    session = SimSession(
        "external", "source", 10, 0, 0, 100, True,
        requests=(SimRequest(0.5, 10, 0),), wake_probability=0.9, state="cold",
    )
    move = PlannedMove("external", "dest", "replay_on_request", 0, ("wan",), ("wan",))
    result = execute(scenario((session,)), model(tmp_path), (move,))
    row = result.sessions[0]
    request_start = [e.time_s for e in result.events if e.event == "request_start"]

    assert row.committed_s == pytest.approx(1)
    assert row.wake_start_s == pytest.approx(1)
    assert row.wake_ready_s == pytest.approx(2.1)  # 100 B / 100 B/s + 10 tok / 100 tok/s
    assert request_start == pytest.approx([2.1])
    assert not result.requests_started_by(2)
    assert result.requests_started_by(2.1)


def test_incomplete_moves_remain_visible(tmp_path):
    session = SimSession("slow", "source", 20, 0, 0, 100)
    move = PlannedMove("slow", "dest", "kv_transfer", 0, ("wan",))
    result = execute(scenario((session,), deadline=1, end=1), model(tmp_path), (move,))
    assert len(result.sessions) == 1
    assert result.sessions[0].committed_s is None
    assert result.completed_sessions == 0
    assert result.network[0].end_s is None
    assert result.network[0].transferred_bytes == 100
    assert result.network[0].remaining_bytes == 100


def test_unmeasured_destination_concurrency_queues_instead_of_overlapping(tmp_path):
    sessions = tuple(SimSession(str(i), "source", 10, 0, 0, 1) for i in range(2))
    moves = tuple(
        PlannedMove(str(i), "dest", "replay", i, ("wan",), ("wan",)) for i in range(2)
    )
    result = execute(scenario(sessions), model(tmp_path), moves)
    ready = sorted(row.initial_ready_s for row in result.sessions)
    assert ready == pytest.approx([0.12, 0.22])
    assert sum(event.event == "endpoint_queued" for event in result.events) == 1


def test_initial_kv_uses_snapshot_and_catch_up_uses_only_new_blocks(tmp_path):
    session = SimSession(
        "active", "source", 10, 0, 0, 1, requests=(SimRequest(0, 10, 0),)
    )
    result = execute(
        scenario((session,)), model(tmp_path, setup=1),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )
    assert [(row.phase, row.bytes) for row in result.network] == [
        ("initial", 100), ("catch_up", 100)
    ]


def test_kv_catch_up_resends_a_changed_partial_block(tmp_path):
    session = SimSession(
        "active", "source", 11, 0, 0, 1, requests=(SimRequest(0, 4, 0),)
    )
    result = execute(
        scenario((session,)), model(tmp_path),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )
    assert [(row.phase, row.bytes) for row in result.network] == [
        ("initial", 200), ("catch_up", 100)
    ]


def test_replay_catch_up_processes_only_tokens_after_snapshot(tmp_path):
    session = SimSession(
        "active", "source", 10, 0, 0, 100, False,
        requests=(SimRequest(0, 10, 0),),
    )
    result = execute(
        scenario((session,)), model(tmp_path),
        (PlannedMove("active", "dest", "replay", 0, ("wan",)),),
    )
    assert [(row.phase, row.bytes) for row in result.network] == [
        ("initial", 100), ("catch_up", 100)
    ]


def test_replay_catch_up_rates_new_tokens_at_full_context(tmp_path):
    replay_rate = {"1": [[1, 100], [10, 100], [20, 10], [1000, 10]],
                   "2": [[1, 50], [1000, 50]]}
    session = SimSession(
        "active", "source", 10, 0, 0, 100, False,
        requests=(SimRequest(0, 10, 0),),
    )
    result = execute(
        scenario((session,)), model(tmp_path, replay_rate=replay_rate),
        (PlannedMove("active", "dest", "replay", 0, ("wan",)),),
    )
    catch_up = next(row for row in result.network if row.phase == "catch_up")
    assert result.sessions[0].catch_up_ready_s - catch_up.end_s == pytest.approx(1)


def test_pause_begins_after_active_request_finishes(tmp_path):
    session = SimSession(
        "active", "source", 10, 0, 0, 1, requests=(SimRequest(0, 200, 0),)
    )
    result = execute(
        scenario((session,)), model(tmp_path),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )
    row = result.sessions[0]
    assert row.initial_ready_s == pytest.approx(1)
    assert row.pause_s == row.idle_s == pytest.approx(2)


def test_unmeasured_serving_concurrency_queues_at_one(tmp_path):
    sessions = tuple(
        SimSession(str(i), "source", 10, 0, 0, 1, requests=(SimRequest(0, 100, 0),))
        for i in range(2)
    )
    result = execute(scenario(sessions), model(tmp_path), ())
    assert [row.start_s for row in result.requests] == pytest.approx([0, 1])
    assert sum(event.event == "serving_queued" for event in result.events) == 1


def test_requests_continue_while_controller_is_planning(tmp_path):
    session = SimSession(
        "active", "source", 10, 0, 0, 1, requests=(SimRequest(0, 10, 0),)
    )
    result = execute(
        scenario((session,), controller_delay=1), model(tmp_path),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )
    assert result.requests[0].start_s == 0
    assert result.network[0].start_s == 1
    assert result.network[0].bytes == 200


def test_queued_request_uses_destination_after_commit(tmp_path):
    sessions = (
        SimSession("busy", "source", 10, 0, 0, 1,
                   requests=(SimRequest(0, 1000, 0),)),
        SimSession("moving", "source", 10, 0, 0, 1,
                   requests=(SimRequest(0, 10, 0),)),
    )
    result = execute(
        scenario(sessions), model(tmp_path),
        (PlannedMove("moving", "dest", "kv_transfer", 0, ("wan",)),),
    )
    request = next(row for row in result.requests if row.session_id == "moving")
    assert request.instance_id == "dest"
    assert request.start_s == pytest.approx(2)


def test_external_replay_avoids_source_egress(tmp_path):
    session = SimSession(
        "external", "source", 10, 0, 0, 100, True,
        requests=(SimRequest(0.5, 10, 0),), state="cold",
    )
    links = (NetworkLink("source-egress", 100), NetworkLink("dest-ingress", 100))
    move = PlannedMove(
        "external", "dest", "replay_on_request", 0,
        ("source-egress", "dest-ingress"), ("dest-ingress",),
    )
    result = execute(scenario((session,), links=links), model(tmp_path), (move,))
    assert [row.path for row in result.network] == [("dest-ingress",)]


def test_invalid_move_and_tensor_parallel_mismatch_hard_fail(tmp_path):
    session = SimSession("s", "source", 10, 0, 0, 1)
    with pytest.raises(ValueError, match="different source"):
        execute(
            scenario((session,)), model(tmp_path),
            (PlannedMove("s", "source", "kv_transfer", 0, ("wan",)),),
        )
    with pytest.raises(ValueError, match="tensor parallelism"):
        execute(scenario((session,), tp=1), model(tmp_path), ())


@pytest.mark.parametrize(("state", "method"), (
    ("active", "replay_on_request"), ("cold", "replay"), ("cold", "kv_transfer"),
))
def test_move_method_must_match_gpu_residency(tmp_path, state, method):
    session = SimSession("s", "source", 10, 0, 0, 1, state=state)
    move = PlannedMove("s", "dest", method, 0, ("wan",), ("wan",))

    with pytest.raises(ValueError, match=f"invalid for a {state} session"):
        execute(scenario((session,)), model(tmp_path), (move,))
