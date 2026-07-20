"""
Claim:
Execution follows background copy, pause, catch-up, route switch, and final node
state while overlapping transfers share network and destination capacity.

Plausible wrong implementations:
- Credit source power when bytes finish instead of when the route switches.
- Serialize simultaneous transfers or exceed a shared link.
- Recalculate unrelated transfers or miss a transfer connected through a second link.
- Stall when shared rates leave a small floating-point byte remainder.
- Change or retain the wrong GPU slot load when one session moves.
- Enter sleep/off before the final source session commits.
- Skip catch-up when a request changes state during background preparation.
- Transfer nonexistent cold KV or defer replay for GPU-resident active KV.
- Add network and destination KV time even though ingestion is pipelined.
- Charge measured total concurrent action power once for every session.
- Fail to lower cached action power when concurrency decreases.
- Mix source and destination action power while updating concurrency.
- Admit destination KV copies after rather than before their bytes move.
- Apply a destination queue globally instead of once per destination instance.
- Report queue depth or bytes without the newly queued transfer.
- Drop zero-delay completion events exactly at the simulation cutoff.
- Allocate queue audit records during summary-only prediction.
- Apply a background pace cap to the paused final catch-up.
- Transfer an unsealed partial block or commit without reconstructing its tail.
- Omit measured replay completion time or emit duplicate network-start events.
- Treat an unused or completed private link as an active bottleneck.
"""

import json
from dataclasses import replace

import pytest

import simulate
from profiles import ModelProfile
from simulate import (ExecutionScenario, ExecutionSimulator, NetworkLink, PlannedMove,
                      PowerNode, ServingInstance, SimRequest, SimSession, execute,
                      fair_link_rates, step_average)


def model(tmp_path, switch=1, destination_rate=1e12, shutdown=2, setup=0, tp=2,
          replay_rate=None, kv_capacity=10_000, kv_action_power=(0, 0), parallel_kv=1,
          parallel_moves=2, kv_source_action_power=(0, 0), replay_completion=0,
          catch_up_fixed=0):
    source = {"kind": "measured", "reference": "hand", "valid_range": [1, 1000], "relative_error": 0}
    rate = {"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]}
    raw = {
        "schema": "queue-haul-model-profile-v3", "profile_id": "hand", "status": "fitted",
        "model": "m", "hardware": "h", "precision": "bf16", "tensor_parallel": tp,
        "gpus_per_node": 2, "power_scope": "gpu", "power_window_s": 1,
        "max_ell": 1, "kv_capacity_tokens": kv_capacity,
        "max_source_streams": parallel_moves,
        "max_destination_replays": 1,
        "max_destination_kv_streams": parallel_kv,
        "sources": {k: source for k in (
            "power", "service", "capacity", "replay", "kv_transfer", "transitions"
        )},
        "cases": {"central": {
            "F": 100, "G": 100, "power_curve": [[0, 10], [0.5, 30], [1, 40]],
            "prefill_tps": rate, "decode_tps": rate, "replay_tps": replay_rate or rate,
            "replay_completion_s": replay_completion,
            "kv_transfer": {"block_tokens": 10, "block_bytes": 100, "setup_s": setup,
                            "destination_bytes_per_s": destination_rate,
                            "initial_completion_s": 0, "catch_up_fixed_s": catch_up_fixed,
                            "tail_replay_tps": 100},
            "switch_s": switch, "sleep_power_delta_w": -8, "sleep_s": 1,
            "shutdown_s": shutdown,
            "action_power_w": {
                "replay": {"1": [0, 0], "2": [0, 0]},
                "kv_transfer": {"1": [kv_source_action_power[0], kv_action_power[0]],
                                "2": [kv_source_action_power[1], kv_action_power[1]]},
                "replay_on_request": {"1": [0, 0], "2": [0, 0]},
                "catch_up": {"1": [0, 0], "2": [0, 0]},
                "sleep": {"1": [0, 0]}, "off": {"1": [0, 0]},
            },
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
        2 if final == "off" else None,
    )


def test_fair_rates_redistribute_capacity_across_two_bottlenecks():
    rates = fair_link_rates({0: ("a",), 1: ("a", "b"), 2: ("b",)}, {"a": 100, "b": 20})
    assert rates == pytest.approx({0: 90, 1: 10, 2: 10})
    assert rates[0] + rates[1] == pytest.approx(100)
    assert rates[1] + rates[2] == pytest.approx(20)


def test_fair_rates_ignore_unused_links():
    paths = {0: ("a",), 1: ("a", "b"), 2: ("b",)}
    links = {"a": 100, "b": 20}

    assert fair_link_rates(paths, {**links, "unused": 1}) == fair_link_rates(paths, links)


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
        physical = set("abc")
        calls.append((tuple(sorted(tuple(link for link in path if link in physical)
                                  for path in active.values())),
                      tuple(sorted(link for link in links if link in physical))))
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
    result = execute(scenario(sessions), model(tmp_path, parallel_kv=2), moves)

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


def test_destination_kv_queue_blocks_bytes_until_slot_free(tmp_path):
    sessions = tuple(SimSession(str(i), "source", 10, 0, 0, 1) for i in range(2))
    moves = tuple(
        PlannedMove(str(i), "dest", "kv_transfer", i, ("wan",)) for i in range(2)
    )
    result = execute(
        scenario(sessions, links=(NetworkLink("wan", 1000),)),
        model(tmp_path, switch=0, destination_rate=100), moves,
    )

    assert [(row.start_s, row.end_s) for row in result.network] == [(0, 1), (1, 2)]
    assert [(row.arrival_s, row.start_s, row.end_s, row.depth_at_arrival,
             row.bytes_at_arrival) for row in result.queues] == [
        (0, 0, 1, 0, 0), (0, 1, 2, 1, 100),
    ]


def test_destination_kv_queue_refills_concurrency_two(tmp_path):
    sessions = tuple(SimSession(str(i), "source", 10, 0, 0, 1) for i in range(3))
    moves = tuple(
        PlannedMove(str(i), "dest", "kv_transfer", i, ("wan",)) for i in range(3)
    )
    result = execute(
        scenario(sessions, links=(NetworkLink("wan", 1000),)),
        model(tmp_path, switch=0, destination_rate=100, parallel_kv=2, parallel_moves=3),
        moves,
    )

    assert [row.initial_ready_s for row in result.sessions] == pytest.approx([2, 2, 3])
    assert [(row.session_id, row.start_s, row.depth_at_arrival, row.bytes_at_arrival)
            for row in result.queues] == [
        ("0", 0, 0, 0), ("1", 0, 0, 0), ("2", 2, 1, 100),
    ]


def test_kv_catch_up_waits_behind_an_earlier_destination_copy(tmp_path):
    sessions = (
        SimSession("growing", "source", 10, 0, 0, 1,
                   requests=(SimRequest(0, 10, 0),)),
        SimSession("waiting", "source", 10, 0, 0, 1),
    )
    moves = tuple(
        PlannedMove(session.session_id, "dest", "kv_transfer", i, ("wan",))
        for i, session in enumerate(sessions)
    )
    result = execute(
        scenario(sessions, links=(NetworkLink("wan", 1000),)),
        model(tmp_path, switch=0, destination_rate=100), moves,
    )

    append = next(row for row in result.queues if row.phase.startswith("append"))
    assert append.session_id == "growing"
    assert (append.arrival_s, append.start_s, append.depth_at_arrival) \
        == pytest.approx((1, 2, 1))


def test_configured_parallel_kv_copy_shares_destination_capacity(tmp_path):
    sessions = tuple(SimSession(str(i), "source", 10, 0, 0, 1) for i in range(2))
    moves = tuple(
        PlannedMove(str(i), "dest", "kv_transfer", i, ("wan",)) for i in range(2)
    )
    result = execute(
        scenario(sessions, links=(NetworkLink("wan", 1000),)),
        model(tmp_path, switch=0, destination_rate=100, parallel_kv=2), moves,
    )

    assert [row.initial_ready_s for row in result.sessions] == pytest.approx([2, 2])


def test_action_power_is_total_for_concurrent_actions(tmp_path):
    sessions = tuple(SimSession(str(i), "source", 10, 0, 0, 1) for i in range(2))
    moves = tuple(
        PlannedMove(str(i), "dest", "kv_transfer", i, ("wan",)) for i in range(2)
    )
    result = execute(
        scenario(sessions), model(tmp_path, kv_action_power=(10, 15), parallel_kv=2), moves,
    )

    baseline = result.power[0][2]
    assert max(point[2] for point in result.power) - baseline == pytest.approx(15)


def test_action_power_tracks_concurrency_per_resource_on_start_and_stop(tmp_path):
    topology = ExecutionScenario(
        10, 20, 0, "awake", 0,
        (PowerNode("src", 1, True), PowerNode("dst", 2, False)),
        (ServingInstance("source", ("src",)), ServingInstance("d0", ("dst",)),
         ServingInstance("d1", ("dst",))),
        (), (NetworkLink("wan", 100),),
    )
    simulator = ExecutionSimulator(topology, model(
        tmp_path, tp=1, kv_action_power=(10, 15), kv_source_action_power=(3, 5),
    ), ())

    simulator._start_action("a", "kv_transfer", instance="d0")
    assert simulator._action_power(False) == pytest.approx(10)
    simulator._start_action("source", "kv_transfer", instance="source")
    assert simulator._action_power(True) == pytest.approx(3)
    assert simulator._action_power(False) == pytest.approx(10)
    simulator._start_action("b", "kv_transfer", instance="d0")
    assert simulator._action_power(False) == pytest.approx(15)
    simulator._start_action("c", "kv_transfer", instance="d1")
    assert simulator._action_power(False) == pytest.approx(25)
    simulator._stop_action("b")
    assert simulator._action_power(False) == pytest.approx(20)
    simulator._stop_action("a")
    simulator._stop_action("c")
    simulator._stop_action("source")
    assert simulator._action_power(True) == 0
    assert simulator._action_power(False) == 0


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
    simulator = ExecutionSimulator(topology, profile, moves, detailed=False)
    summary = simulator.run()

    assert summary.events == summary.requests == summary.network == summary.queues == ()
    assert simulator.queues == []
    assert simulator.kv_records == {}
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
    assert result.migration_makespan_s == off_start
    assert result.final_state_ready_s == result.makespan_s == off_done


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
        2,
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


def test_deferred_replay_leaves_source_local_log_until_wake(tmp_path):
    sessions = (
        SimSession("external", "source", 10, 0, 0, 100, state="cold"),
        SimSession("local", "source", 10, 0, 0, 100, state="cold"),
    )
    moves = (
        PlannedMove("external", "dest", "replay_on_request", 0, ("wan",)),
        PlannedMove("local", "dest", "replay_on_request", 1, ("wan",)),
    )
    result = execute(scenario(sessions), model(tmp_path), moves)
    rows = {row.session_id: row for row in result.sessions}
    assert rows["external"].initial_ready_s == 0
    assert rows["local"].initial_ready_s == 0
    assert not result.network


def test_deferred_replay_waits_for_an_observed_request(tmp_path):
    session = SimSession(
        "external", "source", 10, 0, 0, 100,
        requests=(SimRequest(0.5, 10, 0),), wake_probability=0.9, state="cold",
    )
    move = PlannedMove("external", "dest", "replay_on_request", 0, ("wan",))
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


def test_zero_delay_completion_at_end_is_processed(tmp_path):
    session = SimSession("exact", "source", 10, 0, 0, 1)
    result = execute(
        scenario((session,), deadline=1, end=1), model(tmp_path, switch=0),
        (PlannedMove("exact", "dest", "kv_transfer", 0, ("wan",)),),
    )

    assert result.network[0].end_s == 1
    assert result.queues[0].end_s == 1
    assert result.sessions[0].initial_ready_s == result.sessions[0].committed_s == 1


def test_unmeasured_destination_concurrency_queues_instead_of_overlapping(tmp_path):
    sessions = tuple(SimSession(str(i), "source", 10, 0, 0, 1) for i in range(2))
    moves = tuple(
        PlannedMove(str(i), "dest", "replay", i, ("wan",)) for i in range(2)
    )
    result = execute(scenario(sessions), model(tmp_path), moves)
    ready = sorted(row.initial_ready_s for row in result.sessions)
    assert ready == pytest.approx([0.12, 0.22])
    assert sum(event.event == "endpoint_queued" for event in result.events) == 1


def test_replay_includes_measured_completion_once(tmp_path):
    result = execute(
        scenario((SimSession("active", "source", 10, 0, 0, 100),)),
        model(tmp_path, switch=0, replay_completion=.4),
        (PlannedMove("active", "dest", "replay", 0, ("wan",)),),
    )

    assert result.sessions[0].initial_ready_s == pytest.approx(1.5)


def test_each_network_flow_has_one_start_event(tmp_path):
    result = execute(
        scenario((SimSession("active", "source", 10, 0, 0, 1),)),
        model(tmp_path, switch=0),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )

    assert sum(event.event == "network_start" for event in result.events) == 1


def test_initial_kv_uses_snapshot_and_catch_up_uses_only_new_blocks(tmp_path):
    session = SimSession(
        "active", "source", 10, 0, 0, 1, requests=(SimRequest(0, 10, 0),)
    )
    result = execute(
        scenario((session,)), model(tmp_path, setup=1),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )
    assert [(row.phase, row.bytes) for row in result.network] == [
        ("initial", 100), ("append_initial", 100)
    ]


def test_kv_catch_up_replays_a_changed_partial_tail_without_network_bytes(tmp_path):
    session = SimSession(
        "active", "source", 11, 0, 0, 1, requests=(SimRequest(0, 4, 0),)
    )
    result = execute(
        scenario((session,)), model(tmp_path),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )
    row = result.sessions[0]
    assert [(flow.phase, flow.bytes) for flow in result.network] == [("initial", 100)]
    assert row.catch_up_ready_s - row.catch_up_start_s == pytest.approx(.05)


def test_static_partial_tail_is_reconstructed_without_wan_bytes(tmp_path):
    result = execute(
        scenario((SimSession("active", "source", 11, 0, 0, 1),)),
        model(tmp_path),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )

    row = result.sessions[0]
    assert [(flow.phase, flow.bytes) for flow in result.network] == [("initial", 100)]
    assert row.catch_up_ready_s - row.catch_up_start_s == pytest.approx(.01)


@pytest.mark.parametrize(("growth", "blocks"), ((3, 0), (9, 1), (13, 1), (29, 3)))
def test_append_copy_adds_only_newly_completed_blocks(tmp_path, growth, blocks):
    session = SimSession(
        "active", "source", 11, 0, 0, 1,
        requests=(SimRequest(0, growth, 0),),
    )
    result = execute(
        scenario((session,)), model(tmp_path),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )

    assert sum(
        flow.bytes for flow in result.network if flow.phase != "initial"
    ) == blocks * 100


def test_move_rate_limit_is_a_real_shared_flow_bottleneck(tmp_path):
    session = SimSession("active", "source", 10, 0, 0, 1)
    result = execute(
        scenario((session,)),
        model(tmp_path, switch=0),
        (PlannedMove(
            "active", "dest", "kv_transfer", 0, ("wan",),
            rate_limit_bytes_per_s=50,
        ),),
    )

    assert result.sessions[0].initial_ready_s == pytest.approx(2)


def test_final_catch_up_is_not_background_paced(tmp_path):
    session = SimSession(
        "active", "source", 10, 0, 0, 1,
        requests=(SimRequest(2, 10, 0),),
    )
    result = execute(
        scenario((session,)),
        model(tmp_path, switch=0),
        (PlannedMove(
            "active", "dest", "kv_transfer", 0, ("wan",),
            rate_limit_bytes_per_s=50, quiesce_s=2,
        ),),
    )
    catch_up = next(row for row in result.network if row.phase == "catch_up")

    assert catch_up.end_s - catch_up.start_s == pytest.approx(1)


def test_replay_catch_up_processes_only_tokens_after_snapshot(tmp_path):
    session = SimSession(
        "active", "source", 10, 0, 0, 100,
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
        "active", "source", 10, 0, 0, 100,
        requests=(SimRequest(0, 10, 0),),
    )
    result = execute(
        scenario((session,)), model(tmp_path, replay_rate=replay_rate),
        (PlannedMove("active", "dest", "replay", 0, ("wan",)),),
    )
    catch_up = next(row for row in result.network if row.phase == "catch_up")
    assert result.sessions[0].catch_up_ready_s - catch_up.end_s == pytest.approx(1)


def test_pause_begins_at_quiescence_before_active_request_finishes(tmp_path):
    session = SimSession(
        "active", "source", 10, 0, 0, 1, requests=(SimRequest(0, 200, 0),)
    )
    result = execute(
        scenario((session,)), model(tmp_path),
        (PlannedMove("active", "dest", "kv_transfer", 0, ("wan",)),),
    )
    row = result.sessions[0]
    assert row.initial_ready_s == pytest.approx(1)
    assert row.pause_s == pytest.approx(1)
    assert row.idle_s == pytest.approx(2)


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


def test_source_local_replay_uses_source_egress(tmp_path):
    session = SimSession(
        "external", "source", 10, 0, 0, 100,
        requests=(SimRequest(0.5, 10, 0),), state="cold",
    )
    links = (NetworkLink("source-egress", 100), NetworkLink("dest-ingress", 100))
    move = PlannedMove(
        "external", "dest", "replay_on_request", 0,
        ("source-egress", "dest-ingress"),
    )
    result = execute(scenario((session,), links=links), model(tmp_path), (move,))
    assert [row.path for row in result.network] == [("source-egress", "dest-ingress")]


def test_invalid_move_and_tensor_parallel_mismatch_hard_fail(tmp_path):
    session = SimSession("s", "source", 10, 0, 0, 1)
    with pytest.raises(ValueError, match="different source"):
        execute(
            scenario((session,)), model(tmp_path),
            (PlannedMove("s", "source", "kv_transfer", 0, ("wan",)),),
        )
    with pytest.raises(ValueError, match="tensor parallelism"):
        execute(scenario((session,), tp=1), model(tmp_path), ())


def test_execution_rejects_source_or_destination_kv_overcommit(tmp_path):
    sessions = (SimSession("a", "source-a", 6, 0, 0, 1),
                SimSession("b", "source-b", 6, 0, 0, 1))
    topology = ExecutionScenario(
        20, 30, 0, "awake", 0,
        (PowerNode("src", 2, True), PowerNode("dst", 1, False)),
        (ServingInstance("source-a", ("src",)), ServingInstance("source-b", ("src",)),
         ServingInstance("dest", ("dst",))),
        sessions, (NetworkLink("wan", 100),),
    )
    moves = tuple(PlannedMove(s.session_id, "dest", "kv_transfer", i, ("wan",))
                  for i, s in enumerate(sessions))

    with pytest.raises(ValueError, match="resident KV capacity"):
        execute(topology, model(tmp_path, tp=1, kv_capacity=10), moves)

    cold = tuple(SimSession(s.session_id, s.source_instance, 60, 0, 0, 1, state="cold")
                 for s in sessions)
    execute(replace(topology, sessions=cold), model(tmp_path, tp=1, kv_capacity=10), ())

    growing = SimSession(
        "growing", "source", 10, 0, 0, 1, requests=(SimRequest(0, 6, 0),)
    )
    with pytest.raises(RuntimeError, match="exceeded resident KV capacity"):
        execute(scenario((growing,)), model(tmp_path, kv_capacity=15), ())


@pytest.mark.parametrize(("state", "method"), (
    ("active", "replay_on_request"), ("cold", "replay"), ("cold", "kv_transfer"),
))
def test_move_method_must_match_gpu_residency(tmp_path, state, method):
    session = SimSession("s", "source", 10, 0, 0, 1, state=state)
    move = PlannedMove("s", "dest", method, 0, ("wan",))

    with pytest.raises(ValueError, match=f"invalid for a {state} session"):
        execute(scenario((session,)), model(tmp_path), (move,))
