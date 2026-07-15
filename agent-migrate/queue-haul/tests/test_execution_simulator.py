"""
Claim:
Execution follows background copy, pause, catch-up, route switch, and final node
state while overlapping transfers share every bottleneck.

Plausible wrong implementations:
- Credit source power when bytes finish instead of when the route switches.
- Serialize simultaneous transfers or exceed a shared link.
- Enter sleep/off before the final source session commits.
- Skip catch-up when a request changes state during background preparation.
"""

import json

import pytest

from profiles import ModelProfile
from simulate import (ExecutionScenario, NetworkLink, PlannedMove, PowerNode, ServingInstance,
                      SimRequest, SimSession, execute, fair_link_rates, step_average)


def model(tmp_path, switch=1, block_s=0, shutdown=2):
    source = {"kind": "measured", "reference": "hand", "valid_range": [1, 1000], "relative_error": 0}
    rate = {"1": [[1, 100], [1000, 100]], "2": [[1, 50], [1000, 50]]}
    raw = {
        "schema": "queue-haul-model-profile-v1", "profile_id": "hand", "status": "fitted",
        "model": "m", "hardware": "h", "precision": "bf16", "tensor_parallel": 1,
        "gpus_per_node": 2, "power_scope": "gpu", "power_window_s": 1,
        "max_ell": 1, "max_parallel_moves": 2,
        "max_parallel_replay": 1, "max_parallel_kv": 1,
        "sources": {k: source for k in ("power", "service", "replay", "kv_transfer", "transitions")},
        "cases": {"central": {
            "F": 100, "G": 100, "power_curve": [[0, 10], [0.5, 30], [1, 40]],
            "prefill_tps": rate, "decode_tps": rate, "replay_tps": rate,
            "kv_transfer": {"block_tokens": 10, "block_bytes": 100, "setup_s": 0,
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


def scenario(sessions, deadline=20, end=30, final="awake"):
    return ExecutionScenario(
        deadline, end, 0, final, 0,
        (PowerNode("src", 2, True), PowerNode("dst", 2, False)),
        (ServingInstance("source", ("src", "src")), ServingInstance("dest", ("dst", "dst"))),
        tuple(sessions), (NetworkLink("wan", 100),),
    )


def test_fair_rates_redistribute_capacity_across_two_bottlenecks():
    rates = fair_link_rates({0: ("a",), 1: ("a", "b"), 2: ("b",)}, {"a": 100, "b": 20})
    assert rates == pytest.approx({0: 90, 1: 10, 2: 10})
    assert rates[0] + rates[1] == pytest.approx(100)
    assert rates[1] + rates[2] == pytest.approx(20)


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
    assert {(row.bytes, row.start_s, row.end_s) for row in result.network} == {
        (100, 0, 2)
    }
    before = [p for p in result.power if p[0] < 3]
    after = [p for p in result.power if p[0] >= 3]
    assert len({p[1] for p in before}) == 1
    assert after[-1][1] < before[0][1]


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


def test_deferred_replay_copies_only_source_local_log(tmp_path):
    sessions = (
        SimSession("external", "source", 10, 0, 0, 100, True),
        SimSession("local", "source", 10, 0, 0, 100, False),
    )
    moves = (
        PlannedMove("external", "dest", "replay_on_request", 0, ("wan",)),
        PlannedMove("local", "dest", "replay_on_request", 1, ("wan",)),
    )
    result = execute(scenario(sessions), model(tmp_path), moves)
    rows = {row.session_id: row for row in result.sessions}
    assert rows["external"].initial_ready_s == 0
    assert rows["local"].initial_ready_s == pytest.approx(1)


def test_deferred_replay_waits_for_an_observed_request(tmp_path):
    session = SimSession(
        "external", "source", 10, 0, 0, 100, True,
        requests=(SimRequest(0.5, 10, 0),), wake_probability=0.9,
    )
    move = PlannedMove("external", "dest", "replay_on_request", 0, ("wan",))
    result = execute(scenario((session,)), model(tmp_path), (move,))
    row = result.sessions[0]
    request_start = [e.time_s for e in result.events if e.event == "request_start"]

    assert row.committed_s == pytest.approx(1)
    assert row.wake_start_s == pytest.approx(1)
    assert row.wake_ready_s == pytest.approx(2.1)  # 100 B / 100 B/s + 10 tok / 100 tok/s
    assert request_start == pytest.approx([2.1])


def test_incomplete_moves_remain_visible(tmp_path):
    session = SimSession("slow", "source", 20, 0, 0, 100)
    move = PlannedMove("slow", "dest", "kv_transfer", 0, ("wan",))
    result = execute(scenario((session,), deadline=1, end=1), model(tmp_path), (move,))
    assert len(result.sessions) == 1
    assert result.sessions[0].committed_s is None
    assert result.completed_sessions == 0
    assert result.network[0].end_s is None


def test_unmeasured_destination_concurrency_queues_instead_of_overlapping(tmp_path):
    sessions = tuple(SimSession(str(i), "source", 10, 0, 0, 1) for i in range(2))
    moves = tuple(PlannedMove(str(i), "dest", "replay", i, ("wan",)) for i in range(2))
    result = execute(scenario(sessions), model(tmp_path), moves)
    ready = sorted(row.initial_ready_s for row in result.sessions)
    assert ready == pytest.approx([0.12, 0.22])
    assert sum(event.event == "endpoint_queued" for event in result.events) == 1
