from types import SimpleNamespace

import numpy as np
import pytest

from pool_shed_execution import PooledExecution
from pool_shed_planner import phase_profile, plan_admission, planning_grid, project_queues


def test_secondary_lp_preserves_primary_below_solver_coefficient_cutoff():
    from pool_shed_campaign import PRIMARY_TOL, solve_lp

    table = SimpleNamespace(matrix=np.ones((1, 1)), capacities=np.array([1e6]),
                            gains=np.array([1e-10]), fleet=SimpleNamespace(gpus=1))
    allowed = np.array([True])
    primary = table.gains @ solve_lp(table, allowed, -table.gains)
    chosen = solve_lp(table, allowed, np.ones(1), primary)
    assert table.gains @ chosen == pytest.approx(primary - PRIMARY_TOL, abs=1e-13)
    assert np.all(table.matrix @ chosen <= table.capacities)


def test_lp_removes_exhausted_columns_and_preserves_exact_scaled_bounds(monkeypatch):
    import pool_shed_campaign as campaign

    original, bounds = campaign.linprog, []

    def record(*args, **kwargs):
        bounds.append(kwargs["bounds"].copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign, "linprog", record)
    table = SimpleNamespace(matrix=np.eye(2), capacities=np.array([1e6, 0.]),
                            gains=np.array([1e-10, 1.]), fleet=SimpleNamespace(gpus=1))
    primary = table.gains @ campaign.solve_lp(table, np.ones(2, bool), -table.gains)
    chosen = campaign.solve_lp(table, np.ones(2, bool), np.ones(2), primary)
    assert bounds == pytest.approx(np.array([[[0., 1e6]], [[0., 1e6]]]))
    assert table.gains @ chosen == pytest.approx(primary - campaign.PRIMARY_TOL, abs=1e-13)
    assert chosen[1] == 0


def case(deadline=20.):
    fleet = SimpleNamespace(count=np.array([10.]), context=np.array([100.]), demand=np.array([.05]),
        memory_tokens=np.array([100.]), baseline_kv=0., kv_capacity=1e6, gpus=10, nodes=2,
        gain=np.array([.1]), t1=np.array([4.]), log=np.array([2.]), kv=np.array([100.]),
        metadata={"turn_sequences": [[]], "source_session_rps": 0.})
    timing = {"kappa": 1., "beta": 1., "kv_completion_s": 1., "kv_batch_completion_s": 1., "resident_replay_loss": 1.}
    calibration = {"kv_block_tokens": 1, "kv_block_bytes": 1., "kv_tail_replay_tps": 100., "switch_s": 0., "F": 100., "G": 10.}
    table = SimpleNamespace(fleet=fleet, replay=np.array([[1.], [0.], [1.], [0.]]),
        kv=np.array([[0.], [1.], [0.], [1.]]), route=np.array([0, 0, 1, 1]), load=.5,
        deadline=deadline, endpoint=np.array([100., 100., 200.]), budgets=np.array([200., 200., 200.]),
        timing=timing, nominal_commit=np.array([6., 3., 6., 3.]), gains=np.full(4, .1), fastest=np.array([False]))
    return table, timing, calibration


def test_generated_replay_debt_changes_later_calibrated_compute_time():
    edges, zero = np.array([0., 1., 2.]), np.zeros((2, 2))
    replay = np.array([[10., 1.], [0., 0.]])
    loads, history, _ = project_queues(edges, [.5, .5], [0., 0.], [0., 0.], replay, zero, zero, zero, 10, np.ones(2))
    assert history[:, 0] == pytest.approx([5., 1.])
    assert loads[0] == pytest.approx([.5, .9])
    table, timing, calibration = case()
    forecast_edges = np.array([1., 20.])
    slow = phase_profile(table, np.ones(1), 0, 0, 1., forecast_edges, np.array([[.9], [.5]]), timing, calibration)
    fast = phase_profile(table, np.ones(1), 0, 0, 1., forecast_edges, np.array([[.5], [.5]]), timing, calibration)
    assert slow["finish"] > fast["finish"] + 3.


def test_buffer_recovery_honors_per_batch_cap_and_resident_priority():
    zero, edges = np.zeros((2, 1)), np.array([0., 1.])
    loads, history, debt = project_queues(edges, [0., 0.], [0., 0.], [1., 0.], zero, zero, zero, zero,
                                        10, np.ones(2), [(0, -1, 10., .1)])
    assert loads[0, 0] == pytest.approx(.01)
    assert debt == pytest.approx([.9, 0.])
    _, history, _ = project_queues(edges, [0., 0.], [10., 0.], [1., 0.], zero, zero, zero, zero,
                                  10, np.ones(2), [(0, -1, 10., .1)])
    assert history[0] == pytest.approx([0., 0., 1., 0.])
    loads, _, debt = project_queues(edges, [0., 0.], [0., 0.], [10.1, 0.], zero, zero, zero, zero,
                                    10, np.ones(2), [(0, -1, .1, 1.), (0, -1, 10., 1.)])
    assert debt == pytest.approx([9., 0.])
    assert loads[0, 0] == pytest.approx(.11)


def test_temporal_phase_uses_observed_progress_without_sampled_work():
    table, timing, calibration = case()
    edges, loads = np.array([0., 20.]), np.full((2, 1), .5)
    fresh = phase_profile(table, np.ones(1), 0, 0, 0., edges, loads, timing, calibration, state=1)
    advanced = phase_profile(table, np.ones(1), 0, 0, 0., edges, loads, timing, calibration, state=1, completed_work=2. * np.exp(-.5))
    assert fresh["finish"] - advanced["finish"] == pytest.approx(2.)


def test_future_load_iterations_do_not_change_observed_past_progress(monkeypatch):
    import pool_shed_planner as planner

    original, progress = planner.phase_profile, []

    def record(*args, **kwargs):
        if kwargs.get("state") == 1:
            progress.append(kwargs["completed_work"])
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, "phase_profile", record)
    table, timing, calibration = case()
    engine = PooledExecution(table, timing, calibration)
    engine.admit(np.array([1., 0., 0., 0.]))
    engine.advance(1.)
    planner.plan_admission(engine, table, "queue_haul", calibration=calibration)
    assert len(progress) > 1
    assert len(set(progress)) == 1


def test_initial_plan_is_invariant_to_hidden_execution_timing():
    table, timing, calibration = case()
    first = PooledExecution(table, timing, calibration)
    second = PooledExecution(table, {**timing, "beta": 10., "kappa": .1}, calibration)
    a, when, diagnostics = plan_admission(first, table, "queue_haul", calibration=calibration)
    b, other, _ = plan_admission(second, table, "queue_haul", calibration=calibration)
    assert a == pytest.approx(b)
    assert when == other
    assert diagnostics["max_relative_residual"] <= 1e-8
    assert diagnostics["fixed_obligation_overload"] <= 1e-8
    assert (table.replay + table.kv).T @ a <= table.fleet.count + 1e-8


def test_observed_debt_changes_handoff_feasibility_without_requiring_recovery():
    table, timing, calibration = case(deadline=3.)
    engine = PooledExecution(table, timing, calibration)
    clear, _, _ = plan_admission(engine, table, "queue_haul", calibration=calibration)
    engine.resident_debt[:] = 1000.
    queued, _, _ = plan_admission(engine, table, "queue_haul", calibration=calibration)
    assert table.gains @ clear > .2
    assert queued.sum() == 0


def test_observed_debt_changes_replay_handoff_feasibility():
    table, timing, calibration = case(deadline=8.)
    table.fleet.count[:] = 1.
    engine = PooledExecution(table, timing, calibration)
    clear, _, _ = plan_admission(engine, table, "replay_only", calibration=calibration)
    engine.resident_debt[:] = 1000.
    queued, _, _ = plan_admission(engine, table, "replay_only", calibration=calibration)
    assert clear.sum() == pytest.approx(1., abs=1e-7)
    assert queued.sum() == 0


def test_all_policies_obey_action_masks_and_same_feedback_clock():
    table, timing, calibration = case()
    times = []
    for policy, forbidden in (("replay_only", [1, 3]), ("kv_only", [0, 2]), ("isolated_fastest", [0, 2]), ("greedy", [])):
        engine = PooledExecution(table, timing, calibration)
        chosen, when, _ = plan_admission(engine, table, policy, calibration=calibration)
        assert chosen[forbidden].sum() == 0
        times.append(when)
    assert len(set(times)) == 1


def test_isolated_fastest_refreshes_action_ranking_from_current_debt():
    table, timing, calibration = case()
    table.fleet.t1[:], table.fleet.kv[:] = 2., 400.
    timing.update(kv_completion_s=.1, kv_batch_completion_s=.1)
    engine = PooledExecution(table, timing, calibration)
    clear, _, _ = plan_admission(engine, table, "isolated_fastest", calibration=calibration)
    engine.resident_debt[:] = 1000.
    queued, _, _ = plan_admission(engine, table, "isolated_fastest", calibration=calibration)
    assert clear[[0, 2]].sum() > 0 and clear[[1, 3]].sum() == 0
    assert queued[[1, 3]].sum() > 0 and queued[[0, 2]].sum() == 0


def test_long_horizon_clock_stays_short_early_and_bounded_over_hour():
    grid = planning_grid(0., 3600., np.array([8.]))
    fine = planning_grid(0., 3600., np.array([8.]), .5)
    assert grid[1] == 2.
    assert len(grid) < 16
    assert set(grid).issubset(fine)
    assert planning_grid(grid[1], 3600., np.array([8.])) == pytest.approx(grid[1:])


def test_source_reset_and_absolute_start_change_later_kv_volume():
    table, timing, calibration = case()
    table.fleet.metadata = {"source_session_rps": 1., "sequence_cycle": True,
        "turn_sequences": [[{"context": 100, "prompt": 20, "output": 0}, {"context": 1000, "prompt": 20, "output": 0}]]}
    edges, loads = np.array([0., 20.]), np.full((2, 1), .5)
    first = phase_profile(table, np.ones(1), 1, 0, 0., edges, loads, timing, calibration)
    later = phase_profile(table, np.ones(1), 1, 0, 2., edges, loads, timing, calibration)
    assert later["network"].sum() > first["network"].sum()


def test_no_remaining_capacity_advances_directly_to_deadline():
    table, timing, calibration = case()
    engine = PooledExecution(table, timing, calibration)
    engine.admit(np.array([0., 10., 0., 0.]))
    chosen, when, info = plan_admission(engine, table, "queue_haul", calibration=calibration)
    assert chosen.sum() == 0
    assert when == table.deadline
    assert info["iterations"] == 0


def test_qh_contains_restricted_actions_on_the_same_frozen_temporal_matrix(monkeypatch):
    import pool_shed_planner as planner

    original, frames = planner._choose, []

    def record(*args):
        frames.append(args)
        return original(*args)

    monkeypatch.setattr(planner, "_choose", record)
    table, timing, calibration = case(deadline=5.)
    planner.plan_admission(PooledExecution(table, timing, calibration), table, "queue_haul", calibration=calibration)
    matrix, capacity, gains, debt, fleet, _ = frames[-1]
    optimum = gains @ original(matrix, capacity, gains, debt, fleet, False)
    for action in (0, 1):
        restricted = gains * (np.arange(len(gains)) % 2 == action)
        assert restricted @ original(matrix, capacity, restricted, debt, fleet, False) <= optimum + 1e-8
