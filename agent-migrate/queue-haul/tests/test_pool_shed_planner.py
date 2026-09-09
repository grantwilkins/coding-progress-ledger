from types import SimpleNamespace

import numpy as np
import pytest

from pool_shed_execution import PooledExecution
from pool_shed_planner import phase_profile, plan_admission, planning_grid, project_queues, recovery_prefix


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

    original, bounds = campaign._bounded_lp, []

    def record(cost, matrix, rhs, upper):
        bounds.append(upper.copy())
        return original(cost, matrix, rhs, upper)

    monkeypatch.setattr(campaign, "_bounded_lp", record)
    table = SimpleNamespace(matrix=np.eye(2), capacities=np.array([1e6, 0.]),
                            gains=np.array([1e-10, 1.]), fleet=SimpleNamespace(gpus=1))
    primary = table.gains @ campaign.solve_lp(table, np.ones(2, bool), -table.gains)
    chosen = campaign.solve_lp(table, np.ones(2, bool), np.ones(2), primary)
    assert bounds == pytest.approx(np.array([[1e6], [1e6]]))
    assert table.gains @ chosen == pytest.approx(primary - campaign.PRIMARY_TOL, abs=1e-13)
    assert chosen[1] == 0


@pytest.mark.parametrize('wan,deadline', [(100, 30), (400, 600)])
def test_loaded_coding_secondary_face_solves_without_backend_failure(wan, deadline):
    from pool_shed_campaign import GPUS, initial_admission

    chosen, _, info = initial_admission('coding', 0, GPUS, 8, .25, wan, deadline, False, 'isolated_fastest', 1., 3)
    assert chosen.sum() > 0
    assert info['max_relative_residual'] <= 1e-8


def test_secondary_lp_rejects_a_lost_primary_even_with_optimal_status(monkeypatch):
    import pool_shed_campaign as campaign

    monkeypatch.setattr(campaign, '_bounded_lp', lambda *args: np.zeros(1))
    table = SimpleNamespace(matrix=np.ones((1, 1)), capacities=np.ones(1), gains=np.ones(1), fleet=SimpleNamespace(gpus=1))
    with pytest.raises(RuntimeError, match='preserve the primary'):
        campaign.solve_lp(table, np.array([True]), np.ones(1), primary=.5)


def test_native_scaling_preserves_recorded_coding_source_capacity():
    from pool_shed_campaign import GPUS, POLICIES, initial_admission

    for policy in POLICIES:
        _, _, info = initial_admission('coding', 1, GPUS, 8, .5, 400, 1800, False, policy, 1., 3)
        assert info['max_relative_residual'] <= 1e-8


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


def test_protected_recovery_prefix_preserves_weighted_per_batch_limits():
    engine = SimpleNamespace(now=0., fleet=SimpleNamespace(gpus=1), route=np.array([0, 0]),
        gated=np.ones(2, bool), backlog=np.array([10., 1.]), mass=np.array([.1, 10.]), serving_load=lambda: np.zeros(2))
    edges = np.array([0., 1., 10.1, 19.1, 20.])
    work, end = recovery_prefix(engine, edges)
    assert end == pytest.approx([19.1, 0.])
    assert work.sum() == pytest.approx(11.)
    assert np.all(work <= np.diff(edges) + 1e-12)
    np.testing.assert_array_equal(engine.backlog, [10., 1.])


def test_protected_fixed_gate_and_replay_do_not_double_book_capacity():
    table, timing, calibration = case(30.)
    table.fleet.gpus = 1
    table.fleet.metadata["protect_resident"] = True
    engine = PooledExecution(table, timing, calibration)
    engine.admit(np.array([1., 1., 0., 0.]))
    engine.state[:] = [1, 5]
    engine.remaining[:] = [4., 0.]
    engine.gated[1], engine.backlog[1] = True, .2
    chosen, _, diagnostic = plan_admission(engine, table, "queue_haul")
    assert diagnostic["fixed_obligation_overload"] <= 1e-8
    assert diagnostic["max_relative_residual"] <= 1e-8
    assert np.isfinite(chosen).all()
    np.testing.assert_array_equal(engine.backlog, [0., .2])


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
    assert table.gains @ clear > 0
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


def test_compute_peaks_reserve_short_bursts_without_inflating_network_or_work():
    table, timing, calibration = case(20.)
    table.fleet.metadata["protect_resident"] = True
    edges = np.array([0., 20.])
    profile = phase_profile(table, np.ones(1), 0, 0, 0., edges, np.full((2, 1), .5), timing, calibration)
    assert profile["occupancy"] == pytest.approx([20.])
    assert profile["replay"].sum() < 10.
    assert profile["network"].sum() == pytest.approx(table.fleet.log[0])
    assert profile["service_peak"] == pytest.approx([table.fleet.demand[0] * 20.])
    assert profile["serving"].sum() < profile["service_peak"].sum()


def test_gate_prefix_event_edges_keep_sequential_recovery_and_compute_disjoint():
    engine = SimpleNamespace(now=0., fleet=SimpleNamespace(gpus=1), route=np.array([0]),
        gated=np.ones(1, bool), backlog=np.array([.2]), mass=np.array([1.]), serving_load=lambda: np.zeros(2))
    edges, events = np.array([0., 2.]), []
    recovery_prefix(engine, edges, events)
    edges = np.unique(np.r_[edges, events])
    recovery, end = recovery_prefix(engine, edges)
    compute = (edges[:-1] >= end[0]) * np.diff(edges)
    assert end[0] == pytest.approx(.2)
    assert recovery[0] + compute == pytest.approx(np.diff(edges))


def test_secondary_primary_row_margin_stays_bounded_for_large_gain():
    from pool_shed_campaign import PRIMARY_TOL, solve_lp

    table = SimpleNamespace(matrix=np.ones((1, 1)), capacities=np.ones(1), gains=np.array([100.]), fleet=SimpleNamespace(gpus=1))
    chosen = solve_lp(table, np.ones(1, bool), np.ones(1), primary=100.)
    assert 0 <= 100. - table.gains @ chosen <= 2 * PRIMARY_TOL + 1e-12


@pytest.mark.parametrize("policy", ["queue_haul", "greedy", "kv_only", "isolated_fastest"])
def test_new_admission_cannot_sacrifice_mandatory_route_deadline(policy):
    table, timing, calibration = case(5.)
    table.fleet.metadata["protect_resident"] = True
    table.fleet.t1[:] = 20.
    engine = PooledExecution(table, timing, calibration)
    engine.admit(np.array([1., 0., 0., 0.]))
    chosen, _, info = plan_admission(engine, table, policy, calibration=calibration)
    assert info["mandatory_forecast_finish_s"][0] > table.deadline
    assert chosen[table.route == 0].sum() == 0
    assert info["predicted_shed_fraction"] > 0


@pytest.mark.parametrize("policy", ["queue_haul", "replay_only"])
def test_short_deadline_guard_preserves_original_32_wave_trajectory(policy):
    from pool_shed_calibration import calibration
    from pool_shed_campaign import execute_feedback, forecast

    central = calibration(0)
    table = forecast("coding_long", 0, 66666, 8, .5, 1000, 30)[0]
    result = execute_feedback(table, table, policy, table.timing, central, chunks=32, resolution=1.)
    assert result["shed_fraction"] > .8
    assert result["shed_fraction"] == pytest.approx(result["admitted_shed_fraction"], abs=1e-8)
    assert result["max_relative_residual"] <= 1e-8


def test_zero_nominal_tail_cannot_overlap_mandatory_recovery_prefix():
    table, timing, calibration = case(30.)
    table.fleet.gpus = 1
    table.fleet.metadata["protect_resident"] = True
    engine = PooledExecution(table, timing, calibration)
    engine.admit(np.array([1., 1., 0., 0.]))
    engine.state[:] = [4, 5]
    engine.remaining[:] = [1e-4, 0.]
    engine.phase_replica_seconds[0] = 100.
    engine.gated[1], engine.backlog[1] = True, .2
    _, _, info = plan_admission(engine, table, "greedy")
    assert info["fixed_obligation_overload"] <= 1e-8
    assert info["max_relative_residual"] <= 1e-8


def test_greedy_resource_exhaustion_is_invariant_to_fleet_and_wan_scale():
    from pool_shed_calibration import calibration
    from pool_shed_campaign import execute_feedback, forecast

    central, completed = calibration(0), []
    for gpus in (6400, 64000, 640000):
        table = forecast("measured_pack", 0, gpus, 8, .5, 1000 * gpus / 66666, 60)[0]
        result = execute_feedback(table, table, "greedy", table.timing, central)
        completed.append(result["shed_fraction"])
        assert result["max_relative_residual"] <= 1e-8
        assert all(d["fixed_obligation_overload"] <= 1e-8 for d in result["planning_diagnostics"] if "fixed_obligation_overload" in d)
    assert max(completed) - min(completed) <= 1e-8


def test_secondary_per_batch_cost_is_invariant_to_fleet_scale(monkeypatch):
    import pool_shed_planner as planner

    costs = []
    def capture(matrix, capacity, gains, debt, fleet, greedy):
        costs.append(debt.copy())
        return np.zeros(len(gains))
    monkeypatch.setattr(planner, "_choose", capture)
    for scale in (1., 100.):
        table, timing, calibration = case()
        table.fleet.metadata["protect_resident"] = True
        table.fleet.gpus *= scale
        table.fleet.nodes *= scale
        table.fleet.count *= scale
        table.fleet.kv_capacity *= scale
        table.fleet.gain /= scale
        table.gains /= scale
        table.budgets *= scale
        planner.plan_admission(PooledExecution(table, timing, calibration), table, "queue_haul", iterations=1)
    np.testing.assert_allclose(costs[0], costs[1], rtol=1e-13, atol=1e-13)
