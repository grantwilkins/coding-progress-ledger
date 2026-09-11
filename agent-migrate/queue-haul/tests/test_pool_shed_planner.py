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

    def record(cost, matrix, rhs, upper, certificate=None):
        bounds.append(upper.copy())
        return original(cost, matrix, rhs, upper, certificate)

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


def test_captured_primary_lp_matches_feasible_dual_certificate():
    from pathlib import Path
    from pool_shed_campaign import _bounded_lp

    # Cell 925, QH at 86.583414979 s: internal rescaling produced an invalid x=-3.
    data = np.load(Path(__file__).parent / 'fixtures/pool_shed_primary925.npz')
    cost, matrix, rhs, upper = (data[k] for k in ('cost', 'matrix', 'rhs', 'upper'))
    chosen = _bounded_lp(cost, matrix, rhs, upper)
    row_dual, col_dual = data['row_dual'], data['col_dual']
    assert np.max(row_dual) <= 0
    assert matrix.T @ row_dual + col_dual == pytest.approx(cost, abs=1e-12)
    dual_bound = rhs @ row_dual + upper @ np.minimum(col_dual, 0)
    assert dual_bound == pytest.approx(-1.0000519451158267, abs=1e-12)
    assert cost @ chosen == pytest.approx(dual_bound, abs=1e-10)
    assert chosen.min() >= -1e-10 and np.max(chosen - upper) <= 1e-10
    assert np.max(matrix @ chosen - rhs) <= 1e-10
    replicas = chosen * data['gpus'] / data['column_scale']
    residual = (data['original_matrix'][:, data['ids']] @ replicas - data['capacities']) / np.maximum(data['capacities'], 1)
    assert residual.max() <= 1e-8


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


def test_fixed_clock_keeps_replay_control_independent_of_kv_costs():
    table, timing, calibration = case()
    table.fleet.metadata["planning_reference_s"] = 4.
    original = plan_admission(PooledExecution(table, timing, calibration), table, "replay_only", calibration=calibration)
    table.nominal_commit[1::2] = .01
    table.fleet.kv *= .01
    changed = plan_admission(PooledExecution(table, timing, calibration), table, "replay_only", calibration=calibration)
    np.testing.assert_array_equal(original[0], changed[0])
    assert original[1] == changed[1] == .5
    assert original[2] == changed[2]


def test_shared_continuation_reserves_replay_without_turning_resident_recovery_into_new_arrivals():
    from pool_shed_planner import mandatory_profile
    table, timing, calibration = case()
    table.fleet.metadata["paced_source"] = True
    engine = PooledExecution(table, timing, calibration)
    engine.admit([10., 0., 0., 0.])
    engine.advance(1.)
    assert engine.resident_debt[0] > 0
    fixed, _ = mandatory_profile(engine, table, timing, calibration, np.array([1., 5., 20.]))
    assert fixed["replay"].sum() > 0 and fixed["recovery"].sum() > 0
    assert fixed["buffers"].sum() == pytest.approx(0., abs=1e-8)
    assert engine.now == 1.


def test_late_buffers_cannot_borrow_earlier_idle_service(monkeypatch):
    import pool_shed_planner as planner
    table, timing, calibration = case(10.)
    table.fleet.metadata["paced_source"] = True
    table.fleet.demand[:] = 0.
    monkeypatch.setattr(planner, "planning_grid", lambda *args: np.array([0., 9., 10.]))
    def late_buffer(*args, **kwargs):
        profile = {name: np.zeros(2) for name in ("replay", "kv", "network", "application", "serving", "buffers", "recovery", "occupancy", "service_peak")}
        profile["buffers"][-1], profile["finish"] = 20., 9.
        return profile
    monkeypatch.setattr(planner, "phase_profile", late_buffer)
    _, _, audit = plan_admission(PooledExecution(table, timing, calibration), table, "replay_only", calibration=calibration)
    assert audit["predicted_shed_fraction"] == pytest.approx(.05)


def test_protected_recovery_prefix_preserves_weighted_per_batch_limits():
    engine = SimpleNamespace(now=0., gpus=1, fleet=SimpleNamespace(gpus=1), route=np.array([0, 0]),
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


def test_affine_queue_profile_preserves_compute_and_recovery_order_inside_bins():
    from pool_shed_planner import replica_queue_profile
    profile = replica_queue_profile(np.array([0., 2., 4., 6., 9.]), [(0., 1.), (3., 4.)],
                                    5., 1., .5, .25, 1.)
    assert profile["resident_debt"] == pytest.approx([0., .5, 0., 0.])
    assert profile["buffer_debt"] == pytest.approx([0., 0., .75, 0.])
    assert profile["recovery"].sum() == pytest.approx(2.)


def test_affine_profile_matches_execution_without_idle_gpu_compensation():
    table, timing, calibration = case()
    table.fleet.metadata["resident_affinity"] = True
    edges = np.array([0., 1., 3., 5., 10., 20.])
    profile = phase_profile(table, np.ones(1), 0, 0, 0., edges, np.full((2, 5), .99), timing, calibration)
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1., 0., 0., 0.])
    for k, until in enumerate(edges[1:]):
        engine.advance(until)
        assert engine.resident_debt[0] == pytest.approx(profile["resident_debt"][k], abs=1e-9)
        assert engine.backlog[0] == pytest.approx(profile["buffer_debt"][k], abs=1e-9)
    assert profile["finish"] == pytest.approx(engine.result()["last_completion_s"])


@pytest.mark.parametrize("policy", ["queue_haul", "greedy", "replay_only", "kv_only", "isolated_fastest"])
def test_affine_planner_reserves_disjoint_replicas_for_every_policy(policy):
    table, timing, calibration = case()
    table.fleet.metadata.update(resident_affinity=True, destination_gpus=1)
    engine = PooledExecution(table, timing, calibration)
    chosen, _, audit = plan_admission(engine, table, policy, calibration=calibration)
    footprint = (table.replay.sum(1) > 0).astype(int) + (table.kv.sum(1) > 0)
    for route in (0, 1):
        assert chosen[table.route == route] @ footprint[table.route == route] <= 1 + 1e-8
    assert chosen.sum() > 0 and audit["max_relative_residual"] <= 1e-8
    engine.admit(chosen)


def test_affine_planner_rejects_local_overload_and_preserves_completed_replica_reservations():
    table, timing, calibration = case()
    table.fleet.metadata.update(resident_affinity=True, destination_gpus=1)
    table.fleet.demand[:] = .3
    table.replay[[0, 2], 0] = 2.
    engine = PooledExecution(table, timing, calibration)
    chosen, _, _ = plan_admission(engine, table, "queue_haul", calibration=calibration)
    assert chosen[[0, 2]].sum() == 0.
    engine.admit([0., 1., 0., 0.])
    engine.advance(5.)
    assert engine.state[0] == 6
    chosen, _, _ = plan_admission(engine, table, "queue_haul", calibration=calibration)
    assert chosen[:2].sum() == 0. and chosen[3] > 0
    engine.admit(chosen)


def test_local_recovery_admission_rejects_handoff_that_leaves_its_own_queue():
    table, timing, calibration = case(deadline=8.)
    table.fleet.metadata["resident_affinity"] = True
    raw, _, _ = plan_admission(PooledExecution(table, timing, calibration), table, "replay_only", calibration=calibration)
    assert raw.sum() > 0
    table.fleet.metadata["require_local_recovery"] = True
    for policy in ("queue_haul", "greedy", "replay_only", "kv_only", "isolated_fastest"):
        engine = PooledExecution(table, timing, calibration)
        chosen, _, audit = plan_admission(engine, table, policy, calibration=calibration)
        assert chosen[[0, 2]].sum() == 0
        assert chosen[[1, 3]].sum() > 0 if policy != "replay_only" else chosen.sum() == 0
        assert "forecast criterion" in audit["planning_scope"]
    del table.fleet.metadata["resident_affinity"]
    with pytest.raises(ValueError, match="requires resident affinity"):
        PooledExecution(table, timing, calibration)


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
    for policy, forbidden in (("replay_only", [1, 3]), ("kv_only", [0, 2]), ("isolated_fastest", [0, 2]), ("greedy", []), ("greedy_priced", [])):
        engine = PooledExecution(table, timing, calibration)
        chosen, when, audit = plan_admission(engine, table, policy, calibration=calibration)
        assert chosen[forbidden].sum() == 0
        if policy == "greedy_priced":
            assert len(audit["pricing_certificates"]) == audit["iterations"]
            for certificate in audit["pricing_certificates"]:
                assert certificate["converged"] and 0 <= certificate["absolute_gap"] <= .001
                assert 0 <= certificate["objective"] <= certificate["upper_bound"]
        else:
            assert "pricing_certificates" not in audit
        times.append(when)
    assert len(set(times)) == 1


def test_priced_greedy_wrapper_escapes_the_resource_trap_without_a_generic_solver(monkeypatch):
    import highspy
    import scipy.optimize
    import pool_shed_campaign as campaign
    from pool_shed_planner import _choose, _choose_priced

    def forbidden(*args, **kwargs):
        raise AssertionError("generic solver called")

    for module, name in ((highspy, "Highs"), (scipy.optimize, "linprog"),
                         (scipy.optimize, "minimize"), (campaign, "solve_lp"), (campaign, "_bounded_lp")):
        monkeypatch.setattr(module, name, forbidden)
    matrix = np.column_stack((np.ones(16), np.eye(16)))
    capacity, gains, debt, fleet = np.ones(16), np.r_[1.001, np.ones(16)], np.zeros(17), SimpleNamespace(gpus=1)
    old = _choose(matrix, capacity, gains, debt, fleet, True)
    chosen, certificate = _choose_priced(matrix, capacity, gains, debt, fleet)
    assert gains @ old == pytest.approx(1.001)
    assert chosen == pytest.approx(np.r_[0., np.ones(16)])
    assert matrix @ chosen == pytest.approx(capacity)
    assert certificate["objective"] == pytest.approx(16.)
    assert certificate["upper_bound"] >= 16.
    assert certificate["converged"] and certificate["relative_gap"] <= .001


@pytest.mark.parametrize("bound,error", [(16., "did not close its admission gap"), (1., "failed its certificate")])
def test_priced_greedy_wrapper_rejects_a_false_convergence_certificate(monkeypatch, bound, error):
    import pool_shed_priced_greedy as pricing
    from pool_shed_planner import _choose_priced

    def false_certificate(matrix, capacity, gains, incumbent, debt, **kwargs):
        return incumbent.copy(), dict(objective=float(gains @ incumbent), upper_bound=bound,
            absolute_gap=0., relative_gap=0., iterations=0, converged=True)

    monkeypatch.setattr(pricing, "priced_greedy", false_certificate)
    with pytest.raises(RuntimeError, match=error):
        _choose_priced(np.column_stack((np.ones(16), np.eye(16))), np.ones(16),
                       np.r_[1.001, np.ones(16)], np.zeros(17), SimpleNamespace(gpus=1))


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
    engine = SimpleNamespace(now=0., gpus=1, fleet=SimpleNamespace(gpus=1), route=np.array([0]),
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
@pytest.mark.parametrize("old_mass", [1., 1e-6])
def test_late_migration_keeps_its_resources_without_vetoing_feasible_new_work(policy, old_mass):
    table, timing, calibration = case(5.)
    table.fleet.metadata["protect_resident"] = True
    table.fleet.t1[:] = 20.
    engine = PooledExecution(table, timing, calibration)
    engine.loads[1] = 1.
    engine.admit(np.array([old_mass, 0., 0., 0.]))
    while engine.now < table.deadline:
        chosen, until, info = plan_admission(engine, table, policy, calibration=calibration)
        if engine.now == 0:
            assert info["mandatory_forecast_finish_s"][0] > table.deadline
            assert info["predicted_shed_fraction"] > 0
        assert chosen[table.route == 1].sum() == 0
        assert info["max_relative_residual"] <= 1e-8 and info.get("fixed_obligation_overload", 0.) <= 1e-8
        engine.admit(chosen)
        engine.advance(until)
    assert engine.mass[0] == old_mass and engine.state[0] < 6
    assert engine.result()["shed_fraction"] > 0
    assert not engine.resident_generated.any()


@pytest.mark.parametrize("policy", ["queue_haul", "replay_only"])
def test_short_deadline_handoff_preserves_source_and_destination_queue_contract(policy):
    from pool_shed_calibration import calibration
    from pool_shed_campaign import execute_feedback, forecast

    central = calibration(0)
    table = forecast("coding_long", 0, 66666, 8, .5, 1000, 30)[0]
    result = execute_feedback(table, table, policy, table.timing, central, chunks=32, resolution=1.)
    assert result["shed_fraction"] > .8
    assert result["shed_fraction"] <= result["admitted_shed_fraction"] + 1e-8
    assert max(result["resident_debt_generated_work_s"]) > 0.
    assert np.array(result["resident_debt_generated_work_s"]) - result["resident_debt_recovered_work_s"] == pytest.approx(result["pending_resident_debt_work_s"], abs=1e-7)
    assert not result["resident_latency_validated"]
    assert all(event["completion_s"] <= table.deadline for event in result["completion_events"])
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
    for scale in (1, 100):
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


@pytest.mark.parametrize("scale", [1., 1e6])
@pytest.mark.parametrize("columns", [2, 3])
def test_greedy_unused_dominated_option_cannot_rescale_secondary_cost(scale, columns):
    from pool_shed_planner import _choose

    matrix = np.array([[1., 2., 2.]])[:, :columns]
    gains = np.array([1., 1., 1e-6])[:columns] / scale
    work = np.array([10., 0., 100.])[:columns]
    chosen = _choose(matrix, np.array([scale]), gains, work, SimpleNamespace(gpus=scale), True)
    np.testing.assert_allclose(chosen, np.r_[scale, np.zeros(columns - 1)])
    assert gains @ chosen == pytest.approx(1.)


@pytest.mark.parametrize("second_gain,second_work,winner", [(2., 5., 1), (2., 20., 0), (2. - 1e-8, 0., 0)])
def test_greedy_uses_work_per_gain_only_to_break_equal_primary_density(second_gain, second_work, winner):
    from pool_shed_planner import _choose

    chosen = _choose(np.array([[1., 2.]]), np.ones(1), np.array([1., second_gain]),
                     np.array([10., second_work]), SimpleNamespace(gpus=1), True)
    assert np.flatnonzero(chosen).tolist() == [winner]


def test_mandatory_profile_counts_resident_occupancy_once():
    from pool_shed_planner import mandatory_profile

    table, timing, calibration = case()
    table.fleet.metadata["protect_resident"] = True
    fixed, finish = mandatory_profile(PooledExecution(table, timing, calibration), table, timing, calibration, np.array([0., 1., 3.]))
    np.testing.assert_allclose(fixed["occupancy"], [[5., 10.], [5., 10.]])
    assert not fixed["serving"].any() and not fixed["service_peak"].any()
    assert finish.tolist() == [0., 0.]


def test_mandatory_forecast_marks_deadline_truncated_compute_unfinished():
    from pool_shed_planner import mandatory_profile

    table, timing, calibration = case(1.)
    table.fleet.metadata["protect_resident"] = True
    engine = PooledExecution(table, timing, calibration)
    engine.admit(np.array([1., 0., 0., 0.]))
    engine.state[:] = 1
    fixed, finish = mandatory_profile(engine, table, timing, calibration, np.array([0., 1.]))
    assert finish.tolist() == [table.deadline + 1., 0.]
    assert fixed["occupancy"][0, 0] > table.fleet.gpus * table.load


def test_mandatory_continuation_reserves_later_tails_sharing_current_transfers(monkeypatch):
    import pool_shed_execution as execution
    from pool_shed_planner import mandatory_profile

    table, timing, calibration = case()
    table.fleet.metadata["protect_resident"] = True
    timing["beta"] = 0.
    engine = PooledExecution(table, timing, calibration)
    engine.admit(np.array([2., 2., 0., 0.]))
    engine.state[:], engine.phase_nominal_work[:] = [1, 0], [4., 0.]
    monkeypatch.setattr(execution, "catchup", lambda *args, **kwargs: (100., 1.))
    edges = np.array([0., .5, 1., 2., 3., 10., 20.])
    fixed, finish = mandatory_profile(engine, table, timing, calibration, edges)
    assert fixed["network"].sum() == pytest.approx(600.)
    assert fixed["application"].sum() == pytest.approx(400.)
    assert np.all(fixed["network"].sum(0) <= table.budgets[2] * np.diff(edges) + 1e-9)
    assert np.all(fixed["occupancy"] <= table.fleet.gpus * np.diff(edges) + 1e-9)
    assert 3. <= finish[0] < table.deadline
    np.testing.assert_array_equal(engine.state, [1, 0])


def test_small_fleet_qh_does_not_reserve_wan_for_compute_only_replicas():
    from pool_shed_calibration import calibration
    from pool_shed_campaign import execute_feedback, forecast

    central = calibration(0)
    table = forecast("coding", 0, 6666, 8, .5, 1000, 30)[0]
    result = execute_feedback(table, table, "queue_haul", table.timing, central)
    assert result["shed_fraction"] > .99
    assert result["max_relative_residual"] <= 1e-8
    assert np.array(result["resident_debt_generated_work_s"]) - result["resident_debt_recovered_work_s"] == pytest.approx(result["pending_resident_debt_work_s"], abs=1e-7)


def test_observed_recovery_boundary_is_a_feedback_and_candidate_start_time(monkeypatch):
    import pool_shed_planner as planner

    table, timing, calibration = case(3600.)
    table.fleet.metadata["protect_resident"] = True
    engine = PooledExecution(table, timing, calibration)
    engine.now = 600.
    engine.admit(np.array([0., 1., 0., 0.]))
    engine.state[:], engine.gated[:], engine.backlog[:], engine.remaining[:] = 5, True, 3., 0.
    starts, original = [], planner.phase_profile
    def record(*args, **kwargs):
        starts.append(args[4])
        return original(*args, **kwargs)
    monkeypatch.setattr(planner, "phase_profile", record)
    _, until, _ = planner.plan_admission(engine, table, "queue_haul", iterations=1)
    assert until == 603. and 603. in starts
    assert planner.planning_grid(engine.now, table.deadline, table.nominal_commit, .5)[1] == 864.
    assert engine.now == 600. and engine.backlog.tolist() == [3.]


@pytest.mark.parametrize("action,expected_bytes,expected_compute", [(0, 40., .2), (1, 20., 0.)])
def test_delayed_candidate_snapshots_current_prefix_without_recharging_an_earlier_reset(action, expected_bytes, expected_compute):
    table, timing, calibration = case(5.)
    table.endpoint[:] = table.budgets[:] = 10000.
    timing.update(beta=0., kv_completion_s=0., kv_batch_completion_s=0.)
    calibration.update(replay_context_tokens=[1., 1000.], replay_tps=[100., 100.], replay_completion_s=0.)
    table.fleet.metadata.update(protect_resident=True, source_session_rps=1., sequence_cycle=True,
        turn_sequences=[[{"context": 100, "prompt": 20, "output": 0},
                         {"context": 10, "prompt": 10, "output": 0, "reset": True},
                         {"context": 20, "prompt": 20, "output": 0}]],
        turn_duration_s=[[.1, .1, .1]], turn_work_s=[[.1, .1, .1]])
    profile = phase_profile(table, np.ones(1), action, 0, 1.5, np.array([1.5, 5.]), np.zeros((2, 1)), timing, calibration)
    assert profile["network"].sum() == pytest.approx(expected_bytes)
    assert (profile["replay"] + profile["kv"]).sum() == pytest.approx(expected_compute)
    assert profile["finish"] < 2.


@pytest.mark.parametrize("progress", [{"state": 1}, {"transferred": 1.}])
def test_protected_phase_profile_cannot_resnapshot_an_active_migration(progress):
    table, timing, calibration = case()
    table.fleet.metadata["protect_resident"] = True
    with pytest.raises(ValueError, match="observed-origin engine continuation"):
        phase_profile(table, np.ones(1), 0, 0, 1., np.array([1., 20.]), np.zeros((2, 1)), timing, calibration, **progress)


@pytest.mark.parametrize("payload,expected_finish", [(.4, 2.), (0., .6)])
def test_candidate_network_barriers_preserve_volume_and_zero_payload_timing(payload, expected_finish):
    table, timing, calibration = case(3.)
    table.fleet.metadata["protect_resident"] = True
    table.fleet.kv[:], table.endpoint[:], table.budgets[:] = payload, 1., 1.
    timing.update(beta=0., kv_completion_s=0., kv_batch_completion_s=0.)
    args = (table, np.ones(1), 1, 0, .6, np.arange(4.), np.zeros((2, 3)), timing, calibration)
    raw, causal = phase_profile(*args), phase_profile(*args, causal_network=True)
    assert raw["finish"] == pytest.approx(.6 + payload)
    assert causal["finish"] == pytest.approx(expected_finish)
    assert causal["network"].sum() == pytest.approx(payload)
    assert causal["network"][0] == 0.


def test_mid_bin_delta_wait_is_included_in_source_buffer_interval(monkeypatch):
    import pool_shed_planner as planner

    table, timing, calibration = case(3.)
    table.fleet.metadata["protect_resident"] = True
    table.fleet.kv[:], table.endpoint[:], table.budgets[:] = 0., 1., 1.
    timing["beta"] = 0.
    observed = []
    monkeypatch.setattr(planner, "_quiesce", lambda *args, **kwargs: (.6, table.fleet.context, np.zeros(1, bool), False))
    monkeypatch.setattr(planner, "catchup", lambda *args, **kwargs: (.4, 0.))
    monkeypatch.setattr(planner, "_buffered", lambda fleet, counts, begin, end, *args, **kwargs: (observed.append((begin, end)) or 0., 0.))
    profile = phase_profile(table, np.ones(1), 1, 0, 0., np.arange(4.), np.zeros((2, 3)), timing, calibration, causal_network=True)
    assert profile["network"] == pytest.approx([0., .4, 0.])
    assert profile["finish"] == 2. and observed == [(.6, 2.)]


def test_isolated_fastest_ranking_keeps_raw_singleton_times(monkeypatch):
    import pool_shed_planner as planner

    table, timing, calibration = case()
    table.fleet.metadata["protect_resident"] = True
    observed, original = [], planner.phase_profile
    def record(*args, **kwargs):
        observed.append(kwargs.get("causal_network", False))
        return original(*args, **kwargs)
    monkeypatch.setattr(planner, "phase_profile", record)
    plan_admission(PooledExecution(table, timing, calibration), table, "isolated_fastest", calibration=calibration)
    assert observed[:4 * len(table.fleet.count)] == [False] * (4 * len(table.fleet.count))
    assert all(observed[4 * len(table.fleet.count):])


@pytest.mark.parametrize("offset", [0., np.spacing(1.), 1e-8])
@pytest.mark.parametrize("boundary", ["release", "completion"])
def test_network_bin_time_equivalence_does_not_skip_a_bin_or_move_backwards(offset, boundary):
    table, timing, calibration = case(4.)
    table.endpoint[:], table.budgets[:] = 1., 1.
    timing.update(beta=0., kv_completion_s=0., kv_batch_completion_s=0.)
    table.fleet.kv[:] = .25 if boundary == "release" else 1. + offset
    start = 1. + offset if boundary == "release" else 0.
    profile = phase_profile(table, np.ones(1), 1, 0, start, np.arange(5.), np.zeros((2, 4)), timing, calibration, causal_network=True)
    expected = (2. if offset <= 1e-10 else 3.) if boundary == "release" else (1. + offset if offset <= 1e-10 else 2.)
    assert profile["finish"] == expected
    assert profile["finish"] >= start + table.fleet.kv[0]
    assert profile["network"].sum() == pytest.approx(table.fleet.kv[0])


def test_captured_secondary_lp_recovers_with_valid_dual_and_original_resource_certificate():
    from pathlib import Path
    from pool_shed_campaign import PRIMARY_TOL, _bounded_lp

    data = np.load(Path(__file__).parent / 'fixtures/pool_shed_secondary3104.npz')
    cost, matrix, rhs, upper = (data[k] for k in ('cost', 'matrix', 'rhs', 'upper'))
    with pytest.warns(RuntimeWarning, match='retrying the same LP with IPM'):
        chosen = _bounded_lp(cost, matrix, rhs, upper)
    row_dual, col_dual = data['row_dual'], data['col_dual']
    assert row_dual.max() <= 0
    assert matrix.T @ row_dual + col_dual == pytest.approx(cost, abs=1e-10)
    dual_bound = rhs @ row_dual + upper @ np.minimum(col_dual, 0)
    assert dual_bound == pytest.approx(2.3106690901477123, abs=1e-10)
    assert cost @ chosen == pytest.approx(dual_bound, abs=1e-9)
    assert chosen.min() >= -1e-10 and np.max(chosen - upper) <= 1e-10
    assert np.max(matrix @ chosen - rhs) <= 1e-10
    replicas = chosen * data['gpus'] / data['column_scale']
    assert np.max((data['original_matrix'][:, data['ids']] @ replicas - data['capacities']) / np.maximum(data['capacities'], 1)) <= 1e-8
    assert data['gains'][data['ids']] @ replicas >= data['primary'] - PRIMARY_TOL - 1e-8


def test_both_native_algorithms_nonoptimal_still_fail(monkeypatch):
    import pool_shed_campaign as campaign

    original, attempts = campaign.highspy.Highs, []
    class FailedSolver:
        def __init__(self):
            self.solver = original()
            attempts.append(self)
        def __getattr__(self, name):
            return getattr(self.solver, name)
        def run(self):
            return campaign.highspy.HighsStatus.kWarning
    monkeypatch.setattr(campaign.highspy, 'Highs', FailedSolver)
    with pytest.warns(RuntimeWarning, match='retrying the same LP with IPM'), pytest.raises(RuntimeError):
        campaign._bounded_lp(-np.ones(1), np.ones((1, 1)), np.ones(1), np.ones(1))
    assert len(attempts) == 3
    assert [item.getOptionValue('solver')[1] for item in attempts] == ['simplex', 'ipm', 'simplex']
    assert [item.getOptionValue('simplex_scale_strategy')[1] for item in attempts] == [0, 0, 2]


def test_captured_resolution_lp_recovers_with_scaling_and_a_dual_certificate():
    from pathlib import Path
    from pool_shed_campaign import _bounded_lp

    data = np.load(Path(__file__).parent / 'fixtures/pool_shed_resolution888.npz')
    cost, matrix, rhs, upper = (data[k] for k in ('cost', 'matrix', 'rhs', 'upper'))
    chosen = _bounded_lp(cost, matrix, rhs, upper)
    assert np.max(matrix @ chosen - rhs) <= 1e-10
    assert chosen.min() >= -1e-10 and np.max(chosen - upper) <= 1e-10
    assert data['row_dual'].max() <= 1e-10
    assert matrix.T @ data['row_dual'] + data['col_dual'] == pytest.approx(cost, abs=1e-10)
    bound = rhs @ data['row_dual'] + upper @ np.minimum(data['col_dual'], 0)
    assert cost @ chosen == pytest.approx(bound, abs=1e-8)


@pytest.mark.parametrize('scale', [1e-6, 1., 1e6])
@pytest.mark.parametrize('relative_saving,expected', [(0., 0), (1e-8, 1)])
def test_greedy_secondary_ties_keep_earliest_start_across_roundoff_and_scaling(scale, relative_saving, expected):
    from pool_shed_planner import _choose

    first = 11.097339422110858
    second = np.nextafter(first, -np.inf) if relative_saving == 0 else first * (1 - relative_saving)
    chosen = _choose(np.ones((1, 2)) / scale, np.ones(1), np.full(2, 7.25e-6 / scale),
                     np.array([first, second]) / scale, SimpleNamespace(gpus=1), True)
    assert chosen[expected] == pytest.approx(scale)
    assert chosen[1 - expected] == 0.


def test_optimal_secondary_lp_is_retried_until_original_certificate_passes():
    from pathlib import Path
    from pool_shed_campaign import PRIMARY_TOL, solve_lp

    data = np.load(Path(__file__).parent / 'fixtures/pool_shed_secondary9044.npz')
    table = SimpleNamespace(matrix=data['original_matrix'], capacities=data['capacities'],
                            gains=data['gains'], fleet=SimpleNamespace(gpus=int(data['gpus'])))
    allowed = np.zeros(len(table.gains), bool)
    allowed[data['ids']] = True
    chosen = solve_lp(table, allowed, data['objective'], float(data['primary']))
    assert np.max((table.matrix @ chosen - table.capacities) / np.maximum(table.capacities, 1)) <= 1e-8
    assert table.gains @ chosen >= data['primary'] - PRIMARY_TOL - 1e-8
    scaled = chosen[data['ids']] * data['column_scale'] / data['gpus']
    assert np.max(data['row_dual']) <= 0
    assert data['matrix'].T @ data['row_dual'] + data['col_dual'] == pytest.approx(data['cost'], abs=1e-10)
    bound = data['rhs'] @ data['row_dual'] + data['upper'] @ np.minimum(data['col_dual'], 0)
    assert bound == pytest.approx(2.5373245631700057, abs=1e-10)
    assert data['cost'] @ scaled == pytest.approx(bound, abs=1e-9)


@pytest.mark.parametrize('bad_value,error', [(np.nan, 'invalid replica'), (-1., 'invalid replica'), (2., 'infeasible resource'), (0., 'preserve the primary')])
@pytest.mark.parametrize('both_fail', [False, True])
def test_native_optimal_requires_original_certificate(monkeypatch, bad_value, error, both_fail):
    import pool_shed_campaign as campaign

    original, attempts = campaign.highspy.Highs, []
    class UncertifiedSolver:
        def __init__(self):
            self.solver = original()
            attempts.append(self)
        def __getattr__(self, name):
            return getattr(self.solver, name)
        def getSolution(self):
            return SimpleNamespace(col_value=[bad_value]) if both_fail or self is attempts[0] else self.solver.getSolution()
    monkeypatch.setattr(campaign.highspy, 'Highs', UncertifiedSolver)
    table = SimpleNamespace(matrix=np.ones((1, 1)), capacities=np.ones(1), gains=np.ones(1), fleet=SimpleNamespace(gpus=1))
    with pytest.warns(RuntimeWarning, match='retrying the same LP with IPM'):
        if both_fail:
            with pytest.raises(RuntimeError, match=error):
                campaign.solve_lp(table, np.ones(1, bool), np.ones(1), primary=.5)
        else:
            chosen = campaign.solve_lp(table, np.ones(1, bool), np.ones(1), primary=.5)
            assert chosen[0] == pytest.approx(.5, abs=1e-8)
    assert len(attempts) == (3 if both_fail else 2)
