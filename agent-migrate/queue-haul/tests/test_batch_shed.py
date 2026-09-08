"""Analytical checks for measured batch schedules and pooled admission."""

import numpy as np
import pytest

import pool_shed_campaign as c


def fleet(count=(4, 4), demand=(.1, .1), t1=(1., 3.), log=(1., 2.), kv=(10., 20.)):
    n = len(count)
    return c.Fleet(count=np.array(count), context=np.arange(1, n + 1) * 10.,
                   prompt=np.ones(n), output=np.ones(n), t1=np.array(t1),
                   kv=np.array(kv), log=np.array(log), demand=np.array(demand),
                   templates=[np.repeat(np.arange(n), count).tolist()], gpus=1,
                   kv_capacity=10000., metadata={})


def table(f, replay, kv, deadline=4., load=.5, endpoint=(100., 100., 200.),
          budgets=(100., 100., 100.), kappa=.5, kv_tails=(0., 0.)):
    return c.schedule_table(f, np.array(replay), np.array(kv), load, deadline,
                            np.array(endpoint), np.array(budgets),
                            {"kv_completion_s": kv_tails[0], "kv_batch_completion_s": kv_tails[1], "beta": 0., "kappa": kappa})


def test_batch_wall_time_preserves_singletons_and_counts_repeated_requests():
    replay = np.array([[0, 0], [1, 0], [0, 1], [2, 1], [3, 0]])
    expected = np.array([0., 1., 3., 4., 2.])
    np.testing.assert_allclose(c.batch_time(replay, np.array([1., 3.]), 0., .5, .95), expected)
    np.testing.assert_allclose(c.batch_time(replay, np.array([1., 3.]), np.log(2), .5, .5),
                               expected * np.sqrt(2))


def test_tight_wan_long_context_lp_preserves_physical_feasibility():
    from dataclasses import replace

    base = c.sample_fleet("coding")
    ids = np.resize(np.flatnonzero(base.context >= 24000), 24)
    counts = np.full(24, c.GPUS * 16 // 24)
    counts[-1] += c.GPUS * 16 - counts.sum()
    fields = {key: getattr(base, key)[ids] for key in ("context", "prompt", "output", "t1", "kv", "log", "demand")}
    fields["demand"] *= c.SOURCE_LOAD * c.GPUS / (counts @ fields["demand"])
    f = replace(base, **fields, count=counts)
    samples = c.network_samples()
    endpoint = samples[np.random.default_rng(2001).integers(len(samples), size=8)[3]]
    budgets, timing = c.bandwidth(endpoint, f.nodes, 100), c.calibration(8)["timing"][4]
    r, k = c.include_isolated(*c.library(f), c.isolated_methods(f, .5, endpoint, budgets, timing))
    results = c.compare(c.schedule_table(f, r, k, .5, 35, endpoint, budgets, timing))
    assert max(v["max_relative_residual"] for v in results.values()) <= 1e-8


def test_serial_batch_limit_adds_wall_service_without_a_free_gpu_factor():
    replay, t1 = np.array([[1, 1], [2, 1]]), np.array([1., 3.])
    np.testing.assert_allclose(c.batch_time(replay, t1, 0., 1., .95), [4., 5.])


def test_context_specific_batch_factors_preserve_singletons_and_monotonicity():
    replay = np.array([[0, 0], [1, 0], [0, 1], [2, 1], [1, 2], [3, 0]])
    t1, kappa = np.array([1., 3.]), np.array([.2, .8])
    expected = [0., 1., 3., 3.6, 5.8, 1.4]
    np.testing.assert_allclose(c.batch_time(replay, t1, 0., kappa, .95), expected)
    factors = np.tile(kappa, (len(replay), 1))
    factors[-1] = 1.
    np.testing.assert_allclose(c.batch_time(replay, t1, 0., factors, .95), [*expected[:-1], 3.])
    for extra in np.eye(2, dtype=int):
        assert np.all(c.batch_time(replay + extra, t1, 0., kappa, .95) >= np.array(expected) - 1e-12)


@pytest.mark.parametrize("kappa", (-.1, 1.1, np.array([.5, np.nan])))
def test_unidentified_batch_factors_hard_fail(kappa):
    with pytest.raises(ValueError, match="batch"):
        c.batch_time(np.array([[1, 1]]), np.array([1., 3.]), 0., kappa, .5)


def test_long_context_replay_serializes_the_entire_batch_only_when_selected():
    f = fleet()
    f.metadata = {"packing_context_tokens": [10., 20.], "batch_context_limit": 15.}
    replay = np.array([[1, 0], [1, 1], [0, 0]])
    kv = np.array([[0, 1], [0, 0], [0, 1]])
    t = c.schedule_table(f, replay, kv, .5, 10., np.full(3, 100.), np.full(3, 100.),
                         {"kv_completion_s": 0., "kv_batch_completion_s": 0., "beta": 0., "packing_kappa": [.2, .8]})
    np.testing.assert_allclose(t.duration, [1., 4., 0., 1., 4., 0.])


def test_library_keeps_every_singleton_and_whole_request_counts():
    f = fleet()
    replay, kv = c.library(f)
    patterns = np.column_stack((replay, kv))
    assert len(patterns) == len(np.unique(patterns, axis=0))
    assert np.all(patterns >= 0) and np.all(patterns == np.floor(patterns))
    assert np.all((replay + kv).sum(1) <= 8)
    assert np.all((replay + kv).sum(1) > 0)
    assert np.all(replay + kv <= f.count)
    for j in range(2 * len(f.count)):
        singleton = np.eye(2 * len(f.count), dtype=int)[j]
        assert np.any(np.all(patterns == singleton, axis=1))


def test_shared_wan_limits_planned_volume_across_both_destinations():
    t = table(fleet(t1=(.1, .3)), [[1, 1]], [[0, 0]], budgets=(1., 1., 1.))
    np.testing.assert_allclose(t.duration, .35)
    np.testing.assert_allclose(t.release, 3.65)
    np.testing.assert_allclose(t.log_bytes, 3.)
    np.testing.assert_allclose(t.rate, .75)
    x = c.select(t, "queue_haul")
    assert x.sum() == pytest.approx(4 / 3)
    assert t.gains @ x == pytest.approx(1 / 3)
    result = c.certify(t, x)
    assert result["shed_fraction"] == pytest.approx(1 / 3)
    assert sum(result["action_counts"]) == pytest.approx(8 / 3)
    assert sum(result["action_fractions"]) == pytest.approx(1 / 3)
    assert "last_completion_s" not in result
    assert result["resource_utilization"][-1] == pytest.approx(1.)


def test_isolated_precedence_requires_logs_before_replay():
    t = table(fleet(), [[1, 1]], [[0, 0]], endpoint=(5., 100., 105.))
    assert not t.eligible[t.route == 0].any()
    assert t.eligible[t.route == 1].all()
    x = c.select(t, "queue_haul")
    assert not x[t.route == 0].any()
    assert x[t.route == 1].sum() == pytest.approx(4 / 3.5)


def test_wan_budget_does_not_multiply_with_the_number_of_source_gpus():
    endpoint = np.array([2., 3., 5.])
    np.testing.assert_allclose(c.bandwidth(endpoint, 2, 1.), [4., 6., 10.])
    for gpus in (10000, 20000):
        np.testing.assert_allclose(c.bandwidth(endpoint, gpus, 1e-5), 1250.)
        np.testing.assert_allclose(c.bandwidth(endpoint, gpus, "reference"), endpoint)


def test_gpus_on_one_node_share_its_network_budget():
    f = fleet()
    f.gpus, f.gpus_per_node = 8, 8
    t = table(f, [[0, 0]], [[1, 1]], endpoint=(1., 2., 3.))
    np.testing.assert_allclose(t.budgets, [1., 2., 3.])
    f.gpus = 16
    t = table(f, [[0, 0]], [[1, 1]], endpoint=(1., 2., 3.))
    np.testing.assert_allclose(t.budgets, [2., 4., 6.])


def test_ninety_five_percent_load_keeps_fractional_pooled_admission():
    f = fleet(count=(4,), demand=(.2,), t1=(1.,), log=(0.,), kv=(1.,))
    t = table(f, [[0]], [[1]], load=.95)
    x = c.select(t, "kv_only")
    for route in (0, 1):
        assert x[t.route == route].sum() == pytest.approx(.25)
    result = c.certify(t, x)
    assert sum(result["action_counts"]) == pytest.approx(.5)
    assert result["shed_fraction"] == pytest.approx(.125)
    assert f.baseline_kv == pytest.approx(64.)  # Four ten-token contexts occupy four 16-token blocks.
    low = table(f, [[0]], [[1]], load=.25)
    memory = slice(len(f.count) + 4, len(f.count) + 6)
    np.testing.assert_allclose(t.capacities[memory], low.capacities[memory])
    np.testing.assert_allclose(t.capacities[memory], 9936.)


def test_patterns_cannot_reuse_the_same_source_population():
    f = fleet(count=(4,), demand=(.2,), t1=(1.,), log=(0.,), kv=(1.,))
    t = table(f, [[0]], [[3]], load=.25)
    x = c.select(t, "queue_haul")
    assert x.sum() == pytest.approx(4 / 3)
    result = c.certify(t, x)
    assert sum(result["action_counts"]) == pytest.approx(4.)
    assert result["shed_fraction"] == pytest.approx(1.)


def test_pure_kv_has_no_replay_work_but_keeps_endpoint_capacity():
    f = fleet(count=(4,), demand=(.2,), t1=(1.,), log=(0.,), kv=(1.,))
    t = table(f, [[0]], [[1]], load=.25, deadline=1.)
    assert c.certify(t, c.select(t, "kv_only"))["completed_sessions"] == pytest.approx(4.)
    assert not t.matrix[1:3].any()
    t = table(f, [[0]], [[1]], load=.25, deadline=1., endpoint=(1., 1., 2.))
    assert c.certify(t, c.select(t, "kv_only"))["completed_sessions"] == pytest.approx(2.)
    mixed = table(f, [[1]], [[1]])
    np.testing.assert_equal(mixed.matrix[1:3].sum(0), mixed.duration)


def test_kv_completion_changes_isolated_choice_and_deadline_feasibility():
    f = fleet(count=(4,), demand=(.2,), t1=(1.,), log=(1.,), kv=(10.,))
    r, k = np.array([[0], [1]]), np.array([[1], [0]])
    assert not table(f, r, k).fastest.any()
    t = table(f, r, k, deadline=1.5, kv_tails=(2., 2.))
    assert t.fastest.all()
    assert not t.eligible[t.kv.any(1)].any()
    assert t.eligible[t.replay.any(1)].all()
    assert c.certify(t, c.select(t, "kv_only"))["completed_sessions"] == 0
    boundary = table(f, [[0]], [[1]], deadline=2., kv_tails=(2., 2.))
    assert not boundary.eligible.any()


def test_kv_tail_limits_isolated_deadline_and_consumes_compute_work():
    t = table(fleet(), [[1, 0]], [[0, 1]], kv_tails=(.5, 1.2))
    np.testing.assert_allclose(t.kv_release, 3.5)
    np.testing.assert_allclose(t.rate, 21 / 4)
    np.testing.assert_allclose(t.matrix[2:4].sum(0), t.duration + .5)
    c.certify(t, c.select(t, "queue_haul"))
    t = table(fleet(), [[0, 0]], [[4, 4]], kv_tails=(.5, 1.2))
    np.testing.assert_allclose(t.kv_release, 2.8)


@pytest.mark.parametrize("deadline,log,allowed", [(2., 0., True), (2., 1., False), (1.99, 0., False)])
def test_zero_log_does_not_waive_batch_completion_deadline(deadline, log, allowed):
    f = fleet(count=(4,), demand=(.2,), t1=(2.,), log=(log,), kv=(1.,))
    t = table(f, [[1]], [[0]], deadline=deadline)
    assert t.eligible.all() if allowed else not t.eligible.any()
    x = c.select(t, "queue_haul")
    assert x.sum() > 0 if allowed else x.sum() == 0
    assert np.isfinite(c.certify(t, x)["shed_fraction"])


def test_replay_and_kv_use_total_volume_instead_of_fixed_reservations():
    t = table(fleet(), [[1, 0]], [[0, 1]])
    np.testing.assert_allclose(t.duration, 1.)
    np.testing.assert_allclose(t.release, 3.)
    np.testing.assert_allclose(t.rate, 21 / 4)
    np.testing.assert_allclose((t.log_bytes + t.kv_bytes) / t.deadline, t.rate)


def test_certificate_rejects_a_shared_wan_overbooking():
    t = table(fleet(t1=(.1, .3)), [[1, 1]], [[0, 0]], budgets=(1., 1., 1.))
    with pytest.raises((ValueError, RuntimeError)):
        c.certify(t, np.ones(len(t.gains)))


def test_every_policy_uses_the_same_planning_relaxation():
    f = fleet()
    replay, kv = c.library(f)
    t = c.schedule_table(f, replay, kv, .5, 4., np.array([5., 20., 25.]),
                         np.array([8., 12., 15.]), {"kv_completion_s": 0., "kv_batch_completion_s": 0., "beta": 0., "kappa": .5})
    upper = c.certify(t, c.select(t, "queue_haul"))["shed_fraction"]
    for policy in c.POLICIES:
        x = c.select(t, policy)
        result = c.certify(t, x)
        assert np.all(x >= 0)
        assert not x[~t.eligible].any()
        assert np.all(t.matrix @ x <= t.capacities + 1e-8)
        assert result["shed_fraction"] == pytest.approx(t.gains @ x)
        assert result["shed_fraction"] <= upper + 1e-8
        if policy == "kv_only":
            assert not np.any(x @ t.replay)
        if policy == "replay_only":
            assert not np.any(x @ t.kv)
        if policy == "isolated_fastest":
            assert not np.any((x @ t.kv)[t.fastest])
            assert not np.any((x @ t.replay)[~t.fastest])


def test_lp_shed_increases_with_deadline_and_wan_and_decreases_with_resident_load():
    f = fleet()
    replay, kv = c.library(f)

    def shed(deadline=4., bandwidth=6., load=.5):
        t = c.schedule_table(f, replay, kv, load, deadline, np.full(3, 100.),
                             np.full(3, bandwidth), {"kv_completion_s": 0., "kv_batch_completion_s": 0., "beta": .5, "kappa": .5})
        return c.certify(t, c.select(t, "queue_haul"))["shed_fraction"]

    deadlines = [shed(deadline=d) for d in (1., 2., 4., 8., 16.)]
    bandwidths = [shed(bandwidth=b) for b in (1., 3., 6., 12., 24.)]
    loads = [shed(deadline=16., bandwidth=24., load=u) for u in (.25, .5, .75, .9, .95)]
    assert np.all(np.diff(deadlines) >= -1e-9) and deadlines[-1] > deadlines[0]
    assert np.all(np.diff(bandwidths) >= -1e-9) and bandwidths[-1] > bandwidths[0]
    assert np.all(np.diff(loads) <= 1e-9)
    np.testing.assert_allclose(loads, [1., 1., .625, .25, .125], atol=1e-9)


@pytest.mark.parametrize("policy", ("queue_haul", "greedy", "kv_only", "replay_only", "isolated_fastest"))
def test_destination_names_do_not_change_optimized_shed(policy):
    f = fleet()
    replay, kv = c.library(f)
    endpoint, budgets = np.array([5., 20., 25.]), np.array([8., 12., 15.])
    a = c.schedule_table(f, replay, kv, .5, 4., endpoint, budgets, {"kv_completion_s": 0., "kv_batch_completion_s": 0., "beta": 0., "kappa": .5})
    b = c.schedule_table(f, replay, kv, .5, 4., endpoint[[1, 0, 2]], budgets[[1, 0, 2]],
                         {"kv_completion_s": 0., "kv_batch_completion_s": 0., "beta": 0., "kappa": .5})
    assert a.gains @ c.select(a, policy) == pytest.approx(b.gains @ c.select(b, policy))


def test_planner_reuses_compute_across_the_deadline():
    t = table(fleet(t1=(1., 1.), log=(0., 0.)), np.eye(2), np.zeros((2, 2)))
    chosen = c.select(t, "replay_only")
    assert chosen.sum() == pytest.approx(8.)
    assert c.certify(t, chosen)["shed_fraction"] == pytest.approx(1.)


def test_volume_relaxation_allows_reusing_log_bandwidth_for_kv():
    f = fleet(count=(1, 1), demand=(.4, .4), t1=(8., 8.), log=(2., 2.), kv=(98., 98.))
    t = table(f, [[1, 0], [0, 0]], [[0, 0], [0, 1]], deadline=10.,
              endpoint=(10., 10., 20.), budgets=(10., 10., 10.))
    # Logs transfer at 0–.2s, replay at .2–8.2s, and KV at .2–10s.
    assert c.certify(t, c.select(t, "queue_haul"))["shed_fraction"] == pytest.approx(1.)


def test_equal_optimal_shed_can_use_zero_or_all_kv():
    f = fleet(count=(4,), demand=(.2,), t1=(1.,), log=(0.,), kv=(1.,))
    t = table(f, [[1], [0]], [[0], [1]])
    result = c.optimal_kv_range(t)
    assert result["planned_shed_fraction"] == pytest.approx(1.)
    assert result["minimum_kv_fraction"] == pytest.approx(0.)
    assert result["maximum_kv_fraction"] == pytest.approx(1.)


def test_redundant_network_constraints_do_not_change_greedy_prices():
    f = fleet()
    t = table(f, *c.library(f), endpoint=(5., 20., 25.), budgets=(8., 12., 15.))
    chosen = c.select(t, "greedy")
    t.matrix = np.vstack((t.matrix, np.tile(t.matrix[-1], (4, 1))))
    t.capacities = np.r_[t.capacities, np.repeat(t.capacities[-1], 4)]
    np.testing.assert_allclose(c.select(t, "greedy"), chosen)
    c.certify(t, chosen)


def test_nominal_forecast_includes_final_delta_and_buffered_turns():
    f = fleet(count=(1,), demand=(.2,), t1=(1.,), log=(0.,), kv=(100.,))
    f.context[:] = 100
    f.metadata = {"source_session_rps": 1., "turn_sequences": [[
        {"context": 100, "prompt": 10, "output": 0},
        {"context": 110, "prompt": 20, "output": 0}]]}
    timing = {"beta": 0., "kappa": 1., "kv_completion_s": .5, "kv_batch_completion_s": .5}
    measured = {"forecast_load": .5, "replay_context_tokens": [100., 200.], "replay_tps": [100., 100.],
        "replay_completion_s": 0., "kv_block_tokens": 10, "kv_block_bytes": 10,
        "kv_tail_replay_tps": 10., "F": 10., "G": 10.}
    np.testing.assert_allclose(c.nominal_action(f, np.ones(1), 0, 0, 100., timing, measured),
                               [20., 1.1, 1.3, 2.])
    np.testing.assert_allclose(c.nominal_action(f, np.ones(1), 1, 0, 100., timing, measured),
                               [110., .5, 1.6, 2.])


@pytest.mark.parametrize("kv_tail,selected_action", [(.5, "kv"), (2., "replay")])
def test_debt_tiebreak_preserves_primary_gain_and_has_no_fixed_action_preference(kv_tail, selected_action):
    f = fleet(count=(4,), demand=(.2,), t1=(1.,), log=(0.,), kv=(1.,))
    t = c.schedule_table(f, np.array([[1], [0]]), np.array([[0], [1]]), .5, 4.,
        np.full(3, 100.), np.full(3, 100.), {"beta": 0., "kappa": 1., "resident_replay_loss": .9,
        "kv_completion_s": kv_tail, "kv_batch_completion_s": kv_tail})
    primary = c.solve_lp(t, c.policy_mask(t, "queue_haul"), -t.gains)
    chosen = c.select(t, "queue_haul")
    assert t.gains @ chosen == pytest.approx(t.gains @ primary)
    assert t.gains @ chosen == pytest.approx(1.)
    assert float(chosen @ getattr(t, selected_action)[:, 0]) == pytest.approx(4.)
    c.certify(t, chosen)


def test_service_work_charges_ongoing_arrivals_only_after_nominal_commit():
    f = fleet(count=(4,), demand=(.2,), t1=(1.,), log=(0.,), kv=(10.,))
    t = c.schedule_table(f, np.array([[1], [0]]), np.array([[0], [1]]), .5, 10.,
        np.full(3, 100.), np.full(3, 100.), {"beta": 0., "kappa": 1., "resident_replay_loss": .8,
        "kv_completion_s": .5, "kv_batch_completion_s": .5})
    np.testing.assert_allclose(t.service_time, [2.7, 2.38, 2.7, 2.38])
    np.testing.assert_allclose(t.debt, [.4, .25, .4, .25])
