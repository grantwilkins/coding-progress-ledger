"""Check whole-session power credit and pooled resource conservation independently."""

import gzip
import json
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

import pool_shed_campaign as c


def fleet(count=(2,), demand=(.1,), replay=(1.,), kv=(10.,), log=(1.,), gpus=10):
    return c.Fleet(np.array(count), np.ones(len(count)), np.array(demand), np.array(replay),
                   np.array(kv), np.array(log), gpus, 10000., 300., 100., {})


def test_shared_egress_and_whole_cohort_completion():
    f = fleet()
    plan = np.array([0, 1, 0, 1])
    budgets = np.array([100., 100., 10.])
    endpoint = np.array([100., 100., 200.])
    done, result = c.execute(f, plan, 1.9, budgets, endpoint)
    assert done.sum() == 0  # Both transfers have 9.5/10 bytes; no fractional credit.
    assert result['incomplete_sessions'] == 2
    done, result = c.execute(f, plan, 2., budgets, endpoint)
    np.testing.assert_array_equal(done, plan)
    assert result['network_utilization'][2] == pytest.approx(1.)
    assert result['last_event_s'] == pytest.approx(2.)


def test_endpoint_ceiling_and_staged_replay():
    f = fleet(count=(1,), log=(10.,), replay=(2.,), gpus=1000)
    plan = np.array([1, 0, 0, 0])
    budgets, endpoint = np.full(3, 1e6), np.array([10., 10., 20.])
    assert c.execute(f, plan, 2.999, budgets, endpoint)[0].sum() == 0
    done, result = c.execute(f, plan, 3., budgets, endpoint)
    assert done.sum() == 1  # One second network + two seconds on <=1 GPU.
    assert result['last_event_s'] == pytest.approx(3.)
    assert c.execute(replace(f, log=np.zeros(1)), plan, 2., budgets, endpoint)[0].sum() == 1


def test_fixed_plan_partial_completion_and_admission():
    f = fleet(count=(1, 1), demand=(.1, .1), kv=(5., 20.), replay=(1., 1.), log=(1., 1.))
    plan = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    done, result = c.execute(f, plan, 1., np.full(3, 10.), np.full(3, 100.))
    assert done.sum() == 1
    assert result['incomplete_sessions'] == 1
    assert result['network_utilization'][0] == pytest.approx(1.)
    # Stable cohort/action IDs: first session reserves .6 GPU, second is rejected.
    f = fleet(count=(2,), demand=(.6,), gpus=2)
    done, result = c.execute(f, np.array([0, 2, 0, 0]), 10., np.full(3, 100.), np.full(3, 100.), .8)
    assert result['rejected_sessions'] == 1
    assert done.sum() == 1
    assert result['reserved_serving_utilization'][0] <= 1 + 1e-12


def test_executor_rejects_fractional_or_duplicate_sessions():
    for plan in ([0, .5, 0, 0], [0, 2, 0, 1], [0, -1, 0, 0]):
        with pytest.raises(ValueError, match='whole-session'):
            c.execute(fleet(), np.array(plan), 1., np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match='whole-session'):
        c.execute(fleet(), np.array([0, 1, 0, 0]), np.nan, np.ones(3), np.ones(3))


def test_fair_rates_redistribute_unused_route_capacity():
    rates = c.fair_rates(np.array([10, 10]), np.array([1., 10.]), 50.)
    np.testing.assert_allclose(rates, [1., 4.])


def test_prefill_contention_and_network_blocking_are_separate():
    f = fleet(count=(2,), demand=(.25,), replay=(1.,), log=(0.,), gpus=2)
    plan = np.array([2, 0, 0, 0])
    done, stats = c.execute(f, plan, 2., np.full(3, 100.), np.full(3, 100.))
    assert done.sum() == 0
    assert stats['prefill_peak_ready_sessions'] == [2, 0]
    assert stats['prefill_ready_at_deadline'] == [2, 0]
    assert stats['prefill_network_blocked_at_deadline'] == [0, 0]
    assert stats['prefill_remaining_gpu_s'] == pytest.approx([1., 0.])
    assert stats['prefill_contention_session_s'] == pytest.approx([3., 0.])
    assert c.execute(f, plan, 4., np.full(3, 100.), np.full(3, 100.))[0].sum() == 2
    _, stats = c.execute(replace(f, log=np.array([1000.])), plan, 2., np.full(3, 100.), np.full(3, 100.))
    assert stats['prefill_network_blocked_at_deadline'] == [2, 0]
    assert stats['prefill_contention_session_s'] == [0., 0.]
    _, stats = c.execute(replace(f, log=np.array([100.])), plan, 2., np.full(3, 100.), np.full(3, 100.))
    assert stats['prefill_peak_ready_sessions'] == stats['prefill_ready_at_deadline'] == [2, 0]


@pytest.mark.parametrize('policy', ['queue_haul', 'greedy', 'replay_only'])
def test_policies_preserve_shed_and_balance_symmetric_routes(policy):
    f = fleet(count=(100,), demand=(.08,), replay=(1.,), kv=(1e9,), log=(.001,), gpus=20)
    limits = np.full(3, 100.)
    chosen, stats = c.select(f, 10., limits, limits, policy)
    if policy == 'queue_haul':
        assert stats['planned_shed_w'] == pytest.approx(stats['lp_bound_w'])
    np.testing.assert_array_equal(chosen, [50, 0, 50, 0])
    assert chosen.sum() == 100
    matrix, capacity, _ = c.resources(f, 10., limits, limits)
    assert max((matrix @ chosen / capacity)[3:5]) < .901
    assert c.execute(f, chosen, 10., limits, limits)[0].sum() == 100


def test_greedy_balances_around_existing_allocations_and_preserves_odd_counts():
    f = fleet(count=(101,), demand=(.08,), replay=(1.,), kv=(1e9,), log=(.001,), gpus=20)
    matrix, caps, isolated = c.resources(f, 10., np.full(3, 100.), np.full(3, 100.))
    initial = np.array([40, 0, 0, 0])
    chosen = c.greedy_fill(f.count, np.repeat(f.gain, 4), matrix, caps, isolated <= 10., initial)
    assert chosen.sum() == 101 and abs(chosen[0] - chosen[2]) == 1
    assert np.all(chosen >= initial)
    assert np.all(matrix @ chosen <= caps)
    np.testing.assert_array_equal(initial, [40, 0, 0, 0])


def test_greedy_routes_respect_unequal_budgets_and_endpoint_eligibility():
    f = fleet(count=(100,), demand=(.001,), replay=(100.,), kv=(10.,), log=(1.,), gpus=100)
    budgets = np.array([20., 80., 100.])
    chosen, _ = c.select(f, 1., budgets, np.full(3, 1000.), 'kv_only')
    np.testing.assert_array_equal(chosen, [0, 2, 0, 8])
    chosen, _ = c.select(f, 1., budgets, np.array([1000., 1., 1001.]), 'kv_only')
    np.testing.assert_array_equal(chosen, [0, 2, 0, 0])


@pytest.mark.parametrize('policy', ['greedy', 'kv_only', 'replay_only', 'isolated_fastest'])
def test_greedy_is_invariant_to_destination_labels(policy):
    f = fleet(count=(10, 20), demand=(.2, .4), replay=(.4, .8), kv=(30., 100.), log=(2., 3.), gpus=20)
    budgets, endpoint = np.array([50., 150., 180.]), np.array([30., 100., 130.])
    a, _ = c.select(f, 3., budgets, endpoint, policy)
    b, _ = c.select(f, 3., budgets[[1, 0, 2]], endpoint[[1, 0, 2]], policy)
    np.testing.assert_array_equal(a.reshape(-1, 4)[:, [2, 3, 0, 1]], b.reshape(-1, 4))


def test_lp_bound_against_exhaustive_integer_plans_and_monotonicity():
    f = fleet(count=(1, 1), demand=(.1, .2), kv=(4., 6.), replay=(1., 2.), log=(1., 1.))
    endpoint = np.array([20., 20., 40.])
    bounds = []
    for deadline in (.5, 1., 2., 4.):
        budgets = np.array([3., 4., 5.])
        matrix, capacities, isolated = c.resources(f, deadline, budgets, endpoint)
        optimum = 0.
        for actions in product(range(-1, 4), repeat=2):
            plan = np.zeros(8, int)
            for i, action in enumerate(actions):
                if action >= 0:
                    plan[4*i+action] = 1
            if np.all(matrix @ plan <= capacities) and np.all(isolated[plan > 0] <= deadline):
                optimum = max(optimum, np.repeat(f.gain, 4) @ plan)
        plan, stats = c.select(f, deadline, budgets, endpoint, 'queue_haul')
        assert stats['lp_bound_w'] + 1e-8 >= optimum >= stats['planned_shed_w'] - 1e-8
        assert np.all(plan == np.floor(plan))
        bounds.append(stats['lp_bound_w'])
    assert np.all(np.diff(bounds) >= -1e-8)
    small = c.select(f, 1., np.array([3., 4., 5.]), endpoint, 'queue_haul')[1]
    large = c.select(f, 1., np.array([30., 40., 50.]), endpoint, 'queue_haul')[1]
    assert large['lp_bound_w'] >= small['lp_bound_w']


def test_lp_bound_dominates_all_baselines_under_identical_constraints():
    rng = np.random.default_rng(20260909)
    for _ in range(24):
        n = rng.integers(1, 6)
        f = fleet(count=rng.integers(1, 21, n), demand=rng.uniform(.01, .6, n),
                  replay=rng.uniform(.05, 3., n), kv=rng.uniform(2., 100., n),
                  log=rng.uniform(.1, 5., n), gpus=int(rng.integers(4, 31)))
        f = replace(f, kv_capacity=f.baseline_kv + rng.uniform(5., 80.))
        deadline, budgets, endpoint = rng.uniform(.2, 10.), rng.uniform(2., 80., 3), rng.uniform(1., 40., 3)
        matrix, capacities, isolated = c.resources(f, deadline, budgets, endpoint)
        bound = c.select(f, deadline, budgets, endpoint, 'queue_haul')[1]['lp_bound_w']
        tolerance = 1e-8 * max(1., bound)
        for policy in c.POLICIES:
            plan, stats = c.select(f, deadline, budgets, endpoint, policy)
            assert np.all(plan >= 0) and np.all(plan == np.floor(plan))
            assert np.all(plan.reshape(-1, 4).sum(1) <= f.count)
            assert np.all(matrix @ plan <= capacities * (1 + 1e-8))
            assert np.all(isolated[plan > 0] <= deadline)
            assert stats['planned_shed_w'] == pytest.approx(np.repeat(f.gain, 4) @ plan)
            assert stats['planned_shed_w'] <= bound + tolerance
            done, _ = c.execute(f, plan, deadline, budgets, endpoint)
            assert np.repeat(f.gain, 4) @ done <= stats['planned_shed_w'] + tolerance


def test_volume_feasibility_does_not_guarantee_staged_completion():
    f = fleet(count=(2,), demand=(.5,), replay=(.75,), kv=(100.,), log=(1.,), gpus=4)
    plan, deadline = np.array([2, 0, 0, 0]), 2.
    budgets, endpoint = np.full(3, 2.), np.array([2., 2., 4.])
    matrix, capacities, isolated = c.resources(f, deadline, budgets, endpoint)
    assert np.all(matrix @ plan <= capacities)
    assert np.all(isolated[plan > 0] <= deadline)
    bound = c.select(f, deadline, budgets, endpoint, 'queue_haul')[1]['lp_bound_w']
    assert np.repeat(f.gain, 4) @ plan <= bound + 1e-8
    done, stats = c.execute(f, plan, deadline, budgets, endpoint)
    # One second of log transport leaves one GPU-second for 1.5 GPU-seconds of replay.
    assert done.sum() == 0 and stats['rejected_sessions'] == 0
    assert stats['prefill_remaining_gpu_s'] == pytest.approx([.5, 0.])
    np.testing.assert_array_equal(c.execute(f, plan, 2.5, budgets, endpoint)[0], plan)


def test_greedy_uses_global_action_order_and_count_weighted_prices():
    matrix = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 4., 0., 0., 2., 0., 0., 0.]])
    eligible = np.array([True, True, False, False, True, False, False, False])
    # A's fallback must not precede B's more efficient primary action.
    gains = np.array([10.] * 4 + [12.] * 4)
    chosen = c.greedy_fill(np.array([2, 1]), gains, matrix, np.array([1., 4.]), eligible)
    assert chosen[4] == 1
    assert chosen.sum() == 2


@pytest.mark.parametrize('model', c.MODELS)
def test_measured_population_power_and_baselines(model):
    f = c.sample_fleet(model, 'coding', 8, 0)
    assert f.count.sum() == c.GPUS * 8
    assert f.count @ f.demand == pytest.approx(c.GPUS * .8)
    assert f.count @ f.gain == pytest.approx(f.gpus * (f.power_w - f.idle_w))
    assert f.baseline_kv == pytest.approx(f.count @ f.context * .5 / .8)
    endpoint = np.median(c.network_samples(), axis=0)
    budgets = c.bandwidth(endpoint, f.gpus, 100)
    for policy in c.POLICIES:
        chosen, _ = c.select(f, 30, budgets, endpoint, policy)
        if policy == 'kv_only':
            assert not chosen[::2].any()
        if policy == 'replay_only':
            assert not chosen[1::2].any()


def test_a100_calibration_uses_same_runtime_raw_anchors_and_warm_idle():
    prefill, power = c.calibrations('gpt-oss-20b')
    assert prefill['kv_capacity_tokens'] == 1_936_832
    assert [r['context_tokens'] for r in prefill['curve']] == [2048, 8192, 28672]
    assert power['F_prefill_tps'] == pytest.approx(18156.26919119177)
    assert power['G_decode_tps'] == pytest.approx(1585.5893522872957)
    assert power['evidence']['prior_rational_fit_status'] == 'holdout_failed'
    assert all(119 < curve[0][1] < 120 for curve in power['phase_power']['measured_power_bootstrap'])
    assert sum(power['bootstrap_curve_counts']) == 200
    assert c.configuration()['installed_gpu_w'] == 19_999_800
    f = c.sample_fleet('gpt-oss-20b', 'coding', 8, 0)
    assert 295 < f.power_w < 301
    assert 0 < f.metadata['excluded_states'] < f.metadata['supported_states']
    with pytest.raises(ValueError, match='A100'):
        c.calibrations('qwen3.8-27b')


def test_compact_draws_are_paired_and_reproducible():
    config = c.configuration(True)
    cell = c.cells(config)[0]
    a, b = c.run_cell(config, cell), c.run_cell(config, cell)
    assert c.digest(a) == c.digest(b)
    rows = list(c.draw_rows(a))
    assert len(a['executions']) <= 45
    assert len(rows) == 5 * 3 * 20
    for draw in range(20):
        matched = [r for r in rows if r['draw'] == draw]
        assert len({(r['network_draw'], r['power_draw']) for r in matched}) == 1
        for row in matched:
            assert sum(row['action_counts']) == row['completed_sessions']
            assert sum(row['action_shed_w']) == pytest.approx(row['shed_w'])
            assert 0 <= row['shed_w'] <= row['initial_source_w'] - row['idle_source_w'] + 1e-7
    assert a['plans']['queue_haul']['planned_shed_w'] == rows[0]['planned_shed_w']
    volume = list(c.volume_rows(a))
    assert len(volume) == 6 * config['draws']
    for draw in range(config['draws']):
        matched = {r['policy']: r for r in volume if r['draw'] == draw}
        for policy, row in matched.items():
            assert row['shed_w'] <= matched['lp_bound']['shed_w'] + 1e-6
            if policy != 'lp_bound':
                assert sum(row['action_shed_w']) == pytest.approx(row['shed_w'])
    a['plans']['queue_haul']['lp_bound_w'] = -1.
    with pytest.raises(RuntimeError, match='upper bound'):
        c.audit_plans(a['plans'])


def test_fleet_lp_keeps_small_log_coefficients_at_large_bandwidth():
    f = c.sample_fleet('gpt-oss-20b', 'coding', 8, 0)
    endpoint = np.median(c.network_samples(), axis=0)
    endpoint[2] = endpoint[:2].sum()
    budgets = c.bandwidth(endpoint, f.gpus, 500_000.)  # Deliberate numerical stress, outside default WAN sweep.
    chosen, _ = c.select(f, 1., budgets, endpoint, 'queue_haul')
    matrix, capacities, _ = c.resources(f, 1., budgets, endpoint)
    assert np.max((matrix @ chosen) / capacities) <= 1 + 1e-8


def test_wan_allocation_does_not_scale_with_gpu_count_or_endpoint_ratio():
    endpoint = np.array([2e9, 8e9, 10e9]) / 8
    expected = np.full(3, 100e9 / 8)
    np.testing.assert_array_equal(c.bandwidth(endpoint, 50_000, 100), expected)
    np.testing.assert_array_equal(c.bandwidth(endpoint, 100_000, 100), expected)
    np.testing.assert_array_equal(c.bandwidth(endpoint, 50_000, 'reference'), endpoint)
    assert max(x for x in c.configuration()['wan_gbps'] if x != 'reference') == 400


def test_memory_infeasible_population_is_explicit(monkeypatch):
    f = replace(fleet(), kv_capacity=1.)
    monkeypatch.setattr(c, 'sample_fleet', lambda *args: f)
    config = c.configuration(True)
    result = c.run_cell(config, c.cells(config)[0])
    assert result['status'] == 'memory_infeasible'
    assert 'executions' not in result


def test_checkpoint_reduction_and_hard_failures(tmp_path, monkeypatch):
    config = c.configuration(True)
    config.update(models=[c.MODELS[0]], deadlines=[1], wan_gbps=[40], draws=3)
    monkeypatch.setattr(c, 'configuration', lambda smoke=False: config)
    monkeypatch.setattr(c, 'plot', lambda *args: None)
    c.prepare(tmp_path)
    with pytest.raises(ValueError, match='missing'):
        c.reduce(tmp_path)
    c.run(tmp_path)
    path = next((tmp_path / 'cells').glob('*.gz'))
    before = path.stat().st_mtime_ns
    c.run(tmp_path)
    assert before == path.stat().st_mtime_ns
    summary = c.reduce(tmp_path)
    assert len(summary) == 15
    assert {r['evaluation'] for r in summary} == {'staged_execution'}
    import csv
    with (tmp_path / 'volume-summary.csv').open() as handle:
        volume = list(csv.DictReader(handle))
    assert len(volume) == 6
    assert {r['evaluation'] for r in volume} == {'nominal_volume'}
    audit = json.loads((tmp_path / 'dominance-audit.json').read_text())
    assert audit['cells_checked'] == 1 and audit['volume_bound_violations'] == 0
    with gzip.open(path, 'rt') as handle:
        data = json.load(handle)
    data['executions'].append(data['executions'][0])
    c.write_json(path, data)
    with pytest.raises(ValueError, match='duplicate'):
        c.reduce(tmp_path)
    monkeypatch.setattr(c, 'provenance', lambda: {})
    with pytest.raises(ValueError, match='changed'):
        c.reduce(tmp_path)
