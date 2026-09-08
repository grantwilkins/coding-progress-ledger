"""Campaign provenance, measured inputs, and complete paired reductions."""

import gzip
import json

import numpy as np
import pytest

import pool_shed_campaign as c
from pool_shed_calibration import calibration, replay_seconds


def test_measured_calibration_and_population():
    measured = calibration(2)
    singleton = replay_seconds(measured['calibration_contexts'], measured)
    timing = measured['timing'][0]
    batch = c.batch_time(np.ones((1, 8)), singleton, timing['beta'], timing['kappa'], load=0)[0]
    assert batch == pytest.approx(measured['evidence']['idle_width8_endpoint_s'])
    assert measured['evidence']['replay_holdout']['episodes'] == 220
    assert measured['evidence']['replay_holdout']['p90_relative_error'] < .05
    assert measured['evidence']['transition_passes'] == 9
    assert len(measured['timing']) == 3
    assert measured['active_w'] == pytest.approx(280.8065)
    assert len(measured['power_draws_w']) == 200
    for workload in ('coding', 'measured_pack'):
        fleet = c.sample_fleet(workload)
        assert fleet.count.sum() == 8 * c.GPUS
        assert fleet.count @ fleet.demand == pytest.approx(.8 * c.GPUS)
        assert fleet.count @ fleet.gain == pytest.approx(1.)
        assert fleet.baseline_kv < fleet.kv_capacity
        r, k = c.library(fleet)
        assert len(r) * 2 <= 1344
        assert np.all((r + k).sum(1) <= 8)
        assert np.all(r * k == 0)
        a, b = c.library(fleet, expanded=True)
        assert set(map(tuple, np.c_[r, k])) <= set(map(tuple, np.c_[a, b]))


def test_default_grid_and_paired_endpoints():
    config = c.configuration()
    assert len(c.cells(config)) == 11250
    samples = c.network_samples()
    np.testing.assert_allclose(samples[:, :2].sum(1), samples[:, 2])
    for endpoint in samples:
        budget = c.bandwidth(endpoint, c.GPUS, 40)
        assert max(budget) <= 40e9 / 8
        np.testing.assert_equal(c.bandwidth(endpoint, c.GPUS, 'reference'), endpoint)


def test_lp_scaling_preserves_source_counts_in_the_loaded_coding_case(tmp_path):
    plan = c.prepare(tmp_path)
    result = c.run_cell(plan, (('coding', 1), .75, 4, 100, 30))
    assert result['results']['queue_haul']['initial_nominal_shed_fraction'] == pytest.approx(.625)
    assert result['results']['queue_haul']['shed_fraction'] <= .625 + 1e-8
    assert max(r['max_relative_residual'] for r in result['results'].values()) <= 1e-8


def test_scaling_nodes_and_network_together_preserves_the_policy_tradeoff():
    plan = {'identity': 'scaling', 'config': c.configuration(), 'calibration': calibration(0), 'network_indices': [-1]}
    results = []
    for gpus in (8, 80):
        plan['config']['gpus'] = gpus
        results.append(c.run_cell(plan, (('measured_pack', 0), .5, 0, 5 * gpus / 8, 30))['results'])
    for policy in c.POLICIES:
        assert results[0][policy]['shed_fraction'] == pytest.approx(results[1][policy]['shed_fraction'])
        assert results[0][policy]['admitted_shed_fraction'] == pytest.approx(results[1][policy]['admitted_shed_fraction'])
    assert results[0]['replay_only']['shed_fraction'] > 0


def test_independent_regional_fidelity_passes_without_hiding_the_old_error(tmp_path):
    c.validate(tmp_path)
    validation = json.loads((tmp_path / 'validation.json').read_text())
    report = validation['regional_fidelity']
    assert report['current_pool']['aggregate']['episodes'] == 24
    assert not report['current_pool']['gate_pass'] and report['frozen_oracle']['gate_pass']
    assert validation['regional_execution']['gate_pass']
    assert validation['regional_execution']['aggregate']['mae_s'] < 3
    assert validation['regional_execution']['aggregate']['false_feasible_25s'] == 1


def test_isolated_fastest_masks_exist_in_the_common_library():
    fleet = c.sample_fleet('coding')
    r, k = c.library(fleet)
    for load, wan in ((.25, 10), (.5, 40), (.95, 400)):
        endpoint = c.network_samples()[0]
        r, k = c.include_isolated(r, k, c.isolated_methods(fleet, load, endpoint,
                                  c.bandwidth(endpoint, fleet.gpus, wan), calibration(0)['timing'][0]))
        table = c.schedule_table(fleet, r, k, load, 60, endpoint,
                                 c.bandwidth(endpoint, fleet.gpus, wan), calibration(0)['timing'][0])
        signatures = set(map(tuple, np.c_[r, k]))
        for counts in r + k:
            desired = np.r_[counts * table.fastest, counts * ~table.fastest]
            assert tuple(desired) in signatures


def test_execution_draws_preserve_initial_information_and_admission_accounting():
    plan = {'identity': 'feedback', 'config': c.configuration(), 'calibration': calibration(2), 'network_indices': [-1, 0, 1]}
    results = [c.run_cell(plan, (('coding', 0), .5, draw, 1000, 30))['results'] for draw in range(3)]
    for policy in c.POLICIES:
        for result in results[1:]:
            assert result[policy]['initial_nominal_shed_fraction'] == results[0][policy]['initial_nominal_shed_fraction']
            assert result[policy]['planning_diagnostics'][0] == results[0][policy]['planning_diagnostics'][0]
        assert sum(results[0][policy]['action_fractions']) <= sum(results[0][policy]['admitted_action_fractions']) + 1e-8


def test_campaign_reduction_checks_all_cells_and_policies(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'plot', lambda *args: None)
    plan = c.prepare(tmp_path, smoke=True)
    assert len(c.cells(plan['config'])) == 24
    with pytest.raises(ValueError, match='missing'):
        c.reduce(tmp_path)
    c.run(tmp_path)
    summary = c.reduce(tmp_path)
    assert summary['cells'] == 24
    audit = json.loads((tmp_path / 'dominance-audit.json').read_text())
    assert audit['maxima']['initial_nominal_lp_loss'] <= 1e-8
    c.run(tmp_path)  # Provenance-valid checkpoints may be resumed.
    path = tmp_path / 'cells/000000.json.gz'
    with gzip.open(path, 'rt') as handle:
        value = json.load(handle)
    del value['results']['replay_only']
    c.write_json(path, value)
    with pytest.raises(ValueError, match='missing policy'):
        c.reduce(tmp_path)


def test_configuration_and_provenance_fail_closed(tmp_path):
    for options in ({'resident_loads': [1.]}, {'resident_loads': [.5, .5]}, {'draws': -1},
                    {'snapshots': 0}, {'wan_gbps': [float('nan')]}, {'gpus_per_node': 0}, {'gpus_per_node': 1.5}):
        with pytest.raises(ValueError):
            c.prepare(tmp_path, **options)
    c.prepare(tmp_path, smoke=True)
    with pytest.raises(ValueError, match='different inputs'):
        c.prepare(tmp_path, smoke=True, draws=2)
    plan = json.loads((tmp_path / 'plan.json').read_text())
    plan['config']['source_load'] = .9
    c.write_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match='identity'):
        c.load_plan(tmp_path)
