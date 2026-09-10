"""Campaign provenance, measured inputs, and complete paired reductions."""

import csv
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
    for workload in c.WORKLOADS:
        fleet = c.sample_fleet(workload)
        assert fleet.count.sum() == 8 * c.GPUS
        assert fleet.count @ fleet.demand == pytest.approx(.8 * c.GPUS)
        assert fleet.count @ fleet.gain == pytest.approx(1.)
        assert fleet.baseline_kv < fleet.kv_capacity
        r, k = c.library(fleet)
        assert len(r) * 2 <= 2400
        assert np.all((r + k).sum(1) <= 8)
        assert np.all(r * k == 0)
        a, b = c.library(fleet, expanded=True)
        assert set(map(tuple, np.c_[r, k])) <= set(map(tuple, np.c_[a, b]))


def test_default_grid_and_paired_endpoints():
    from pool_shed_execution import destination_gpus

    config = c.configuration()
    assert config['gpus'] == 6666 and config['installed_gpu_w'] == 1999800
    assert config['workloads'] == ['coding', 'coding_long']
    assert {cell[0] for cell in c.cells(config)} == {('coding', i) for i in range(4)} | {('coding_long', 0)}
    assert len(c.cells(config)) == 11250
    assert len(c.cells(c.configuration(smoke=True))) == 24
    for workload in config['workloads']:
        fleet = c.sample_fleet(workload, gpus=config['gpus'])
        assert destination_gpus(fleet) == fleet.gpus == 6666
    samples = c.network_samples()
    np.testing.assert_allclose(samples[:, :2].sum(1), samples[:, 2])
    for endpoint in samples:
        budget = c.bandwidth(endpoint, c.GPUS, 40)
        assert max(budget) <= 40e9 / 8
        np.testing.assert_equal(c.bandwidth(endpoint, c.GPUS, 'reference'), endpoint)


def test_lp_scaling_preserves_source_counts_in_the_loaded_coding_case(tmp_path):
    plan = c.prepare(tmp_path)
    result = c.run_cell(plan, (('coding', 1), .75, 4, 100, 30))
    assert 0 < result['results']['queue_haul']['initial_nominal_shed_fraction'] <= .625 + 1e-8
    assert result['results']['queue_haul']['shed_fraction'] <= .625 + 1e-8
    assert max(r['max_relative_residual'] for r in result['results'].values()) <= 1e-8


def test_scaling_nodes_and_network_together_preserves_the_policy_tradeoff():
    plan = {'identity': 'scaling', 'config': c.configuration(), 'calibration': calibration(0), 'network_indices': [-1]}
    results = []
    for gpus in (6400, 64000):
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
            for desired in (np.r_[counts * table.fastest, np.zeros_like(counts)],
                            np.r_[np.zeros_like(counts), counts * ~table.fastest]):
                assert not desired.any() or tuple(desired) in signatures


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


def test_single_cell_reduction_retains_resident_pooling_diagnostics(tmp_path, monkeypatch):
    config = {**c.configuration(), 'workloads': ['coding'], 'resident_loads': [.5],
              'deadlines': [30], 'wan_gbps': [1000], 'snapshots': 1, 'draws': 0}
    monkeypatch.setattr(c, 'configuration', lambda smoke=False: config.copy())
    monkeypatch.setattr(c, 'plot', lambda *args: None)
    c.prepare(tmp_path)
    c.run(tmp_path)
    assert c.reduce(tmp_path)['cells'] == 1
    rows = list(csv.DictReader((tmp_path / 'scenarios.csv').open()))
    assert len(rows) == len(c.POLICIES)
    for row in rows:
        assert float(row['resident_displaced_work_s']) - float(row['resident_pool_compensation_work_s']) == pytest.approx(float(row['resident_debt_generated_work_s']), abs=1e-6)
        assert 'no resident GPU affinity' in row['service_recovery_scope']


def test_configuration_and_provenance_fail_closed(tmp_path):
    for options in ({'resident_loads': [1.]}, {'resident_loads': [.5, .5]}, {'draws': -1},
                    {'snapshots': 0}, {'wan_gbps': [float('nan')]}, {'gpus_per_node': 0}, {'gpus_per_node': 1.5}):
        with pytest.raises(ValueError):
            c.prepare(tmp_path, **options)
    c.prepare(tmp_path, smoke=True)
    with pytest.raises(ValueError, match='different inputs'):
        c.prepare(tmp_path, smoke=True, draws=2)
    plan = json.loads((tmp_path / 'plan.json').read_text())
    version = plan['config']['solver_version']
    plan['config']['solver_version'] = 'stale'
    plan['identity'] = c.digest({'config': plan['config'], 'sources': plan['sources']})
    c.write_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match='solver version'):
        c.load_plan(tmp_path)
    plan['config']['solver_version'] = version
    plan['identity'] = c.digest({'config': plan['config'], 'sources': plan['sources']})
    plan['config']['source_load'] = .9
    c.write_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match='identity'):
        c.load_plan(tmp_path)


def test_source_pacing_uses_measured_service_capacity_and_long_recorded_contexts():
    ordinary, longer = [c.sample_fleet(w) for w in ('coding', 'coding_long')]
    assert np.average(longer.context, weights=longer.count) > np.average(ordinary.context, weights=ordinary.count)
    assert np.min(longer.context) >= 24576
    assert np.max(longer.context) <= max(calibration(0)['replay_context_tokens'])
    assert ordinary.metadata['source_session_rps'] < .1
    assert ordinary.metadata['timing_load_factor'] < .1
    assert not ordinary.metadata['protect_resident'] and ordinary.metadata['paced_source']
    assert len(set(ordinary.metadata['source_phase_s'])) == len(ordinary.count)
    assert 0 < min(ordinary.metadata['source_phase_s']) < max(ordinary.metadata['source_phase_s']) < 1 / ordinary.metadata['source_session_rps']
    assert ordinary.count @ ordinary.demand == pytest.approx(c.SOURCE_LOAD * ordinary.gpus)
    assert all(max(durations) < 1 / ordinary.metadata['source_session_rps'] for durations in ordinary.metadata['turn_duration_s'])


def test_baselines_share_every_mixed_action_batch_projection():
    fleet = c.sample_fleet('coding_long')
    replay, kv = c.library(fleet)
    replay, kv = c.include_isolated(replay, kv, np.arange(len(fleet.count)) % 2 == 0)
    signatures = set(map(tuple, np.c_[replay, kv]))
    zero = np.zeros(replay.shape[1])
    assert all(tuple(np.r_[row, zero]) in signatures for row in replay if row.any())
    assert all(tuple(np.r_[zero, row]) in signatures for row in kv if row.any())


def test_mixed_batch_resources_are_additive_pure_action_choices():
    fleet = c.sample_fleet('measured_pack')
    replay, kv = np.array([[1, 1, 1, 1, 0, 0, 0, 0.]]), np.array([[0, 0, 0, 0, 1, 1, 1, 1.]])
    endpoint = c.network_samples()[0]
    args = (.5, 60, endpoint, c.bandwidth(endpoint, fleet.nodes, 1000), calibration(0)['timing'][0])
    mixed = c.schedule_table(fleet, replay, kv, *args)
    pure = c.schedule_table(fleet, np.vstack((replay, np.zeros_like(replay))), np.vstack((np.zeros_like(kv), kv)), *args)
    np.testing.assert_allclose(mixed.matrix[:, 0], pure.matrix[:, :2].sum(1), atol=1e-8)
    assert mixed.gains[0] == pytest.approx(pure.gains[:2].sum())


def inherited_case(tmp_path, monkeypatch):
    import hashlib

    config = c.configuration(smoke=True)
    sources, parent_sources = {'pool_shed_campaign.py': 'current'}, {'pool_shed_campaign.py': 'previous'}
    plan = {'config': config, 'sources': sources, 'identity': c.digest({'config': config, 'sources': sources})}
    parent = c.digest({'config': config, 'sources': parent_sources})
    cell = c.cells(config)[0]
    path = tmp_path / 'cells/000000.json.gz'
    c.write_json(path, {'identity': parent, 'cell': cell, 'status': 'complete', 'results': dict.fromkeys(c.POLICIES, {})})
    manifest = {'parent_identity': parent, 'parent_config': dict(config), 'parent_sources': parent_sources,
                'cells_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
    c.write_json(tmp_path / 'inherited-checkpoints.json', manifest)
    plan['inherited_checkpoints_sha256'] = hashlib.sha256((tmp_path / 'inherited-checkpoints.json').read_bytes()).hexdigest()
    c.write_json(tmp_path / 'plan.json', plan)
    monkeypatch.setattr(c, 'calibration', lambda *args: {})
    monkeypatch.setattr(c, 'provenance', lambda *args: sources)
    return plan, manifest, path, cell


def test_inherited_checkpoint_requires_pinned_bytes_and_preserves_original_identity(tmp_path, monkeypatch):
    plan, manifest, path, cell = inherited_case(tmp_path, monkeypatch)
    before = path.read_bytes()
    loaded = c.load_plan(tmp_path)
    value = c.read_checkpoint(path, loaded, cell)
    assert value['identity'] == manifest['parent_identity'] != loaded['identity']
    assert path.read_bytes() == before
    unlisted = path.with_name('000001.json.gz')
    unlisted.write_bytes(before)
    with pytest.raises(ValueError, match='invalid cell'):
        c.read_checkpoint(unlisted, loaded, cell)
    value['identity'] = plan['identity']
    c.write_json(path, value)
    with pytest.raises(ValueError, match='bytes changed'):
        c.read_checkpoint(path, loaded, cell)


@pytest.mark.parametrize('change', ['manifest', 'config', 'parent_identity', 'filename'])
def test_inherited_manifest_fails_closed(tmp_path, monkeypatch, change):
    import hashlib

    plan, manifest, path, cell = inherited_case(tmp_path, monkeypatch)
    if change == 'config':
        manifest['parent_config']['source_load'] = .7
    elif change == 'parent_identity':
        manifest['parent_identity'] = 'unknown'
    elif change == 'filename':
        manifest['cells_sha256']['../000000.json.gz'] = manifest['cells_sha256'][path.name]
    else:
        manifest['cells_sha256'][path.name] = 'changed'
    c.write_json(tmp_path / 'inherited-checkpoints.json', manifest)
    if change != 'manifest':
        plan['inherited_checkpoints_sha256'] = hashlib.sha256((tmp_path / 'inherited-checkpoints.json').read_bytes()).hexdigest()
        c.write_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match='inherited'):
        c.load_plan(tmp_path)


def test_unpinned_parent_checkpoint_is_rejected_and_missing_inherited_cell_not_recomputed(tmp_path, monkeypatch):
    plan, manifest, path, cell = inherited_case(tmp_path, monkeypatch)
    loaded = c.load_plan(tmp_path)
    del plan['inherited_checkpoints_sha256']
    c.write_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match='invalid cell'):
        c.read_checkpoint(path, c.load_plan(tmp_path), cell)
    path.unlink()
    monkeypatch.setattr(c, 'load_plan', lambda out: loaded)
    monkeypatch.setattr(c, 'run_cell', lambda *args: pytest.fail('must not replace missing inherited bytes'))
    with pytest.raises(ValueError, match='missing inherited'):
        c.run(tmp_path)


def test_unpinned_serialized_inheritance_cache_is_rejected(tmp_path, monkeypatch):
    plan, manifest, path, cell = inherited_case(tmp_path, monkeypatch)
    plan["inherited_checkpoints"] = manifest
    del plan["inherited_checkpoints_sha256"]
    c.write_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="pinned manifest"):
        c.load_plan(tmp_path)
