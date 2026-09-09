from types import SimpleNamespace

import numpy as np

import pool_replay_validation as validation


def fleet_and_manifest():
    rows = [{'session_id': 'long', 'turn': 0, 'input_tokens_total': 27000,
             'newly_append_tokens': 100, 'output_tokens': 10, 'reset': False},
            {'session_id': 'long', 'turn': 1, 'input_tokens_total': 1000,
             'newly_append_tokens': 1000, 'output_tokens': 10, 'reset': True},
            {'session_id': 'unsupported', 'turn': 0, 'input_tokens_total': 32000,
             'newly_append_tokens': 32, 'output_tokens': 1000, 'reset': False}]
    sequence = [{'context': r['input_tokens_total'] - r['newly_append_tokens'],
                 'prompt': r['newly_append_tokens'], 'output': r['output_tokens'],
                 'reset': r['reset']} for r in rows[:2]]
    fleet = SimpleNamespace(count=np.array([8]), metadata={
        'sampled_states': rows[:1], 'turn_sequences': [sequence], 'turn_offset': [0],
        'excluded_states': 1, 'exclusion_reason': 'unsupported', 'reference_rps': .4})
    return fleet, {'manifest': {'splits': {'coding': {'train': ['long', 'unsupported']}}},
                   'traces': rows}


def test_freeze_keeps_short_reset_after_initial_long_context():
    fleet, manifest = fleet_and_manifest()
    result = validation.trajectories(fleet, manifest, 32256)
    assert result['recorded_rows'] == manifest['traces'][:2]
    assert result['turn_sequences'][0][1] == {'context': 0, 'prompt': 1000, 'output': 10, 'reset': True}
    assert result['excluded_trajectories'][0]['session_id'] == 'unsupported'
    assert result['excluded_trajectories'][0]['turns'] == [0]
    assert result['initial_scout_rps_per_gpu'] == .2


def test_frozen_matrix_preserves_pairs_order_and_acquisition_limit():
    fleet, manifest = fleet_and_manifest()
    plan = validation.make_plan({'replay_context_tokens': [2048, 32256]},
                                {w: fleet for w in validation.WORKLOADS}, manifest)
    assert len(plan['unloaded_trials']) == 24
    assert len(plan['main_episodes']) == 24
    assert len(plan['followups']) == 4
    for row in plan['unloaded_trials']:
        assert row['order'].index('initial') < row['order'].index('catch_up')
        assert row['retained_tokens'] + row['append_tokens'] + 512 <= 32768
    for workload in validation.WORKLOADS:
        for slot in (0, 1):
            arms = [[r['arm'] for r in plan['main_episodes'] if r['seed'] == seed
                     and r['workload'] == workload and r['rate_slot'] == slot]
                    for seed in validation.SEEDS]
            assert arms[0] == arms[1][::-1]
    budget = plan['budget']
    assert sum(budget[k] for k in ('unloaded_cap_s', 'scout_cap_s', 'main_cap_s',
                                  'followup_cap_s', 'startup_and_cleanup_reserve_s')) == 9000
    assert plan['simulator_verification']['evaluations'] == 20
