import json
from collections import Counter
from pathlib import Path

import pytest

from planner import plan
from pool_planner import candidate_table
from power_model import ExpectedPower
from quick_action_mix import private_bytes, problem_for, profile_for, schedule
from simulate import predict


@pytest.fixture(scope='module')
def inputs():
    return json.loads((Path(__file__).resolve().parents[1] /
                       'outputs/quick-action-mix-20260910/report.json').read_text())


def model_profile(inputs, index):
    return profile_for(inputs['models'][index], [c for pack in inputs['context_packs'] for c in pack])


def test_private_payloads_preserve_restoration_evidence(inputs):
    expected = (789577728, 865075200, 2260729856)
    for model, size in zip(inputs['models'], expected):
        assert private_bytes(32256, model['private_geometry']) == size
        profile = profile_for(model, [14080, 30720, 31488])
        for context in (14080, 30720, 31488):
            assert profile.case().kv_transfer.sealed_bytes(context) == private_bytes(context, model['private_geometry'])
        with pytest.raises(ValueError, match='outside'):
            profile_for(model, [40000])


def test_deadline_is_migration_budget_and_credits_are_equal(inputs):
    for index in range(3):
        profile = model_profile(inputs, index)
        problem, architecture = problem_for(inputs['context_packs'][0], inputs['session_orders'][0],
                                            profile, inputs['bandwidth_mbps'], 5)
        table = candidate_table(problem, profile, architecture, 'normal', ExpectedPower(problem, profile))
        assert table.migration_horizon_s == 5
        assert len({round(c.credit, 12) for c in table.candidates}) == 1
        assert min(c.credit for c in table.candidates) > 0
        assert len(architecture.pools) == 2
        assert all(t.kv_capacity_tokens == profile.kv_capacity_tokens for t in architecture.types)


def test_qwen_deadline_changes_real_planner_actions_and_attainment(inputs):
    profile = model_profile(inputs, 2)
    contexts, order = inputs['context_packs'][0], inputs['session_orders'][0]
    short = schedule(contexts, order, profile, inputs['bandwidth_mbps'], 5)
    long = schedule(contexts, order, profile, inputs['bandwidth_mbps'], 30)
    assert 'not_moved' in {r['action'] for r in short}
    assert 'replay' in {r['action'] for r in short}
    assert all(r['action'] == 'kv_transfer' and r['destination'] == 'germany' for r in long)
    assert all(r['finish_s'] <= 30 for r in long)


def test_saved_cases_match_independent_planner_and_simulator(inputs):
    for index, model in enumerate(inputs['models']):
        profile = model_profile(inputs, index)
        for case in model['cases'][::100]:
            draw, deadline = case['draw'], case['deadline_s']
            problem, architecture = problem_for(inputs['context_packs'][draw], inputs['session_orders'][draw],
                                                profile, inputs['bandwidth_mbps'], deadline)
            result = plan(problem, profile, {}, 'greedy', destination=architecture, admission_mode='normal')
            execution = predict(problem, profile, result.moves, destination=architecture)
            finishes = {r.session_id: r.committed_s for r in execution.sessions}
            selected = {r.session_id: r for r in result.moves}
            for row in case['actions']:
                key = str(row['session'])
                assert row['selected_action'] == (selected[key].method if key in selected else 'not_moved')
                assert row['finish_s'] == finishes.get(key)
                assert (row['action'] != 'not_moved') == (key in selected and finishes[key] is not None
                                                        and finishes[key] <= deadline + 1e-8)
        for summary in model['summary']:
            actions = [r for c in model['cases'] if c['deadline_s'] == summary['deadline_s'] for r in c['actions']]
            counts = Counter(r['action'] for r in actions)
            assert summary['counts'] == {a: counts[a] for a in summary['counts']}
            assert sum(counts.values()) == inputs['resamples'] * 8
