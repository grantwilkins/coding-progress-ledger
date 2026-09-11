from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import planner_sparse_scaling as scaling


def test_individual_histories_preserve_block_mix_phases_and_per_gpu_capacity():
    base = scaling.q.Fleet(count=np.array([32., 32.]), context=np.array([100., 200.]),
        prompt=np.array([1., 2.]), output=np.array([3., 4.]), t1=np.array([.1, .2]),
        kv=np.array([10., 20.]), log=np.array([2., 4.]), demand=np.array([.01, .02]),
        templates=[[0, 1]], gpus=8, kv_capacity=10000., gpus_per_node=8,
        metadata=dict(source_session_rps=.25, source_phase_s=[1., 2.],
                      turn_sequences=[[dict(output=3)], [dict(output=4)]], source_power=dict(full=1.)))
    fleet = scaling.individual_fleet(base, 128)
    assert len(fleet.count) == fleet.count.sum() == 128
    assert fleet.gpus == fleet.metadata['destination_gpus'] == 16
    assert fleet.kv_capacity / fleet.gpus == base.kv_capacity / base.gpus
    assert fleet.count @ fleet.demand / fleet.gpus == pytest.approx(base.count @ base.demand / base.gpus)
    np.testing.assert_array_equal(fleet.context[:64], fleet.context[64:])
    assert fleet.metadata['source_phase_s'][:64] == fleet.metadata['source_phase_s'][64:]
    assert fleet.metadata['source_session_rps'] == .25
    assert sum(fleet.templates, []) == list(range(128)) and all(len(t) == 8 for t in fleet.templates)
    assert len(set(fleet.metadata['individual_history_ids'])) == 128
    fleet.context[0] = 999
    assert base.context[0] == 100 and 'source_power' in base.metadata
    with pytest.raises(ValueError, match='64-history'):
        scaling.individual_fleet(base, 65)


@pytest.mark.parametrize('policy', scaling.POLICIES)
def test_timing_observers_preserve_production_admission_and_sparse_witness(policy):
    fleet = SimpleNamespace(count=np.array([10.]), context=np.array([100.]), demand=np.array([.05]),
        memory_tokens=np.array([100.]), baseline_kv=0., kv_capacity=1e6, gpus=10, nodes=2,
        gain=np.array([.1]), t1=np.array([4.]), log=np.array([2.]), kv=np.array([100.]),
        metadata=dict(turn_sequences=[[]], source_session_rps=0., resident_affinity=True))
    timing = dict(kappa=1., beta=1., kv_completion_s=1., kv_batch_completion_s=1., resident_replay_loss=1.)
    measured = dict(kv_block_tokens=1, kv_block_bytes=1., kv_tail_replay_tps=100., switch_s=0., F=100., G=10.)
    table = SimpleNamespace(fleet=fleet, replay=csr_matrix([[1.], [0.], [1.], [0.]]),
        kv=csr_matrix([[0.], [1.], [0.], [1.]]), route=np.array([0, 0, 1, 1]), load=.5,
        deadline=20., endpoint=np.array([100., 100., 200.]), budgets=np.array([200., 200., 200.]),
        timing=timing, nominal_commit=np.array([6., 3., 6., 3.]), gains=np.full(4, .1), fastest=np.array([False]))
    expected = scaling.planner.plan_admission(scaling.PooledExecution(table, timing, measured), table, policy, calibration=measured)
    stats, inputs, witnesses = scaling.admission(table, policy, measured)
    np.testing.assert_allclose(witnesses['admitted'], expected[0], atol=1e-12)
    assert stats['next_decision_s'] == expected[1]
    assert stats['max_relative_constraint_residual'] <= 1e-8
    assert stats['first_admission_s'] >= stats['selection_s'] >= stats['primary_selection_s'] > 0
    assert inputs['matrix_data'].size == stats['matrix_nnz']
    if policy == 'greedy_priced':
        assert stats['absolute_gap'] <= .001 and stats['lp_primary_s'] == stats['lp_secondary_s'] == 0
        assert 'dual' in witnesses
    else:
        assert stats['lp_primary_s'] > 0 and stats['lp_secondary_s'] > 0 and 'primary' in witnesses


def test_plot_uses_distinct_total_and_primary_metrics_and_canonical_styles(tmp_path, monkeypatch):
    import plot_style

    calls, figures = [], []
    apply = plot_style.apply
    monkeypatch.setattr(plot_style, 'apply', lambda: (calls.append(True), apply()))
    close = plt.close
    monkeypatch.setattr(plt, 'close', lambda fig: figures.append(fig) if hasattr(fig, 'axes') else close(fig))
    records = [dict(sessions=64, policies=[dict(policy=p, total_planning_s=10. + i,
                    primary_selection_s=.1 + i) for i, p in enumerate(scaling.POLICIES)])]
    scaling.plot(records, tmp_path)
    assert calls == [True]
    assert (tmp_path / 'planner_scaling.png').is_file() and (tmp_path / 'planner_scaling.pdf').is_file()
    left, right = figures[-1].axes
    assert left.lines[0].get_ydata()[0] == 10. and right.lines[0].get_ydata()[0] == .1
    assert left.lines[1].get_color() == plot_style.POLICY_COLORS['greedy_priced']
    assert left.get_xlabel() == 'Individual source histories'
    close(figures[-1])
