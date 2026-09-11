import numpy as np
import pytest

from pool_shed_priced_greedy import priced_greedy


def trap(size=16):
    return np.column_stack((np.ones(size), np.eye(size))), np.ones(size), np.r_[1.001, np.ones(size)]


def test_prices_escape_irrevocable_greedy_trap_without_a_generic_solver(monkeypatch):
    import highspy
    import scipy.optimize
    import pool_shed_campaign

    def forbidden(*args, **kwargs):
        raise AssertionError("generic solver called")

    monkeypatch.setattr(highspy, 'Highs', forbidden)
    monkeypatch.setattr(scipy.optimize, 'linprog', forbidden)
    monkeypatch.setattr(scipy.optimize, 'minimize', forbidden)
    monkeypatch.setattr(pool_shed_campaign, 'solve_lp', forbidden)
    matrix, capacity, gains = trap()
    chosen, info = priced_greedy(matrix, capacity, gains, np.r_[1., np.zeros(16)], max_iterations=100)
    assert matrix @ chosen == pytest.approx(capacity)
    assert info['objective'] == pytest.approx(16.)
    assert info['upper_bound'] >= 16.
    assert info['converged'] and info['relative_gap'] <= .001


@pytest.mark.parametrize('scale', [1e-6, 1., 1e6])
def test_fleet_and_resource_units_preserve_objective_and_certificate(scale):
    matrix = np.array([[2., 1., 0.], [1., 0., 2.]])
    capacity, gains, incumbent = np.array([3., 2.]), np.array([2.5, 1.5, 2.]), np.array([1., 0., 0.])
    _, reference = priced_greedy(matrix, capacity, gains, incumbent, max_iterations=500)
    for row_scale, column_scale in ((np.ones(2), np.ones(3)), (np.array([1e-12, 1e12]), np.array([1e-6, 1., 1e6]))):
        transformed = matrix * row_scale[:, None] * column_scale
        limits, values = capacity * scale * row_scale, gains * column_scale / scale
        chosen, info = priced_greedy(transformed, limits, values, incumbent * scale / column_scale, max_iterations=500)
        assert np.max((transformed @ chosen - limits) / np.maximum(limits, 1)) <= 1e-8
        assert info['objective'] == pytest.approx(reference['objective'], abs=1e-12)
        assert info['upper_bound'] == pytest.approx(reference['upper_bound'], abs=1e-12)
        assert info['objective'] <= 6.5 + 1e-12 and info['upper_bound'] >= 6.5 - 1e-12


def test_near_ties_use_debt_but_bound_covers_the_actual_maximum():
    chosen, info = priced_greedy(np.ones((1, 2)), np.ones(1), np.array([1., 1 - 1e-13]), np.zeros(2),
                                 debt=np.array([10., 0.]), max_iterations=2)
    assert chosen == pytest.approx([0., 1.])
    assert info['upper_bound'] >= 1.
    assert info['converged'] and info['iterations'] == 0


def test_roundoff_in_the_incumbent_is_repaired_without_mutating_inputs():
    args = [np.ones((1, 1)), np.ones(1), np.ones(1), np.array([1 + 1e-12])]
    originals = [x.copy() for x in args]
    chosen, info = priced_greedy(*args)
    assert chosen == pytest.approx([1.]) and info['objective'] == 1.
    assert info['absolute_gap'] == info['upper_bound'] - info['objective']
    for original, actual in zip(originals, args):
        np.testing.assert_array_equal(original, actual)


@pytest.mark.parametrize('budget', [0, 1])
def test_iteration_budget_preserves_incumbent_and_a_valid_bound(budget):
    matrix, capacity, gains = trap()
    chosen, info = priced_greedy(matrix, capacity, gains, np.r_[1., np.zeros(16)], max_iterations=budget)
    assert info['objective'] >= 1.001
    assert info['upper_bound'] >= 16.
    assert info['iterations'] <= budget
    assert info['converged'] == (info['relative_gap'] <= .001)
    if budget == 0:
        assert not info['converged']
    assert np.all(matrix @ chosen <= capacity)


def test_absolute_tolerance_preserves_the_actual_relative_gap():
    matrix, capacity, gains = trap()
    chosen, info = priced_greedy(matrix, capacity, gains * 1e-5, np.r_[1., np.zeros(16)], absolute_tolerance=.001)
    assert info['converged'] and info['absolute_gap'] <= .001
    assert info['relative_gap'] > .9 and info['iterations'] == 0
    assert np.all(matrix @ chosen <= capacity)


def test_blocked_columns_have_zero_capacity_dual_coverage():
    chosen, info = priced_greedy(np.eye(2), np.array([0., 1.]), np.array([3., 2.]), np.zeros(2))
    assert chosen == pytest.approx([0., 1.])
    assert info['objective'] == pytest.approx(2.)
    assert info['upper_bound'] >= 2. and info['converged']


@pytest.mark.parametrize('matrix,capacity,gains', [
    (np.empty((0, 0)), np.empty(0), np.empty(0)),
    (np.zeros((2, 3)), np.ones(2), np.zeros(3)),
    (np.eye(2), np.zeros(2), np.ones(2)),
])
def test_empty_zero_or_blocked_optimum(matrix, capacity, gains):
    chosen, info = priced_greedy(matrix, capacity, gains, np.zeros(len(gains)))
    assert not chosen.any()
    assert info == dict(objective=0., upper_bound=0., absolute_gap=0., relative_gap=0., iterations=0, converged=True)


@pytest.mark.parametrize('matrix,capacity', [(np.zeros((1, 1)), np.ones(1)), (np.empty((0, 1)), np.empty(0))])
def test_positive_gain_without_a_consumed_resource_is_unbounded(matrix, capacity):
    with pytest.raises(ValueError, match='unbounded'):
        priced_greedy(matrix, capacity, np.ones(1), np.zeros(1))


@pytest.mark.parametrize('field', ['matrix', 'capacity', 'gains', 'incumbent', 'debt'])
@pytest.mark.parametrize('bad', [-1., np.inf, np.nan])
def test_invalid_data_fails(field, bad):
    args = dict(matrix=np.ones((1, 1)), capacity=np.ones(1), gains=np.ones(1), incumbent=np.zeros(1), debt=np.zeros(1))
    args[field].flat[0] = bad
    with pytest.raises(ValueError, match='finite and nonnegative'):
        priced_greedy(**args)


@pytest.mark.parametrize('capacity,incumbent', [(1., 1.01), (0., 1e-20), (1e-12, 2e-12)])
def test_infeasible_incumbent_fails(capacity, incumbent):
    with pytest.raises(ValueError, match='infeasible'):
        priced_greedy(np.ones((1, 1)), np.array([capacity]), np.ones(1), np.array([incumbent]))


@pytest.mark.parametrize('kwargs', [dict(tolerance=0), dict(tolerance=np.nan), dict(max_iterations=-1), dict(max_iterations=1.5),
                                  dict(absolute_tolerance=-1), dict(absolute_tolerance=np.inf)])
def test_invalid_controls_fail(kwargs):
    with pytest.raises(ValueError, match='tolerance or iteration'):
        priced_greedy(np.ones((1, 1)), np.ones(1), np.ones(1), np.zeros(1), **kwargs)


def test_inconsistent_dimensions_fail():
    with pytest.raises(ValueError, match='dimensions'):
        priced_greedy(np.ones((1, 2)), np.ones(1), np.ones(1), np.zeros(1))


def test_extreme_gain_range_retains_a_finite_honest_certificate():
    chosen, info = priced_greedy(np.eye(2), np.ones(2), np.array([1., 1e-320]), np.zeros(2),
                                 tolerance=1e-14, max_iterations=7500)
    assert np.isfinite(info['upper_bound'])
    assert np.all(chosen <= 1) and info['objective'] == pytest.approx(1.)
    assert info['upper_bound'] >= 1. and not info['converged']


@pytest.mark.parametrize('seed', range(8))
def test_sparse_coordinate_certificate_brackets_an_independent_lp(seed):
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix

    rng = np.random.default_rng(seed)
    matrix = rng.uniform(.05, 1., (8, 24)) * (rng.random((8, 24)) < .3)
    matrix[0] += .01
    capacity, gains = rng.uniform(1, 3, 8), rng.uniform(.01, .1, 24)
    sparse = csc_matrix(matrix)
    chosen, info = priced_greedy(sparse, capacity, gains, np.zeros(24), absolute_tolerance=.001,
                                 tolerance=1e-12, return_prices=True)
    optimum = linprog(-gains, A_ub=sparse, b_ub=capacity, bounds=(0, None), method='highs')
    assert optimum.success
    assert info['converged'] and info['absolute_gap'] <= .001
    assert np.max((matrix @ chosen) / capacity) <= 1 + 1e-8
    assert np.all(matrix.T @ info['dual_prices'] >= gains * (1 - 1e-10))
    assert capacity @ info['dual_prices'] == pytest.approx(info['upper_bound'])
    assert info['objective'] <= -optimum.fun + 1e-8 <= info['upper_bound'] + 1e-8
    dense, _ = priced_greedy(matrix, capacity, gains, np.zeros(24), absolute_tolerance=.001, tolerance=1e-12)
    np.testing.assert_allclose(chosen, dense, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('gains,secondary', [
    ([1., 1 - 8e-13, 1 - 12e-13], [10., 0., 0.]),
    ([1., 1., 1.], [1 + 8e-13, 1., 1 + 12e-13]),
    ([1., 1 - 8e-13, 1 - 12e-13], [1 + 8e-13, 1., 0.]),
])
def test_native_seed_reprices_all_near_ties_before_selecting(gains, secondary):
    from pool_shed_planner import _choose

    matrix, gains, secondary = np.ones((1, 3)), np.asarray(gains), np.asarray(secondary)
    debt = gains * secondary
    expected = _choose(matrix, np.ones(1), gains, debt, None, True)
    chosen, info = priced_greedy(matrix, np.ones(1), gains, np.zeros(3), debt, max_iterations=0)
    np.testing.assert_array_equal(chosen, expected)
    assert info['iterations'] == 0


def test_corrupt_native_dual_cannot_pass_original_unit_checks(monkeypatch):
    import _queue_haul_native

    monkeypatch.setattr(_queue_haul_native, 'packing_coordinate', lambda *a: (np.ones(2), np.zeros(2), 0))
    with pytest.raises(RuntimeError, match='dual certificate'):
        priced_greedy(np.eye(2), np.ones(2), np.ones(2), np.zeros(2))


@pytest.mark.parametrize('format', ['csc', 'csr', 'coo', 'lil', 'dok'])
def test_sparse_formats_preserve_coefficients_and_solution(format):
    from scipy.sparse import eye

    chosen, info = priced_greedy(eye(2, format=format), np.ones(2), np.ones(2), np.zeros(2))
    np.testing.assert_array_equal(chosen, np.ones(2))
    assert info['converged']


def test_feedback_compression_preserves_the_original_primary_certificate():
    from pathlib import Path
    from pool_shed_planner import _choose_priced

    path = Path(__file__).resolve().parents[1] / 'outputs/a100-native-greedy-20260911/validation/native/feedback-regression.npz'
    with np.load(path) as data:
        matrix, capacity, gains, debt, reference, dual = (data[key] for key in
            ('matrix', 'capacity', 'gains', 'debt', 'reference_allocation', 'reference_dual'))
    chosen, info = _choose_priced(matrix, capacity, gains, debt, None)
    positive = capacity > 0
    assert np.isfinite(chosen).all() and np.all(chosen >= 0)
    assert not np.any((matrix @ chosen)[~positive] > 0)
    assert np.max((matrix @ chosen)[positive] / capacity[positive]) <= 1 + 1e-8
    assert np.all(matrix.T @ dual >= gains * (1 - 1e-10))
    assert gains @ chosen >= gains @ reference - .001
    assert 0 <= capacity @ dual - gains @ chosen <= .001
    assert info['converged'] and info['absolute_gap'] <= .001
    assert np.count_nonzero(chosen) < np.count_nonzero(reference)


def test_greedy_covering_witness_tightens_zero_budget_bound():
    matrix, capacity, gains = np.array([[1., 1.], [1., 0.]]), np.ones(2), np.ones(2)
    chosen, info = priced_greedy(matrix, capacity, gains, np.zeros(2), max_iterations=0, return_prices=True)
    assert info['iterations'] == 0 and info['objective'] == pytest.approx(1.)
    assert info['upper_bound'] == pytest.approx(1., rel=2e-12)
    assert np.all(matrix.T @ info['dual_prices'] >= gains * (1 - 1e-10))
    assert np.all(matrix @ chosen <= capacity * (1 + 1e-10))


def test_late_feedback_uses_a_tight_covering_witness_without_relaxing_primal_checks():
    from pathlib import Path
    from pool_shed_planner import _choose_priced

    path = Path(__file__).resolve().parents[1] / 'outputs/a100-native-greedy-20260911/validation/native/convergence-regression.npz'
    with np.load(path) as data:
        matrix, capacity, gains, debt, reference, dual = (data[key] for key in
            ('matrix', 'capacity', 'gains', 'debt', 'reference_allocation', 'reference_dual'))
    chosen, info = _choose_priced(matrix, capacity, gains, debt, None)
    positive = capacity > 0
    assert np.isfinite(chosen).all() and np.all(chosen >= 0)
    assert not np.any((matrix @ chosen)[~positive] > 0)
    assert np.max((matrix @ chosen)[positive] / capacity[positive]) <= 1 + 1e-8
    assert np.all(matrix.T @ dual >= gains * (1 - 1e-10))
    assert -1e-8 <= gains @ (reference - chosen) <= .001
    assert 0 <= capacity @ dual - gains @ chosen <= .001
    assert info['converged'] and info['absolute_gap'] <= .001
