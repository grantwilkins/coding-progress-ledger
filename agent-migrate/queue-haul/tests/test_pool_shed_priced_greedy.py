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
        assert info['objective'] <= 6.5 <= info['upper_bound']


def test_near_ties_use_debt_but_bound_covers_the_actual_maximum():
    chosen, info = priced_greedy(np.ones((1, 2)), np.ones(1), np.array([1., 1 - 1e-13]), np.zeros(2),
                                 debt=np.array([10., 0.]), max_iterations=2)
    assert chosen == pytest.approx([0., 1.])
    assert info['upper_bound'] >= 1.
    assert info['converged'] and info['iterations'] == 1


def test_roundoff_in_the_incumbent_is_repaired_without_mutating_inputs():
    args = [np.ones((1, 1)), np.ones(1), np.ones(1), np.array([1 + 1e-12])]
    originals = [x.copy() for x in args]
    chosen, info = priced_greedy(*args)
    assert chosen == pytest.approx([1.]) and info['objective'] == 1.
    assert info['absolute_gap'] == info['upper_bound'] - info['objective']
    for original, actual in zip(originals, args):
        np.testing.assert_array_equal(original, actual)


@pytest.mark.parametrize('budget', [0, 1])
def test_iteration_exhaustion_preserves_incumbent_and_a_valid_bound(budget):
    matrix, capacity, gains = trap()
    chosen, info = priced_greedy(matrix, capacity, gains, np.r_[1., np.zeros(16)], max_iterations=budget)
    assert info['objective'] >= 1.001
    assert info['upper_bound'] >= 16.
    assert info['iterations'] == budget and not info['converged']
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


def test_price_floor_keeps_the_certificate_finite_after_large_log_separation(monkeypatch):
    original, clipped = np.exp, []

    def record(values):
        if values.min() == -700.:
            clipped.append(True)
        return original(values)

    monkeypatch.setattr(np, 'exp', record)
    chosen, info = priced_greedy(np.eye(2), np.ones(2), np.array([1., 1e-320]), np.zeros(2),
                                 tolerance=1e-14, max_iterations=7500)
    assert clipped and np.isfinite(info['upper_bound'])
    assert np.all(chosen <= 1) and info['objective'] == pytest.approx(1.)
    assert info['upper_bound'] >= 1. and not info['converged']
