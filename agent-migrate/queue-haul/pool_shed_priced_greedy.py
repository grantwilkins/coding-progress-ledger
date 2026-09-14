"""Sparse native coordinate packing with an original-unit primal–dual certificate."""

import numpy as np
from scipy.sparse import csc_matrix, issparse


def priced_greedy(matrix, capacity, gains, incumbent, debt=None, tolerance=.001, max_iterations=10000,
                  absolute_tolerance=0., return_prices=False):
    capacity, gains, incumbent = (np.asarray(x, float) for x in (capacity, gains, incumbent))
    debt = np.zeros_like(gains) if debt is None else np.asarray(debt, float)
    if (np.ndim(matrix) != 2 or capacity.shape != (np.shape(matrix)[0],) or gains.shape != (np.shape(matrix)[1],)
            or incumbent.shape != gains.shape or debt.shape != gains.shape):
        raise ValueError("inconsistent packing dimensions")
    data = matrix.tocoo(copy=False).data if issparse(matrix) else np.asarray(matrix, float)
    if any(not np.isfinite(x).all() or np.any(x < 0) for x in (data, capacity, gains, incumbent, debt)):
        raise ValueError("packing inputs must be finite and nonnegative")
    matrix = csc_matrix(matrix, dtype=float, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("packing coefficients exceed finite precision")
    if (not np.isfinite(tolerance) or not 0 < tolerance < 1
            or not np.isfinite(absolute_tolerance) or absolute_tolerance < 0 or isinstance(max_iterations, bool)
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0):
        raise ValueError("invalid packing tolerance or iteration budget")
    consumed = matrix @ incumbent
    zero_rows = matrix[capacity == 0].tocoo()
    blocked = np.zeros(len(gains), bool)
    blocked[zero_rows.col] = True
    rows, ids = capacity > 0, np.flatnonzero((gains > 0) & ~blocked)
    utilization = np.max(consumed[rows] / capacity[rows], initial=0.)
    if (not np.isfinite(consumed).all() or np.any(incumbent[blocked] > 0)
            or not np.isfinite(utilization) or utilization > 1 + 1e-8):
        raise ValueError("infeasible packing incumbent")
    best = incumbent.copy() / max(1., utilization)
    prices = np.zeros(len(capacity))
    np.maximum.at(prices, np.flatnonzero(~rows)[zero_rows.row], gains[zero_rows.col] / zero_rows.data)
    objective, bound, iteration = float(gains @ best), 0., 0
    if not np.isfinite(prices).all() or not np.isfinite(objective):
        raise ValueError("packing certificate exceeds finite precision")
    if len(ids):
        from _queue_haul_native import packing_coordinate

        normalized = matrix[rows][:, ids].multiply(1 / capacity[rows, None]).tocsc()
        largest = normalized.max(axis=0).toarray().ravel() if normalized.shape[0] else np.zeros(len(ids))
        if np.any(largest == 0):
            raise ValueError("unbounded positive-gain packing column")
        upper = 1 / largest
        normalized.data /= np.repeat(largest, np.diff(normalized.indptr))
        value, secondary = gains[ids] * upper, debt[ids] / gains[ids]
        if np.any(value == 0) or any(not np.isfinite(x).all() for x in (normalized.data, upper, value, secondary)):
            raise ValueError("packing normalization exceeds finite precision")
        allocation, dual, iteration = packing_coordinate(*normalized.shape, normalized.indptr, normalized.indices,
            normalized.data, value, best[ids] / upper, secondary, int(max_iterations), tolerance, absolute_tolerance)
        best = np.zeros(len(gains))
        best[ids] = upper * allocation
        prices[rows] = dual / capacity[rows]
        objective, bound = float(gains @ best), float(capacity @ prices)
    else:
        best = np.zeros(len(gains))
        objective = 0.
    consumed, covered = matrix @ best, matrix.T @ prices
    if (not np.isfinite(best).all() or not np.isfinite(consumed).all() or np.any(best < 0)
            or np.max(consumed[rows] / capacity[rows], initial=0.) > 1 + 1e-8
            or np.any(best[blocked] > 0)):
        raise RuntimeError("packing primal certificate failed")
    if (not np.isfinite(prices).all() or not np.isfinite(covered).all() or np.any(prices < 0)
            or np.any(covered < gains * (1 - 1e-10)) or not np.isfinite(bound)
            or bound < objective * (1 - 1e-10)):
        raise RuntimeError("packing dual certificate failed")
    gap = max(0., (bound - objective) / bound) if bound else 0.
    absolute_gap = max(0., bound - objective)
    info = dict(objective=objective, upper_bound=bound, absolute_gap=absolute_gap, relative_gap=gap,
                iterations=iteration, converged=absolute_gap <= max(absolute_tolerance, tolerance * bound))
    if return_prices:
        info["dual_prices"] = prices
    return best, info
