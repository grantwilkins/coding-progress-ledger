"""Multiplicative-price packing with a fixed-matrix primal–dual certificate."""

import numpy as np


def priced_greedy(matrix, capacity, gains, incumbent, debt=None, tolerance=.001, max_iterations=200000,
                  absolute_tolerance=0.):
    matrix, capacity, gains, incumbent = (np.asarray(x, float) for x in (matrix, capacity, gains, incumbent))
    debt = np.zeros_like(gains) if debt is None else np.asarray(debt, float)
    if (matrix.ndim != 2 or capacity.shape != (matrix.shape[0],) or gains.shape != (matrix.shape[1],)
            or incumbent.shape != gains.shape or debt.shape != gains.shape):
        raise ValueError("inconsistent packing dimensions")
    if any(not np.isfinite(x).all() or np.any(x < 0) for x in (matrix, capacity, gains, incumbent, debt)):
        raise ValueError("packing inputs must be finite and nonnegative")
    if (not np.isfinite(tolerance) or not 0 < tolerance < 1
            or not np.isfinite(absolute_tolerance) or absolute_tolerance < 0 or isinstance(max_iterations, bool)
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0):
        raise ValueError("invalid packing tolerance or iteration budget")
    consumed = matrix @ incumbent
    blocked = np.any((matrix > 0) & (capacity[:, None] == 0), axis=0)
    rows, ids = capacity > 0, np.flatnonzero((gains > 0) & ~blocked)
    utilization = np.max(consumed[rows] / capacity[rows], initial=0.)
    if (not np.isfinite(consumed).all() or np.any(incumbent[blocked] > 0)
            or not np.isfinite(utilization) or utilization > 1 + 1e-8):
        raise ValueError("infeasible packing incumbent")
    best = incumbent.copy() / max(1., utilization)
    prices = np.zeros(len(capacity))
    prices[~rows] = np.max(np.divide(gains, matrix[~rows], out=np.zeros_like(matrix[~rows]),
                                   where=matrix[~rows] > 0), axis=1, initial=0.)
    objective, bound, iteration = float(gains @ best), 0., 0
    if not np.isfinite(prices).all() or not np.isfinite(objective):
        raise ValueError("packing certificate exceeds finite precision")
    if len(ids):
        normalized = matrix[rows][:, ids] / capacity[rows, None]
        largest = np.max(normalized, axis=0, initial=0.)
        if np.any(largest == 0):
            raise ValueError("unbounded positive-gain packing column")
        upper = 1 / largest
        normalized *= upper
        value, secondary = gains[ids] * upper, debt[ids] / gains[ids]
        if any(not np.isfinite(x).all() for x in (normalized, upper, value, secondary)):
            raise ValueError("packing normalization exceeds finite precision")
        logs = np.zeros(len(normalized))
        total, usage, profit = np.zeros(len(ids)), np.zeros(len(normalized)), 0.
        bound = np.inf
        for iteration in range(max_iterations + 1):
            # A positive price floor preserves a valid dual when exponentials would underflow.
            weights = np.exp(np.maximum(logs, -700.))
            ratios = value / (normalized.T @ weights)
            maximum = float(ratios.max())
            candidate = float(weights.sum() * maximum * (1 + 1e-12))
            if not np.isfinite(candidate):
                raise RuntimeError("nonfinite packing dual bound")
            if candidate < bound:
                prices[rows] = weights * maximum * (1 + 1e-12) / capacity[rows]
                bound = float(capacity @ prices)
            if bound - objective <= max(absolute_tolerance, tolerance * bound) or iteration == max_iterations:
                break
            tied = np.flatnonzero(ratios >= maximum * (1 - 1e-12))
            costs = secondary[tied]
            j = int(tied[np.flatnonzero(costs <= costs.min() * (1 + 1e-12))[0]])
            if iteration % 20000 == 0:
                total.fill(0)
                usage.fill(0)
                profit = 0.
            total[j] += 1
            usage += normalized[:, j]
            profit += value[j]
            scale = max(1., usage.max())
            if profit / scale > objective:
                best = np.zeros(len(gains))
                best[ids] = upper * total / scale
                objective = float(gains @ best)
            logs += .1 * .3 ** (iteration // 20000) * normalized[:, j]
            logs -= logs.max()
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
    return best, dict(objective=objective, upper_bound=bound, absolute_gap=absolute_gap, relative_gap=gap,
                      iterations=iteration, converged=absolute_gap <= max(absolute_tolerance, tolerance * bound))
