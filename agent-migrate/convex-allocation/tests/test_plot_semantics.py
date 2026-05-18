"""
Claim:
Convergence plots report the best feasible original-objective gap to the CVXPY
oracle, not a shed or capacity feasibility residual.

Plausible wrong implementations:
- Plot the penalized objective instead of the original objective.
- Turn infeasible or missing best-feasible values into finite plotted gaps.
- Plot signed solver noise below the oracle as a negative objective gap.
"""

from __future__ import annotations

import numpy as np

from experiments.run_catalog_sweep import _best_objective_gap


def test_best_objective_gap_keeps_missing_feasible_iterates_masked():
    hist = {"best_feasible_objective": np.array([np.nan, 11.0])}
    gap = _best_objective_gap(hist, 10.0)
    assert np.isnan(gap[0])
    assert gap[1] == 0.1


def test_best_objective_gap_clips_solver_noise_below_oracle():
    hist = {"best_feasible_objective": np.array([9.999999, 10.5])}
    np.testing.assert_allclose(_best_objective_gap(hist, 10.0), [0.0, 0.05])
