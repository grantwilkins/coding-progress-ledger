"""
Claim:
Convergence plots report real feasibility residuals, with semilog views masking
zero residuals instead of drawing artificial positive floors.

Plausible wrong implementations:
- Clip zero feasibility gaps to a fake positive value for semilog plotting.
- Use only shed violation and ignore capacity violations.
- Sum constraint violations instead of reporting the maximum normalized residual.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.run_catalog_sweep import _feasibility_gap, _positive_for_semilog


def test_semilog_masks_nonpositive_values_without_changing_positive_gaps():
    plotted = _positive_for_semilog(np.array([-1.0, 0.0, 1e-3]))
    assert np.isnan(plotted[0])
    assert np.isnan(plotted[1])
    assert plotted[2] == 1e-3


def test_feasibility_gap_is_max_normalized_constraint_violation():
    problem = SimpleNamespace(B_shed=10.0)
    hist = {
        "shed_violation": np.array([0.0, 2.0, 1.0]),
        "max_net_util": np.array([0.9, 1.1, 0.8]),
        "max_prefill_util": np.array([1.2, 0.95, 1.05]),
    }
    np.testing.assert_allclose(_feasibility_gap(problem, hist), [0.2, 0.2, 0.1])
