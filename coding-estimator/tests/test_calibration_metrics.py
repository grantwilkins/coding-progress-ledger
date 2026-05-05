"""
Claim:
- brier(y, p) = mean((p - y)^2).
- expected_calibration_error(y, p, n_bins) = sum_b (n_b/n) * |avg_p_b - avg_y_b|,
  with equal-width bins on [0, 1].
- reliability_table(y, p, n_bins) emits one row per equal-width bin on
  [0, 1]; per-bin gap = avg_predicted - avg_observed; empty bins emit
  count=0 with None means and None gap.

Plausible wrong implementations:
- Brier returns sum or sum/2 instead of mean
- ECE uses unweighted mean abs gap (no n_b/n weighting)
- ECE uses equal-frequency bins instead of equal-width
- Reliability table puts all rows of p=1.0 into bin n_bins (off-by-one),
  losing the right edge
- Empty-bin row drops out instead of being represented as count=0
- gap sign is flipped (avg_y - avg_p)
"""

from __future__ import annotations

import numpy as np
import pytest

from coding_estimator.calibration.metrics import (
    brier,
    expected_calibration_error,
    reliability_table,
)


def test_brier_hand_worked_mean():
    """y=[0,1,1,0], p=[0.0,1.0,0.5,0.5].
    (0-0)^2 + (1-1)^2 + (1-0.5)^2 + (0-0.5)^2 = 0 + 0 + 0.25 + 0.25 = 0.5
    mean = 0.5/4 = 0.125."""
    y = np.array([0, 1, 1, 0])
    p = np.array([0.0, 1.0, 0.5, 0.5])
    assert brier(y, p) == pytest.approx(0.125)


def test_ece_is_count_weighted_not_uniform():
    """Construct a bin distribution where unweighted-mean and weighted-ECE
    diverge: 9 rows with p=0.95, y=1 (gap=0.05); 1 row with p=0.05, y=1
    (gap=0.95). Equal-width 10-bin ECE is (9/10)*0.05 + (1/10)*0.95 = 0.140.
    The unweighted-mean wrong impl returns (0.05 + 0.95)/2 = 0.500."""
    y = np.array([1] * 9 + [1])
    p = np.array([0.95] * 9 + [0.05])
    got = expected_calibration_error(y, p, n_bins=10)
    assert got == pytest.approx(0.14, abs=1e-9)


def test_ece_uses_equal_width_not_equal_frequency_bins():
    """All p in [0.0, 0.1) — they all fall in bin 0. y all 1. Average p
    in bin 0 = mean of values which is small; average y = 1; gap large.
    With equal-frequency binning the algorithm would split the rows into
    n_bins quantile groups and produce gap≈0 within each (avg_p == p in
    each one-row group); ECE -> 0."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 0.1, size=10)
    y = np.ones(10, dtype=int)
    got = expected_calibration_error(y, p, n_bins=10)
    assert got > 0.85, f"equal-width ECE on this case must be ~ 1 - mean(p), got {got}"


def test_reliability_emits_n_bins_rows_even_when_some_are_empty():
    rt = reliability_table(np.array([1, 0]), np.array([0.95, 0.05]), n_bins=5)
    assert len(rt) == 5
    # only bins 0 and 4 are non-empty
    counts = rt.sort_values("bin_index")["count"].tolist()
    assert counts == [1, 0, 0, 0, 1]
    # empty bins must NOT carry 0.0 for averages and gap (pandas may
    # coerce None to NaN in a float column — both are "missing")
    import pandas as pd

    for col in ("avg_predicted", "avg_observed", "gap"):
        for empty_bin in (1, 2, 3):
            cell = rt.loc[rt["bin_index"] == empty_bin, col].iloc[0]
            assert cell is None or pd.isna(cell), (
                f"empty bin must be None/NaN, got {cell!r} in column {col}"
            )
            # And critically must NOT be 0 (the key wrong impl).
            assert not (isinstance(cell, (int, float)) and cell == 0.0)


def test_reliability_gap_is_avg_predicted_minus_avg_observed():
    """y=0, p=0.7 -> gap = 0.7 - 0 = +0.7 (overpredicting). Sign matters
    because the calibration report displays gap directly."""
    rt = reliability_table(np.array([0]), np.array([0.7]), n_bins=10)
    row = rt[rt["count"] == 1].iloc[0]
    assert row["gap"] == pytest.approx(0.7)


def test_reliability_right_edge_p1_lands_in_last_bin():
    """digitize boundary regression: p=1.0 must NOT spill into bin n_bins
    (off-by-one); it should land in bin n_bins - 1."""
    rt = reliability_table(np.array([1, 1, 1]), np.array([1.0, 1.0, 1.0]), n_bins=4)
    assert int(rt.loc[rt["bin_index"] == 3, "count"].iloc[0]) == 3
    assert int(rt["count"].sum()) == 3


def test_brier_matches_ece_when_bin_collapses_perfectly_calibrated():
    """If p == y on every row, Brier = 0 and ECE = 0 (every bin's gap = 0)."""
    y = np.array([0, 1, 1, 0, 1])
    p = y.astype(float)
    assert brier(y, p) == 0.0
    assert expected_calibration_error(y, p, n_bins=10) == 0.0
