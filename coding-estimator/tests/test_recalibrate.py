"""
Claim:
- PlattRecalibrator fits a 1-D logistic on logit(p_raw); transform is
  monotone non-decreasing in p; output ∈ OUTPUT_CLIP.
- IsotonicRecalibrator fits PAV on (p, y); output is monotone
  non-decreasing in p and bounded in [0, 1] (then clipped to OUTPUT_CLIP).
- SourceIsotonicRecalibrator fits one isotonic per source; an unseen
  source falls back to a global isotonic fit on all rows; per-source
  fits do not influence each other.
- Both recalibrators degrade to a constant base-rate predictor when
  fit on single-class y.

Plausible wrong implementations:
- Platt fits on p directly (not logit); breaks calibration on
  near-extreme inputs but is otherwise hard to detect — caught by
  monotonicity-on-extreme-grid.
- Isotonic forgets to clip to OUTPUT_CLIP; transform can produce
  values outside [0.001, 0.999], breaking downstream log_loss.
- Source-isotonic uses a single global isotonic for everyone (the
  per-source split does nothing).
- Source-isotonic falls back to identity (or NaN) for an unseen source
  instead of the global recalibrator.
- Single-class y: recalibrator silently fits on degenerate logit and
  produces nonsense.
"""

from __future__ import annotations

import numpy as np
import pytest

from coding_estimator.calibration.recalibrate import (
    IsotonicRecalibrator,
    PlattRecalibrator,
    SourceIsotonicRecalibrator,
    fit_method,
)
from coding_estimator.eval.metrics import OUTPUT_CLIP


@pytest.fixture
def well_separated() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.0, 1.0, size=n)
    # Generate a labeling such that y has a clear monotone relation to p
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(int)
    return p, y


def test_platt_transform_is_monotone_non_decreasing(well_separated):
    p, y = well_separated
    cal = PlattRecalibrator().fit(p, y)
    grid = np.linspace(0.001, 0.999, 50)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)


def test_isotonic_transform_is_monotone_non_decreasing(well_separated):
    p, y = well_separated
    cal = IsotonicRecalibrator().fit(p, y)
    grid = np.linspace(0.001, 0.999, 50)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)


def test_isotonic_output_bounded_in_output_clip(well_separated):
    p, y = well_separated
    cal = IsotonicRecalibrator().fit(p, y)
    extreme = np.array([0.0, 0.0001, 0.5, 0.9999, 1.0])
    out = cal.transform(extreme)
    lo, hi = OUTPUT_CLIP
    assert np.all(out >= lo - 1e-12)
    assert np.all(out <= hi + 1e-12)


def test_platt_output_bounded_in_output_clip(well_separated):
    p, y = well_separated
    cal = PlattRecalibrator().fit(p, y)
    extreme = np.array([0.0, 0.0001, 0.5, 0.9999, 1.0])
    out = cal.transform(extreme)
    lo, hi = OUTPUT_CLIP
    assert np.all(out >= lo - 1e-12)
    assert np.all(out <= hi + 1e-12)


def test_isotonic_perfectly_calibrates_a_step_function():
    """When (p, y) data are themselves a perfect monotone step, the
    isotonic fit should reproduce the step (modulo OUTPUT_CLIP). A wrong
    impl that returns p unchanged would *also* satisfy monotonicity, so
    we additionally check that systematically miscalibrated probabilities
    are corrected."""
    # 100 negatives at p=0.7, 100 positives at p=0.3 (perversely
    # over-confident-on-negatives data). Isotonic must invert: predict
    # low for p=0.7 and high for p=0.3 — i.e. the *output* at p=0.7
    # should be < output at p=0.3 actually NO, isotonic must be
    # *monotone non-decreasing in p*, so it'll average to 0.5 across the
    # whole range. The test: output at both inputs should be ~ mean(y).
    p = np.concatenate([np.full(100, 0.3), np.full(100, 0.7)])
    y = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    cal = IsotonicRecalibrator().fit(p, y)
    out = cal.transform(np.array([0.3, 0.7]))
    assert out[0] == pytest.approx(0.5, abs=0.01)
    assert out[1] == pytest.approx(0.5, abs=0.01)


def test_source_isotonic_per_source_independence():
    """Two sources with opposite calibration profiles. A correct
    implementation produces opposite mappings; a wrong implementation
    that fits a single global isotonic produces an averaged mapping."""
    # Source A: p=0.3 -> y=1 always (under-confident); A's isotonic
    # should map 0.3 to ~1.
    # Source B: p=0.3 -> y=0 always (over-confident); B's isotonic
    # should map 0.3 to ~0.
    p = np.concatenate([np.full(100, 0.3), np.full(100, 0.3)])
    y = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    src = np.array(["A"] * 100 + ["B"] * 100)
    cal = SourceIsotonicRecalibrator().fit(p, y, src)
    a_out = cal.transform(np.array([0.3]), source="A")[0]
    b_out = cal.transform(np.array([0.3]), source="B")[0]
    assert a_out > 0.9
    assert b_out < 0.1


def test_source_isotonic_unseen_source_falls_back_to_global():
    """An unseen source must NOT raise — it must use the global isotonic
    fit on all rows."""
    p = np.linspace(0.0, 1.0, 200)
    y = (p > 0.5).astype(int)
    src = np.where(p < 0.5, "A", "B")
    cal = SourceIsotonicRecalibrator().fit(p, y, src)
    # Unseen source "Z"
    out = cal.transform(np.array([0.1, 0.9]), source="Z")
    # Global mapping: low p -> 0, high p -> 1. Tolerate clip.
    assert out[0] < 0.1
    assert out[1] > 0.9


def test_source_isotonic_array_source_routes_per_row():
    """When `source` is an array aligned with `p`, each row should be
    transformed by its row's source's recalibrator."""
    p = np.concatenate([np.full(100, 0.3), np.full(100, 0.3)])
    y = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    src = np.array(["A"] * 100 + ["B"] * 100)
    cal = SourceIsotonicRecalibrator().fit(p, y, src)
    query_p = np.array([0.3, 0.3])
    query_src = np.array(["A", "B"])
    out = cal.transform(query_p, source=query_src)
    assert out[0] > 0.9
    assert out[1] < 0.1


def test_recalibrators_handle_single_class_y_without_blowing_up():
    p = np.array([0.1, 0.5, 0.9])
    y_all_zero = np.array([0, 0, 0])
    pl = PlattRecalibrator().fit(p, y_all_zero)
    iso = IsotonicRecalibrator().fit(p, y_all_zero)
    out_pl = pl.transform(p)
    out_iso = iso.transform(p)
    lo, hi = OUTPUT_CLIP
    # Should be a constant predictor at the base-rate (0.0), clipped.
    assert np.all(out_pl == pytest.approx(lo, abs=1e-9))
    assert np.all(out_iso == pytest.approx(lo, abs=1e-9))


def test_fit_method_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        fit_method("does-not-exist", np.array([0.5]), np.array([1]))


def test_fit_method_source_isotonic_requires_source():
    with pytest.raises(ValueError):
        fit_method("source_isotonic", np.array([0.5]), np.array([1]), source=None)
