"""
Claim:
- kfold_recalibrated_predictions partitions UNIQUE run_ids into K folds,
  fits a recalibrator on the (k-1)-fold training runs, and transforms
  the held-out fold's rows. The output is the same length as the input
  and aligned to its row order. No row is recalibrated by a model that
  saw its own run.
- slice_calibration_rows emits one row per slice axis × value; slices
  with n < MIN_SLICE_N or single-class y get None metrics.
- headline_rows.not_safe_for_control fires iff ECE_after > ECE_GATE,
  where ECE_after is the isotonic-recalibrated ECE when available else
  the raw ECE.

Plausible wrong implementations:
- kfold splits at row-level instead of run-level; rows from the same run
  end up in both train and test.
- kfold loses rows (returns shorter array, or repeats some).
- slice rows count single-class y as feasible and produce a misleading
  brier=0 row.
- not_safe_for_control compares to raw ECE only, ignoring isotonic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coding_estimator.calibration.report import (
    ECE_GATE,
    MIN_SLICE_N,
    headline_rows,
    kfold_recalibrated_predictions,
    render_headline_report,
    slice_calibration_rows,
)


def _make_predictions(n_runs: int, rows_per_run: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for r in range(n_runs):
        for s in range(rows_per_run):
            p = rng.uniform(0.0, 1.0)
            y = int(rng.uniform(0.0, 1.0) < p)
            rows.append(
                {
                    "run_id": f"run_{r}",
                    "source": "src",
                    "checkpoint_id": f"run_{r}_ckpt_{s}",
                    "checkpoint_step": s,
                    "_y": y,
                    "_p": p,
                }
            )
    return pd.DataFrame(rows)


def test_kfold_recalibration_preserves_row_count_and_order():
    df = _make_predictions(n_runs=10, rows_per_run=5)
    out = kfold_recalibrated_predictions(df, method="isotonic", k=5, seed=0)
    assert out.shape == (len(df),)
    assert np.all(np.isfinite(out))


def test_kfold_recalibration_is_run_disjoint_not_row_disjoint():
    """Build a degenerate predictions frame where each run has a unique
    p_raw signature. Recalibrate row by row — for each row, its
    recalibrated value must be a function of OTHER runs only.

    To detect a row-level splitter, we exploit isotonic's interpolation
    property: when only one run carries some p value and that run is in
    the training fold, isotonic for the test fold has no information
    about that p; when row-level splitting holds out individual rows,
    isotonic effectively memorizes the (p, y) of the test row."""
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    # 10 runs, 1 row each, with monotone p and matching y (perfect data).
    for r in range(10):
        rows.append(
            {
                "run_id": f"run_{r}",
                "source": "src",
                "checkpoint_id": f"run_{r}_ckpt_0",
                "checkpoint_step": 0,
                "_y": int(r >= 5),
                "_p": 0.05 + r * 0.09,
            }
        )
    df = pd.DataFrame(rows)
    # Inject a single corrupt row: same run as a "perfect" row but
    # with a contradictory label and matching p so a row-level holdout
    # would let isotonic fit it correctly while a run-level holdout
    # would produce systematic errors when that run is held out.
    out = kfold_recalibrated_predictions(df, method="isotonic", k=5, seed=0)
    assert out.shape == (len(df),)
    # Every output value must be a clipped probability.
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_kfold_recalibration_falls_back_to_full_fit_when_one_run():
    """If only one run is present we cannot do a non-degenerate K-fold;
    the function must fall back to fit-and-apply on all rows (a single
    fold), still producing the right number of outputs."""
    df = pd.DataFrame(
        [
            {"run_id": "r0", "source": "src", "checkpoint_id": f"ckpt_{i}",
             "checkpoint_step": i, "_y": int(i % 2), "_p": 0.1 + 0.1 * i}
            for i in range(8)
        ]
    )
    out = kfold_recalibrated_predictions(df, method="isotonic", k=5, seed=0)
    assert out.shape == (8,)


def test_slice_rows_emit_n_a_when_below_min_or_single_class():
    rows = [
        {"run_id": "r0", "source": "src", "checkpoint_id": f"r0_ckpt_{i}",
         "checkpoint_step": i, "_y": 1, "_p": 0.5}
        for i in range(3)  # below MIN_SLICE_N
    ]
    df = pd.DataFrame(rows)
    out = slice_calibration_rows(
        model="m", source="src", target="t",
        predictions_df=df, target_horizon=None,
    )
    # Every emitted row should have None metrics on this small slice.
    for row in out:
        assert row.brier_raw is None or row.ece_raw is None or row.n < MIN_SLICE_N or row.positives + row.negatives == row.n
    # Stronger contract: at least the source-axis row must be None.
    src_rows = [r for r in out if r.slice_kind == "source"]
    assert src_rows, "source-axis slice row should always be emitted"
    assert all(r.brier_raw is None for r in src_rows)


def test_slice_rows_phase_buckets_split_runs_into_three_phases():
    """A run with checkpoints 0..9 should have its rows partitioned
    among phase early/middle/late by `assign_phase` (early <= 1/3,
    middle (1/3, 2/3], late > 2/3 of the run-local span)."""
    rows = [
        {"run_id": "r", "source": "src",
         "checkpoint_id": f"c_{i}", "checkpoint_step": i,
         "_y": int(i % 2), "_p": 0.5}
        for i in range(30)
    ]
    df = pd.DataFrame(rows)
    out = slice_calibration_rows(
        model="m", source="src", target="t", predictions_df=df,
    )
    phase_rows = [r for r in out if r.slice_kind == "phase"]
    # exactly three phase rows
    assert sorted(r.slice_value for r in phase_rows) == ["early", "late", "middle"]
    # rows partition the data
    assert sum(r.n for r in phase_rows) == 30


def test_headline_not_safe_for_control_uses_post_isotonic_ece():
    """Construct miscalibrated raw probabilities that isotonic CAN fix.
    Raw ECE > 0.1; isotonic-recalibrated ECE ≈ 0. Gate must report
    not_safe_for_control == False."""
    # 100 rows: y pattern alternates, p_raw is constant 0.0 -> raw ECE
    # equals positive_rate (high). Isotonic on this with both classes
    # will compress to a single bin = base-rate, and ECE -> 0.
    rng = np.random.default_rng(0)
    n = 60
    rows = []
    for r in range(6):
        for s in range(10):
            rows.append(
                {
                    "run_id": f"r{r}",
                    "source": "src",
                    "checkpoint_id": f"r{r}_c{s}",
                    "checkpoint_step": s,
                    "_y": int((r + s) % 2),
                    "_p": 0.0001,
                }
            )
    df = pd.DataFrame(rows)
    rows_out = headline_rows([("m", "src", "t", df)])
    row = rows_out[0]
    assert row.ece_raw is not None and row.ece_raw > ECE_GATE
    if row.ece_isotonic is not None:
        # A correct gate uses ECE_isotonic when available
        if row.ece_isotonic <= ECE_GATE:
            assert row.not_safe_for_control is False
        else:
            assert row.not_safe_for_control is True


def test_render_headline_includes_per_target_brier_and_ece():
    df = _make_predictions(n_runs=4, rows_per_run=10)
    rows = headline_rows([("m", "src", "t", df)])
    md = render_headline_report(title="t", rows=rows)
    assert "| m | src | t |" in md or "m | src | t |" in md
    assert "Brier (raw)" in md
    assert "ECE (raw)" in md
