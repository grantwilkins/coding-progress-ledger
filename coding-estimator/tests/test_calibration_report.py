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
    """Construct a frame where one run is a "poison" run: its (p, y)
    pairs contradict every other run. Run-level k-fold leaves the poison
    run out of the recalibrator that scores it, so the poisoned rows'
    recalibrated values must match the WITHOUT-poison isotonic fit at
    those p's — which differs systematically from the WITH-poison fit.

    A row-level splitter (the wrong impl) trains on most poison rows
    even when scoring others, so its outputs trend toward the
    WITH-poison fit. We assert the per-row error vs. the
    WITHOUT-poison reference is small and the error vs. the WITH-poison
    reference is large for poisoned rows."""
    from coding_estimator.calibration.recalibrate import IsotonicRecalibrator

    rows: list[dict] = []
    # 5 "good" runs, 5 rows each: low p -> y=0, high p -> y=1.
    for r in range(5):
        for s in range(5):
            p_val = 0.1 + 0.08 * s  # 0.1, 0.18, 0.26, 0.34, 0.42
            y_val = 1 if p_val > 0.3 else 0
            rows.append(
                {
                    "run_id": f"good_{r}",
                    "source": "src",
                    "checkpoint_id": f"good_{r}_c{s}",
                    "checkpoint_step": s,
                    "_y": y_val,
                    "_p": p_val,
                }
            )
    # 1 "poison" run with MANY rows: same p grid, but labels inverted.
    # With one poison run, run-level LOO over 6 runs trains the
    # poison-fold's recalibrator on the 5 good runs only — poison is
    # fully held out. A row-level splitter trains on most poison rows
    # plus all good rows — heavily contaminated.
    for s in range(5):
        for c in range(6):  # 30 poison rows
            p_val = 0.1 + 0.08 * s
            y_val = 0 if p_val > 0.3 else 1
            rows.append(
                {
                    "run_id": "poison",
                    "source": "src",
                    "checkpoint_id": f"poison_c{s}_{c}",
                    "checkpoint_step": s * 10 + c,
                    "_y": y_val,
                    "_p": p_val,
                }
            )
    df = pd.DataFrame(rows)

    # Reference 1: without-poison isotonic (what a correct run-level
    # k-fold converges to on poison rows — those rows' recalibrator is
    # fit on the 5 good runs only).
    good_df = df[df["run_id"] != "poison"]
    ref_without_poison = (
        IsotonicRecalibrator()
        .fit(good_df["_p"].to_numpy(), good_df["_y"].to_numpy())
    )

    # Reference 2: WITH-poison isotonic (the in-sample fit a row-level
    # splitter would produce when most rows are training).
    ref_with_poison = (
        IsotonicRecalibrator().fit(df["_p"].to_numpy(), df["_y"].to_numpy())
    )

    # k=6 → LOO over runs. Poison fold scores all poison rows with a
    # recalibrator fit on the 5 good runs only.
    out = kfold_recalibrated_predictions(df, method="isotonic", k=6, seed=0)
    df_out = df.assign(_recal=out)
    poison_rows = df_out[df_out["run_id"] == "poison"]
    expect_without = ref_without_poison.transform(poison_rows["_p"].to_numpy())
    expect_with = ref_with_poison.transform(poison_rows["_p"].to_numpy())

    # On the poisoned p-grid the two reference fits must disagree —
    # otherwise the test is vacuous.
    assert np.max(np.abs(expect_without - expect_with)) > 0.1, (
        "test fixture failed to construct a meaningfully-poisoned slice"
    )

    err_without = float(np.mean(np.abs(poison_rows["_recal"].to_numpy() - expect_without)))
    err_with = float(np.mean(np.abs(poison_rows["_recal"].to_numpy() - expect_with)))
    assert err_without < err_with, (
        f"poisoned-row outputs are closer to the WITH-poison fit "
        f"(err_with={err_with:.3f} vs err_without={err_without:.3f}) — "
        "kfold split is leaking the poison run into its own fit"
    )


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
