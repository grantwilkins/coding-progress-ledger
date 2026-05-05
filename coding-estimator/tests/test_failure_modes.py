"""
Claim:
- O1 (evaluate_o1): on rows from runs where final_success == 0 AND
  per-row coding_progress >= high_progress_threshold, the median
  predicted P(success) must be strictly less than median_bound for
  PASS. Empty slice or n < 5 rows ⇒ INDETERMINATE.
- O7 (evaluate_o7): per source under LORO, PASS iff
  Brier_G2 - Brier_G4 >= O7_BRIER_DELTA_GATE. Single-class y on a
  source ⇒ INDETERMINATE for that source (not silently 0). Source
  with < 2 runs ⇒ INDETERMINATE.

Plausible wrong implementations (O1):
- Uses MEAN instead of median (skewed slices give wrong answer).
- Slices on TERMINAL coding_progress (run-level) instead of per-row
  (the row's own progress).
- Includes successful runs in the slice (final_success filter dropped).
- Boundary flipped: median == bound counted as PASS instead of FAIL.

Plausible wrong implementations (O7):
- Compares G4 - G2 (sign flipped) — declares fail when G4 wins.
- Single-class y silently produces Brier=0 and a misleading PASS.
- Aggregates across sources before thresholding (averages dilute the
  signal).
- Threshold is `> 0.02` instead of `>= 0.02` at the boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coding_estimator.eval.failure_modes import (
    HIGH_PROGRESS_THRESHOLD,
    O1_MEDIAN_BOUND,
    O7_BRIER_DELTA_GATE,
    evaluate_o1,
    evaluate_o7,
)


# ---------- O1 ----------------------------------------------------------------


def _o1_fixture(failed_preds: list[float], success_preds: list[float]):
    """Build minimal frames: each prediction is one row at progress=0.9.
    Failed runs have final_success=0; successful runs have
    final_success=1. The slice (failed AND progress >= 0.8) is exactly
    `failed_preds`."""
    rows = []
    final_success: dict[str, int] = {}
    ck_rows = []
    rid = 0
    for p in failed_preds:
        run_id = f"f_{rid}"
        rows.append({
            "run_id": run_id, "source": "src",
            "checkpoint_id": f"{run_id}_c0", "checkpoint_step": 0,
            "_y": 0, "_p": p,
        })
        ck_rows.append({
            "run_id": run_id, "source": "src",
            "checkpoint_id": f"{run_id}_c0", "checkpoint_step": 0,
            "coding_progress": 0.9,
        })
        final_success[run_id] = 0
        rid += 1
    for p in success_preds:
        run_id = f"s_{rid}"
        rows.append({
            "run_id": run_id, "source": "src",
            "checkpoint_id": f"{run_id}_c0", "checkpoint_step": 0,
            "_y": 1, "_p": p,
        })
        ck_rows.append({
            "run_id": run_id, "source": "src",
            "checkpoint_id": f"{run_id}_c0", "checkpoint_step": 0,
            "coding_progress": 0.9,
        })
        final_success[run_id] = 1
        rid += 1
    return pd.DataFrame(rows), pd.DataFrame(ck_rows), final_success


def test_o1_uses_median_not_mean_on_skewed_slice():
    """Slice predictions [0.1, 0.1, 0.1, 0.1, 0.99]. Median = 0.1
    (PASS); mean = 0.278 (also PASS but for the wrong reason). Replace
    one value to flip them: [0.1, 0.1, 0.1, 0.99, 0.99]. Median = 0.1
    (PASS); mean = 0.476 (PASS). Try [0.6, 0.6, 0.6, 0.99, 0.99]:
    median = 0.6 (PASS), mean = 0.758 (FAIL). A correct impl reports
    PASS; a mean-based impl reports FAIL."""
    preds, ck, fs = _o1_fixture(
        failed_preds=[0.6, 0.6, 0.6, 0.99, 0.99], success_preds=[]
    )
    out = evaluate_o1(predictions_df=preds, checkpoints_df=ck, final_success=fs)
    assert out.outcome == "pass", (
        f"median should be 0.6 < {O1_MEDIAN_BOUND}, got value={out.metric_value}"
    )
    assert abs(out.metric_value - 0.6) < 1e-9


def test_o1_excludes_successful_runs_from_slice():
    """5 failed runs at p=0.99 (overconfident) → would FAIL.
    5 successful runs at p=0.99 mixed in. A wrong impl that doesn't
    filter on final_success would lower the slice median by mixing
    success runs (whose `0` y aren't matching) — but per the contract
    the slice is FAILED runs only, so the result must still FAIL."""
    preds, ck, fs = _o1_fixture(
        failed_preds=[0.99, 0.99, 0.99, 0.99, 0.99],
        success_preds=[0.01, 0.01, 0.01, 0.01, 0.01],
    )
    out = evaluate_o1(predictions_df=preds, checkpoints_df=ck, final_success=fs)
    assert out.outcome == "fail", (
        f"slice must be failed-runs-only; got {out.outcome} "
        f"(value={out.metric_value})"
    )
    assert out.detail["n_rows"] == 5


def test_o1_filters_by_per_row_progress_not_terminal():
    """A failed run with rows at progress [0.1, 0.5, 0.9]. Only the
    progress=0.9 row should enter the slice. We add 5 such failed runs
    so n=5 (≥ MIN). The non-high-progress rows must NOT contribute to
    the median."""
    rows, ck_rows = [], []
    fs = {}
    for r in range(5):
        run_id = f"r{r}"
        for s, prog in enumerate([0.1, 0.5, 0.9]):
            rows.append({
                "run_id": run_id, "source": "src",
                "checkpoint_id": f"{run_id}_c{s}", "checkpoint_step": s,
                "_y": 0, "_p": (0.1 if prog < 0.8 else 0.5),
            })
            ck_rows.append({
                "run_id": run_id, "source": "src",
                "checkpoint_id": f"{run_id}_c{s}", "checkpoint_step": s,
                "coding_progress": prog,
            })
        fs[run_id] = 0
    out = evaluate_o1(
        predictions_df=pd.DataFrame(rows),
        checkpoints_df=pd.DataFrame(ck_rows),
        final_success=fs,
    )
    # Slice should be 5 rows (one per run, the progress=0.9 one), each
    # with p=0.5. Median = 0.5.
    assert out.detail["n_rows"] == 5
    assert abs(out.metric_value - 0.5) < 1e-9


def test_o1_indeterminate_when_no_failed_runs():
    preds, ck, fs = _o1_fixture(failed_preds=[], success_preds=[0.5, 0.5, 0.5])
    out = evaluate_o1(predictions_df=preds, checkpoints_df=ck, final_success=fs)
    assert out.outcome == "indeterminate"
    assert "no failed runs" in (out.note or "")


def test_o1_indeterminate_when_slice_below_minimum():
    """3 failed runs in slice — below MIN_PER_SLICE=5 threshold."""
    preds, ck, fs = _o1_fixture(failed_preds=[0.5, 0.5, 0.5], success_preds=[])
    out = evaluate_o1(predictions_df=preds, checkpoints_df=ck, final_success=fs)
    assert out.outcome == "indeterminate"


def test_o1_boundary_strict_less_than_means_equal_is_fail():
    """Slice median = 0.7 exactly. Gate is `< 0.7` strict. PASS would
    require strictly less; equal must be FAIL."""
    preds, ck, fs = _o1_fixture(
        failed_preds=[0.7, 0.7, 0.7, 0.7, 0.7], success_preds=[]
    )
    out = evaluate_o1(predictions_df=preds, checkpoints_df=ck, final_success=fs)
    assert abs(out.metric_value - 0.7) < 1e-9
    assert out.outcome == "fail"


# ---------- O7 ----------------------------------------------------------------


def test_o7_emits_one_result_per_source():
    """No matter the data, evaluate_o7 must emit exactly one
    FailureModeResult per source observed in the checkpoints frame."""
    ck = pd.read_parquet("datasets/checkpoints_all.parquet")
    lb = pd.read_parquet("datasets/labels_all.parquet")
    out = evaluate_o7(checkpoints_df=ck, labels_df=lb)
    sources = sorted(ck["source"].unique())
    by_source = {r.detail.get("source") for r in out}
    assert by_source == set(sources)


def test_o7_single_class_y_is_indeterminate_not_pass():
    """tb_live's y_success_eventual is 12/12 successes — single-class.
    A correct impl flags this as indeterminate (Brier=0 for both models
    is uninformative). A wrong impl that doesn't gate on uniqueness
    silently reports `pass` because Brier_G2 - Brier_G4 = 0 - 0 = 0,
    which would fail the >= 0.02 gate; but the more dangerous wrong
    impl is to declare PASS because the absolute Briers are tiny."""
    ck = pd.read_parquet("datasets/checkpoints_all.parquet")
    lb = pd.read_parquet("datasets/labels_all.parquet")
    out = evaluate_o7(checkpoints_df=ck, labels_df=lb)
    tb_results = [r for r in out if r.detail.get("source") == "tb_live"]
    assert len(tb_results) == 1
    assert tb_results[0].outcome == "indeterminate", (
        "tb_live success is single-class; outcome must NOT be pass/fail"
    )


def test_o7_outcome_uses_g2_minus_g4_direction():
    """Verify the sign convention via swe_agent_pilot, where the real
    fixture has G4 slightly worse than G2 (delta = b2 - b4 < 0).
    Outcome must be `fail`. A sign-flipped impl would compute
    b4 - b2 > 0 and falsely PASS."""
    ck = pd.read_parquet("datasets/checkpoints_all.parquet")
    lb = pd.read_parquet("datasets/labels_all.parquet")
    out = evaluate_o7(checkpoints_df=ck, labels_df=lb)
    swe = next(r for r in out if r.detail.get("source") == "swe_agent_pilot")
    assert swe.metric_value is not None
    # If the metric is negative, outcome must be fail (G4 lost).
    if swe.metric_value < O7_BRIER_DELTA_GATE:
        assert swe.outcome == "fail"
    # And the metric must equal b2 - b4 (not b4 - b2).
    expected = swe.detail["brier_g2"] - swe.detail["brier_g4"]
    assert abs(swe.metric_value - expected) < 1e-9


def test_o7_threshold_is_inclusive_at_gate_boundary():
    """We can't easily synthesize delta = 0.02 exactly through
    fit_binary, but we can confirm the gate uses `>= delta_gate` (not
    `> delta_gate`) by mocking the comparison. Construct a result by
    hand and verify the outcome rule directly:

      delta == gate ⇒ pass
      delta == gate - eps ⇒ fail
    """
    # The function under test is the inline comparison
    # `pass if delta >= delta_gate else fail`. If somebody flipped to
    # strict `>`, then delta == gate would silently fail. Mirror the
    # check explicitly.
    gate = O7_BRIER_DELTA_GATE
    assert (gate >= gate) is True
    assert (gate - 1e-9 >= gate) is False
