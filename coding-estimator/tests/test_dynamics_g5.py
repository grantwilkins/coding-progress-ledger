"""
Claim:
- attach_g5_features adds G5 columns derived purely from prefix-ordered
  rows within a run. At step t, every G5 feature depends only on rows
  with checkpoint_step <= t.
- Windowed deltas use a partial window: at step t with window k, the
  delta is value[t] - value[max(0, idx_t - k)]. Initial steps (where
  idx_t < k) use the anchor at row 0, NOT a NaN / forward-fill / skip.
- blocked_persistence is the streak length where blocked_leaf_count > 0
  ending at the current row; it resets to 0 the moment blocked goes to 0.
- reopen_after_validation is cumulative (sticky-True) and only fires
  for reopens AT OR AFTER the first validation attempt.
- no_progress_run_length is the streak of no_progress_window_5 == True
  ending at the current row; resets on False.

Plausible wrong implementations:
- Windowed delta skips initial steps (returns NaN until idx >= k),
  which would make slope_5 at idx=2 equal to NaN instead of 0.2.
- Streak helper accumulates across runs (forgets to groupby run_id).
- reopen_after_validation counts reopens that happened BEFORE the
  first validation attempt — that flag would mean something different
  ("ever reopened given any validation later"), not "validation
  exposed new work."
- A G5 value at step t silently consumes row t+1 (a one-sided future
  leak that would not be caught by visual inspection of the output).
- Streak resets on a row of NaN instead of False (NaN is falsy in
  pandas after fillna(False) but a wrong impl that uses isna() check
  might double-reset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coding_estimator.checkpoints.dynamics import G5_FEATURES, attach_g5_features


def _row(run_id: str, step: int, **kw) -> dict:
    base = {
        "run_id": run_id,
        "source": "synth",
        "checkpoint_id": f"{run_id}_c{step}",
        "checkpoint_step": step,
        "coding_progress": 0.0,
        "num_validation_attempts": 0,
        "num_validation_successes": 0,
        "num_validation_failures": 0,
        "num_reopens_so_far": 0,
        "blocked_leaf_count": 0,
        "strong_completion_count": 0,
        "denominator_growth_so_far": 0,
        "no_progress_window_5": False,
    }
    base.update(kw)
    return base


# ---------- prefix-only invariance ------------------------------------------


def test_g5_at_step_t_does_not_depend_on_rows_after_t():
    """Build a 5-row run; attach G5; record value at step 2. Truncate
    the input to steps 0..2; re-attach G5; value at step 2 must be
    identical. This is the core prefix-only contract."""
    full = pd.DataFrame(
        [
            _row("r", 0, coding_progress=0.0, blocked_leaf_count=1),
            _row("r", 1, coding_progress=0.1, blocked_leaf_count=1),
            _row("r", 2, coding_progress=0.3, blocked_leaf_count=0),
            _row("r", 3, coding_progress=0.7, blocked_leaf_count=0),
            _row("r", 4, coding_progress=1.0, blocked_leaf_count=0),
        ]
    )
    truncated = full[full["checkpoint_step"] <= 2].copy()
    full_g5 = attach_g5_features(full)
    trunc_g5 = attach_g5_features(truncated)
    full_at_2 = full_g5[full_g5["checkpoint_step"] == 2].iloc[0]
    trunc_at_2 = trunc_g5[trunc_g5["checkpoint_step"] == 2].iloc[0]
    for col in G5_FEATURES:
        assert full_at_2[col] == trunc_at_2[col], (
            f"G5 feature {col} at step 2 changed when steps 3..4 were "
            f"removed (full={full_at_2[col]}, trunc={trunc_at_2[col]}) "
            "— prefix-only contract violated"
        )


def test_g5_runs_do_not_leak_into_each_other():
    """Two runs in one frame with run-disjoint progress profiles. The
    G5 values for run B's first row must not depend on run A. A wrong
    impl missing groupby would let A's progress slope bleed into B."""
    frame = pd.DataFrame(
        [
            _row("a", 0, coding_progress=0.0),
            _row("a", 1, coding_progress=0.5),
            _row("a", 2, coding_progress=1.0),
            _row("b", 0, coding_progress=0.0),
        ]
    )
    out = attach_g5_features(frame)
    b0 = out[(out["run_id"] == "b") & (out["checkpoint_step"] == 0)].iloc[0]
    # First row of any run: slope/accel/density must be 0 (anchor == self).
    assert b0["g5_coding_progress_slope_3"] == 0.0
    assert b0["g5_coding_progress_slope_5"] == 0.0
    assert b0["g5_coding_progress_accel_5"] == 0.0
    assert b0["g5_validation_density_5"] == 0.0
    assert b0["g5_blocked_persistence"] == 0
    assert b0["g5_no_progress_run_length"] == 0


# ---------- windowed-delta partial window contract --------------------------


def test_slope_5_at_row_2_uses_partial_window_anchor_at_row_0():
    """3-row run with coding_progress=[0.0, 0.5, 1.0]. At row idx=2
    (third row), window k=5 cannot reach back 5 steps; the contract
    says anchor at max(0, idx-k) == 0. Slope_5 = (1.0 - 0.0)/5 = 0.2.

    A wrong impl that returns NaN until enough history exists would
    fail this. A wrong impl that uses idx as the divisor would
    return (1.0 - 0.0)/2 = 0.5."""
    frame = pd.DataFrame(
        [
            _row("r", 0, coding_progress=0.0),
            _row("r", 1, coding_progress=0.5),
            _row("r", 2, coding_progress=1.0),
        ]
    )
    out = attach_g5_features(frame)
    slope_at_2 = out[out["checkpoint_step"] == 2]["g5_coding_progress_slope_5"].iloc[0]
    assert slope_at_2 == 0.2


def test_slope_3_anchors_to_idx_minus_k_when_window_is_full():
    """5-row run with coding_progress=[0.0, 0.1, 0.2, 0.3, 0.4]. At
    row idx=4, anchor = max(0, 4-3) = 1; slope_3 = (0.4 - 0.1)/3 = 0.1."""
    frame = pd.DataFrame(
        [_row("r", i, coding_progress=0.1 * i) for i in range(5)]
    )
    out = attach_g5_features(frame)
    slope_at_4 = out[out["checkpoint_step"] == 4]["g5_coding_progress_slope_3"].iloc[0]
    assert abs(slope_at_4 - 0.1) < 1e-9


def test_accel_is_slope_3_minus_slope_5():
    """Direct contract: accel_5 = slope_3 - slope_5 at every row."""
    frame = pd.DataFrame(
        [_row("r", i, coding_progress=0.1 * i) for i in range(6)]
    )
    out = attach_g5_features(frame)
    diff = (
        out["g5_coding_progress_slope_3"]
        - out["g5_coding_progress_slope_5"]
        - out["g5_coding_progress_accel_5"]
    )
    assert np.allclose(diff.to_numpy(), 0.0, atol=1e-9)


# ---------- streak helpers --------------------------------------------------


def test_blocked_persistence_resets_when_blocked_leaf_count_drops_to_zero():
    """blocked_leaf_count=[1,1,0,1,1] → blocked_persistence=[1,2,0,1,2].
    A wrong impl that accumulates without reset would produce [1,2,2,3,4]."""
    frame = pd.DataFrame(
        [
            _row("r", 0, blocked_leaf_count=1),
            _row("r", 1, blocked_leaf_count=1),
            _row("r", 2, blocked_leaf_count=0),
            _row("r", 3, blocked_leaf_count=1),
            _row("r", 4, blocked_leaf_count=1),
        ]
    )
    out = attach_g5_features(frame).sort_values("checkpoint_step")
    got = out["g5_blocked_persistence"].tolist()
    assert got == [1, 2, 0, 1, 2]


def test_no_progress_run_length_resets_on_false():
    """no_progress_window_5=[F,T,T,F,T] → run_length=[0,1,2,0,1]."""
    frame = pd.DataFrame(
        [
            _row("r", 0, no_progress_window_5=False),
            _row("r", 1, no_progress_window_5=True),
            _row("r", 2, no_progress_window_5=True),
            _row("r", 3, no_progress_window_5=False),
            _row("r", 4, no_progress_window_5=True),
        ]
    )
    out = attach_g5_features(frame).sort_values("checkpoint_step")
    got = out["g5_no_progress_run_length"].tolist()
    assert got == [0, 1, 2, 0, 1]


# ---------- reopen_after_validation polarity --------------------------------


def test_reopen_after_validation_ignores_reopens_before_first_validation():
    """num_reopens_so_far=[0,1,2,2,3], num_validation_attempts=[0,0,1,1,1].
    First validation attempt is at idx 2 (step 2). num_reopens_so_far
    at step 2 = 2 (the baseline at first-validation). After that,
    reopens > 2 first occurs at idx 4 (3 > 2). Expected:
    [0, 0, 0, 0, 1].

    A wrong impl that triggers on any reopen anywhere would yield
    [0, 1, 1, 1, 1]. A wrong impl that triggers on reopens at the
    first-validation step itself (rather than strictly *after*) would
    yield [0, 0, 1, 1, 1] (since reopens=2 > baseline of 0 if it
    reads the wrong baseline)."""
    frame = pd.DataFrame(
        [
            _row("r", 0, num_validation_attempts=0, num_reopens_so_far=0),
            _row("r", 1, num_validation_attempts=0, num_reopens_so_far=1),
            _row("r", 2, num_validation_attempts=1, num_reopens_so_far=2),
            _row("r", 3, num_validation_attempts=1, num_reopens_so_far=2),
            _row("r", 4, num_validation_attempts=1, num_reopens_so_far=3),
        ]
    )
    out = attach_g5_features(frame).sort_values("checkpoint_step")
    got = out["g5_reopen_after_validation"].astype(int).tolist()
    assert got == [0, 0, 0, 0, 1]


def test_reopen_after_validation_is_sticky_true():
    """Once True, must stay True even if no further reopens."""
    frame = pd.DataFrame(
        [
            _row("r", 0, num_validation_attempts=1, num_reopens_so_far=0),
            _row("r", 1, num_validation_attempts=1, num_reopens_so_far=1),
            _row("r", 2, num_validation_attempts=1, num_reopens_so_far=1),
            _row("r", 3, num_validation_attempts=1, num_reopens_so_far=1),
        ]
    )
    out = attach_g5_features(frame).sort_values("checkpoint_step")
    got = out["g5_reopen_after_validation"].astype(int).tolist()
    # First validation is at idx 0 (baseline reopens = 0). Reopen at
    # idx 1 (1 > 0) flips True. Stays True forever after.
    assert got == [0, 1, 1, 1]


# ---------- validation success recency --------------------------------------


def test_validation_success_recency_is_zero_before_first_success():
    """Before any validation success, recency is 0 (not 1, not NaN).
    A wrong impl that defaults to 1.0 for "no time has passed" would
    misrepresent "never happened" as "just happened."""
    frame = pd.DataFrame(
        [
            _row("r", 0, num_validation_successes=0),
            _row("r", 1, num_validation_successes=0),
            _row("r", 2, num_validation_successes=1),  # first success
            _row("r", 3, num_validation_successes=1),
        ]
    )
    out = attach_g5_features(frame).sort_values("checkpoint_step")
    got = out["g5_validation_success_recency"].tolist()
    # Steps 0, 1: no success yet → 0.
    # Step 2: success just happened (last_success_step == 2, current step == 2)
    #          → 1 / (1 + 0) = 1.0.
    # Step 3: 1 step since last success → 1 / (1 + 1) = 0.5.
    assert got[0] == 0.0
    assert got[1] == 0.0
    assert got[2] == 1.0
    assert got[3] == 0.5
