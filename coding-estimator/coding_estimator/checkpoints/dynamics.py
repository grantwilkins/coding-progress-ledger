"""G5 — ledger-dynamics features.

Derived from the existing checkpoint frame (per-run, prefix-ordered) as
a post-processing layer. Avoids touching the upstream replay builder.

Each G5 feature is a pure function of the existing columns + prefix
order within a run. Prefix-only — the row at step `t` only consumes
rows at steps `<= t`. The dataclass `G5_FEATURES` enumerates the
columns added; `attach_g5_features` adds them to a checkpoints
frame in place of returning a new frame.

Features added:
- `g5_coding_progress_slope_3`     — Δ coding_progress over last 3 steps
- `g5_coding_progress_slope_5`     — Δ coding_progress over last 5 steps
- `g5_coding_progress_accel_5`     — slope_3 − slope_5 (positive ⇒ accel)
- `g5_validation_density_5`        — Δ num_validation_attempts over last 5 steps
- `g5_validation_success_recency`  — 1 / (1 + steps_since_last_validation_success)
- `g5_blocked_persistence`         — streak length where blocked_leaf_count > 0
- `g5_reopen_after_validation`     — bool: any reopen following a validation event
- `g5_evidence_rate_5`             — Δ strong_completion_count over last 5 steps
- `g5_denominator_growth_rate_5`   — Δ denominator_growth_so_far over last 5 steps
- `g5_no_progress_run_length`      — streak length of no_progress_window_5 == True

All windowed deltas are computed prefix-only: at step `t`, only
checkpoints with `checkpoint_step <= t` are visible. Initial steps
where the window is shorter than `k` use the partial window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

G5_FEATURES: tuple[str, ...] = (
    "g5_coding_progress_slope_3",
    "g5_coding_progress_slope_5",
    "g5_coding_progress_accel_5",
    "g5_validation_density_5",
    "g5_validation_success_recency",
    "g5_blocked_persistence",
    "g5_reopen_after_validation",
    "g5_evidence_rate_5",
    "g5_denominator_growth_rate_5",
    "g5_no_progress_run_length",
)


def _windowed_delta(series: pd.Series, k: int) -> pd.Series:
    """At position i, return series[i] - series[max(0, i-k)]. Aligns
    to the input's index. Pure prefix-only — only positions ≤ i are
    consumed. Initial positions where the window is shorter than k
    use the partial window (anchor = position 0)."""
    n = len(series)
    if n == 0:
        return series.copy()
    arr = series.to_numpy(dtype=float, copy=False)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        anchor = max(0, i - k)
        out[i] = arr[i] - arr[anchor]
    return pd.Series(out, index=series.index)


def _streak_true(mask: pd.Series) -> pd.Series:
    """Run-length-encoded streak: at position i, return the length of
    the current True run ending at i. False positions reset to 0."""
    n = len(mask)
    arr = mask.fillna(False).astype(bool).to_numpy()
    out = np.zeros(n, dtype=int)
    streak = 0
    for i in range(n):
        streak = streak + 1 if arr[i] else 0
        out[i] = streak
    return pd.Series(out, index=mask.index)


def _validation_success_recency(g: pd.DataFrame) -> pd.Series:
    """1 / (1 + steps since the most recent validation success). 0
    when there has been no success yet."""
    n_succ = g.get("num_validation_successes", pd.Series(0, index=g.index)).fillna(0).astype(int)
    steps = g.get("checkpoint_step", pd.Series(range(len(g)), index=g.index)).astype(int)
    n = len(g)
    out = np.zeros(n, dtype=float)
    last_success_step = None
    succ_arr = n_succ.to_numpy()
    step_arr = steps.to_numpy()
    prev_succ = 0
    for i in range(n):
        if succ_arr[i] > prev_succ:
            last_success_step = int(step_arr[i])
        prev_succ = int(succ_arr[i])
        if last_success_step is None:
            out[i] = 0.0
        else:
            out[i] = 1.0 / (1.0 + (int(step_arr[i]) - last_success_step))
    return pd.Series(out, index=g.index)


def _reopen_after_validation(g: pd.DataFrame) -> pd.Series:
    """1 if a reopen ever occurred at or after the first validation
    attempt, else 0. Cumulative; once True, stays True."""
    reopens = g.get("num_reopens_so_far", pd.Series(0, index=g.index)).fillna(0).astype(int)
    attempts = g.get("num_validation_attempts", pd.Series(0, index=g.index)).fillna(0).astype(int)
    n = len(g)
    out = np.zeros(n, dtype=int)
    reopens_at_first_validation: int | None = None
    flag = 0
    a_arr = attempts.to_numpy()
    r_arr = reopens.to_numpy()
    for i in range(n):
        if reopens_at_first_validation is None and a_arr[i] > 0:
            reopens_at_first_validation = int(r_arr[i])
        if reopens_at_first_validation is not None and r_arr[i] > reopens_at_first_validation:
            flag = 1
        out[i] = flag
    return pd.Series(out, index=g.index)


def _attach_g5_to_run(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("checkpoint_step", kind="mergesort").reset_index(drop=True)
    progress = g.get("coding_progress", pd.Series(0.0, index=g.index)).fillna(0.0).astype(float)
    slope_3 = _windowed_delta(progress, 3) / 3.0
    slope_5 = _windowed_delta(progress, 5) / 5.0
    accel = slope_3 - slope_5
    val_attempts = (
        g.get("num_validation_attempts", pd.Series(0, index=g.index))
        .fillna(0).astype(float)
    )
    val_density_5 = _windowed_delta(val_attempts, 5) / 5.0
    val_success_recency = _validation_success_recency(g)
    blocked_count = (
        g.get("blocked_leaf_count", pd.Series(0, index=g.index))
        .fillna(0).astype(int) > 0
    )
    blocked_persistence = _streak_true(blocked_count)
    reopen_after_val = _reopen_after_validation(g)
    strong_completion = (
        g.get("strong_completion_count", pd.Series(0, index=g.index))
        .fillna(0).astype(float)
    )
    evidence_rate_5 = _windowed_delta(strong_completion, 5) / 5.0
    denom_growth = (
        g.get("denominator_growth_so_far", pd.Series(0, index=g.index))
        .fillna(0).astype(float)
    )
    denom_rate_5 = _windowed_delta(denom_growth, 5) / 5.0
    no_progress_5 = (
        g.get("no_progress_window_5", pd.Series(False, index=g.index))
        .fillna(False).astype(bool)
    )
    no_progress_run = _streak_true(no_progress_5)

    g["g5_coding_progress_slope_3"] = slope_3.astype(float)
    g["g5_coding_progress_slope_5"] = slope_5.astype(float)
    g["g5_coding_progress_accel_5"] = accel.astype(float)
    g["g5_validation_density_5"] = val_density_5.astype(float)
    g["g5_validation_success_recency"] = val_success_recency.astype(float)
    g["g5_blocked_persistence"] = blocked_persistence.astype(int)
    g["g5_reopen_after_validation"] = reopen_after_val.astype(int)
    g["g5_evidence_rate_5"] = evidence_rate_5.astype(float)
    g["g5_denominator_growth_rate_5"] = denom_rate_5.astype(float)
    g["g5_no_progress_run_length"] = no_progress_run.astype(int)
    return g


def attach_g5_features(checkpoints_df: pd.DataFrame) -> pd.DataFrame:
    """Add the G5 columns to a checkpoints frame. Returns a new frame
    with the original index preserved (rows reordered within each run
    by checkpoint_step, then re-concatenated)."""
    if checkpoints_df.empty:
        out = checkpoints_df.copy()
        for c in G5_FEATURES:
            out[c] = pd.Series(dtype=float if c.startswith("g5_") else int)
        return out
    pieces: list[pd.DataFrame] = []
    for run_id, g in checkpoints_df.groupby("run_id", sort=True):
        pieces.append(_attach_g5_to_run(g))
    return pd.concat(pieces, ignore_index=True)


@dataclass(frozen=True)
class G5Spec:
    """Mirrors `BaselineSpec` so the harness can fit on G5 features."""

    name: str = "g5_dynamics"

    def feature_cols_for(self, _sources: tuple[str, ...]) -> tuple[str, ...]:
        return G5_FEATURES


__all__ = [
    "G5_FEATURES",
    "G5Spec",
    "attach_g5_features",
]
