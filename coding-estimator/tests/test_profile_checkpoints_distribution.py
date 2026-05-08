"""F2 checkpoint-distribution profile (Workstream F2).

Claim:
    `distributions(checkpoints_df)` produces one
    `CheckpointDistribution` per source. Bucket counts use half-open
    `[lo, hi)` intervals; the last bucket extends to `1.000001` so
    that the value `1.0` exactly lands in `[0.75, 1.0]`. Bucket counts
    sum to the count of NON-NULL values in the source's column
    (no double counting; values strictly above 1.0 are dropped).

    `validation_started_rate`, `validation_complete_rate`, and
    `blocked_rate` are means over the boolean-coerced non-null cells.
    `blocked_rate` is the fraction of checkpoints with
    `blocked_leaf_count > 0`.

Plausible wrong implementations:
    - closed buckets `[lo, hi]` so the value 0.25 falls in BOTH
      `[0.0, 0.25]` and `[0.25, 0.5]` (double-count).
    - upper bound exactly 1.0 instead of 1.000001 → the value 1.0
      lands in NO bucket (silent drop of the most-common terminal
      progress value).
    - rate computed against `len(grp)` instead of non-null count
      (under-rates when many cells are null).
    - `blocked_rate` uses `>= 0` instead of `> 0` (every checkpoint
      with even zero blocked leaves counts as blocked).
"""

from __future__ import annotations

import pandas as pd

from coding_estimator.profile.checkpoints_distribution import (
    PROGRESS_BUCKETS,
    _bucket_counts,
    distributions,
)


def _bucket_for(value: float) -> str:
    """Hand-evaluation of which bucket a value should land in per
    the documented half-open contract. Used as the test oracle
    instead of calling _bucket_counts (which is the unit under test)."""
    if 0.0 <= value < 0.25:
        return "[0.0, 0.25)"
    if 0.25 <= value < 0.5:
        return "[0.25, 0.5)"
    if 0.5 <= value < 0.75:
        return "[0.5, 0.75)"
    if 0.75 <= value <= 1.0:
        return "[0.75, 1.0]"
    raise AssertionError(f"value {value} is outside [0,1]")


def test_value_one_lands_in_last_bucket_not_dropped() -> None:
    """REGRESSION: an upper bound of exactly 1.0 (instead of
    1.000001) would silently drop the value 1.0 — i.e. every
    terminal-step checkpoint at full progress disappears from the
    distribution. The 1.000001 sentinel is the contract."""
    s = pd.Series([1.0, 1.0, 1.0])
    counts = _bucket_counts(s, PROGRESS_BUCKETS)
    assert counts["[0.75, 1.0]"] == 3
    assert sum(counts.values()) == 3  # nothing dropped


def test_boundary_at_quarter_lands_in_upper_bucket_not_lower() -> None:
    """Half-open [lo, hi): 0.25 lands in [0.25, 0.5), NOT in
    [0.0, 0.25). A closed-on-the-left convention would double-count
    or wrongly bin."""
    s = pd.Series([0.25, 0.5, 0.75])
    counts = _bucket_counts(s, PROGRESS_BUCKETS)
    assert counts["[0.0, 0.25)"] == 0
    assert counts["[0.25, 0.5)"] == 1
    assert counts["[0.5, 0.75)"] == 1
    assert counts["[0.75, 1.0]"] == 1


def test_value_above_one_is_dropped_silently() -> None:
    """Values >1.0 (rare rounding artefact) fall into no bucket.
    The bucket-sum will be less than n_values. This documents the
    behaviour so a future regression that wraps such values into
    [0.75, 1.0] is detected."""
    s = pd.Series([0.5, 1.000002])
    counts = _bucket_counts(s, PROGRESS_BUCKETS)
    assert sum(counts.values()) == 1


def test_bucket_counts_sum_equals_non_null_count() -> None:
    """For valid [0,1] inputs, bucket counts sum exactly to the count
    of non-null cells (no over-count, no under-count)."""
    s = pd.Series([0.0, 0.1, 0.25, 0.49, 0.5, 0.74, 0.75, 1.0, None, None])
    counts = _bucket_counts(s, PROGRESS_BUCKETS)
    assert sum(counts.values()) == 8  # 8 non-null in [0,1]


def test_bucket_counts_metamorphic_on_uniform_input() -> None:
    """Hand-checkable: 100 evenly-spaced values in [0,1] (steps of
    0.01) should put 25 values in each of the first three buckets and
    26 values in the last (because 1.0 is in the last bucket too)."""
    s = pd.Series([i / 100 for i in range(101)])  # 0.00, 0.01, ..., 1.00
    counts = _bucket_counts(s, PROGRESS_BUCKETS)
    assert counts["[0.0, 0.25)"] == 25  # 0..24
    assert counts["[0.25, 0.5)"] == 25  # 25..49
    assert counts["[0.5, 0.75)"] == 25  # 50..74
    assert counts["[0.75, 1.0]"] == 26  # 75..100 (inclusive of 1.0)


def test_distributions_one_row_per_source() -> None:
    """Per-source independence. A frame with two sources must produce
    exactly two rows (sorted by source id)."""
    df = pd.DataFrame({
        "source": ["a", "a", "b"],
        "run_id": ["r1", "r1", "r2"],
        "checkpoint_step": [0, 1, 0],
        "coding_progress": [0.0, 0.5, 1.0],
    })
    dists = distributions(df)
    assert [d.source for d in dists] == ["a", "b"]
    assert dists[0].n_checkpoints == 2
    assert dists[1].n_checkpoints == 1


def test_validation_started_rate_is_mean_not_count() -> None:
    """`validation_started_rate` for 3 trues out of 5 unmasked rows
    is 0.6, NOT 3."""
    df = pd.DataFrame({
        "source": ["a"] * 5,
        "run_id": ["r1"] * 5,
        "checkpoint_step": list(range(5)),
        "validation_started": [True, True, True, False, False],
    })
    [d] = distributions(df)
    assert d.validation_started_rate == 0.6


def test_blocked_rate_uses_strict_greater_than_zero() -> None:
    """`blocked_rate` = mean of `blocked_leaf_count > 0`. A row with
    blocked_leaf_count=0 must NOT count as blocked. A `>= 0` regression
    would inflate the rate to 1.0."""
    df = pd.DataFrame({
        "source": ["a"] * 4,
        "run_id": ["r1"] * 4,
        "checkpoint_step": list(range(4)),
        "blocked_leaf_count": [0, 0, 1, 2],
    })
    [d] = distributions(df)
    assert d.blocked_rate == 0.5  # 2 of 4 rows have count > 0


def test_leaf_count_quantiles_hand_checkable() -> None:
    """Quantiles use linear interpolation by default in pandas. For
    [1, 2, 3, 4]: p25=1.75, p50=2.5, p75=3.25. Pinning these values
    catches a regression that switches to nearest-neighbor or
    midpoint."""
    df = pd.DataFrame({
        "source": ["a"] * 4,
        "run_id": ["r1"] * 4,
        "checkpoint_step": list(range(4)),
        "active_leaf_count": [1, 2, 3, 4],
    })
    [d] = distributions(df)
    assert d.leaf_count_p25 == 1.75
    assert d.leaf_count_p50 == 2.5
    assert d.leaf_count_p75 == 3.25


def test_rate_returns_none_when_column_absent_or_all_null() -> None:
    """All-null or absent column gives `None` (not 0.0). The
    distinction matters: 0.0 means "we observed and saw zero", None
    means "we don't have the signal". A regression that returned 0.0
    would silently mask data-availability issues."""
    df_absent = pd.DataFrame({
        "source": ["a", "a"],
        "run_id": ["r1", "r1"],
        "checkpoint_step": [0, 1],
    })
    [d] = distributions(df_absent)
    assert d.validation_started_rate is None

    df_allnull = pd.DataFrame({
        "source": ["a", "a"],
        "run_id": ["r1", "r1"],
        "checkpoint_step": [0, 1],
        "validation_started": [None, None],
    })
    [d2] = distributions(df_allnull)
    assert d2.validation_started_rate is None
