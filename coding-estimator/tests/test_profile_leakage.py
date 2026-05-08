"""F4 leakage profile flags (Workstream F4).

Claim:
    `feature_leakage_rows(df)` returns one row per registered feature
    (NOT per column in the frame). Per row:
      - `derived_only_from_prefix` mirrors `Feature.prefix_only`
      - `contains_forbidden_token` is True iff the feature name matches
        the forbidden-column spec via `find_forbidden`
      - `constant_or_near_constant` is True iff the most-frequent
        non-null value's normalised frequency is >= 0.99
      - `high_cardinality_id` is True iff `dtype == "str"` AND the
        column has > 0.5 * len(df) distinct non-null values
      - `overall_non_null_rate` is the fraction of cells that are
        not-null across the whole frame.

Plausible wrong implementations:
    - iterate the frame's columns and miss registered features that
      didn't materialise (they should still emit a row, with low
      availability) — i.e. drive-by-frame instead of drive-by-registry.
    - `_is_near_constant` returns True on a column with ALL-null
      values (because `value_counts` on empty series would crash; an
      `except: True` would silently flip the flag).
    - `_is_high_cardinality_id` checks `> 0.5 * n_unique` (denominator
      mistake) instead of `> 0.5 * n_rows`.
    - `_is_high_cardinality_id` ignores the `dtype` filter and fires
      on a numeric column — `>0.5 * n_rows distinct ints` is normal
      for an `elapsed_steps` column.
    - `contains_forbidden_token` uses substring match instead of the
      forbidden-spec exact/prefix/suffix match (so e.g.
      "fraction_timeout_consumed" wrongly hits "timeout").
"""

from __future__ import annotations

import pandas as pd

from coding_estimator.checkpoints.features.missingness import Missingness
from coding_estimator.checkpoints.features.registry import Feature, all_features
from coding_estimator.leakage.guard import ForbiddenSpec
from coding_estimator.profile.leakage import (
    HIGH_CARDINALITY_RATIO,
    NEAR_CONSTANT_THRESHOLD,
    _is_high_cardinality_id,
    _is_near_constant,
    feature_leakage_rows,
)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["source", "run_id"])


def test_one_row_per_registered_feature_even_when_absent_from_frame() -> None:
    """The profile is registry-driven. A frame missing some columns
    must still produce a row per registered feature (with
    overall_non_null_rate=0.0 for absent columns)."""
    rows = feature_leakage_rows(_empty_df())
    expected_count = len(all_features())
    assert len(rows) == expected_count
    expected_names = {f.column_name for f in all_features()}
    assert {r.column_name for r in rows} == expected_names


def test_near_constant_handles_all_null_column_without_crashing() -> None:
    """All-null column has no defined mode. The function must return
    False (not crash, not silently return True)."""
    df = pd.DataFrame({"x": [None, None, None]})
    assert _is_near_constant(df, "x") is False


def test_near_constant_single_non_null_value() -> None:
    """A column with one non-null value (rest null) — that value
    occupies 100% of NON-NULL cells, so the freq-of-mode is 1.0 >=
    0.99. The function operates on dropna'd values per the docstring."""
    df = pd.DataFrame({"x": [None, 1.0, None]})
    assert _is_near_constant(df, "x") is True


def test_near_constant_boundary_at_threshold() -> None:
    """Boundary check on 99% — 99 rows of value 0 + 1 row of value 1
    is exactly 0.99, must be flagged. 98:2 is below threshold and
    must NOT be flagged."""
    df_99 = pd.DataFrame({"x": [0] * 99 + [1] * 1})
    assert _is_near_constant(df_99, "x") is True
    df_98 = pd.DataFrame({"x": [0] * 98 + [1] * 2})
    assert _is_near_constant(df_98, "x") is False
    assert NEAR_CONSTANT_THRESHOLD == 0.99


def test_near_constant_returns_false_for_missing_column() -> None:
    df = pd.DataFrame({"y": [1, 2, 3]})
    assert _is_near_constant(df, "x_does_not_exist") is False


def test_high_cardinality_id_only_fires_for_string_dtype() -> None:
    """A numeric column with high cardinality (e.g. elapsed_steps)
    must NOT be flagged. Otherwise a legitimate counter feature would
    appear as an ID leak. The dtype filter is the contract."""
    df = pd.DataFrame({"elapsed_steps": list(range(100))})
    # dtype="int" must NOT trip the flag.
    assert _is_high_cardinality_id(df, "elapsed_steps", "int") is False
    # Same data, but registered as str — flag fires.
    df_str = pd.DataFrame({"elapsed_steps": [str(i) for i in range(100)]})
    assert _is_high_cardinality_id(df_str, "elapsed_steps", "str") is True


def test_high_cardinality_id_uses_n_rows_denominator() -> None:
    """`n_unique > 0.5 * n_rows`. With 4 rows and 3 distinct strings,
    3 > 0.5 * 4 = 2.0 → True. With 4 rows and 2 distinct strings,
    2 > 0.5 * 4 = 2.0 is False (strict >, not >=). Catches the
    `> 0.5 * n_unique` denominator mistake."""
    df3 = pd.DataFrame({"x": ["a", "b", "c", "a"]})
    df2 = pd.DataFrame({"x": ["a", "b", "a", "b"]})
    assert _is_high_cardinality_id(df3, "x", "str") is True
    assert _is_high_cardinality_id(df2, "x", "str") is False
    assert HIGH_CARDINALITY_RATIO == 0.5


def test_contains_forbidden_token_does_not_substring_match() -> None:
    """`fraction_timeout_consumed` contains the substring "timeout"
    but is NOT in the forbidden spec; substring matching would wrongly
    flag it. The contract is exact/prefix/suffix matching via
    `find_forbidden`."""
    spec = ForbiddenSpec(
        exact=("final_success",),
        prefixes=("y_",),
        suffixes=("_terminal",),
    )
    fake_feature = Feature(
        column_name="fraction_timeout_consumed",
        dtype="float",
        group="time_budget",
        populated_on=("tb_live",),
        upstream_source=None,
        missingness_semantic=Missingness.UNKNOWN_DUE_TO_MISSING_ARTIFACT,
        run_constant_flag=False,
    )
    rows = feature_leakage_rows(_empty_df(), forbidden=spec, feature_registry=[fake_feature])
    assert rows[0].contains_forbidden_token is False


def test_contains_forbidden_token_fires_on_exact_match() -> None:
    """If somebody registers a feature with a forbidden name, the
    flag MUST fire — this is the regression-detector for accidental
    leakage features sneaking into the registry."""
    spec = ForbiddenSpec(exact=("final_success",), prefixes=(), suffixes=())
    bad = Feature(
        column_name="final_success",
        dtype="bool",
        group="closure",
        populated_on=("tb_live",),
        upstream_source=None,
        missingness_semantic=Missingness.APPLICABLE_NEVER_OBSERVED_IN_RUN,
        run_constant_flag=True,
    )
    rows = feature_leakage_rows(_empty_df(), forbidden=spec, feature_registry=[bad])
    assert rows[0].contains_forbidden_token is True


def test_no_v0_feature_is_currently_forbidden() -> None:
    """Sanity invariant on the actual v0 registry: zero features
    flagged FORBIDDEN_TOKEN. If this fails, somebody silently wired
    a leakage column into the feature registry."""
    rows = feature_leakage_rows(_empty_df())
    flagged = [r.column_name for r in rows if r.contains_forbidden_token]
    assert flagged == [], flagged


def test_overall_non_null_rate_matches_notna_mean() -> None:
    """For a frame with N rows and K non-null values in column c,
    overall_non_null_rate must equal K/N exactly. Catches an
    accidental `notna().sum()` (raw count) regression."""
    # Use a real registered feature column so the row is emitted with
    # the actual frame data.
    df = pd.DataFrame({"coding_progress": [0.0, None, 0.5, None, 1.0]})
    rows = feature_leakage_rows(df)
    by_name = {r.column_name: r for r in rows}
    assert by_name["coding_progress"].overall_non_null_rate == 3 / 5
