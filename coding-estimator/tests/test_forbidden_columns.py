"""Forbidden-column guard exercises exact / prefix / suffix matching.

Claim:
    Suffix matching uses str.endswith(suffix), not substring containment.
    A column whose name *contains* a forbidden suffix elsewhere in the
    string must NOT be rejected; only true suffixes must.

Plausible wrong implementations:
    - `if suffix in col` instead of `col.endswith(suffix)` -> over-matches
    - applying prefixes as suffixes (or vice versa) -> wrong directional check
    - case-insensitive match -> "Final" wrongly accepted/rejected
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

from coding_estimator.leakage.guard import (
    assert_no_forbidden,
    find_forbidden,
    load_forbidden_spec,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "forbidden_columns.json"


def test_schema_validates() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_clean_frame_passes() -> None:
    df = pd.DataFrame({"run_id": ["a"], "checkpoint_step": [1], "active_leaf_count": [3]})
    assert_no_forbidden(df)


@pytest.mark.parametrize(
    "bad_col",
    [
        "y_success_eventual",
        "label_progress",
        "final_success",
        "verifier_exit_code",
        "test_output_path",
        "eval_log_size",
        "final_diff_lines",
        "summary_by_category_total",
        "shape_label",
        "shape_tags",
        "checkpoint_event_index_at_terminal",
        "final_artifact_without_validation",
        "anything_at_terminal",  # suffix _at_terminal
        "score_final",            # suffix _final
    ],
)
def test_forbidden_column_rejected(bad_col: str) -> None:
    df = pd.DataFrame({"run_id": ["a"], bad_col: [1]})
    with pytest.raises(ValueError, match="forbidden columns"):
        assert_no_forbidden(df)


def test_find_returns_sorted_unique() -> None:
    cols = [
        "y_success_eventual",
        "y_future_progress_drop_h5",
        "active_leaf_count",
        "final_success",
    ]
    out = find_forbidden(cols)
    assert out == sorted(set(out))
    assert "active_leaf_count" not in out
    assert "final_success" in out


def test_load_spec() -> None:
    spec = load_forbidden_spec()
    assert "final_success" in spec.exact
    assert "y_" in spec.prefixes
    assert "_final" in spec.suffixes


def test_suffix_match_is_endswith_not_substring() -> None:
    # `_final` is a forbidden suffix. `score_finally` only CONTAINS that
    # substring; it does not end with it -> must be allowed.
    df_ok = pd.DataFrame({"score_finally": [1.0]})
    assert_no_forbidden(df_ok)
    # The true suffix case must still be rejected.
    df_bad = pd.DataFrame({"score_final": [1.0]})
    with pytest.raises(ValueError, match="forbidden columns"):
        assert_no_forbidden(df_bad)


def test_at_terminal_endswith_only() -> None:
    # `_at_terminal` is a forbidden suffix. The legitimate `checkpoint_event_index`
    # feature must pass; only the terminal-leakage variant must be rejected.
    df_ok = pd.DataFrame({"checkpoint_event_index": [12]})
    assert_no_forbidden(df_ok)
    df_bad = pd.DataFrame({"checkpoint_event_index_at_terminal": [12]})
    with pytest.raises(ValueError, match="forbidden columns"):
        assert_no_forbidden(df_bad)


def test_submit_without_validation_so_far_is_NOT_forbidden() -> None:
    """The prefix-only flag must coexist with the terminal forbidden column."""
    df = pd.DataFrame({"submit_without_validation_so_far": [True]})
    assert_no_forbidden(df)
