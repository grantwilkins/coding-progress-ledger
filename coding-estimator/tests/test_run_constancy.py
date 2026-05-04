"""Run-constancy audit triggers on the (source, y_submit_without_validation) pair."""

from __future__ import annotations

import pandas as pd
import pytest

from coding_estimator.leakage.run_constancy import (
    assert_clean,
    audit,
    is_run_constant,
    load_spec,
)


def _frame() -> pd.DataFrame:
    """4 runs × 3 ckpts. `source` is run-constant by construction, and
    `y_submit_without_validation` is replicated within each run (= run-constant)."""
    rows = []
    for r, src, swv in [
        ("r0", "tb_live", 0),
        ("r1", "tb_live", 1),
        ("r2", "swe_agent_pilot", 1),
        ("r3", "hermes_pilot", 0),
    ]:
        for c in range(3):
            rows.append(
                {
                    "run_id": r,
                    "source": src,
                    "active_leaf_count": c,  # NOT run-constant
                    "y_submit_without_validation": swv,
                    "y_future_progress_drop_h5": (c % 2),
                }
            )
    return pd.DataFrame(rows)


def test_spec_loads() -> None:
    spec = load_spec()
    assert "source" in spec.declared
    assert spec.fail_on_run_constant_pair is True


def test_is_run_constant() -> None:
    df = _frame()
    assert is_run_constant(df, "source")
    assert is_run_constant(df, "y_submit_without_validation")
    assert not is_run_constant(df, "active_leaf_count")
    assert not is_run_constant(df, "y_future_progress_drop_h5")


def test_audit_flags_source_pair() -> None:
    df = _frame()
    offenders = audit(
        df,
        feature_columns=["source", "active_leaf_count"],
        target_columns=["y_submit_without_validation", "y_future_progress_drop_h5"],
    )
    assert ("source", "y_submit_without_validation") in offenders
    # active_leaf_count is run-VARYING, so it must not appear.
    assert all(f != "active_leaf_count" for f, _ in offenders)
    # The run-varying target must not appear either.
    assert all(t != "y_future_progress_drop_h5" for _, t in offenders)


def test_assert_clean_raises() -> None:
    df = _frame()
    with pytest.raises(ValueError, match="run-constant"):
        assert_clean(
            df,
            feature_columns=["source", "active_leaf_count"],
            target_columns=["y_submit_without_validation"],
        )


def test_assert_clean_passes_when_no_pair() -> None:
    df = _frame()
    assert_clean(
        df,
        feature_columns=["active_leaf_count"],
        target_columns=["y_submit_without_validation"],
    )


def test_undeclared_run_constant_feature_not_flagged() -> None:
    """A feature that is empirically run-constant but NOT in the registry
    is allowed (the registry is the explicit list of dangerous features)."""
    df = _frame()
    df["task_id"] = df["run_id"]  # empirically run-constant, not declared
    offenders = audit(
        df,
        feature_columns=["task_id"],
        target_columns=["y_submit_without_validation"],
    )
    assert offenders == []
