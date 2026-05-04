"""Disjointness on every scheme; jsonschema validation; warning on synthetic temporal.

Claim:
    For LORO over a frame with N unique run_ids, sp.loro produces N folds;
    in each fold (train_run_ids, test_run_ids) is a partition of those N
    run_ids: their union covers every input run, their intersection is
    empty, and |test_run_ids| == 1.

Plausible wrong implementations:
    - test_run_ids contains a run not in the input (e.g. mislabeled)
    - the held-out run leaks into train_run_ids (off-by-one when constructing
      the complement)
    - n_folds != n_runs (e.g. iterates over checkpoints not unique runs)
    - LTFO accidentally puts some runs of the held-out family into train
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

from coding_estimator.splits import protocol as sp

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "split_schema.json").read_text()
)


def _frame(n_runs: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_runs):
        rid = f"r{i}"
        src = "tb_live" if i < 3 else "swe_agent_pilot"
        fam = f"fam_{i // 2}"
        for c in range(3):
            rows.append(
                {
                    "run_id": rid,
                    "source": src,
                    "task_family": fam,
                    "checkpoint_wall_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
                    "timestamp_quality": "real",
                    "checkpoint_step": c,
                }
            )
    return pd.DataFrame(rows)


def test_schema_valid() -> None:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_loro_disjoint_and_validates() -> None:
    df = _frame()
    s = sp.loro(df)
    sp.assert_disjoint(s)
    jsonschema.validate(s.to_dict(), SCHEMA)
    assert {f.test_run_ids[0] for f in s.folds} == set(df["run_id"].unique())
    for f in s.folds:
        assert len(f.train_run_ids) == df["run_id"].nunique() - 1


def test_ltfo_disjoint() -> None:
    df = _frame()
    s = sp.ltfo(df)
    sp.assert_disjoint(s)
    jsonschema.validate(s.to_dict(), SCHEMA)


def test_loso_disjoint_and_two_folds() -> None:
    df = _frame()
    s = sp.loso(df)
    sp.assert_disjoint(s)
    jsonschema.validate(s.to_dict(), SCHEMA)
    assert len(s.folds) == 2
    held_sources = {df[df["run_id"].isin(f.test_run_ids)]["source"].iloc[0] for f in s.folds}
    assert held_sources == {"tb_live", "swe_agent_pilot"}


def test_holdout_disjoint_and_deterministic() -> None:
    df = _frame()
    a = sp.holdout(df, seed=42)
    b = sp.holdout(df, seed=42)
    sp.assert_disjoint(a)
    assert a.to_dict() == b.to_dict()


def test_temporal_warns_on_synthetic() -> None:
    df = _frame()
    df["timestamp_quality"] = "synthetic"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = sp.temporal(df, train_frac=0.5)
        assert any("synthetic" in str(w.message) for w in caught)
    sp.assert_disjoint(s)


def test_temporal_orders_by_start_time() -> None:
    df = _frame()
    s = sp.temporal(df, train_frac=0.5)
    assert s.folds[0].train_run_ids == ("r0", "r1", "r2")


def test_assert_disjoint_catches_overlap() -> None:
    bad = sp.Split(
        scheme="loro",
        seed=0,
        folds=(sp.Fold("f", ("r1", "r2"), ("r2",)),),
    )
    with pytest.raises(ValueError, match="r2"):
        sp.assert_disjoint(bad)


def test_loro_train_is_exact_complement_of_test() -> None:
    # For every fold: train_run_ids ∪ test_run_ids must equal the full set
    # of unique run_ids, AND the intersection must be empty.
    df = _frame()
    runs_in = set(df["run_id"].unique())
    s = sp.loro(df)
    assert len(s.folds) == len(runs_in)
    for fold in s.folds:
        train = set(fold.train_run_ids)
        test = set(fold.test_run_ids)
        assert len(test) == 1, "LORO test partition must hold exactly one run"
        assert train | test == runs_in, "train ∪ test must cover all input runs"
        assert train & test == set(), "train ∩ test must be empty"


def test_ltfo_held_out_family_runs_never_in_train() -> None:
    # Every run that belongs to the held-out family must appear ONLY in the
    # test partition of its fold and NEVER in the training partition.
    df = _frame()
    s = sp.ltfo(df)
    fam_to_runs: dict[str, set[str]] = {}
    for _, sub in df[["run_id", "task_family"]].drop_duplicates().groupby("task_family"):
        fam_to_runs[str(sub["task_family"].iloc[0])] = set(sub["run_id"])
    for fold in s.folds:
        # fold_id is "ltfo::<family>"
        family = fold.fold_id.split("::", 1)[1]
        held_runs = fam_to_runs[family]
        assert held_runs.issubset(set(fold.test_run_ids))
        assert held_runs.isdisjoint(set(fold.train_run_ids))


def test_temporal_requires_timestamps() -> None:
    df = _frame()
    df["checkpoint_wall_time"] = None
    with pytest.raises(ValueError, match="timestamps"):
        sp.temporal(df)
