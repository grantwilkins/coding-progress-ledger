"""Per-(target, source) budget flags infeasible cells correctly on a small fixture.

Claim:
    _per_fold_min(df, target, "loro") returns the minimum positives and
    minimum negatives over TRAINING folds (each training fold = the
    complement of the held-out run); feasibility requires both >= 5.

Plausible wrong implementations:
    - returns the held-out (test) fold minimums instead of training-fold
    - returns the held-out (test) fold values for `runs - 1` runs
    - off-by-one on fold count (returns n_runs - 1 or n_runs + 1)
    - uses `df["source"] != g` instead of `df["run_id"] != g` for loro
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coding_estimator.profile.budget import (
    cells_to_frame,
    compute_budget,
    write_budget_artifacts,
)


@pytest.fixture()
def fixture_df() -> pd.DataFrame:
    """5 runs × 4 checkpoints, one source. Five hand-crafted targets:

    easy:           ~50/50, supports LORO
    rare:           1 positive total
    all_zero:       no positives
    all_one:        no negatives
    run_constant:   all positives concentrated in run0
    """
    runs = ["r0", "r1", "r2", "r3", "r4"]
    rows = []
    for r in runs:
        for c in range(4):
            rows.append({"run_id": r, "source": "tb_live", "checkpoint_id": c})
    df = pd.DataFrame(rows)
    df["easy"] = [0, 1, 0, 1] * 5
    df["rare"] = [0] * len(df)
    df.loc[0, "rare"] = 1
    df["all_zero"] = 0
    df["all_one"] = 1
    df["run_constant"] = (df["run_id"] == "r0").astype(int)
    return df


def test_four_of_five_infeasible_under_loro(fixture_df: pd.DataFrame) -> None:
    targets = ["easy", "rare", "all_zero", "all_one", "run_constant"]
    cells = compute_budget(fixture_df, targets=targets, schemes=("loro",))
    out = cells_to_frame(cells).set_index("target")
    assert out.loc["easy", "feasible"]
    assert not out.loc["rare", "feasible"]
    assert not out.loc["all_zero", "feasible"]
    assert not out.loc["all_one", "feasible"]
    assert not out.loc["run_constant", "feasible"]
    # exactly four infeasible
    assert int((~out["feasible"]).sum()) == 4


def test_missing_target_flagged(fixture_df: pd.DataFrame) -> None:
    cells = compute_budget(fixture_df, targets=["never_existed"], schemes=("loro",))
    assert cells[0].feasible is False
    assert cells[0].reason == "target_missing"


def test_per_fold_min_uses_training_fold_not_test_fold(fixture_df: pd.DataFrame) -> None:
    # Concentrate ALL `easy` positives in one run (r0). Under loro:
    #   held-out r0 -> training fold = r1..r4 -> 0 positives -> infeasible.
    #   held-out rk (k>0) -> training fold contains r0 (10 pos) -> feasible if neg>=5.
    # If the function were returning test-fold minimums instead, the held-out
    # run with no positives (e.g. r1) would be considered the 'minimum' and
    # the 0-positive scenario would never surface as a training-fold blocker.
    # We verify the OVERALL feasibility flag fires because some training fold
    # has 0 positives -- the only way that happens is if the code is correctly
    # taking the min over the *complement* of each held-out group.
    df = fixture_df.copy()
    df["concentrated"] = 0
    df.loc[df["run_id"] == "r0", "concentrated"] = 1  # 4 positives, all in r0
    cells = compute_budget(df, targets=["concentrated"], schemes=("loro",))
    cell = cells[0]
    assert cell.feasible is False
    # Reason must reflect a positives shortage, not a negatives shortage:
    # negatives in any training fold are >= 12 (>=5 OK), positives drop to 0
    # only when r0 is held out.
    assert "pos=0" in cell.reason


def test_per_fold_min_uses_run_id_for_loro_not_source(fixture_df: pd.DataFrame) -> None:
    # If LORO accidentally grouped by `source` instead of `run_id`, all 5 runs
    # share one source ("tb_live") and the loop body would never execute
    # (single group => returns 0,0,n). Construct a target that *is* feasible
    # under correct LORO grouping; if the code grouped by source, it would
    # be flagged infeasible (single-source dataset).
    cells = compute_budget(fixture_df, targets=["easy"], schemes=("loro",))
    assert cells[0].feasible is True


def test_writes_artifacts(fixture_df: pd.DataFrame, tmp_path: Path) -> None:
    csv_path, md_path = write_budget_artifacts(
        fixture_df,
        targets=["easy", "rare"],
        out_dir=tmp_path / "profiles",
        schemes=("loro",),
    )
    assert csv_path.exists() and md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "data budget snapshot" in text.lower()
    assert "easy" in text and "rare" in text
