"""Per-(target, source) budget flags infeasible cells correctly on a small fixture."""

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
