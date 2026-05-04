"""Per-(target, source, split) data-budget snapshot.

Cells are flagged not-feasible when they cannot support honest evaluation.
The harness consumes these flags and skips infeasible cells with an
'n/a (insufficient data)' marker rather than emitting silent zeros.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from coding_estimator.io import write_csv

SplitScheme = Literal["loro", "ltfo", "loso", "holdout"]
MIN_PER_FOLD = 5


@dataclass(frozen=True)
class BudgetCell:
    target: str
    source: str
    split_scheme: str
    runs: int
    checkpoints: int
    positives: int
    negatives: int
    masked: int
    feasible: bool
    reason: str


def _per_fold_min(
    df: pd.DataFrame, target: str, scheme: str
) -> tuple[int, int, int]:
    """Return (min_pos_in_a_fold, min_neg_in_a_fold, n_folds) for run-level
    cross-validation. For loro the held-out fold is one run."""
    if scheme not in {"loro", "ltfo", "loso", "holdout"}:
        raise ValueError(f"unsupported split_scheme: {scheme}")
    pos_col = df[target]
    if scheme == "loro":
        groups = df["run_id"].unique()
        if len(groups) < 2:
            return 0, 0, len(groups)
        # For each held-out run, what does the *training* fold look like?
        min_pos = min(int(pos_col[df["run_id"] != g].sum()) for g in groups)
        min_neg = min(
            int((1 - pos_col[df["run_id"] != g].fillna(0)).sum()) for g in groups
        )
        return min_pos, min_neg, len(groups)
    # ltfo / loso / holdout: in v0 we treat them by their group columns
    group_col = {"ltfo": "task_family", "loso": "source"}.get(scheme, "run_id")
    if group_col not in df.columns:
        return 0, 0, 0
    groups = df[group_col].unique()
    if len(groups) < 2:
        return 0, 0, len(groups)
    min_pos = min(int(pos_col[df[group_col] != g].sum()) for g in groups)
    min_neg = min(int((1 - pos_col[df[group_col] != g].fillna(0)).sum()) for g in groups)
    return min_pos, min_neg, len(groups)


def compute_budget(
    df: pd.DataFrame,
    *,
    targets: Iterable[str],
    sources: Iterable[str] | None = None,
    schemes: Iterable[str] = ("loro",),
) -> list[BudgetCell]:
    cells: list[BudgetCell] = []
    sources_list = list(sources) if sources is not None else sorted(df["source"].unique())
    for source in sources_list:
        sub_all = df[df["source"] == source]
        for target in targets:
            if target not in sub_all.columns:
                cells.append(
                    BudgetCell(target, source, "n/a", 0, 0, 0, 0, 0, False, "target_missing")
                )
                continue
            masked = int(sub_all[target].isna().sum())
            sub = sub_all[sub_all[target].notna()].copy()
            sub[target] = sub[target].astype(int)
            runs = int(sub["run_id"].nunique())
            checkpoints = int(len(sub))
            positives = int(sub[target].sum())
            negatives = checkpoints - positives
            for scheme in schemes:
                if checkpoints == 0:
                    cells.append(
                        BudgetCell(
                            target, source, scheme, runs, 0, 0, 0, masked, False, "no_rows"
                        )
                    )
                    continue
                min_pos, min_neg, _ = _per_fold_min(sub, target, scheme)
                feasible = min_pos >= MIN_PER_FOLD and min_neg >= MIN_PER_FOLD
                reason = (
                    "ok"
                    if feasible
                    else f"min_per_fold<{MIN_PER_FOLD} (pos={min_pos},neg={min_neg})"
                )
                cells.append(
                    BudgetCell(
                        target,
                        source,
                        scheme,
                        runs,
                        checkpoints,
                        positives,
                        negatives,
                        masked,
                        feasible,
                        reason,
                    )
                )
    return cells


def cells_to_frame(cells: Iterable[BudgetCell]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in cells])


def render_markdown(cells: Iterable[BudgetCell]) -> str:
    df = cells_to_frame(cells).sort_values(["source", "target", "split_scheme"])
    lines = [
        "# Data budget snapshot",
        "",
        "| source | target | scheme | runs | ckpts | pos | neg | masked | feasible | reason |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.source} | {r.target} | {r.split_scheme} | {r.runs} | {r.checkpoints} | "
            f"{r.positives} | {r.negatives} | {r.masked} | {r.feasible} | {r.reason} |"
        )
    return "\n".join(lines) + "\n"


def write_budget_artifacts(
    df: pd.DataFrame,
    *,
    targets: Iterable[str],
    out_dir: Path,
    schemes: Iterable[str] = ("loro",),
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = compute_budget(df, targets=targets, schemes=schemes)
    csv_path = write_csv(
        cells_to_frame(cells),
        out_dir / "data_budget.csv",
        sort_by=["source", "target", "split_scheme"],
    )
    md_path = out_dir / "data_budget.md"
    md_path.write_text(render_markdown(cells), encoding="utf-8", newline="\n")
    return csv_path, md_path
