"""K1 — TB-only checkpoint evaluation.

Run the v0 logreg/G4 model + G2 baseline on `tb_live` alone under LORO,
then report headline Brier/AUROC/ECE per (model, target) and a feasibility
note when N is small.

This module is read-only over the prepared artifacts; the calling
script (`scripts/run_tb_live_eval.py`) handles I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY, BaselineSpec
from coding_estimator.calibration.metrics import expected_calibration_error
from coding_estimator.eval.bootstrap import bootstrap_brier_ci, brier_per_run
from coding_estimator.eval.harness import EvalCell, evaluate_cell, predict_cell
from coding_estimator.eval.metrics import auroc, brier
from coding_estimator.splits.protocol import loro

TB_LIVE = "tb_live"
K1_TARGETS: tuple[str, ...] = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
    "y_submit_without_validation",
)
K1_MODELS: tuple[BaselineSpec, ...] = (TIME_ONLY, LEDGER_BASIC)


@dataclass(frozen=True)
class TBOnlyCell:
    target: str
    model: str
    n_runs: int
    n_checkpoints: int
    positive_rate: float
    auroc: float | None
    brier: float
    ece: float
    brier_ci_low: float
    brier_ci_high: float
    note: str | None = None


def evaluate_tb_only(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    bootstrap_b: int = 1000,
    bootstrap_seed: int = 0,
) -> tuple[list[TBOnlyCell], list[EvalCell]]:
    """Returns (TBOnlyCells_for_report, raw_EvalCells_for_csv)."""
    sub = checkpoints_df[checkpoints_df["source"] == TB_LIVE]
    if sub["run_id"].nunique() < 2:
        return [], []
    split = loro(sub)
    out_cells: list[TBOnlyCell] = []
    raw_cells: list[EvalCell] = []
    for spec in K1_MODELS:
        for target in K1_TARGETS:
            preds = predict_cell(
                checkpoints_df=sub,
                labels_df=labels_df,
                target=target,
                spec=spec,
                split=split,
                sources_in_train=(TB_LIVE,),
            )
            raw_cells.append(
                evaluate_cell(
                    checkpoints_df=sub,
                    labels_df=labels_df,
                    target=target,
                    spec=spec,
                    split=split,
                    source_slice=TB_LIVE,
                    sources_in_train=(TB_LIVE,),
                    feasible=True,
                    bootstrap_b=bootstrap_b,
                    bootstrap_seed=bootstrap_seed,
                )
            )
            if preds.empty:
                continue
            y = preds["_y"].astype(int).to_numpy()
            p = preds["_p"].astype(float).to_numpy()
            y_by_run = {
                str(rid): grp["_y"].astype(int).to_numpy()
                for rid, grp in preds.groupby("run_id", sort=True)
            }
            p_by_run = {
                str(rid): grp["_p"].astype(float).to_numpy()
                for rid, grp in preds.groupby("run_id", sort=True)
            }
            lo, hi = bootstrap_brier_ci(
                brier_per_run(y_by_run, p_by_run),
                b=bootstrap_b, seed=bootstrap_seed,
            )
            out_cells.append(
                TBOnlyCell(
                    target=target,
                    model=spec.name,
                    n_runs=int(preds["run_id"].nunique()),
                    n_checkpoints=int(len(preds)),
                    positive_rate=float(y.mean()),
                    auroc=auroc(y, p),
                    brier=brier(y, p),
                    ece=expected_calibration_error(y, p, n_bins=3),
                    brier_ci_low=lo,
                    brier_ci_high=hi,
                    note=None if len(np.unique(y)) >= 2 else "single-class y",
                )
            )
    return out_cells, raw_cells


def render_tb_only_report(
    cells: list[TBOnlyCell], *, summary: str | None = None
) -> str:
    from datetime import UTC, datetime

    lines = [
        "# TB-live only — checkpoint evaluation (K1)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
    ]
    if summary:
        lines.extend([summary, ""])
    if not cells:
        lines.append("_No tb_live cells produced predictions; tb_live is below the LORO budget._")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "Per-target metrics on `tb_live` under LORO. ECE uses 3 equal-width bins (10-bin "
            "ECE is unestimable at N=12).",
            "",
            "| target | model | n_runs | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | ECE_3bin | note |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for r in sorted(cells, key=lambda r: (r.target, r.model)):
        ci = f"[{r.brier_ci_low:.3f}, {r.brier_ci_high:.3f}]"
        auroc_s = "n/a" if r.auroc is None else f"{r.auroc:.3f}"
        lines.append(
            f"| {r.target} | {r.model} | {r.n_runs} | {r.n_checkpoints} | "
            f"{r.positive_rate:.3f} | {auroc_s} | {r.brier:.3f} | {ci} | "
            f"{r.ece:.3f} | {r.note or ''} |"
        )
    lines.append("")
    by_target: dict[str, dict[str, TBOnlyCell]] = {}
    for r in cells:
        by_target.setdefault(r.target, {})[r.model] = r
    lines.append("## G4 vs G2 (Brier)")
    lines.append("")
    lines.append("| target | G2 Brier | G4 Brier | Δ (G4 - G2) | G4 wins-or-ties |")
    lines.append("|---|---:|---:|---:|:---:|")
    for target in sorted(by_target):
        bag = by_target[target]
        g2 = bag.get("time_only")
        g4 = bag.get("ledger_basic")
        if g2 is None or g4 is None:
            continue
        delta = g4.brier - g2.brier
        lines.append(
            f"| {target} | {g2.brier:.3f} | {g4.brier:.3f} | {delta:+.3f} | "
            f"{'yes' if delta <= 0 else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def write_tb_only_report(path: Path, cells: list[TBOnlyCell], summary: str | None = None) -> Path:
    md = render_tb_only_report(cells, summary=summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "TB_LIVE",
    "TBOnlyCell",
    "evaluate_tb_only",
    "render_tb_only_report",
    "write_tb_only_report",
]
