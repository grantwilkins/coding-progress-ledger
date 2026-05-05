#!/usr/bin/env python
"""G5 driver — evaluate G2 / G4 / G5 / G4+G5 across the headline targets.

The recentered v0 question: are ledger features (and especially their
*dynamics* — slopes, streaks, recency) predictive of process dynamics
(progress drops, validation new-work) where they are not predictive of
terminal success?

Outputs a per-source LORO comparison table for each target.

Usage:
    uv run python scripts/run_g5_eval.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out-dir reports/g5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY
from coding_estimator.baselines.base import BaselineSpec
from coding_estimator.baselines.ledger_dynamics import LEDGER_DYNAMICS
from coding_estimator.checkpoints.dynamics import G5_FEATURES, attach_g5_features
from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.bootstrap import bootstrap_brier_ci, brier_per_run
from coding_estimator.eval.harness import predict_cell
from coding_estimator.eval.metrics import auroc, brier
from coding_estimator.io import write_csv, write_json
from coding_estimator.splits.protocol import loro

HEADLINE_TARGETS: tuple[str, ...] = (
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
    "y_success_eventual",
)


def _g4_plus_g5_spec() -> BaselineSpec:
    base = LEDGER_BASIC.feature_cols_for(())
    cols = base + G5_FEATURES
    return BaselineSpec(
        name="g4_plus_g5", feature_cols_for=lambda _s: cols
    )


@dataclass(frozen=True)
class G5Cell:
    source: str
    target: str
    model: str
    n_runs: int
    n_checkpoints: int
    positive_rate: float
    auroc: float | None
    brier: float
    brier_ci_low: float
    brier_ci_high: float
    note: str | None = None


def _evaluate(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    spec: BaselineSpec,
    target: str,
    source: str,
    bootstrap_b: int,
) -> G5Cell | None:
    sub = checkpoints_df[checkpoints_df["source"] == source]
    if sub["run_id"].nunique() < 2:
        return None
    preds = predict_cell(
        checkpoints_df=sub,
        labels_df=labels_df,
        target=target,
        spec=spec,
        split=loro(sub),
        sources_in_train=(source,),
    )
    if preds.empty:
        return None
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
        brier_per_run(y_by_run, p_by_run), b=bootstrap_b, seed=0
    )
    return G5Cell(
        source=source,
        target=target,
        model=spec.name,
        n_runs=int(preds["run_id"].nunique()),
        n_checkpoints=int(len(preds)),
        positive_rate=float(y.mean()),
        auroc=auroc(y, p),
        brier=brier(y, p),
        brier_ci_low=lo,
        brier_ci_high=hi,
        note=None if len(np.unique(y)) >= 2 else "single-class y",
    )


def run(*, checkpoints_path: Path, labels_path: Path, out_dir: Path) -> Path:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    checkpoints_df = attach_g5_features(checkpoints_df)
    labels_df = pd.read_parquet(labels_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = (TIME_ONLY, LEDGER_BASIC, LEDGER_DYNAMICS, _g4_plus_g5_spec())
    cells: list[G5Cell] = []
    for source in sorted(checkpoints_df["source"].unique()):
        for target in HEADLINE_TARGETS:
            for spec in specs:
                cell = _evaluate(
                    checkpoints_df=checkpoints_df,
                    labels_df=labels_df,
                    spec=spec,
                    target=target,
                    source=source,
                    bootstrap_b=500,
                )
                if cell is not None:
                    cells.append(cell)

    csv_path = write_csv(
        pd.DataFrame([asdict(c) for c in cells]),
        out_dir / "g5_metrics.csv",
        sort_by=["source", "target", "model"],
    )
    write_json(
        [asdict(c) for c in cells], out_dir / "g5_metrics.json"
    )
    md_path = out_dir / "g5_eval.md"
    md_path.write_text(_render(cells), encoding="utf-8", newline="\n")
    return md_path


def _render(cells: list[G5Cell]) -> str:
    lines = [
        "# G5 ledger-dynamics evaluation",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
        "G2 (`time_only`) vs G4 (`ledger_basic`) vs G5 (`ledger_dynamics`) "
        "vs `g4_plus_g5` per source under LORO across the recentered "
        "v0 headline targets. Lower Brier is better. Run-level "
        "bootstrap 95% CI (B=500).",
        "",
    ]
    by_target_source: dict[tuple[str, str], dict[str, G5Cell]] = {}
    for c in cells:
        by_target_source.setdefault((c.target, c.source), {})[c.model] = c

    primary = ("y_future_progress_drop_h5", "y_validation_new_work_h5")
    secondary = ("y_success_eventual",)

    def _section(title: str, targets: tuple[str, ...]) -> list[str]:
        out = [f"## {title}", "", "| target | source | model | n_runs | n_ckpts | pos | AUROC | Brier | Brier 95% CI | note |", "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
        for target in targets:
            for (t, source), cells_by_model in sorted(by_target_source.items()):
                if t != target:
                    continue
                for model in ("time_only", "ledger_basic", "ledger_dynamics", "g4_plus_g5"):
                    c = cells_by_model.get(model)
                    if c is None:
                        continue
                    auroc_s = "n/a" if c.auroc is None else f"{c.auroc:.3f}"
                    out.append(
                        f"| {c.target} | {c.source} | {c.model} | {c.n_runs} | "
                        f"{c.n_checkpoints} | {c.positive_rate:.3f} | {auroc_s} | "
                        f"{c.brier:.3f} | [{c.brier_ci_low:.3f}, {c.brier_ci_high:.3f}] | "
                        f"{c.note or ''} |"
                    )
        out.append("")
        return out

    lines.extend(_section("Primary headline targets (process dynamics)", primary))
    lines.extend(_section("Secondary headline target (terminal success — negative result)", secondary))

    # Δ summary
    lines.append("## Δ Brier vs G2 by (target, source)")
    lines.append("")
    lines.append("Positive = G2 better; negative = the named model better.")
    lines.append("")
    lines.append("| target | source | G2 Brier | G4 - G2 | G5 - G2 | (G4+G5) - G2 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for (target, source), cells_by_model in sorted(by_target_source.items()):
        g2 = cells_by_model.get("time_only")
        if g2 is None:
            continue
        def _delta(name: str) -> str:
            c = cells_by_model.get(name)
            return "n/a" if c is None else f"{c.brier - g2.brier:+.3f}"
        lines.append(
            f"| {target} | {source} | {g2.brier:.3f} | "
            f"{_delta('ledger_basic')} | {_delta('ledger_dynamics')} | "
            f"{_delta('g4_plus_g5')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        out_dir=args.out_dir,
    )
    print(f"wrote g5 eval to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
