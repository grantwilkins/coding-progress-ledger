"""L3 — retrospective→live transfer evaluation.

Train a logistic regression on the G4 ledger-basic feature groups using
`swe_agent_pilot` ∪ `hermes_pilot_h5_v2`, then evaluate on `tb_live`.
Repeat with each feature group ablated to identify which groups carry
the transfer signal.

A baseline is also run with only `elapsed_steps` (G2 reference) so the
report shows whether the ledger features add anything over time-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from coding_estimator.baselines.base import BaselineSpec, fit_binary
from coding_estimator.calibration.metrics import expected_calibration_error
from coding_estimator.checkpoints.features.registry import GROUPS
from coding_estimator.eval.bootstrap import bootstrap_brier_ci, brier_per_run
from coding_estimator.eval.metrics import OUTPUT_CLIP, auroc, brier

LEDGER_GROUPS: tuple[str, ...] = ("closure", "frontier", "instability", "discovery")
RETRO_SOURCES: tuple[str, ...] = ("swe_agent_pilot", "hermes_pilot_h5_v2")
LIVE_SOURCE: str = "tb_live"
TARGETS: tuple[str, ...] = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
    "y_submit_without_validation",
)


def _group_columns(groups: tuple[str, ...]) -> tuple[str, ...]:
    cols: list[str] = []
    for g in groups:
        for f in GROUPS[g]:
            if f.dtype not in ("int", "float", "bool"):
                continue
            cols.append(f.column_name)
    return tuple(cols)


def _spec_for(name: str, groups: tuple[str, ...]) -> BaselineSpec:
    cols = _group_columns(groups)
    return BaselineSpec(name=name, feature_cols_for=lambda _src: cols)


def _join_target(
    checkpoints_df: pd.DataFrame, labels_df: pd.DataFrame, target: str
) -> pd.DataFrame:
    lab = labels_df[
        (labels_df["target_name"] == target)
        & (~labels_df["is_masked"].astype(bool))
    ][["run_id", "checkpoint_id", "label_value"]]
    if lab.empty:
        return pd.DataFrame()
    j = checkpoints_df.merge(lab, on=["run_id", "checkpoint_id"], how="inner")
    j = j.rename(columns={"label_value": "_y"})
    j["_y"] = j["_y"].astype(float)
    return j


@dataclass(frozen=True)
class TransferRow:
    target: str
    config: str
    feature_group_ablated: str | None
    n_train_runs: int
    n_test_runs: int
    n_test_checkpoints: int
    positive_rate: float
    auroc: float | None
    brier: float
    brier_ci_low: float
    brier_ci_high: float
    ece: float
    note: str | None = None


def evaluate_transfer(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    bootstrap_b: int = 1000,
    bootstrap_seed: int = 0,
) -> list[TransferRow]:
    rows: list[TransferRow] = []
    train_mask = checkpoints_df["source"].isin(RETRO_SOURCES)
    test_mask = checkpoints_df["source"] == LIVE_SOURCE
    train_df = checkpoints_df[train_mask]
    test_df = checkpoints_df[test_mask]
    if train_df.empty or test_df.empty:
        return rows

    configs: list[tuple[str, BaselineSpec, str | None]] = [
        ("g2_time_only", _spec_for("g2_time_only", ()), None),
    ]
    configs[0] = (
        "g2_time_only",
        BaselineSpec(name="g2_time_only", feature_cols_for=lambda _s: ("elapsed_steps",)),
        None,
    )
    configs.append(
        ("g4_full", _spec_for("g4_full", LEDGER_GROUPS), None),
    )
    for group in LEDGER_GROUPS:
        rest = tuple(g for g in LEDGER_GROUPS if g != group)
        configs.append((f"g4_minus_{group}", _spec_for(f"g4_minus_{group}", rest), group))

    for target in TARGETS:
        joined = _join_target(checkpoints_df, labels_df, target)
        if joined.empty:
            continue
        train = joined[joined["source"].isin(RETRO_SOURCES)]
        test = joined[joined["source"] == LIVE_SOURCE]
        if train.empty or test.empty:
            continue
        y_train = train["_y"].astype(int).to_numpy()
        for config_name, spec, ablated in configs:
            sources_in_train = tuple(sorted(train["source"].unique()))
            try:
                fitted = fit_binary(spec, train, y_train, sources_in_train)
                probs = np.clip(fitted.predict_proba(test), *OUTPUT_CLIP)
            except (ValueError, KeyError) as exc:
                rows.append(
                    TransferRow(
                        target=target,
                        config=config_name,
                        feature_group_ablated=ablated,
                        n_train_runs=int(train["run_id"].nunique()),
                        n_test_runs=int(test["run_id"].nunique()),
                        n_test_checkpoints=0,
                        positive_rate=float(test["_y"].mean()),
                        auroc=None,
                        brier=float("nan"),
                        brier_ci_low=float("nan"),
                        brier_ci_high=float("nan"),
                        ece=float("nan"),
                        note=f"fit failed: {exc}",
                    )
                )
                continue
            scored = test.assign(_p=probs)
            y_by_run: dict[str, np.ndarray] = {}
            p_by_run: dict[str, np.ndarray] = {}
            for rid, sub in scored.groupby("run_id", sort=True):
                key = str(rid)
                y_by_run[key] = sub["_y"].astype(int).to_numpy()
                p_by_run[key] = sub["_p"].to_numpy()
            y_all = np.concatenate([y_by_run[r] for r in sorted(y_by_run)])
            p_all = np.concatenate([p_by_run[r] for r in sorted(y_by_run)])
            lo, hi = bootstrap_brier_ci(
                brier_per_run(y_by_run, p_by_run),
                b=bootstrap_b, seed=bootstrap_seed,
            )
            rows.append(
                TransferRow(
                    target=target,
                    config=config_name,
                    feature_group_ablated=ablated,
                    n_train_runs=int(train["run_id"].nunique()),
                    n_test_runs=int(test["run_id"].nunique()),
                    n_test_checkpoints=int(len(y_all)),
                    positive_rate=float(y_all.mean()),
                    auroc=auroc(y_all, p_all),
                    brier=brier(y_all, p_all),
                    brier_ci_low=lo,
                    brier_ci_high=hi,
                    ece=expected_calibration_error(y_all, p_all, n_bins=3),
                    note=None,
                )
            )
    return rows


def render_transfer_report(rows: list[TransferRow], *, summary: str | None = None) -> str:
    lines = [
        "# Retrospective → live transfer (L3)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
    ]
    if summary:
        lines.extend([summary, ""])
    if not rows:
        lines.append(
            "_No transfer rows produced — either retrospective or live source missing labels._"
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "Train: `swe_agent_pilot ∪ hermes_pilot_h5_v2`. Test: `tb_live`. "
            "ECE uses 3 equal-width bins (10-bin ECE is unestimable at N=12).",
            "",
            "## Per-target metrics",
            "",
            "| target | config | ablated | n_train | n_test | n_ckpts | pos | AUROC | Brier | Brier 95% CI | ECE_3bin |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for r in sorted(rows, key=lambda r: (r.target, r.config)):
        ci = (
            "n/a"
            if np.isnan(r.brier_ci_low) or np.isnan(r.brier_ci_high)
            else f"[{r.brier_ci_low:.3f}, {r.brier_ci_high:.3f}]"
        )
        auroc_s = "n/a" if r.auroc is None else f"{r.auroc:.3f}"
        brier_s = "n/a" if np.isnan(r.brier) else f"{r.brier:.3f}"
        ece_s = "n/a" if np.isnan(r.ece) else f"{r.ece:.3f}"
        lines.append(
            f"| {r.target} | {r.config} | {r.feature_group_ablated or '-'} | "
            f"{r.n_train_runs} | {r.n_test_runs} | {r.n_test_checkpoints} | "
            f"{r.positive_rate:.3f} | {auroc_s} | {brier_s} | {ci} | {ece_s} |"
        )
    lines.append("")
    by_target: dict[str, dict[str, TransferRow]] = {}
    for r in rows:
        by_target.setdefault(r.target, {})[r.config] = r
    lines.append("## Ablation Δ (Brier vs `g4_full`)")
    lines.append("")
    lines.append(
        "Positive Δ ⇒ removing the group **hurt** transfer ⇒ that group carried signal.\n"
    )
    lines.append("| target | full Brier | minus_closure | minus_frontier | minus_instability | minus_discovery |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for target in sorted(by_target):
        bag = by_target[target]
        full = bag.get("g4_full")
        if full is None or np.isnan(full.brier):
            continue
        deltas = []
        for group in LEDGER_GROUPS:
            row = bag.get(f"g4_minus_{group}")
            if row is None or np.isnan(row.brier):
                deltas.append("n/a")
            else:
                deltas.append(f"{row.brier - full.brier:+.3f}")
        lines.append(
            f"| {target} | {full.brier:.3f} | {deltas[0]} | {deltas[1]} | {deltas[2]} | {deltas[3]} |"
        )
    return "\n".join(lines) + "\n"


def write_transfer_report(
    path: Path, rows: list[TransferRow], summary: str | None = None
) -> Path:
    md = render_transfer_report(rows, summary=summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "LEDGER_GROUPS",
    "RETRO_SOURCES",
    "LIVE_SOURCE",
    "TARGETS",
    "TransferRow",
    "evaluate_transfer",
    "render_transfer_report",
    "write_transfer_report",
]
