"""P1 — v0 no-regression gate.

Evaluates every P1 condition (a–h from TASKS.md § Workstream P) and
returns a structured verdict per condition plus a single overall
verdict. All inputs are derived from the same predictions/audit
artifacts used by other workstreams; no new training happens here.

Each `GateCondition.outcome` is `pass` | `fail` | `indeterminate`. The
overall verdict is:
- `pass`  iff every REQUIRED condition is `pass`.
- `fail`  iff any REQUIRED condition is `fail`.
- `indeterminate` otherwise.

Conditions are tagged required / informational explicitly so the
report can show which ones gate the verdict and which ones are
context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY
from coding_estimator.calibration.metrics import expected_calibration_error
from coding_estimator.calibration.recalibrate import IsotonicRecalibrator
from coding_estimator.eval.harness import predict_cell
from coding_estimator.eval.metrics import brier
from coding_estimator.leakage.guard import find_forbidden, load_forbidden_spec
from coding_estimator.leakage.run_constancy import audit as run_constancy_audit
from coding_estimator.splits.protocol import Fold, Split, loro

OutcomeT = Literal["pass", "fail", "indeterminate"]

HEADLINE_TARGETS: tuple[str, ...] = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
    "y_submit_without_validation",
)
TB_LIVE = "tb_live"
RETRO_SOURCES: tuple[str, ...] = ("swe_agent_pilot", "hermes_pilot_h5_v2")

P1B_ECE_DELTA_GATE: float = 0.05
P1D_LOSO_DELTA_GATE: float = 0.05
P1A_TIE_TOL: float = 1e-6  # numeric tolerance for "tie"


def _g4_wins_or_ties(brier_g2: float, brier_g4: float) -> bool:
    """G4 wins or ties iff its Brier is at most G2's Brier plus
    `P1A_TIE_TOL` (lower Brier is better)."""
    return brier_g4 <= brier_g2 + P1A_TIE_TOL


def _decide_verdict(conditions: list["GateCondition"]) -> OutcomeT:
    required = [c for c in conditions if c.required]
    if not required:
        return "indeterminate"
    if any(c.outcome == "fail" for c in required):
        return "fail"
    if all(c.outcome == "pass" for c in required):
        return "pass"
    return "indeterminate"


@dataclass(frozen=True)
class GateCondition:
    condition_id: str
    name: str
    required: bool
    outcome: OutcomeT
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateReport:
    verdict: OutcomeT
    conditions: list[GateCondition]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "conditions": [
                {
                    **asdict(c),
                    "evidence": dict(c.evidence),
                }
                for c in self.conditions
            ],
        }


# ---------- helpers ---------------------------------------------------------


def _per_source_loro_predictions(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    spec,
    target: str,
    source: str,
) -> pd.DataFrame:
    sub = checkpoints_df[checkpoints_df["source"] == source]
    if sub["run_id"].nunique() < 2:
        return pd.DataFrame()
    return predict_cell(
        checkpoints_df=sub,
        labels_df=labels_df,
        target=target,
        spec=spec,
        split=loro(sub),
        sources_in_train=(source,),
    )


def _loso_predictions(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    spec,
    target: str,
    test_source: str,
) -> pd.DataFrame:
    sources = sorted(checkpoints_df["source"].unique())
    if test_source not in sources or len(sources) < 2:
        return pd.DataFrame()
    test_runs = tuple(
        sorted(
            checkpoints_df.loc[
                checkpoints_df["source"] == test_source, "run_id"
            ].unique()
        )
    )
    train_runs = tuple(
        sorted(
            checkpoints_df.loc[
                checkpoints_df["source"] != test_source, "run_id"
            ].unique()
        )
    )
    if not test_runs or not train_runs:
        return pd.DataFrame()
    train_sources = tuple(s for s in sources if s != test_source)
    fold = Fold(
        fold_id=f"loso::{test_source}",
        train_run_ids=train_runs,
        test_run_ids=test_runs,
    )
    split = Split(scheme="loso", seed=0, folds=(fold,))
    return predict_cell(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        target=target,
        spec=spec,
        split=split,
        sources_in_train=train_sources,
    )


# ---------- P1.a ------------------------------------------------------------


def evaluate_p1a(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> GateCondition:
    """G4 ties or beats G2 on at least one v0 headline (target, source)
    under LORO. Lower Brier is better. Wins or ties on point estimate."""
    rows: list[dict[str, Any]] = []
    any_pass = False
    for source in sorted(checkpoints_df["source"].unique()):
        for target in HEADLINE_TARGETS:
            g2 = _per_source_loro_predictions(
                checkpoints_df=checkpoints_df,
                labels_df=labels_df,
                spec=TIME_ONLY,
                target=target,
                source=source,
            )
            g4 = _per_source_loro_predictions(
                checkpoints_df=checkpoints_df,
                labels_df=labels_df,
                spec=LEDGER_BASIC,
                target=target,
                source=source,
            )
            if g2.empty or g4.empty:
                continue
            b2 = brier(g2["_y"].astype(int).to_numpy(), g2["_p"].astype(float).to_numpy())
            b4 = brier(g4["_y"].astype(int).to_numpy(), g4["_p"].astype(float).to_numpy())
            row = {
                "source": source,
                "target": target,
                "brier_g2": float(b2),
                "brier_g4": float(b4),
                "wins_or_ties": _g4_wins_or_ties(float(b2), float(b4)),
            }
            rows.append(row)
            if row["wins_or_ties"]:
                any_pass = True
    if not rows:
        return GateCondition(
            condition_id="P1.a",
            name="G4 ties or beats G2 on at least one (target, source) under LORO",
            required=True,
            outcome="indeterminate",
            summary="no LORO predictions produced for any (target, source)",
            evidence={"rows": []},
        )
    outcome: OutcomeT = "pass" if any_pass else "fail"
    summary = (
        "G4 wins or ties G2 on "
        f"{sum(1 for r in rows if r['wins_or_ties'])} of {len(rows)} "
        "(target, source) cells."
    )
    return GateCondition(
        condition_id="P1.a",
        name="G4 ties or beats G2 on at least one (target, source) under LORO",
        required=True,
        outcome=outcome,
        summary=summary,
        evidence={"rows": rows},
    )


# ---------- P1.b ------------------------------------------------------------


class InsufficientRunsForRecalibrationError(RuntimeError):
    """Raised when run-disjoint isotonic recalibration is impossible
    (< 2 unique runs). The caller must treat this as INDETERMINATE
    rather than fall back to in-sample fitting — fitting on test data
    would silently give the gate an artificially low ECE."""


def _isotonic_recal_oof(predictions_df: pd.DataFrame) -> np.ndarray:
    """K-fold isotonic recalibration over run_ids. Mirrors
    `calibration.report.kfold_recalibrated_predictions` but kept local
    to avoid a circular dependency with the report module.

    Raises `InsufficientRunsForRecalibrationError` when `< 2` unique
    runs are present — P1.b's gate explicitly requires run-disjoint
    recalibration; an in-sample fallback would corrupt the gate.
    """
    if predictions_df.empty:
        return np.array([], dtype=float)
    runs = predictions_df["run_id"].astype(str).to_numpy()
    p_arr = predictions_df["_p"].astype(float).to_numpy()
    y_arr = predictions_df["_y"].astype(int).to_numpy()
    unique_runs = np.array(sorted(set(runs.tolist())))
    if len(unique_runs) < 2:
        raise InsufficientRunsForRecalibrationError(
            f"need ≥ 2 runs for run-disjoint recalibration, got {len(unique_runs)}"
        )
    out = np.empty_like(p_arr)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(unique_runs))
    ordered = unique_runs[perm]
    folds = np.array_split(ordered, min(5, len(ordered)))
    for fold_runs in folds:
        test_set = set(fold_runs.tolist())
        test_mask = np.array([r in test_set for r in runs])
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        y_train = y_arr[train_mask]
        if len(np.unique(y_train)) < 2:
            out[test_mask] = p_arr[test_mask]
            continue
        cal = IsotonicRecalibrator().fit(p_arr[train_mask], y_train)
        out[test_mask] = cal.transform(p_arr[test_mask])
    return out


def evaluate_p1b(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str = "y_success_eventual",
) -> GateCondition:
    """ECE_3bin after isotonic recalibration on tb_live under LORO must
    not increase by more than 0.05 going from G2 to G4."""
    g2 = _per_source_loro_predictions(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        spec=TIME_ONLY,
        target=target,
        source=TB_LIVE,
    )
    g4 = _per_source_loro_predictions(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        spec=LEDGER_BASIC,
        target=target,
        source=TB_LIVE,
    )
    if g2.empty or g4.empty:
        return GateCondition(
            condition_id="P1.b",
            name=(
                "ECE_3bin (after isotonic) on tb_live LORO does not "
                "increase by > 0.05 from G2 to G4"
            ),
            required=True,
            outcome="indeterminate",
            summary=f"no LORO predictions on tb_live for {target}",
            evidence={"target": target},
        )
    y2 = g2["_y"].astype(int).to_numpy()
    y4 = g4["_y"].astype(int).to_numpy()
    if len(np.unique(y2)) < 2 or len(np.unique(y4)) < 2:
        return GateCondition(
            condition_id="P1.b",
            name=(
                "ECE_3bin (after isotonic) on tb_live LORO does not "
                "increase by > 0.05 from G2 to G4"
            ),
            required=True,
            outcome="indeterminate",
            summary=(
                f"single-class y on tb_live for `{target}` "
                "(N=12 cohort is currently 12/12 successes)"
            ),
            evidence={
                "target": target,
                "g2_pos_rate": float(y2.mean()),
                "g4_pos_rate": float(y4.mean()),
            },
        )
    try:
        p2_iso = _isotonic_recal_oof(g2)
        p4_iso = _isotonic_recal_oof(g4)
    except InsufficientRunsForRecalibrationError as exc:
        return GateCondition(
            condition_id="P1.b",
            name=(
                "ECE_3bin (after isotonic) on tb_live LORO does not "
                "increase by > 0.05 from G2 to G4"
            ),
            required=True,
            outcome="indeterminate",
            summary=(
                "tb_live LORO predictions span < 2 runs; run-disjoint "
                "recalibration is impossible without falling back to "
                "in-sample fitting (which would corrupt the gate)"
            ),
            evidence={"target": target, "exception": str(exc)},
        )
    ece_g2 = expected_calibration_error(y2, p2_iso, n_bins=3)
    ece_g4 = expected_calibration_error(y4, p4_iso, n_bins=3)
    delta = float(ece_g4 - ece_g2)
    outcome: OutcomeT = "pass" if delta <= P1B_ECE_DELTA_GATE else "fail"
    return GateCondition(
        condition_id="P1.b",
        name=(
            "ECE_3bin (after isotonic) on tb_live LORO does not "
            "increase by > 0.05 from G2 to G4"
        ),
        required=True,
        outcome=outcome,
        summary=(
            f"ECE_3bin G2={ece_g2:.3f}, G4={ece_g4:.3f}, Δ={delta:+.3f} "
            f"(threshold +{P1B_ECE_DELTA_GATE:.2f})"
        ),
        evidence={
            "target": target,
            "ece_3bin_g2": float(ece_g2),
            "ece_3bin_g4": float(ece_g4),
            "delta": delta,
        },
    )


# ---------- P1.c ------------------------------------------------------------


def evaluate_p1c(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str = "y_success_eventual",
    bootstrap_b: int = 2000,
) -> GateCondition:
    """Combined-retrospective LORO: G4 beats G2 on `y_success_eventual`
    Brier with run-level bootstrap 95% CI excluding zero, on
    swe_agent_pilot ∪ hermes_pilot_h5_v2.

    The combined frame is treated as a single source for LORO. We fit
    G4 / G2 in the harness with `sources_in_train` reflecting the
    actual training sources observed.
    """
    sub = checkpoints_df[checkpoints_df["source"].isin(RETRO_SOURCES)]
    has_hermes_labels = (
        labels_df.loc[
            (labels_df["source"] == "hermes_pilot_h5_v2")
            & (labels_df["target_name"] == target),
            "label_value",
        ].notna().any()
    )
    if sub.empty:
        return GateCondition(
            condition_id="P1.c",
            name="Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero",
            required=True,
            outcome="indeterminate",
            summary="no retrospective sources present",
            evidence={"target": target},
        )
    if not has_hermes_labels:
        # The plan's contract is `swe ∪ hermes`. With hermes labels
        # missing the test is not the test the plan asked for —
        # `indeterminate`, not a degraded fail/pass on swe alone.
        return GateCondition(
            condition_id="P1.c",
            name="Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero",
            required=True,
            outcome="indeterminate",
            summary=(
                "hermes_pilot_h5_v2 labels not built into "
                "`datasets/labels_all.parquet` — combined retrospective "
                "is not testable as the plan defines it"
            ),
            evidence={
                "target": target,
                "missing_source_labels": "hermes_pilot_h5_v2",
                "note": (
                    "swe_agent_pilot-only result is available in the "
                    "Workstream H baselines; do NOT promote that to "
                    "the combined-retrospective gate"
                ),
            },
        )
    if sub["run_id"].nunique() < 2:
        return GateCondition(
            condition_id="P1.c",
            name="Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero",
            required=True,
            outcome="indeterminate",
            summary="< 2 runs in combined retrospective frame",
            evidence={"target": target},
        )
    train_sources = tuple(sorted(sub["source"].unique()))
    split = loro(sub)
    g2 = predict_cell(
        checkpoints_df=sub,
        labels_df=labels_df,
        target=target,
        spec=TIME_ONLY,
        split=split,
        sources_in_train=train_sources,
    )
    g4 = predict_cell(
        checkpoints_df=sub,
        labels_df=labels_df,
        target=target,
        spec=LEDGER_BASIC,
        split=split,
        sources_in_train=train_sources,
    )
    if g2.empty or g4.empty:
        return GateCondition(
            condition_id="P1.c",
            name="Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero",
            required=True,
            outcome="indeterminate",
            summary="combined LORO produced no predictions",
            evidence={"target": target, "train_sources": train_sources},
        )
    # Pair predictions by row to compute per-run Δ Brier.
    keys = ["run_id", "checkpoint_id"]
    paired = g4[keys + ["_y", "_p"]].rename(columns={"_p": "_p_g4"}).merge(
        g2[keys + ["_p"]].rename(columns={"_p": "_p_g2"}),
        on=keys,
        how="inner",
    )
    if paired.empty:
        return GateCondition(
            condition_id="P1.c",
            name="Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero",
            required=True,
            outcome="indeterminate",
            summary="paired predictions empty",
            evidence={"target": target},
        )
    y_arr = paired["_y"].astype(int).to_numpy()
    p_g2 = paired["_p_g2"].astype(float).to_numpy()
    p_g4 = paired["_p_g4"].astype(float).to_numpy()
    if len(np.unique(y_arr)) < 2:
        return GateCondition(
            condition_id="P1.c",
            name="Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero",
            required=True,
            outcome="indeterminate",
            summary=f"single-class y on retrospective frame for `{target}`",
            evidence={"target": target, "n_rows": int(len(y_arr))},
        )
    # Run-level bootstrap of (Brier_G2 - Brier_G4): positive ⇒ G4 wins.
    runs = sorted(paired["run_id"].astype(str).unique())
    by_run: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for rid in runs:
        m = paired["run_id"].astype(str) == rid
        by_run[rid] = (
            y_arr[m.to_numpy()],
            p_g2[m.to_numpy()],
            p_g4[m.to_numpy()],
        )
    rng = np.random.default_rng(0)
    deltas = np.empty(bootstrap_b, dtype=float)
    n = len(runs)
    for i in range(bootstrap_b):
        idx = rng.integers(0, n, size=n)
        ys: list[np.ndarray] = []
        p2s: list[np.ndarray] = []
        p4s: list[np.ndarray] = []
        for j in idx:
            y_b, p_g2_b, p_g4_b = by_run[runs[j]]
            ys.append(y_b)
            p2s.append(p_g2_b)
            p4s.append(p_g4_b)
        y_all = np.concatenate(ys)
        deltas[i] = float(
            np.mean((np.concatenate(p2s) - y_all) ** 2)
            - np.mean((np.concatenate(p4s) - y_all) ** 2)
        )
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    point = float(brier(y_arr, p_g2) - brier(y_arr, p_g4))
    excludes_zero = lo > 0.0
    outcome: OutcomeT = "pass" if excludes_zero else "fail"
    note = None
    if not has_hermes_labels:
        note = (
            "hermes_pilot_h5_v2 labels not built; combined retrospective "
            "is currently swe_agent_pilot alone — plan assumed both"
        )
    return GateCondition(
        condition_id="P1.c",
        name="Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero",
        required=True,
        outcome=outcome,
        summary=(
            f"Δ Brier (G2 − G4) = {point:+.3f}, 95% CI = [{lo:+.3f}, {hi:+.3f}]"
            + ("; CI excludes zero" if excludes_zero else "; CI INCLUDES zero")
        ),
        evidence={
            "target": target,
            "train_sources": train_sources,
            "delta_brier_point": point,
            "delta_brier_ci_low": lo,
            "delta_brier_ci_high": hi,
            "n_runs": n,
            "n_rows": int(len(y_arr)),
            "note": note,
        },
    )


# ---------- P1.d ------------------------------------------------------------


def evaluate_p1d(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str = "y_success_eventual",
) -> GateCondition:
    """LOSO->tb_live Brier ≤ within-source LORO Brier on tb_live + 0.05."""
    within = _per_source_loro_predictions(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        spec=LEDGER_BASIC,
        target=target,
        source=TB_LIVE,
    )
    transfer = _loso_predictions(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        spec=LEDGER_BASIC,
        target=target,
        test_source=TB_LIVE,
    )
    if within.empty or transfer.empty:
        return GateCondition(
            condition_id="P1.d",
            name="LOSO->tb_live Brier ≤ within-source LORO Brier + 0.05",
            required=True,
            outcome="indeterminate",
            summary="missing within-source or LOSO predictions on tb_live",
            evidence={"target": target},
        )
    y_within = within["_y"].astype(int).to_numpy()
    y_transfer = transfer["_y"].astype(int).to_numpy()
    if len(np.unique(y_within)) < 2 or len(np.unique(y_transfer)) < 2:
        return GateCondition(
            condition_id="P1.d",
            name="LOSO->tb_live Brier ≤ within-source LORO Brier + 0.05",
            required=True,
            outcome="indeterminate",
            summary=f"single-class y on tb_live for `{target}`",
            evidence={"target": target},
        )
    b_within = brier(y_within, within["_p"].astype(float).to_numpy())
    b_transfer = brier(y_transfer, transfer["_p"].astype(float).to_numpy())
    delta = float(b_transfer - b_within)
    outcome: OutcomeT = "pass" if delta <= P1D_LOSO_DELTA_GATE else "fail"
    return GateCondition(
        condition_id="P1.d",
        name="LOSO->tb_live Brier ≤ within-source LORO Brier + 0.05",
        required=True,
        outcome=outcome,
        summary=(
            f"LORO Brier={b_within:.3f}, LOSO Brier={b_transfer:.3f}, "
            f"Δ={delta:+.3f} (threshold +{P1D_LOSO_DELTA_GATE:.2f})"
        ),
        evidence={
            "target": target,
            "brier_within_source_loro": float(b_within),
            "brier_loso_to_tb_live": float(b_transfer),
            "delta": delta,
        },
    )


# ---------- P1.e ------------------------------------------------------------


def evaluate_p1e(checkpoints_df: pd.DataFrame) -> GateCondition:
    """Forbidden-column audit: zero hits in the checkpoints frame."""
    spec = load_forbidden_spec()
    hits = find_forbidden(checkpoints_df.columns, spec=spec)
    outcome: OutcomeT = "pass" if not hits else "fail"
    return GateCondition(
        condition_id="P1.e",
        name="Forbidden-column audit: zero hits",
        required=True,
        outcome=outcome,
        summary=(
            "no forbidden columns in the checkpoints frame"
            if outcome == "pass"
            else f"forbidden columns present: {hits}"
        ),
        evidence={
            "forbidden_exact_count": len(spec.exact),
            "forbidden_prefix_count": len(spec.prefixes),
            "forbidden_suffix_count": len(spec.suffixes),
            "hits": list(hits),
        },
    )


# ---------- P1.f ------------------------------------------------------------


def evaluate_p1f(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> GateCondition:
    """G4 training fold run-constancy: zero joint (feature, target)
    pairs that are run-constant within any fold's training data."""
    feature_cols = LEDGER_BASIC.feature_cols_for(())
    feature_cols_present = tuple(c for c in feature_cols if c in checkpoints_df.columns)
    audits: list[dict[str, Any]] = []
    fail_count = 0
    total_count = 0
    skipped_no_labels = 0
    skipped_empty_join = 0
    for source in sorted(checkpoints_df["source"].unique()):
        sub = checkpoints_df[checkpoints_df["source"] == source]
        if sub["run_id"].nunique() < 2:
            continue
        split = loro(sub)
        for fold in split.folds:
            for target in HEADLINE_TARGETS:
                lab = labels_df[
                    (labels_df["target_name"] == target)
                    & (~labels_df["is_masked"].astype(bool))
                ][["run_id", "checkpoint_id", "label_value"]]
                if lab.empty:
                    skipped_no_labels += 1
                    continue
                train = sub[sub["run_id"].isin(set(fold.train_run_ids))]
                joined = train.merge(lab, on=["run_id", "checkpoint_id"], how="inner")
                if joined.empty:
                    skipped_empty_join += 1
                    continue
                joined = joined.rename(columns={"label_value": "__target__"})
                pairs = run_constancy_audit(
                    joined,
                    feature_columns=feature_cols_present,
                    target_columns=("__target__",),
                )
                total_count += 1
                if pairs:
                    fail_count += 1
                    audits.append(
                        {
                            "source": source,
                            "target": target,
                            "fold_id": fold.fold_id,
                            "run_constant_pairs": [list(p) for p in pairs],
                        }
                    )
    outcome: OutcomeT = "pass" if fail_count == 0 else "fail"
    if total_count == 0:
        outcome = "indeterminate"
    return GateCondition(
        condition_id="P1.f",
        name="G4 training-fold run-constancy: zero joint (feature, target) pairs",
        required=True,
        outcome=outcome,
        summary=(
            f"audited {total_count} (source, target, fold) cells; "
            f"{fail_count} have run-constant pairs; "
            f"skipped {skipped_no_labels + skipped_empty_join} cells "
            f"({skipped_no_labels} no labels, {skipped_empty_join} empty join)"
        ),
        evidence={
            "audits": audits,
            "audited_cells": total_count,
            "skipped_no_labels": skipped_no_labels,
            "skipped_empty_join": skipped_empty_join,
        },
    )


# ---------- P1.g ------------------------------------------------------------


D5_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "n_runs_audited",
    "n_checkpoints_audited",
    "findings",
    "clean",
)


def evaluate_p1g(d5_audit_path: Path | None) -> GateCondition:
    """Online-vs-offline parity is deferred (Workstream M); replaced
    by the D5 behavioral leakage audit. Pass requires the audit JSON
    to ship with structured content (`schema_version`,
    `n_runs_audited`, `n_checkpoints_audited`, `findings`, `clean`),
    `clean: true`, and `n_runs_audited > 0`. A bare `{"clean": true}`
    is rejected — anyone could write that."""
    if d5_audit_path is None or not d5_audit_path.exists():
        return GateCondition(
            condition_id="P1.g",
            name="D5 behavioral leakage audit (Workstream M deferred)",
            required=True,
            outcome="indeterminate",
            summary=(
                "D5 audit artifact not provided; Workstream M is "
                "deferred — re-evaluate this condition once D5 ships "
                f"with required fields {list(D5_REQUIRED_FIELDS)}"
            ),
            evidence={"d5_audit_path": str(d5_audit_path) if d5_audit_path else None},
        )
    import json

    try:
        audit = json.loads(d5_audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return GateCondition(
            condition_id="P1.g",
            name="D5 behavioral leakage audit (Workstream M deferred)",
            required=True,
            outcome="fail",
            summary=f"D5 audit at `{d5_audit_path}` is not valid JSON: {exc}",
            evidence={"d5_audit_path": str(d5_audit_path)},
        )
    missing = [f for f in D5_REQUIRED_FIELDS if f not in audit]
    if missing:
        return GateCondition(
            condition_id="P1.g",
            name="D5 behavioral leakage audit (Workstream M deferred)",
            required=True,
            outcome="fail",
            summary=(
                f"D5 audit at `{d5_audit_path}` is missing required "
                f"fields {missing}; a bare `{{\"clean\": true}}` is not "
                "sufficient — see D5_REQUIRED_FIELDS"
            ),
            evidence={
                "d5_audit_path": str(d5_audit_path),
                "missing_fields": missing,
                "present_keys": sorted(audit.keys()),
            },
        )
    if int(audit.get("n_runs_audited", 0)) == 0:
        return GateCondition(
            condition_id="P1.g",
            name="D5 behavioral leakage audit (Workstream M deferred)",
            required=True,
            outcome="fail",
            summary="D5 audit reports zero runs audited — vacuous pass blocked",
            evidence={"d5_audit_path": str(d5_audit_path)},
        )
    findings = audit.get("findings", [])
    clean = bool(audit.get("clean", False))
    outcome: OutcomeT = "pass" if clean and not findings else "fail"
    return GateCondition(
        condition_id="P1.g",
        name="D5 behavioral leakage audit (Workstream M deferred)",
        required=True,
        outcome=outcome,
        summary=(
            f"D5 audit clean ({audit['n_runs_audited']} runs, "
            f"{audit['n_checkpoints_audited']} checkpoints; 0 findings)"
            if outcome == "pass"
            else f"D5 audit reports {len(findings)} findings or `clean: false`"
        ),
        evidence={
            "d5_audit_path": str(d5_audit_path),
            "schema_version": audit.get("schema_version"),
            "n_runs_audited": audit.get("n_runs_audited"),
            "n_checkpoints_audited": audit.get("n_checkpoints_audited"),
            "n_findings": len(findings),
        },
    )


# ---------- P1.h ------------------------------------------------------------


def evaluate_p1h(p1a: GateCondition) -> GateCondition:
    """If every P1.a winning cell is `y_submit_without_validation`, the
    headline win is a data property (run-constant target) not skill, and
    P1.h's caveat MUST be applied (verdict downgraded). This condition
    is REQUIRED — only-SWV wins must block the gate. The condition
    PASSES when the caveat is *not triggered* (winners span more than
    one target) or when there are no winners at all (P1.a already
    fails, so no caveat is needed)."""
    rows = p1a.evidence.get("rows", [])
    winning = [r for r in rows if r.get("wins_or_ties")]
    if not winning:
        return GateCondition(
            condition_id="P1.h",
            name="Submit-without-validation caveat",
            required=True,
            outcome="pass",
            summary=(
                "P1.a has no winners ⇒ no submit-without-validation "
                "caveat needed (the prior gate already does the work)"
            ),
            evidence={"winning_cells": []},
        )
    only_swv = all(r["target"] == "y_submit_without_validation" for r in winning)
    return GateCondition(
        condition_id="P1.h",
        name="Submit-without-validation caveat",
        required=True,
        outcome="pass" if not only_swv else "fail",
        summary=(
            "winning cells span multiple targets — caveat does not apply"
            if not only_swv
            else "BLOCKER: every P1.a winning cell is on "
            "`y_submit_without_validation`; that target is run-constant, "
            "so a non-trivial AUROC at non-terminal t is a data property, "
            "NOT model skill — gate must NOT pass on this evidence alone"
        ),
        evidence={"winning_cells": winning, "only_swv": only_swv},
    )


# ---------- aggregator ------------------------------------------------------


def evaluate_gate(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    d5_audit_path: Path | None = None,
) -> GateReport:
    p1a = evaluate_p1a(checkpoints_df=checkpoints_df, labels_df=labels_df)
    conditions: list[GateCondition] = [
        p1a,
        evaluate_p1b(checkpoints_df=checkpoints_df, labels_df=labels_df),
        evaluate_p1c(checkpoints_df=checkpoints_df, labels_df=labels_df),
        evaluate_p1d(checkpoints_df=checkpoints_df, labels_df=labels_df),
        evaluate_p1e(checkpoints_df),
        evaluate_p1f(checkpoints_df=checkpoints_df, labels_df=labels_df),
        evaluate_p1g(d5_audit_path),
        evaluate_p1h(p1a),
    ]
    return GateReport(verdict=_decide_verdict(conditions), conditions=conditions)


# ---------- renderer --------------------------------------------------------


def _outcome_badge(o: OutcomeT) -> str:
    return {"pass": "✅ pass", "fail": "❌ fail", "indeterminate": "⚠️ indeterminate"}[o]


def _executive_summary(report: GateReport) -> list[str]:
    """At-a-glance: which required conditions blocked the verdict?"""
    fails = [
        c for c in report.conditions if c.required and c.outcome == "fail"
    ]
    indets = [
        c for c in report.conditions
        if c.required and c.outcome == "indeterminate"
    ]
    lines: list[str] = []
    if report.verdict == "pass":
        lines.append("**All required conditions PASS.**")
    else:
        chunks: list[str] = []
        if fails:
            chunks.append(
                "FAIL on " + ", ".join(f"`{c.condition_id}`" for c in fails)
            )
        if indets:
            chunks.append(
                "INDETERMINATE on " + ", ".join(f"`{c.condition_id}`" for c in indets)
            )
        if chunks:
            lines.append("**Blocked by:** " + "; ".join(chunks) + ".")
        else:
            lines.append("**Blocked.**")
    return lines


def _render_p1a_evidence(rows: list[dict]) -> list[str]:
    out = [
        "| source | target | Brier G2 | Brier G4 | wins or ties |",
        "|---|---|---:|---:|:---:|",
    ]
    for r in sorted(rows, key=lambda r: (r.get("source", ""), r.get("target", ""))):
        wins = "✅" if r.get("wins_or_ties") else "❌"
        out.append(
            f"| {r.get('source', '?')} | {r.get('target', '?')} | "
            f"{r.get('brier_g2', float('nan')):.3f} | "
            f"{r.get('brier_g4', float('nan')):.3f} | {wins} |"
        )
    return out


def _render_evidence(condition_id: str, evidence: dict[str, Any]) -> list[str]:
    """Per-condition evidence renderer. P1.a gets a markdown sub-table;
    every other condition uses bullets."""
    if condition_id == "P1.a":
        rows = evidence.get("rows", [])
        if rows:
            return _render_p1a_evidence(rows)
    out: list[str] = []
    for k, v in sorted(evidence.items()):
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out.append(f"- `{k}`:")
            for item in v:
                pretty = ", ".join(f"{ik}={iv}" for ik, iv in sorted(item.items()))
                out.append(f"  - {pretty}")
        else:
            out.append(f"- `{k}`: {v}")
    return out


def render_gate_report(report: GateReport, *, summary: str | None = None) -> str:
    lines = [
        "# Estimator go/no-go gate (Workstream P)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
        f"## Overall verdict: {_outcome_badge(report.verdict).upper()}",
        "",
    ]
    lines.extend(_executive_summary(report))
    lines.append("")
    lines.append(
        "**v0 framing (recentered).** The primary v0 headline is "
        "**process-dynamics prediction** (`y_future_progress_drop_h5`, "
        "`y_validation_new_work_h5`). Terminal success "
        "(`y_success_eventual`) is reported as a secondary / negative "
        "result: ledger features do not yet beat elapsed time on it at "
        "this N. See `reports/V0_FINDINGS.md` for the publishable "
        "story; this gate keeps its original P1.a–h structure so "
        "no-regression on success is still measured."
    )
    lines.append("")
    if summary:
        lines.extend([summary, ""])
    lines.extend(
        [
            "Required conditions must all be `pass` for the overall verdict to "
            "be `pass`. Any `fail` on a required condition forces `fail`. "
            "Otherwise the verdict is `indeterminate`.",
            "",
            "## Condition summary",
            "",
            "| id | required | outcome | summary |",
            "|---|:---:|---|---|",
        ]
    )
    for c in report.conditions:
        req = "yes" if c.required else "no"
        lines.append(f"| `{c.condition_id}` | {req} | {_outcome_badge(c.outcome)} | {c.summary} |")
    lines.append("")
    for c in report.conditions:
        lines.append(f"## {c.condition_id} — {c.name}")
        lines.append("")
        lines.append(f"- outcome: {_outcome_badge(c.outcome)}")
        lines.append(f"- required: {'yes' if c.required else 'no'}")
        lines.append(f"- summary: {c.summary}")
        if c.evidence:
            lines.append("")
            lines.append("### Evidence")
            lines.append("")
            lines.extend(_render_evidence(c.condition_id, c.evidence))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_gate_report(
    path: Path, report: GateReport, summary: str | None = None
) -> Path:
    md = render_gate_report(report, summary=summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "HEADLINE_TARGETS",
    "P1A_TIE_TOL",
    "P1B_ECE_DELTA_GATE",
    "P1D_LOSO_DELTA_GATE",
    "GateCondition",
    "GateReport",
    "evaluate_p1a",
    "evaluate_p1b",
    "evaluate_p1c",
    "evaluate_p1d",
    "evaluate_p1e",
    "evaluate_p1f",
    "evaluate_p1g",
    "evaluate_p1h",
    "evaluate_gate",
    "render_gate_report",
    "write_gate_report",
]
