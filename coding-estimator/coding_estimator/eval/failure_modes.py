"""Workstream O — adversarial failure-mode tests for the v0 estimator.

O1 — progress-overconfidence: median P(success) on high-progress
     failures must stay below a measurable bound. Existence of the
     bound is the gate; threshold is tunable.
O5 — source-leakage: adding the `source_task` feature group must not
     materially improve LOSO performance. If it does, the model was
     memorizing dataset identity.
O7 — timeout-bias: G4 (ledger-basic) must beat G2 (time-only) on
     terminal-success Brier by ≥ 0.02 absolute. If not, the ledger is
     not adding information beyond elapsed time. The strongest
     scientific gate.

Each test returns a structured dict {result: pass|fail|indeterminate,
metric: float|None, threshold: float, note: str|None} so the P1 rollup
can quote it verbatim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY, BaselineSpec
from coding_estimator.checkpoints.features.registry import GROUPS
from coding_estimator.eval.harness import predict_cell
from coding_estimator.eval.metrics import brier
from coding_estimator.splits.protocol import Fold, Split, loro

OutcomeT = Literal["pass", "fail", "indeterminate"]

HEADLINE_TARGET: str = "y_success_eventual"
TB_LIVE: str = "tb_live"

# O1
HIGH_PROGRESS_THRESHOLD: float = 0.8
O1_MEDIAN_BOUND: float = 0.7  # plan default; overridable

# O7
O7_BRIER_DELTA_GATE: float = 0.02

# O5
O5_LOSO_BRIER_DELTA_GATE: float = 0.02


@dataclass(frozen=True)
class FailureModeResult:
    test_id: str
    outcome: OutcomeT
    metric_name: str
    metric_value: float | None
    threshold: float
    note: str | None = None
    detail: dict[str, float | int | str | None] = field(default_factory=dict)


def _to_dict(result: FailureModeResult) -> dict:
    out = asdict(result)
    out["detail"] = {k: v for k, v in result.detail.items()}
    return out


# ---------- O1 progress-overconfidence ---------------------------------------


def evaluate_o1(
    *,
    predictions_df: pd.DataFrame,
    checkpoints_df: pd.DataFrame,
    final_success: dict[str, int],
    high_progress_threshold: float = HIGH_PROGRESS_THRESHOLD,
    median_bound: float = O1_MEDIAN_BOUND,
) -> FailureModeResult:
    """Slice: rows from runs where final_success == 0 AND coding_progress
    at the row >= `high_progress_threshold`. Pass iff median predicted
    P(success) on this slice is strictly less than `median_bound`."""
    if predictions_df.empty:
        return FailureModeResult(
            test_id="O1",
            outcome="indeterminate",
            metric_name="median_p_success_on_high_progress_failures",
            metric_value=None,
            threshold=median_bound,
            note="no predictions",
        )
    failed_runs = {rid for rid, fs in final_success.items() if fs == 0}
    if not failed_runs:
        return FailureModeResult(
            test_id="O1",
            outcome="indeterminate",
            metric_name="median_p_success_on_high_progress_failures",
            metric_value=None,
            threshold=median_bound,
            note="no failed runs in cohort — slice empty",
        )
    keep_cols = [c for c in ("run_id", "checkpoint_id") if c in checkpoints_df.columns]
    progress_df = checkpoints_df[keep_cols + ["coding_progress"]]
    joined = predictions_df.merge(progress_df, on=keep_cols, how="left")
    failed_rows = joined[joined["run_id"].astype(str).isin(failed_runs)]
    high_progress = failed_rows[
        failed_rows["coding_progress"].astype(float) >= high_progress_threshold
    ]
    n = len(high_progress)
    if n < 5:
        return FailureModeResult(
            test_id="O1",
            outcome="indeterminate",
            metric_name="median_p_success_on_high_progress_failures",
            metric_value=None,
            threshold=median_bound,
            note=f"only {n} rows in slice; require ≥ 5",
            detail={"n_rows": n, "n_failed_runs": len(failed_runs)},
        )
    median_p = float(np.median(high_progress["_p"].astype(float).to_numpy()))
    outcome: OutcomeT = "pass" if median_p < median_bound else "fail"
    return FailureModeResult(
        test_id="O1",
        outcome=outcome,
        metric_name="median_p_success_on_high_progress_failures",
        metric_value=median_p,
        threshold=median_bound,
        note=None,
        detail={
            "n_rows": n,
            "n_failed_runs": len(failed_runs),
            "high_progress_threshold": high_progress_threshold,
        },
    )


# ---------- O5 source-leakage -------------------------------------------------


def _source_task_numeric_columns() -> tuple[str, ...]:
    return tuple(
        f.column_name
        for f in GROUPS["source_task"]
        if f.dtype in ("int", "float", "bool")
    )


def _g4_plus_source_task_spec(available: tuple[str, ...]) -> BaselineSpec:
    """G4 features + every source_task numeric column that's actually
    present in `available` (the checkpoints frame columns). Source_task
    features that aren't built into the v0 checkpoint frame are
    silently dropped."""
    base = LEDGER_BASIC.feature_cols_for(())
    extras = tuple(c for c in _source_task_numeric_columns() if c in available)
    cols = base + extras

    def feature_cols_for(_sources: tuple[str, ...]) -> tuple[str, ...]:
        return cols

    return BaselineSpec(
        name="g4_plus_source_task", feature_cols_for=feature_cols_for
    )


def _loso_split(
    checkpoints_df: pd.DataFrame, test_source: str
) -> tuple[Split, tuple[str, ...]] | None:
    sources = sorted(checkpoints_df["source"].unique())
    if test_source not in sources or len(sources) < 2:
        return None
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
        return None
    fold = Fold(
        fold_id=f"loso::{test_source}",
        train_run_ids=train_runs,
        test_run_ids=test_runs,
    )
    train_sources = tuple(s for s in sources if s != test_source)
    return Split(scheme="loso", seed=0, folds=(fold,)), train_sources


def evaluate_o5(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    test_source: str = TB_LIVE,
    target: str = HEADLINE_TARGET,
    delta_gate: float = O5_LOSO_BRIER_DELTA_GATE,
) -> FailureModeResult:
    """LOSO -> test_source. Compare G4 (ledger-basic, excludes
    source_task) against G4 + source_task. If adding source_task moves
    Brier by less than `delta_gate`, source_task carries no
    discriminative information at LOSO time → no source-identity
    memorization. Pass iff |delta| < delta_gate."""
    split_info = _loso_split(checkpoints_df, test_source)
    if split_info is None:
        return FailureModeResult(
            test_id="O5",
            outcome="indeterminate",
            metric_name="loso_brier_delta_with_source_task",
            metric_value=None,
            threshold=delta_gate,
            note=f"LOSO target source `{test_source}` not present",
        )
    split, train_sources = split_info
    g4_preds = predict_cell(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        target=target,
        spec=LEDGER_BASIC,
        split=split,
        sources_in_train=train_sources,
    )
    plus_spec = _g4_plus_source_task_spec(tuple(checkpoints_df.columns))
    extras_added = tuple(
        c for c in plus_spec.feature_cols_for(()) if c not in LEDGER_BASIC.feature_cols_for(())
    )
    if not extras_added:
        return FailureModeResult(
            test_id="O5",
            outcome="indeterminate",
            metric_name="loso_brier_delta_with_source_task",
            metric_value=None,
            threshold=delta_gate,
            note=(
                "no source_task numeric columns present in checkpoints "
                "frame — comparison is identity, test is vacuous"
            ),
            detail={"source_task_columns_present": ""},
        )
    plus_preds = predict_cell(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        target=target,
        spec=plus_spec,
        split=split,
        sources_in_train=train_sources,
    )
    if g4_preds.empty or plus_preds.empty:
        return FailureModeResult(
            test_id="O5",
            outcome="indeterminate",
            metric_name="loso_brier_delta_with_source_task",
            metric_value=None,
            threshold=delta_gate,
            note="no LOSO predictions",
        )
    y_g4 = g4_preds["_y"].astype(int).to_numpy()
    p_g4 = g4_preds["_p"].astype(float).to_numpy()
    y_plus = plus_preds["_y"].astype(int).to_numpy()
    p_plus = plus_preds["_p"].astype(float).to_numpy()
    if len(np.unique(y_g4)) < 2 or len(np.unique(y_plus)) < 2:
        return FailureModeResult(
            test_id="O5",
            outcome="indeterminate",
            metric_name="loso_brier_delta_with_source_task",
            metric_value=None,
            threshold=delta_gate,
            note=f"single-class y on LOSO -> {test_source}",
            detail={"n_g4": int(len(y_g4)), "n_plus": int(len(y_plus))},
        )
    b_g4 = brier(y_g4, p_g4)
    b_plus = brier(y_plus, p_plus)
    delta = b_plus - b_g4
    outcome: OutcomeT = "pass" if abs(delta) < delta_gate else "fail"
    return FailureModeResult(
        test_id="O5",
        outcome=outcome,
        metric_name="loso_brier_delta_with_source_task",
        metric_value=float(delta),
        threshold=delta_gate,
        note=None,
        detail={
            "brier_g4": float(b_g4),
            "brier_g4_plus_source_task": float(b_plus),
            "test_source": test_source,
            "target": target,
            "source_task_columns_added": ",".join(extras_added),
        },
    )


# ---------- O7 timeout-bias ---------------------------------------------------


def evaluate_o7(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str = HEADLINE_TARGET,
    delta_gate: float = O7_BRIER_DELTA_GATE,
) -> list[FailureModeResult]:
    """Per source under LORO: compare G2 (time_only) against G4
    (ledger_basic) on `target`. PASS iff `Brier_G2 - Brier_G4 >=
    delta_gate` (ledger beats time-only by enough margin to claim it
    adds information beyond elapsed time). FAIL iff
    `Brier_G2 - Brier_G4 < delta_gate`. INDETERMINATE iff y is
    single-class on the source.

    O7 is reported per source (not collapsed) because the question
    "does the ledger help on THIS source" is the actual scientific
    claim; collapsing to a global mean averages signal across sources
    of very different sizes.
    """
    out: list[FailureModeResult] = []
    sources = sorted(checkpoints_df["source"].unique())
    for source in sources:
        sub = checkpoints_df[checkpoints_df["source"] == source]
        if sub["run_id"].nunique() < 2:
            out.append(
                FailureModeResult(
                    test_id="O7",
                    outcome="indeterminate",
                    metric_name="brier_g2_minus_brier_g4",
                    metric_value=None,
                    threshold=delta_gate,
                    note=f"{source}: < 2 runs",
                    detail={"source": source},
                )
            )
            continue
        split = loro(sub)
        g2_preds = predict_cell(
            checkpoints_df=sub,
            labels_df=labels_df,
            target=target,
            spec=TIME_ONLY,
            split=split,
            sources_in_train=(source,),
        )
        g4_preds = predict_cell(
            checkpoints_df=sub,
            labels_df=labels_df,
            target=target,
            spec=LEDGER_BASIC,
            split=split,
            sources_in_train=(source,),
        )
        if g2_preds.empty or g4_preds.empty:
            out.append(
                FailureModeResult(
                    test_id="O7",
                    outcome="indeterminate",
                    metric_name="brier_g2_minus_brier_g4",
                    metric_value=None,
                    threshold=delta_gate,
                    note=f"{source}: no LORO predictions",
                    detail={"source": source},
                )
            )
            continue
        y2 = g2_preds["_y"].astype(int).to_numpy()
        p2 = g2_preds["_p"].astype(float).to_numpy()
        y4 = g4_preds["_y"].astype(int).to_numpy()
        p4 = g4_preds["_p"].astype(float).to_numpy()
        if len(np.unique(y2)) < 2 or len(np.unique(y4)) < 2:
            out.append(
                FailureModeResult(
                    test_id="O7",
                    outcome="indeterminate",
                    metric_name="brier_g2_minus_brier_g4",
                    metric_value=None,
                    threshold=delta_gate,
                    note=f"{source}: single-class y on `{target}`",
                    detail={"source": source, "target": target},
                )
            )
            continue
        b2 = brier(y2, p2)
        b4 = brier(y4, p4)
        delta = b2 - b4
        outcome: OutcomeT = "pass" if delta >= delta_gate else "fail"
        out.append(
            FailureModeResult(
                test_id="O7",
                outcome=outcome,
                metric_name="brier_g2_minus_brier_g4",
                metric_value=float(delta),
                threshold=delta_gate,
                note=None,
                detail={
                    "source": source,
                    "target": target,
                    "brier_g2": float(b2),
                    "brier_g4": float(b4),
                    "n_checkpoints": int(len(y2)),
                },
            )
        )
    return out


# ---------- Renderer ---------------------------------------------------------


def _fmt(v: float | int | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_failure_mode_report(
    *,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
    summary: str | None = None,
) -> str:
    lines = [
        "# Failure-mode tests (Workstream O)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
    ]
    if summary:
        lines.extend([summary, ""])

    def _row(r: FailureModeResult) -> str:
        return (
            f"| {r.test_id} | **{r.outcome}** | {r.metric_name} | "
            f"{_fmt(r.metric_value)} | {_fmt(r.threshold)} | {r.note or ''} |"
        )

    lines.extend(
        [
            "## Headline outcomes",
            "",
            "| test | outcome | metric | value | threshold | note |",
            "|---|---|---|---:|---:|---|",
            _row(o1),
            _row(o5),
        ]
    )
    for r in o7:
        lines.append(_row(r))
    lines.append("")

    lines.append("## O1 — progress-overconfidence")
    lines.append("")
    lines.append(
        f"Slice: rows where `final_success == 0` AND "
        f"`coding_progress >= {HIGH_PROGRESS_THRESHOLD}`. Gate: median "
        f"`P(success)` on slice must be < {O1_MEDIAN_BOUND}."
    )
    lines.append("")
    if o1.detail:
        for k, v in sorted(o1.detail.items()):
            lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## O5 — source-leakage")
    lines.append("")
    lines.append(
        f"LOSO -> `{TB_LIVE}` on `{HEADLINE_TARGET}`. Compare G4 vs "
        f"G4+source_task; `|Brier_plus - Brier_g4| < "
        f"{O5_LOSO_BRIER_DELTA_GATE:.2f}` ⇒ pass."
    )
    lines.append("")
    if o5.detail:
        for k, v in sorted(o5.detail.items()):
            lines.append(f"- `{k}`: {_fmt(v) if isinstance(v, float) else v}")
    lines.append("")

    lines.append("## O7 — timeout-bias")
    lines.append("")
    lines.append(
        f"Per source under LORO on `{HEADLINE_TARGET}`. Pass iff "
        f"`Brier_G2 - Brier_G4 >= {O7_BRIER_DELTA_GATE:.2f}` (the ledger "
        f"adds information beyond elapsed time). Indeterminate when y "
        f"is single-class on the source."
    )
    lines.append("")
    lines.extend(
        [
            "| source | outcome | Brier G2 | Brier G4 | Δ (G2 - G4) | n |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for r in o7:
        d = r.detail
        lines.append(
            f"| {d.get('source', '?')} | {r.outcome} | {_fmt(d.get('brier_g2'))} "
            f"| {_fmt(d.get('brier_g4'))} | {_fmt(r.metric_value)} | "
            f"{d.get('n_checkpoints', '?')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_failure_mode_report(
    path: Path,
    *,
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
    summary: str | None = None,
) -> Path:
    md = render_failure_mode_report(o1=o1, o5=o5, o7=o7, summary=summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


def all_results_json(
    o1: FailureModeResult,
    o5: FailureModeResult,
    o7: list[FailureModeResult],
) -> dict:
    return {
        "O1": _to_dict(o1),
        "O5": _to_dict(o5),
        "O7": [_to_dict(r) for r in o7],
    }


__all__ = [
    "HIGH_PROGRESS_THRESHOLD",
    "O1_MEDIAN_BOUND",
    "O5_LOSO_BRIER_DELTA_GATE",
    "O7_BRIER_DELTA_GATE",
    "FailureModeResult",
    "evaluate_o1",
    "evaluate_o5",
    "evaluate_o7",
    "render_failure_mode_report",
    "write_failure_mode_report",
    "all_results_json",
]
