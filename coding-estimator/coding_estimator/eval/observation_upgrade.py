"""Observation-upgrade evaluation on tb_live_v2 exact-task holdout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coding_estimator.baselines import BaselineSpec, LEDGER_BASIC, TIME_ONLY
from coding_estimator.baselines.observation_basic import OBSERVATION_BASIC
from coding_estimator.checkpoints.dynamics import G5_FEATURES, attach_g5_features
from coding_estimator.eval.harness import EvalCell, evaluate_cell, predict_cell
from coding_estimator.eval.metrics import brier
from coding_estimator.eval.tb_live_v2 import (
    SCHEME_LABELS,
    TBLiveV2Profile,
    build_tb_live_v2_profile,
    build_tb_live_v2_splits,
    exact_task_group_map,
)
from coding_estimator.profile.budget import compute_budget
from coding_estimator.splits.protocol import Split

TB_LIVE_V2 = "tb_live_v2"
EVAL_TARGETS: tuple[str, ...] = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
)
EVAL_MODELS = (TIME_ONLY, LEDGER_BASIC, OBSERVATION_BASIC)


def _g5_cols(_sources: tuple[str, ...]) -> tuple[str, ...]:
    return G5_FEATURES


def _g4_plus_g5_cols(_sources: tuple[str, ...]) -> tuple[str, ...]:
    return LEDGER_BASIC.feature_cols_for(_sources) + G5_FEATURES


G5_DIAGNOSTIC = BaselineSpec(name="g5_dynamics", feature_cols_for=_g5_cols)
G4_PLUS_G5_DIAGNOSTIC = BaselineSpec(name="g4_plus_g5", feature_cols_for=_g4_plus_g5_cols)


@dataclass(frozen=True)
class SliceSummary:
    slice_name: str
    model: str
    n_rows: int
    mean_label: float
    mean_prediction: float
    brier: float


def _wide_targets(labels_df: pd.DataFrame) -> pd.DataFrame:
    sub = labels_df[
        (labels_df["source"] == TB_LIVE_V2)
        & (labels_df["target_name"].isin(EVAL_TARGETS))
        & (~labels_df["is_masked"].astype(bool))
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot(
        index=["run_id", "source", "checkpoint_id"],
        columns="target_name",
        values="label_value",
    ).reset_index()


def _budget_lookup(labels_df: pd.DataFrame, exact_groups: dict[str, str]) -> dict[tuple[str, str], object]:
    wide = _wide_targets(labels_df)
    if wide.empty:
        return {}
    wide = wide.assign(task_family=wide["run_id"].map(exact_groups))
    return {
        (cell.target, cell.split_scheme): cell
        for cell in compute_budget(
            wide,
            targets=EVAL_TARGETS,
            sources=(TB_LIVE_V2,),
            schemes=("ltfo", "loro", "holdout"),
        )
    }


def evaluate_observation_upgrade(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    bootstrap_b: int = 1000,
    bootstrap_seed: int = 0,
) -> list[EvalCell]:
    sub = checkpoints_df[checkpoints_df["source"] == TB_LIVE_V2].copy()
    if sub.empty:
        raise ValueError("no tb_live_v2 checkpoints present")
    exact_groups = exact_task_group_map(
        checkpoints_df=sub,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    budget = _budget_lookup(labels_df, exact_groups)
    splits = build_tb_live_v2_splits(
        checkpoints_df=sub,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )

    cells: list[EvalCell] = []
    for scheme in ("ltfo", "loro", "holdout"):
        split = splits[scheme]
        source_slice = SCHEME_LABELS[scheme]
        for target in EVAL_TARGETS:
            budget_cell = budget.get((target, scheme))
            feasible = budget_cell is not None and budget_cell.feasible
            note = None if feasible else ("no budget row" if budget_cell is None else budget_cell.reason)
            for spec in EVAL_MODELS:
                if feasible:
                    cells.append(
                        evaluate_cell(
                            checkpoints_df=sub,
                            labels_df=labels_df,
                            target=target,
                            spec=spec,
                            split=split,
                            source_slice=source_slice,
                            sources_in_train=(TB_LIVE_V2,),
                            feasible=True,
                            bootstrap_b=bootstrap_b,
                            bootstrap_seed=bootstrap_seed,
                        )
                    )
                else:
                    cells.append(
                        EvalCell(
                            target=target,
                            model=spec.name,
                            scheme=split.scheme,
                            source_slice=source_slice,
                            feasible=False,
                            n_runs_train=None,
                            n_runs_test=None,
                            n_checkpoints_test=None,
                            positive_rate_data=None,
                            predicted_positive_rate=None,
                            auroc=None,
                            brier=None,
                            log_loss=None,
                            ece=None,
                            brier_ci_low=None,
                            brier_ci_high=None,
                            note=note or "insufficient data",
                        )
                    )
    return cells


def _exact_task_predictions(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    spec,
    target: str,
) -> pd.DataFrame:
    sub = checkpoints_df[checkpoints_df["source"] == TB_LIVE_V2].copy()
    split = build_tb_live_v2_splits(
        checkpoints_df=sub,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )["ltfo"]
    preds = predict_cell(
        checkpoints_df=sub,
        labels_df=labels_df,
        target=target,
        spec=spec,
        split=split,
        sources_in_train=(TB_LIVE_V2,),
    )
    if preds.empty:
        return preds
    keep = sub[
        [
            "run_id",
            "checkpoint_id",
            "coding_progress",
            "num_progress_drops_so_far",
            "task_id",
            "arm",
            "model_name",
        ]
    ].drop_duplicates(["run_id", "checkpoint_id"])
    return preds.merge(keep, on=["run_id", "checkpoint_id"], how="left")


def build_success_slice_summaries(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> list[SliceSummary]:
    out: list[SliceSummary] = []
    for spec in EVAL_MODELS:
        preds = _exact_task_predictions(
            checkpoints_df=checkpoints_df,
            labels_df=labels_df,
            manifest_df=manifest_df,
            spec=spec,
            target="y_success_eventual",
        )
        if preds.empty:
            continue
        slices = {
            "high_progress_failure": preds[(preds["_y"] == 0) & (preds["coding_progress"] >= 0.75)],
            "low_progress_success": preds[(preds["_y"] == 1) & (preds["coding_progress"] <= 0.25)],
            "recovery_after_drop_success": preds[(preds["_y"] == 1) & (preds["num_progress_drops_so_far"] > 0)],
        }
        for slice_name, frame in slices.items():
            if frame.empty:
                continue
            out.append(
                SliceSummary(
                    slice_name=slice_name,
                    model=spec.name,
                    n_rows=int(len(frame)),
                    mean_label=float(frame["_y"].mean()),
                    mean_prediction=float(frame["_p"].mean()),
                    brier=float(brier(frame["_y"].astype(int).to_numpy(), frame["_p"].to_numpy())),
                )
            )
    return out


def build_success_diagnostic_cells(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    bootstrap_b: int = 1000,
    bootstrap_seed: int = 0,
) -> list[EvalCell]:
    sub = checkpoints_df[checkpoints_df["source"] == TB_LIVE_V2].copy()
    sub = attach_g5_features(sub)
    split = build_tb_live_v2_splits(
        checkpoints_df=sub,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )["ltfo"]
    out: list[EvalCell] = []
    for spec in (TIME_ONLY, LEDGER_BASIC, G5_DIAGNOSTIC, G4_PLUS_G5_DIAGNOSTIC, OBSERVATION_BASIC):
        out.append(
            evaluate_cell(
                checkpoints_df=sub,
                labels_df=labels_df,
                target="y_success_eventual",
                spec=spec,
                split=split,
                source_slice="exact-task-holdout",
                sources_in_train=(TB_LIVE_V2,),
                feasible=True,
                bootstrap_b=bootstrap_b,
                bootstrap_seed=bootstrap_seed,
            )
        )
    return out


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_observation_upgrade_report(
    *,
    cells: list[EvalCell],
    profile: TBLiveV2Profile,
    slices: list[SliceSummary],
    diagnostics: list[EvalCell] | None = None,
) -> str:
    frame = pd.DataFrame([cell.__dict__ for cell in cells])
    exact = frame[frame["source_slice"] == "exact-task-holdout"].copy()

    lines = [
        "# Observation Upgrade Evaluation",
        "",
        "Exact-task holdout on `tb_live_v2` comparing `time_only`, `ledger_basic`, and `observation_basic`.",
        "",
        "Arm and `model_name` are descriptive metadata only. They are not part of the headline estimator feature set.",
        "",
        "## Corpus",
        "",
        f"- runs: {profile.n_runs}",
        f"- successes: {profile.n_success}",
        f"- failures: {profile.n_fail}",
        f"- exact tasks: {profile.n_exact_tasks}",
        "",
        "## Per-Arm Outcome Breakdown",
        "",
        "| arm | model | runs | successes | failures | success rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in profile.arm_rows:
        lines.append(
            f"| {row.name} | {row.model_name or ''} | {row.n_runs} | {row.n_success} | {row.n_fail} | {row.success_rate:.3f} |"
        )

    lines.extend(
        [
            "",
        "## Exact-Task Holdout",
        "",
        "| target | model | feasible | AUROC | Brier | 95% CI | ECE | note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in exact.sort_values(["target", "model"]).iterrows():
        ci = "n/a"
        if pd.notna(row["brier_ci_low"]) and pd.notna(row["brier_ci_high"]):
            ci = f"[{row['brier_ci_low']:.3f}, {row['brier_ci_high']:.3f}]"
        lines.append(
            f"| {row['target']} | {row['model']} | {bool(row['feasible'])} | "
            f"{_fmt_metric(row['auroc'])} | {_fmt_metric(row['brier'])} | {ci} | "
            f"{_fmt_metric(row['ece'])} | {row['note'] or ''} |"
        )

    success_rows = exact[exact["target"] == "y_success_eventual"].copy()
    if not success_rows.empty:
        g2_brier = float(success_rows.loc[success_rows["model"] == "time_only", "brier"].iloc[0])
        g4_row = success_rows.loc[success_rows["model"] == "ledger_basic"].iloc[0]
        obs_row = success_rows.loc[success_rows["model"] == "observation_basic"].iloc[0]
        lines.extend(
            [
                "",
                "## Completion-Risk Retest",
                "",
                "This is the X2-style re-test after adding transcript/verifier-derived observation features.",
                "",
                "| model | AUROC | Brier | Δ Brier vs G2 | ECE |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in success_rows.sort_values("model").iterrows():
            lines.append(
                f"| {row['model']} | {_fmt_metric(row['auroc'])} | {_fmt_metric(row['brier'])} | "
                f"{(g2_brier - float(row['brier'])):+.3f} | {_fmt_metric(row['ece'])} |"
            )

    if diagnostics:
        diag_df = pd.DataFrame([cell.__dict__ for cell in diagnostics]).sort_values("model")
        lines.extend(
            [
                "",
                "## Supplementary Success Diagnostics",
                "",
                "Exact-task holdout on `y_success_eventual` including the existing G5 dynamics probes.",
                "",
                "| model | AUROC | Brier | ECE | note |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for _, row in diag_df.iterrows():
            lines.append(
                f"| {row['model']} | {_fmt_metric(row['auroc'])} | {_fmt_metric(row['brier'])} | {_fmt_metric(row['ece'])} | {row['note'] or ''} |"
            )

    if slices:
        slice_df = pd.DataFrame([slice_.__dict__ for slice_ in slices])
        lines.extend(
            [
                "",
                "## Success Slices",
                "",
                "| slice | model | n | mean label | mean P(success) | Brier |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in slice_df.sort_values(["slice_name", "model"]).iterrows():
            lines.append(
                f"| {row['slice_name']} | {row['model']} | {int(row['n_rows'])} | "
                f"{row['mean_label']:.3f} | {row['mean_prediction']:.3f} | {row['brier']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `observation_basic` tests the instrumentation bottleneck directly by adding structured transcript-visible validation, error, and oracle-read signals on top of `ledger_basic`.",
            f"- Terminal success remains a bounded / negative headline on `tb_live_v2`: `observation_basic` does not beat `ledger_basic` on Brier ({obs_row['brier']:.3f} vs {g4_row['brier']:.3f}), and its AUROC is {_fmt_metric(obs_row['auroc'])} vs {_fmt_metric(g4_row['auroc'])}.",
            "- `y_validation_new_work_h5` may remain infeasible on `tb_live_v2`; that is a substrate fact, not a modeling failure.",
            "- Verifier terminal events are emitted after the last transcript step, so they do not leak into earlier prefixes.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_observation_upgrade_report(
    out_path: Path,
    *,
    cells: list[EvalCell],
    profile: TBLiveV2Profile,
    slices: list[SliceSummary],
    diagnostics: list[EvalCell] | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_observation_upgrade_report(
            cells=cells,
            profile=profile,
            slices=slices,
            diagnostics=diagnostics,
        ),
        encoding="utf-8",
    )
    return out_path


def build_profile(manifest_df: pd.DataFrame) -> TBLiveV2Profile:
    return build_tb_live_v2_profile(manifest_df=manifest_df)
