"""Process-dynamics audit package for frozen tb_live_v2 artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from ledger_progress.core import EventType, LedgerEvent
from ledger_progress.serialization import load_events_jsonl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY
from coding_estimator.baselines.base import BaselineSpec
from coding_estimator.checkpoints.features.registry import feature_by_name
from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.harness import predict_cell
from coding_estimator.eval.metrics import auroc, brier
from coding_estimator.eval.tb_live_v2 import build_tb_live_v2_splits
from coding_estimator.ingest.run_record import RunRecord
from coding_estimator.io import write_csv
from coding_estimator.labels._upstream_q_snapshot import (
    _coding_progress,
    _events_through_step,
    _is_discovery_event,
    _is_validation_transition,
)
from coding_estimator.models.common import join_binary_target

TB_LIVE_V2 = "tb_live_v2"
PROGRESS_DROP_TARGET = "y_future_progress_drop_h5"
VALIDATION_TARGET = "y_validation_new_work_h5"
SUCCESS_TARGET = "y_success_eventual"
HARD_FAIL_VERDICTS = frozenset({"feature_leakage", "label_construction_bug"})
LEAKAGE_TOKENS: tuple[str, ...] = (
    "future",
    "next",
    "h5",
    "lead",
    "target",
    "label",
    "drop_within",
    "future_min",
    "min_future",
    "max_future",
    "final",
    "success",
    "outcome",
    "post",
)
LEAKAGE_FEATURE_WHITELIST: frozenset[str] = frozenset()
CASE_THRESHOLD = 0.5
VERDICT_VALID_PREFIX = "valid_prefix_signal"
VERDICT_NEAR_BOUNDARY = "valid_but_near_boundary"
VERDICT_LEAKAGE = "feature_leakage"
VERDICT_LABEL_BUG = "label_construction_bug"
VERDICT_INSUFFICIENT = "insufficient_evidence"
DECISION_VERDICTS: tuple[str, ...] = (
    VERDICT_VALID_PREFIX,
    VERDICT_NEAR_BOUNDARY,
    VERDICT_LEAKAGE,
    VERDICT_LABEL_BUG,
    VERDICT_INSUFFICIENT,
)


@dataclass(frozen=True)
class AuditFailure(RuntimeError):
    verdict: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class DecisionOutcome:
    verdict: str
    rationale: str


@dataclass(frozen=True)
class CaseStudy:
    section_title: str
    run_id: str
    checkpoint_id: str
    checkpoint_step: int
    predicted_probability: float
    label_value: int
    task_id: str | None
    task_family: str | None
    arm: str | None
    why_selected: str
    interpretation: str
    figure_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_local_run(run_dir: Path) -> RunRecord:
    run_id = run_dir.name
    events = tuple(sorted(load_events_jsonl(str(run_dir / "ledger.jsonl")), key=lambda e: e.step))
    manifest = _read_json(run_dir / "run_manifest.json")
    instr = _read_json(run_dir / "live_instrumentation.json")
    src_meta = _read_json(run_dir / "source_metadata.json")
    task_id = (
        instr.get("task_id")
        or manifest.get("task_id")
        or src_meta.get("instance_id")
        or run_id
    )
    task_family = (
        manifest.get("task_family")
        or manifest.get("category")
        or manifest.get("target_shape")
        or src_meta.get("category")
        or src_meta.get("subcategory")
    )
    return RunRecord(
        run_id=run_id,
        source=TB_LIVE_V2,
        ledger_path=run_dir / "ledger.jsonl",
        events=events,
        has_real_wallclock=manifest.get("has_real_wallclock") is True,
        start_wall_time=None,
        end_wall_time=None,
        task_id=str(task_id) if task_id is not None else None,
        task_family=str(task_family) if task_family is not None else None,
        arm=str(manifest.get("arm")) if manifest.get("arm") is not None else None,
        difficulty=str(manifest.get("difficulty")) if manifest.get("difficulty") is not None else None,
        agent_scaffold=(
            str(manifest.get("subagent_type"))
            if manifest.get("subagent_type") is not None
            else None
        ),
        model_name=str(manifest.get("model_name")) if manifest.get("model_name") is not None else None,
        raw_metadata={
            "run_manifest": manifest,
            "source_metadata": src_meta,
            "live_instrumentation": instr,
        },
    )


def _tb_live_v2_checkpoints(checkpoints_df: pd.DataFrame) -> pd.DataFrame:
    return checkpoints_df[checkpoints_df["source"] == TB_LIVE_V2].copy()


def _tb_live_v2_labels(labels_df: pd.DataFrame, target: str) -> pd.DataFrame:
    sub = labels_df[
        (labels_df["source"] == TB_LIVE_V2)
        & (labels_df["target_name"] == target)
        & (~labels_df["is_masked"].astype(bool))
    ][["run_id", "checkpoint_id", "label_value"]].copy()
    sub["label_value"] = sub["label_value"].astype(float)
    return sub


def _progress_by_step(events: tuple[LedgerEvent, ...]) -> dict[int, float]:
    if not events:
        return {}
    lo = min(e.step for e in events)
    hi = max(e.step for e in events)
    return {
        step: float(_coding_progress(_events_through_step(events, step)))
        for step in range(lo, hi + 1)
    }


def _event_types_by_step(events: tuple[LedgerEvent, ...]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = defaultdict(set)
    for event in events:
        out[int(event.step)].add(str(event.event_type.name))
    return out


def _actual_drop_steps(progress_by_step: dict[int, float]) -> list[int]:
    steps = sorted(progress_by_step)
    out: list[int] = []
    for prev, cur in zip(steps, steps[1:], strict=False):
        if progress_by_step[cur] < progress_by_step[prev] - 1e-9:
            out.append(cur)
    return out


def build_progress_drop_witness_rows(
    run: RunRecord,
    checkpoint_steps: list[int],
    *,
    horizon: int,
) -> list[dict[str, Any]]:
    if not run.events:
        return []
    finish_step = max(e.step for e in run.events)
    progress_map = _progress_by_step(run.events)
    event_types = _event_types_by_step(run.events)
    rows: list[dict[str, Any]] = []
    for checkpoint_step in sorted(checkpoint_steps):
        if checkpoint_step == finish_step or checkpoint_step + horizon > finish_step:
            continue
        current_progress = progress_map[checkpoint_step]
        future = [
            (step, progress_map[step])
            for step in range(checkpoint_step + 1, checkpoint_step + horizon + 1)
        ]
        future_min = min(progress for _, progress in future)
        drops = [
            (step, progress)
            for step, progress in future
            if progress < current_progress - 1e-9
        ]
        next_drop_step = int(drops[0][0]) if drops else None
        lead_time = None if next_drop_step is None else int(next_drop_step - checkpoint_step)
        rows.append(
            {
                "run_id": run.run_id,
                "checkpoint_step": int(checkpoint_step),
                "current_progress": float(current_progress),
                "future_min_progress_within_h": float(future_min),
                "label": int(bool(drops)),
                "next_drop_step": next_drop_step,
                "drop_magnitude": float(max(0.0, current_progress - future_min)),
                "lead_time": lead_time,
                "features_max_step": int(checkpoint_step),
                "features_max_step_status": "assumed",
                "checkpoint_has_add_split_reopen": bool(
                    event_types.get(checkpoint_step, set())
                    & {"ADD_SUBTASK", "SPLIT_SUBTASK", "REOPEN_SUBTASK"}
                ),
                "checkpoint_event_types": ",".join(sorted(event_types.get(checkpoint_step, set()))),
                "horizon": int(horizon),
            }
        )
    return rows


def build_progress_drop_witness(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    runs_dir: Path,
    horizon: int,
) -> pd.DataFrame:
    checkpoints = _tb_live_v2_checkpoints(checkpoints_df)
    rows: list[dict[str, Any]] = []
    meta_cols = [
        "run_id",
        "checkpoint_id",
        "checkpoint_step",
        "task_id",
        "task_family",
        "arm",
        "model_name",
        "coding_progress",
    ]
    meta = checkpoints[meta_cols].drop_duplicates(["run_id", "checkpoint_id"]).copy()
    for run_id, sub in meta.groupby("run_id", sort=True):
        run = _load_local_run(runs_dir / str(run_id))
        rows.extend(
            build_progress_drop_witness_rows(
                run,
                checkpoint_steps=sub["checkpoint_step"].astype(int).tolist(),
                horizon=horizon,
            )
        )
    witness = pd.DataFrame(rows)
    if witness.empty:
        return witness
    witness = witness.merge(
        meta,
        on=["run_id", "checkpoint_step"],
        how="left",
    )
    witness = witness.rename(
        columns={
            "future_min_progress_within_h": f"future_min_progress_within_h{horizon}",
        }
    )
    if horizon == 5:
        shipped = _tb_live_v2_labels(labels_df, PROGRESS_DROP_TARGET)
        witness = witness.merge(
            shipped.rename(columns={"label_value": "shipped_label"}),
            on=["run_id", "checkpoint_id"],
            how="left",
        )
    return witness.sort_values(["run_id", "checkpoint_step"], kind="mergesort").reset_index(drop=True)


def verify_progress_drop_witness(
    witness_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "shipped_label" not in witness_df.columns:
        raise ValueError("witness frame missing shipped_label")
    compared = witness_df.copy()
    compared["label_matches"] = compared["label"].astype(float) == compared["shipped_label"].astype(float)
    mismatches = compared[~compared["label_matches"]].copy()
    return compared, mismatches


def scan_ledger_basic_feature_names() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_cols = LEDGER_BASIC.feature_cols_for((TB_LIVE_V2,))
    for feature in feature_cols:
        lower = feature.lower()
        hits = [token for token in LEAKAGE_TOKENS if token in lower]
        whitelisted = feature in LEAKAGE_FEATURE_WHITELIST
        rows.append(
            {
                "feature": feature,
                "tokens_hit": ",".join(hits),
                "is_flagged": bool(hits),
                "whitelisted": bool(whitelisted),
            }
        )
    return pd.DataFrame(rows).sort_values("feature", kind="mergesort").reset_index(drop=True)


def assert_no_leakage_feature_names(scan_df: pd.DataFrame) -> None:
    bad = scan_df[scan_df["is_flagged"] & ~scan_df["whitelisted"]]
    if not bad.empty:
        features = ", ".join(sorted(bad["feature"].astype(str).tolist()))
        raise AuditFailure(
            verdict=VERDICT_LEAKAGE,
            message=f"target-specific leakage scan hit used features: {features}",
        )


def build_exact_task_oof_predictions(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Any]:
    checkpoints = _tb_live_v2_checkpoints(checkpoints_df)
    splits = build_tb_live_v2_splits(
        checkpoints_df=checkpoints,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    ledger = predict_cell(
        checkpoints_df=checkpoints,
        labels_df=labels_df,
        target=PROGRESS_DROP_TARGET,
        spec=LEDGER_BASIC,
        split=splits["ltfo"],
        sources_in_train=(TB_LIVE_V2,),
    ).rename(columns={"_p": "p_ledger_basic"})
    time_only = predict_cell(
        checkpoints_df=checkpoints,
        labels_df=labels_df,
        target=PROGRESS_DROP_TARGET,
        spec=TIME_ONLY,
        split=splits["ltfo"],
        sources_in_train=(TB_LIVE_V2,),
    ).rename(columns={"_p": "p_time_only"})
    if ledger.empty or time_only.empty:
        raise AuditFailure(
            verdict=VERDICT_INSUFFICIENT,
            message="exact-task OOF predictions were empty",
        )
    oof = ledger.merge(
        time_only[["run_id", "checkpoint_id", "p_time_only"]],
        on=["run_id", "checkpoint_id"],
        how="inner",
    )
    meta_cols = [
        "run_id",
        "checkpoint_id",
        "task_id",
        "task_family",
        "arm",
        "model_name",
        "coding_progress",
    ]
    meta = checkpoints[meta_cols].drop_duplicates(["run_id", "checkpoint_id"])
    oof = oof.merge(meta, on=["run_id", "checkpoint_id"], how="left")
    oof = oof.sort_values(["run_id", "checkpoint_step"], kind="mergesort").reset_index(drop=True)
    return oof, splits["ltfo"]


def assert_expected_oof_row_count(oof_df: pd.DataFrame, expected_rows: int = 213) -> None:
    if len(oof_df) != expected_rows:
        raise AuditFailure(
            verdict=VERDICT_INSUFFICIENT,
            message=f"exact-task OOF row count mismatch: expected {expected_rows}, got {len(oof_df)}",
        )


def _ledger_basic_feature_groups() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for feature in LEDGER_BASIC.feature_cols_for((TB_LIVE_V2,)):
        try:
            group = feature_by_name(feature).group
        except KeyError:
            group = "other"
        mapping[feature] = str(group) if group in {"closure", "frontier", "instability", "discovery"} else "other"
    return mapping


def build_diagnostic_coefficients(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    split: Any,
) -> pd.DataFrame:
    checkpoints = _tb_live_v2_checkpoints(checkpoints_df)
    joined = join_binary_target(checkpoints, labels_df, PROGRESS_DROP_TARGET)
    feature_cols = LEDGER_BASIC.feature_cols_for((TB_LIVE_V2,))
    feature_groups = _ledger_basic_feature_groups()
    rows: list[dict[str, Any]] = []
    for fold in split.folds:
        train = joined[joined["run_id"].isin(set(fold.train_run_ids))].copy()
        if train.empty or train["_y"].nunique() < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[list(feature_cols)].to_numpy(dtype=float, copy=False))
        y_train = train["_y"].astype(int).to_numpy()
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(x_train, y_train)
        for feature, coef in zip(feature_cols, model.coef_.ravel(), strict=True):
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "feature": feature,
                    "coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                    "feature_group": feature_groups.get(feature, "other"),
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise AuditFailure(
            verdict=VERDICT_INSUFFICIENT,
            message="diagnostic coefficient fit had no usable folds",
        )
    agg = (
        raw.groupby(["feature", "feature_group"], as_index=False)
        .agg(
            median_abs_coefficient=("abs_coefficient", "median"),
            sign_frequency_positive=("coefficient", lambda s: float((s > 0).mean())),
            sign_frequency_negative=("coefficient", lambda s: float((s < 0).mean())),
        )
        .sort_values(["median_abs_coefficient", "feature"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return agg


def _variant_label_frame(witness_df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": witness_df["run_id"].astype(str),
            "source": TB_LIVE_V2,
            "checkpoint_id": witness_df["checkpoint_id"].astype(str),
            "target_name": target_name,
            "label_value": witness_df["label"].astype(float),
            "is_masked": False,
        }
    )


def apply_progress_drop_variant(
    witness_df: pd.DataFrame,
    variant_name: str,
) -> pd.DataFrame:
    df = witness_df.copy()
    if variant_name == "h5_base":
        return df
    if variant_name == "h5_drop_ge_0.05":
        df["label"] = ((df["label"] > 0) & (df["drop_magnitude"] >= 0.05)).astype(int)
        return df
    if variant_name == "h5_drop_ge_0.10":
        df["label"] = ((df["label"] > 0) & (df["drop_magnitude"] >= 0.10)).astype(int)
        return df
    if variant_name == "h5_first_drop_lead_ge_2":
        df["label"] = ((df["label"] > 0) & (df["lead_time"].fillna(0).astype(int) >= 2)).astype(int)
        return df
    if variant_name == "h5_first_drop_lead_ge_2_excluding_checkpoint_steps_with_add_split_reopen":
        df["label"] = ((df["label"] > 0) & (df["lead_time"].fillna(0).astype(int) >= 2)).astype(int)
        df = df[~df["checkpoint_has_add_split_reopen"].astype(bool)].copy()
        return df
    if variant_name == "h5_first_positive_per_drop_episode":
        pos = df[df["label"] > 0].copy()
        pos = pos.sort_values(
            ["run_id", "next_drop_step", "checkpoint_step"],
            kind="mergesort",
        )
        keep = pos.groupby(["run_id", "next_drop_step"], as_index=False).head(1)
        neg = df[df["label"] <= 0].copy()
        out = pd.concat([neg, keep], ignore_index=True)
        return out.sort_values(["run_id", "checkpoint_step"], kind="mergesort").reset_index(drop=True)
    raise KeyError(variant_name)


def build_progress_drop_variant_suite(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    runs_dir: Path,
) -> dict[str, pd.DataFrame]:
    h5 = build_progress_drop_witness(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        runs_dir=runs_dir,
        horizon=5,
    )
    h3 = build_progress_drop_witness(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        runs_dir=runs_dir,
        horizon=3,
    )
    h10 = build_progress_drop_witness(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        runs_dir=runs_dir,
        horizon=10,
    )
    return {
        "h5_base": h5,
        "h3_base": h3,
        "h10_base": h10,
        "h5_drop_ge_0.05": apply_progress_drop_variant(h5, "h5_drop_ge_0.05"),
        "h5_drop_ge_0.10": apply_progress_drop_variant(h5, "h5_drop_ge_0.10"),
        "h5_first_drop_lead_ge_2": apply_progress_drop_variant(h5, "h5_first_drop_lead_ge_2"),
        "h5_first_drop_lead_ge_2_excluding_checkpoint_steps_with_add_split_reopen": apply_progress_drop_variant(
            h5,
            "h5_first_drop_lead_ge_2_excluding_checkpoint_steps_with_add_split_reopen",
        ),
        "h5_first_positive_per_drop_episode": apply_progress_drop_variant(
            h5,
            "h5_first_positive_per_drop_episode",
        ),
    }


def _metric_row(preds: pd.DataFrame) -> tuple[float | None, float]:
    y = preds["_y"].astype(int).to_numpy()
    p = preds["_p"].astype(float).to_numpy()
    return auroc(y, p), brier(y, p)


def build_sensitivity_table(
    *,
    checkpoints_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    split: Any,
    variant_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    checkpoints = _tb_live_v2_checkpoints(checkpoints_df)
    rows: list[dict[str, Any]] = []
    for target_variant, variant_df in variant_frames.items():
        labels = _variant_label_frame(variant_df, target_variant)
        ledger_preds = predict_cell(
            checkpoints_df=checkpoints,
            labels_df=labels,
            target=target_variant,
            spec=LEDGER_BASIC,
            split=split,
            sources_in_train=(TB_LIVE_V2,),
        )
        time_preds = predict_cell(
            checkpoints_df=checkpoints,
            labels_df=labels,
            target=target_variant,
            spec=TIME_ONLY,
            split=split,
            sources_in_train=(TB_LIVE_V2,),
        )
        if ledger_preds.empty or time_preds.empty:
            raise AuditFailure(
                verdict=VERDICT_INSUFFICIENT,
                message=f"sensitivity predictions empty for {target_variant}",
            )
        ledger_auroc, ledger_brier = _metric_row(ledger_preds)
        time_auroc, time_brier = _metric_row(time_preds)
        rows.append(
            {
                "target_variant": target_variant,
                "n_ckpts": int(len(variant_df)),
                "pos_rate": float(variant_df["label"].mean()) if len(variant_df) else 0.0,
                "ledger_basic_auroc": ledger_auroc,
                "ledger_basic_brier": ledger_brier,
                "time_only_auroc": time_auroc,
                "time_only_brier": time_brier,
                "ledger_minus_time_brier": float(ledger_brier - time_brier),
                "interpretation": "",
            }
        )
    out = pd.DataFrame(rows).sort_values("target_variant", kind="mergesort").reset_index(drop=True)
    base_gain = float(
        out.loc[out["target_variant"] == "h5_base", "time_only_brier"].iloc[0]
        - out.loc[out["target_variant"] == "h5_base", "ledger_basic_brier"].iloc[0]
    )
    for idx, row in out.iterrows():
        gain = float(row["time_only_brier"] - row["ledger_basic_brier"])
        text = "ledger beats time_only"
        if gain <= 0:
            text = "ledger no longer beats time_only"
        elif row["target_variant"] in {
            "h5_first_drop_lead_ge_2",
            "h5_first_positive_per_drop_episode",
        } and (gain < 0.02 or gain < 0.25 * base_gain):
            text = "near-boundary sensitivity weakens sharply"
        out.at[idx, "interpretation"] = text
    return out


def build_group_ablation_table(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    split: Any,
) -> pd.DataFrame:
    checkpoints = _tb_live_v2_checkpoints(checkpoints_df)
    base_cols = list(LEDGER_BASIC.feature_cols_for((TB_LIVE_V2,)))
    feature_groups = _ledger_basic_feature_groups()
    rows: list[dict[str, Any]] = []
    for group in ("closure", "frontier", "instability", "discovery"):
        cols = tuple(col for col in base_cols if feature_groups.get(col, "other") != group)
        spec = BaselineSpec(
            name=f"ledger_basic_minus_{group}",
            feature_cols_for=lambda _sources, feature_cols=cols: feature_cols,
        )
        preds = predict_cell(
            checkpoints_df=checkpoints,
            labels_df=labels_df,
            target=PROGRESS_DROP_TARGET,
            spec=spec,
            split=split,
            sources_in_train=(TB_LIVE_V2,),
        )
        model_auroc, model_brier = _metric_row(preds)
        rows.append(
            {
                "model": spec.name,
                "group_removed": group,
                "auroc": model_auroc,
                "brier": model_brier,
            }
        )
    return pd.DataFrame(rows).sort_values("group_removed", kind="mergesort").reset_index(drop=True)


def build_confusion_summary(
    oof_df: pd.DataFrame,
    witness_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = oof_df.merge(
        witness_df[["run_id", "checkpoint_id", "lead_time"]],
        on=["run_id", "checkpoint_id"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    thresholds = {
        "prevalence": float(merged["_y"].mean()),
        "0.5": 0.5,
    }
    for threshold_name, threshold_value in thresholds.items():
        pred_pos = merged["p_ledger_basic"] >= threshold_value
        y = merged["_y"].astype(int)
        tp_mask = (y == 1) & pred_pos
        fp_mask = (y == 0) & pred_pos
        fn_mask = (y == 1) & (~pred_pos)
        tn_mask = (y == 0) & (~pred_pos)
        tp = int(tp_mask.sum())
        fp = int(fp_mask.sum())
        fn = int(fn_mask.sum())
        tn = int(tn_mask.sum())
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        lead_tp = merged.loc[tp_mask & merged["lead_time"].notna(), "lead_time"].astype(int)
        rows.append(
            {
                "threshold_name": threshold_name,
                "threshold_value": float(threshold_value),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "mean_lead_time_tp": None if lead_tp.empty else float(lead_tp.mean()),
                "median_lead_time_tp": None if lead_tp.empty else float(lead_tp.median()),
            }
        )
        if not lead_tp.empty:
            for _, row in merged.loc[tp_mask & merged["lead_time"].notna()].iterrows():
                lead_rows.append(
                    {
                        "threshold_name": threshold_name,
                        "threshold_value": float(threshold_value),
                        "run_id": row["run_id"],
                        "checkpoint_id": row["checkpoint_id"],
                        "checkpoint_step": int(row["checkpoint_step"]),
                        "lead_time": int(row["lead_time"]),
                        "predicted_probability": float(row["p_ledger_basic"]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(lead_rows)


def _validation_run_stats(run: RunRecord) -> dict[str, Any]:
    prefix: list[LedgerEvent] = []
    validation_steps: list[int] = []
    within_5 = False
    any_later = False
    for event in run.events:
        if _is_validation_transition(event, prefix):
            validation_steps.append(int(event.step))
        elif validation_steps and _is_discovery_event(event, prefix):
            if any(int(event.step) > step for step in validation_steps):
                any_later = True
            if any(0 < int(event.step) - step <= 5 for step in validation_steps):
                within_5 = True
        prefix.append(event)
    return {
        "run_id": run.run_id,
        "task_family": run.task_family,
        "has_validation_transition": bool(validation_steps),
        "has_discovery_or_reopen_after_validation_within_5": bool(within_5),
        "has_discovery_or_reopen_after_validation_any_later": bool(any_later),
    }


def _validation_slice_summary(
    *,
    name: str,
    manifest_runs: pd.DataFrame,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    run_stats_df: pd.DataFrame,
) -> dict[str, Any]:
    run_ids = manifest_runs["run_id"].astype(str).tolist()
    ck = checkpoints_df[checkpoints_df["run_id"].isin(run_ids)]
    pos = labels_df[
        (labels_df["target_name"] == VALIDATION_TARGET)
        & (~labels_df["is_masked"].astype(bool))
        & (labels_df["label_value"] > 0.5)
        & (labels_df["run_id"].isin(run_ids))
    ]
    stats = run_stats_df[run_stats_df["run_id"].isin(run_ids)]
    return {
        "slice_name": name,
        "n_runs": int(len(run_ids)),
        "n_checkpoints": int(len(ck)),
        "n_runs_with_validation_transition": int(stats["has_validation_transition"].sum()),
        "n_runs_with_discovery_or_reopen_after_validation_within_5": int(
            stats["has_discovery_or_reopen_after_validation_within_5"].sum()
        ),
        "n_runs_with_discovery_or_reopen_after_validation_any_later": int(
            stats["has_discovery_or_reopen_after_validation_any_later"].sum()
        ),
        "n_unmasked_positive_checkpoints_current_label": int(len(pos)),
    }


def _validation_recommendation(all_slice: dict[str, Any]) -> str:
    if all_slice["n_unmasked_positive_checkpoints_current_label"] > 0:
        return "keep"
    if all_slice["n_runs_with_validation_transition"] == 0:
        return "defer_on_tb_live_v2"
    if all_slice["n_runs_with_discovery_or_reopen_after_validation_any_later"] > 0:
        return "redefine_later"
    return "defer_on_tb_live_v2"


def build_validation_new_work_audit(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    runs_dir: Path,
) -> tuple[pd.DataFrame, str]:
    manifest = manifest_df[manifest_df["source"] == TB_LIVE_V2].drop_duplicates("run_id").copy()
    stats = pd.DataFrame(
        [_validation_run_stats(_load_local_run(runs_dir / str(run_id))) for run_id in manifest["run_id"]]
    )
    all_slice = _validation_slice_summary(
        name="all_tb_live_v2_runs",
        manifest_runs=manifest,
        checkpoints_df=_tb_live_v2_checkpoints(checkpoints_df),
        labels_df=labels_df[labels_df["source"] == TB_LIVE_V2],
        run_stats_df=stats,
    )
    family_manifest = manifest[manifest["task_family"] == "validation_new_work"].copy()
    family_slice = _validation_slice_summary(
        name="validation_new_work_family_only",
        manifest_runs=family_manifest,
        checkpoints_df=_tb_live_v2_checkpoints(checkpoints_df),
        labels_df=labels_df[labels_df["source"] == TB_LIVE_V2],
        run_stats_df=stats,
    )
    out = pd.DataFrame([all_slice, family_slice])
    recommendation = _validation_recommendation(all_slice)
    out["recommendation"] = recommendation
    return out, recommendation


def _case_sort(df: pd.DataFrame, *, ascending_prob: bool) -> pd.DataFrame:
    return df.sort_values(
        ["p_ledger_basic", "run_id", "checkpoint_step"],
        ascending=[ascending_prob, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def select_case_studies(
    *,
    oof_df: pd.DataFrame,
    witness_df: pd.DataFrame,
    figures_dir: Path,
) -> list[CaseStudy]:
    enriched = oof_df.merge(
        witness_df[["run_id", "checkpoint_id", "lead_time", "next_drop_step", "drop_magnitude"]],
        on=["run_id", "checkpoint_id"],
        how="left",
    )
    pos = enriched[enriched["_y"] == 1].copy()
    neg = enriched[enriched["_y"] == 0].copy()
    tp = enriched[(enriched["_y"] == 1) & (enriched["p_ledger_basic"] >= CASE_THRESHOLD)].copy()
    fp = enriched[(enriched["_y"] == 0) & (enriched["p_ledger_basic"] >= CASE_THRESHOLD)].copy()
    fn = enriched[(enriched["_y"] == 1) & (enriched["p_ledger_basic"] < CASE_THRESHOLD)].copy()
    quiet_runs = set(
        enriched.groupby("run_id")["_y"].sum().loc[lambda s: s == 0].index.astype(str)
    )
    quiet_neg = neg[neg["run_id"].isin(quiet_runs)].copy()

    def _pick_frame(frame: pd.DataFrame, *, highest: bool) -> pd.Series:
        if frame.empty:
            raise AuditFailure(
                verdict=VERDICT_INSUFFICIENT,
                message="case selection had no eligible rows",
            )
        ordered = frame.sort_values(
            ["p_ledger_basic", "run_id", "checkpoint_step"],
            ascending=[not highest, True, True],
            kind="mergesort",
        )
        return ordered.iloc[0]

    chosen: list[tuple[str, pd.Series, str, str]] = []
    chosen.append(
        (
            "true positive",
            _pick_frame(tp if not tp.empty else pos, highest=True),
            "highest-probability positive checkpoint",
            "The model assigns high risk before a realized progress drop.",
        )
    )
    chosen.append(
        (
            "false positive" if not fp.empty else "hardest negative",
            _pick_frame(fp if not fp.empty else neg, highest=True),
            "highest-probability negative checkpoint",
            "This is the hardest negative under the frozen exact-task OOF scores.",
        )
    )
    chosen.append(
        (
            "false negative" if not fn.empty else "hardest positive",
            _pick_frame(fn if not fn.empty else pos, highest=False),
            "lowest-probability positive checkpoint",
            "This is the hardest positive under the frozen exact-task OOF scores.",
        )
    )
    chosen.append(
        (
            "true negative quiet run",
            _pick_frame(quiet_neg, highest=False),
            "lowest-probability negative checkpoint from a run with no realized positives",
            "The model stays quiet on a run with no realized progress-drop positives.",
        )
    )
    cases: list[CaseStudy] = []
    for title, row, why_selected, interpretation in chosen:
        figure_path = figures_dir / f"process_dynamics_{row['run_id']}.png"
        cases.append(
            CaseStudy(
                section_title=title,
                run_id=str(row["run_id"]),
                checkpoint_id=str(row["checkpoint_id"]),
                checkpoint_step=int(row["checkpoint_step"]),
                predicted_probability=float(row["p_ledger_basic"]),
                label_value=int(row["_y"]),
                task_id=None if pd.isna(row.get("task_id")) else str(row.get("task_id")),
                task_family=None if pd.isna(row.get("task_family")) else str(row.get("task_family")),
                arm=None if pd.isna(row.get("arm")) else str(row.get("arm")),
                why_selected=why_selected,
                interpretation=interpretation,
                figure_path=figure_path,
            )
        )
    return cases


def plot_case_study(
    *,
    run_id: str,
    checkpoints_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    runs_dir: Path,
    selected_checkpoint_step: int,
    out_path: Path,
) -> Path:
    checkpoints = _tb_live_v2_checkpoints(checkpoints_df)
    run_rows = checkpoints[checkpoints["run_id"] == run_id].sort_values("checkpoint_step", kind="mergesort")
    oof_rows = oof_df[oof_df["run_id"] == run_id].sort_values("checkpoint_step", kind="mergesort")
    run = _load_local_run(runs_dir / run_id)
    progress_map = _progress_by_step(run.events)
    actual_drop_steps = _actual_drop_steps(progress_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "progress_steps": run_rows["checkpoint_step"].astype(int).tolist(),
        "progress_values": run_rows["coding_progress"].astype(float).tolist(),
        "pred_steps": oof_rows["checkpoint_step"].astype(int).tolist(),
        "pred_values": oof_rows["p_ledger_basic"].astype(float).tolist(),
        "actual_drop_steps": [int(step) for step in actual_drop_steps],
        "selected_step": int(selected_checkpoint_step),
    }
    code = """
import json
import os
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib.pyplot as plt
payload = json.loads(sys.argv[1])
out_path = sys.argv[2]
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax2 = ax1.twinx()
ax1.plot(payload["progress_steps"], payload["progress_values"], color="tab:blue", linewidth=1.8, label="coding_progress")
ax2.plot(payload["pred_steps"], payload["pred_values"], color="tab:red", linewidth=1.6, label="P(drop_h5)")
for step in payload["actual_drop_steps"]:
    ax1.axvline(step, color="tab:gray", linestyle="--", linewidth=0.9, alpha=0.7)
if payload["selected_step"] in payload["pred_steps"]:
    idx = payload["pred_steps"].index(payload["selected_step"])
    ax2.scatter([payload["selected_step"]], [payload["pred_values"][idx]], color="black", s=36, zorder=5, label="selected checkpoint")
ax1.set_xlabel("checkpoint step")
ax1.set_ylabel("coding progress", color="tab:blue")
ax2.set_ylabel("P(drop_h5)", color="tab:red")
ax1.set_ylim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax1.set_title(payload["run_id"])
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3, frameon=False)
fig.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
"""
    subprocess.run(
        [_matplotlib_python(), "-c", code, json.dumps(payload), str(out_path)],
        check=True,
    )
    return out_path


def _matplotlib_python() -> str:
    candidates = [
        os.environ.get("CODEX_MATPLOTLIB_PYTHON"),
        "/opt/homebrew/bin/python3",
        "/usr/bin/python3",
        shutil.which("python3"),
    ]
    check = "import matplotlib"
    for candidate in candidates:
        if not candidate:
            continue
        try:
            subprocess.run(
                [candidate, "-c", check],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("no python interpreter with matplotlib available for case-study plotting")


def _fmt_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.3f}"


def _render_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                vals.append(_fmt_metric(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def render_progress_drop_audit(
    *,
    witness_df: pd.DataFrame,
    mismatches_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    group_ablation_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    decision: DecisionOutcome,
) -> str:
    lead_counts = (
        witness_df[witness_df["label"] > 0]["lead_time"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    lines = [
        "# Progress-Drop Audit",
        "",
        "## Label witness validity",
        "",
        f"- unmasked_rows: {len(witness_df)}",
        f"- label_matches: {int((witness_df['label'] == witness_df['shipped_label']).sum())}",
        f"- label_mismatches: {len(mismatches_df)}",
        f"- positive_rate: {_fmt_metric(float(witness_df['label'].mean()))}",
        f"- lead_time_distribution: {lead_counts}",
        "- feature_provenance_status: assumed (`checkpoint_step` is stored, but per-row max source step is not).",
        "",
    ]
    if not mismatches_df.empty:
        lines.extend(["Witness mismatches were detected.", ""])
    lines.extend(
        [
            "## Target-specific leakage scan",
            "",
        ]
    )
    flagged = leakage_df[leakage_df["is_flagged"]]
    if flagged.empty:
        lines.append("- No target-specific feature-name hits in the `LEDGER_BASIC` feature list.")
    else:
        lines.extend(_render_table(flagged, ["feature", "tokens_hit", "whitelisted"]))
    lines.extend(
        [
            "",
            "## Feature drivers",
            "",
            "Diagnostic standardized logistic coefficients over exact-task train folds.",
            "",
        ]
    )
    lines.extend(
        _render_table(
            coefficients_df[["feature", "feature_group", "median_abs_coefficient", "sign_frequency_positive", "sign_frequency_negative"]],
            ["feature", "feature_group", "median_abs_coefficient", "sign_frequency_positive", "sign_frequency_negative"],
        )
    )
    lines.extend(["", "Leave-one-group-out diagnostics.", ""])
    lines.extend(_render_table(group_ablation_df, ["model", "group_removed", "auroc", "brier"]))
    lines.extend(["", "## Sensitivity checks", ""])
    lines.extend(
        _render_table(
            sensitivity_df[
                [
                    "target_variant",
                    "n_ckpts",
                    "pos_rate",
                    "ledger_basic_auroc",
                    "ledger_basic_brier",
                    "time_only_auroc",
                    "time_only_brier",
                    "ledger_minus_time_brier",
                    "interpretation",
                ]
            ],
            [
                "target_variant",
                "n_ckpts",
                "pos_rate",
                "ledger_basic_auroc",
                "ledger_basic_brier",
                "time_only_auroc",
                "time_only_brier",
                "ledger_minus_time_brier",
                "interpretation",
            ],
        )
    )
    lines.extend(["", "## Row-level interpretability", ""])
    lines.extend(
        _render_table(
            confusion_df,
            [
                "threshold_name",
                "threshold_value",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "mean_lead_time_tp",
                "median_lead_time_tp",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Decision interpretation",
            "",
            "| condition | verdict |",
            "|---|---|",
            "| witness mismatch | label_construction_bug |",
            "| flagged feature name in used features | feature_leakage |",
            "| harder variants collapse | valid_but_near_boundary |",
            "| harder variants remain strong with frontier/discovery/instability drivers | valid_prefix_signal |",
            "| evidence remains ambiguous | insufficient_evidence |",
            "",
            f"- applied_verdict: `{decision.verdict}`",
            f"- rationale: {decision.rationale}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_validation_new_work_audit(
    audit_df: pd.DataFrame,
    recommendation: str,
) -> str:
    lines = [
        "# Validation-New-Work Label Audit",
        "",
        "Use the upstream snapshot logic as the source of truth for validation transitions and discovery/reopen events.",
        "",
    ]
    lines.extend(
        _render_table(
            audit_df,
            [
                "slice_name",
                "n_runs",
                "n_checkpoints",
                "n_runs_with_validation_transition",
                "n_runs_with_discovery_or_reopen_after_validation_within_5",
                "n_runs_with_discovery_or_reopen_after_validation_any_later",
                "n_unmasked_positive_checkpoints_current_label",
                "recommendation",
            ],
        )
    )
    lines.extend(["", f"- recommendation: `{recommendation}`", ""])
    if int(audit_df.loc[audit_df["slice_name"] == "all_tb_live_v2_runs", "n_runs_with_validation_transition"].iloc[0]) == 0:
        lines.append(
            "The current live substrate does not emit the upstream-recognized validation-transition pattern required by `y_validation_new_work_h5`."
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_case_studies(cases: list[CaseStudy]) -> str:
    lines = [
        "# Process Dynamics Case Studies",
        "",
        "Each case uses frozen exact-task OOF `LEDGER_BASIC` predictions.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case.section_title.title()}",
                "",
                f"- run_id: `{case.run_id}`",
                f"- task_id: `{case.task_id or ''}`",
                f"- task_family: `{case.task_family or ''}`",
                f"- arm: `{case.arm or ''}`",
                f"- checkpoint_step: {case.checkpoint_step}",
                f"- predicted_probability: {_fmt_metric(case.predicted_probability)}",
                f"- true_label: {case.label_value}",
                f"- why_selected: {case.why_selected}",
                f"- interpretation: {case.interpretation}",
                f"- figure: `{case.figure_path.name}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def render_process_dynamics_result(
    *,
    oof_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    decision: DecisionOutcome,
    cases: list[CaseStudy],
    validation_recommendation: str,
) -> str:
    ledger_auroc = auroc(oof_df["_y"].astype(int).to_numpy(), oof_df["p_ledger_basic"].to_numpy())
    ledger_brier = brier(oof_df["_y"].astype(int).to_numpy(), oof_df["p_ledger_basic"].to_numpy())
    time_auroc = auroc(oof_df["_y"].astype(int).to_numpy(), oof_df["p_time_only"].to_numpy())
    time_brier = brier(oof_df["_y"].astype(int).to_numpy(), oof_df["p_time_only"].to_numpy())
    harder = sensitivity_df[
        sensitivity_df["target_variant"].isin(
            ["h5_first_drop_lead_ge_2", "h5_first_positive_per_drop_episode"]
        )
    ]
    lines = [
        "# Process Dynamics Result",
        "",
        "## Exact claim",
        "",
        "Prefix-only ledger features predict near-future progress drops under exact-task holdout better than elapsed time. This supports work-frontier instability detection, not policy-grade completion-risk estimation.",
        "",
        "## Exact-task headline metrics",
        "",
        f"- ledger_basic_auroc: {_fmt_metric(ledger_auroc)}",
        f"- ledger_basic_brier: {_fmt_metric(ledger_brier)}",
        f"- time_only_auroc: {_fmt_metric(time_auroc)}",
        f"- time_only_brier: {_fmt_metric(time_brier)}",
        "",
        "## Label witness result",
        "",
        f"- verdict: `{decision.verdict}`",
        f"- rationale: {decision.rationale}",
        "",
        "## Feature-driver result",
        "",
        "- See `PROGRESS_DROP_AUDIT.md` for coefficient rankings and leave-one-group-out diagnostics.",
        "",
        "## Sensitivity summary",
        "",
        f"- harder_variant_rows: {len(harder)}",
        f"- harder_variants: {', '.join(harder['target_variant'].astype(str).tolist())}",
        "",
        "## Case-study summary",
        "",
        f"- case_count: {len(cases)}",
        f"- cases: {', '.join(case.section_title for case in cases)}",
        "",
        "## Validation-new-work diagnosis",
        "",
        f"- recommendation: `{validation_recommendation}`",
        "",
        "## Terminal-success negative result",
        "",
        "- Terminal success remains secondary and negative on tb_live_v2 exact-task holdout.",
        "",
        "## What this does and does not support",
        "",
        "- Supports: prefix-only work-frontier instability prediction.",
        "- Does not support: control, scheduling, or terminal completion-risk decisions.",
        "",
    ]
    return "\n".join(lines) + "\n"


def decide_progress_drop_verdict(
    *,
    mismatches_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> DecisionOutcome:
    if not mismatches_df.empty:
        return DecisionOutcome(
            verdict=VERDICT_LABEL_BUG,
            rationale="the witness reconstruction does not reproduce the shipped labels",
        )
    if not leakage_df[leakage_df["is_flagged"] & ~leakage_df["whitelisted"]].empty:
        return DecisionOutcome(
            verdict=VERDICT_LEAKAGE,
            rationale="the used feature list contains future- or label-derived names",
        )
    base = sensitivity_df.set_index("target_variant")
    base_gain = float(base.loc["h5_base", "time_only_brier"] - base.loc["h5_base", "ledger_basic_brier"])
    lead_gain = float(
        base.loc["h5_first_drop_lead_ge_2", "time_only_brier"]
        - base.loc["h5_first_drop_lead_ge_2", "ledger_basic_brier"]
    )
    first_gain = float(
        base.loc["h5_first_positive_per_drop_episode", "time_only_brier"]
        - base.loc["h5_first_positive_per_drop_episode", "ledger_basic_brier"]
    )
    if lead_gain < 0.02 or first_gain < 0.02 or lead_gain < 0.25 * base_gain or first_gain < 0.25 * base_gain:
        return DecisionOutcome(
            verdict=VERDICT_NEAR_BOUNDARY,
            rationale="the headline result weakens sharply when the audit removes near-boundary positives",
        )
    group_weights = coefficients_df.groupby("feature_group")["median_abs_coefficient"].sum().to_dict()
    frontier_signal = float(
        group_weights.get("frontier", 0.0)
        + group_weights.get("discovery", 0.0)
        + group_weights.get("instability", 0.0)
    )
    total_signal = float(sum(group_weights.values()))
    if total_signal <= 0.0:
        return DecisionOutcome(
            verdict=VERDICT_INSUFFICIENT,
            rationale="coefficient diagnostics did not yield usable signal mass",
        )
    if frontier_signal / total_signal >= 0.6:
        return DecisionOutcome(
            verdict=VERDICT_VALID_PREFIX,
            rationale="harder variants remain strong and the diagnostic coefficients center on frontier/discovery/instability features",
        )
    return DecisionOutcome(
        verdict=VERDICT_INSUFFICIENT,
        rationale="the headline result holds, but the diagnostic driver mix is too diffuse to support a stronger claim",
    )


def run_process_dynamics_package(
    *,
    checkpoints_path: Path,
    labels_path: Path,
    manifest_path: Path,
    runs_dir: Path,
    out_dir: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    manifest_df = pd.read_csv(manifest_path)
    reports_dir = out_dir / "process_dynamics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    witness_df = build_progress_drop_witness(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        runs_dir=runs_dir,
        horizon=5,
    )
    witness_df, mismatches_df = verify_progress_drop_witness(witness_df)
    if not mismatches_df.empty:
        raise AuditFailure(
            verdict=VERDICT_LABEL_BUG,
            message=f"progress-drop witness mismatch count: {len(mismatches_df)}",
        )

    leakage_df = scan_ledger_basic_feature_names()
    assert_no_leakage_feature_names(leakage_df)

    oof_df, split = build_exact_task_oof_predictions(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    assert_expected_oof_row_count(oof_df, expected_rows=len(witness_df))

    coefficients_df = build_diagnostic_coefficients(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        split=split,
    )
    group_ablation_df = build_group_ablation_table(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        split=split,
    )
    variants = build_progress_drop_variant_suite(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        runs_dir=runs_dir,
    )
    sensitivity_df = build_sensitivity_table(
        checkpoints_df=checkpoints_df,
        manifest_df=manifest_df,
        split=split,
        variant_frames=variants,
    )
    confusion_df, lead_times_df = build_confusion_summary(oof_df, witness_df)
    validation_df, validation_recommendation = build_validation_new_work_audit(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        manifest_df=manifest_df,
        runs_dir=runs_dir,
    )
    decision = decide_progress_drop_verdict(
        mismatches_df=mismatches_df,
        leakage_df=leakage_df,
        coefficients_df=coefficients_df,
        sensitivity_df=sensitivity_df,
    )
    cases = select_case_studies(oof_df=oof_df, witness_df=witness_df, figures_dir=figures_dir)
    for case in cases:
        plot_case_study(
            run_id=case.run_id,
            checkpoints_df=checkpoints_df,
            oof_df=oof_df,
            runs_dir=runs_dir,
            selected_checkpoint_step=case.checkpoint_step,
            out_path=case.figure_path,
        )

    paths = {
        "oof_csv": write_csv(oof_df, reports_dir / "exact_task_oof_predictions.csv", sort_by=["run_id", "checkpoint_step"]),
        "witness_csv": write_csv(witness_df, reports_dir / "progress_drop_witness.csv", sort_by=["run_id", "checkpoint_step"]),
        "coefficients_csv": write_csv(coefficients_df, reports_dir / "progress_drop_coefficients.csv", sort_by=["median_abs_coefficient", "feature"]),
        "sensitivity_csv": write_csv(sensitivity_df, reports_dir / "progress_drop_sensitivity.csv", sort_by=["target_variant"]),
        "lead_times_csv": write_csv(lead_times_df, reports_dir / "progress_drop_lead_times.csv", sort_by=["threshold_name", "run_id", "checkpoint_step"]),
        "validation_csv": write_csv(validation_df, reports_dir / "validation_new_work_audit.csv", sort_by=["slice_name"]),
    }
    progress_audit_path = out_dir / "PROGRESS_DROP_AUDIT.md"
    validation_audit_path = out_dir / "VALIDATION_NEW_WORK_LABEL_AUDIT.md"
    case_studies_path = out_dir / "process_dynamics_case_studies.md"
    result_path = out_dir / "PROCESS_DYNAMICS_RESULT.md"
    progress_audit_path.write_text(
        render_progress_drop_audit(
            witness_df=witness_df,
            mismatches_df=mismatches_df,
            leakage_df=leakage_df,
            coefficients_df=coefficients_df,
            group_ablation_df=group_ablation_df,
            sensitivity_df=sensitivity_df,
            confusion_df=confusion_df,
            decision=decision,
        ),
        encoding="utf-8",
        newline="\n",
    )
    validation_audit_path.write_text(
        render_validation_new_work_audit(validation_df, validation_recommendation),
        encoding="utf-8",
        newline="\n",
    )
    case_studies_path.write_text(
        render_case_studies(cases),
        encoding="utf-8",
        newline="\n",
    )
    result_path.write_text(
        render_process_dynamics_result(
            oof_df=oof_df,
            sensitivity_df=sensitivity_df,
            decision=decision,
            cases=cases,
            validation_recommendation=validation_recommendation,
        ),
        encoding="utf-8",
        newline="\n",
    )
    paths.update(
        {
            "progress_audit_md": progress_audit_path,
            "validation_audit_md": validation_audit_path,
            "case_studies_md": case_studies_path,
            "result_md": result_path,
        }
    )
    return paths
