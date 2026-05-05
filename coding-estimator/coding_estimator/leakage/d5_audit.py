"""D5 — behavioral leakage audit, structured JSON form.

Replaces the bare `{clean: true}` form rejected by P1.g. Emits a JSON
artifact with `schema_version`, `n_runs_audited`, `n_checkpoints_audited`,
`findings`, `clean` plus per-section detail. P1.g loads it; the human
report still lives at `reports/CHECKPOINT_CONSTRUCTION_AUDIT.md`.

Four sections, each contributes findings:

1. STRUCTURAL — forbidden-column matches (exact / prefix / suffix).
2. PREFIX_TRUNCATION — rebuild a row from a ledger truncated at
   `mid_step`; the row must byte-equal the row built from the full
   ledger at the same step (modulo `is_terminal_checkpoint`).
3. SHUFFLE — train G4 with labels shuffled across runs; run-level
   bootstrap AUROC must straddle 0.5 (a leaky model would predict
   the shuffled labels well above chance).
4. RUN_CONSTANCY — joint run-constant (feature, target) pairs in
   any G4 training fold.

Any non-empty finding ⇒ `clean: false` ⇒ P1.g fails.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC
from coding_estimator.checkpoints.build import build_run_rows
from coding_estimator.eval.harness import predict_cell
from coding_estimator.eval.metrics import auroc
from coding_estimator.ingest.run_record import RunRecord, load_run
from coding_estimator.leakage.guard import find_forbidden, load_forbidden_spec
from coding_estimator.leakage.run_constancy import audit as run_constancy_audit
from coding_estimator.splits.protocol import loro

D5_SCHEMA_VERSION = "1.0.0"
SHUFFLE_SEEDS: tuple[int, ...] = (0, 1, 2)
SHUFFLE_AUROC_TOLERANCE: float = 0.10  # |AUROC - 0.5| > this ⇒ finding


@dataclass(frozen=True)
class Finding:
    section: str
    kind: str
    detail: str
    severity: str = "block"  # "block" or "info"


def _structural(checkpoints_df: pd.DataFrame) -> tuple[list[Finding], dict[str, Any]]:
    spec = load_forbidden_spec()
    hits = find_forbidden(checkpoints_df.columns, spec=spec)
    findings: list[Finding] = []
    if hits:
        findings.append(
            Finding(
                section="structural",
                kind="forbidden_column",
                detail=f"forbidden columns present: {hits}",
            )
        )
    return findings, {
        "forbidden_exact": len(spec.exact),
        "forbidden_prefix": len(spec.prefixes),
        "forbidden_suffix": len(spec.suffixes),
        "hits": list(hits),
    }


def _prefix_truncation(
    checkpoints_df: pd.DataFrame, *, sample_runs: int = 4
) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    if "run_id" not in checkpoints_df.columns or "source" not in checkpoints_df.columns:
        findings.append(
            Finding(
                section="prefix_truncation",
                kind="frame_missing_columns",
                detail="checkpoints frame lacks run_id or source",
            )
        )
        return findings, {"sampled_runs": 0}
    sampled = (
        checkpoints_df[["source", "run_id"]]
        .drop_duplicates()
        .head(sample_runs)
        .itertuples(index=False)
    )
    n_checked = 0
    n_skipped = 0
    differing_runs: list[dict[str, Any]] = []
    for source, run_id in sampled:
        try:
            run = load_run(source, run_id)
        except (FileNotFoundError, OSError, ValueError) as exc:
            findings.append(
                Finding(
                    section="prefix_truncation",
                    kind="run_load_failure",
                    detail=f"{source}/{run_id}: {exc}",
                )
            )
            continue
        if len(run.events) < 3:
            n_skipped += 1
            continue
        event_steps = sorted({e.step for e in run.events})
        mid_step = event_steps[len(event_steps) // 2]
        full_rows = build_run_rows(run)
        try:
            mid_full = next(r for r in full_rows if r["checkpoint_step"] == mid_step)
        except StopIteration:
            n_skipped += 1
            continue
        truncated_events = tuple(e for e in run.events if e.step <= mid_step)
        trunc_run = RunRecord(
            run_id=run.run_id,
            source=run.source,
            ledger_path=run.ledger_path,
            events=truncated_events,
            has_real_wallclock=run.has_real_wallclock,
            start_wall_time=run.start_wall_time,
            end_wall_time=run.end_wall_time,
            task_id=run.task_id,
            task_family=run.task_family,
            agent_scaffold=run.agent_scaffold,
            model_name=run.model_name,
            raw_metadata=run.raw_metadata,
        )
        try:
            mid_trunc = next(
                r for r in build_run_rows(trunc_run) if r["checkpoint_step"] == mid_step
            )
        except StopIteration:
            findings.append(
                Finding(
                    section="prefix_truncation",
                    kind="truncated_build_missing_row",
                    detail=f"{source}/{run_id} at step {mid_step}",
                )
            )
            continue
        full_cmp = {k: v for k, v in mid_full.items() if k != "is_terminal_checkpoint"}
        trunc_cmp = {k: v for k, v in mid_trunc.items() if k != "is_terminal_checkpoint"}
        n_checked += 1
        if full_cmp != trunc_cmp:
            differing = sorted(k for k in full_cmp if full_cmp[k] != trunc_cmp.get(k))
            differing_runs.append(
                {"source": source, "run_id": run_id, "step": mid_step, "columns": differing[:8]}
            )
            findings.append(
                Finding(
                    section="prefix_truncation",
                    kind="row_diverges_after_truncation",
                    detail=(
                        f"{source}/{run_id} at step {mid_step}: "
                        f"differing columns {differing[:8]}"
                    ),
                )
            )
    return findings, {
        "sampled_runs": n_checked,
        "skipped_runs": n_skipped,
        "differing_runs": differing_runs,
    }


def _shuffle(
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    targets: Iterable[str],
    *,
    tolerance: float = SHUFFLE_AUROC_TOLERANCE,
) -> tuple[list[Finding], dict[str, Any]]:
    """Shuffle labels across runs (preserving feature row order). Train
    G4 LORO on the shuffled labels, predict, compute AUROC. A leaky
    pipeline keeps AUROC well above 0.5 even when labels are random."""
    findings: list[Finding] = []
    detail: dict[str, Any] = {"sources": {}, "tolerance": tolerance}
    for source in sorted(checkpoints_df["source"].unique()):
        sub = checkpoints_df[checkpoints_df["source"] == source]
        if sub["run_id"].nunique() < 3:
            continue
        per_target: dict[str, dict[str, float | str | None]] = {}
        for target in targets:
            real_lab = labels_df[
                (labels_df["source"] == source)
                & (labels_df["target_name"] == target)
                & (~labels_df["is_masked"].astype(bool))
            ][["run_id", "checkpoint_id", "label_value"]]
            if real_lab.empty:
                continue
            run_ids = sorted(real_lab["run_id"].unique())
            run_label = (
                real_lab.drop_duplicates("run_id")
                .set_index("run_id")["label_value"]
                .astype(float)
                .to_dict()
            )
            if len(set(run_label.values())) < 2:
                per_target[target] = {
                    "auroc_mean": None,
                    "note": "single-class run-level y; shuffle test uninformative",
                }
                continue
            aurocs: list[float] = []
            for seed in SHUFFLE_SEEDS:
                rng = np.random.default_rng(seed)
                shuffled_run_ids = list(run_ids)
                rng.shuffle(shuffled_run_ids)
                run_to_shuffled = dict(zip(run_ids, shuffled_run_ids, strict=True))
                shuffled_lab = real_lab.copy()
                shuffled_lab["label_value"] = shuffled_lab["run_id"].map(
                    lambda r: run_label[run_to_shuffled[r]]
                )
                # Build a synthetic labels_df with shuffled rows for this target.
                masked = labels_df[
                    (labels_df["source"] == source)
                    & (labels_df["target_name"] == target)
                ]
                # Replace the un-masked rows' label_value with the shuffled values.
                synth = masked.copy()
                synth = synth.merge(
                    shuffled_lab.rename(columns={"label_value": "_shuf"}),
                    on=["run_id", "checkpoint_id"],
                    how="left",
                )
                mask_unmask = ~synth["is_masked"].astype(bool)
                synth.loc[mask_unmask, "label_value"] = synth.loc[mask_unmask, "_shuf"]
                synth = synth.drop(columns=["_shuf"])
                # Inject back into labels_df only for this (source, target).
                others = labels_df[
                    ~(
                        (labels_df["source"] == source)
                        & (labels_df["target_name"] == target)
                    )
                ]
                synth_all = pd.concat([others, synth], ignore_index=True)
                preds = predict_cell(
                    checkpoints_df=sub,
                    labels_df=synth_all,
                    target=target,
                    spec=LEDGER_BASIC,
                    split=loro(sub),
                    sources_in_train=(source,),
                )
                if preds.empty:
                    continue
                y = preds["_y"].astype(int).to_numpy()
                p = preds["_p"].astype(float).to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                a = auroc(y, p)
                if a is not None:
                    aurocs.append(a)
            if not aurocs:
                per_target[target] = {
                    "auroc_mean": None,
                    "note": "all shuffles produced single-class y or empty preds",
                }
                continue
            mean_auroc = float(np.mean(aurocs))
            per_target[target] = {
                "auroc_mean": mean_auroc,
                "auroc_seeds": [float(a) for a in aurocs],
                "note": None,
            }
            if abs(mean_auroc - 0.5) > tolerance:
                findings.append(
                    Finding(
                        section="shuffle",
                        kind="shuffled_auroc_excursion",
                        detail=(
                            f"{source} / {target}: mean AUROC on label-shuffled "
                            f"data is {mean_auroc:.3f}; |Δ from 0.5| > {tolerance}"
                        ),
                    )
                )
        if per_target:
            detail["sources"][source] = per_target
    return findings, detail


def _run_constancy(
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    targets: Iterable[str],
) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    feat_cols = tuple(
        c for c in LEDGER_BASIC.feature_cols_for(()) if c in checkpoints_df.columns
    )
    n_audited = 0
    offenders: list[dict[str, Any]] = []
    for source in sorted(checkpoints_df["source"].unique()):
        sub = checkpoints_df[checkpoints_df["source"] == source]
        if sub["run_id"].nunique() < 2:
            continue
        split = loro(sub)
        for fold in split.folds:
            train = sub[sub["run_id"].isin(set(fold.train_run_ids))]
            for target in targets:
                lab = labels_df[
                    (labels_df["source"] == source)
                    & (labels_df["target_name"] == target)
                    & (~labels_df["is_masked"].astype(bool))
                ][["run_id", "checkpoint_id", "label_value"]]
                if lab.empty:
                    continue
                joined = train.merge(lab, on=["run_id", "checkpoint_id"], how="inner")
                if joined.empty:
                    continue
                joined = joined.rename(columns={"label_value": "__target__"})
                pairs = run_constancy_audit(
                    joined,
                    feature_columns=feat_cols,
                    target_columns=("__target__",),
                )
                n_audited += 1
                if pairs:
                    offenders.append(
                        {
                            "source": source,
                            "target": target,
                            "fold_id": fold.fold_id,
                            "pairs": [list(p) for p in pairs],
                        }
                    )
                    findings.append(
                        Finding(
                            section="run_constancy",
                            kind="joint_run_constant_pair",
                            detail=(
                                f"{source} / {target} / {fold.fold_id}: "
                                f"{[list(p) for p in pairs]}"
                            ),
                        )
                    )
    return findings, {"audited_cells": n_audited, "offenders": offenders}


@dataclass(frozen=True)
class D5Audit:
    schema_version: str
    n_runs_audited: int
    n_checkpoints_audited: int
    findings: list[dict[str, Any]]
    clean: bool
    sections: dict[str, Any] = field(default_factory=dict)


D5_HEADLINE_TARGETS: tuple[str, ...] = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
)


def run_d5_audit(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    targets: Iterable[str] = D5_HEADLINE_TARGETS,
    sample_runs_for_truncation: int = 4,
) -> D5Audit:
    findings: list[Finding] = []
    sections: dict[str, Any] = {}

    f, d = _structural(checkpoints_df)
    findings.extend(f)
    sections["structural"] = d

    f, d = _prefix_truncation(
        checkpoints_df, sample_runs=sample_runs_for_truncation
    )
    findings.extend(f)
    sections["prefix_truncation"] = d

    f, d = _shuffle(checkpoints_df, labels_df, list(targets))
    findings.extend(f)
    sections["shuffle"] = d

    f, d = _run_constancy(checkpoints_df, labels_df, list(targets))
    findings.extend(f)
    sections["run_constancy"] = d

    return D5Audit(
        schema_version=D5_SCHEMA_VERSION,
        n_runs_audited=int(checkpoints_df["run_id"].nunique()),
        n_checkpoints_audited=int(len(checkpoints_df)),
        findings=[asdict(x) for x in findings],
        clean=not findings,
        sections=sections,
    )


def write_d5_audit(audit: D5Audit, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(asdict(audit), fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def render_d5_summary_md(audit: D5Audit) -> str:
    lines = [
        "# D5 behavioral leakage audit",
        "",
        f"- schema_version: `{audit.schema_version}`",
        f"- n_runs_audited: {audit.n_runs_audited}",
        f"- n_checkpoints_audited: {audit.n_checkpoints_audited}",
        f"- clean: **{audit.clean}**",
        f"- findings: {len(audit.findings)}",
        "",
        "## Section results",
        "",
        "| section | findings | detail key counts |",
        "|---|---:|---|",
    ]
    by_section: dict[str, int] = {}
    for f in audit.findings:
        by_section[f["section"]] = by_section.get(f["section"], 0) + 1
    for sec in ("structural", "prefix_truncation", "shuffle", "run_constancy"):
        n = by_section.get(sec, 0)
        d = audit.sections.get(sec, {})
        lines.append(f"| {sec} | {n} | keys={sorted(d.keys())} |")
    lines.append("")
    if audit.findings:
        lines.append("## Findings")
        lines.append("")
        for f in audit.findings:
            lines.append(f"- `{f['section']}` / `{f['kind']}`: {f['detail']}")
    return "\n".join(lines) + "\n"


def write_d5_summary_md(audit: D5Audit, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_d5_summary_md(audit), encoding="utf-8", newline="\n")
    return path


__all__ = [
    "D5_SCHEMA_VERSION",
    "D5_HEADLINE_TARGETS",
    "D5Audit",
    "run_d5_audit",
    "write_d5_audit",
    "write_d5_summary_md",
    "render_d5_summary_md",
]
