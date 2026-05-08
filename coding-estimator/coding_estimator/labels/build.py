"""Combined long-form label table builder (Workstream E7).

Produces one row per `(run_id, checkpoint_id, target_name)` covering
every v0 target in `labels.registry.V0_TARGETS`. Columns follow
`schemas/label_schema.json` with the `target_horizon` object flattened
into `target_horizon_units` / `target_horizon_value`.

Output:
  datasets/labels_<source>.parquet
  datasets/labels_all.parquet (combined across canonical sources)

The CLI hard-fails on unresolvable label runs (load_final_label raises);
upstream `combined_manifest` is the place to log skips, not this table.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.policy import p_step_checkpoints
from coding_estimator.ingest import paths
from coding_estimator.ingest.labels import (
    UnresolvableLabelError,
    load_final_label,
)
from coding_estimator.ingest.run_record import RunRecord, load_run
from coding_estimator.ingest.sources import SOURCES, canonical_sources
from coding_estimator.io import write_parquet
from coding_estimator.labels.dynamics import (
    H5,
    DynamicsLabel,
    future_progress_drop_h5,
    validation_new_work_h5,
)
from coding_estimator.labels.terminal import TerminalLabels, terminal_labels

SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class LabelRow:
    run_id: str
    source: str
    checkpoint_id: str
    checkpoint_step: int
    is_terminal_checkpoint: bool
    task_id: str | None
    task_family: str | None
    arm: str | None
    difficulty: str | None
    agent_scaffold: str | None
    model_name: str | None
    target_name: str
    target_family: str
    target_horizon_units: str
    target_horizon_value: int | None
    label_value: float | None
    is_masked: bool
    mask_reason: str | None
    label_available: bool
    schema_version: str

    def to_schema_dict(self) -> dict:
        """View matching `schemas/label_schema.json` (nested target_horizon
        and only the schema's required + optional keys). Parquet storage
        keeps the horizon flattened; this view is what schema validators
        consume."""
        return {
            "run_id": self.run_id,
            "source": self.source,
            "checkpoint_id": self.checkpoint_id,
            "target_name": self.target_name,
            "target_family": self.target_family,
            "target_horizon": {
                "units": self.target_horizon_units,
                "value": self.target_horizon_value,
            },
            "label_value": self.label_value,
            "is_masked": self.is_masked,
            "mask_reason": self.mask_reason,
            "schema_version": self.schema_version,
        }


def _terminal_row(
    run: RunRecord,
    checkpoint_id: str,
    checkpoint_step: int,
    is_terminal: bool,
    target_name: str,
    target_family: str,
    value: float | bool | int | None,
) -> LabelRow:
    coerced = (
        None
        if value is None
        else (float(value) if isinstance(value, bool) else float(value))
    )
    return LabelRow(
        run_id=run.run_id,
        source=run.source,
        checkpoint_id=checkpoint_id,
        checkpoint_step=checkpoint_step,
        is_terminal_checkpoint=is_terminal,
        task_id=run.task_id,
        task_family=run.task_family,
        arm=run.arm,
        difficulty=run.difficulty,
        agent_scaffold=run.agent_scaffold,
        model_name=run.model_name,
        target_name=target_name,
        target_family=target_family,
        target_horizon_units="terminal",
        target_horizon_value=None,
        label_value=coerced,
        is_masked=False,
        mask_reason=None,
        label_available=coerced is not None,
        schema_version=SCHEMA_VERSION,
    )


def _dynamics_row(
    run: RunRecord,
    checkpoint_id: str,
    checkpoint_step: int,
    is_terminal: bool,
    target_name: str,
    target_family: str,
    horizon: int,
    label: DynamicsLabel,
) -> LabelRow:
    coerced = None if label.value is None else float(label.value)
    return LabelRow(
        run_id=run.run_id,
        source=run.source,
        checkpoint_id=checkpoint_id,
        checkpoint_step=checkpoint_step,
        is_terminal_checkpoint=is_terminal,
        task_id=run.task_id,
        task_family=run.task_family,
        arm=run.arm,
        difficulty=run.difficulty,
        agent_scaffold=run.agent_scaffold,
        model_name=run.model_name,
        target_name=target_name,
        target_family=target_family,
        target_horizon_units="steps",
        target_horizon_value=horizon,
        label_value=coerced,
        is_masked=label.is_masked,
        mask_reason=label.mask_reason,
        label_available=not label.is_masked,
        schema_version=SCHEMA_VERSION,
    )


def build_run_label_rows(run: RunRecord, term: TerminalLabels) -> list[LabelRow]:
    steps = p_step_checkpoints(run)
    terminal_step = steps[-1]
    finish_step = term.y_finish_step
    rows: list[LabelRow] = []
    for t in steps:
        cid = f"{run.run_id}::{t}"
        is_terminal = t == terminal_step

        rows.append(
            _terminal_row(
                run, cid, t, is_terminal,
                "y_success_eventual", "success",
                term.y_success_eventual,
            )
        )
        rows.append(
            _terminal_row(
                run, cid, t, is_terminal,
                "y_finish_step", "success",
                term.y_finish_step,
            )
        )
        rows.append(
            _terminal_row(
                run, cid, t, is_terminal,
                "y_finish_seconds", "success",
                term.y_finish_seconds,
            )
        )
        rows.append(
            _terminal_row(
                run, cid, t, is_terminal,
                "y_timeout", "success",
                term.y_timeout,
            )
        )
        rows.append(
            _terminal_row(
                run, cid, t, is_terminal,
                "y_submit_without_validation", "submission",
                term.y_submit_without_validation,
            )
        )
        rows.append(
            _dynamics_row(
                run, cid, t, is_terminal,
                "y_future_progress_drop_h5", "progress_dynamics", H5,
                future_progress_drop_h5(
                    run, t, is_terminal=is_terminal, finish_step=finish_step
                ),
            )
        )
        rows.append(
            _dynamics_row(
                run, cid, t, is_terminal,
                "y_validation_new_work_h5", "validation", H5,
                validation_new_work_h5(
                    run, t, is_terminal=is_terminal, finish_step=finish_step
                ),
            )
        )
    return rows


@dataclass(frozen=True)
class SourceLabelStats:
    source_id: str
    n_runs_total: int
    n_runs_labeled: int
    n_runs_unresolvable: int
    n_runs_malformed: int

    def warn_if_empty(self) -> None:
        if self.n_runs_labeled == 0 and self.n_runs_total > 0:
            import warnings
            warnings.warn(
                f"{self.source_id}: 0 of {self.n_runs_total} runs produced "
                f"labels (unresolvable={self.n_runs_unresolvable}, "
                f"malformed={self.n_runs_malformed}). Source registry caveat?",
                stacklevel=3,
            )


def build_source_labels(
    source_id: str, run_ids: Iterable[str] | None = None
) -> tuple[pd.DataFrame, SourceLabelStats]:
    if source_id not in SOURCES:
        raise KeyError(source_id)
    if run_ids is None:
        run_ids = paths.list_run_ids(source_id)
    rows: list[LabelRow] = []
    n_total = n_unresolvable = n_malformed = n_labeled = 0
    for rid in run_ids:
        n_total += 1
        try:
            run = load_run(source_id, rid)
        except (FileNotFoundError, OSError, ValueError):
            n_malformed += 1
            continue
        if not run.events:
            n_malformed += 1
            continue
        try:
            label = load_final_label(run)
        except UnresolvableLabelError:
            n_unresolvable += 1
            continue
        n_labeled += 1
        term = terminal_labels(run, label)
        rows.extend(build_run_label_rows(run, term))
    df = pd.DataFrame([asdict(r) for r in rows])
    stats = SourceLabelStats(
        source_id=source_id,
        n_runs_total=n_total,
        n_runs_labeled=n_labeled,
        n_runs_unresolvable=n_unresolvable,
        n_runs_malformed=n_malformed,
    )
    return df, stats


def write_source_labels(source_id: str, out_dir: Path) -> tuple[Path, SourceLabelStats]:
    df, stats = build_source_labels(source_id)
    stats.warn_if_empty()
    path = write_parquet(
        df,
        out_dir / f"labels_{source_id}.parquet",
        sort_by=["source", "run_id", "checkpoint_step", "target_name"],
    )
    return path, stats


def _source_ids_or_canonical(source_ids: Iterable[str] | None) -> list[str]:
    if source_ids is None:
        return [s.source_id for s in canonical_sources()]
    ordered = list(dict.fromkeys(source_ids))
    missing = [sid for sid in ordered if sid not in SOURCES]
    if missing:
        raise KeyError(f"unknown source(s): {missing}")
    return ordered


def write_combined_labels(
    out_dir: Path,
    source_ids: Iterable[str] | None = None,
) -> tuple[Path, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for source_id in _source_ids_or_canonical(source_ids):
        df, stats = build_source_labels(source_id)
        stats.warn_if_empty()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    csv_path = write_parquet(
        df,
        out_dir / "labels_all.parquet",
        sort_by=["source", "run_id", "checkpoint_step", "target_name"],
    )
    return csv_path, df
