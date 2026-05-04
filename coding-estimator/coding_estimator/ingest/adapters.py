"""Per-source ingestion adapters.

Each canonical source has its own adapter that lists run_ids
deterministically, loads RunRecord per run, attaches a final label (or
records 'unresolvable' as the skip reason), and emits one manifest row
per run -- including malformed runs, so § 0.9 compliance is auditable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from coding_estimator.ingest import paths
from coding_estimator.ingest.labels import (
    FinalLabel,
    UnresolvableLabelError,
    load_final_label,
)
from coding_estimator.ingest.run_record import load_run
from coding_estimator.ingest.sources import SOURCES, canonical_sources
from coding_estimator.io import write_csv


@dataclass(frozen=True)
class RunManifestRow:
    run_id: str
    source: str
    ledger_path: str
    ledger_event_count: int
    has_real_wallclock: bool
    start_wall_time: str | None
    end_wall_time: str | None
    task_id: str | None
    task_family: str | None
    agent_scaffold: str | None
    model_name: str | None
    final_success: bool | None
    final_success_source: str
    timeout: bool
    finish_step: int | None
    finish_seconds: float | None
    notes: str


def _row_for_run(source_id: str, run_id: str) -> RunManifestRow:
    """Build one manifest row, including 'unresolvable_label' or
    'malformed_run' as notes when the run cannot be processed."""
    try:
        run = load_run(source_id, run_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return RunManifestRow(
            run_id=run_id,
            source=source_id,
            ledger_path="",
            ledger_event_count=0,
            has_real_wallclock=False,
            start_wall_time=None,
            end_wall_time=None,
            task_id=None,
            task_family=None,
            agent_scaffold=None,
            model_name=None,
            final_success=None,
            final_success_source="missing",
            timeout=False,
            finish_step=None,
            finish_seconds=None,
            notes=f"malformed_run: {type(exc).__name__}: {exc}",
        )
    label: FinalLabel | None
    note = ""
    try:
        label = load_final_label(run)
    except UnresolvableLabelError as exc:
        label = None
        note = f"unresolvable_label: {exc}"
    return RunManifestRow(
        run_id=run.run_id,
        source=run.source,
        ledger_path=str(run.ledger_path),
        ledger_event_count=len(run.events),
        has_real_wallclock=run.has_real_wallclock,
        start_wall_time=run.start_wall_time.isoformat() if run.start_wall_time else None,
        end_wall_time=run.end_wall_time.isoformat() if run.end_wall_time else None,
        task_id=run.task_id,
        task_family=run.task_family,
        agent_scaffold=run.agent_scaffold,
        model_name=run.model_name,
        final_success=label.final_success if label else None,
        final_success_source=label.final_success_source if label else "missing",
        timeout=label.timeout if label else False,
        finish_step=label.finish_step if label else None,
        finish_seconds=label.finish_seconds if label else None,
        notes=note,
    )


def ingest_source(source_id: str) -> list[RunManifestRow]:
    if source_id not in SOURCES:
        raise KeyError(source_id)
    run_ids = paths.list_run_ids(source_id)
    return [_row_for_run(source_id, rid) for rid in run_ids]


def write_source_manifest(source_id: str, out_dir: Path) -> tuple[Path, list[RunManifestRow]]:
    rows = ingest_source(source_id)
    df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = write_csv(
        df,
        out_dir / f"{source_id}.csv",
        sort_by=["source", "run_id"],
    )
    return csv_path, rows


def ingest_canonical_sources(out_dir: Path) -> dict[str, list[RunManifestRow]]:
    """Ingest the three canonical v0 sources and write per-source manifests."""
    out: dict[str, list[RunManifestRow]] = {}
    for s in canonical_sources():
        _, rows = write_source_manifest(s.source_id, out_dir)
        out[s.source_id] = rows
    return out


def to_frame(rows: Iterable[RunManifestRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


FINAL_SUCCESS_SOURCE_ENUM = frozenset(
    {"verifier_exit", "swe_agent_target", "hermes_resolved", "manual", "missing"}
)


def write_combined_manifest(out_dir: Path) -> tuple[Path, pd.DataFrame]:
    """Write datasets/manifests/all_runs.csv covering every canonical
    source. Validates the final_success_source enum on the way out."""
    by_source = ingest_canonical_sources(out_dir)
    all_rows: list[RunManifestRow] = []
    for rows in by_source.values():
        all_rows.extend(rows)
    df = to_frame(all_rows)
    bad = set(df["final_success_source"].unique()) - FINAL_SUCCESS_SOURCE_ENUM
    if bad:
        raise ValueError(f"final_success_source values not in enum: {bad}")
    csv_path = write_csv(
        df,
        out_dir / "all_runs.csv",
        sort_by=["source", "run_id"],
    )
    return csv_path, df
