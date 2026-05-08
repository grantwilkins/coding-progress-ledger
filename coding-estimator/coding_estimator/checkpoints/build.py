"""Checkpoint dataset builder.

For one source, walk every canonical run, run prefix_replay at every
P_step checkpoint, run each feature builder, and assemble a single
parquet frame with identity + feature columns.

Calls `assert_no_forbidden` immediately before writing (AGENTS.md
invariant 1: forbidden-column guard at every choke point).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.features import (
    closure,
    discovery,
    evidence,
    frontier,
    instability,
    observation,
    stalling,
    time_budget,
    validation,
)
from coding_estimator.checkpoints.features.registry import all_features
from coding_estimator.checkpoints.policy import p_step_checkpoints
from coding_estimator.checkpoints.replay import prefix_replay
from coding_estimator.ingest.run_record import RunRecord, load_run
from coding_estimator.ingest.sources import SOURCES, canonical_sources
from coding_estimator.io import write_parquet
from coding_estimator.leakage.guard import assert_no_forbidden
from coding_estimator.leakage.run_constancy import is_run_constant

SCHEMA_VERSION = "0.1.0"
DEFAULT_SOURCE_PROTOCOL_VERSION = "v1"


def _builder_commit_sha() -> str:
    """Resolve the current coding-estimator HEAD. Falls back to a
    placeholder if we are not in a git checkout (e.g. CI tests in
    a fresh tmpdir)."""
    here = Path(__file__).resolve()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=here.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()[:7]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0000000"


def _max_observation_step_used(run: RunRecord, checkpoint_step: int) -> int:
    path = run.ledger_path.parent / "observation_events.jsonl"
    if not path.is_file():
        return 0
    max_step = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        step = int((json.loads(line)).get("step", 0))
        if step <= checkpoint_step:
            max_step = max(max_step, step)
    return max_step


def build_run_rows(run: RunRecord) -> list[dict]:
    """Build all checkpoint rows for a single run."""
    src = SOURCES[run.source]
    sha = _builder_commit_sha()
    rows: list[dict] = []
    steps = p_step_checkpoints(run)
    terminal = steps[-1]
    for t in steps:
        state = prefix_replay(run, t)
        max_ledger_step = max((event.step for event in state.events_so_far), default=0)
        identity = {
            "run_id": run.run_id,
            "source": run.source,
            "checkpoint_id": f"{run.run_id}::{t}",
            "checkpoint_step": t,
            "max_ledger_step_used": max_ledger_step,
            "max_observation_step_used": _max_observation_step_used(run, t),
            "checkpoint_event_index": len(state.events_so_far) - 1
            if state.events_so_far
            else 0,
            "is_terminal_checkpoint": (t == terminal),
            "timestamp_quality": src.timestamp_quality,
            "ledger_path": str(run.ledger_path),
            "schema_version": SCHEMA_VERSION,
            "builder_commit_sha": sha,
            "source_protocol_version": DEFAULT_SOURCE_PROTOCOL_VERSION,
            "task_id": run.task_id,
            "task_family": run.task_family,
            "arm": run.arm,
            "difficulty": run.difficulty,
            "agent_scaffold": run.agent_scaffold,
            "model_name": run.model_name,
        }
        # checkpoint_wall_time + checkpoint_elapsed_seconds populated only on
        # real-wallclock runs.
        if run.has_real_wallclock and state.events_so_far:
            identity["checkpoint_wall_time"] = state.events_so_far[-1].timestamp
        else:
            identity["checkpoint_wall_time"] = None

        feats: dict = {}
        feats.update(frontier.compute(state))
        feats.update(closure.compute(state))
        feats.update(discovery.compute(state))
        feats.update(instability.compute(state, run))
        feats.update(stalling.compute(state))
        feats.update(validation.compute(state))
        feats.update(evidence.compute(state))
        feats.update(time_budget.compute(state, run))
        feats.update(observation.compute(t, run))

        # Bring elapsed_wall_time forward into the identity slot too so
        # downstream consumers do not have to reach into features.
        identity["checkpoint_elapsed_seconds"] = feats.get("elapsed_wall_time")
        identity["checkpoint_fraction_timeout"] = feats.get("fraction_timeout_consumed")

        rows.append({**identity, **feats})
    return rows


def build_source_frame(source_id: str, run_ids: list[str] | None = None) -> pd.DataFrame:
    """Build the full checkpoint frame for one source, optionally
    filtered to a subset of run_ids."""
    if source_id not in SOURCES:
        raise KeyError(source_id)
    if run_ids is None:
        from coding_estimator.ingest.paths import list_run_ids
        run_ids = list_run_ids(source_id)
    rows: list[dict] = []
    for rid in run_ids:
        try:
            run = load_run(source_id, rid)
        except (FileNotFoundError, OSError, ValueError):
            # Per § 0.9: skip-and-record. The combined manifest is
            # the place that records the skip; the checkpoint frame
            # simply omits the run.
            continue
        if not run.events:
            continue
        rows.extend(build_run_rows(run))
    df = pd.DataFrame(rows)
    return df


def apply_canonical_fills(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-feature canonical fills from the registry.

    Producer-side, called from `write_source_checkpoints`. Without this,
    every consumer of the parquet has to remember to fill (sklearn
    refuses NaN; the harness, plot scripts, and future model trainers
    all hit it). Per AGENTS.md invariant 7, the missingness contract
    says count-features at "applicable absent so far" SHOULD be 0, not
    null — so applying the fill brings the stored frame in line with
    the contract.

    Cells that remain null after this call are exactly the
    `unknown_due_to_missing_artifact` and `not_applicable_to_source`
    cases — the genuinely-missing ones. The F11 missingness audit
    already exempts the former; the latter is structural.
    """
    if "source" not in df.columns or df.empty:
        return df
    out = df.copy()
    for source_id, idx in out.groupby("source").groups.items():
        for f in all_features():
            if f.column_name not in out.columns:
                continue
            fill = f.canonical_fill_for(str(source_id))
            if fill is None:
                continue
            col = out.loc[idx, f.column_name]
            out.loc[idx, f.column_name] = col.fillna(fill)
    return out


def _source_ids_or_canonical(source_ids: list[str] | tuple[str, ...] | None) -> list[str]:
    if source_ids is None:
        return [s.source_id for s in canonical_sources()]
    missing = [sid for sid in source_ids if sid not in SOURCES]
    if missing:
        raise KeyError(f"unknown source(s): {missing}")
    return list(dict.fromkeys(source_ids))


def write_source_checkpoints(
    source_id: str,
    out_path: Path,
    run_ids: list[str] | None = None,
) -> tuple[Path, pd.DataFrame]:
    df = build_source_frame(source_id, run_ids=run_ids)
    df = apply_canonical_fills(df)
    # Defense in depth: the forbidden-column guard fires here, NOT just
    # at schema-load time. Any future regression that joins terminal
    # leakage into the frame will hit this.
    assert_no_forbidden(df)
    # Run-constancy audit: warn (do not fail) on (run-constant feature,
    # run-constant target) pairs. Targets aren't joined here, but if a
    # future caller passes a frame with terminal labels, this will
    # surface them.
    _maybe_warn_on_run_constancy(df)
    csv_path = write_parquet(
        df,
        out_path,
        sort_by=["source", "run_id", "checkpoint_step"],
    )
    return csv_path, df


def write_combined_checkpoints(
    out_path: Path,
    source_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Build and write the combined checkpoint frame.

    Defaults to the canonical v0 sources. Callers may pass an explicit
    source list to build training artifacts that include non-canonical
    live corpora such as `tb_live_v2`.
    """
    frames = [
        frame
        for source_id in _source_ids_or_canonical(source_ids)
        if not (frame := build_source_frame(source_id)).empty
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = apply_canonical_fills(df)
    assert_no_forbidden(df)
    _maybe_warn_on_run_constancy(df)
    path = write_parquet(
        df,
        out_path,
        sort_by=["source", "run_id", "checkpoint_step"],
    )
    return path, df


def _maybe_warn_on_run_constancy(df: pd.DataFrame) -> None:
    """Best-effort sanity: if any column name starts with 'y_' AND is
    run-constant, AND any feature column is run-constant, emit a hard
    error. We expect callers to keep labels OUT of the checkpoint frame
    in v0; this guard catches accidents."""
    if "run_id" not in df.columns:
        return
    label_cols = [c for c in df.columns if c.startswith("y_")]
    if not label_cols:
        return
    rc_labels = [c for c in label_cols if is_run_constant(df, c)]
    if not rc_labels:
        return
    raise ValueError(
        f"checkpoint frame contains run-constant label columns {rc_labels}; "
        "labels must be joined LATER (Workstream E), not at checkpoint build time"
    )
