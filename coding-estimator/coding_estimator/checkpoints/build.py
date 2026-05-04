"""Checkpoint dataset builder.

For one source, walk every canonical run, run prefix_replay at every
P_step checkpoint, run each feature builder, and assemble a single
parquet frame with identity + feature columns.

Calls `assert_no_forbidden` immediately before writing (AGENTS.md
invariant 1: forbidden-column guard at every choke point).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.features import (
    closure,
    discovery,
    evidence,
    frontier,
    instability,
    stalling,
    time_budget,
    validation,
)
from coding_estimator.checkpoints.policy import p_step_checkpoints
from coding_estimator.checkpoints.replay import prefix_replay
from coding_estimator.ingest.run_record import RunRecord, load_run
from coding_estimator.ingest.sources import SOURCES
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


def build_run_rows(run: RunRecord) -> list[dict]:
    """Build all checkpoint rows for a single run."""
    src = SOURCES[run.source]
    sha = _builder_commit_sha()
    rows: list[dict] = []
    steps = p_step_checkpoints(run)
    terminal = steps[-1]
    for t in steps:
        state = prefix_replay(run, t)
        identity = {
            "run_id": run.run_id,
            "source": run.source,
            "checkpoint_id": f"{run.run_id}::{t}",
            "checkpoint_step": t,
            "checkpoint_event_index": len(state.events_so_far) - 1
            if state.events_so_far
            else 0,
            "is_terminal_checkpoint": (t == terminal),
            "timestamp_quality": src.timestamp_quality,
            "ledger_path": str(run.ledger_path),
            "schema_version": SCHEMA_VERSION,
            "builder_commit_sha": sha,
            "source_protocol_version": DEFAULT_SOURCE_PROTOCOL_VERSION,
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


def write_source_checkpoints(
    source_id: str,
    out_path: Path,
    run_ids: list[str] | None = None,
) -> tuple[Path, pd.DataFrame]:
    df = build_source_frame(source_id, run_ids=run_ids)
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
