"""Source registry: enumerate every upstream source and lock the canonical
choices for v0. Per-source caveats live here so adapters and reports can
read a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TimestampQuality = Literal["real", "synthetic", "synthetic_backfill", "none"]
LabelFieldPath = Literal[
    "source_metadata.final_success",
    "live_instrumentation.verifier_pass",
    "summary_by_category.final_success",
    "run_manifest.final_success",
    "unresolvable",
]


@dataclass(frozen=True)
class Source:
    source_id: str
    runs_dir: str  # relative to ledger_root
    timestamp_quality: TimestampQuality
    label_field_path: LabelFieldPath
    schema_version: str
    canonical_for_v0: bool
    known_caveats: tuple[str, ...]
    protocol_doc: str | None = None


SOURCES: dict[str, Source] = {
    "swe_agent_pilot": Source(
        source_id="swe_agent_pilot",
        runs_dir="runs/swe_agent_pilot",
        timestamp_quality="none",
        label_field_path="source_metadata.final_success",
        schema_version="0.1.0",
        canonical_for_v0=True,
        known_caveats=(
            "retrospective annotation: events were tagged knowing run outcome",
            "no event timestamps; step-only ordering",
        ),
        protocol_doc="../coding-progress-ledger/docs",
    ),
    "swe_agent_pilot_v3": Source(
        source_id="swe_agent_pilot_v3",
        runs_dir="runs/swe_agent_pilot_v3",
        timestamp_quality="none",
        label_field_path="source_metadata.final_success",
        schema_version="0.1.0",
        canonical_for_v0=False,
        known_caveats=(
            "revised retrospective protocol; reserved for parity comparisons (Workstream L)",
        ),
    ),
    "swe_agent_live": Source(
        source_id="swe_agent_live",
        runs_dir="runs/swe_agent_live",
        timestamp_quality="synthetic",
        label_field_path="source_metadata.final_success",
        schema_version="0.1.0",
        canonical_for_v0=False,
        known_caveats=("synthetic timestamps via sidecar replay; reserved for sidecar tests",),
    ),
    "swe_agent_live_wallclock": Source(
        source_id="swe_agent_live_wallclock",
        runs_dir="runs/swe_agent_live_wallclock",
        timestamp_quality="synthetic_backfill",
        label_field_path="source_metadata.final_success",
        schema_version="0.1.0",
        canonical_for_v0=False,
        known_caveats=(
            "wallclock back-filled per upstream WORKSTREAM_N_TB_PLAN.md; "
            "do NOT mix into headline pools",
        ),
    ),
    "hermes_pilot": Source(
        source_id="hermes_pilot",
        runs_dir="runs/hermes_pilot",
        timestamp_quality="synthetic",
        label_field_path="source_metadata.final_success",
        schema_version="0.1.0",
        canonical_for_v0=False,
        known_caveats=("retrospective LLM annotation; superseded by hermes_pilot_h5_v2",),
    ),
    "hermes_pilot_h5": Source(
        source_id="hermes_pilot_h5",
        runs_dir="runs/hermes_pilot_h5",
        timestamp_quality="synthetic",
        label_field_path="source_metadata.final_success",
        schema_version="0.1.0",
        canonical_for_v0=False,
        known_caveats=("intermediate Hermes annotation; superseded by hermes_pilot_h5_v2",),
    ),
    "hermes_pilot_h5_v2": Source(
        source_id="hermes_pilot_h5_v2",
        runs_dir="runs/hermes_pilot_h5_v2",
        timestamp_quality="synthetic",
        label_field_path="source_metadata.final_success",
        schema_version="0.1.0",
        canonical_for_v0=True,
        known_caveats=(
            "retrospective LLM annotation: outcome-aware event tagging is unfixable here",
            "many runs have source_metadata.final_success == null and must be skipped",
        ),
    ),
    "tb_live": Source(
        source_id="tb_live",
        runs_dir="runs/tb_live",
        timestamp_quality="real",
        label_field_path="live_instrumentation.verifier_pass",
        schema_version="0.1.0",
        canonical_for_v0=True,
        known_caveats=(
            "12 first-party live runs; verifier_pass is the canonical success signal",
            "summary_by_category.final_success is null on every run; do NOT use it",
        ),
    ),
}


def canonical_sources() -> list[Source]:
    return [s for s in SOURCES.values() if s.canonical_for_v0]


def source(source_id: str) -> Source:
    if source_id not in SOURCES:
        raise KeyError(f"unknown source: {source_id}")
    return SOURCES[source_id]
