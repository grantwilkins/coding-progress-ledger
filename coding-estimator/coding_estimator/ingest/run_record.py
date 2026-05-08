"""RunRecord: immutable per-run handle that downstream code consumes.

Reads `ledger.jsonl` via the upstream serializer (so we inherit its
schema validation) and merges per-source side files (`run_manifest.json`,
`source_metadata.json`, `live_instrumentation.json`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ledger_progress.core import LedgerEvent
from ledger_progress.serialization import load_events_jsonl

from coding_estimator.ingest.paths import run_dir
from coding_estimator.ingest.sources import SOURCES


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    source: str
    ledger_path: Path
    events: tuple[LedgerEvent, ...]
    has_real_wallclock: bool
    start_wall_time: datetime | None
    end_wall_time: datetime | None
    task_id: str | None
    task_family: str | None
    arm: str | None
    difficulty: str | None
    agent_scaffold: str | None
    model_name: str | None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(text)


def _events_sorted(events: list[LedgerEvent]) -> tuple[LedgerEvent, ...]:
    # Python's sorted is stable: ties on step preserve file order.
    return tuple(sorted(events, key=lambda e: e.step))


def load_run(source_id: str, run_id: str) -> RunRecord:
    if source_id not in SOURCES:
        raise KeyError(f"unknown source: {source_id}")
    rd = run_dir(source_id, run_id)
    ledger_path = rd / "ledger.jsonl"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"ledger.jsonl missing: {ledger_path}")
    raw_events = load_events_jsonl(str(ledger_path))
    events = _events_sorted(raw_events)

    instr = _read_json(rd / "live_instrumentation.json")
    src_meta = _read_json(rd / "source_metadata.json")
    manifest = _read_json(rd / "run_manifest.json")

    timestamp_source = instr.get("timestamp_source")
    manifest_has_real_wallclock = manifest.get("has_real_wallclock") is True
    has_real_wallclock = SOURCES[source_id].timestamp_quality == "real" and (
        timestamp_source == "wallclock" or manifest_has_real_wallclock
    )

    timestamps_present = [e.timestamp for e in events if e.timestamp]
    if has_real_wallclock and timestamps_present:
        start_wall_time = _parse_iso(timestamps_present[0])
        end_wall_time = _parse_iso(timestamps_present[-1])
    else:
        start_wall_time = None
        end_wall_time = None

    task_id = (
        instr.get("task_id")
        or manifest.get("task_id")
        or src_meta.get("instance_id")
        or run_id
    )
    task_family = (
        src_meta.get("category")
        or src_meta.get("subcategory")
        or manifest.get("category")
        or manifest.get("target_shape")
    )
    arm = manifest.get("arm")
    difficulty = src_meta.get("difficulty") or manifest.get("difficulty")
    model_name = src_meta.get("model_name") or manifest.get("model_name")
    agent_scaffold = (
        src_meta.get("source")
        or manifest.get("subagent_type")
        or manifest.get("created_by")
    )

    raw = {
        "run_manifest": manifest,
        "source_metadata": src_meta,
        "live_instrumentation": instr,
    }
    return RunRecord(
        run_id=run_id,
        source=source_id,
        ledger_path=ledger_path,
        events=events,
        has_real_wallclock=has_real_wallclock,
        start_wall_time=start_wall_time,
        end_wall_time=end_wall_time,
        task_id=str(task_id) if task_id is not None else None,
        task_family=str(task_family) if task_family is not None else None,
        arm=str(arm) if arm is not None else None,
        difficulty=str(difficulty) if difficulty is not None else None,
        agent_scaffold=str(agent_scaffold) if agent_scaffold is not None else None,
        model_name=str(model_name) if model_name is not None else None,
        raw_metadata=raw,
    )
