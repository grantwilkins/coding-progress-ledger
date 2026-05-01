#!/usr/bin/env python3
"""Stream a normalized SWE-agent trace through the live ledger sidecar.

This is the N3 portability hook: SWE-agent-shaped steps become the N1
wire format, then the sidecar owns ledger generation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress.sidecar import LedgerSidecar


SCHEMA_VERSION = "1.0"
COPY_ARTIFACTS = (
    "task.md",
    "source_trace.json",
    "normalized_trace.json",
    "trajectory_summary.md",
    "final_diff.patch",
    "test_output.txt",
    "source_metadata.json",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    clock = _build_synthetic_clock(args.synthetic_clock_start, args.synthetic_step_seconds)
    try:
        materialize_live_run(
            args.source_run_dir,
            args.output_run_dir,
            args.run_id,
            clock=clock,
            timestamp_source="synthetic" if clock is not None else "replay",
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def materialize_live_run(
    source_run_dir: Path,
    output_run_dir: Path,
    run_id: str | None = None,
    clock: Callable[[], str] | None = None,
    timestamp_source: str = "replay",
) -> list[dict[str, Any]]:
    source_run_dir = Path(source_run_dir)
    output_run_dir = Path(output_run_dir)
    if (output_run_dir / "ledger.jsonl").exists():
        raise FileExistsError(f"{output_run_dir / 'ledger.jsonl'} already exists")

    normalized = _load_normalized(source_run_dir)
    run_id = run_id or normalized["instance_id"]
    clock = clock or _utc_now
    events = list(wire_events(normalized, run_id, clock))
    if not events:
        raise ValueError("normalized trace produced no agent steps")

    output_run_dir.mkdir(parents=True, exist_ok=True)
    _copy_context(source_run_dir, output_run_dir)
    _write_jsonl(output_run_dir / "wire_events.jsonl", events)
    _write_live_metadata(source_run_dir, output_run_dir, normalized, run_id, events, timestamp_source)

    sidecar = LedgerSidecar(output_run_dir, "swe_agent", root_task=_root_task(source_run_dir, normalized))
    for event in events:
        sidecar.process_event(event)
    return events


def wire_events(
    normalized: dict[str, Any],
    run_id: str,
    clock: Callable[[], str],
) -> Iterable[dict[str, Any]]:
    events = normalized.get("events")
    if not isinstance(events, list):
        raise ValueError("normalized trace missing events list")
    assistant_indices = [
        i for i, event in enumerate(events)
        if event.get("role") == "assistant" and (event.get("command") or event.get("action") or event.get("tool_name"))
    ]
    for position, index in enumerate(assistant_indices):
        event = events[index]
        tool_event = events[index + 1] if index + 1 < len(events) and events[index + 1].get("role") == "tool" else {}
        is_last = position == len(assistant_indices) - 1
        yield {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "step": event["step_index"],
            "timestamp": clock(),
            "agent_step": {
                "thought": event.get("thought"),
                "action": event.get("action"),
                "command": event.get("command"),
                "files_touched": event.get("files_touched") or [],
                "observation": tool_event.get("observation"),
                "exit_status": normalized.get("exit_status") if is_last else None,
                "tool_name": event.get("tool_name"),
            },
            "ledger_ops": [],
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a normalized SWE-agent run through the live sidecar.")
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--output-run-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--synthetic-clock-start",
        help="ISO-8601 timestamp to seed a synthetic per-step clock (e.g. 2026-05-01T00:00:00+00:00).",
    )
    parser.add_argument(
        "--synthetic-step-seconds",
        type=float,
        default=30.0,
        help="Seconds advanced per wire event when --synthetic-clock-start is set (default 30).",
    )
    return parser.parse_args(argv)


def _build_synthetic_clock(start: str | None, step_seconds: float) -> Callable[[], str] | None:
    if start is None:
        return None
    if step_seconds <= 0:
        raise ValueError("--synthetic-step-seconds must be positive")
    base = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    delta = timedelta(seconds=step_seconds)
    state = {"t": base - delta}

    def clock() -> str:
        state["t"] = state["t"] + delta
        return state["t"].isoformat()

    return clock


def _load_normalized(source_run_dir: Path) -> dict[str, Any]:
    path = source_run_dir / "normalized_trace.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is required")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("instance_id"), str) or not data["instance_id"]:
        raise ValueError("normalized trace missing instance_id")
    return data


def _copy_context(source_run_dir: Path, output_run_dir: Path) -> None:
    for name in COPY_ARTIFACTS:
        source = source_run_dir / name
        if source.exists():
            shutil.copyfile(source, output_run_dir / name)
    run_notes = output_run_dir / "run_notes.md"
    if not run_notes.exists():
        run_notes.write_text("# Run Notes\n\nLive sidecar run from normalized SWE-agent steps.\n", encoding="utf-8")


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n" for event in events), encoding="utf-8")


def _write_live_metadata(
    source_run_dir: Path,
    output_run_dir: Path,
    normalized: dict[str, Any],
    run_id: str,
    events: list[dict[str, Any]],
    timestamp_source: str,
) -> None:
    timestamps = [event["timestamp"] for event in events]
    span_seconds = (
        (datetime.fromisoformat(timestamps[-1]) - datetime.fromisoformat(timestamps[0])).total_seconds()
        if len(timestamps) >= 2 else 0.0
    )
    metadata = {
        "mode": "normalized_trace_live_sidecar_replay",
        "source_run_dir": str(source_run_dir),
        "run_id": run_id,
        "instance_id": normalized["instance_id"],
        "final_success": normalized.get("final_success"),
        "generated_at": _utc_now(),
        "wire_event_count": len(events),
        "timestamp_source": timestamp_source,
        "first_event_timestamp": timestamps[0] if timestamps else None,
        "last_event_timestamp": timestamps[-1] if timestamps else None,
        "timestamp_span_seconds": span_seconds,
    }
    (output_run_dir / "live_instrumentation.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _root_task(source_run_dir: Path, normalized: dict[str, Any]) -> str:
    task = source_run_dir / "task.md"
    if task.exists():
        return task.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    return normalized["instance_id"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
