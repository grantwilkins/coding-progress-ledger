from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger_progress import CODING_CATEGORIES, EventType, Status, SubtaskCategory, replay, score
from ledger_progress.run_manager import resolve_final_success, stale_summary_warning
from ledger_progress.serialization import event_from_dict

import importlib.util

"""
Build event-level and step-level ledger observation datasets.

Event rows preserve replay fidelity: one row per LedgerEvent prefix.
Step rows are derived from event rows by retaining the final event state for
each (run_id, step), then recomputing deltas across retained step rows.

Step-level drop sources use a v0 attribution fallback: for a negative step
delta, material negative event-level drops between the previous retained step
and the current retained step are aggregated by drop source. A single material
source is reported directly; multiple material sources report "mixed"; if no
negative event-level row exists despite a negative step-level delta, "mixed" is
reported. Non-negative step deltas always report "none".
"""


RESCORE_PATH = ROOT / "scripts" / "rescore_suite_by_category.py"
spec = importlib.util.spec_from_file_location("rescore_suite_by_category", RESCORE_PATH)
rescore = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rescore)


ALL_CATEGORIES = tuple(SubtaskCategory)
EPSILON = 1e-12
DATASET_FIELDS = [
    "run_id",
    "step",
    "event_index",
    "event_type",
    "subtask_id",
    "coding_progress",
    "overall_progress",
    "active_coding_weight",
    "completed_coding_weight",
    "active_overall_weight",
    "completed_overall_weight",
    "active_coding_leaves",
    "completed_coding_leaves",
    "active_overall_leaves",
    "completed_overall_leaves",
    "num_splits_so_far",
    "num_reopens_so_far",
    "num_invalidations_so_far",
    "delta_coding_progress",
    "delta_overall_progress",
    "coding_drop_source",
    "overall_drop_source",
    "final_success",
    "final_success_source",
    "native_coding_progress",
    "native_overall_progress",
    "native_active_coding_weight",
    "native_completed_coding_weight",
    "native_active_overall_weight",
    "native_completed_overall_weight",
    "native_active_coding_leaves",
    "native_completed_coding_leaves",
    "native_active_overall_leaves",
    "native_completed_overall_leaves",
    "native_delta_coding_progress",
    "native_delta_overall_progress",
    "native_coding_drop_source",
    "native_overall_drop_source",
    "category_resolution_mode",
    "category_overrides_applied",
    "product_progress",
    "validation_progress",
    "investigation_progress",
    "step_added_subtasks",
    "step_split_events",
    "step_reopen_events",
    "step_invalidation_events",
    "step_product_completes",
    "step_validation_completes",
    "step_investigation_completes",
    "step_strong_completions",
    "step_manual_only_completions",
    "steps_since_progress_increase",
    "steps_since_completion",
    "steps_since_subtask_added",
    "cum_strong_completions",
    "cum_manual_only_completions",
    "elapsed_seconds",
    "seconds_since_last_event",
    "seconds_since_progress_increase",
    "events_per_minute",
]
EVENTS_PER_MINUTE_WINDOW = 5


@dataclass
class BuildResult:
    event_rows: list[dict[str, Any]]
    step_rows: list[dict[str, Any]]
    event_summaries: list[dict[str, Any]]
    step_summaries: list[dict[str, Any]]
    warnings: list[str]

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.event_rows

    @property
    def run_summaries(self) -> list[dict[str, Any]]:
        return self.event_summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build checkpoint-level observations from ledger runs.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-csv", default="datasets/ledger_observations_v0.csv", help="Backward-compatible event-level CSV alias.")
    parser.add_argument("--output-event-csv", default="datasets/ledger_observations_v0_event.csv")
    parser.add_argument("--output-step-csv", default="datasets/ledger_observations_v0_step.csv")
    parser.add_argument("--summary-md", default="datasets/ledger_observations_v0_summary.md")
    args = parser.parse_args(argv)

    result = build_dataset(Path(args.runs_dir))
    write_dataset_csv(Path(args.output_event_csv), result.event_rows)
    write_dataset_csv(Path(args.output_step_csv), result.step_rows)
    write_dataset_csv(Path(args.output_csv), result.event_rows)
    write_summary_md(Path(args.summary_md), result)
    return 0


def build_dataset(runs_dir: Path) -> BuildResult:
    ledger_paths = sorted(runs_dir.glob("**/ledger.jsonl"), key=lambda path: _run_id(runs_dir, path))
    event_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not ledger_paths:
        warnings.append(f"no ledger.jsonl files found under {runs_dir}")

    for ledger_path in ledger_paths:
        run_dir = ledger_path.parent
        run_id = _run_id(runs_dir, ledger_path)
        before_hash = _sha256(ledger_path)
        raw_events = _load_raw_events(ledger_path)
        summary_json, summary_warnings = _load_summary(run_dir, run_id)
        warnings.extend(summary_warnings)
        resolved_raw_events, resolution = resolve_categories(run_id, raw_events, summary_json)
        native_events = [event_from_dict(event) for event in raw_events]
        resolved_events = [event_from_dict(event) for event in resolved_raw_events]
        final_success, final_success_source = resolve_final_success(run_dir, summary_json)
        if final_success is None:
            warnings.append(f"{run_id}: final_success is unknown")
        stale_warning = stale_summary_warning(run_dir, summary_json)
        if stale_warning:
            warnings.append(f"{run_id}: {stale_warning}")

        run_rows = build_run_rows(
            run_id=run_id,
            native_events=native_events,
            resolved_events=resolved_events,
            final_success=final_success,
            final_success_source=final_success_source,
            category_resolution_mode=resolution["mode"],
            category_overrides_applied=resolution["overrides_applied"],
        )
        event_timestamps = {
            index: getattr(event, "timestamp", None)
            for index, event in enumerate(resolved_events)
        }
        run_step_rows = build_step_rows(run_rows, _event_context(resolved_events), event_timestamps)
        event_rows.extend(run_rows)
        step_rows.extend(run_step_rows)
        event_summaries.append(summarize_run(run_id, run_rows, final_success, resolution))
        step_summaries.append(summarize_run(run_id, run_step_rows, final_success, resolution))

        after_hash = _sha256(ledger_path)
        if after_hash != before_hash:
            raise RuntimeError(f"builder mutated {ledger_path}")

    return BuildResult(
        event_rows=event_rows,
        step_rows=step_rows,
        event_summaries=event_summaries,
        step_summaries=step_summaries,
        warnings=warnings,
    )


def build_run_rows(
    *,
    run_id: str,
    native_events: list[Any],
    resolved_events: list[Any],
    final_success: bool | None,
    final_success_source: str,
    category_resolution_mode: str,
    category_overrides_applied: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_resolved: dict[str, Any] | None = None
    previous_native: dict[str, Any] | None = None
    counters = {
        "num_splits_so_far": 0,
        "num_reopens_so_far": 0,
        "num_invalidations_so_far": 0,
    }
    first_ts: datetime | None = None
    prev_ts: datetime | None = None
    last_increase_ts: datetime | None = None
    rolling_window: deque[datetime] = deque(maxlen=EVENTS_PER_MINUTE_WINDOW)
    for index, (native_event, resolved_event) in enumerate(zip(native_events, resolved_events)):
        _update_counters(counters, resolved_event.event_type)
        native_metrics = _metrics(native_events[: index + 1])
        resolved_metrics = _metrics(resolved_events[: index + 1])
        delta_coding = _delta(previous_resolved, resolved_metrics, "coding_progress")
        delta_overall = _delta(previous_resolved, resolved_metrics, "overall_progress")
        native_delta_coding = _delta(previous_native, native_metrics, "coding_progress")
        native_delta_overall = _delta(previous_native, native_metrics, "overall_progress")
        row = {
            "run_id": run_id,
            "step": resolved_event.step,
            "event_index": index,
            "event_type": resolved_event.event_type.value,
            "subtask_id": resolved_event.subtask_id or "",
            "coding_progress": resolved_metrics["coding_progress"],
            "overall_progress": resolved_metrics["overall_progress"],
            "active_coding_weight": resolved_metrics["active_coding_weight"],
            "completed_coding_weight": resolved_metrics["completed_coding_weight"],
            "active_overall_weight": resolved_metrics["active_overall_weight"],
            "completed_overall_weight": resolved_metrics["completed_overall_weight"],
            "active_coding_leaves": resolved_metrics["active_coding_leaves"],
            "completed_coding_leaves": resolved_metrics["completed_coding_leaves"],
            "active_overall_leaves": resolved_metrics["active_overall_leaves"],
            "completed_overall_leaves": resolved_metrics["completed_overall_leaves"],
            **counters,
            "delta_coding_progress": delta_coding,
            "delta_overall_progress": delta_overall,
            "coding_drop_source": _drop_source(previous_resolved, resolved_metrics, "coding_progress", delta_coding),
            "overall_drop_source": _drop_source(previous_resolved, resolved_metrics, "overall_progress", delta_overall),
            "final_success": _csv_bool(final_success),
            "final_success_source": final_success_source if final_success is not None else "unknown",
            "native_coding_progress": native_metrics["coding_progress"],
            "native_overall_progress": native_metrics["overall_progress"],
            "native_active_coding_weight": native_metrics["active_coding_weight"],
            "native_completed_coding_weight": native_metrics["completed_coding_weight"],
            "native_active_overall_weight": native_metrics["active_overall_weight"],
            "native_completed_overall_weight": native_metrics["completed_overall_weight"],
            "native_active_coding_leaves": native_metrics["active_coding_leaves"],
            "native_completed_coding_leaves": native_metrics["completed_coding_leaves"],
            "native_active_overall_leaves": native_metrics["active_overall_leaves"],
            "native_completed_overall_leaves": native_metrics["completed_overall_leaves"],
            "native_delta_coding_progress": native_delta_coding,
            "native_delta_overall_progress": native_delta_overall,
            "native_coding_drop_source": _drop_source(previous_native, native_metrics, "coding_progress", native_delta_coding),
            "native_overall_drop_source": _drop_source(previous_native, native_metrics, "overall_progress", native_delta_overall),
            "category_resolution_mode": category_resolution_mode,
            "category_overrides_applied": category_overrides_applied,
            "product_progress": resolved_metrics["product_progress"],
            "validation_progress": resolved_metrics["validation_progress"],
            "investigation_progress": resolved_metrics["investigation_progress"],
            "step_added_subtasks": 0,
            "step_split_events": 0,
            "step_reopen_events": 0,
            "step_invalidation_events": 0,
            "step_product_completes": 0,
            "step_validation_completes": 0,
            "step_investigation_completes": 0,
            "step_strong_completions": 0,
            "step_manual_only_completions": 0,
            "steps_since_progress_increase": 0,
            "steps_since_completion": 0,
            "steps_since_subtask_added": 0,
            "cum_strong_completions": 0,
            "cum_manual_only_completions": 0,
            "elapsed_seconds": "",
            "seconds_since_last_event": "",
            "seconds_since_progress_increase": "",
            "events_per_minute": "",
        }
        ts = _parse_iso_timestamp(getattr(resolved_event, "timestamp", None))
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            row["elapsed_seconds"] = (ts - first_ts).total_seconds()
            row["seconds_since_last_event"] = (ts - prev_ts).total_seconds() if prev_ts is not None else 0.0
            if delta_coding > EPSILON:
                last_increase_ts = ts
            row["seconds_since_progress_increase"] = (
                (ts - last_increase_ts).total_seconds() if last_increase_ts is not None else (ts - first_ts).total_seconds()
            )
            rolling_window.append(ts)
            row["events_per_minute"] = _events_per_minute(rolling_window)
            prev_ts = ts
        rows.append(row)
        previous_resolved = resolved_metrics
        previous_native = native_metrics
    return rows


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _events_per_minute(window: deque) -> Any:
    if len(window) < 2:
        return ""
    span_seconds = (window[-1] - window[0]).total_seconds()
    if span_seconds <= 0:
        return ""
    return len(window) / (span_seconds / 60.0)


def build_step_rows(
    event_rows: list[dict[str, Any]],
    event_context: dict[int, dict[str, Any]] | None = None,
    event_timestamps: dict[int, str | None] | None = None,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    for row in event_rows:
        rows_by_step.setdefault(int(row["step"]), []).append(row)
    for step in sorted(rows_by_step):
        retained.append(max(rows_by_step[step], key=lambda row: int(row["event_index"])).copy())

    previous: dict[str, Any] | None = None
    previous_event_index = -1
    cum_strong = 0
    cum_manual = 0
    last_increase_step: int | None = None
    last_completion_step: int | None = None
    last_added_step: int | None = None
    prev_step_ts: datetime | None = None
    last_increase_ts_step: datetime | None = None
    first_step_ts: datetime | None = None
    for row in retained:
        current_event_index = int(row["event_index"])
        interval_rows = [
            event_row
            for event_row in event_rows
            if previous_event_index < int(event_row["event_index"]) <= current_event_index
        ]
        _recompute_step_delta_and_source(
            row,
            previous,
            interval_rows,
            progress_field="coding_progress",
            delta_field="delta_coding_progress",
            source_field="coding_drop_source",
        )
        _recompute_step_delta_and_source(
            row,
            previous,
            interval_rows,
            progress_field="overall_progress",
            delta_field="delta_overall_progress",
            source_field="overall_drop_source",
        )
        _recompute_step_delta_and_source(
            row,
            previous,
            interval_rows,
            progress_field="native_coding_progress",
            delta_field="native_delta_coding_progress",
            source_field="native_coding_drop_source",
        )
        _recompute_step_delta_and_source(
            row,
            previous,
            interval_rows,
            progress_field="native_overall_progress",
            delta_field="native_delta_overall_progress",
            source_field="native_overall_drop_source",
        )

        ctx = event_context or {}
        added = sum(1 for r in interval_rows if r["event_type"] == "add_subtask")
        splits = sum(1 for r in interval_rows if r["event_type"] == "split_subtask")
        reopens = sum(1 for r in interval_rows if r["event_type"] == "reopen_subtask")
        invalidations = sum(1 for r in interval_rows if r["event_type"] == "invalidate_subtask")
        prod_completes = val_completes = inv_completes = 0
        strong = manual = 0
        for r in interval_rows:
            ec = ctx.get(int(r["event_index"]))
            if ec is None or not ec.get("is_completion"):
                continue
            if ec["category"] == "product":
                prod_completes += 1
            elif ec["category"] == "validation":
                val_completes += 1
            elif ec["category"] == "investigation":
                inv_completes += 1
            if ec["evidence_strong"]:
                strong += 1
            elif ec["evidence_manual_only"]:
                manual += 1
        cum_strong += strong
        cum_manual += manual
        step_int = int(row["step"])
        if float(row["delta_coding_progress"]) > EPSILON:
            last_increase_step = step_int
        if (added + splits + prod_completes + val_completes + inv_completes) > 0 and any(
            r["event_type"] in {"update_status"} and ctx.get(int(r["event_index"]), {}).get("is_completion")
            for r in interval_rows
        ):
            last_completion_step = step_int
        if added > 0 or splits > 0:
            last_added_step = step_int
        row["step_added_subtasks"] = added
        row["step_split_events"] = splits
        row["step_reopen_events"] = reopens
        row["step_invalidation_events"] = invalidations
        row["step_product_completes"] = prod_completes
        row["step_validation_completes"] = val_completes
        row["step_investigation_completes"] = inv_completes
        row["step_strong_completions"] = strong
        row["step_manual_only_completions"] = manual
        row["cum_strong_completions"] = cum_strong
        row["cum_manual_only_completions"] = cum_manual
        row["steps_since_progress_increase"] = (step_int - last_increase_step) if last_increase_step is not None else step_int
        row["steps_since_completion"] = (step_int - last_completion_step) if last_completion_step is not None else step_int
        row["steps_since_subtask_added"] = (step_int - last_added_step) if last_added_step is not None else step_int

        ts = _parse_iso_timestamp((event_timestamps or {}).get(current_event_index))
        if ts is not None:
            if first_step_ts is None:
                first_step_ts = ts
            row["seconds_since_last_event"] = (ts - prev_step_ts).total_seconds() if prev_step_ts is not None else 0.0
            if float(row["delta_coding_progress"]) > EPSILON:
                last_increase_ts_step = ts
            row["seconds_since_progress_increase"] = (
                (ts - last_increase_ts_step).total_seconds() if last_increase_ts_step is not None else (ts - first_step_ts).total_seconds()
            )
            prev_step_ts = ts

        previous = row
        previous_event_index = current_event_index
    return retained


def _event_context(resolved_events: list[Any]) -> dict[int, dict[str, Any]]:
    classify = rescore.classify_evidence
    strong_types = rescore.STRONG_EVIDENCE_TYPES
    categories: dict[str, str] = {}
    out: dict[int, dict[str, Any]] = {}
    for index, event in enumerate(resolved_events):
        et = event.event_type.value
        payload = event.payload
        if et == EventType.ADD_SUBTASK.value:
            categories[event.subtask_id] = (
                payload.get("category").value if hasattr(payload.get("category"), "value")
                else str(payload.get("category", "product"))
            )
        elif et == EventType.SPLIT_SUBTASK.value:
            for child in payload.get("children", []):
                cat = child.get("category")
                categories[child["id"]] = cat.value if hasattr(cat, "value") else str(cat or categories.get(event.subtask_id, "product"))
        is_completion = (et == EventType.UPDATE_STATUS.value and str(payload.get("status", "")).endswith("complete"))
        evidence = payload.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        types = classify(list(evidence)) if evidence else set()
        out[index] = {
            "category": categories.get(event.subtask_id, "product"),
            "is_completion": is_completion,
            "evidence_strong": bool(strong_types & types),
            "evidence_manual_only": types == {"manual_note"},
        }
    return out


def _recompute_step_delta_and_source(
    row: dict[str, Any],
    previous: dict[str, Any] | None,
    interval_rows: list[dict[str, Any]],
    *,
    progress_field: str,
    delta_field: str,
    source_field: str,
) -> None:
    if previous is None:
        row[delta_field] = 0.0
        row[source_field] = "none"
        return

    delta = float(row[progress_field]) - float(previous[progress_field])
    row[delta_field] = delta
    row[source_field] = _step_drop_source(interval_rows, delta_field, source_field, delta)


def _step_drop_source(
    interval_rows: list[dict[str, Any]],
    delta_field: str,
    source_field: str,
    step_delta: float,
) -> str:
    if step_delta >= -EPSILON:
        return "none"

    contributions: dict[str, float] = {}
    for row in interval_rows:
        event_delta = float(row[delta_field])
        source = row.get(source_field, "none")
        if event_delta < -EPSILON and source != "none":
            contributions[source] = contributions.get(source, 0.0) + abs(event_delta)
    if not contributions:
        return "mixed"

    material = sorted(
        source
        for source, contribution in contributions.items()
        if contribution >= abs(step_delta) * rescore.DROP_MATERIALITY_THRESHOLD - EPSILON
    )
    if not material:
        return "mixed"
    return material[0] if len(material) == 1 else "mixed"


def resolve_categories(
    run_id: str,
    raw_events: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = deepcopy(raw_events)
    summary_categories = summary.get("subtask_categories") if isinstance(summary.get("subtask_categories"), dict) else {}
    explicit = 0
    missing = 0

    for event in resolved:
        event_type = event["event_type"]
        payload = event["payload"]
        if event_type == EventType.ADD_SUBTASK.value:
            subtask_id = event["subtask_id"]
            if "category" in payload:
                explicit += 1
            else:
                missing += 1
                payload["category"] = _resolved_category(run_id, subtask_id, payload["description"], summary_categories)
        elif event_type == EventType.SPLIT_SUBTASK.value:
            for child in payload["children"]:
                child_id = child["id"]
                if "category" in child:
                    explicit += 1
                else:
                    missing += 1
                    child["category"] = _resolved_category(run_id, child_id, child["description"], summary_categories)

    if missing == 0:
        mode = "native"
    elif explicit == 0:
        mode = "legacy_inferred"
    else:
        mode = "mixed"
    return resolved, {"mode": mode, "overrides_applied": missing}


def summarize_run(
    run_id: str,
    rows: list[dict[str, Any]],
    final_success: bool | None,
    resolution: dict[str, Any],
) -> dict[str, Any]:
    final = rows[-1]
    coding_drops = [row for row in rows if float(row["delta_coding_progress"]) < -EPSILON]
    overall_drops = [row for row in rows if float(row["delta_overall_progress"]) < -EPSILON]
    return {
        "run_id": run_id,
        "event_rows": len(rows),
        "final_success": final_success,
        "final_coding_progress": final["coding_progress"],
        "final_overall_progress": final["overall_progress"],
        "diverges": abs(float(final["coding_progress"]) - float(final["overall_progress"])) > EPSILON,
        "coding_nonmonotonic": bool(coding_drops),
        "largest_coding_drop": min((float(row["delta_coding_progress"]) for row in rows), default=0.0),
        "largest_overall_drop": min((float(row["delta_overall_progress"]) for row in rows), default=0.0),
        "largest_coding_drop_source": _largest_drop_source(rows, "delta_coding_progress", "coding_drop_source"),
        "largest_overall_drop_source": _largest_drop_source(rows, "delta_overall_progress", "overall_drop_source"),
        "native_resolved_mismatch": any(
            abs(float(row["coding_progress"]) - float(row["native_coding_progress"])) > EPSILON
            or abs(float(row["overall_progress"]) - float(row["native_overall_progress"])) > EPSILON
            for row in rows
        ),
        **resolution,
    }


def write_dataset_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, result: BuildResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_summaries = result.event_summaries
    step_summaries = result.step_summaries
    total = len(step_summaries)
    successful = sum(1 for item in step_summaries if item["final_success"] is True)
    failed = sum(1 for item in step_summaries if item["final_success"] is False)
    unknown = total - successful - failed
    native_mismatches = [item for item in event_summaries if item["native_resolved_mismatch"]]
    event_nonmonotonic = [item for item in event_summaries if item["coding_nonmonotonic"]]
    step_nonmonotonic = [item for item in step_summaries if item["coding_nonmonotonic"]]
    differing_largest_drops = _largest_drop_differences(event_summaries, step_summaries)
    multiple_event_steps = _runs_with_multiple_events_at_same_step(result.event_rows)
    warnings = list(result.warnings)
    warnings.extend(f"{item['run_id']}: native/resolved metrics differ" for item in native_mismatches)

    lines = [
        "# Ledger Observations v0 Summary",
        "",
        "Event rows preserve replay fidelity with one row per LedgerEvent prefix. Step rows keep the final state for each (run_id, step) and are intended for plotting and later modeling-oriented analysis.",
        "",
        "## Totals",
        "",
        f"- Total runs: {total}",
        f"- Event rows: {len(result.event_rows)}",
        f"- Step rows: {len(result.step_rows)}",
        f"- Successful runs: {successful}",
        f"- Failed runs: {failed}",
        f"- Unknown success runs: {unknown}",
        "",
        "## Category Resolution",
        "",
        "Event rows by category resolution mode:",
        "",
        *_count_lines(_category_mode_counts(result.event_rows)),
        "",
        "Step rows by category resolution mode:",
        "",
        *_count_lines(_category_mode_counts(result.step_rows)),
        "",
        "Runs with native/resolved metric mismatch: "
        + _inline_run_list([item["run_id"] for item in native_mismatches]),
        "",
        "## Non-monotonic Coding Progress",
        "",
        "Event-level: " + _inline_run_list([item["run_id"] for item in event_nonmonotonic]),
        "",
        "Step-level: " + _inline_run_list([item["run_id"] for item in step_nonmonotonic]),
        "",
        "## Largest Event-Level Coding Drops",
        "",
        *_drop_lines(event_summaries, "largest_coding_drop", "largest_coding_drop_source"),
        "",
        "## Largest Step-Level Coding Drops",
        "",
        *_drop_lines(step_summaries, "largest_coding_drop", "largest_coding_drop_source"),
        "",
        "## Largest Event-Level Overall Drops",
        "",
        *_drop_lines(event_summaries, "largest_overall_drop", "largest_overall_drop_source"),
        "",
        "## Largest Step-Level Overall Drops",
        "",
        *_drop_lines(step_summaries, "largest_overall_drop", "largest_overall_drop_source"),
        "",
        "## Event vs Step",
        "",
        "Runs where event-level and step-level largest coding drops differ: "
        + _inline_run_list([item["run_id"] for item in differing_largest_drops]),
        "",
        "Runs with multiple events at the same step: "
        + _inline_run_list(multiple_event_steps),
        "",
        "## Success / Progress Quadrants",
        "",
        *_quadrant_lines(result.step_rows),
        "",
        "## Sanity Check Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines).rstrip() + "\n")


def _metrics(events: list[Any]) -> dict[str, Any]:
    ledger = replay(events)
    coding = score(ledger, CODING_CATEGORIES)
    overall = score(ledger, ALL_CATEGORIES)
    product = score(ledger, (SubtaskCategory.PRODUCT,))
    validation = score(ledger, (SubtaskCategory.VALIDATION,))
    investigation = score(ledger, (SubtaskCategory.INVESTIGATION,))
    return {
        "coding_progress": coding.progress,
        "overall_progress": overall.progress,
        "active_coding_weight": coding.active_weight,
        "completed_coding_weight": coding.complete_weight,
        "active_overall_weight": overall.active_weight,
        "completed_overall_weight": overall.complete_weight,
        "active_coding_leaves": coding.active_leaf_count,
        "completed_coding_leaves": coding.complete_leaf_count,
        "active_overall_leaves": overall.active_leaf_count,
        "completed_overall_leaves": overall.complete_leaf_count,
        "product_progress": product.progress,
        "validation_progress": validation.progress,
        "investigation_progress": investigation.progress,
        "coding_snapshot": _leaf_snapshot(ledger, CODING_CATEGORIES),
        "overall_snapshot": _leaf_snapshot(ledger, ALL_CATEGORIES),
    }


def _leaf_snapshot(ledger: Any, categories: tuple[SubtaskCategory, ...]) -> dict[str, dict[str, Any]]:
    selected = set(categories)
    active_ids = {
        subtask_id
        for subtask_id, subtask in ledger.subtasks.items()
        if subtask.status not in {Status.INVALIDATED, Status.DELETED}
    }
    parents = {
        subtask.parent_id
        for subtask in ledger.subtasks.values()
        if subtask.parent_id in active_ids and subtask.status not in {Status.INVALIDATED, Status.DELETED}
    }
    snapshot: dict[str, dict[str, Any]] = {}
    for leaf_id in active_ids - parents:
        subtask = ledger.subtasks[leaf_id]
        if subtask.category not in selected:
            continue
        snapshot[leaf_id] = {
            "category": subtask.category.value,
            "active_weight": subtask.weight,
            "complete_weight": subtask.weight if subtask.status is Status.COMPLETE else 0.0,
            "status": subtask.status.value,
        }
    return snapshot


def _drop_source(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    progress_key: str,
    delta: float,
) -> str:
    if previous is None or delta >= -EPSILON:
        return "none"
    snapshot_key = "coding_snapshot" if progress_key == "coding_progress" else "overall_snapshot"
    before_snapshot = previous[snapshot_key]
    after_snapshot = current[snapshot_key]
    changed_ids = {
        leaf_id
        for leaf_id in set(before_snapshot) | set(after_snapshot)
        if before_snapshot.get(leaf_id) != after_snapshot.get(leaf_id)
    }
    if not changed_ids:
        return "none"

    before_complete = sum(leaf["complete_weight"] for leaf in before_snapshot.values())
    before_active = sum(leaf["active_weight"] for leaf in before_snapshot.values())
    before_progress = previous[progress_key]
    drop = abs(delta)
    contributions: dict[str, float] = {}
    for leaf_id in changed_ids:
        before_leaf = before_snapshot.get(leaf_id)
        after_leaf = after_snapshot.get(leaf_id)
        leaf = after_leaf or before_leaf
        category = leaf["category"]
        delta_complete = (after_leaf or {}).get("complete_weight", 0.0) - (before_leaf or {}).get("complete_weight", 0.0)
        delta_active = (after_leaf or {}).get("active_weight", 0.0) - (before_leaf or {}).get("active_weight", 0.0)
        next_active = before_active + delta_active
        next_progress = (before_complete + delta_complete) / next_active if next_active else 0.0
        contribution = max(0.0, before_progress - next_progress)
        if contribution > EPSILON:
            contributions[category] = contributions.get(category, 0.0) + contribution

    material = sorted(
        category
        for category, contribution in contributions.items()
        if contribution >= drop * rescore.DROP_MATERIALITY_THRESHOLD - EPSILON
    )
    if not material:
        return "none"
    return material[0] if len(material) == 1 else "mixed"


def _resolved_category(
    run_id: str,
    subtask_id: str,
    description: str,
    summary_categories: dict[str, Any],
) -> str:
    if isinstance(summary_categories.get(subtask_id), str):
        return SubtaskCategory(summary_categories[subtask_id]).value
    return rescore.category_for(run_id, subtask_id, description).value


def _load_raw_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_summary(run_dir: Path, run_id: str) -> tuple[dict[str, Any], list[str]]:
    path = run_dir / "summary_by_category.json"
    if not path.exists():
        return {}, [f"{run_id}: summary_by_category.json is missing"]
    try:
        return json.loads(path.read_text()), []
    except json.JSONDecodeError as exc:
        return {}, [f"{run_id}: summary_by_category.json is invalid: {exc}"]


def _run_id(runs_dir: Path, ledger_path: Path) -> str:
    try:
        return ledger_path.parent.relative_to(runs_dir).as_posix()
    except ValueError:
        return ledger_path.parent.name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_counters(counters: dict[str, int], event_type: EventType) -> None:
    if event_type is EventType.SPLIT_SUBTASK:
        counters["num_splits_so_far"] += 1
    elif event_type is EventType.REOPEN_SUBTASK:
        counters["num_reopens_so_far"] += 1
    elif event_type is EventType.INVALIDATE_SUBTASK:
        counters["num_invalidations_so_far"] += 1


def _delta(previous: dict[str, Any] | None, current: dict[str, Any], key: str) -> float:
    if previous is None:
        return 0.0
    return current[key] - previous[key]


def _csv_bool(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _largest_drop_source(rows: list[dict[str, Any]], delta_key: str, source_key: str) -> str:
    if not rows:
        return "none"
    row = min(rows, key=lambda item: float(item[delta_key]))
    return row[source_key] if float(row[delta_key]) < -EPSILON else "none"


def _run_list(items: list[dict[str, Any]], empty: str) -> str:
    if not items:
        return empty
    return ", ".join(f"`{item['run_id']}`" for item in sorted(items, key=lambda item: item["run_id"])) + "."


def _inline_run_list(run_ids: list[str]) -> str:
    if not run_ids:
        return "none"
    return ", ".join(f"`{run_id}`" for run_id in sorted(run_ids)) + "."


def _category_mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("category_resolution_mode", "missing"))
        counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def _count_lines(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- `{name}`: {count}" for name, count in counts.items()]


def _largest_drop_differences(
    event_summaries: list[dict[str, Any]],
    step_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    step_by_run = {summary["run_id"]: summary for summary in step_summaries}
    differences = []
    for event_summary in event_summaries:
        run_id = event_summary["run_id"]
        step_summary = step_by_run[run_id]
        event_drop = abs(float(event_summary["largest_coding_drop"]))
        step_drop = abs(float(step_summary["largest_coding_drop"]))
        if abs(event_drop - step_drop) > EPSILON:
            differences.append({
                "run_id": run_id,
                "event_drop": event_drop,
                "step_drop": step_drop,
            })
    return differences


def _runs_with_multiple_events_at_same_step(rows: list[dict[str, Any]]) -> list[str]:
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["run_id"]), int(row["step"]))
        counts[key] = counts.get(key, 0) + 1
    return sorted({run_id for (run_id, _), count in counts.items() if count > 1})


def _quadrant_lines(step_rows: list[dict[str, Any]]) -> list[str]:
    final_rows: dict[str, dict[str, Any]] = {}
    for row in step_rows:
        final_rows[str(row["run_id"])] = row

    quadrants = {
        "Success + high progress": [],
        "Success + low progress": [],
        "Failure + high progress": [],
        "Failure + low progress": [],
        "Unknown success": [],
    }
    for run_id, row in sorted(final_rows.items()):
        progress = float(row["coding_progress"])
        success = str(row["final_success"])
        high = progress >= 0.8 - EPSILON
        if success == "true" and high:
            quadrants["Success + high progress"].append(run_id)
        elif success == "true":
            quadrants["Success + low progress"].append(run_id)
        elif success == "false" and high:
            quadrants["Failure + high progress"].append(run_id)
        elif success == "false":
            quadrants["Failure + low progress"].append(run_id)
        else:
            quadrants["Unknown success"].append(run_id)
    return [f"- {name}: {_inline_run_list(run_ids)}" for name, run_ids in quadrants.items()]


def _drop_lines(summaries: list[dict[str, Any]], drop_key: str, source_key: str) -> list[str]:
    drops = sorted(
        (item for item in summaries if float(item[drop_key]) < -EPSILON),
        key=lambda item: float(item[drop_key]),
    )[:10]
    if not drops:
        return ["- none"]
    return [
        f"- `{item['run_id']}`: {abs(float(item[drop_key])):.6f} ({item[source_key]})"
        for item in drops
    ]


if __name__ == "__main__":
    raise SystemExit(main())
