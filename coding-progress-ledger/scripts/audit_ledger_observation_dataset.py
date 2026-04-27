from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EPSILON = 1e-9
HIGH_PROGRESS_THRESHOLD = 0.8
LARGE_NATIVE_RESOLVED_DIVERGENCE_THRESHOLD = 0.05
SUBSTANTIAL_EVENT_STEP_DROP_DIFF_THRESHOLD = 0.05

IDENTIFIER_FIELDS = ("run_id", "step", "event_index", "event_type")
PROGRESS_FIELDS = (
    "coding_progress",
    "overall_progress",
    "native_coding_progress",
    "native_overall_progress",
)
DELTA_FIELDS = (
    "delta_coding_progress",
    "delta_overall_progress",
    "native_delta_coding_progress",
    "native_delta_overall_progress",
)
WEIGHT_PAIRS = (
    ("completed_coding_weight", "active_coding_weight"),
    ("completed_overall_weight", "active_overall_weight"),
    ("native_completed_coding_weight", "native_active_coding_weight"),
    ("native_completed_overall_weight", "native_active_overall_weight"),
)
LEAF_PAIRS = (
    ("completed_coding_leaves", "active_coding_leaves"),
    ("completed_overall_leaves", "active_overall_leaves"),
    ("native_completed_coding_leaves", "native_active_coding_leaves"),
    ("native_completed_overall_leaves", "native_active_overall_leaves"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ledger observation dataset coherence.")
    parser.add_argument("--input-csv", default="datasets/ledger_observations_v0.csv")
    parser.add_argument("--output-md", default="datasets/ledger_observations_v0_audit.md")
    parser.add_argument("--output-json", default="datasets/ledger_observations_v0_audit.json")
    args = parser.parse_args(argv)

    summary = audit_dataset(Path(args.input_csv))
    write_json(Path(args.output_json), summary)
    write_markdown(Path(args.output_md), summary)
    return 0


def audit_dataset(input_csv: Path) -> dict[str, Any]:
    rows = _load_rows(input_csv)
    rows_by_run = _rows_by_run(rows)
    integrity, parse_failures = integrity_checks(rows_by_run)
    category_resolution = category_resolution_diagnostics(rows)
    drops = drop_diagnostics(rows_by_run)
    event_vs_step = event_vs_step_comparison(rows_by_run)
    quadrants = success_progress_quadrants(rows_by_run)
    warnings = build_warnings(integrity, category_resolution, event_vs_step, parse_failures)

    return {
        "totals": {
            "rows": len(rows),
            "runs": len(rows_by_run),
            "input_csv": str(input_csv),
        },
        "integrity": integrity,
        "category_resolution": category_resolution,
        "drops": drops,
        "event_vs_step": event_vs_step,
        "success_progress_quadrants": quadrants,
        "warnings": warnings,
    }


def integrity_checks(rows_by_run: dict[str, list[dict[str, str]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    invalid_progress: list[dict[str, Any]] = []
    completed_exceeds_active: list[dict[str, Any]] = []
    delta_mismatches: list[dict[str, Any]] = []
    first_delta_nonzero: list[dict[str, Any]] = []
    missing_identifiers: list[dict[str, Any]] = []
    invalid_success_metadata: list[dict[str, Any]] = []
    unknown_success_metadata: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []

    for run_id, rows in rows_by_run.items():
        for row in rows:
            row_ref = _row_ref(row)
            for field in IDENTIFIER_FIELDS:
                if not row.get(field):
                    missing_identifiers.append({**row_ref, "field": field})
            for field in PROGRESS_FIELDS:
                value = _float(row, field, row_ref, parse_failures)
                if value is not None and (value < -EPSILON or value > 1.0 + EPSILON):
                    invalid_progress.append({**row_ref, "field": field, "value": value})
            for complete_field, active_field in WEIGHT_PAIRS:
                complete = _float(row, complete_field, row_ref, parse_failures)
                active = _float(row, active_field, row_ref, parse_failures)
                if complete is not None and active is not None and complete > active + EPSILON:
                    completed_exceeds_active.append({
                        **row_ref,
                        "completed_field": complete_field,
                        "active_field": active_field,
                        "completed": complete,
                        "active": active,
                    })
            for complete_field, active_field in LEAF_PAIRS:
                complete = _int(row, complete_field, row_ref, parse_failures)
                active = _int(row, active_field, row_ref, parse_failures)
                if complete is not None and active is not None and complete > active:
                    completed_exceeds_active.append({
                        **row_ref,
                        "completed_field": complete_field,
                        "active_field": active_field,
                        "completed": complete,
                        "active": active,
                    })
            success = (row.get("final_success") or "").strip().lower()
            success_source = (row.get("final_success_source") or "").strip().lower()
            if success == "" and success_source == "unknown":
                unknown_success_metadata.append(row_ref)
            elif success not in {"true", "false"}:
                invalid_success_metadata.append({**row_ref, "final_success": row.get("final_success", ""), "final_success_source": row.get("final_success_source", "")})

        first = rows[0]
        first_ref = _row_ref(first)
        for field in DELTA_FIELDS:
            value = _float(first, field, first_ref, parse_failures)
            if value is not None and abs(value) > EPSILON:
                first_delta_nonzero.append({**first_ref, "field": field, "value": value})

        for previous, current in zip(rows, rows[1:]):
            current_ref = _row_ref(current)
            checks = (
                ("delta_coding_progress", "coding_progress"),
                ("delta_overall_progress", "overall_progress"),
                ("native_delta_coding_progress", "native_coding_progress"),
                ("native_delta_overall_progress", "native_overall_progress"),
            )
            for delta_field, progress_field in checks:
                previous_progress = _float(previous, progress_field, _row_ref(previous), parse_failures)
                current_progress = _float(current, progress_field, current_ref, parse_failures)
                observed_delta = _float(current, delta_field, current_ref, parse_failures)
                if previous_progress is None or current_progress is None or observed_delta is None:
                    continue
                expected_delta = current_progress - previous_progress
                if abs(observed_delta - expected_delta) > EPSILON:
                    delta_mismatches.append({
                        **current_ref,
                        "field": delta_field,
                        "observed": observed_delta,
                        "expected": expected_delta,
                    })

    return {
        "invalid_progress": invalid_progress,
        "completed_exceeds_active": completed_exceeds_active,
        "delta_mismatches": delta_mismatches,
        "first_delta_nonzero": first_delta_nonzero,
        "missing_identifiers": missing_identifiers,
        "invalid_success_metadata": invalid_success_metadata,
        "unknown_success_metadata": unknown_success_metadata,
        "parse_failures": parse_failures,
        "passed": not any([
            invalid_progress,
            completed_exceeds_active,
            delta_mismatches,
            first_delta_nonzero,
            missing_identifiers,
            invalid_success_metadata,
            parse_failures,
        ]),
    }, parse_failures


def category_resolution_diagnostics(rows: list[dict[str, str]]) -> dict[str, Any]:
    mode_counts = Counter((row.get("category_resolution_mode") or "missing") for row in rows)
    mismatch_runs: set[str] = set()
    max_coding_diff: dict[str, float] = defaultdict(float)
    drop_source_differences: list[dict[str, Any]] = []

    for row in rows:
        run_id = row.get("run_id", "")
        coding_diff = abs(_float_value(row.get("coding_progress")) - _float_value(row.get("native_coding_progress")))
        overall_diff = abs(_float_value(row.get("overall_progress")) - _float_value(row.get("native_overall_progress")))
        if coding_diff > EPSILON or overall_diff > EPSILON:
            mismatch_runs.add(run_id)
        max_coding_diff[run_id] = max(max_coding_diff[run_id], coding_diff)

        coding_sources_differ = row.get("coding_drop_source") != row.get("native_coding_drop_source")
        overall_sources_differ = row.get("overall_drop_source") != row.get("native_overall_drop_source")
        if coding_sources_differ or overall_sources_differ:
            drop_source_differences.append({
                **_row_ref(row),
                "coding_drop_source": row.get("coding_drop_source", ""),
                "native_coding_drop_source": row.get("native_coding_drop_source", ""),
                "overall_drop_source": row.get("overall_drop_source", ""),
                "native_overall_drop_source": row.get("native_overall_drop_source", ""),
            })

    return {
        "rows_by_category_resolution_mode": dict(sorted(mode_counts.items())),
        "runs_with_native_resolved_metric_mismatch": sorted(mismatch_runs),
        "max_abs_native_resolved_coding_progress_diff_by_run": {
            run_id: diff for run_id, diff in sorted(max_coding_diff.items()) if diff > EPSILON
        },
        "drop_source_differences": drop_source_differences,
    }


def drop_diagnostics(rows_by_run: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    negative_coding = []
    negative_overall = []
    coding_source_counts: Counter[str] = Counter()
    overall_source_counts: Counter[str] = Counter()
    largest_coding_by_run: dict[str, dict[str, Any]] = {}
    largest_overall_by_run: dict[str, dict[str, Any]] = {}

    for run_id, rows in rows_by_run.items():
        coding_drop = _largest_drop(rows, "delta_coding_progress", "coding_drop_source")
        overall_drop = _largest_drop(rows, "delta_overall_progress", "overall_drop_source")
        largest_coding_by_run[run_id] = coding_drop
        largest_overall_by_run[run_id] = overall_drop
        for row in rows:
            coding_delta = _float_value(row.get("delta_coding_progress"))
            overall_delta = _float_value(row.get("delta_overall_progress"))
            if coding_delta < -EPSILON:
                negative_coding.append(_row_ref(row))
                coding_source_counts[row.get("coding_drop_source") or "missing"] += 1
            if overall_delta < -EPSILON:
                negative_overall.append(_row_ref(row))
                overall_source_counts[row.get("overall_drop_source") or "missing"] += 1

    return {
        "negative_coding_delta_count": len(negative_coding),
        "negative_overall_delta_count": len(negative_overall),
        "largest_coding_drops_by_run": largest_coding_by_run,
        "largest_overall_drops_by_run": largest_overall_by_run,
        "coding_drop_source_counts": dict(sorted(coding_source_counts.items())),
        "overall_drop_source_counts": dict(sorted(overall_source_counts.items())),
    }


def event_vs_step_comparison(rows_by_run: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    differing_runs: list[dict[str, Any]] = []
    multiple_event_steps: dict[str, list[int]] = {}
    largest_event_coding_drops: dict[str, dict[str, Any]] = {}
    largest_step_coding_drops: dict[str, dict[str, Any]] = {}

    for run_id, rows in rows_by_run.items():
        event_drop = _largest_drop(rows, "delta_coding_progress", "coding_drop_source")
        step_rows, multi_steps = _step_level_rows(rows)
        step_drop = _largest_drop_from_progress(step_rows, "coding_progress", "coding_drop_source")
        largest_event_coding_drops[run_id] = event_drop
        largest_step_coding_drops[run_id] = step_drop
        if multi_steps:
            multiple_event_steps[run_id] = multi_steps
        if abs(event_drop["amount"] - step_drop["amount"]) > EPSILON:
            differing_runs.append({
                "run_id": run_id,
                "event_level_largest_coding_drop": event_drop["amount"],
                "step_level_largest_coding_drop": step_drop["amount"],
                "difference": abs(event_drop["amount"] - step_drop["amount"]),
            })

    return {
        "largest_event_level_coding_drops_by_run": largest_event_coding_drops,
        "largest_step_level_coding_drops_by_run": largest_step_coding_drops,
        "runs_where_largest_coding_drop_differs": differing_runs,
        "runs_with_multiple_events_at_same_step": multiple_event_steps,
    }


def success_progress_quadrants(rows_by_run: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    buckets = {
        "success_high_progress": [],
        "success_low_progress": [],
        "failure_high_progress": [],
        "failure_low_progress": [],
        "unknown_success": [],
    }
    for run_id, rows in rows_by_run.items():
        final = rows[-1]
        progress = _float_value(final.get("coding_progress"))
        success = (final.get("final_success") or "").strip().lower()
        high = progress >= HIGH_PROGRESS_THRESHOLD - EPSILON
        if success == "true" and high:
            buckets["success_high_progress"].append(run_id)
        elif success == "true":
            buckets["success_low_progress"].append(run_id)
        elif success == "false" and high:
            buckets["failure_high_progress"].append(run_id)
        elif success == "false":
            buckets["failure_low_progress"].append(run_id)
        else:
            buckets["unknown_success"].append(run_id)

    return {
        "high_progress_threshold": HIGH_PROGRESS_THRESHOLD,
        **{key: sorted(value) for key, value in buckets.items()},
    }


def build_warnings(
    integrity: dict[str, Any],
    category_resolution: dict[str, Any],
    event_vs_step: dict[str, Any],
    parse_failures: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if integrity["invalid_progress"]:
        warnings.append(f"invalid progress values: {len(integrity['invalid_progress'])}")
    if integrity["completed_exceeds_active"]:
        warnings.append(f"completed values exceed active values: {len(integrity['completed_exceeds_active'])}")
    if integrity["delta_mismatches"] or integrity["first_delta_nonzero"]:
        count = len(integrity["delta_mismatches"]) + len(integrity["first_delta_nonzero"])
        warnings.append(f"delta mismatch/nonzero first delta: {count}")
    if integrity["invalid_success_metadata"]:
        warnings.append(f"missing or invalid success metadata: {len(integrity['invalid_success_metadata'])}")
    if integrity["unknown_success_metadata"]:
        warnings.append(f"explicitly unknown success metadata: {len(integrity['unknown_success_metadata'])}")
    if parse_failures:
        warnings.append(f"numeric parse failures: {len(parse_failures)}")

    large_divergences = [
        {"run_id": run_id, "max_diff": diff}
        for run_id, diff in category_resolution["max_abs_native_resolved_coding_progress_diff_by_run"].items()
        if diff > LARGE_NATIVE_RESOLVED_DIVERGENCE_THRESHOLD + EPSILON
    ]
    if large_divergences:
        warnings.append(f"large native/resolved divergence: {len(large_divergences)} runs")
    category_resolution["large_native_resolved_divergences"] = large_divergences

    substantial_step_differences = [
        item
        for item in event_vs_step["runs_where_largest_coding_drop_differs"]
        if item["difference"] > SUBSTANTIAL_EVENT_STEP_DROP_DIFF_THRESHOLD + EPSILON
    ]
    if substantial_step_differences:
        warnings.append(f"event-level and step-level largest drops differ substantially: {len(substantial_step_differences)} runs")
    event_vs_step["substantial_largest_drop_differences"] = substantial_step_differences
    return warnings


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ledger Observations v0 Audit",
        "",
        "Audit of checkpoint-level observation CSV coherence.",
        "",
        "## Totals",
        "",
        f"- Rows: {summary['totals']['rows']}",
        f"- Runs: {summary['totals']['runs']}",
        f"- Integrity passed: {_yes_no(summary['integrity']['passed'])}",
        "",
        "## Integrity",
        "",
        f"- Invalid progress values: {len(summary['integrity']['invalid_progress'])}",
        f"- Completed > active failures: {len(summary['integrity']['completed_exceeds_active'])}",
        f"- Delta mismatches: {len(summary['integrity']['delta_mismatches'])}",
        f"- First-row nonzero deltas: {len(summary['integrity']['first_delta_nonzero'])}",
        f"- Missing identifiers: {len(summary['integrity']['missing_identifiers'])}",
        f"- Invalid success metadata: {len(summary['integrity']['invalid_success_metadata'])}",
        f"- Unknown success metadata: {len(summary['integrity']['unknown_success_metadata'])}",
        "",
        "## Category Resolution",
        "",
        _dict_table(summary["category_resolution"]["rows_by_category_resolution_mode"], "Mode", "Rows"),
        "",
        "Runs with native/resolved metric mismatch: "
        + _inline_list(summary["category_resolution"]["runs_with_native_resolved_metric_mismatch"]),
        "",
        "## Drops",
        "",
        f"- Negative coding deltas: {summary['drops']['negative_coding_delta_count']}",
        f"- Negative overall deltas: {summary['drops']['negative_overall_delta_count']}",
        "",
        "Coding drop sources:",
        "",
        _dict_table(summary["drops"]["coding_drop_source_counts"], "Source", "Count"),
        "",
        "Overall drop sources:",
        "",
        _dict_table(summary["drops"]["overall_drop_source_counts"], "Source", "Count"),
        "",
        "## Event vs Step",
        "",
        "Runs where largest event-level and step-level coding drops differ: "
        + _inline_list([item["run_id"] for item in summary["event_vs_step"]["runs_where_largest_coding_drop_differs"]]),
        "",
        "Runs with multiple events at the same step: "
        + _inline_list(summary["event_vs_step"]["runs_with_multiple_events_at_same_step"].keys()),
        "",
        "## Success / Progress Quadrants",
        "",
        f"- Success + high progress: {_inline_list(summary['success_progress_quadrants']['success_high_progress'])}",
        f"- Success + low progress: {_inline_list(summary['success_progress_quadrants']['success_low_progress'])}",
        f"- Failure + high progress: {_inline_list(summary['success_progress_quadrants']['failure_high_progress'])}",
        f"- Failure + low progress: {_inline_list(summary['success_progress_quadrants']['failure_low_progress'])}",
        f"- Unknown success: {_inline_list(summary['success_progress_quadrants']['unknown_success'])}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in summary["warnings"]) if summary["warnings"] else lines.append("- none")
    path.write_text("\n".join(lines).rstrip() + "\n")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _rows_by_run(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("run_id", "")].append(row)
    return {
        run_id: sorted(run_rows, key=lambda row: _int_value(row.get("event_index")))
        for run_id, run_rows in sorted(grouped.items())
    }


def _step_level_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[int]]:
    by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_step[_int_value(row.get("step"))].append(row)
    step_rows = [max(items, key=lambda row: _int_value(row.get("event_index"))) for _, items in sorted(by_step.items())]
    multi_steps = [step for step, items in sorted(by_step.items()) if len(items) > 1]
    return step_rows, multi_steps


def _largest_drop(rows: list[dict[str, str]], delta_field: str, source_field: str) -> dict[str, Any]:
    best = {"amount": 0.0, "event_index": None, "step": None, "source": "none"}
    for row in rows:
        delta = _float_value(row.get(delta_field))
        if delta < -best["amount"] - EPSILON:
            best = {
                "amount": abs(delta),
                "event_index": _int_value(row.get("event_index")),
                "step": _int_value(row.get("step")),
                "source": row.get(source_field) or "none",
            }
    return best


def _largest_drop_from_progress(rows: list[dict[str, str]], progress_field: str, source_field: str) -> dict[str, Any]:
    best = {"amount": 0.0, "event_index": None, "step": None, "source": "none"}
    for previous, current in zip(rows, rows[1:]):
        before = _float_value(previous.get(progress_field))
        after = _float_value(current.get(progress_field))
        drop = before - after
        if drop > best["amount"] + EPSILON:
            best = {
                "amount": drop,
                "event_index": _int_value(current.get("event_index")),
                "step": _int_value(current.get("step")),
                "source": current.get(source_field) or "none",
            }
    return best


def _float(row: dict[str, str], field: str, row_ref: dict[str, Any], failures: list[dict[str, Any]]) -> float | None:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        failures.append({**row_ref, "field": field, "value": row.get(field, "")})
        return None


def _int(row: dict[str, str], field: str, row_ref: dict[str, Any], failures: list[dict[str, Any]]) -> int | None:
    try:
        return int(float(row.get(field, "")))
    except (TypeError, ValueError):
        failures.append({**row_ref, "field": field, "value": row.get(field, "")})
        return None


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _row_ref(row: dict[str, str]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id", ""),
        "event_index": _int_value(row.get("event_index")),
        "step": _int_value(row.get("step")),
        "event_type": row.get("event_type", ""),
    }


def _dict_table(values: dict[str, Any], key_header: str, value_header: str) -> str:
    if not values:
        return "none"
    lines = [f"| {key_header} | {value_header} |", "| --- | ---: |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def _inline_list(values: Any) -> str:
    values = list(values)
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in sorted(values))


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
