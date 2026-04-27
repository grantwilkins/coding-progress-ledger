from __future__ import annotations

import csv
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger_progress import Status, SubtaskCategory, replay, score
from ledger_progress.serialization import event_from_dict


RUNS_DIR = ROOT / "runs"

CODING_CATEGORIES = (
    SubtaskCategory.PRODUCT,
    SubtaskCategory.VALIDATION,
    SubtaskCategory.INVESTIGATION,
)
EXCLUDED_CATEGORIES = (
    SubtaskCategory.ARTIFACT,
    SubtaskCategory.DOCUMENTATION,
    SubtaskCategory.ENVIRONMENT,
)
ALL_CATEGORIES = tuple(SubtaskCategory)
EPSILON = 1e-12
DROP_MATERIALITY_THRESHOLD = 0.20
STRONG_EVIDENCE_TYPES = {"test_output", "diff", "file_exists", "command_output"}
CONTRACT_UNDERSTANDING_TERMS = ("understand", "requirement", "expected", "contract", "task", "issue")

EVIDENCE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("test_output", ("pytest", "unittest", "npm test", "node --test", "test_output.txt", "passed", "failed")),
    ("diff", ("final_diff.patch", "diff", "patch", "changed", "modified", "implementation", "function", "class")),
    ("file_exists", ("exists", "created", "wrote file", "artifact present")),
    ("command_output", ("stdout", "stderr", "command output", "cli invocation", "python -m")),
    ("contract_text", ("task.md", "readme", "issue statement", "expected behavior", "contract")),
)
SOURCE_PATH_PATTERN = re.compile(r"(^|\s|`)[\w./-]+\.(py|js|ts|jsx|tsx)\b", re.IGNORECASE)


CATEGORY_OVERRIDES: dict[str, dict[str, SubtaskCategory]] = {
    "negative_control_incomplete_budget_limited": {
        "S1": SubtaskCategory.PRODUCT,
        "S2": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.VALIDATION,
        "S5": SubtaskCategory.PRODUCT,
        "S6": SubtaskCategory.PRODUCT,
    },
    "negative_control_monotonic_one_line": {
        "S1": SubtaskCategory.PRODUCT,
        "S2": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.VALIDATION,
    },
    "control_coding_complete_artifacts_incomplete": {
        "S1": SubtaskCategory.PRODUCT,
        "S2": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.ARTIFACT,
    },
    "control_high_progress_wrong_solution": {
        "S1": SubtaskCategory.INVESTIGATION,
        "S2": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.VALIDATION,
    },
    "control_monotonic_incomplete_failure": {
        "S1": SubtaskCategory.INVESTIGATION,
        "S2": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.PRODUCT,
        "S5": SubtaskCategory.VALIDATION,
    },
    "task_1_parser_timezone_offset": {
        "S1": SubtaskCategory.PRODUCT,
        "S2": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.INVESTIGATION,
        "S4": SubtaskCategory.PRODUCT,
        "S4.1": SubtaskCategory.PRODUCT,
        "S4.2": SubtaskCategory.PRODUCT,
        "S5": SubtaskCategory.ARTIFACT,
        "S6": SubtaskCategory.VALIDATION,
    },
    "task_2_cli_output_flag": {
        "S1": SubtaskCategory.INVESTIGATION,
        "S2": SubtaskCategory.PRODUCT,
        "S3": SubtaskCategory.VALIDATION,
        "S3.1": SubtaskCategory.VALIDATION,
        "S3.2": SubtaskCategory.VALIDATION,
        "S3.3": SubtaskCategory.VALIDATION,
        "S4": SubtaskCategory.PRODUCT,
        "S5": SubtaskCategory.ARTIFACT,
        "S9": SubtaskCategory.ENVIRONMENT,
    },
    "task_3_config_error_type": {
        "S1": SubtaskCategory.PRODUCT,
        "S2": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.PRODUCT,
        "S3.1": SubtaskCategory.PRODUCT,
        "S3.2": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.VALIDATION,
        "S5": SubtaskCategory.ARTIFACT,
    },
    "task_4_csv_messy_aggregation": {
        "S1": SubtaskCategory.VALIDATION,
        "S2": SubtaskCategory.PRODUCT,
        "S3": SubtaskCategory.VALIDATION,
        "S4": SubtaskCategory.VALIDATION,
        "S4.1": SubtaskCategory.VALIDATION,
        "S4.2": SubtaskCategory.VALIDATION,
        "S4.3": SubtaskCategory.VALIDATION,
        "S5": SubtaskCategory.PRODUCT,
        "S6": SubtaskCategory.PRODUCT,
        "S7": SubtaskCategory.PRODUCT,
    },
    "task_5_reset_state_reducer": {
        "S1": SubtaskCategory.PRODUCT,
        "S2": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.PRODUCT,
        "S3.1": SubtaskCategory.PRODUCT,
        "S3.2": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.ARTIFACT,
        "S5": SubtaskCategory.ARTIFACT,
        "S6": SubtaskCategory.DOCUMENTATION,
        "S7": SubtaskCategory.ENVIRONMENT,
    },
    "task_6_async_stale_result": {
        "S1": SubtaskCategory.VALIDATION,
        "S2": SubtaskCategory.PRODUCT,
        "S3": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.VALIDATION,
        "S4.1": SubtaskCategory.VALIDATION,
        "S4.2": SubtaskCategory.VALIDATION,
        "S4.3": SubtaskCategory.ENVIRONMENT,
        "S5": SubtaskCategory.PRODUCT,
        "S9": SubtaskCategory.ENVIRONMENT,
    },
    "task_7_refactor_validation_split": {
        "S1": SubtaskCategory.PRODUCT,
        "S2": SubtaskCategory.PRODUCT,
        "S3": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.VALIDATION,
        "S4.1": SubtaskCategory.VALIDATION,
        "S4.2": SubtaskCategory.VALIDATION,
        "S4.3": SubtaskCategory.VALIDATION,
        "S5": SubtaskCategory.ARTIFACT,
    },
    "task_8_package_import_failure": {
        "S1": SubtaskCategory.PRODUCT,
        "S2": SubtaskCategory.VALIDATION,
        "S2.1": SubtaskCategory.VALIDATION,
        "S2.2": SubtaskCategory.VALIDATION,
        "S2.3": SubtaskCategory.VALIDATION,
        "S3": SubtaskCategory.PRODUCT,
        "S4": SubtaskCategory.VALIDATION,
        "S5": SubtaskCategory.ARTIFACT,
        "S9": SubtaskCategory.INVESTIGATION,
    },
}


def main() -> None:
    run_dirs = sorted((path.parent for path in RUNS_DIR.glob("*/ledger.jsonl")), key=lambda path: run_sort_key(path.name))
    summaries = []
    for run_dir in run_dirs:
        summaries.append(rescore_run(run_dir))
    write_suite_summary_table(RUNS_DIR / "SUITE_SUMMARY.md", summaries)
    write_category_summary(RUNS_DIR / "SUITE_CATEGORY_SUMMARY.md", summaries)


def rescore_run(run_dir: Path) -> dict[str, Any]:
    run_id = run_dir.name
    raw_events = [json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines() if line.strip()]
    annotated_events, category_map = annotate_categories(run_id, raw_events)
    events = [event_from_dict(event) for event in annotated_events]
    steps = sorted({event.step for event in events})
    rows = [progress_row(events, step) for step in steps]
    write_progress_csv(run_dir / "progress_by_category.csv", rows)

    final = rows[-1]
    original = load_original_summary(run_id, runs_dir=run_dir.parent)
    final_success, final_success_source = final_success_from_metadata(run_dir, original)
    evidence_audit = audit_completion_evidence(annotated_events)
    overall_drop_detail = largest_drop_detail(rows, "overall_progress")
    coding_drop_detail = largest_drop_detail(rows, "coding_progress")
    coding_drop_contributions = largest_drop_category_contributions(events, rows, "coding_progress", CODING_CATEGORIES)
    overall_drop_contributions = largest_drop_category_contributions(events, rows, "overall_progress", ALL_CATEGORIES)
    summary = {
        "task_id": run_id,
        "coding_categories": [category.value for category in CODING_CATEGORIES],
        "excluded_categories": [category.value for category in EXCLUDED_CATEGORIES],
        "subtask_categories": category_map,
        "final_coding_progress": final["coding_progress"],
        "final_overall_progress": final["overall_progress"],
        "final_coding_complete_weight": final["coding_complete_weight"],
        "final_coding_active_weight": final["coding_active_weight"],
        "final_overall_complete_weight": final["overall_complete_weight"],
        "final_overall_active_weight": final["overall_active_weight"],
        "coding_largest_drop": coding_drop_detail["amount"],
        "overall_largest_drop": overall_drop_detail["amount"],
        "largest_coding_drop": coding_drop_detail["amount"],
        "largest_overall_drop": overall_drop_detail["amount"],
        "coding_nonmonotonic": coding_drop_detail["amount"] > EPSILON,
        "overall_nonmonotonic": overall_drop_detail["amount"] > EPSILON,
        "nonmonotonic_coding": coding_drop_detail["amount"] > EPSILON,
        "nonmonotonic_overall": overall_drop_detail["amount"] > EPSILON,
        "final_success": final_success,
        "final_success_source": final_success_source,
        "historical_subtasks_created": count_subtasks_created(annotated_events),
        "active_coding_leaves_final": final["coding_active_leaf_count"],
        "completed_coding_leaves_final": final["coding_complete_leaf_count"],
        "active_overall_leaves_final": final["overall_active_leaf_count"],
        "completed_overall_leaves_final": final["overall_complete_leaf_count"],
        "excluded_active_weight_final": final["excluded_active_weight"],
        "excluded_completed_weight_final": final["excluded_complete_weight"],
        "excluded_categories_final": {
            category.value: {
                "active": final[f"{category.value}_active_weight"],
                "complete": final[f"{category.value}_complete_weight"],
            }
            for category in EXCLUDED_CATEGORIES
        },
        "category_active_weight_final": {
            category.value: final[f"{category.value}_active_weight"] for category in ALL_CATEGORIES
        },
        "category_completed_weight_final": {
            category.value: final[f"{category.value}_complete_weight"] for category in ALL_CATEGORIES
        },
        "largest_coding_drop_source": largest_drop_source(events, rows, "coding_progress", CODING_CATEGORIES),
        "largest_overall_drop_source": largest_drop_source(events, rows, "overall_progress", ALL_CATEGORIES),
        "largest_coding_drop_category_contributions": coding_drop_contributions,
        "largest_overall_drop_category_contributions": overall_drop_contributions,
        "evidence_audit_status": evidence_audit["status"],
        "evidence_audit_by_category": evidence_audit["by_category"],
        "evidence_audit_weak_categories": evidence_audit["weak_categories"],
        "weak_completion_evidence_count": evidence_audit["weak_completion_evidence_count"],
        "weak_completion_evidence": evidence_audit["weak_completion_evidence"],
    }
    summary["largest_overall_drop_detail"] = overall_drop_detail
    summary["largest_coding_drop_detail"] = coding_drop_detail
    summary["largest_overall_drop_mostly_excluded"] = summary["largest_overall_drop_source"] in {
        SubtaskCategory.ARTIFACT.value,
        SubtaskCategory.DOCUMENTATION.value,
        SubtaskCategory.ENVIRONMENT.value,
    }
    (run_dir / "summary_by_category.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def annotate_categories(run_id: str, events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    annotated = deepcopy(events)
    category_map: dict[str, str] = {}
    for event in annotated:
        event_type = event["event_type"]
        if event_type == "add_subtask":
            subtask_id = event["subtask_id"]
            category = category_for(run_id, subtask_id, event["payload"]["description"], event["payload"].get("category"))
            event["payload"]["category"] = category.value
            category_map[subtask_id] = category.value
        elif event_type == "split_subtask":
            for child in event["payload"]["children"]:
                category = category_for(run_id, child["id"], child["description"], child.get("category"))
                child["category"] = category.value
                category_map[child["id"]] = category.value
    return annotated, dict(sorted(category_map.items()))


def category_for(run_id: str, subtask_id: str, description: str, explicit_category: str | None = None) -> SubtaskCategory:
    override = CATEGORY_OVERRIDES.get(run_id, {}).get(subtask_id)
    if override is not None:
        return override
    if explicit_category is not None:
        return SubtaskCategory(explicit_category)
    return infer_category(description)


def infer_category(description: str) -> SubtaskCategory:
    text = description.lower()
    if any(term in text for term in ("readme", "run notes", "transcript", "document")):
        return SubtaskCategory.DOCUMENTATION
    if any(term in text for term in ("artifact", "export", "ledger", "progress csv", "final diff", "bundle")):
        return SubtaskCategory.ARTIFACT
    if any(term in text for term in ("install", "tooling", "venv", "plugin", "environment")):
        return SubtaskCategory.ENVIRONMENT
    if any(term in text for term in ("test", "pytest", "regression", "validate", "validation", "verify")):
        return SubtaskCategory.VALIDATION
    if any(term in text for term in ("confirm", "compare", "contract", "baseline", "investigate")):
        return SubtaskCategory.INVESTIGATION
    return SubtaskCategory.PRODUCT


def count_subtasks_created(events: list[dict[str, Any]]) -> int:
    total = 0
    for event in events:
        if event["event_type"] == "add_subtask":
            total += 1
        elif event["event_type"] == "split_subtask":
            total += len(event["payload"]["children"])
    return total


def final_success_from_metadata(run_dir: Path, summary: dict[str, Any]) -> tuple[bool | None, str]:
    if isinstance(summary.get("final_success"), bool):
        return summary["final_success"], "summary.final_success"
    test_status = summary.get("test_status")
    if isinstance(test_status, str):
        normalized = test_status.lower()
        if normalized == "passed":
            return True, "summary.test_status"
        if normalized == "failed":
            return False, "summary.test_status"
    test_output = run_dir / "test_output.txt"
    if test_output.exists():
        text = test_output.read_text(errors="ignore").lower()
        if "failed" in text or "error" in text:
            return False, "inferred_from_test_output"
        if "passed" in text or " ok" in text:
            return True, "inferred_from_test_output"
    return None, "absent"


def progress_row(events, step: int) -> dict[str, Any]:
    ledger = replay([event for event in events if event.step <= step])
    overall = score(ledger)
    coding = score(ledger, categories=CODING_CATEGORIES)
    excluded = score(ledger, categories=EXCLUDED_CATEGORIES)
    row: dict[str, Any] = {
        "step": step,
        "overall_complete_weight": overall.complete_weight,
        "overall_active_weight": overall.active_weight,
        "overall_progress": overall.progress,
        "overall_complete_leaf_count": overall.complete_leaf_count,
        "overall_active_leaf_count": overall.active_leaf_count,
        "coding_complete_weight": coding.complete_weight,
        "coding_active_weight": coding.active_weight,
        "coding_progress": coding.progress,
        "coding_complete_leaf_count": coding.complete_leaf_count,
        "coding_active_leaf_count": coding.active_leaf_count,
        "excluded_complete_weight": excluded.complete_weight,
        "excluded_active_weight": excluded.active_weight,
        "excluded_progress": excluded.progress,
        "excluded_complete_leaf_count": excluded.complete_leaf_count,
        "excluded_active_leaf_count": excluded.active_leaf_count,
    }
    for category in ALL_CATEGORIES:
        obs = score(ledger, categories=(category,))
        row[f"{category.value}_complete_weight"] = obs.complete_weight
        row[f"{category.value}_active_weight"] = obs.active_weight
        row[f"{category.value}_progress"] = obs.progress
    return row


def write_progress_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "step",
        "overall_complete_weight",
        "overall_active_weight",
        "overall_progress",
        "overall_complete_leaf_count",
        "overall_active_leaf_count",
        "coding_complete_weight",
        "coding_active_weight",
        "coding_progress",
        "coding_complete_leaf_count",
        "coding_active_leaf_count",
        "excluded_complete_weight",
        "excluded_active_weight",
        "excluded_progress",
        "excluded_complete_leaf_count",
        "excluded_active_leaf_count",
    ]
    for category in ALL_CATEGORIES:
        fieldnames.extend(
            [
                f"{category.value}_complete_weight",
                f"{category.value}_active_weight",
                f"{category.value}_progress",
            ]
        )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def largest_drop(rows: list[dict[str, Any]], key: str) -> float:
    return largest_drop_detail(rows, key)["amount"]


def largest_drop_detail(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    best = {"amount": 0.0, "from_step": None, "to_step": None}
    for before, after in zip(rows, rows[1:]):
        drop = before[key] - after[key]
        if drop > best["amount"] + EPSILON:
            best = {"amount": drop, "from_step": before["step"], "to_step": after["step"]}
    return best


def is_nonmonotonic(rows: list[dict[str, Any]], key: str) -> bool:
    return largest_drop(rows, key) > EPSILON


def largest_drop_source(events, rows: list[dict[str, Any]], key: str, categories: tuple[SubtaskCategory, ...]) -> str:
    drop = largest_drop_detail(rows, key)["amount"]
    if drop <= EPSILON:
        return "none"

    contributions = largest_drop_category_contributions(events, rows, key, categories)
    material_sources = sorted(
        category
        for category, contribution in contributions.items()
        if contribution >= drop * DROP_MATERIALITY_THRESHOLD - EPSILON
    )
    if not material_sources:
        return "none"
    if len(material_sources) == 1:
        return material_sources[0]
    return "mixed"


def largest_drop_category_contributions(
    events,
    rows: list[dict[str, Any]],
    key: str,
    categories: tuple[SubtaskCategory, ...],
) -> dict[str, float]:
    detail = largest_drop_detail(rows, key)
    if detail["amount"] <= EPSILON:
        return {}

    before = next(row for row in rows if row["step"] == detail["from_step"])
    before_snapshot = active_leaf_snapshot(events, detail["from_step"], categories)
    after_snapshot = active_leaf_snapshot(events, detail["to_step"], categories)
    changed_ids = {
        leaf_id
        for leaf_id in set(before_snapshot) | set(after_snapshot)
        if before_snapshot.get(leaf_id) != after_snapshot.get(leaf_id)
    }
    contributions: dict[str, float] = {}
    before_complete = sum(leaf["complete_weight"] for leaf in before_snapshot.values())
    before_active = sum(leaf["active_weight"] for leaf in before_snapshot.values())
    before_progress = before["overall_progress" if key == "overall_progress" else "coding_progress"]

    for leaf_id in changed_ids:
        before_leaf = before_snapshot.get(leaf_id)
        after_leaf = after_snapshot.get(leaf_id)
        category = (after_leaf or before_leaf)["category"]
        delta_complete = (after_leaf or {}).get("complete_weight", 0.0) - (before_leaf or {}).get("complete_weight", 0.0)
        delta_active = (after_leaf or {}).get("active_weight", 0.0) - (before_leaf or {}).get("active_weight", 0.0)
        next_active = before_active + delta_active
        next_progress = (before_complete + delta_complete) / next_active if next_active else 0.0
        contributions[category] = contributions.get(category, 0.0) + max(0.0, before_progress - next_progress)

    return dict(sorted((category, round(contribution, 12)) for category, contribution in contributions.items() if contribution > EPSILON))


def active_leaf_snapshot(events, step: int, categories: tuple[SubtaskCategory, ...]) -> dict[str, dict[str, Any]]:
    ledger = replay([event for event in events if event.step <= step])
    selected = set(categories)
    active_ids = {sid for sid, subtask in ledger.subtasks.items() if subtask.status not in {Status.INVALIDATED, Status.DELETED}}
    parents = {
        subtask.parent_id
        for subtask in ledger.subtasks.values()
        if subtask.parent_id in active_ids and subtask.status not in {Status.INVALIDATED, Status.DELETED}
    }
    snapshot = {}
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


def largest_overall_drop_mostly_excluded(rows: list[dict[str, Any]]) -> bool:
    detail = largest_drop_detail(rows, "overall_progress")
    if detail["amount"] <= EPSILON:
        return False
    for before, after in zip(rows, rows[1:]):
        if before["step"] == detail["from_step"] and after["step"] == detail["to_step"]:
            coding_drop = before["coding_progress"] - after["coding_progress"]
            excluded_drop = before["excluded_progress"] - after["excluded_progress"]
            excluded_active_increase = after["excluded_active_weight"] > before["excluded_active_weight"] + EPSILON
            excluded_completed_decrease = after["excluded_complete_weight"] + EPSILON < before["excluded_complete_weight"]
            return (
                coding_drop <= detail["amount"] * 0.25
                and (excluded_drop > EPSILON or excluded_active_increase or excluded_completed_decrease)
            )
    return False


def audit_completion_evidence(events: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_by_subtask: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    categories: dict[str, SubtaskCategory] = {}
    weak: list[dict[str, Any]] = []
    audited_count = 0
    audited_by_category = {category: 0 for category in CODING_CATEGORIES}
    weak_by_category = {category: [] for category in CODING_CATEGORIES}

    for event in events:
        event_type = event["event_type"]
        subtask_id = event.get("subtask_id")
        payload = event["payload"]
        if event_type == "add_subtask":
            descriptions[subtask_id] = payload["description"]
            categories[subtask_id] = SubtaskCategory(payload.get("category", SubtaskCategory.PRODUCT.value))
        elif event_type == "split_subtask":
            parent_category = categories.get(subtask_id, SubtaskCategory.PRODUCT)
            for child in payload["children"]:
                descriptions[child["id"]] = child["description"]
                categories[child["id"]] = SubtaskCategory(child.get("category", parent_category.value))

        if subtask_id and "evidence" in payload:
            evidence_by_subtask.setdefault(subtask_id, []).extend(payload["evidence"])

        if event_type == "update_status" and payload.get("status") == Status.COMPLETE.value:
            category = categories.get(subtask_id, SubtaskCategory.PRODUCT)
            if category not in {SubtaskCategory.PRODUCT, SubtaskCategory.VALIDATION, SubtaskCategory.INVESTIGATION}:
                continue
            audited_count += 1
            audited_by_category[category] += 1
            evidence = evidence_by_subtask.get(subtask_id, [])
            evidence_types = sorted(classify_evidence(evidence))
            strong = bool(STRONG_EVIDENCE_TYPES & set(evidence_types))
            if (
                not strong
                and category is SubtaskCategory.INVESTIGATION
                and "contract_text" in evidence_types
                and any(term in descriptions.get(subtask_id, "").lower() for term in CONTRACT_UNDERSTANDING_TERMS)
            ):
                strong = True
            if not strong:
                weak_item = {
                    "step": event["step"],
                    "subtask_id": subtask_id,
                    "category": category.value,
                    "description": descriptions.get(subtask_id, ""),
                    "evidence_types": evidence_types or ["manual_note"],
                }
                weak.append(weak_item)
                weak_by_category[category].append(weak_item)

    if audited_count == 0:
        status = "not_applicable"
    elif weak:
        status = "weak"
    else:
        status = "strong"
    return {
        "status": status,
        "by_category": evidence_audit_by_category(audited_by_category, weak_by_category),
        "weak_categories": [
            category.value
            for category in CODING_CATEGORIES
            if weak_by_category[category]
        ],
        "weak_completion_evidence_count": len(weak),
        "weak_completion_evidence": weak,
    }


def evidence_audit_by_category(
    audited_by_category: dict[SubtaskCategory, int],
    weak_by_category: dict[SubtaskCategory, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    by_category = {}
    for category in CODING_CATEGORIES:
        audited_count = audited_by_category[category]
        weak_items = weak_by_category[category]
        if audited_count == 0:
            status = "not_applicable"
        elif weak_items:
            status = "weak"
        else:
            status = "strong"
        by_category[category.value] = {
            "status": status,
            "audited_completion_count": audited_count,
            "weak_completion_evidence_count": len(weak_items),
            "weak_subtask_ids": [item["subtask_id"] for item in weak_items],
        }
    return by_category


def classify_evidence(evidence: list[str]) -> set[str]:
    evidence_types: set[str] = set()
    for item in evidence:
        text = item.lower()
        for evidence_type, terms in EVIDENCE_PATTERNS:
            if any(term in text for term in terms):
                evidence_types.add(evidence_type)
        if SOURCE_PATH_PATTERN.search(item):
            evidence_types.add("diff")
    if not evidence_types:
        evidence_types.add("manual_note")
    return evidence_types


def write_suite_summary_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    table = suite_summary_table(summaries)
    if not path.exists():
        path.write_text("# Suite Summary\n\n" + "\n".join(table) + "\n")
        return

    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("| Run |")), None)
    if start is None:
        path.write_text(path.read_text().rstrip() + "\n\n" + "\n".join(table) + "\n")
        return

    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    updated = lines[:start] + table + lines[end:]
    path.write_text("\n".join(updated).rstrip() + "\n")


def suite_summary_table(summaries: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Run | final_coding_progress | final_overall_progress | coding_complete_weight_final | coding_active_weight_final | overall_complete_weight_final | overall_active_weight_final | active_coding_leaves_final | completed_coding_leaves_final | active_overall_leaves_final | completed_overall_leaves_final | historical_subtasks_created | coding_largest_drop | overall_largest_drop | largest_coding_drop_source | largest_overall_drop_source | excluded_active_weight_final | excluded_completed_weight_final | coding_nonmonotonic | final_success | final_success_source | evidence_audit_status | weak_completion_evidence_count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |",
    ]
    for summary in sorted(summaries, key=lambda item: run_sort_key(item["task_id"])):
        run_id = summary["task_id"]
        lines.append(
            "| `{run}` | {coding:.3f} | {overall:.3f} | {coding_complete_weight:.3f} | {coding_active_weight:.3f} | {overall_complete_weight:.3f} | {overall_active_weight:.3f} | {active_coding} | {completed_coding} | {active_overall} | {completed_overall} | {historical} | {coding_drop:.3f} | {overall_drop:.3f} | {coding_source} | {overall_source} | {excluded_active:.3f} | {excluded_completed:.3f} | {coding_nm} | {success} | {success_source} | {audit} | {weak_count} |".format(
                run=run_id,
                coding=summary["final_coding_progress"],
                overall=summary["final_overall_progress"],
                coding_complete_weight=summary["final_coding_complete_weight"],
                coding_active_weight=summary["final_coding_active_weight"],
                overall_complete_weight=summary["final_overall_complete_weight"],
                overall_active_weight=summary["final_overall_active_weight"],
                active_coding=summary["active_coding_leaves_final"],
                completed_coding=summary["completed_coding_leaves_final"],
                active_overall=summary["active_overall_leaves_final"],
                completed_overall=summary["completed_overall_leaves_final"],
                historical=summary["historical_subtasks_created"],
                coding_drop=summary["coding_largest_drop"],
                overall_drop=summary["overall_largest_drop"],
                coding_source=summary["largest_coding_drop_source"],
                overall_source=summary["largest_overall_drop_source"],
                excluded_active=summary["excluded_active_weight_final"],
                excluded_completed=summary["excluded_completed_weight_final"],
                coding_nm=yes_no(summary["coding_nonmonotonic"]),
                success=yes_no(summary["final_success"]) if isinstance(summary["final_success"], bool) else "unknown",
                success_source=summary["final_success_source"],
                audit=summary["evidence_audit_status"],
                weak_count=summary["weak_completion_evidence_count"],
            )
        )
    return lines


def load_original_summary(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    path = runs_dir / run_id / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_sort_key(run_id: str) -> tuple[int, int, str]:
    if run_id.startswith("task_"):
        parts = run_id.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            return (0, int(parts[1]), run_id)
    return (1, 0, run_id)


def write_category_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    remaining_nonmonotonic = [summary["task_id"] for summary in summaries if summary["coding_nonmonotonic"]]
    excluded_drop_sources = [
        summary["task_id"]
        for summary in summaries
        if summary["largest_overall_drop_source"] in {
            SubtaskCategory.ARTIFACT.value,
            SubtaskCategory.DOCUMENTATION.value,
            SubtaskCategory.ENVIRONMENT.value,
        }
    ]
    final_divergence = [
        summary["task_id"]
        for summary in summaries
        if abs(summary["final_overall_progress"] - summary["final_coding_progress"]) > EPSILON
    ]
    monotonic_incomplete_failures = [
        summary["task_id"]
        for summary in summaries
        if summary["final_success"] is False and not summary["coding_nonmonotonic"] and summary["final_coding_progress"] < 1.0
    ]
    high_progress_failures = [
        summary["task_id"]
        for summary in summaries
        if summary["final_success"] is False and summary["final_coding_progress"] >= 0.8
    ]
    weak_evidence_runs = [
        summary["task_id"]
        for summary in summaries
        if summary["weak_completion_evidence_count"] > 0
    ]
    excluded_weight = [
        summary["task_id"]
        for summary in summaries
        if summary["excluded_active_weight_final"] > EPSILON
    ]

    lines = [
        "# Suite Category Summary",
        "",
        "Coding progress includes product, validation, and investigation leaves. Artifact, documentation, and environment leaves are excluded from coding progress. The underlying scoring rule is unchanged: progress is completed active leaf weight divided by active leaf weight after normal leaf reduction.",
        "",
        *suite_summary_table(summaries),
        "",
        "## Non-monotonicity After Filtering",
        "",
        sentence_list(
            "Coding progress remains non-monotonic for",
            remaining_nonmonotonic,
            "No runs remain non-monotonic after category filtering.",
        ),
        "",
        "## Bookkeeping-driven Largest Drops",
        "",
        sentence_list(
            "The largest overall drop source is excluded artifact/documentation/environment work for",
            excluded_drop_sources,
            "No run has a largest overall drop primarily caused by excluded work.",
        ),
        "",
        "## Final Progress Divergence",
        "",
        sentence_list(
            "Final overall progress and final coding progress diverge for",
            final_divergence,
            "No run has final-value divergence between overall progress and coding progress.",
        ),
        "",
        "Runs with excluded active weight at the final step: "
        + (", ".join(f"`{run_id}`" for run_id in excluded_weight) if excluded_weight else "none")
        + ".",
        "",
        "## Failure Modes",
        "",
        sentence_list(
            "Monotonic incomplete failures",
            monotonic_incomplete_failures,
            "No monotonic incomplete failures are present.",
        ),
        "",
        sentence_list(
            "High-progress failed runs",
            high_progress_failures,
            "No high-progress failed runs are present.",
        ),
        "",
        "## Evidence Audit",
        "",
        sentence_list(
            "Runs with weak product/validation completion evidence",
            weak_evidence_runs,
            "No runs have weak product/validation completion evidence.",
        ),
        "",
        "## Audit Resolution",
        "",
        "Category filtering separates coding progress from run-management work without changing ledger scoring semantics or rewriting historical ledgers. The added controls make the remaining distinctions explicit: progress is not success, coding progress can differ from overall progress, failures can be monotonic, and weak evidence remains an audit finding rather than a replay failure.",
        "",
    ]
    path.write_text("\n".join(lines))


def sentence_list(prefix: str, items: list[str], empty: str) -> str:
    if not items:
        return empty
    return prefix + " " + ", ".join(f"`{item}`" for item in items) + "."


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
