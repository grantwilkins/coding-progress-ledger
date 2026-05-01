"""Assign qualitative shape tags to ledger runs (W2).

Tags are *audit* labels derived from ledger structure plus run-note
citations; they are not training targets. Anchors per W4: the
`high_progress_failure` threshold is coding_progress >= 0.70; the
`no_validation_frontier` tag means no VALIDATION subtask was attempted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger_progress import EventType, Status, SubtaskCategory, replay
from ledger_progress.run_manager import resolve_final_success
from ledger_progress.serialization import event_from_dict


HIGH_PROGRESS_THRESHOLD = 0.70

SHAPE_TAGS = (
    "high_progress_failure",
    "low_progress_success",
    "stuck_loop",
    "submit_without_validation",
    "no_validation_frontier",
    "validation_induced_reopen",
    "scope_discovery_after_high_progress",
    "hidden_work_gap",
    "nonmonotone_recovery",
)

HIDDEN_WORK_PHRASES = (
    "hidden-work",
    "hidden work",
    "DID NOT trigger",
    "did not trigger",
    "uninformative",
    "repro was insufficient",
    "insufficient",
)

VALIDATION_REOPEN_PHRASES = (
    "repro",
    "traceback",
    "pytest",
    "re-run",
    "still emits",
    "still raises",
    "rerun",
)


@dataclass
class RunLabels:
    run_id: str
    final_coding_progress: float
    final_success: bool | None
    final_success_source: str
    tags: set[str] = field(default_factory=set)
    clean_success: bool = False
    notes: list[str] = field(default_factory=list)


def _load_events(run_dir: Path):
    events = []
    with (run_dir / "ledger.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(event_from_dict(json.loads(line)))
    return events


def _evidence_matches(ledger, phrases: tuple[str, ...]) -> bool:
    needles = tuple(p.lower() for p in phrases)
    for event in ledger.events:
        if event.event_type is not EventType.UPDATE_STATUS:
            continue
        for ev in event.payload.get("evidence", []) or []:
            if any(n in ev.lower() for n in needles):
                return True
    return False


def _has_blocked_subtask(ledger) -> bool:
    return any(s.status is Status.BLOCKED for s in ledger.subtasks.values())


def _has_validation_subtask(ledger) -> bool:
    return any(s.category is SubtaskCategory.VALIDATION for s in ledger.subtasks.values())


def _has_completed_validation(ledger) -> bool:
    return any(
        s.category is SubtaskCategory.VALIDATION and s.status is Status.COMPLETE
        for s in ledger.subtasks.values()
    )


def _has_completed_artifact_submit(ledger) -> bool:
    for sub in ledger.subtasks.values():
        if sub.category is not SubtaskCategory.ARTIFACT or sub.status is not Status.COMPLETE:
            continue
        desc = (sub.description or "").lower()
        if "submit" in desc:
            return True
    # Fall back to any artifact-complete with submit in its evidence text.
    for event in ledger.events:
        if event.event_type is not EventType.UPDATE_STATUS:
            continue
        sid = event.subtask_id
        if sid is None:
            continue
        sub = ledger.subtasks.get(sid)
        if sub is None or sub.category is not SubtaskCategory.ARTIFACT:
            continue
        if event.payload.get("status") != "complete":
            continue
        for ev in event.payload.get("evidence", []) or []:
            if "submit" in ev.lower():
                return True
    return False


def _validation_induced_reopen(ledger) -> bool:
    for event in ledger.events:
        if event.event_type is not EventType.REOPEN_SUBTASK:
            continue
        reason = (event.reason or "") + " " + str(event.payload.get("reason") or "")
        if any(phrase in reason.lower() for phrase in VALIDATION_REOPEN_PHRASES):
            return True
    return False


def _scope_discovery_after_high_progress(events) -> bool:
    """A PRODUCT or INVESTIGATION subtask is added *after* a REOPEN_SUBTASK
    event — i.e. the agent thought it was done and new coding work surfaced.
    Sequential annotation layout (add-then-complete) does not trigger this;
    only post-reopen coding additions do."""
    seen_reopen = False
    for event in events:
        if event.event_type is EventType.REOPEN_SUBTASK:
            seen_reopen = True
            continue
        if not seen_reopen:
            continue
        if event.event_type is EventType.ADD_SUBTASK:
            cat = event.payload.get("category")
            if cat in {"product", "investigation"}:
                return True
    return False


def _nonmonotone_recovery(ledger, summary: dict, success: bool | None) -> bool:
    if not summary.get("nonmonotonic_coding"):
        return False
    if not any(e.event_type is EventType.REOPEN_SUBTASK for e in ledger.events):
        return False
    if success is not True:
        return False
    return summary.get("final_coding_progress", 0.0) >= 1.0 - 1e-9


def label_run(run_dir: Path) -> RunLabels:
    events = _load_events(run_dir)
    ledger = replay(events)
    summary = json.loads((run_dir / "summary_by_category.json").read_text())
    success, source = resolve_final_success(run_dir, summary)
    coding = float(summary.get("final_coding_progress", 0.0))

    out = RunLabels(
        run_id=run_dir.name,
        final_coding_progress=coding,
        final_success=success,
        final_success_source=source,
    )
    if success is False and coding >= HIGH_PROGRESS_THRESHOLD:
        out.tags.add("high_progress_failure")
    if success is True and coding < HIGH_PROGRESS_THRESHOLD:
        out.tags.add("low_progress_success")
    if _has_blocked_subtask(ledger):
        out.tags.add("stuck_loop")
    if not _has_validation_subtask(ledger):
        out.tags.add("no_validation_frontier")
    if _has_completed_artifact_submit(ledger) and not _has_completed_validation(ledger):
        out.tags.add("submit_without_validation")
    if _validation_induced_reopen(ledger):
        out.tags.add("validation_induced_reopen")
    if _scope_discovery_after_high_progress(events):
        out.tags.add("scope_discovery_after_high_progress")
    if _evidence_matches(ledger, HIDDEN_WORK_PHRASES):
        out.tags.add("hidden_work_gap")
    if _nonmonotone_recovery(ledger, summary, success):
        out.tags.add("nonmonotone_recovery")

    out.clean_success = (
        success is True
        and "low_progress_success" not in out.tags
        and "submit_without_validation" not in out.tags
        and "no_validation_frontier" not in out.tags
        and "high_progress_failure" not in out.tags
    )
    return out


def label_runs(runs_dir: Path) -> list[RunLabels]:
    rows = []
    for sub in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not (sub / "ledger.jsonl").is_file():
            continue
        if not (sub / "summary_by_category.json").is_file():
            continue
        rows.append(label_run(sub))
    return rows


def write_csv(rows: list[RunLabels], out_path: Path) -> None:
    fields = [
        "run_id",
        "final_success",
        "final_success_source",
        "final_coding_progress",
        "clean_success",
        *SHAPE_TAGS,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([
                row.run_id,
                "" if row.final_success is None else str(row.final_success),
                row.final_success_source,
                f"{row.final_coding_progress:.4f}",
                "true" if row.clean_success else "false",
                *("true" if tag in row.tags else "false" for tag in SHAPE_TAGS),
            ])


def write_report(rows: list[RunLabels], out_path: Path, runs_dir: Path) -> None:
    counts = Counter()
    for r in rows:
        for tag in r.tags:
            counts[tag] += 1
    lines = [
        f"# Shape labels report for `{runs_dir.as_posix()}`",
        "",
        "Shape tags are **audit labels**, not final model targets. They are derived",
        "from ledger structure (status, category, REOPEN events) plus evidence-text",
        "citations. Anchors per W4: `high_progress_failure` fires at",
        f"coding_progress >= {HIGH_PROGRESS_THRESHOLD}; `no_validation_frontier`",
        "means no VALIDATION subtask was attempted.",
        "",
        f"Runs labeled: **{len(rows)}**",
        "",
        "## Tag counts",
        "",
        "| Tag | Count |",
        "|---|---:|",
    ]
    for tag in SHAPE_TAGS:
        lines.append(f"| `{tag}` | {counts.get(tag, 0)} |")
    lines.append(f"| `clean_success` | {sum(1 for r in rows if r.clean_success)} |")
    lines += ["", "## Per-run tags", "",
              "| run_id | success | coding | clean | tags |",
              "|---|:---:|---:|:---:|---|"]
    for r in rows:
        tags = ", ".join(sorted(r.tags)) or "—"
        succ = "" if r.final_success is None else ("✓" if r.final_success else "✗")
        clean = "✓" if r.clean_success else ""
        lines.append(
            f"| `{r.run_id}` | {succ} | {r.final_coding_progress:.3f} | {clean} | {tags} |"
        )
    lines += ["", "## Caveats", "",
              "- Labels are derived from ledger fields plus evidence text.",
              "- `hidden_work_gap` requires explicit annotator-cited phrases (e.g. ",
              "  \"DID NOT trigger\", \"hidden-work\"); silent gaps will not be flagged.",
              "- `validation_induced_reopen` keys on reason strings naming repro / ",
              "  pytest / Traceback / re-run / still emits.",
              "- These tags are intended as the audit surface that distinguishes a",
              "  progress=1.0 success from a progress=1.0 submit-without-test; they",
              "  are not yet training labels for any predictive model.",
              ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, help="Directory containing run subdirectories")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = label_runs(args.runs_dir)
    write_csv(rows, args.csv)
    write_report(rows, args.report, args.runs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
