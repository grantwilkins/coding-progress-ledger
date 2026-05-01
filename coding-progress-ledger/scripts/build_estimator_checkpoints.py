"""W3 — Build the estimator checkpoint feature table.

One row per retained step from the step-level observation table. All
feature columns are derivable from events at-or-before the row's step
(no future leakage). Label columns (`final_success`, `finish_step`,
`success_by_horizon`, `shape_tags`) are explicitly suffixed and must
not be consumed as features by any downstream model.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger_progress import EventType, Status, SubtaskCategory, replay
from ledger_progress.queries import CODING_CATEGORIES
from ledger_progress.run_manager import resolve_final_success
from ledger_progress.scoring import score
from ledger_progress.serialization import event_from_dict


_RESCORE_PATH = ROOT / "scripts" / "rescore_suite_by_category.py"
_spec = importlib.util.spec_from_file_location("rescore_suite_by_category", _RESCORE_PATH)
_rescore = importlib.util.module_from_spec(_spec)
sys.modules["rescore_suite_by_category"] = _rescore
_spec.loader.exec_module(_rescore)
classify_evidence = _rescore.classify_evidence
STRONG_EVIDENCE_TYPES = _rescore.STRONG_EVIDENCE_TYPES


FEATURE_COLUMNS = [
    "run_id", "step",
    # frontier size
    "active_leaf_count", "active_coding_leaf_count", "active_validation_leaf_count",
    # closure
    "completed_leaf_count", "coding_progress", "validation_progress",
    # instability
    "num_reopens_so_far", "num_invalidations_so_far", "largest_progress_drop_so_far",
    # discovery
    "num_splits_so_far", "steps_since_new_subtask", "denominator_growth_so_far",
    # stalls
    "steps_since_completion", "blocked_leaf_count", "repeated_observation_loop_flag",
    # validation
    "validation_started", "validation_complete", "validation_failed",
    "submit_without_validation",
    # evidence
    "strong_completion_count", "manual_only_completion_count",
    "weak_product_completion_count",
]
LABEL_COLUMNS = [
    "label_final_success",
    "label_finish_step",
    "label_success_by_horizon",
    "label_shape_tags",
]
ALL_COLUMNS = FEATURE_COLUMNS + LABEL_COLUMNS


@dataclass
class _RunState:
    events_so_far: list
    num_reopens: int = 0
    num_invalidations: int = 0
    num_splits: int = 0
    largest_drop: float = 0.0
    last_coding: float = 0.0
    last_completion_step: int | None = None
    last_add_step: int | None = None
    initial_denom: float | None = None
    strong_completions: int = 0
    manual_only_completions: int = 0
    weak_product_completions: int = 0
    completed_subtask_ids: set = None
    repeated_loop_flag: bool = False
    validation_started: bool = False
    validation_complete: bool = False
    validation_failed: bool = False
    submit_without_validation: bool = False

    def __post_init__(self):
        if self.completed_subtask_ids is None:
            self.completed_subtask_ids = set()


def _load_events(run_dir: Path):
    events = []
    with (run_dir / "ledger.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(event_from_dict(json.loads(line)))
    return events


def _is_active(status: Status) -> bool:
    return status not in {Status.INVALIDATED, Status.DELETED}


def _frontier_features(ledger):
    active = [s for s in ledger.subtasks.values()
              if _is_active(s.status) and s.status is not Status.COMPLETE]
    leaves = []
    parent_ids = {s.parent_id for s in ledger.subtasks.values()
                  if s.parent_id is not None and _is_active(ledger.subtasks[s.parent_id].status)}
    for s in active:
        if s.id in parent_ids:
            continue
        leaves.append(s)
    coding_leaves = [s for s in leaves if s.category in CODING_CATEGORIES]
    val_leaves = [s for s in leaves if s.category is SubtaskCategory.VALIDATION]
    completed = [s for s in ledger.subtasks.values() if s.status is Status.COMPLETE]
    blocked = [s for s in ledger.subtasks.values() if s.status is Status.BLOCKED]
    return {
        "active_leaf_count": len(leaves),
        "active_coding_leaf_count": len(coding_leaves),
        "active_validation_leaf_count": len(val_leaves),
        "completed_leaf_count": len(completed),
        "blocked_leaf_count": len(blocked),
    }


def _closure_features(ledger):
    coding = score(ledger, CODING_CATEGORIES).progress
    val = score(ledger, (SubtaskCategory.VALIDATION,)).progress
    return {"coding_progress": coding, "validation_progress": val}


def _denominator(ledger) -> float:
    total = 0.0
    for s in ledger.subtasks.values():
        if s.category not in CODING_CATEGORIES:
            continue
        if not _is_active(s.status):
            continue
        total += s.weight
    return total


def _process_event(state: _RunState, event):
    state.events_so_far.append(event)
    ledger = replay(state.events_so_far)
    if event.event_type is EventType.REOPEN_SUBTASK:
        state.num_reopens += 1
    elif event.event_type is EventType.INVALIDATE_SUBTASK:
        state.num_invalidations += 1
    elif event.event_type is EventType.SPLIT_SUBTASK:
        state.num_splits += 1
    elif event.event_type is EventType.ADD_SUBTASK:
        state.last_add_step = event.step
    elif event.event_type is EventType.UPDATE_STATUS:
        sid = event.subtask_id
        new_status = event.payload.get("status")
        if new_status == "complete" and sid is not None and sid in ledger.subtasks:
            sub = ledger.subtasks[sid]
            if sid not in state.completed_subtask_ids:
                state.completed_subtask_ids.add(sid)
                state.last_completion_step = event.step
                evidence = event.payload.get("evidence", []) or []
                ev_types = classify_evidence(evidence) if evidence else {"manual_note"}
                strong = bool(STRONG_EVIDENCE_TYPES & ev_types)
                if strong:
                    state.strong_completions += 1
                else:
                    state.manual_only_completions += 1
                    if sub.category is SubtaskCategory.PRODUCT:
                        state.weak_product_completions += 1
            if sub.category is SubtaskCategory.VALIDATION:
                state.validation_complete = True
                state.validation_started = True
        elif new_status == "blocked" and sid is not None and sid in ledger.subtasks:
            sub = ledger.subtasks[sid]
            reason = (event.payload.get("reason") or event.reason or "").lower()
            if "loop" in reason or "stuck" in reason:
                state.repeated_loop_flag = True
            if sub.category is SubtaskCategory.VALIDATION:
                state.validation_failed = True
                state.validation_started = True
        elif new_status == "in_progress" and sid is not None and sid in ledger.subtasks:
            sub = ledger.subtasks[sid]
            if sub.category is SubtaskCategory.VALIDATION:
                state.validation_started = True

    new_coding = score(ledger, CODING_CATEGORIES).progress
    drop = max(0.0, state.last_coding - new_coding)
    if drop > state.largest_drop:
        state.largest_drop = drop
    state.last_coding = new_coding

    has_artifact_submit = any(
        s.category is SubtaskCategory.ARTIFACT
        and s.status is Status.COMPLETE
        and ("submit" in (s.description or "").lower()
             or any("submit" in ev.lower() for ev in s.evidence))
        for s in ledger.subtasks.values()
    )
    has_val_subtask = any(s.category is SubtaskCategory.VALIDATION for s in ledger.subtasks.values())
    state.submit_without_validation = has_artifact_submit and not (state.validation_complete and has_val_subtask)
    if has_val_subtask:
        for s in ledger.subtasks.values():
            if s.category is SubtaskCategory.VALIDATION and s.status is not Status.NOT_STARTED:
                state.validation_started = True

    if state.initial_denom is None:
        d = _denominator(ledger)
        if d > 0:
            state.initial_denom = d


def _row_at_step(state: _RunState, step: int) -> dict:
    ledger = replay(state.events_so_far)
    frontier = _frontier_features(ledger)
    closure = _closure_features(ledger)
    denom_now = _denominator(ledger)
    denom_growth = denom_now - state.initial_denom if state.initial_denom is not None else 0.0
    steps_since_new = step - state.last_add_step if state.last_add_step is not None else step
    steps_since_completion = (
        step - state.last_completion_step if state.last_completion_step is not None else step
    )
    return {
        "step": step,
        **frontier,
        **closure,
        "num_reopens_so_far": state.num_reopens,
        "num_invalidations_so_far": state.num_invalidations,
        "largest_progress_drop_so_far": round(state.largest_drop, 6),
        "num_splits_so_far": state.num_splits,
        "steps_since_new_subtask": steps_since_new,
        "denominator_growth_so_far": round(denom_growth, 6),
        "steps_since_completion": steps_since_completion,
        "repeated_observation_loop_flag": state.repeated_loop_flag,
        "validation_started": state.validation_started,
        "validation_complete": state.validation_complete,
        "validation_failed": state.validation_failed,
        "submit_without_validation": state.submit_without_validation,
        "strong_completion_count": state.strong_completions,
        "manual_only_completion_count": state.manual_only_completions,
        "weak_product_completion_count": state.weak_product_completions,
    }


def _retained_steps(step_csv: Path, run_id: str) -> list[int]:
    steps = []
    with step_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["run_id"] == run_id:
                steps.append(int(row["step"]))
    return steps


def _all_run_ids(step_csv: Path) -> list[str]:
    seen = []
    with step_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["run_id"] not in seen:
                seen.append(row["run_id"])
    return seen


def _load_shape_tags(shape_csv: Path | None) -> dict[str, str]:
    if shape_csv is None or not shape_csv.exists():
        return {}
    out = {}
    with shape_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tags = sorted(
                col for col in reader.fieldnames
                if col not in {"run_id", "final_success", "final_success_source",
                               "final_coding_progress", "clean_success"}
                and row.get(col) == "true"
            )
            out[row["run_id"]] = ";".join(tags)
    return out


def _format_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".") or "0"
    if v is None:
        return ""
    return str(v)


def build_checkpoints(
    runs_dir: Path,
    step_csv: Path,
    horizon: int,
    shape_labels_csv: Path | None,
) -> list[dict]:
    shape_tags = _load_shape_tags(shape_labels_csv)
    rows = []
    run_id_to_dir = {p.name: p for p in runs_dir.iterdir() if p.is_dir()}
    for run_id in _all_run_ids(step_csv):
        run_dir = run_id_to_dir.get(run_id)
        if run_dir is None:
            continue
        events = _load_events(run_dir)
        retained = _retained_steps(step_csv, run_id)
        retained_set = set(retained)
        finish_step = max((e.step for e in events), default=0)
        success, _ = resolve_final_success(run_dir)
        success_by_horizon = (success is True and finish_step <= horizon)

        state = _RunState(events_so_far=[])
        emitted_steps = set()
        i = 0
        for event in events:
            _process_event(state, event)
            i += 1
            next_event_step = events[i].step if i < len(events) else None
            cur_step = event.step
            if cur_step in retained_set and (next_event_step is None or next_event_step != cur_step):
                if cur_step in emitted_steps:
                    continue
                emitted_steps.add(cur_step)
                row = _row_at_step(state, cur_step)
                row["run_id"] = run_id
                row["label_final_success"] = success
                row["label_finish_step"] = finish_step
                row["label_success_by_horizon"] = success_by_horizon
                row["label_shape_tags"] = shape_tags.get(run_id, "")
                rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ALL_COLUMNS)
        for row in rows:
            writer.writerow([_format_value(row.get(col)) for col in ALL_COLUMNS])


def write_summary(rows: list[dict], out_path: Path, horizon: int, sources: dict) -> None:
    by_run = defaultdict(int)
    for row in rows:
        by_run[row["run_id"]] += 1
    lines = [
        "# Estimator checkpoint table — summary (W3)",
        "",
        "Derived feature table for downstream estimators. Each row is a",
        "checkpoint corresponding to one retained step from the step-level",
        "observation table. **All feature columns are derivable from events at",
        "or before the row's step**; nothing leaks future state.",
        "",
        f"- Source step CSV: `{sources['step_csv']}`",
        f"- Source runs dir: `{sources['runs_dir']}`",
        f"- Shape labels source: `{sources['shape_labels']}`" if sources.get("shape_labels") else "- Shape labels source: (none)",
        f"- Horizon for `label_success_by_horizon`: **{horizon} steps**",
        f"- Total checkpoints: **{len(rows)}**",
        f"- Distinct runs: **{len(by_run)}**",
        "",
        "## Column groups",
        "",
        "| Group | Columns |",
        "|---|---|",
        f"| frontier | active_leaf_count, active_coding_leaf_count, active_validation_leaf_count |",
        f"| closure | completed_leaf_count, coding_progress, validation_progress |",
        f"| instability | num_reopens_so_far, num_invalidations_so_far, largest_progress_drop_so_far |",
        f"| discovery | num_splits_so_far, steps_since_new_subtask, denominator_growth_so_far |",
        f"| stalls | steps_since_completion, blocked_leaf_count, repeated_observation_loop_flag |",
        f"| validation | validation_started, validation_complete, validation_failed, submit_without_validation |",
        f"| evidence | strong_completion_count, manual_only_completion_count, weak_product_completion_count |",
        f"| **labels (never features)** | label_final_success, label_finish_step, label_success_by_horizon, label_shape_tags |",
        "",
        "## Per-run checkpoint counts",
        "",
        "| run_id | checkpoints |",
        "|---|---:|",
    ]
    for run_id in sorted(by_run):
        lines.append(f"| `{run_id}` | {by_run[run_id]} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- `label_*` columns must not be used as features. Tests assert the",
        "  prefix; training pipelines should drop them at the schema layer.",
        "- `repeated_observation_loop_flag` keys on \"loop\"/\"stuck\" in the",
        "  block-event reason text. Live ledgers without explicit blocked",
        "  semantics will leave it false.",
        "- `denominator_growth_so_far` measures total active coding-category",
        "  weight added since the first non-empty checkpoint; SPLIT events",
        "  preserve denominator and won't move it.",
        "- Legacy retrospective rows (no timestamps) remain supported; the",
        "  table consumes step indices, not wall-clock seconds.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--step-csv", type=Path, required=True)
    parser.add_argument("--shape-labels", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=30,
                        help="Steps for label_success_by_horizon (default 30)")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    rows = build_checkpoints(args.runs_dir, args.step_csv, args.horizon, args.shape_labels)
    write_csv(rows, args.out_csv)
    write_summary(rows, args.out_summary, args.horizon, {
        "step_csv": args.step_csv.as_posix(),
        "runs_dir": args.runs_dir.as_posix(),
        "shape_labels": args.shape_labels.as_posix() if args.shape_labels else None,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
