"""Q2 — Build channel-native prediction labels for W3 checkpoints.

For each `(run_id, step)` pair in the W3 estimator checkpoint table,
emit five binary labels defined in `docs/Q_TARGETS.md`:

  future_progress_drop                horizon-dependent
  product_reopened_after_completion   horizon-dependent
  validation_exposes_new_work         horizon-dependent
  stuck_loop_next_window              horizon-dependent (masked)
  submit_without_validation_state     terminal (constant per run)

Window convention: half-open `(S, S+H]` for horizon-dependent labels.
Categories are resolved at event time, never from a future replay.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger_progress import EventType, SubtaskCategory, replay
from ledger_progress.queries import CODING_CATEGORIES
from ledger_progress.scoring import score
from ledger_progress.serialization import event_from_dict


TARGET_COLUMNS = (
    "future_progress_drop",
    "product_reopened_after_completion",
    "validation_exposes_new_work",
    "stuck_loop_next_window",
    "submit_without_validation_state",
)
OUTPUT_COLUMNS = ("run_id", "step", "horizon_steps", *TARGET_COLUMNS)
DISCOVERY_CATEGORIES = {SubtaskCategory.PRODUCT, SubtaskCategory.INVESTIGATION}


def _load_events(run_dir: Path) -> list:
    with (run_dir / "ledger.jsonl").open() as fh:
        return [event_from_dict(json.loads(line)) for line in fh if line.strip()]


def _checkpoints_by_run(checkpoint_csv: Path) -> dict[str, list[dict]]:
    by_run: dict[str, list[dict]] = {}
    with checkpoint_csv.open() as fh:
        for row in csv.DictReader(fh):
            by_run.setdefault(row["run_id"], []).append(row)
    for rows in by_run.values():
        rows.sort(key=lambda r: int(r["step"]))
    return by_run


def _events_through_step(events: list, step: int) -> list:
    return [e for e in events if e.step <= step]


def _events_in_open_window(events: list, low_exclusive: int, high_inclusive: int) -> list:
    return [e for e in events if low_exclusive < e.step <= high_inclusive]


def _coding_progress(events_so_far: list) -> float:
    return score(replay(events_so_far), CODING_CATEGORIES).progress if events_so_far else 0.0


def _category_at(events_so_far: list, subtask_id: str | None) -> SubtaskCategory | None:
    if not subtask_id:
        return None
    sub = replay(events_so_far).subtasks.get(subtask_id)
    return sub.category if sub else None


def _add_subtask_category(event) -> SubtaskCategory:
    raw = event.payload.get("category", "product")
    return raw if isinstance(raw, SubtaskCategory) else SubtaskCategory(raw)


def _is_validation_transition(event, prefix_events: list) -> bool:
    if event.event_type is not EventType.UPDATE_STATUS:
        return False
    if event.payload.get("status") not in {"complete", "blocked"}:
        return False
    return _category_at(prefix_events, event.subtask_id) is SubtaskCategory.VALIDATION


def _is_discovery_event(event, prefix_events: list) -> bool:
    if event.event_type is EventType.ADD_SUBTASK:
        return _add_subtask_category(event) in DISCOVERY_CATEGORIES
    if event.event_type is EventType.REOPEN_SUBTASK:
        return _category_at(prefix_events, event.subtask_id) in DISCOVERY_CATEGORIES
    return False


def _is_stuck_loop_block(event) -> bool:
    if event.event_type is not EventType.UPDATE_STATUS:
        return False
    if event.payload.get("status") != "blocked":
        return False
    reason = (event.payload.get("reason") or event.reason or "").lower()
    return "loop" in reason or "stuck" in reason


def label_future_progress_drop(
    events: list, checkpoint_step: int, horizon: int, current_progress: float
) -> bool:
    prefix = _events_through_step(events, checkpoint_step)
    for e in _events_in_open_window(events, checkpoint_step, checkpoint_step + horizon):
        prefix.append(e)
        if _coding_progress(prefix) < current_progress - 1e-9:
            return True
    return False


def label_product_reopened(events: list, checkpoint_step: int, horizon: int) -> bool:
    prefix = _events_through_step(events, checkpoint_step)
    for e in _events_in_open_window(events, checkpoint_step, checkpoint_step + horizon):
        if e.event_type is EventType.REOPEN_SUBTASK:
            if _category_at(prefix, e.subtask_id) is SubtaskCategory.PRODUCT:
                return True
        prefix.append(e)
    return False


def label_validation_exposes_new_work(
    events: list, checkpoint_step: int, horizon: int
) -> bool:
    prefix = _events_through_step(events, checkpoint_step)
    saw_validation = False
    for e in _events_in_open_window(events, checkpoint_step, checkpoint_step + horizon):
        if not saw_validation and _is_validation_transition(e, prefix):
            saw_validation = True
        elif saw_validation and _is_discovery_event(e, prefix):
            return True
        prefix.append(e)
    return False


def label_stuck_loop_next_window(
    events: list, checkpoint_step: int, horizon: int, repeated_loop_already: bool
) -> bool:
    if repeated_loop_already:
        return False
    for e in _events_in_open_window(events, checkpoint_step, checkpoint_step + horizon):
        if _is_stuck_loop_block(e):
            return True
    return False


def label_submit_without_validation_state(checkpoint_rows: list[dict]) -> bool:
    return checkpoint_rows[-1].get("submit_without_validation", "false") == "true"


# Backwards-compatible aliases for tests that imported the underscore-prefixed names.
_label_future_progress_drop = label_future_progress_drop
_label_product_reopened = label_product_reopened
_label_validation_exposes_new_work = label_validation_exposes_new_work
_label_stuck_loop_next_window = label_stuck_loop_next_window


def build_labels(runs_dir: Path, checkpoint_csv: Path, horizon: int) -> list[dict]:
    by_run = _checkpoints_by_run(checkpoint_csv)
    run_dir_for = {p.name: p for p in runs_dir.iterdir() if p.is_dir()}
    rows: list[dict] = []
    for run_id, ckpts in by_run.items():
        run_dir = run_dir_for.get(run_id)
        if run_dir is None:
            continue
        events = _load_events(run_dir)
        terminal_swv = label_submit_without_validation_state(ckpts)
        for ckpt in ckpts:
            step = int(ckpt["step"])
            prefix = _events_through_step(events, step)
            current_progress = _coding_progress(prefix)
            already_loop = ckpt.get("repeated_observation_loop_flag", "false") == "true"
            rows.append({
                "run_id": run_id,
                "step": step,
                "horizon_steps": horizon,
                "future_progress_drop": label_future_progress_drop(
                    events, step, horizon, current_progress),
                "product_reopened_after_completion": label_product_reopened(
                    events, step, horizon),
                "validation_exposes_new_work": label_validation_exposes_new_work(
                    events, step, horizon),
                "stuck_loop_next_window": label_stuck_loop_next_window(
                    events, step, horizon, already_loop),
                "submit_without_validation_state": terminal_swv,
            })
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow([
                row["run_id"], row["step"], row["horizon_steps"],
                *("true" if row[c] else "false" for c in TARGET_COLUMNS),
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-csv", type=Path, required=True)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    rows = build_labels(args.runs_dir, args.checkpoint_csv, args.horizon_steps)
    write_csv(rows, args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
