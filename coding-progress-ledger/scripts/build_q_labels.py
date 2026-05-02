"""Q2 — Build channel-native prediction labels for W3 checkpoints.

For each (run_id, step) pair in the W3 checkpoint table, compute the
five Q1 targets defined in `docs/Q_TARGETS.md`. Labels look strictly
*after* step S in the same run; features at step S never see them.
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

from ledger_progress import EventType, Status, SubtaskCategory, replay
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


def _load_events(run_dir: Path):
    events = []
    with (run_dir / "ledger.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(event_from_dict(json.loads(line)))
    return events


def _checkpoint_steps_by_run(checkpoint_csv: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with checkpoint_csv.open() as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["run_id"], []).append(row)
    for run_id in out:
        out[run_id].sort(key=lambda r: int(r["step"]))
    return out


def _coding_progress_at(events_so_far: list) -> float:
    return score(replay(events_so_far), CODING_CATEGORIES).progress


def _category_at_event(events_so_far: list, subtask_id: str | None) -> SubtaskCategory | None:
    if not subtask_id:
        return None
    ledger = replay(events_so_far)
    sub = ledger.subtasks.get(subtask_id)
    return sub.category if sub else None


def _label_future_progress_drop(
    events: list, checkpoint_step: int, horizon: int, current_progress: float
) -> bool:
    upper = checkpoint_step + horizon
    cur_events = [e for e in events if e.step <= checkpoint_step]
    for e in events:
        if e.step <= checkpoint_step:
            continue
        if e.step > upper:
            break
        cur_events.append(e)
        prog = score(replay(cur_events), CODING_CATEGORIES).progress
        if prog < current_progress - 1e-9:
            return True
    return False


def _label_product_reopened(events: list, checkpoint_step: int, horizon: int) -> bool:
    upper = checkpoint_step + horizon
    cur_events = [e for e in events if e.step <= checkpoint_step]
    for e in events:
        if e.step <= checkpoint_step:
            continue
        if e.step > upper:
            break
        if e.event_type is EventType.REOPEN_SUBTASK:
            cat = _category_at_event(cur_events, e.subtask_id)
            if cat is SubtaskCategory.PRODUCT:
                return True
        cur_events.append(e)
    return False


def _label_validation_exposes_new_work(
    events: list, checkpoint_step: int, horizon: int
) -> bool:
    upper = checkpoint_step + horizon
    cur_events = [e for e in events if e.step <= checkpoint_step]
    saw_validation_event = False
    for e in events:
        if e.step <= checkpoint_step:
            continue
        if e.step > upper:
            break
        if not saw_validation_event:
            if e.event_type is EventType.UPDATE_STATUS:
                new_status = e.payload.get("status")
                if new_status in {"complete", "blocked"}:
                    cat = _category_at_event(cur_events, e.subtask_id)
                    if cat is SubtaskCategory.VALIDATION:
                        saw_validation_event = True
                        cur_events.append(e)
                        continue
        else:
            if e.event_type is EventType.ADD_SUBTASK:
                cat_str = e.payload.get("category", "product")
                cat = SubtaskCategory(cat_str) if isinstance(cat_str, str) else cat_str
                if cat in {SubtaskCategory.PRODUCT, SubtaskCategory.INVESTIGATION}:
                    return True
            elif e.event_type is EventType.REOPEN_SUBTASK:
                cat = _category_at_event(cur_events, e.subtask_id)
                if cat in {SubtaskCategory.PRODUCT, SubtaskCategory.INVESTIGATION}:
                    return True
        cur_events.append(e)
    return False


def _label_stuck_loop_next_window(
    events: list,
    checkpoint_step: int,
    horizon: int,
    repeated_loop_already: bool,
) -> bool:
    if repeated_loop_already:
        return False
    upper = checkpoint_step + horizon
    for e in events:
        if e.step <= checkpoint_step:
            continue
        if e.step > upper:
            break
        if e.event_type is EventType.UPDATE_STATUS:
            new_status = e.payload.get("status")
            reason = (e.payload.get("reason") or e.reason or "").lower()
            if new_status == "blocked" and ("loop" in reason or "stuck" in reason):
                return True
    return False


def _label_submit_without_validation_terminal(rows_for_run: list[dict]) -> bool:
    last = rows_for_run[-1]
    return last.get("submit_without_validation", "false") == "true"


def build_labels(
    runs_dir: Path,
    checkpoint_csv: Path,
    horizon: int,
) -> list[dict]:
    by_run = _checkpoint_steps_by_run(checkpoint_csv)
    run_id_to_dir = {p.name: p for p in runs_dir.iterdir() if p.is_dir()}
    rows = []
    for run_id, ckpt_rows in by_run.items():
        run_dir = run_id_to_dir.get(run_id)
        if run_dir is None:
            continue
        events = _load_events(run_dir)
        terminal_swv = _label_submit_without_validation_terminal(ckpt_rows)
        for ckpt in ckpt_rows:
            step = int(ckpt["step"])
            cur_events = [e for e in events if e.step <= step]
            current_progress = _coding_progress_at(cur_events) if cur_events else 0.0
            repeated_loop_already = ckpt.get("repeated_observation_loop_flag", "false") == "true"
            rows.append({
                "run_id": run_id,
                "step": step,
                "horizon_steps": horizon,
                "future_progress_drop": _label_future_progress_drop(
                    events, step, horizon, current_progress),
                "product_reopened_after_completion": _label_product_reopened(
                    events, step, horizon),
                "validation_exposes_new_work": _label_validation_exposes_new_work(
                    events, step, horizon),
                "stuck_loop_next_window": _label_stuck_loop_next_window(
                    events, step, horizon, repeated_loop_already),
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
                row["run_id"],
                row["step"],
                row["horizon_steps"],
                "true" if row["future_progress_drop"] else "false",
                "true" if row["product_reopened_after_completion"] else "false",
                "true" if row["validation_exposes_new_work"] else "false",
                "true" if row["stuck_loop_next_window"] else "false",
                "true" if row["submit_without_validation_state"] else "false",
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
