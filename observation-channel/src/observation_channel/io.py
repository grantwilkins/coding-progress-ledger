from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator

from .models import Row, Summary, Turn


ROW_FIELDS = ["step", "total", "done", "current_category", "current_unit_age", "had_stuck_episode", "kind", "tool"]
SUMMARY_FIELDS = ["instance_id", "final_total", "final_done", "had_stuck_episode", "exit_status"]


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield obj


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_turns(path: Path) -> list[Turn]:
    return [Turn.from_json(obj) for obj in read_jsonl(path)]


def write_turns(path: Path, turns: Iterable[Turn]) -> None:
    write_jsonl(path, (turn.to_json() for turn in turns))


def write_rows_csv(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def write_summaries_csv(path: Path, summaries: Iterable[Summary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_csv_row())
