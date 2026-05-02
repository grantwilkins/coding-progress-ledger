#!/usr/bin/env python3
"""Hermes pilot importer (HP2). Mirrors import_swe_agent_trace.py."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.normalize_hermes_trace import normalize_row, render_summary


PRE_ANNOTATION_ARTIFACTS: Tuple[str, ...] = (
    "task.md",
    "source_trace.json",
    "normalized_trace.json",
    "trajectory_summary.md",
    "final_diff.patch",
    "test_output.txt",
    "run_notes.md",
    "source_metadata.json",
)

PLACEHOLDER_PATCH = "(no final_diff in source — Hermes traces ship no upstream patch)\n"
PLACEHOLDER_TEST = "(no test_output in source — Hermes traces ship no upstream eval logs)\n"

RUN_NOTES_TEMPLATE = """\
# Run notes — {pilot_id}

Source: Hermes retrospective pilot (HP2)
instance_id: `{instance_id}`
model_name: `{model_name}`
category: `{category}` / `{subcategory}`
final_success: null (Hermes has no upstream label)

## TODO — annotation
- [ ] Read trajectory_summary.md before opening source.
- [ ] Decide step-level subtasks per protocol.
- [ ] Capture evidence quotes (step indices) in payloads.
"""


@dataclass(frozen=True)
class PilotRow:
    pilot_id: str
    instance_id: str
    model_name: str
    category: str
    subcategory: str
    raw_path_or_dataset_index: str

    @classmethod
    def from_csv(cls, row: Dict[str, str]) -> "PilotRow":
        return cls(
            pilot_id=row["pilot_id"],
            instance_id=row["instance_id"],
            model_name=row["model_name"],
            category=row.get("category", ""),
            subcategory=row.get("subcategory", ""),
            raw_path_or_dataset_index=row.get("raw_path_or_dataset_index", ""),
        )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import Hermes pilot to run dirs.")
    p.add_argument("--sample-csv", required=True, type=Path)
    p.add_argument("--runs-dir", required=True, type=Path)
    p.add_argument("--raw-cache-dir", type=Path, default=None)
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args(argv)


def _load_pilots(path: Path) -> List[PilotRow]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [PilotRow.from_csv(r) for r in csv.DictReader(fh)]


def _resolve_cache_path(pilot: PilotRow, cache_dir: Path) -> Path:
    direct = cache_dir / f"{pilot.pilot_id}.json"
    if direct.is_file():
        return direct
    tail = pilot.raw_path_or_dataset_index.rsplit(":", 1)[-1]
    by_index = cache_dir / pilot.model_name / f"{tail}.json"
    if by_index.is_file():
        return by_index
    raise FileNotFoundError(
        f"raw cache missing for {pilot.pilot_id}: tried {direct} and {by_index}"
    )


def _write_task_md(run_dir: Path, normalized: Dict[str, object], pilot: PilotRow) -> None:
    issue = normalized.get("issue_text") or ""
    if not isinstance(issue, str):
        issue = ""
    if issue:
        body = (
            f"# Task — {pilot.pilot_id} ({pilot.instance_id})\n\n"
            f"{issue}\n"
        )
    else:
        body = (
            f"# Task — {pilot.pilot_id} ({pilot.instance_id})\n\n"
            "(no task description in source trace)\n"
        )
    (run_dir / "task.md").write_text(body, encoding="utf-8")


def _write_run_notes(run_dir: Path, pilot: PilotRow) -> None:
    body = RUN_NOTES_TEMPLATE.format(
        pilot_id=pilot.pilot_id,
        instance_id=pilot.instance_id,
        model_name=pilot.model_name,
        category=pilot.category,
        subcategory=pilot.subcategory,
    )
    (run_dir / "run_notes.md").write_text(body, encoding="utf-8")


def _write_source_metadata(run_dir: Path, pilot: PilotRow, normalized: Dict[str, object]) -> None:
    md = {
        "source": "hermes_agent_reasoning",
        "pilot_id": pilot.pilot_id,
        "instance_id": pilot.instance_id,
        "model_name": pilot.model_name,
        "category": pilot.category,
        "subcategory": pilot.subcategory,
        "final_success": None,
        "trajectory_length": normalized["trajectory_length"],
        "annotation_mode": "not_annotated",
    }
    (run_dir / "source_metadata.json").write_text(
        json.dumps(md, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_placeholders(run_dir: Path) -> None:
    (run_dir / "final_diff.patch").write_text(PLACEHOLDER_PATCH, encoding="utf-8")
    (run_dir / "test_output.txt").write_text(PLACEHOLDER_TEST, encoding="utf-8")


def import_one(pilot: PilotRow, cache_dir: Path, runs_dir: Path) -> Path:
    cache_path = _resolve_cache_path(pilot, cache_dir)
    run_dir = runs_dir / pilot.pilot_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache_path, run_dir / "source_trace.json")
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    normalized = normalize_row(raw, source="hermes_agent_reasoning", model_name=pilot.model_name)
    (run_dir / "normalized_trace.json").write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "trajectory_summary.md").write_text(render_summary(normalized), encoding="utf-8")
    _write_task_md(run_dir, normalized, pilot)
    _write_run_notes(run_dir, pilot)
    _write_placeholders(run_dir)
    _write_source_metadata(run_dir, pilot, normalized)
    return run_dir


def verify_run(run_dir: Path, pilot: PilotRow) -> List[str]:
    errors: List[str] = []
    for name in PRE_ANNOTATION_ARTIFACTS:
        p = run_dir / name
        if not p.is_file():
            errors.append(f"{run_dir.name}: missing {name}")
            continue
        if p.stat().st_size == 0:
            errors.append(f"{run_dir.name}: {name} is empty")
    if (run_dir / "ledger.jsonl").is_file():
        errors.append(f"{run_dir.name}: unexpected ledger.jsonl")
    return errors


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    pilots = _load_pilots(args.sample_csv)

    if args.verify_only:
        all_errors: List[str] = []
        for pilot in pilots:
            run_dir = args.runs_dir / pilot.pilot_id
            if not run_dir.is_dir():
                all_errors.append(f"{pilot.pilot_id}: run dir missing at {run_dir}")
                continue
            all_errors.extend(verify_run(run_dir, pilot))
        for e in all_errors:
            print(f"[import_hermes_trace] VERIFY: {e}", file=sys.stderr)
        if all_errors:
            return 1
        print(f"[import_hermes_trace] verify ok: {len(pilots)} run dirs", file=sys.stderr)
        return 0

    if args.raw_cache_dir is None:
        print("[import_hermes_trace] FATAL: --raw-cache-dir required unless --verify-only", file=sys.stderr)
        return 2

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    all_errors: List[str] = []
    for pilot in pilots:
        run_dir = import_one(pilot, args.raw_cache_dir, args.runs_dir)
        errors = verify_run(run_dir, pilot)
        for e in errors:
            print(f"[import_hermes_trace] VERIFY: {e}", file=sys.stderr)
        all_errors.extend(errors)
        print(f"[import_hermes_trace] imported {pilot.pilot_id} -> {run_dir}", file=sys.stderr)
    if all_errors:
        return 1
    print(f"[import_hermes_trace] imported {len(pilots)} run dirs into {args.runs_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
