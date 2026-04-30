#!/usr/bin/env python3
"""Spec-driven annotation driver (D4/E1 scaffolding).

Each annotated pilot is a pair of files in ``--specs-dir``:

    <pilot_id>.json        # event spec + quality fields
    <pilot_id>.notes.md    # human prose for run_notes.md (with placeholders)

The driver replays the spec through ``LedgerSession``, exports
``ledger.jsonl`` into the matching ``--runs-dir/<pilot_id>/``, calls
``ledger-run export-run`` to derive progress.csv etc., substitutes
``{{PROGRESS_OVERALL}}`` and ``{{PROGRESS_CODING}}`` in the notes
template, and writes ``annotation_quality.json``. Idempotent.

This driver is **source-agnostic** — the specs say which run dir they
target. It works for SWE-agent, future scaffolds, or any source whose
runs follow the framework's run-dir layout.

Spec format (JSON):

    {
      "pilot_id": "swe_agent_pilot_s_01",
      "instance_id": "...",                 # informational; sanity check
      "root_task": "...",
      "events": [
        {"op": "add",        "step": N, "id": "S1", "category": "INVESTIGATION", "description": "..."},
        {"op": "complete",   "step": N, "id": "S1", "evidence": "..." | ["...", ...]},
        {"op": "start",      "step": N, "id": "S1", "evidence": "...?"},
        {"op": "block",      "step": N, "id": "S1", "reason": "...", "evidence": "...?"},
        {"op": "reopen",     "step": N, "id": "S1", "reason": "..."},
        {"op": "invalidate", "step": N, "id": "S1", "reason": "..."},
        {"op": "split",      "step": N, "id": "S1", "reason": "...",
         "children": [{"description": "...", "category": "PRODUCT"}]}
      ],
      "quality": {                          # D5 fields except number_of_subtasks (driver fills)
        "annotator": "...",
        "annotation_time_minutes": 0,
        "number_of_uncertain_events": 0,
        "number_of_evidence_gaps": 0,
        "whether_final_success_used_only_at_end": true,
        "whether_progress_forced": false,
        "whether_schema_gap_found": false
      }
    }

Subtask `id` values in the spec MUST match LedgerSession's
auto-numbering (``S1``, ``S2``, ...) in the order events appear. The
driver asserts this so a typo or reordering bug fails loudly rather
than silently mis-targeting an event.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress import LedgerSession, SubtaskCategory  # noqa: E402
from ledger_progress.queries import CODING_CATEGORIES  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORY_BY_NAME: Dict[str, SubtaskCategory] = {c.name: c for c in SubtaskCategory}


def _category(name: str) -> SubtaskCategory:
    if name not in CATEGORY_BY_NAME:
        raise ValueError(
            f"unknown category {name!r}; expected one of {sorted(CATEGORY_BY_NAME)}"
        )
    return CATEGORY_BY_NAME[name]


def _op_add(s: LedgerSession, ev: Dict[str, Any]) -> None:
    sid = s.add(
        ev["description"],
        step=ev["step"],
        category=_category(ev["category"]),
        weight=ev.get("weight", 1.0),
        parent_id=ev.get("parent_id"),
    )
    asserted = ev.get("id")
    if asserted is not None and sid != asserted:
        raise ValueError(
            f"event at step {ev['step']!r}: spec asserts id={asserted!r} but session generated {sid!r}; "
            "events out of insertion order or id renumbering needed"
        )


def _op_start(s: LedgerSession, ev: Dict[str, Any]) -> None:
    s.start(ev["id"], step=ev["step"], evidence=ev.get("evidence"))


def _op_complete(s: LedgerSession, ev: Dict[str, Any]) -> None:
    s.complete(ev["id"], ev["evidence"], step=ev["step"])


def _op_block(s: LedgerSession, ev: Dict[str, Any]) -> None:
    s.block(ev["id"], step=ev["step"], reason=ev["reason"], evidence=ev.get("evidence"))


def _op_reopen(s: LedgerSession, ev: Dict[str, Any]) -> None:
    s.reopen(ev["id"], step=ev["step"], reason=ev["reason"])


def _op_invalidate(s: LedgerSession, ev: Dict[str, Any]) -> None:
    s.invalidate(ev["id"], step=ev["step"], reason=ev["reason"])


def _op_split(s: LedgerSession, ev: Dict[str, Any]) -> None:
    descs = [c["description"] for c in ev["children"]]
    cats: List[Optional[SubtaskCategory]] = [
        _category(c["category"]) if "category" in c else None for c in ev["children"]
    ]
    use_cats = cats if any(c is not None for c in cats) else None
    s.split(ev["id"], descs, step=ev["step"], reason=ev["reason"], categories=use_cats)


OP_HANDLERS: Dict[str, Callable[[LedgerSession, Dict[str, Any]], None]] = {
    "add": _op_add,
    "start": _op_start,
    "complete": _op_complete,
    "block": _op_block,
    "reopen": _op_reopen,
    "invalidate": _op_invalidate,
    "split": _op_split,
}


def build_session(spec: Dict[str, Any]) -> LedgerSession:
    """Construct a LedgerSession by replaying the spec's events."""
    s = LedgerSession(spec["root_task"])
    for ev in spec["events"]:
        op = ev.get("op")
        handler = OP_HANDLERS.get(op)
        if handler is None:
            raise ValueError(
                f"unknown op {op!r}; expected one of {sorted(OP_HANDLERS)}"
            )
        handler(s, ev)
    return s


def emit_one(spec_path: Path, runs_dir: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    notes_path = spec_path.parent / (spec_path.stem + ".notes.md")
    if not notes_path.is_file():
        raise FileNotFoundError(
            f"notes file expected at {notes_path} (paired with {spec_path.name})"
        )
    notes_template = notes_path.read_text(encoding="utf-8")

    pilot_id = spec["pilot_id"]
    run_dir = runs_dir / pilot_id
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"run dir {run_dir} not found; run the importer first"
        )

    session = build_session(spec)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))

    subprocess.run(
        ["uv", "run", "ledger-run", "export-run", str(run_dir)],
        check=True,
        cwd=str(REPO_ROOT),
    )

    overall = session.score()
    coding = session.score(categories=CODING_CATEGORIES)
    notes_body = notes_template.replace(
        "{{PROGRESS_OVERALL}}", f"{overall.progress:.2f}"
    ).replace(
        "{{PROGRESS_CODING}}", f"{coding.progress:.2f}"
    )
    (run_dir / "run_notes.md").write_text(notes_body, encoding="utf-8")

    quality = dict(spec.get("quality", {}))
    quality["number_of_subtasks"] = len(session.ledger.subtasks)
    (run_dir / "annotation_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"[annotate_pilots_from_spec] {pilot_id}: "
        f"{quality['number_of_subtasks']} subtasks, "
        f"progress={overall.progress:.3f} (coding={coding.progress:.3f})"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay annotation specs into ledger artifacts.")
    parser.add_argument("--specs-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Optional: limit to one or more pilot_ids (matched against spec basename).",
    )
    args = parser.parse_args(argv)

    specs = sorted(args.specs_dir.glob("*.json"))
    if not specs:
        raise FileNotFoundError(f"no .json specs in {args.specs_dir}")

    if args.only:
        wanted = set(args.only)
        specs = [p for p in specs if p.stem in wanted]
        if not specs:
            raise FileNotFoundError(
                f"no specs matched --only {sorted(wanted)} in {args.specs_dir}"
            )

    for spec_path in specs:
        emit_one(spec_path, args.runs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
