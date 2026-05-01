#!/usr/bin/env python3
"""J2: enforce explicit category on every add_subtask / split child.

Reads ledger.jsonl files under one or more run directory roots and
reports any event whose payload (or split child) is missing the
`category` field. Returns non-zero if any are found.

Legacy toy/control runs predate the native-category convention; pass
their roots via --legacy-root to acknowledge them as exempt without
silently passing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger_progress.core import EventType  # noqa: E402


def _load_events(ledger_path: Path) -> list[dict]:
    return [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def offending_events(ledger_path: Path) -> list[dict]:
    out = []
    for index, event in enumerate(_load_events(ledger_path)):
        et = event.get("event_type")
        sid = event.get("subtask_id")
        payload = event.get("payload") or {}
        if et == EventType.ADD_SUBTASK.value and "category" not in payload:
            out.append({"event_index": index, "step": event.get("step"), "subtask_id": sid, "event_type": et, "child_id": None})
        elif et == EventType.SPLIT_SUBTASK.value:
            for child in payload.get("children", []):
                if "category" not in child:
                    out.append({"event_index": index, "step": event.get("step"), "subtask_id": sid, "event_type": et, "child_id": child.get("id")})
    return out


def scan(roots: Iterable[Path], legacy_roots: set[Path]) -> dict:
    enforced: dict[str, list[dict]] = {}
    legacy_passed: list[str] = []
    for root in roots:
        for ledger in sorted(root.resolve().rglob("ledger.jsonl")):
            run_id = str(ledger.parent.relative_to(ROOT.resolve()))
            if any(_is_under(ledger.parent, lr) for lr in legacy_roots):
                legacy_passed.append(run_id)
                continue
            offenders = offending_events(ledger)
            if offenders:
                enforced[run_id] = offenders
    return {"violating_runs": enforced, "legacy_exempt_runs": sorted(legacy_passed)}


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], type=Path,
                        help="Root containing run dirs (repeat for multiple).")
    parser.add_argument("--legacy-root", action="append", default=[], type=Path,
                        help="Run-dir root to exempt (path filter; not silent).")
    args = parser.parse_args(argv)
    if not args.root:
        args.root = [ROOT / "runs/swe_agent_pilot", ROOT / "runs/swe_agent_pilot_v3"]
    legacy = {p.resolve() for p in args.legacy_root}
    report = scan(args.root, legacy)

    violating = report["violating_runs"]
    legacy_exempt = report["legacy_exempt_runs"]
    if legacy_exempt:
        print(f"legacy-exempt runs (path-filtered): {len(legacy_exempt)}")
        for run_id in legacy_exempt:
            print(f"  {run_id}")
    if not violating:
        print(f"all enforced runs use native categories ({sum(1 for _ in args.root)} root(s))")
        return 0
    for run_id, offenders in sorted(violating.items()):
        for o in offenders:
            tag = f"  step={o['step']}  event_index={o['event_index']}  event_type={o['event_type']}  subtask_id={o['subtask_id']}"
            if o["child_id"] is not None:
                tag += f"  child_id={o['child_id']}"
            print(f"{run_id}{tag}")
    print(f"\n{sum(len(v) for v in violating.values())} non-native event(s) across {len(violating)} run(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
