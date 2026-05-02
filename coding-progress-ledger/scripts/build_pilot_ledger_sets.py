"""T4 — Wrap the 20 SWE-agent pilots as singleton LedgerSets and emit
a 20-member rollup set.

Singleton: runs/swe_agent_pilot/<pilot_id>/set.jsonl, one member with
weight 1.0 pointing to the pilot's ledger.jsonl (relative path).

Rollup: runs/swe_agent_pilot/pilot_rollup_set.jsonl, 20 members weight
1.0 each, member_id = pilot_id, ledger_ref relative to the rollup file's
parent directory.

No source_trace.json or ledger.jsonl is read for content beyond the
score; nothing is mutated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger_progress import LedgerSetSession


PILOT_DIR = ROOT / "runs" / "swe_agent_pilot"


def _pilot_ids(pilot_dir: Path) -> list[str]:
    return sorted(p.name for p in pilot_dir.iterdir() if p.is_dir() and p.name.startswith("swe_agent_pilot_"))


def write_singleton_sets(pilot_dir: Path) -> list[str]:
    written = []
    for pilot_id in _pilot_ids(pilot_dir):
        run = pilot_dir / pilot_id
        if not (run / "ledger.jsonl").exists():
            raise FileNotFoundError(f"missing ledger.jsonl in {run}")
        session = LedgerSetSession(pilot_id)
        session.add_member("ledger.jsonl", weight=1.0, member_id=pilot_id)
        out = run / "set.jsonl"
        session.export_jsonl(str(out))
        written.append(str(out.relative_to(ROOT)))
    return written


def write_rollup_set(pilot_dir: Path) -> str:
    session = LedgerSetSession("swe_agent_pilot_rollup")
    for pilot_id in _pilot_ids(pilot_dir):
        ref = f"{pilot_id}/ledger.jsonl"
        session.add_member(ref, weight=1.0, member_id=pilot_id)
    out = pilot_dir / "pilot_rollup_set.jsonl"
    session.export_jsonl(str(out))
    return str(out.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    args = parser.parse_args()

    singletons = write_singleton_sets(args.pilot_dir)
    rollup = write_rollup_set(args.pilot_dir)
    for s in singletons:
        print(s)
    print(rollup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
