from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger_progress import active_incomplete_coding_leaves, from_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List active incomplete coding leaves from a ledger JSONL file.")
    parser.add_argument("ledger_jsonl", help="Path to ledger.jsonl")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)

    ledger = from_jsonl(args.ledger_jsonl)
    rows = [
        {
            "id": subtask.id,
            "description": subtask.description,
            "category": subtask.category.value,
            "status": subtask.status.value,
            "weight": subtask.weight,
        }
        for subtask in active_incomplete_coding_leaves(ledger)
    ]
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print("{id}\t{category}\t{status}\t{weight:g}\t{description}".format(**row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
