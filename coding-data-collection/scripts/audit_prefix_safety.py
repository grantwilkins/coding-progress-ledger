from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_data_collection.audits import prefix_safety_report, write_report
from coding_data_collection.observation import read_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit checkpoint prefix safety.")
    parser.add_argument("--checkpoints-json", required=True)
    parser.add_argument("--observation-events", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    checkpoints = json.loads(Path(args.checkpoints_json).read_text(encoding="utf-8"))
    observations = read_jsonl(Path(args.observation_events))
    report = prefix_safety_report(checkpoints, observations)
    write_report(Path(args.out), report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

