from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_data_collection.artifacts import write_json
from coding_data_collection.ledger_sidecar_audit import corpus_ledger_sidecar_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ledger sidecar-generated artifact surfaces.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    report = corpus_ledger_sidecar_report([Path(run_dir) for run_dir in args.run_dirs])
    if args.output:
        write_json(Path(args.output), report)
    else:
        sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
