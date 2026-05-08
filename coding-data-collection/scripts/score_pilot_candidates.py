from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_data_collection.task_scoring import write_candidate_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score and select Terminal-Bench pilot candidates.")
    parser.add_argument(
        "--input",
        default="manifests/pilots/terminal_bench_candidate_calibration.csv",
    )
    parser.add_argument(
        "--output",
        default="manifests/pilots/terminal_bench_candidate_scores.csv",
    )
    parser.add_argument("--selected-count", type=int, default=12)
    args = parser.parse_args(argv)

    path = write_candidate_scores(
        Path(args.input),
        Path(args.output),
        selected_count=args.selected_count,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
