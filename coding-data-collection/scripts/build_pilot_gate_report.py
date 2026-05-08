from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_data_collection.pilot_gates import pilot_gate_report, write_pilot_gate_outputs


def _run_dirs_from_args(paths: list[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for text in paths:
        path = Path(text)
        if (path / "run_manifest.json").is_file():
            run_dirs.append(path)
        else:
            run_dirs.extend(sorted(child for child in path.iterdir() if child.is_dir()))
    return run_dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Terminal-Bench pilot gate report.")
    parser.add_argument("paths", nargs="+", help="Run directories or corpus directories.")
    parser.add_argument("--estimator-artifact-dir", type=Path, required=True)
    parser.add_argument("--out", default="reports/PILOT_GATE_REPORT.json")
    parser.add_argument("--failure-out", default="reports/PILOT_FAILURE_ANALYSIS.md")
    args = parser.parse_args(argv)

    report = pilot_gate_report(
        _run_dirs_from_args(args.paths),
        estimator_artifact_dir=args.estimator_artifact_dir,
    )
    write_pilot_gate_outputs(
        report=report,
        report_path=Path(args.out),
        failure_path=Path(args.failure_out),
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
