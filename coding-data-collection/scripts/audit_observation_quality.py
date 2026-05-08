from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_data_collection.artifacts import write_json
from coding_data_collection.observation_quality import corpus_observation_quality_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute observation event quality metrics for run dirs.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--min-exit-code-coverage", type=float, default=1.0)
    parser.add_argument("--min-stdout-coverage", type=float, default=1.0)
    parser.add_argument("--min-stderr-coverage", type=float, default=1.0)
    args = parser.parse_args(argv)

    report = corpus_observation_quality_report([Path(run_dir) for run_dir in args.run_dirs])
    gates = {
        "exit_code_coverage": report["shell_exit_code_coverage"] >= args.min_exit_code_coverage,
        "stdout_snippet_coverage": report["shell_stdout_snippet_coverage"] >= args.min_stdout_coverage,
        "stderr_snippet_coverage": report["shell_stderr_snippet_coverage"] >= args.min_stderr_coverage,
        "per_run_quality": report["passed"],
    }
    report["gates"] = gates
    report["passed"] = all(gates.values())

    if args.output:
        write_json(Path(args.output), report)
    else:
        sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
