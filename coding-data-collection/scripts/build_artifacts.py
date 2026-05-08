from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_data_collection.estimator_artifacts import (
    build_estimator_artifacts,
    validate_estimator_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage collection runs and ask coding-estimator to build estimator artifacts."
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--source-id", default="terminal_bench_pilot")
    parser.add_argument("--estimator-root", default="../coding-estimator")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        help="Corpus root containing run directories with run_manifest.json.",
    )
    parser.add_argument("--replace-staged-runs", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir or f"datasets/{args.corpus_id}")
    if args.validate_only:
        report = validate_estimator_artifacts(artifact_dir)
    else:
        run_dirs = [Path(path) for path in args.run_dir]
        for root_path in args.run_root:
            root = Path(root_path)
            run_dirs.extend(
                sorted(child for child in root.iterdir() if child.is_dir() and (child / "run_manifest.json").is_file())
            )
        if not run_dirs:
            parser.error("--run-dir or --run-root is required unless --validate-only is set")
        result = build_estimator_artifacts(
            corpus_id=args.corpus_id,
            source_id=args.source_id,
            run_dirs=run_dirs,
            estimator_root=Path(args.estimator_root),
            artifact_dir=artifact_dir,
            replace_staged_runs=args.replace_staged_runs,
        )
        report = {
            **result.validation_report,
            "corpus_id": result.corpus_id,
            "source_id": result.source_id,
            "staged_runs_dir": str(result.staged_runs_dir),
            "source_manifest": str(result.manifest_path),
        }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
