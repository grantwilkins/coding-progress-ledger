#!/usr/bin/env python
"""Build the combined training inputs and gate artifacts.

Usage:
    uv run python scripts/prepare_training_artifacts.py \
        --datasets-dir datasets \
        --reports-dir reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_estimator.models.readiness import write_training_readiness_artifacts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts = write_training_readiness_artifacts(
        dataset_dir=args.datasets_dir,
        reports_dir=args.reports_dir,
    )
    for path in (
        artifacts.checkpoints_path,
        artifacts.labels_path,
        artifacts.manifest_path,
        artifacts.audit_path,
        artifacts.gate_report_path,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
