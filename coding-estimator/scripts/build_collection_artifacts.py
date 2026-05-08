#!/usr/bin/env python
"""Build estimator artifacts for a first-party collection source."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.build import write_source_checkpoints
from coding_estimator.checkpoints.features.registry import all_features
from coding_estimator.ingest.sources import SOURCES
from coding_estimator.labels.build import write_source_labels


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _write_feature_manifest(
    *,
    source: str,
    checkpoints: pd.DataFrame,
    labels: pd.DataFrame,
    out_path: Path,
) -> Path:
    payload = {
        "generator": "coding_estimator.scripts.build_collection_artifacts",
        "source": source,
        "checkpoint_rows": int(len(checkpoints)),
        "label_rows": int(len(labels)),
        "checkpoint_provenance_columns": [
            "checkpoint_step",
            "max_ledger_step_used",
            "max_observation_step_used",
        ],
        "feature_columns": [
            {
                "column_name": feature.column_name,
                "group": feature.group,
                "dtype": feature.dtype,
                "populated_on": list(feature.populated_on),
                "upstream_source": feature.upstream_source,
                "prefix_only": feature.prefix_only,
            }
            for feature in all_features()
        ],
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def _write_out_of_run_prevalence_predictions(
    *,
    checkpoints: pd.DataFrame,
    labels: pd.DataFrame,
    out_path: Path,
) -> Path:
    rows: list[dict] = []
    if not checkpoints.empty and not labels.empty:
        label_rows = labels[~labels["is_masked"].astype(bool)]
        joined = checkpoints[
            ["run_id", "source", "checkpoint_id", "checkpoint_step"]
        ].merge(
            label_rows[["run_id", "checkpoint_id", "target_name", "label_value"]],
            on=["run_id", "checkpoint_id"],
            how="inner",
        )
        for target_name, target in joined.groupby("target_name", sort=True):
            global_mean = float(target["label_value"].astype(float).mean())
            for _, row in target.iterrows():
                train = target[target["run_id"] != row["run_id"]]
                prediction = (
                    float(train["label_value"].astype(float).mean())
                    if not train.empty
                    else global_mean
                )
                rows.append(
                    {
                        "run_id": row["run_id"],
                        "source": row["source"],
                        "checkpoint_id": row["checkpoint_id"],
                        "checkpoint_step": int(row["checkpoint_step"]),
                        "target_name": target_name,
                        "model_name": "out_of_run_prevalence_baseline",
                        "prediction": prediction,
                    }
                )
    pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "source",
            "checkpoint_id",
            "checkpoint_step",
            "target_name",
            "model_name",
            "prediction",
        ],
    ).to_parquet(out_path, index=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path, checkpoints = write_source_checkpoints(
        args.source,
        args.out_dir / "checkpoints.parquet",
    )
    labels_path, _stats = write_source_labels(args.source, args.out_dir)
    canonical_labels_path = args.out_dir / "labels.parquet"
    shutil.copyfile(labels_path, canonical_labels_path)
    labels = pd.read_parquet(canonical_labels_path)

    predictions_path = _write_out_of_run_prevalence_predictions(
        checkpoints=checkpoints,
        labels=labels,
        out_path=args.out_dir / "estimator_predictions.parquet",
    )
    manifest_path = _write_feature_manifest(
        source=args.source,
        checkpoints=checkpoints,
        labels=labels,
        out_path=args.out_dir / "checkpoint_feature_manifest.json",
    )

    for path in (checkpoint_path, canonical_labels_path, predictions_path, manifest_path):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
