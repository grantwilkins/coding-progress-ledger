"""Shape labels (slicing only, NOT prediction targets in v0).

Wraps the upstream `label_observation_shapes.label_run` snapshot
(`_upstream_shapes_snapshot.py`). Output: one row per (source, run_id)
with each upstream shape tag as a `shape_<tag>` boolean column.

Sources without `summary_by_category.json` (live sources) are skipped;
the upstream shape labeler hard-requires that file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from coding_estimator.ingest import paths
from coding_estimator.ingest.sources import SOURCES
from coding_estimator.io import write_parquet
from coding_estimator.labels._upstream_shapes_snapshot import SHAPE_TAGS, label_run


def shape_rows_for_source(source_id: str) -> list[dict]:
    if source_id not in SOURCES:
        raise KeyError(source_id)
    rows: list[dict] = []
    for run_id in paths.list_run_ids(source_id):
        run_dir = paths.run_dir(source_id, run_id)
        if not (run_dir / "summary_by_category.json").is_file():
            continue
        rl = label_run(run_dir)
        row: dict = {
            "run_id": run_id,
            "source": source_id,
            "final_coding_progress": rl.final_coding_progress,
            "final_success": rl.final_success,
            "final_success_source": rl.final_success_source,
            "clean_success": rl.clean_success,
        }
        for tag in SHAPE_TAGS:
            row[f"shape_{tag}"] = tag in rl.tags
        rows.append(row)
    return rows


def write_source_shape_labels(source_id: str, out_dir: Path) -> Path:
    rows = shape_rows_for_source(source_id)
    df = pd.DataFrame(rows)
    return write_parquet(
        df, out_dir / f"shapes_{source_id}.parquet", sort_by=["source", "run_id"]
    )
