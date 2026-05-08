# H Estimator Artifact Production Status

Status: done for the current smoke corpus bridge.

Implementation boundary:

- `coding-data-collection` stages collection run directories and validates the
  artifact/provenance contract.
- `coding-estimator` owns checkpoint feature construction, label construction,
  prediction artifact construction, and feature manifest emission.
- No estimator model or feature logic lives in this repository.

Smoke corpus verification:

```text
source_id=terminal_bench_pilot
corpus_id=d_smoke_estimator
run_count=4
checkpoint_rows=19
prefix_provenance_complete=true
passed=true
```

Generated artifacts:

```text
datasets/d_smoke_estimator/checkpoints.parquet
datasets/d_smoke_estimator/labels.parquet
datasets/d_smoke_estimator/estimator_predictions.parquet
datasets/d_smoke_estimator/checkpoint_feature_manifest.json
datasets/d_smoke_estimator/estimator_source_manifest.json
reports/H_ESTIMATOR_ARTIFACT_REPORT.json
```

Prefix provenance gate:

- `max_ledger_step_used` exists for every checkpoint row.
- `max_observation_step_used` exists for every checkpoint row.
- Neither provenance column exceeds `checkpoint_step`.

Estimator-side source exposure:

- `terminal_bench_pilot` is registered in `coding-estimator`.
- Collection runs are staged under
  `../coding-estimator/runs/terminal_bench_pilot/` as symlinks to the
  collection run directories.
