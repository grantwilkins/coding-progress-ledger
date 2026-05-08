# Artifact Layout

Required for every completed run:

```text
runs/<corpus>/<run_id>/
  task.md
  task_metadata.json
  environment_manifest.json
  protocol_manifest.json
  transcript.jsonl
  observation_events.jsonl
  events.jsonl
  ledger.jsonl
  progress.csv
  progress_by_category.csv
  summary_by_category.json
  run_manifest.json
  verifier_output.txt
  run_notes.md
```

Required after estimator build:

```text
checkpoints.parquet
labels.parquet
estimator_predictions.parquet
checkpoint_feature_manifest.json
```

Conditional:

```text
final_diff.patch
patch_predictions.json
container_logs.txt
docker_build.log
harbor_job_metadata.json
swe_bench_patch.json
```

Infrastructure failures still require:

```text
task_metadata.json
environment_manifest.json
protocol_manifest.json
run_manifest.json
run_notes.md
```

`artifact_incomplete` follows the same minimum manifest set and preserves any
additional partial artifacts for audit. It is never treated as a completed
terminal outcome.

Schema validation:

```bash
uv run python scripts/validate_run.py runs/<corpus>/<run_id>
```
