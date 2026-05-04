# runs

Materialized run artifacts for toy controls, retrospective pilots, live instrumentation experiments, and validation scenarios.

## Common run artifacts

Most run directories include:

- `task.md`: run objective/spec.
- `ledger.jsonl`: append-only event source.
- `progress.csv`: replayed progress curve.
- `summary_by_category.json`: category-level progress summary.
- `run_notes.md`: qualitative interpretation and caveats.
- Optional provenance and outputs: `source_trace.json`, `normalized_trace.json`, `test_output.txt`, `final_diff.patch`, `live_instrumentation.json`, `wire_events.jsonl`.

## Major groups

- `task_*`, `control_*`, `negative_control_*`: synthetic toy/control benchmark runs.
- `swe_agent_pilot*`: retrospective SWE-agent pilot and reannotation outputs.
- `hermes_pilot*`: retrospective Hermes pilot outputs.
- `swe_agent_live*`: live SWE-agent instrumentation runs (wallclock and non-wallclock variants).
- `tb_live/`: live benchmark task runs.
- `live_validation/`: explicit scenario-based validation checks.

## Suite-level docs

- `SUITE_SUMMARY.md`: aggregate benchmark summary.
- `SUITE_CATEGORY_SUMMARY.md`: category-level aggregate breakdown.
- `LEDGER_AUDIT.md`: evidence/quality audit notes.
- `LIVE_VALIDATION_SUMMARY.md`: live validation rollup.

## Recompute helper

Use `ledger-run export-run <run_dir>` to regenerate derived CSV/summary artifacts from `ledger.jsonl`.
