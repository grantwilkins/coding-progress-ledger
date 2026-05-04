# datasets

Derived tables and analysis reports built from replayed ledger events.

## What lives here

- Observation datasets: `*_observations*.csv`, `ledger_observations_v0*.csv`.
- Checkpoint tables: `*_estimator_checkpoints*.csv`.
- Label/shape tables: `*_q_labels.csv`, `*_shape_labels.csv`.
- Audits: `*_audit.json`, `*_audit.md`.
- Summaries/reports: `*_summary.md`, `*_report.md`, `RESULTS_DISCLAIMERS.md`.

## Data characteristics

- Rebuildable from committed ledgers, manifests, and scripts.
- Deterministic outputs for fixed inputs.
- Research/diagnostic artifacts, not ground-truth performance claims.

## Regeneration

Primary builders live in `scripts/` (for example `build_ledger_observation_dataset.py`, `build_q_labels.py`, `build_estimator_checkpoints.py`, audit scripts).
