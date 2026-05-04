# scripts

Deterministic build/import/audit scripts for inventories, pilot runs, observation datasets, and reports.

## Categories

- Source prep: `*_inventory.py`, `sample_*_pilot*.py`, `populate_*_pilot_cache.py`.
- Import/normalize: `normalize_*_trace.py`, `import_*_trace.py`.
- Annotation + sets: `annotate_pilots_from_spec.py`, `build_pilot_ledger_sets.py`.
- Dataset builders: `build_ledger_observation_dataset.py`, `build_q_labels.py`, `build_estimator_checkpoints.py`, `label_observation_shapes.py`.
- Audits/comparisons: `audit_*`, `compare_annotations.py`, `collect_schema_gaps.py`, `rescore_suite_by_category.py`.
- Live/TB ops: `run_swe_agent_live_sidecar.py`, `tb_emit.py`, `validate_tb_run.py`.

## Usage

Run scripts from repo root with `uv run python scripts/<name>.py ...`.

Scripts are expected to fail loudly on invalid inputs.
