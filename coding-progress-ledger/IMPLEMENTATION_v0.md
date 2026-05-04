# IMPLEMENTATION v0

This document describes what is implemented in this repository today for coding-progress ledgers, how SWE-agent/Hermes retrospective pipelines are built, and what is validated.

It is a methodology and infrastructure handoff, not a model-performance claim.

## 1. Scope

The repository provides:

- An append-only ledger model for coding progress.
- Deterministic replay/scoring/serialization.
- Source ingestion pipelines for SWE-agent and Hermes traces.
- Retrospective annotation workflows that materialize ledger events from visible trace evidence.
- Derived observation datasets and audits.
- Live-run sidecar/run-manager primitives for forward instrumentation.

The repository explicitly does not claim that progress equals correctness or that progress alone predicts final success.

## 2. Core Ledger Semantics

Package: `ledger_progress/`.

Progress definition:

```text
progress = completed active leaf weight / total active leaf weight
```

Semantics:

- Event log is the source of truth (`ledger.jsonl`).
- Only active leaves are scored.
- Splits, reopens, invalidations, and new discoveries can decrease progress.
- Completion requires evidence.
- Progress is orthogonal to final success labels.

Primary event/status/category enums are in `ledger_progress/core.py`.

## 3. Key Package Components

- `session.py`: write API (`add`, `start`, `complete`, `block`, `reopen`, `split`, `invalidate`, export methods).
- `scoring.py`: score computation on replayed state.
- `queries.py`: query helpers and `CODING_CATEGORIES`.
- `serialization.py`: JSONL/dataclass IO.
- `run_manager.py`: `ledger-run` CLI (`init-run`, `export-run`, `capture-tests`, `capture-diff`, `check-run`, `summarize-run`).
- `sidecar.py`: live instrumentation support.
- `set_core.py`, `set_session.py`, `set_serialization.py`: set-level abstractions.
- `adapters/`: source-specific mapping helpers.

## 4. Source Pipelines

### 4.1 SWE-agent pipeline

Directories:

- `external_data/swe_agent/`: source policy, manifests, sampling policy.
- `annotations/swe_agent_pilot*`: retrospective annotation specs.
- `runs/swe_agent_pilot*`, `runs/swe_agent_live*`: materialized runs.
- `datasets/`: derived observation/checkpoint/label artifacts.

Core scripts:

- `scripts/swe_agent_inventory.py`
- `scripts/sample_swe_agent_pilot.py`
- `scripts/populate_swe_agent_pilot_cache.py`
- `scripts/normalize_swe_agent_trace.py`
- `scripts/import_swe_agent_trace.py`
- `scripts/annotate_pilots_from_spec.py`

Determinism choices:

- Stable filtering/sorting and seeded sampling.
- Explicit parse-status reporting instead of silent drops.
- Byte-stable CSV writing.

### 4.2 Hermes pipeline

Directories:

- `external_data/hermes/`: source/policy/manifests/cache artifacts.
- `annotations/hermes_pilot/`: annotation specs.
- `runs/hermes_pilot*`: materialized retrospective runs.
- `datasets/`: Hermes observation/checkpoint/label outputs.

Core scripts:

- `scripts/hermes_inventory.py`
- `scripts/sample_hermes_pilot.py`, `scripts/sample_hermes_pilot_v2.py`
- `scripts/normalize_hermes_trace.py`
- `scripts/import_hermes_trace.py`
- `scripts/auto_annotate_hermes.py` (plus audit/compare passes)

## 5. Retrospective Annotation Protocol

High-level flow:

1. Select deterministic pilot samples from source inventories.
2. Normalize/import each source trace into a run directory.
3. Apply annotation specs to generate ledger events.
4. Replay events to produce progress curves and category summaries.
5. Build event/step observation datasets.
6. Run audits for evidence quality, schema consistency, and annotation agreement.

Important separation:

- Source trace: immutable input.
- Retrospective ledger: human/automation-derived annotation artifact.
- Observation dataset: derived replay artifact.
- Completion label: external final outcome metadata.

## 6. Dataset/Reporting Layer

Primary builders (scripts):

- `build_ledger_observation_dataset.py`
- `build_q_labels.py`
- `build_estimator_checkpoints.py`
- `label_observation_shapes.py`
- `q_baselines.py`
- `smoke_test_completion_prediction.py`

Audits/comparisons:

- `audit_ledger_observation_dataset.py`
- `audit_pilot_evidence.py`
- `compare_annotations.py`
- `collect_schema_gaps.py`
- `build_live_parity_report.py`

Outputs live in `datasets/` and `runs/*` summary docs.

## 7. Live Instrumentation Surface

Current live-oriented assets include:

- `scripts/run_swe_agent_live_sidecar.py`
- `ledger_progress/sidecar.py`
- `ledger_progress/run_manager.py`
- TB task tooling (`scripts/tb_emit.py`, `scripts/validate_tb_run.py`, `tasks/tb_live/`, `runs/tb_live/`)

Goal: support automated progress checking/querying during active long-horizon runs, while preserving the same append-only semantics validated in retrospective pipelines.

## 8. Validation Coverage

Test suite: `tests/`.

Coverage includes:

- Core ledger invariants and scoring semantics.
- Serialization and replay roundtrips.
- SWE-agent/Hermes normalize/import behavior.
- Observation dataset construction and audits.
- Run manager + sidecar behavior.
- Category and checkpoint invariants.

Run command:

```sh
uv run pytest
```

## 9. Known Limits

- Retrospective ledgers model visible trace evidence, not hidden internal reasoning.
- Progress values are protocol-sensitive and should be interpreted with audits/notes.
- High progress does not imply correctness; low progress does not imply failure.
- Cross-source comparability requires explicit protocol parity checks.

## 10. Operational Summary

If you need to reproduce or extend this work:

1. Start from source policy/manifests under `external_data/`.
2. Use deterministic scripts in `scripts/` to inventory/sample/import.
3. Materialize ledgers from specs in `annotations/`.
4. Regenerate observations/audits in `datasets/`.
5. Validate with `uv run pytest`.
6. Use `runs/` outputs plus protocol docs in `docs/` for interpretation.
