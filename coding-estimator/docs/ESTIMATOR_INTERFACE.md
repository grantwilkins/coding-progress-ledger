# Estimator interface

The estimator reads a single tabular form: a **checkpoint frame**. Each row
is one belief-state instance, indexed by `(run_id, checkpoint_id)`.

This document is normative. Schemas under `schemas/` are the
machine-readable form; this file is the prose.

## Required across all sources

| column | type | meaning |
|---|---|---|
| `run_id` | str | unique within `source` |
| `source` | str | one of the values in `schemas/checkpoint_schema.json::source.enum` |
| `checkpoint_id` | str | stable id within run; usually `f"{run_id}::{checkpoint_step}"` |
| `checkpoint_step` | int | step index used as `t` for prefix-only features |
| `checkpoint_event_index` | int | absolute event index in `ledger.jsonl` at `t` |
| `is_terminal_checkpoint` | bool | true iff `t == finish_step` |
| `ledger_path` | str | repo-relative path to the ledger file used |
| `schema_version` | str | semver of the dataset build |
| `builder_commit_sha` | str | `git rev-parse HEAD` of `coding-estimator` at build time |
| `source_protocol_version` | str | upstream protocol marker |

## Required when available, nullable otherwise

| column | populated_on | nullable_on |
|---|---|---|
| `checkpoint_wall_time` | tb_live, tb_live_v2, swe_agent_live_wallclock | swe_agent_pilot, hermes_pilot* |
| `checkpoint_elapsed_seconds` | tb_live, tb_live_v2, *_wallclock | retrospective sources |
| `checkpoint_fraction_timeout` | tb_live | else |
| `timestamp_quality` | always | enum: real, synthetic, synthetic_backfill, missing |

When a column is "nullable on a source", every row from that source must
emit `null` (not zero, not a sentinel). The leakage and budget audits will
surface populated-vs-null violations.

## Identity invariants

- `(source, run_id, checkpoint_id)` is unique across the dataset.
- `(source, run_id, checkpoint_step)` is unique within a run; this is the
  primary sort key for stable artifacts.
- Exactly one row per run has `is_terminal_checkpoint == true`.

## Features and labels

- Feature columns are catalogued in `docs/ESTIMATOR_FEATURE_GROUPS.md` and
  validated against `schemas/feature_schema.json`.
- Label columns are catalogued in `docs/ESTIMATOR_TARGETS.md` and pivot to a
  long-form table per `schemas/label_schema.json` (one row per
  (run_id, checkpoint_id, target_name)).
- The forbidden-column guard (`schemas/forbidden_columns.json`) is run on
  every checkpoint frame before it is written. No `final_*`, `verifier_*`,
  `summary_by_category_*`, `y_*`, or `label_*` columns may co-exist with
  features in the same frame.
