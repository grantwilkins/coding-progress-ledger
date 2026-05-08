# Sources

The estimator reads from up to nine upstream sources. Three are
**canonical for v0**; the rest are reserved for parity comparisons or
future ingestion work.

| source_id | canonical | runs | timestamps | label field |
|---|---|---|---|---|
| `swe_agent_pilot` | ✓ | 20 | none (step-only) | `source_metadata.final_success` |
| `swe_agent_pilot_v3` |  | 20 | none | `source_metadata.final_success` |
| `swe_agent_live` |  | 20 | synthetic | `source_metadata.final_success` |
| `swe_agent_live_wallclock` |  | 20 | synthetic_backfill | `source_metadata.final_success` |
| `hermes_pilot` |  | ~30 | synthetic | `source_metadata.final_success` |
| `hermes_pilot_h5` |  | larger | synthetic | `source_metadata.final_success` |
| `hermes_pilot_h5_v2` | ✓ | revised | synthetic | `source_metadata.final_success` |
| `tb_live` | ✓ | 12 | **real** | `live_instrumentation.verifier_pass` |
| `tb_live_v2` |  | 102 | **real** | `run_manifest.final_success` |

The machine-readable form is `coding_estimator/ingest/sources.py`.

## Canonical-source decisions (locked)

- **swe_agent canonical:** `swe_agent_pilot` (20 runs, original protocol).
  `swe_agent_pilot_v3` is reserved for parity-of-protocol comparisons in
  Workstream L; `swe_agent_live` is sidecar-feasibility only.
- **hermes canonical:** `hermes_pilot_h5_v2` (most recent, revised
  annotation protocol).
- **live canonical:** `tb_live` (12 first-party runs).
- **selectable live extension:** `tb_live_v2` (102 first-party runs,
  repo-local corpus) can be included in explicit training-artifact
  builds without changing the default v0 bundle.
- **`swe_agent_live_wallclock`:** do **not** mix into headline pools; its
  wall-clock is back-filled synthetic per upstream
  `WORKSTREAM_N_TB_PLAN.md`.

## Annotation-leakage acknowledgment (mandatory)

> Retrospective sources (`swe_agent_pilot`, `hermes_pilot*`) were
> annotated post-hoc with knowledge of the run's outcome. This means
> event-categorization decisions may carry annotator-outcome
> information that is unfixable at the estimator layer. Any model
> trained or evaluated on retrospective sources inherits this leakage;
> it must be named in every report and treated as an upper bound on
> "realistic" performance, not a faithful estimate.

## Source-specific success-resolution caveats

- `tb_live`: upstream `resolve_final_success` returns `(None, "unknown")`
  on every tb_live run because `summary_by_category.final_success` is
  `null` and `run_manifest.json` has no label. The canonical success
  signal is `live_instrumentation.verifier_pass`. The label loader
  (Workstream C3) is responsible for that fallback.
- `tb_live_v2`: final success is recorded directly in
  `run_manifest.final_success` and resolves from the internal verifier.
  These runs live under the local `coding-estimator/runs/tb_live_v2/`
  tree rather than the external `coding-progress-ledger` checkout.
- `hermes_pilot_h5_v2`: many runs have `source_metadata.final_success ==
  null`. Per § 0.9, those runs must be **skipped**, not fabricated. The
  combined manifest records them so the budget snapshot can flag the
  deficit.

## Upstream protocol pointers

- `../coding-progress-ledger/AGENTS.md` — ledger-side rules of engagement.
- `../coding-progress-ledger/docs/SCHEMA_DECISION.md` — event-schema
  history.
- `../coding-progress-ledger/docs/WORKSTREAM_N_TB_PLAN.md` — TB-12 live
  pipeline; canonical for tb_live behavior.
