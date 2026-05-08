# Checkpoint construction audit

> ⚠️ **Retrospective annotation caveat.** This report draws on
> retrospective sources (`hermes_pilot_h5_v2`, `swe_agent_pilot`). These ledgers were annotated
> post-hoc with knowledge of the run's outcome. Annotator-outcome
> leakage is unfixable at the estimator layer; treat any reported
> performance as an upper bound on "realistic" performance, not a
> faithful estimate. See `docs/SOURCES.md` for the full statement.

> ℹ️ **Live-source framing.** `tb_live` / `tb_live_v2` measure
> **online realism**, not just model performance. Use live-source
> reports to confirm the pipeline produces honest online checkpoint
> features; do not optimize against a single live cohort in isolation.


## Run and checkpoint counts per source

- `hermes_pilot_h5_v2`: 30 runs, 896 checkpoints
- `swe_agent_pilot`: 20 runs, 599 checkpoints
- `tb_live_v2`: 102 runs, 703 checkpoints
- `tb_live`: 12 runs, 83 checkpoints

## Feature columns by group

- **closure**: 5 present, 0 missing
- **discovery**: 7 present, 0 missing
- **evidence**: 6 present, 0 missing
- **frontier**: 3 present, 0 missing
- **instability**: 6 present, 0 missing
- **stalling**: 10 present, 0 missing
- **time_budget**: 5 present, 0 missing
- **validation**: 11 present, 0 missing

## Forbidden-column audit

PASS — no forbidden columns detected on the checkpoint frame.

## Behavioral prefix-truncation audit

PASS — no leakage detected on 2 sampled runs.

## Run-constancy audit

_PLACEHOLDER — populated by D5 once D3 feature builders ship. (no feature/target columns supplied)_

## Missingness by feature and source

- `hermes_pilot_h5_v2`: 896 rows; 4 features with any null
- `swe_agent_pilot`: 599 rows; 4 features with any null
- `tb_live_v2`: 703 rows; 3 features with any null
- `tb_live`: 83 rows; 3 features with any null

## Label balance by target and source

_PLACEHOLDER — populated by D5 once D3 feature builders ship. (needs Workstream E labels)_

## Live-source row examples

| run_id | source | checkpoint_step | active_leaf_count | coding_progress |
|---|---|---|---|---|
| b-tree-on-disk | tb_live | 0 | 0 | 0.0 |
| b-tree-on-disk | tb_live | 1 | 1 | 0.0 |
| b-tree-on-disk | tb_live | 2 | 2 | 0.5 |

## Retrospective-source row examples

| run_id | source | checkpoint_step | active_leaf_count | coding_progress |
|---|---|---|---|---|
| swe_agent_pilot_f_01 | swe_agent_pilot | 0 | 0 | 0.0 |
| swe_agent_pilot_f_01 | swe_agent_pilot | 1 | 0 | 0.0 |
| swe_agent_pilot_f_01 | swe_agent_pilot | 2 | 1 | 0.0 |

---
Overall: PASS
