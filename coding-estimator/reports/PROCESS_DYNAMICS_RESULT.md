# Process Dynamics Result

## Exact claim

Prefix-only ledger features predict near-future progress drops under exact-task holdout better than elapsed time. This supports work-frontier instability detection, not policy-grade completion-risk estimation.

## Exact-task headline metrics

- ledger_basic_auroc: 1.000
- ledger_basic_brier: 0.004
- time_only_auroc: 0.832
- time_only_brier: 0.125

## Label witness result

- verdict: `insufficient_evidence`
- rationale: the headline result holds, but the diagnostic driver mix is too diffuse to support a stronger claim

## Feature-driver result

- See `PROGRESS_DROP_AUDIT.md` for coefficient rankings and leave-one-group-out diagnostics.

## Sensitivity summary

- harder_variant_rows: 2
- harder_variants: h5_first_drop_lead_ge_2, h5_first_positive_per_drop_episode

## Case-study summary

- case_count: 4
- cases: true positive, hardest negative, false negative, true negative quiet run

## Validation-new-work diagnosis

- recommendation: `defer_on_tb_live_v2`

## Terminal-success negative result

- Terminal success remains secondary and negative on tb_live_v2 exact-task holdout.

## What this does and does not support

- Supports: prefix-only work-frontier instability prediction.
- Does not support: control, scheduling, or terminal completion-risk decisions.

