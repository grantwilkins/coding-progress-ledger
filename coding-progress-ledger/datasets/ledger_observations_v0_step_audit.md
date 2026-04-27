# Ledger Observations v0 Audit

Audit of checkpoint-level observation CSV coherence.

## Totals

- Rows: 198
- Runs: 18
- Integrity passed: yes

## Integrity

- Invalid progress values: 0
- Completed > active failures: 0
- Delta mismatches: 0
- First-row nonzero deltas: 0
- Missing identifiers: 0
- Invalid success metadata: 0
- Unknown success metadata: 0

## Category Resolution

| Mode | Rows |
| --- | ---: |
| `legacy_inferred` | 149 |
| `mixed` | 40 |
| `native` | 9 |

Runs with native/resolved metric mismatch: `control_coding_complete_artifacts_incomplete`, `live_validation/03_evidence_audit_by_category`, `live_validation/05_active_incomplete_coding_leaves`, `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`

## Drops

- Negative coding deltas: 21
- Negative overall deltas: 29

Coding drop sources:

| Source | Count |
| --- | ---: |
| `mixed` | 1 |
| `product` | 8 |
| `validation` | 12 |

Overall drop sources:

| Source | Count |
| --- | ---: |
| `artifact` | 8 |
| `documentation` | 1 |
| `environment` | 2 |
| `mixed` | 2 |
| `product` | 8 |
| `validation` | 8 |

## Event vs Step

Runs where largest event-level and step-level coding drops differ: none

Runs with multiple events at the same step: none

## Success / Progress Quadrants

- Success + high progress: `control_coding_complete_artifacts_incomplete`, `live_validation/01_suite_summary_weight_source`, `live_validation/02_drop_category_contributions`, `live_validation/03_evidence_audit_by_category`, `live_validation/04_docs_progress_not_success`, `live_validation/05_active_incomplete_coding_leaves`, `negative_control_monotonic_one_line`, `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`
- Success + low progress: none
- Failure + high progress: `control_high_progress_wrong_solution`
- Failure + low progress: `control_monotonic_incomplete_failure`, `negative_control_incomplete_budget_limited`
- Unknown success: none

## Warnings

- large native/resolved divergence: 10 runs
