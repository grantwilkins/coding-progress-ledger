# Progress-Drop Audit

## Label witness validity

- unmasked_rows: 213
- label_matches: 213
- label_mismatches: 0
- positive_rate: 0.146
- lead_time_distribution: {1.0: 15, 2.0: 13, 3.0: 1, 4.0: 2}
- feature_provenance_status: assumed (`checkpoint_step` is stored, but per-row max source step is not).

## Target-specific leakage scan

- No target-specific feature-name hits in the `LEDGER_BASIC` feature list.

## Feature drivers

Diagnostic standardized logistic coefficients over exact-task train folds.

| feature | feature_group | median_abs_coefficient | sign_frequency_positive | sign_frequency_negative |
|---|---|---|---|---|
| investigation_progress | closure | 1.745 | 1.000 | 0.000 |
| coding_progress | closure | 1.642 | 1.000 | 0.000 |
| completed_leaf_count | closure | 0.735 | 1.000 | 0.000 |
| num_progress_drops_so_far | instability | 0.637 | 0.000 | 1.000 |
| steps_since_last_drop | instability | 0.309 | 0.000 | 1.000 |
| largest_progress_drop_so_far | instability | 0.294 | 0.000 | 1.000 |
| new_leaf_count_last_5_steps | discovery | 0.231 | 0.040 | 0.960 |
| product_progress | closure | 0.205 | 0.960 | 0.040 |
| active_coding_leaf_count | frontier | 0.203 | 0.000 | 1.000 |
| steps_since_new_subtask | discovery | 0.195 | 1.000 | 0.000 |
| active_leaf_count | frontier | 0.150 | 0.000 | 1.000 |
| denominator_growth_so_far | discovery | 0.150 | 0.000 | 1.000 |
| num_adds_so_far | discovery | 0.150 | 0.000 | 1.000 |
| new_leaf_count_last_1_steps | discovery | 0.066 | 0.040 | 0.960 |
| new_leaf_count_last_3_steps | discovery | 0.009 | 0.160 | 0.840 |
| active_validation_leaf_count | frontier | 0.000 | 0.000 | 0.000 |
| num_deletes_so_far | instability | 0.000 | 0.000 | 0.000 |
| num_invalidations_so_far | instability | 0.000 | 0.000 | 0.000 |
| num_reopens_so_far | instability | 0.000 | 0.000 | 0.000 |
| num_splits_so_far | discovery | 0.000 | 0.000 | 0.000 |
| validation_progress | closure | 0.000 | 0.000 | 0.000 |

Leave-one-group-out diagnostics.

| model | group_removed | auroc | brier |
|---|---|---|---|
| ledger_basic_minus_closure | closure | 0.790 | 0.103 |
| ledger_basic_minus_discovery | discovery | 1.000 | 0.004 |
| ledger_basic_minus_frontier | frontier | 1.000 | 0.005 |
| ledger_basic_minus_instability | instability | 1.000 | 0.008 |

## Sensitivity checks

| target_variant | n_ckpts | pos_rate | ledger_basic_auroc | ledger_basic_brier | time_only_auroc | time_only_brier | ledger_minus_time_brier | interpretation |
|---|---|---|---|---|---|---|---|---|
| h10_base | 37 | 0.135 | 0.975 | 0.046 | 0.850 | 0.135 | -0.089 | ledger beats time_only |
| h3_base | 399 | 0.150 | 0.984 | 0.020 | 0.804 | 0.119 | -0.099 | ledger beats time_only |
| h5_base | 213 | 0.146 | 1.000 | 0.004 | 0.832 | 0.125 | -0.121 | ledger beats time_only |
| h5_drop_ge_0.05 | 213 | 0.146 | 1.000 | 0.004 | 0.832 | 0.125 | -0.121 | ledger beats time_only |
| h5_drop_ge_0.10 | 213 | 0.146 | 1.000 | 0.004 | 0.832 | 0.125 | -0.121 | ledger beats time_only |
| h5_first_drop_lead_ge_2 | 213 | 0.075 | 0.989 | 0.023 | 0.557 | 0.072 | -0.049 | ledger beats time_only |
| h5_first_drop_lead_ge_2_excluding_checkpoint_steps_with_add_split_reopen | 148 | 0.095 | 0.987 | 0.023 | 0.736 | 0.098 | -0.075 | ledger beats time_only |
| h5_first_positive_per_drop_episode | 200 | 0.090 | 1.000 | 0.007 | 0.773 | 0.087 | -0.079 | ledger beats time_only |

## Row-level interpretability

| threshold_name | threshold_value | tp | fp | fn | tn | precision | recall | mean_lead_time_tp | median_lead_time_tp |
|---|---|---|---|---|---|---|---|---|---|
| prevalence | 0.146 | 31 | 1 | 0 | 181 | 0.969 | 1.000 | 1.677 | 2.000 |
| 0.5 | 0.500 | 30 | 0 | 1 | 182 | 1.000 | 0.968 | 1.700 | 2.000 |

## Decision interpretation

| condition | verdict |
|---|---|
| witness mismatch | label_construction_bug |
| flagged feature name in used features | feature_leakage |
| harder variants collapse | valid_but_near_boundary |
| harder variants remain strong with frontier/discovery/instability drivers | valid_prefix_signal |
| evidence remains ambiguous | insufficient_evidence |

- applied_verdict: `insufficient_evidence`
- rationale: the headline result holds, but the diagnostic driver mix is too diffuse to support a stronger claim

