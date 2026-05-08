# Profiling go/no-go gate (F11)

## Verdict: **PASS**

## Checks

### [PASS] C1_terminal_class_balance

```
target=y_success_eventual; qualifying sources (>= 5 of each class): ['swe_agent_pilot']
  - swe_agent_pilot: positives=10, negatives=10
  - tb_live: positives=12, negatives=0
```

### [PASS] C2_wallclock_coverage

```
qualifying sources (>= 50 wallclock checkpoints): ['tb_live']
  - hermes_pilot_h5_v2: wallclock_checkpoints=0
  - swe_agent_pilot: wallclock_checkpoints=0
  - tb_live: wallclock_checkpoints=83
```

### [PASS] C3_global_feature_missingness

```
features over 95% missing: 0 of 50 applicable features (exempted: unknown_due_to_missing_artifact)

```

### [PASS] C4_no_forbidden_columns

```
no forbidden columns
```

### [PASS] C5_closure_cross_source_ks

```
closure features with worst-pair KS < 0.5: ['completed_leaf_count', 'coding_progress', 'validation_progress', 'product_progress', 'investigation_progress']
  - completed_leaf_count: worst-pair KS = 0.228 (3 pairs)
  - coding_progress: worst-pair KS = 0.177 (3 pairs)
  - validation_progress: worst-pair KS = 0.486 (3 pairs)
  - product_progress: worst-pair KS = 0.213 (3 pairs)
  - investigation_progress: worst-pair KS = 0.351 (3 pairs)
```

