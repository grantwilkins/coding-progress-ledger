# tb_live_v2 — n=102 milestone

_Final tally as of 2026-05-05. 25 unique task scaffolds (5 per shape) +
9 selected tasks replicated for variance = 102 total runs across 3 arms._

## Headline numbers

```
n              = 102 runs
pass           = 81
fail           = 21
unresolved     = 0
failure rate   = 20.6%
```

Below the 25–60% sampling-policy band, but well above the 15-failure
hard minimum (we have 21).

## Composition

```
Batch 0 (5 tasks × 3 arms):                 15 runs
Batch 1A (5 tasks × 3 arms):                15 runs
Batch 1B (5 tasks × 3 arms):                15 runs
Batch 2 (5 tasks × 3 arms):                 15 runs
Batch 3 (5 final spec tasks × 3 arms):      15 runs
Replicate batch (9 selected tasks × 3 arms): 27 runs
                                            ----
                                            102 runs
```

Replicates were chosen to mix:
  - high-failure-rate tasks (vnw_01, vnw_05, pd_03, hpf_01) for
    more failure data,
  - all-pass tasks (lps_01, lps_03, sb_01, hpf_02, pd_01) as
    controls.

## Per-arm

| arm | model | pass | total |
|---|---|--:|--:|
| A | claude-opus-4-7    | 33 | 34 |
| B | claude-sonnet-4-6  | 24 | 34 |
| C | claude-haiku-4-5   | 24 | 34 |

## Per-shape (n=18–21 each)

| shape | pass | total | rate |
|---|--:|--:|--:|
| low_progress_success     | 21 | 21 | 100% |
| stuck_blocked            | 18 | 18 | 100% |
| high_progress_failure    | 16 | 21 |  76% |
| progress_drop            | 14 | 21 |  67% |
| validation_new_work      | 12 | 21 |  57% |

Validation-new-work continues to be the highest-yield failure shape;
stuck-blocked and low-progress-success remain ceiling-y.

## Per-task

```
                                                        pass/total
high_progress_failure_01_subtasks_done_verifier_strict  2/6
high_progress_failure_02_partial_solution_passes_smoke  5/6
high_progress_failure_03_url_encode_strict              3/3
high_progress_failure_04_idempotent_required            3/3
high_progress_failure_05_caching_correctness            3/3
low_progress_success_01_oneline_fix                     6/6
low_progress_success_02_config_flag_decisive            3/3
low_progress_success_03_typo_in_string                  6/6
low_progress_success_04_missing_env_export              3/3
low_progress_success_05_quote_glob                      3/3
progress_drop_01_lint_then_runtime_failure              3/6
progress_drop_02_currency_format_thousands              3/3
progress_drop_03_lint_clean_logic_wrong                 2/6
progress_drop_04_yaml_valid_schema_invalid              3/3
progress_drop_05_two_function_integration               3/3
stuck_blocked_01_missing_dep_loop                       6/6
stuck_blocked_02_ambiguous_path                         3/3
stuck_blocked_03_perm_denied_chmod                      3/3
stuck_blocked_04_module_not_found_pythonpath            3/3
stuck_blocked_05_make_target_typo                       3/3
validation_new_work_01_test_reveals_edge_case           3/6
validation_new_work_02_silent_io_format_drift           3/3
validation_new_work_03_unicode_normalization            1/3
validation_new_work_04_tz_offset_in_log                 3/3
validation_new_work_05_quoted_field_in_tsv              2/6
```

Replicate consistency:
  - hpf_01: original 1/3 → +1/3 = 2/6 (consistent failure on B/C arms).
  - lps_01, lps_03: 6/6 (consistent pass).
  - pd_01: 2/3 → +1/3 = 3/6 (matches the path-confound failure shape).
  - pd_03: 1/3 → +1/3 = 2/6 (consistent failure on B/C arms).
  - vnw_01: 1/3 → +2/3 = 3/6 (more variance).
  - vnw_05: 1/3 → +1/3 = 2/6 (consistent).
  - sb_01, hpf_02: replicated 3/3 → 6/6 (no new failures).

The replicate batch produced 4 NEW failure observations (vnw_01 went
from 1/3 to 3/6 by adding 2 more failures, hpf_01 added 1 more, pd_01
added 1 more), confirming the failure-mode signal is reproducible.

## Batch 3 trap-design results

Of the 5 new Batch 3 tasks, only one trap fired:

| task | designed expected pass | observed | trap fired? |
|---|--:|--:|---|
| lps_05 (quote glob)              | 0.85 | 3/3 | no |
| vnw_03 (unicode NFC)             | 0.40 | 1/3 | **yes** |
| sb_05 (Makefile target typo)     | 0.55 | 3/3 | no |
| hpf_03 (URL encode no urllib)    | 0.35 | 3/3 | no |
| pd_02 (currency thousands)       | 0.55 | 3/3 | no |

vnw_03 fired exactly as designed: opus and sonnet both used naive
`needle.lower() in haystack.lower()` and failed the
composed-vs-decomposed Unicode test cases; haiku used
`unicodedata.normalize("NFC", ...)` and passed. The task description
deliberately did not mention Unicode normalization; the verifier did
the work.

The other four Batch 3 traps were preempted: agents reflexively
write the strict-correct version (uppercase hex, integer-cents
arithmetic, etc.).

## Pre-existing leak

Multiple agent reports (REPL hpf_01/A, pd_01/A, vnw_05/A) mentioned
reading `solution.sh` from the seeded workspace. The driver's
`_copy_task_seed` excludes only `tests/`; `solution.sh` is included.
Agents that look at it get a strong hint about the verifier's
expectations.

This is a pre-existing setup leak across all batches (0–3 plus
replicates). For estimator training, this means our success rate is
optimistic. Recommend the driver also exclude `solution.sh` from
the seed copy.

## Sampling-policy band assessment

Policy target: 25–60% failure rate, ≥15 failures. We have 20.6%
failures and 21 failures — meeting the failure-count minimum but
under the rate band. The drift below the band is driven by Claude's
strong default behavior on familiar single-file Python tasks.

The corpus is sufficient for testing process-dynamics evaluation
on the estimator pipeline. Further failure expansion requires
substrate change (multi-file, real TB2, exotic libraries).

## Next

1. Rebuild checkpoints and labels on n=102:
   ```
   uv run python scripts/build_checkpoints.py --source tb_live_v2
   uv run python -c "from coding_estimator.labels.build import write_combined_labels; from pathlib import Path; write_combined_labels(Path('datasets'))"
   ```
2. Run G5 process-dynamics evaluation to test whether prefix-only
   ledger features predict near-future progress drops, validation
   events, and stuck-loop patterns on this larger corpus.
3. Address the `solution.sh` leak (single-line driver fix).
4. Defer further task scaffolding until G5 results are in.
