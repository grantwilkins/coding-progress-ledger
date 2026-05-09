# Cached Annotator Diagnostic

This diagnostic runs the current annotator over every locally cached raw row and writes full local CSVs under ignored `observation-channel/data/diagnostics/cached_annotator/`. The committed files here are the report, plots, compact trace-level data, plot aggregates, conditional-prefix cohorts, and pathology candidates.

## Run Summary

- Raw rows scanned: 94,345
- Valid annotated traces: 94,175
- Local turn rows written: 4,769,485
- Parse/no-turn failures recorded: 170
- Critical pathology candidates: 170

### Source Summary

| source | traces | median_final_total | mean_final_total | p90_final_total | median_turns | stuck_rate |
| --- | --- | --- | --- | --- | --- | --- |
| hermes | 7644 | 6.000 | 7.051 | 14.000 | 28.000 | 0.000 |
| swe-agent | 80036 | 7.000 | 10.925 | 24.000 | 35.000 | 0.118 |
| terminalbench | 6495 | 4.000 | 8.756 | 20.000 | 18.000 | 0.046 |

The stuck predicate was tightened after inspecting 20 Hermes traces previously flagged as stuck. The triggering observation bodies were empty strings produced from structured tool acknowledgements/errors, not meaningful repeated output. The annotator now ignores repeated empty or short ack-like observations unless the body contains an explicit error marker or is non-trivial in length. With that repair, Hermes stuck flags drop from 5,525 traces to 2 traces.

## 1. Final Unit Count

![Final total distribution](plots/01_final_total_distribution.png)

Observed `D_T` is source-dependent. hermes: median 6.0, mean 7.1, p90 14.0; swe-agent: median 7.0, mean 10.9, p90 24.0; terminalbench: median 4.0, mean 8.8, p90 20.0.

## 2. Trace Length Versus Units

![Trace length versus units](plots/02_trace_length_vs_units.png)

Trace length and final unit count are related but visibly not interchangeable. Long traces can still coalesce into modest unit counts, while compact traces can accumulate many units when actions switch category or target frequently.

## 3. Sample Trajectories

![Sample trajectories](plots/03_sample_trajectories.png)

The sampled trajectories are a qualitative sanity check for monotone `D_t` and `N_t` behavior. Automated invariant failures, if any, are listed in `pathology_candidates.csv`.

## 4. Conditional Final Counts

![Conditional final count distributions](plots/04_conditional_final_total.png)

The selected supported prefix cohorts have median IQR change of -37.5% versus the uncensored marginal `D_T` IQR (8.0). The plot excludes right-tail-censored SWE-Agent traces from the histograms and annotates the censored fraction for each cohort. These four hand-selected/simple fallback prefixes do not yet establish a robust estimator signal.

## 5. Category Mix

![Category mix](plots/05_category_mix.png)

`NONE` is included for turns before any unit is open or where no current category exists. The plot is a prefix-level diagnostic, not a final taxonomy of trace work.

## Pathology Candidates

| rule | severity | count |
| --- | --- | --- |
| no_coalescing_every_action_new_unit | warning | 619 |
| parse_error | critical | 170 |
| unexpected_terminal_category | warning | 21759 |
| very_long_stuck_continuation | warning | 937 |

The pathology log is automated only. Rows in `pathology_candidates.csv` are candidates for review, not human-confirmed annotation failures.

Cross-tabulation shows `unexpected_terminal_category` is mostly a capped-run artifact: 21,026 of 21,759 cases are `submitted (exit_context)`, with 643 clean `submitted` cases left for follow-up. The no-coalescing audit found path-tracker over-splitting on SWE-Agent edit line ranges and shell flags; after tightening those rules, no-coalescing warnings drop from 4,436 to 619. The five reviewed remaining cases are mostly short category-alternation traces or honest multi-file scaffolding. Long stuck continuations fire around the middle of traces overall: median stuck progress is 0.49, with interquartile range 0.31-0.70.

## Estimator Readiness

The current simple prefix cohorts do not provide a strong go signal after fixing path-target over-splitting and excluding the censored SWE-Agent right tail. Cautious next step: treat `swe-agent` traces with `final_total > 103` as right-censored or filter them from training, then search for better prefix features/cohorts before committing to the estimator. The parse/critical-candidate affected trace rate is 0.18%.

## Right-Tail Censoring

The `swe-agent` `final_total > 103` folded tail contains 290 traces after the path-tracker repair. Their exit statuses are 250 `submitted (exit_context)`, 14 `exit_context`, and 26 `early_exit`. This remains consistent with a capped agent loop rather than ordinary completed trajectories. These traces are flagged in `trace_summary.csv` with `censored_right_tail=True` and summarized in `censoring_summary.csv`.

## Files

- `trace_summary.csv`: one compact row per raw trace, including parse errors.
- `final_total_distribution.csv`, `trace_length_distribution.csv`, `category_mix.csv`: plot-ready aggregates.
- `conditional_prefix_cohorts.csv`: all traces matching selected supported prefix states.
- `censoring_summary.csv`: exit-status breakdown for right-tail-censored traces.
- `pathology_exit_status_crosstab.csv`: pathology warning counts by source and exit status.
- `no_coalescing_review.csv`: five manually inspected no-coalescing examples.
- `stuck_timing.csv`: stuck-fire step, trace length, and normalized stuck progress.
- `pathology_candidates.csv`: automated sanity-check candidates.
- Local only: `observation-channel/data/diagnostics/cached_annotator/turns.csv` and `traces.csv`.
