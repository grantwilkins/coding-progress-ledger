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
| hermes | 7644 | 6.000 | 7.087 | 14.000 | 28.000 | 0.723 |
| swe-agent | 80036 | 8.000 | 13.492 | 29.000 | 35.000 | 0.115 |
| terminalbench | 6495 | 4.000 | 8.731 | 20.000 | 18.000 | 0.114 |

## 1. Final Unit Count

![Final total distribution](plots/01_final_total_distribution.png)

Observed `D_T` is source-dependent. hermes: median 6.0, mean 7.1, p90 14.0; swe-agent: median 8.0, mean 13.5, p90 29.0; terminalbench: median 4.0, mean 8.7, p90 20.0.

## 2. Trace Length Versus Units

![Trace length versus units](plots/02_trace_length_vs_units.png)

Trace length and final unit count are related but visibly not interchangeable. Long traces can still coalesce into modest unit counts, while compact traces can accumulate many units when actions switch category or target frequently.

## 3. Sample Trajectories

![Sample trajectories](plots/03_sample_trajectories.png)

The sampled trajectories are a qualitative sanity check for monotone `D_t` and `N_t` behavior. Automated invariant failures, if any, are listed in `pathology_candidates.csv`.

## 4. Conditional Final Counts

![Conditional final count distributions](plots/04_conditional_final_total.png)

The selected supported prefix cohorts have median IQR reduction of 10.0% versus the marginal `D_T` IQR (10.0). This is a coarse signal check, not a trained estimator.

## 5. Category Mix

![Category mix](plots/05_category_mix.png)

`NONE` is included for turns before any unit is open or where no current category exists. The plot is a prefix-level diagnostic, not a final taxonomy of trace work.

## Pathology Candidates

| rule | severity | count |
| --- | --- | --- |
| no_coalescing_every_action_new_unit | warning | 4436 |
| parse_error | critical | 170 |
| unexpected_terminal_category | warning | 21759 |
| very_long_stuck_continuation | warning | 999 |

The pathology log is automated only. Rows in `pathology_candidates.csv` are candidates for review, not human-confirmed annotation failures.

## Estimator Readiness

The supported prefix cohorts show visible signal for an estimator. Cautious go: use these tables to prototype an estimator, but inspect the automated pathology candidates first. The combined parse/critical-candidate rate is 0.36%.

## Files

- `trace_summary.csv`: one compact row per raw trace, including parse errors.
- `final_total_distribution.csv`, `trace_length_distribution.csv`, `category_mix.csv`: plot-ready aggregates.
- `conditional_prefix_cohorts.csv`: all traces matching selected supported prefix states.
- `pathology_candidates.csv`: automated sanity-check candidates.
- Local only: `observation-channel/data/diagnostics/cached_annotator/turns.csv` and `traces.csv`.
