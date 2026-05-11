# Progress-Belief Tracker

Replayable live estimator over final work using empirical Bayes, GBM, and filtered combinations.

## Split Summary

| source | train_traces | eval_traces | total_traces |
| --- | ---: | ---: | ---: |
| hermes | 6115 | 1529 | 7644 |
| swe-agent | 64028 | 16008 | 80036 |
| terminalbench | 5196 | 1299 | 6495 |

## Final-Work Summary

| variant | n | median_abs_error | median_interval80_width |
| --- | ---: | ---: | ---: |
| eb_direct | 917339 | 2.0 | 18.0 |
| eb_filter | 917339 | 2.0 | 16.0 |
| gbm_direct | 918612 | 2.3868894553447735 | 18.886018994229246 |
| eb_gbm_mixed_filter | 917339 | 2.0 | 15.0 |

## Artifacts

- [Progress beliefs](progress_beliefs.csv)
- [Claim calibration pairs](belief_threshold_pairs.csv)
- [Belief summary](belief_summary.csv)

![Trace belief examples](trace_belief_examples.png)
