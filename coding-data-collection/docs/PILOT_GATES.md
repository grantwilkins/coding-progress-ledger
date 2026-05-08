# Pilot Gates

Do not scale beyond the 12-task / 24-run Terminal-Bench pilot unless all
gates pass:

```text
median transcript steps >= 15
validation_attempt in >= 50% of runs
validation_fail_observed in >= 25% of runs
progress_drop in >= 20% of runs
terminal failure rate between 25% and 70%
>= 5 high-progress failures or verifier disagreements
median observation events per run >= 10
shell exit_code coverage >= 95%
shell stdout/stderr snippet coverage >= 80%
prefix provenance present on 100% of checkpoints
0 oracle/test/gold leakage incidents
verifier outcomes reproducible on rerun sample
```

If any gate fails, write `reports/PILOT_FAILURE_ANALYSIS.md` and do not
run the 40-task / 80-run scale batch.

