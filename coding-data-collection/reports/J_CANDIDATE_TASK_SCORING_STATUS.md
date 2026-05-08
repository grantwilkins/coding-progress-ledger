# J Candidate Task Scoring Status

Status: done for pre-pilot selection.

Inputs:

```text
manifests/pilots/terminal_bench_candidate_calibration.csv
```

Output:

```text
manifests/pilots/terminal_bench_candidate_scores.csv
```

Selection summary:

```text
candidates_scored=14
selected_for_pilot=12
selected_categories=9
```

Selected tasks:

```text
adaptive-rejection-sampler
extract-safely
broken-python
grid-pattern-transform
fix-permissions
aimo-airline-departures
csv-to-parquet
attention-mil
count-dataset-tokens
terminal-bench/query-optimize
create-bucket
terminal-bench/regex-log
```

Notes:

- The scoring formula is exactly the formula in `TASKS.md`.
- Selection first preserves category coverage, then fills remaining slots by
  priority and operational risk.
- Two selected tasks are Harbor registry candidates. They are retained as
  contingency candidates because only eleven calibrated HF/archive candidates
  are currently available locally; they should be replaced by HF-extractable
  tasks before K if the pilot remains strictly `hf_archive_custom`.
