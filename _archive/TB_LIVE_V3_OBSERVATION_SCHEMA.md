# TB Live V3 Observation Schema

`tb_live_v3` keeps `ledger.jsonl` unchanged and adds a parallel
`observation_events.jsonl` file for structured transcript/verifier
signals that should not be collapsed into ledger categories.

## New file

Per run:

```text
runs/tb_live_v3/<run_id>/observation_events.jsonl
```

Each line is a JSON object:

```json
{
  "schema_version": "0.1.0",
  "run_id": "task_x__armB__abcd1234",
  "step": 7,
  "observed_ts": "2026-05-05T08:10:17Z",
  "source_artifact": "transcript.jsonl",
  "event_type": "validation_attempt",
  "payload": {
    "command": "pytest -q",
    "summary": "run tests",
    "after_solution_oracle_read": false
  }
}
```

## Event semantics

- `step` is the earliest transcript-visible step where the observation is available.
- Verifier terminal events (`verifier_pass`, `verifier_fail`,
  `verifier_disagreement`) are emitted at `max_transcript_step + 1` so
  they remain invisible to earlier checkpoints.
- `source_artifact` identifies provenance:
  `transcript.jsonl`, `run_manifest.json`, `verifier_output.txt`,
  `task_tests`.

## Required event types

- `validation_attempt`
- `validation_pass_observed`
- `validation_fail_observed`
- `error_observed`
- `error_repeated`
- `environment_blocked`
- `product_file_written`
- `expected_file_missing`
- `agent_claims_done`
- `verifier_pass`
- `verifier_fail`
- `verifier_disagreement`
- `solution_oracle_read`

## Relationship to existing files

- `transcript.jsonl` remains the raw action trace.
- `events.jsonl` remains the coarse ledger-sidecar input.
- `ledger.jsonl`, `progress.csv`, and `summary_by_category.json`
  continue to define progress semantics.
- `observation_events.jsonl` is additive and measurement-only.
- `run_manifest.json` and `verifier_output.txt` provide terminal
  verifier outcome and failure type.
- `final_diff.patch` and `task.md` remain audit aids, not headline
  checkpoint features.
