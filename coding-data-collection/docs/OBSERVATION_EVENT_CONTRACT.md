# Observation Event Contract

Schema version: `0.2.0`

`observation_events.jsonl` preserves transcript and verifier signals that
must not be collapsed into ledger progress semantics.

Each line:

```json
{
  "schema_version": "0.2.0",
  "run_id": "run-id",
  "step": 17,
  "observed_ts": "2026-05-05T00:00:00Z",
  "source_artifact": "transcript.jsonl",
  "event_type": "validation_fail_observed",
  "payload": {
    "command": "pytest -q",
    "exit_code": 1,
    "stdout_snippet": "...",
    "stderr_snippet": "...",
    "normalized_error_signature": "...",
    "visible_to_agent": true
  }
}
```

`payload.visible_to_agent` is required for every event. Events sourced from
`verifier_output.txt` or `oracle_workspace_snapshot` must set it to `false`.

Required event families:

```text
validation_attempt
validation_pass_observed
validation_fail_observed
error_observed
error_repeated
environment_blocked
product_file_written
product_file_edited
expected_file_missing
agent_claims_done
verifier_pass
verifier_fail
verifier_disagreement
oracle_artifact_read
```

Verifier events are emitted only at `max_transcript_step + 1` or later.
They are not available to preterminal checkpoint features.

The validator rejects post-terminal event families emitted at or before the
last transcript step, even if the JSON shape is otherwise valid.

## Quality Metrics

`scripts/audit_observation_quality.py` computes pilot-gate metrics over one or
more run directories:

```text
shell_exit_code_coverage
shell_stdout_snippet_coverage
shell_stderr_snippet_coverage
terminal_events_visible_to_agent
hidden_phase_events_visible_to_agent
observation_schema_valid
median_observation_events_per_run
```

Snippet coverage is based on capture-field presence, not non-empty output. A
command that produced no stdout still counts as covered if `stdout_snippet` or
`obs_snippet` is present with an empty captured value.

The audit reports the later pilot gate
`median_observation_events_per_run >= 10` separately from smoke-quality pass
status. The four-run smoke corpus is expected to exercise the metric surface,
not to satisfy that pilot-scale density gate.
