# Native Observation Events Status

Date: 2026-05-05

## Scope

Workstream F is complete for the current HF custom Docker smoke path.

## Implemented

Code:

```text
src/coding_data_collection/observation.py
src/coding_data_collection/observation_quality.py
scripts/audit_observation_quality.py
tests/test_observation_quality.py
```

The run validator already validates `observation_events.jsonl` against
`schemas/observation_event.schema.json` and enforces post-terminal verifier
timing.

The Docker smoke runner emits observation events from the same transcript rows
used by the HF smoke runs.

## Quality Audit

Command:

```bash
uv run python scripts/audit_observation_quality.py \
  runs/d_smoke/noop_aimo \
  runs/d_smoke/oracle_hello_world \
  runs/d_smoke/oracle_grid_pattern_transform \
  runs/d_smoke/oracle_aimo_airline_departures \
  --output reports/OBSERVATION_F_QUALITY_REPORT.json
```

Result:

```text
smoke_quality_passed=true
run_count=4
shell_rows=11
shell_exit_code_coverage=1.0
shell_stdout_snippet_coverage=1.0
shell_stderr_snippet_coverage=1.0
median_observation_events_per_run=4.0
pilot_gates.median_observation_events_per_run_passed=false
```

Every run had schema-valid observation events and zero terminal verifier events
visible to the agent. Verifier/oracle phase validation observations are also
non-agent-visible and counted by `hidden_phase_events_visible_to_agent`.

The smoke corpus intentionally does not pass the later pilot observation-density
gate of median observation events per run >= 10; F only wires and reports the
gate.

## Verification

```bash
uv run pytest tests/test_observation_quality.py tests/test_observation_and_audits.py tests/test_schema_validation_semantics.py
# 16 passed

uv run pytest tests
# 54 passed
```
