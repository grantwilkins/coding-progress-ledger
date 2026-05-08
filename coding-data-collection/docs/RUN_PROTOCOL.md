# Run Protocol

Protocol version: `0.1.0`

Each run has three phases:

1. Prepare an agent-visible workspace.
2. Run the agent and collect `transcript.jsonl`.
3. Finalize by emitting observation events, ledger wire events, replayed
   ledger artifacts, verifier outputs, manifests, and notes.

The run protocol is intentionally a coordination layer. Ledger replay and
progress scoring are delegated to `coding-progress-ledger`. Downstream
estimation consumes the trace tuple after this repository has finished
collection and audits.

## Required Version Fields

Every run records:

```text
run_protocol_version
artifact_layout_version
observation_event_schema_version
ledger_wire_schema_version
benchmark_adapter_version
coding_progress_ledger_sha
coding_data_collection_sha
```

JSON artifacts are validated against committed schemas in `schemas/` before
trace collection. JSONL artifacts are validated line-by-line:

```text
run_manifest.json              schemas/run_manifest.schema.json
task_metadata.json             schemas/task_metadata.schema.json
environment_manifest.json      schemas/environment_manifest.schema.json
protocol_manifest.json         schemas/protocol_manifest.schema.json
observation_events.jsonl       schemas/observation_event.schema.json
events.jsonl                   schemas/ledger_wire_event.schema.json
```

## Run Status

```text
completed_success
completed_failure
agent_timeout
agent_crash
verifier_timeout
verifier_crash
docker_build_failure
environment_setup_failure
artifact_incomplete
infrastructure_failure
quarantined_leakage
```

Terminal-success analysis includes only `completed_success` and
`completed_failure`.

Process-dynamics analysis may include `agent_timeout` when transcript,
observation events, and ledger artifacts are valid.

Artifact quality audits include all statuses.

## Artifact-Incomplete Policy

`artifact_incomplete` preserves every artifact that was produced unless a
leakage quarantine requires removing raw agent-visible material. It is not
included in terminal-success or process-dynamics analysis. The preserved
partial run remains available only for artifact-quality and harness-debug
audits.
