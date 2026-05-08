# coding-data-collection

Produces `(transcript, observations, ledger events, verifier outcome, wallclock)` tuples for downstream estimation.

This repository owns trace production only:

- prepare isolated task workspaces;
- run model-agent, scripted-model, and substrate-smoke collection jobs;
- capture `transcript.jsonl`, `observation_events.jsonl`, `events.jsonl`, and replayed `ledger.jsonl`;
- run deterministic verifiers after the agent phase;
- record `run_manifest.json` with `started_at`, `ended_at`, and `wallclock_seconds`;
- audit artifact completeness, leakage/redaction safety, sidecar replay, and verifier determinism.

It does not define progress semantics, train models, build prediction datasets,
or decide adaptive collection policy. Historical v0 planning and reports live under
`../_archive/coding-data-collection-v0/`.

## Core Commands

```bash
uv run python scripts/run_model_agent_trace.py --help
uv run python scripts/run_trace_batch.py plan.json --out execution.json
uv run python scripts/audit_trace_run.py runs/<corpus>/<run_id>
uv run pytest tests
```
