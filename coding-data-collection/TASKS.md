# TASKS - coding-data-collection

Mission: produce `(transcript, observations, ledger events, verifier outcome,
wallclock)` tuples for downstream estimation.

Active scope:

- Keep model-agent, scripted-model, and substrate-smoke trace collection working.
- Keep Docker sandboxing, model-client adapters, leakage/redaction checks,
  verifier determinism replay, sidecar replay, and run manifests maintained.
- Keep required completed-run artifacts small and tied to trace consumption:
  `task.md`, manifests, transcript, observation events, wire ledger events,
  replayed ledger, verifier output, and run manifest wallclock fields.
- Run batches from explicit plan JSON, with no hidden candidate expansion or
  adaptive replacement.

Out of scope:

- Progress scoring semantics.
- Model training.
- Prediction dataset construction.
- Failure-selection policy.
- Adaptive collection policy.

Current follow-ups:

- Keep `scripts/run_trace_batch.py` aligned with new provider and sandbox knobs.
- Add more trace-audit fixtures when new collection kinds are introduced.
- Preserve historical v0 material under `_archive/coding-data-collection-v0/`.
