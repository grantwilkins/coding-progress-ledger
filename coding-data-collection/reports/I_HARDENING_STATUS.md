# I Audits and Hardening Status

Status: done for pre-K smoke corpus hardening.

Implemented:

- Visibility-aware redaction audit over:
  - `task.md`
  - agent-visible `transcript.jsonl` content
  - agent-visible `events.jsonl::agent_step` content
  - agent-visible `observation_events.jsonl` payloads
- Validation-attempt precision sample over emitted positives.
- Validation-attempt miss sample over non-attempt shell rows.
- Corpus artifact completeness and schema validation command.

Command:

```bash
uv run python scripts/audit_corpus_artifacts.py runs/d_smoke \
  --out reports/I_HARDENING_AUDIT_REPORT.json
```

Result:

```text
passed=true
run_count=4
redaction.leakage_incidents=0
artifact_completeness.passed=true
validation_attempt_precision.sample_precision=1.0
validation_attempt_precision.recall_miss_rate=0.0
```

Notes:

- Harness Docker wrapper command strings are not treated as agent-visible
  task content. Their stdout/stderr/observation payloads are still scanned.
- Hidden verifier/oracle rows are retained for auditability but must remain
  non-agent-visible.
