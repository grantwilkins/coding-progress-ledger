# L Pilot Gate Report Status

Status: gate machinery done; current smoke corpus is no-go for scale.

Implemented:

- Fail-closed gate aggregator for every gate in `docs/PILOT_GATES.md`.
- JSON gate report writer.
- Failure-analysis writer whenever any gate fails.

Command:

```bash
uv run python scripts/build_pilot_gate_report.py runs/d_smoke \
  --estimator-artifact-dir datasets/d_smoke_estimator \
  --out reports/PILOT_GATE_REPORT.json \
  --failure-out reports/PILOT_FAILURE_ANALYSIS.md
```

Current smoke-corpus result:

```text
passed=false
failed_gates:
  median_transcript_steps
  high_progress_failures_or_disagreements
  median_observation_events_per_run
```

Passing gate evidence:

```text
terminal_failure_rate=0.25
validation_attempt_run_fraction=1.0
validation_fail_observed_run_fraction=0.25
progress_drop_run_fraction=1.0
shell_exit_code_coverage=1.0
shell_stdout_snippet_coverage=1.0
shell_stderr_snippet_coverage=1.0
prefix_provenance_present=true
estimator_prefix_safety_and_alignment=true
leakage_incidents=0
verifier_determinism_passed=true
```

Scale decision:

- Do not run Workstream M.
- K can only proceed after the I/K preflight and after replacing the smoke
  corpus with the real 24-run pilot corpus.
