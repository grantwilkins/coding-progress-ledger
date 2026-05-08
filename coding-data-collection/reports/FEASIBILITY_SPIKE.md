# Feasibility Spike

Date: 2026-05-05

## Decision

Primary path: `hf_archive_custom`

Use HF archive extraction and custom Docker execution as the primary pilot
substrate. Keep Harbor as a secondary oracle/verifier smoke path for tasks that
can be migrated or downloaded reliably, but do not make Harbor the
instrumented collection substrate.

## Evidence Summary

Harbor:

- Remote Harbor Hub registry path attempted with
  `terminal-bench/terminal-bench-2` and task `terminal-bench/fix-git`.
- Registry lookup failed with a PostgREST statement timeout before task
  download.
- Local Harbor path succeeded after migrating the HF Terminal-Bench archive
  task `aimo-airline-departures` into Harbor format.
- Oracle reward: `1.0`.
- Harbor artifacts were sufficient for oracle/verifier smoke evidence, but the
  observed job output did not expose a native per-step agent transcript stream.

HF archive custom:

- Downloaded and extracted one `ia03/terminal-bench` archive:
  `aimo-airline-departures`.
- Archive contained `Dockerfile`, `docker-compose.yaml`, `run-tests.sh`,
  `solution.sh`, `task.yaml`, and `tests/test_outputs.py`.
- Agent workspace preparation skipped the oracle solution, hidden tests,
  verifier script, task YAML, Docker metadata, and canary-bearing files.
- Custom Docker build and verifier rerun passed with `4 passed`.

## Artifact Trees

Harbor local oracle run:

```text
runs/feasibility/harbor_native_aimo_run/
  task.md
  task_metadata.json
  environment_manifest.json
  protocol_manifest.json
  transcript.jsonl
  observation_events.jsonl
  events.jsonl
  ledger.jsonl
  progress.csv
  progress_by_category.csv
  summary_by_category.json
  run_manifest.json
  verifier_output.txt
  run_notes.md
  harbor_job_metadata.json
  final_diff.patch
  test_output.txt
```

HF archive custom run:

```text
runs/feasibility/hf_archive_custom_aimo_run/
  task.md
  task_metadata.json
  environment_manifest.json
  protocol_manifest.json
  transcript.jsonl
  observation_events.jsonl
  events.jsonl
  ledger.jsonl
  progress.csv
  progress_by_category.csv
  summary_by_category.json
  run_manifest.json
  verifier_output.txt
  run_notes.md
  final_diff.patch
  test_output.txt
```

Raw supporting outputs:

```text
runs/feasibility/harbor_local_aimo/2026-05-05__14-37-58/
/private/tmp/houdini_tb_hf/aimo-airline-departures/
/private/tmp/houdini_harbor_migrated/aimo-airline-departures/
```

The `/private/tmp` paths are scratch evidence only and must not be committed.

## Feasibility Questions

Can oracle/test/verifier internals be hidden from the agent phase?

Yes for `hf_archive_custom`. `prepare_run.py` skipped `solution.sh`, `tests/`,
`run-tests.sh`, `task.yaml`, `Dockerfile`, and `docker-compose.yaml`.
Leakage scan on the prepared agent workspace passed.

Can per-step transcript events be captured without fighting Harbor?

No evidence for Harbor-native capture. Harbor produced job/trial logs and
agent/verifier outputs, not a stable per-step transcript stream. Custom
execution can emit one transcript row per harness/agent/tool step directly.

Can native `observation_events.jsonl` be emitted at the right step?

Yes. Both run-shaped feasibility directories emit verifier events at
`max(transcript.step) + 1`; schema validation passed.

Can `events.jsonl` replay through the `coding-progress-ledger` sidecar?

Yes. `scripts/finalize_run.py` replayed both feasibility transcripts through
the sidecar and produced `ledger.jsonl`, `progress.csv`,
`progress_by_category.csv`, and `summary_by_category.json`.

Can the verifier be rerun from a clean verifier phase with same result?

Yes for `hf_archive_custom`: Docker built the extracted task image, mounted the
task archive read-only only for verifier/oracle execution, and reran pytest
with `4 passed`. Harbor local migration also returned reward `1.0`.

Does Harbor expose enough hooks, or is HF archive custom execution needed?

HF archive custom execution is needed for the pilot. Harbor remains useful for
oracle/verifier smoke checks, but the observed CLI surface does not provide the
agent-phase instrumentation hooks needed for native transcript and observation
event emission.

## Blockers

- Harbor remote registry lookup timed out for `terminal-bench/fix-git` on
  2026-05-05.
- Full pilot still requires Workstreams H-J: estimator artifact production,
  audits, task scoring, and pilot orchestration.
- The HF task's `run-tests.sh` installs packages over the network. Pilot policy
  needs explicit task-level network exceptions or prebuilt/cacheable verifier
  images.

## Go/No-Go

Go for implementing the pilot substrate on `hf_archive_custom`.

No-go for launching the 24-run pilot today. `scripts/run_pilot.py` should remain
blocked until the later isolation, instrumentation, prefix-safety, and pilot
gate workstreams are complete.
