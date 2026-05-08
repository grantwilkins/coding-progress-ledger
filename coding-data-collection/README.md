# coding-data-collection

Experiment harness and data factory for long-horizon coding-agent traces.

This repository coordinates `../coding-progress-ledger` and
`../coding-estimator` operationally without absorbing their scientific
responsibilities.

```text
orchestration/adapters/artifacts -> coding-data-collection
progress semantics/replay/scoring -> coding-progress-ledger
features/labels/models/eval      -> coding-estimator
```

Primary first substrate: Terminal-Bench / Terminal-Bench 2.0.
SWE-bench Pro remains inspect-only until the Terminal-Bench pilot passes.

## Boundary

This repo may:

- inspect benchmark sources;
- choose and document collection policy;
- prepare isolated agent workspaces;
- emit raw transcripts and observation events;
- call `coding-progress-ledger` sidecar replay;
- call `coding-estimator` artifact builders;
- audit leakage, artifact completeness, prefix safety, and pilot gates.

This repo must not:

- redefine ledger progress;
- reimplement ledger replay or scoring;
- add estimator model classes;
- redefine estimator features or labels;
- expose oracle solutions, gold patches, hidden tests, or verifier internals
  to the agent phase.

## First Milestone

The first deliverable is the Phase 0 feasibility spike:

1. Run one Terminal-Bench task through Harbor oracle.
2. Run one task through `ia03/terminal-bench` archive extraction if feasible.
3. For each path, prove oracle/test hiding, transcript capture, native
   `observation_events.jsonl`, ledger sidecar replay, verifier
   reproducibility, and artifact completeness.
4. Choose Harbor-native, HF-archive custom, or hybrid before the 24-run pilot.

