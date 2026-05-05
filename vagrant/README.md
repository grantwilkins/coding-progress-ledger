# vagrant-agent

State-mobility layer between agent harnesses and serving/runtime backends. Given an agent workflow trace and a placement-change event, decides what state moves together, what splits, and how each state object is materialized at the destination.

This is **not** a new agent harness or a new serving engine. It is a derivation pipeline:

```text
trace -> manifest -> placement plan -> cost estimate -> plot
```

## Status

Pre-MVP. See `TASKS.md` for the working backlog. Workstream A (trace) is in progress.

## Read first

- `CLAUDE.md` — full rules and reuse contract.
- `AGENTS.md` — coding rules (succinct, hard-fail, test, commit).
- `TASKS.md` — the authoritative backlog with workstreams and gates.

## Sibling repos

- `../coding-progress-ledger/` — the ledger framework. Vagrant imports from this; do not fork it.
- `../coding-estimator/`, `../coding-data-collection/` — downstream/upstream of the ledger.
