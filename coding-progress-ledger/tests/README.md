# tests

Pytest suite for ledger semantics, adapters, scripts, live-run utilities, and dataset invariants.

## Coverage themes

- Core semantics: replay, scoring, categories, serialization, sessions.
- Source adapters: SWE-agent and Hermes normalize/import behavior.
- Dataset pipeline: observation tables, labels, checkpoints, audits.
- Run tooling: run manager, sidecar/live query behavior.
- Protocol hardening: invariants and regression tests for known failure modes.

## Run

```sh
uv run pytest
```
