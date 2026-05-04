# ledger_progress

Core package implementing append-only progress ledgers, replay, scoring, and run utilities.

## Modules

- `core.py`: event/status/category enums and replay engine.
- `session.py`: `LedgerSession` write API.
- `scoring.py`: leaf-based progress computation.
- `queries.py`: reusable query helpers and coding-category filters.
- `serialization.py`: JSONL and dataclass serialization helpers.
- `run_manager.py`: filesystem run management + CLI integration.
- `sidecar.py`: live instrumentation sidecar helpers.
- `set_*`: multi-run set abstractions.
- `adapters/`: trace-adapter glue code.

## Invariants

- Event log is the source of truth.
- Progress is about discovered active leaf work, not final correctness.
- Reopen/split/invalidate semantics are first-class and can reduce progress.
