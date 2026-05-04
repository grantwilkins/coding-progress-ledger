# annotations/swe_agent_pilot

Canonical SWE-agent pilot annotation specs for the 20-run balanced sample.

## Contents

- `swe_agent_pilot_*.json`: structured ledgers-to-build specs.
- `swe_agent_pilot_*.notes.md`: run-level evidence interpretation.

## Role in pipeline

These specs drive retrospective ledger materialization under `runs/swe_agent_pilot/` and downstream observation datasets in `datasets/`.

## Constraints

- Keep event ordering stable.
- Preserve explicit evidence mapping in notes.
- Treat these specs as versioned research artifacts.
