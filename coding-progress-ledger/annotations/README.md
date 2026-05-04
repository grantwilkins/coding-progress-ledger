# annotations

Retrospective ledger annotation specs and notes used to materialize `runs/*` ledgers from imported traces.

## Subfolders

- `hermes_pilot/`: first Hermes pilot annotations (`.json` specs + `.notes.md` rationale).
- `swe_agent_pilot/`: canonical SWE-agent pilot annotations.
- `swe_agent_pilot_v2/`: protocol-adjusted reannotation subset.
- `swe_agent_pilot_v3/`: later reannotation subset used for agreement/protocol checks.

## File conventions

- `*.json`: structured annotation spec consumed by `scripts/annotate_pilots_from_spec.py`.
- `*.notes.md`: human reasoning, evidence references, and unresolved judgment calls.

## How annotations are used

1. Import/normalize source traces into run directories.
2. Apply spec files to emit append-only ledger events.
3. Export `progress.csv` and category summaries.
4. Build step/event observation datasets and audits in `datasets/`.

## Guardrails

- Keep annotation files deterministic and replayable.
- Do not edit generated `runs/*/ledger.jsonl` history in place.
- If protocol changes, create a new versioned annotation folder rather than rewriting prior versions.
