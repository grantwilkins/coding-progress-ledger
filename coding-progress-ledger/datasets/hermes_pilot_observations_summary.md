# Ledger Observations v0 Summary

Event rows preserve replay fidelity with one row per LedgerEvent prefix. Step rows keep the final state for each (run_id, step) and are intended for plotting and later modeling-oriented analysis.

## Totals

- Total runs: 5
- Event rows: 61
- Step rows: 59
- Successful runs: 0
- Failed runs: 0
- Unknown success runs: 5

## Category Resolution

Event rows by category resolution mode:

- `native`: 61

Step rows by category resolution mode:

- `native`: 59

Runs with native/resolved metric mismatch: none

## Non-monotonic Coding Progress

Event-level: `hermes_pilot_01`, `hermes_pilot_02`, `hermes_pilot_03`, `hermes_pilot_04`, `hermes_pilot_05`.

Step-level: `hermes_pilot_01`, `hermes_pilot_02`, `hermes_pilot_03`, `hermes_pilot_04`, `hermes_pilot_05`.

## Largest Event-Level Coding Drops

- `hermes_pilot_01`: 0.500000 (product)
- `hermes_pilot_02`: 0.500000 (validation)
- `hermes_pilot_03`: 0.500000 (product)
- `hermes_pilot_04`: 0.500000 (product)
- `hermes_pilot_05`: 0.500000 (product)

## Largest Step-Level Coding Drops

- `hermes_pilot_01`: 0.500000 (product)
- `hermes_pilot_02`: 0.500000 (validation)
- `hermes_pilot_03`: 0.500000 (product)
- `hermes_pilot_04`: 0.500000 (product)
- `hermes_pilot_05`: 0.500000 (product)

## Largest Event-Level Overall Drops

- `hermes_pilot_01`: 0.500000 (product)
- `hermes_pilot_02`: 0.500000 (validation)
- `hermes_pilot_03`: 0.500000 (product)
- `hermes_pilot_04`: 0.500000 (product)
- `hermes_pilot_05`: 0.333333 (product)

## Largest Step-Level Overall Drops

- `hermes_pilot_01`: 0.500000 (product)
- `hermes_pilot_02`: 0.500000 (validation)
- `hermes_pilot_03`: 0.500000 (product)
- `hermes_pilot_04`: 0.500000 (product)
- `hermes_pilot_05`: 0.333333 (product)

## Event vs Step

Runs where event-level and step-level largest coding drops differ: none

Runs with multiple events at the same step: `hermes_pilot_05`.

## Success / Progress Quadrants

- Success + high progress: none
- Success + low progress: none
- Failure + high progress: none
- Failure + low progress: none
- Unknown success: `hermes_pilot_01`, `hermes_pilot_02`, `hermes_pilot_03`, `hermes_pilot_04`, `hermes_pilot_05`.

## Sanity Check Warnings

- hermes_pilot_01: final_success is unknown
- hermes_pilot_02: final_success is unknown
- hermes_pilot_03: final_success is unknown
- hermes_pilot_04: final_success is unknown
- hermes_pilot_05: final_success is unknown
