# Ledger Observations v0 Summary

Event rows preserve replay fidelity with one row per LedgerEvent prefix. Step rows keep the final state for each (run_id, step) and are intended for plotting and later modeling-oriented analysis.

## Totals

- Total runs: 30
- Event rows: 370
- Step rows: 370
- Successful runs: 0
- Failed runs: 0
- Unknown success runs: 30

## Category Resolution

Event rows by category resolution mode:

- `native`: 370

Step rows by category resolution mode:

- `native`: 370

Runs with native/resolved metric mismatch: none

## Non-monotonic Coding Progress

Event-level: `hermes_pilot_h5_002`, `hermes_pilot_h5_003`, `hermes_pilot_h5_004`, `hermes_pilot_h5_005`, `hermes_pilot_h5_007`, `hermes_pilot_h5_008`, `hermes_pilot_h5_009`, `hermes_pilot_h5_010`, `hermes_pilot_h5_011`, `hermes_pilot_h5_012`, `hermes_pilot_h5_013`, `hermes_pilot_h5_014`, `hermes_pilot_h5_015`, `hermes_pilot_h5_017`, `hermes_pilot_h5_018`, `hermes_pilot_h5_019`, `hermes_pilot_h5_020`, `hermes_pilot_h5_022`, `hermes_pilot_h5_023`, `hermes_pilot_h5_024`, `hermes_pilot_h5_025`, `hermes_pilot_h5_027`, `hermes_pilot_h5_028`, `hermes_pilot_h5_029`, `hermes_pilot_h5_030`.

Step-level: `hermes_pilot_h5_002`, `hermes_pilot_h5_003`, `hermes_pilot_h5_004`, `hermes_pilot_h5_005`, `hermes_pilot_h5_007`, `hermes_pilot_h5_008`, `hermes_pilot_h5_009`, `hermes_pilot_h5_010`, `hermes_pilot_h5_011`, `hermes_pilot_h5_012`, `hermes_pilot_h5_013`, `hermes_pilot_h5_014`, `hermes_pilot_h5_015`, `hermes_pilot_h5_017`, `hermes_pilot_h5_018`, `hermes_pilot_h5_019`, `hermes_pilot_h5_020`, `hermes_pilot_h5_022`, `hermes_pilot_h5_023`, `hermes_pilot_h5_024`, `hermes_pilot_h5_025`, `hermes_pilot_h5_027`, `hermes_pilot_h5_028`, `hermes_pilot_h5_029`, `hermes_pilot_h5_030`.

## Largest Event-Level Coding Drops

- `hermes_pilot_h5_002`: 0.500000 (validation)
- `hermes_pilot_h5_004`: 0.500000 (validation)
- `hermes_pilot_h5_005`: 0.500000 (validation)
- `hermes_pilot_h5_007`: 0.500000 (validation)
- `hermes_pilot_h5_008`: 0.500000 (investigation)
- `hermes_pilot_h5_010`: 0.500000 (validation)
- `hermes_pilot_h5_011`: 0.500000 (investigation)
- `hermes_pilot_h5_012`: 0.500000 (validation)
- `hermes_pilot_h5_013`: 0.500000 (investigation)
- `hermes_pilot_h5_014`: 0.500000 (validation)

## Largest Step-Level Coding Drops

- `hermes_pilot_h5_002`: 0.500000 (validation)
- `hermes_pilot_h5_004`: 0.500000 (validation)
- `hermes_pilot_h5_005`: 0.500000 (validation)
- `hermes_pilot_h5_007`: 0.500000 (validation)
- `hermes_pilot_h5_008`: 0.500000 (investigation)
- `hermes_pilot_h5_010`: 0.500000 (validation)
- `hermes_pilot_h5_011`: 0.500000 (investigation)
- `hermes_pilot_h5_012`: 0.500000 (validation)
- `hermes_pilot_h5_013`: 0.500000 (investigation)
- `hermes_pilot_h5_014`: 0.500000 (validation)

## Largest Event-Level Overall Drops

- `hermes_pilot_h5_002`: 0.500000 (validation)
- `hermes_pilot_h5_004`: 0.500000 (validation)
- `hermes_pilot_h5_005`: 0.500000 (validation)
- `hermes_pilot_h5_007`: 0.500000 (validation)
- `hermes_pilot_h5_008`: 0.500000 (investigation)
- `hermes_pilot_h5_010`: 0.500000 (validation)
- `hermes_pilot_h5_011`: 0.500000 (investigation)
- `hermes_pilot_h5_012`: 0.500000 (validation)
- `hermes_pilot_h5_013`: 0.500000 (investigation)
- `hermes_pilot_h5_014`: 0.500000 (validation)

## Largest Step-Level Overall Drops

- `hermes_pilot_h5_002`: 0.500000 (validation)
- `hermes_pilot_h5_004`: 0.500000 (validation)
- `hermes_pilot_h5_005`: 0.500000 (validation)
- `hermes_pilot_h5_007`: 0.500000 (validation)
- `hermes_pilot_h5_008`: 0.500000 (investigation)
- `hermes_pilot_h5_010`: 0.500000 (validation)
- `hermes_pilot_h5_011`: 0.500000 (investigation)
- `hermes_pilot_h5_012`: 0.500000 (validation)
- `hermes_pilot_h5_013`: 0.500000 (investigation)
- `hermes_pilot_h5_014`: 0.500000 (validation)

## Event vs Step

Runs where event-level and step-level largest coding drops differ: `hermes_pilot_h5_017`.

Runs with multiple events at the same step: none

## Success / Progress Quadrants

- Success + high progress: none
- Success + low progress: none
- Failure + high progress: none
- Failure + low progress: none
- Unknown success: `hermes_pilot_h5_001`, `hermes_pilot_h5_002`, `hermes_pilot_h5_003`, `hermes_pilot_h5_004`, `hermes_pilot_h5_005`, `hermes_pilot_h5_006`, `hermes_pilot_h5_007`, `hermes_pilot_h5_008`, `hermes_pilot_h5_009`, `hermes_pilot_h5_010`, `hermes_pilot_h5_011`, `hermes_pilot_h5_012`, `hermes_pilot_h5_013`, `hermes_pilot_h5_014`, `hermes_pilot_h5_015`, `hermes_pilot_h5_016`, `hermes_pilot_h5_017`, `hermes_pilot_h5_018`, `hermes_pilot_h5_019`, `hermes_pilot_h5_020`, `hermes_pilot_h5_021`, `hermes_pilot_h5_022`, `hermes_pilot_h5_023`, `hermes_pilot_h5_024`, `hermes_pilot_h5_025`, `hermes_pilot_h5_026`, `hermes_pilot_h5_027`, `hermes_pilot_h5_028`, `hermes_pilot_h5_029`, `hermes_pilot_h5_030`.

## Sanity Check Warnings

- hermes_pilot_h5_001: final_success is unknown
- hermes_pilot_h5_002: final_success is unknown
- hermes_pilot_h5_003: final_success is unknown
- hermes_pilot_h5_004: final_success is unknown
- hermes_pilot_h5_005: final_success is unknown
- hermes_pilot_h5_006: final_success is unknown
- hermes_pilot_h5_007: final_success is unknown
- hermes_pilot_h5_008: final_success is unknown
- hermes_pilot_h5_009: final_success is unknown
- hermes_pilot_h5_010: final_success is unknown
- hermes_pilot_h5_011: final_success is unknown
- hermes_pilot_h5_012: final_success is unknown
- hermes_pilot_h5_013: final_success is unknown
- hermes_pilot_h5_014: final_success is unknown
- hermes_pilot_h5_015: final_success is unknown
- hermes_pilot_h5_016: final_success is unknown
- hermes_pilot_h5_017: final_success is unknown
- hermes_pilot_h5_018: final_success is unknown
- hermes_pilot_h5_019: final_success is unknown
- hermes_pilot_h5_020: final_success is unknown
- hermes_pilot_h5_021: final_success is unknown
- hermes_pilot_h5_022: final_success is unknown
- hermes_pilot_h5_023: final_success is unknown
- hermes_pilot_h5_024: final_success is unknown
- hermes_pilot_h5_025: final_success is unknown
- hermes_pilot_h5_026: final_success is unknown
- hermes_pilot_h5_027: final_success is unknown
- hermes_pilot_h5_028: final_success is unknown
- hermes_pilot_h5_029: final_success is unknown
- hermes_pilot_h5_030: final_success is unknown
