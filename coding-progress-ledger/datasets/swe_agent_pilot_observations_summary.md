# Ledger Observations v0 Summary

Event rows preserve replay fidelity with one row per LedgerEvent prefix. Step rows keep the final state for each (run_id, step) and are intended for plotting and later modeling-oriented analysis.

## Totals

- Total runs: 20
- Event rows: 202
- Step rows: 191
- Successful runs: 7
- Failed runs: 13
- Unknown success runs: 0

## Category Resolution

Event rows by category resolution mode:

- `mixed`: 192
- `native`: 10

Step rows by category resolution mode:

- `mixed`: 181
- `native`: 10

Runs with native/resolved metric mismatch: `swe_agent_pilot_s_03`.

## Non-monotonic Coding Progress

Event-level: `swe_agent_pilot_f_01`, `swe_agent_pilot_f_02`, `swe_agent_pilot_f_03`, `swe_agent_pilot_f_04`, `swe_agent_pilot_f_05`, `swe_agent_pilot_f_06`, `swe_agent_pilot_f_07`, `swe_agent_pilot_f_08`, `swe_agent_pilot_f_09`, `swe_agent_pilot_f_10`, `swe_agent_pilot_s_01`, `swe_agent_pilot_s_02`, `swe_agent_pilot_s_03`, `swe_agent_pilot_s_04`, `swe_agent_pilot_s_05`, `swe_agent_pilot_s_06`, `swe_agent_pilot_s_07`, `swe_agent_pilot_s_08`, `swe_agent_pilot_s_09`, `swe_agent_pilot_s_10`.

Step-level: `swe_agent_pilot_f_01`, `swe_agent_pilot_f_02`, `swe_agent_pilot_f_03`, `swe_agent_pilot_f_04`, `swe_agent_pilot_f_05`, `swe_agent_pilot_f_06`, `swe_agent_pilot_f_07`, `swe_agent_pilot_f_08`, `swe_agent_pilot_f_09`, `swe_agent_pilot_f_10`, `swe_agent_pilot_s_01`, `swe_agent_pilot_s_02`, `swe_agent_pilot_s_03`, `swe_agent_pilot_s_04`, `swe_agent_pilot_s_05`, `swe_agent_pilot_s_06`, `swe_agent_pilot_s_07`, `swe_agent_pilot_s_08`, `swe_agent_pilot_s_09`, `swe_agent_pilot_s_10`.

## Largest Event-Level Coding Drops

- `swe_agent_pilot_f_01`: 0.500000 (product)
- `swe_agent_pilot_f_02`: 0.500000 (investigation)
- `swe_agent_pilot_f_03`: 0.500000 (investigation)
- `swe_agent_pilot_f_04`: 0.500000 (product)
- `swe_agent_pilot_f_05`: 0.500000 (investigation)
- `swe_agent_pilot_f_06`: 0.500000 (investigation)
- `swe_agent_pilot_f_07`: 0.500000 (validation)
- `swe_agent_pilot_f_08`: 0.500000 (investigation)
- `swe_agent_pilot_f_09`: 0.500000 (product)
- `swe_agent_pilot_f_10`: 0.500000 (investigation)

## Largest Step-Level Coding Drops

- `swe_agent_pilot_s_04`: 0.666667 (mixed)
- `swe_agent_pilot_f_01`: 0.500000 (product)
- `swe_agent_pilot_f_02`: 0.500000 (investigation)
- `swe_agent_pilot_f_03`: 0.500000 (investigation)
- `swe_agent_pilot_f_04`: 0.500000 (product)
- `swe_agent_pilot_f_05`: 0.500000 (investigation)
- `swe_agent_pilot_f_06`: 0.500000 (investigation)
- `swe_agent_pilot_f_07`: 0.500000 (validation)
- `swe_agent_pilot_f_08`: 0.500000 (investigation)
- `swe_agent_pilot_f_09`: 0.500000 (product)

## Largest Event-Level Overall Drops

- `swe_agent_pilot_f_01`: 0.500000 (product)
- `swe_agent_pilot_f_02`: 0.500000 (investigation)
- `swe_agent_pilot_f_03`: 0.500000 (investigation)
- `swe_agent_pilot_f_04`: 0.500000 (product)
- `swe_agent_pilot_f_05`: 0.500000 (investigation)
- `swe_agent_pilot_f_06`: 0.500000 (investigation)
- `swe_agent_pilot_f_07`: 0.500000 (validation)
- `swe_agent_pilot_f_08`: 0.500000 (investigation)
- `swe_agent_pilot_f_09`: 0.500000 (product)
- `swe_agent_pilot_f_10`: 0.500000 (investigation)

## Largest Step-Level Overall Drops

- `swe_agent_pilot_s_04`: 0.666667 (mixed)
- `swe_agent_pilot_f_01`: 0.500000 (product)
- `swe_agent_pilot_f_02`: 0.500000 (investigation)
- `swe_agent_pilot_f_03`: 0.500000 (investigation)
- `swe_agent_pilot_f_04`: 0.500000 (product)
- `swe_agent_pilot_f_05`: 0.500000 (investigation)
- `swe_agent_pilot_f_06`: 0.500000 (investigation)
- `swe_agent_pilot_f_07`: 0.500000 (validation)
- `swe_agent_pilot_f_08`: 0.500000 (investigation)
- `swe_agent_pilot_f_09`: 0.500000 (product)

## Event vs Step

Runs where event-level and step-level largest coding drops differ: `swe_agent_pilot_s_04`.

Runs with multiple events at the same step: `swe_agent_pilot_f_01`, `swe_agent_pilot_f_04`, `swe_agent_pilot_f_05`, `swe_agent_pilot_f_09`, `swe_agent_pilot_s_01`, `swe_agent_pilot_s_03`, `swe_agent_pilot_s_04`, `swe_agent_pilot_s_07`, `swe_agent_pilot_s_09`, `swe_agent_pilot_s_10`.

## Success / Progress Quadrants

- Success + high progress: `swe_agent_pilot_s_01`, `swe_agent_pilot_s_02`, `swe_agent_pilot_s_05`, `swe_agent_pilot_s_07`, `swe_agent_pilot_s_08`, `swe_agent_pilot_s_10`.
- Success + low progress: `swe_agent_pilot_s_04`.
- Failure + high progress: `swe_agent_pilot_f_06`, `swe_agent_pilot_f_09`, `swe_agent_pilot_s_03`, `swe_agent_pilot_s_06`, `swe_agent_pilot_s_09`.
- Failure + low progress: `swe_agent_pilot_f_01`, `swe_agent_pilot_f_02`, `swe_agent_pilot_f_03`, `swe_agent_pilot_f_04`, `swe_agent_pilot_f_05`, `swe_agent_pilot_f_07`, `swe_agent_pilot_f_08`, `swe_agent_pilot_f_10`.
- Unknown success: none

## Sanity Check Warnings

- swe_agent_pilot_s_03: native/resolved metrics differ
