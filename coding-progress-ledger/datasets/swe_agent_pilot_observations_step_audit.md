# Ledger Observations v0 Audit

Audit of checkpoint-level observation CSV coherence.

## Totals

- Rows: 191
- Runs: 20
- Integrity passed: yes

## Integrity

- Invalid progress values: 0
- Completed > active failures: 0
- Delta mismatches: 0
- First-row nonzero deltas: 0
- Missing identifiers: 0
- Invalid success metadata: 0
- Unknown success metadata: 0

## Category Resolution

| Mode | Rows |
| --- | ---: |
| `native` | 191 |

Runs with native/resolved metric mismatch: none

## Drops

- Negative coding deltas: 58
- Negative overall deltas: 63

Coding drop sources:

| Source | Count |
| --- | ---: |
| `investigation` | 14 |
| `mixed` | 1 |
| `product` | 25 |
| `validation` | 18 |

Overall drop sources:

| Source | Count |
| --- | ---: |
| `artifact` | 5 |
| `investigation` | 14 |
| `mixed` | 1 |
| `product` | 25 |
| `validation` | 18 |

## Event vs Step

Runs where largest event-level and step-level coding drops differ: none

Runs with multiple events at the same step: none

## Success / Progress Quadrants

- Success + high progress: `swe_agent_pilot_s_01`, `swe_agent_pilot_s_02`, `swe_agent_pilot_s_03`, `swe_agent_pilot_s_05`, `swe_agent_pilot_s_06`, `swe_agent_pilot_s_07`, `swe_agent_pilot_s_08`, `swe_agent_pilot_s_09`, `swe_agent_pilot_s_10`
- Success + low progress: `swe_agent_pilot_s_04`
- Failure + high progress: `swe_agent_pilot_f_06`, `swe_agent_pilot_f_09`
- Failure + low progress: `swe_agent_pilot_f_01`, `swe_agent_pilot_f_02`, `swe_agent_pilot_f_03`, `swe_agent_pilot_f_04`, `swe_agent_pilot_f_05`, `swe_agent_pilot_f_07`, `swe_agent_pilot_f_08`, `swe_agent_pilot_f_10`
- Unknown success: none

## Warnings

- none
