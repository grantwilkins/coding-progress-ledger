# Evidence availability audit (K1)

Reuses `scripts/rescore_suite_by_category.py` (`audit_completion_evidence`, `classify_evidence`). Strong evidence = `test_output | diff | file_exists | command_output` (plus `contract_text` for INVESTIGATION leaves on understanding-style descriptions).

## 1. Headline

- Pilots audited: **20**
- Total completion events: **81**
- Completions with at least one strong evidence type: **79** (98%)
- Completions with `manual_note` only: **2** (2%)

## 2. Evidence-type counts (across all completion events)

| Evidence type | Count |
|---|---:|
| `tool_action` | 65 |
| `diff` | 47 |
| `test_output` | 5 |
| `file_exists` | 2 |
| `manual_note` | 2 |
| `command_output` | 1 |

## 2b. Evidence levels (K4)

| Level | product | validation | investigation | total |
|---|---:|---:|---:|---:|
| mechanical | 24 | 11 | 30 | 65 |
| trace_semantic | 0 | 0 | 0 | 0 |
| annotator_judgment | 0 | 1 | 1 | 2 |

Levels: `mechanical` = test/command output, diff, file_exists, tool_action; `trace_semantic` = contract_text on understanding-style leaves; `annotator_judgment` = manual_note fallback. Trace_semantic counts are candidates for live sidecar automation.

## 3. Weak completions by category

| Category | Audited completions | Weak | Weak rate |
|---|---:|---:|---:|
| product | 24 | 0 | 0% |
| validation | 12 | 1 | 8% |
| investigation | 31 | 1 | 3% |

## 4. Per-pilot status

| Pilot | Status | Weak total | weak prod / val / inv |
|---|---|---:|---|
| `swe_agent_pilot_f_01` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_02` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_03` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_04` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_05` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_06` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_07` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_08` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_09` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_10` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_01` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_02` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_03` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_04` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_05` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_06` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_07` | weak | 2 | 0 / 1 / 1 |
| `swe_agent_pilot_s_08` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_09` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_10` | strong | 0 | 0 / 0 / 0 |

## 5. Notes on interpretation

- Weak evidence is a **signal**, not a replay failure. The framework allows `manual_note` evidence on completed subtasks; this audit measures *how often* that fallback fires.
- A pilot's status is `weak` if any audited completion has only weak evidence; `strong` if all completions have at least one strong evidence type. `not_applicable` if the pilot has zero PRODUCT/VALIDATION/INVESTIGATION completions in scope.
- The `contract_text` carve-out for INVESTIGATION captures "understanding what the issue asks for" as legitimate completion evidence on a leaf whose description is about contract-reading.

## 6. Pointers

- Classifier: `scripts/rescore_suite_by_category.py:classify_evidence`
- Audit fn: `scripts/rescore_suite_by_category.py:audit_completion_evidence`
- This script: `scripts/audit_pilot_evidence.py`
