# Evidence availability audit (K1)

Reuses `scripts/rescore_suite_by_category.py` (`audit_completion_evidence`, `classify_evidence`). Strong evidence = `test_output | diff | file_exists | command_output` (plus `contract_text` for INVESTIGATION leaves on understanding-style descriptions).

## 1. Headline

- Pilots audited: **20**
- Total completion events: **81**
- Completions with at least one strong evidence type: **51** (63%)
- Completions with `manual_note` only: **30** (37%)

## 2. Evidence-type counts (across all completion events)

| Evidence type | Count |
|---|---:|
| `diff` | 47 |
| `manual_note` | 30 |
| `test_output` | 5 |
| `file_exists` | 2 |
| `command_output` | 1 |

## 3. Weak completions by category

| Category | Audited completions | Weak | Weak rate |
|---|---:|---:|---:|
| product | 24 | 20 | 83% |
| validation | 12 | 1 | 8% |
| investigation | 31 | 1 | 3% |

## 4. Per-pilot status

| Pilot | Status | Weak total | weak prod / val / inv |
|---|---|---:|---|
| `swe_agent_pilot_f_01` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_f_02` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_03` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_04` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_f_05` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_f_06` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_f_07` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_f_08` | weak | 2 | 2 / 0 / 0 |
| `swe_agent_pilot_f_09` | weak | 2 | 2 / 0 / 0 |
| `swe_agent_pilot_f_10` | strong | 0 | 0 / 0 / 0 |
| `swe_agent_pilot_s_01` | weak | 2 | 2 / 0 / 0 |
| `swe_agent_pilot_s_02` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_s_03` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_s_04` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_s_05` | weak | 2 | 2 / 0 / 0 |
| `swe_agent_pilot_s_06` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_s_07` | weak | 3 | 1 / 1 / 1 |
| `swe_agent_pilot_s_08` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_s_09` | weak | 1 | 1 / 0 / 0 |
| `swe_agent_pilot_s_10` | weak | 1 | 1 / 0 / 0 |

## 5. Notes on interpretation

- Weak evidence is a **signal**, not a replay failure. The framework allows `manual_note` evidence on completed subtasks; this audit measures *how often* that fallback fires.
- A pilot's status is `weak` if any audited completion has only weak evidence; `strong` if all completions have at least one strong evidence type. `not_applicable` if the pilot has zero PRODUCT/VALIDATION/INVESTIGATION completions in scope.
- The `contract_text` carve-out for INVESTIGATION captures "understanding what the issue asks for" as legitimate completion evidence on a leaf whose description is about contract-reading.

## 6. Pointers

- Classifier: `scripts/rescore_suite_by_category.py:classify_evidence`
- Audit fn: `scripts/rescore_suite_by_category.py:audit_completion_evidence`
- This script: `scripts/audit_pilot_evidence.py`
