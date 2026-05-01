# Inter-annotator comparison

Comparison of two independent annotation passes over the same SWE-agent pilots. Per-pilot metrics: final coding-progress delta, leaf count delta, category-vector L1 distance, REOPEN/BLOCK count deltas. "v1" is the original annotator; "v2" is the second.

## Aggregate

- Pairs compared: **5**
- Mean coding-progress delta (v2 − v1): **+0.100**
- Mean *absolute* coding-progress delta: **0.100**
- Mean leaf count delta (v2 − v1): **+1.00**
- Mean category-vector L1 distance: **1.80**
- Verdict distribution: {'low': 2, 'moderate': 2, 'high': 1}

## Per pilot

| pilot | v1 progress | v2 progress | Δ | v1 leaves | v2 leaves | cat L1 | Δ reopens | Δ blocks | verdict |
|-------|------------:|------------:|--:|----------:|----------:|-------:|----------:|---------:|--------:|
| `swe_agent_pilot_f_01` | 0.67 | 1.00 | +0.33 | 4 | 3 | 1 | +0 | +0 | low |
| `swe_agent_pilot_f_03` | 0.50 | 0.67 | +0.17 | 2 | 3 | 1 | +0 | +0 | moderate |
| `swe_agent_pilot_f_06` | 1.00 | 1.00 | +0.00 | 5 | 7 | 2 | +0 | +0 | moderate |
| `swe_agent_pilot_s_01` | 1.00 | 1.00 | +0.00 | 6 | 7 | 1 | +0 | +0 | high |
| `swe_agent_pilot_s_03` | 1.00 | 1.00 | +0.00 | 6 | 8 | 4 | -1 | +0 | low |

## Per-pilot detail

### `swe_agent_pilot_f_01`

- v1 root_task: "Remove getip.php request to soon-decommissioned SL6 server"
- v2 root_task: "Remove or replace the http://simprod.icecube.wisc.edu/downloads/getip.php request (server is being decommissioned)"

| field | v1 | v2 |
|-------|----|----|
| n_leaves | 4 | 3 |
| coding_progress | 0.667 | 1.000 |
| overall_progress | 0.750 | 1.000 |
| category_counts | {'INVESTIGATION': 1, 'PRODUCT': 1, 'VALIDATION': 1, 'ARTIFACT': 1} | {'INVESTIGATION': 1, 'PRODUCT': 1, 'ARTIFACT': 1} |
| status_counts | {'COMPLETE': 3, 'NOT_STARTED': 1} | {'COMPLETE': 3} |
| n_reopens | 0 | 0 |
| n_blocks | 0 | 0 |
| annotation_minutes | 20 | 18 |

### `swe_agent_pilot_f_03`

- v1 root_task: "Fix configparser downcasing keys in unrelated sections"
- v2 root_task: "Stop configparser from downcasing keys in unrelated sections (e.g. DJANGO_SETTINGS_MODULE under [tool:pytest]) so setup-cfg-fmt round-trips them"

| field | v1 | v2 |
|-------|----|----|
| n_leaves | 2 | 3 |
| coding_progress | 0.500 | 0.667 |
| overall_progress | 0.500 | 0.667 |
| category_counts | {'INVESTIGATION': 2} | {'INVESTIGATION': 2, 'VALIDATION': 1} |
| status_counts | {'COMPLETE': 1, 'BLOCKED': 1} | {'COMPLETE': 2, 'BLOCKED': 1} |
| n_reopens | 0 | 0 |
| n_blocks | 1 | 1 |
| annotation_minutes | 40 | 35 |

### `swe_agent_pilot_f_06`

- v1 root_task: "Add support for inserting decimal.Decimal values into Spanner NUMERIC fields"
- v2 root_task: "Add support for mapping python decimal.Decimal to Spanner NUMERIC param type so float-decimal values can be inserted into a NUMERIC field"

| field | v1 | v2 |
|-------|----|----|
| n_leaves | 5 | 7 |
| coding_progress | 1.000 | 1.000 |
| overall_progress | 1.000 | 1.000 |
| category_counts | {'INVESTIGATION': 2, 'PRODUCT': 1, 'VALIDATION': 1, 'ARTIFACT': 1} | {'INVESTIGATION': 2, 'VALIDATION': 2, 'PRODUCT': 1, 'ENVIRONMENT': 1, 'ARTIFACT': 1} |
| status_counts | {'COMPLETE': 5} | {'COMPLETE': 7} |
| n_reopens | 0 | 0 |
| n_blocks | 0 | 0 |
| annotation_minutes | 22 | 28 |

### `swe_agent_pilot_s_01`

- v1 root_task: "Fix incorrect counting for binary logical operators"
- v2 root_task: "Drop the nesting increment for sequences of binary logical operators (B3) in cognitive_complexity, per the issue spec"

| field | v1 | v2 |
|-------|----|----|
| n_leaves | 6 | 7 |
| coding_progress | 1.000 | 1.000 |
| overall_progress | 1.000 | 1.000 |
| category_counts | {'INVESTIGATION': 1, 'PRODUCT': 2, 'VALIDATION': 2, 'ARTIFACT': 1} | {'INVESTIGATION': 1, 'PRODUCT': 3, 'VALIDATION': 2, 'ARTIFACT': 1} |
| status_counts | {'COMPLETE': 6} | {'COMPLETE': 7} |
| n_reopens | 0 | 0 |
| n_blocks | 0 | 0 |
| annotation_minutes | 35 | 25 |

### `swe_agent_pilot_s_03`

- v1 root_task: "Fix UnboundLocalError in ConstantTeambuilder for showdown-format teams without items"
- v2 root_task: "Fix UnboundLocalError in Teambuilder.parse_showdown_team when a Pokemon line has no '@ <item>'"

| field | v1 | v2 |
|-------|----|----|
| n_leaves | 6 | 8 |
| coding_progress | 1.000 | 1.000 |
| overall_progress | 1.000 | 1.000 |
| category_counts | {'INVESTIGATION': 2, 'PRODUCT': 2, 'VALIDATION': 1, 'ARTIFACT': 1} | {'INVESTIGATION': 2, 'VALIDATION': 3, 'ENVIRONMENT': 1, 'PRODUCT': 1, 'ARTIFACT': 1} |
| status_counts | {'COMPLETE': 6} | {'COMPLETE': 8} |
| n_reopens | 1 | 0 |
| n_blocks | 0 | 0 |
| annotation_minutes | 22 | 30 |

