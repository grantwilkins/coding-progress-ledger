# Schema gaps collected from pilot run notes (I1)

Pilots scanned: **20**. 
Pilots with `whether_schema_gap_found = true`: **2**. 
Pilots reporting 'None' in § 8: **18**.

## 1. Pilots that flagged a schema gap

### swe_agent_pilot_f_02

**One real gap, surfaced by f_02 and resolved before annotating:**
the original stuck-loop rule only covered cycles of identical
commands. f_02's failure mode is "agent varies the command on every
step but every tool response is identical" -- a tool-response loop
rather than a command loop. The literal command-loop rule did not
trigger. Refined § 6 to include variant (b): tool-response-loop,
fires on three identical/near-identical tool responses regardless
of query variation. Annotated under the refined rule.

### swe_agent_pilot_f_07

**One real gap, surfaced by f_07 and resolved before annotating:**
the original stuck-loop rule (general § 6) was ambiguous on cycle
length — the wording "same sequence of N ≥ 3 commands" could be read
as "the cycle is at least 3 commands long". f_07 has both a
1-command cycle (`edit 1:20` ×5) and a 2-command cycle
(`edit 5:5` / `edit 21:21` ×~55). Neither would have triggered under
the strict reading, but both are obviously stuck. Refined the rule to
"cycle of any length, including 1 or 2; mark blocked at the earliest
step where any such pattern hits its third iteration" before
annotating.

No other gaps. Categories, statuses, event types, and step-numbering
all covered the trace.

## 2. Pilots that explicitly reported no schema gap

18 pilots: `swe_agent_pilot_f_01`, `swe_agent_pilot_f_03`, `swe_agent_pilot_f_04`, `swe_agent_pilot_f_05`, `swe_agent_pilot_f_06`, `swe_agent_pilot_f_08`, `swe_agent_pilot_f_09`, `swe_agent_pilot_f_10`, `swe_agent_pilot_s_01`, `swe_agent_pilot_s_02`, `swe_agent_pilot_s_03`, `swe_agent_pilot_s_04`, `swe_agent_pilot_s_05`, `swe_agent_pilot_s_06`, `swe_agent_pilot_s_07`, `swe_agent_pilot_s_08`, `swe_agent_pilot_s_09`, `swe_agent_pilot_s_10`

## 4. Cross-workstream findings (post-pilot)

### v1 inconsistently applied Pitfall #8 across harness-terminated failure pilots

- **Severity:** annoying
- **Class:** protocol application (not schema)
- **Source:** H4 GATE_RESULT § 7

The HIGH-severity H3 revision (Pitfall #8: bug-fix tasks always carry an implicit `VALIDATION` leaf) was applied by v1 to `f_01` / `f_04` / `s_04` (submit-without-test) but not to `f_02` / `f_03` / `f_07` / `f_10` (harness-forced termination mid-loop). The schema is fine; this is a protocol-application gap. Fix: re-emit specs for the four pilots adding a not_started VAL leaf. Estimated effort ~30 min.

### Builder reports `category_resolution_mode = mixed` for 181/191 SWE-agent step rows

- **Severity:** note
- **Class:** category resolution / pipeline
- **Source:** datasets/swe_agent_pilot_observations_step_audit.md

Annotation specs assign categories explicitly on every add/split, yet the builder records 'mixed' for nearly all step rows and 'native' for only 10. One run (`s_03`) carries a 'large native/resolved divergence' warning. This is investigated and resolved in J1; logged here so the gap is visible in the schema-gap collection.

### `final_success` heuristic from `test_output.txt` mis-classified 3 SWE-agent successes

- **Severity:** blocker (resolved)
- **Class:** label leakage / heuristic drift
- **Source:** datasets/observation_distribution_comparison.md § 3.6

Pre-fix: builder's `resolve_final_success` keyword-scanned `test_output.txt`, misclassifying `s_03` / `s_06` / `s_09` as failures. Fix (commit 7df39ba): honor `source_metadata.json:final_success` whenever the importer pinned an authoritative label. Listed here as a load-bearing pilot finding even though it was schema-adjacent (heuristic-driven) rather than schema-shaped.

## 5. Classification summary

| Gap | Class | Severity | Source/Status |
|---|---|---|---|
| `f_02` stuck-loop rule covered command loops only | missing protocol coverage | annoying | in-pilot, resolved |
| `f_07` stuck-loop rule ambiguous on cycle length | missing protocol coverage | annoying | in-pilot, resolved |
| v1 inconsistently applied Pitfall #8 across harness-terminated failure pilots | protocol application (not schema) | annoying | H4 GATE_RESULT § 7 |
| Builder reports `category_resolution_mode = mixed` for 181/191 SWE-agent step rows | category resolution / pipeline | note | datasets/swe_agent_pilot_observations_step_audit.md |
| `final_success` heuristic from `test_output.txt` mis-classified 3 SWE-agent successes | label leakage / heuristic drift | blocker (resolved) | datasets/observation_distribution_comparison.md § 3.6 |
