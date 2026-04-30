# Pilot annotation summary (E3) — SWE-agent retrospective pilot N=20

This document satisfies `TASKS.md` § Workstream E, task **E3**. It is
a derived statistical and qualitative report over the 20 annotated
pilot runs at `runs/swe_agent_pilot/swe_agent_pilot_*/`. It contains
no upstream trace content; it summarizes the annotator-product
artifacts.

## 1. Headline numbers

| Metric                              | Value                                                                |
|-------------------------------------|----------------------------------------------------------------------|
| Pilots annotated                    | **20** (all)                                                         |
| Success / failure split (upstream)  | 10 / 10                                                              |
| Median subtasks per pilot           | **5**                                                                |
| Total annotation time               | **492 min** (~8.2 h)                                                 |
| Median annotation time per pilot    | **21 min** (range 12–55, mean 24.6)                                  |
| Median final overall progress       | **0.92** (across all 20)                                             |
| Median final coding-progress, success traces | **1.00**                                                    |
| Median final coding-progress, failure traces | **0.69**                                                    |
| Pilots with at least one BLOCKED leaf | **6 / 20**                                                         |
| Pilots with at least one REOPEN event | **3 / 20**                                                         |
| Pilots reporting `whether_schema_gap_found` | **2 / 20** (`f_02`, `f_07` — protocol refinements landed) |

## 2. Per-pilot table

| pilot                          | upstream | steps | leaves | blocked | reopens | overall | coding | annot. min |
|--------------------------------|----------|-------|--------|---------|---------|---------|--------|------------|
| swe_agent_pilot_s_01           | True     |  43   |  6     |  0      |  0      | 1.00    | 1.00   | 35         |
| swe_agent_pilot_s_02           | True     |  27   |  5     |  0      |  0      | 1.00    | 1.00   | 18         |
| swe_agent_pilot_s_03           | True     |  37   |  6     |  0      |  1      | 1.00    | 1.00   | 22         |
| swe_agent_pilot_s_04           | True     |  17   |  4     |  0      |  0      | 0.75    | 0.67   | 12         |
| swe_agent_pilot_s_05           | True     |  33   |  5     |  0      |  1      | 1.00    | 1.00   | 18         |
| swe_agent_pilot_s_06           | True     |  29   |  5     |  0      |  0      | 1.00    | 1.00   | 16         |
| swe_agent_pilot_s_07           | True     |  23   |  5     |  0      |  0      | 1.00    | 1.00   | 18         |
| swe_agent_pilot_s_08           | True     |  29   |  4     |  0      |  0      | 1.00    | 1.00   | 20         |
| swe_agent_pilot_s_09           | True     |  19   |  4     |  0      |  0      | 1.00    | 1.00   | 15         |
| swe_agent_pilot_s_10           | True     |  23   |  5     |  0      |  0      | 1.00    | 1.00   | 18         |
| swe_agent_pilot_f_01           | False    |  17   |  4     |  0      |  0      | 0.75    | 0.67   | 20         |
| swe_agent_pilot_f_02           | False    | 509   |  2     |  1      |  0      | 0.50    | 0.50   | 35         |
| swe_agent_pilot_f_03           | False    | 113   |  2     |  1      |  0      | 0.50    | 0.50   | 40         |
| swe_agent_pilot_f_04           | False    |  19   |  4     |  0      |  0      | 0.75    | 0.67   | 12         |
| swe_agent_pilot_f_05           | False    |  35   |  5     |  2      |  0      | 0.60    | 0.60   | 28         |
| swe_agent_pilot_f_06           | False    |  33   |  5     |  0      |  0      | 1.00    | 1.00   | 22         |
| swe_agent_pilot_f_07           | False    | 183   |  3     |  1      |  0      | 0.67    | 0.67   | 55         |
| swe_agent_pilot_f_08           | False    |  77   |  7     |  1      |  0      | 0.71    | 0.71   | 35         |
| swe_agent_pilot_f_09           | False    |  41   |  6     |  0      |  1      | 0.83    | 0.80   | 28         |
| swe_agent_pilot_f_10           | False    |  81   |  3     |  1      |  0      | 0.67    | 0.67   | 25         |

## 3. Notable shapes

### High-progress failure (the protocol's predicted shape)

- **`swe_agent_pilot_f_06` at coding-progress 1.00**, `final_success=False`. Every subtask the agent surfaced reached `complete` with in-trace evidence. The failure sits entirely in undiscovered hidden work: the agent's `reproduce.py` returned "Script completed successfully, no errors." (step 7) — i.e. the repro never actually triggered the bug — and the agent moved on without noticing. This is the canonical "all discovered work done; failure in undiscovered hidden work" case from general § 3 of the protocol, recorded in `runs/swe_agent_pilot/swe_agent_pilot_f_06/run_notes.md` § 6 as a hidden-work gap.

### Low-progress success (the "validated by chance" shape)

- **`swe_agent_pilot_s_04` at coding-progress 0.67**, `final_success=True`. The agent located, edited, and submitted in 17 steps without running any in-trace validation. Identical ledger shape to `f_01` and `f_04` (both `False`). The progress signal correctly says "validation never started" regardless of upstream label.

### Non-monotonic runs (REOPEN events)

Three runs surface real non-monotonicity:

- **`s_03`**: REOPEN at step 22 because step 21's repro re-run still emitted Traceback after the first `__init__.py` edit. Second `__init__.py` edit + a teambuilder.py edit fixed it.
- **`s_05`**: REOPEN at step 26 because step 25's repro output was unchanged — the first `core.py:366` edit didn't recurse the encoder.
- **`f_09`**: REOPEN at step 38 on the VALIDATION leaf because the agent issued one final test-file edit *after* the last in-trace pytest, leaving the submitted state unvalidated.

In each case the progress curve dips when the REOPEN fires, exactly as protocol § 4 anticipates. None were smoothed out.

### Stuck-loop variants (BLOCKED leaves)

Six runs hit at least one BLOCKED leaf via the § 6 stuck-loop rules:

- **`f_03`**: 4-command cycle (`search_file 'configparser'` / `goto 156` / `search_file 'test.ini'` / "no matches"), iter 3 begins step 22, pattern still active at step 112.
- **`f_07`**: cycle-of-1 (`edit 1:20` ×5) at step 40 escalating into a 2-command oscillation (`edit 5:5` / `edit 21:21` ×~55) through step 182.
- **`f_02`**: tool-response-loop variant. The agent issues `find_file` with ~250 different keywords ("ast", "parse", "amalgamated", "harmonized", "spirit", "autonomy", ...) all returning identical "No matches found"; iter 3 of the response-pattern at step 17.
- **`f_05`**: 2-command oscillation `edit 77:NN` / `pytest` / `edit 77:NN+3` from step 22 with intermittent syntax-error rejections.
- **`f_08`**: `fields.py` scroll-only loop (4 consecutive `scroll_down` then mixed `scroll_up`/`scroll_down`) with no edit, iter 3 at step 64.
- **`f_10`**: `edit 17:32` repeated ~30+ times, all rejected with the same syntax-error message.

## 4. Schema gaps and protocol refinements forced by the pilot

Two pilots reported `whether_schema_gap_found=True`. Both surfaced real protocol refinements that landed before annotating the affected pilot:

1. **`f_07` (183 steps)** forced the **cycle-of-1 / cycle-of-2 expansion** of the stuck-loop rule. The original rule said "same sequence of N≥3 commands verbatim", which was ambiguous on cycle length. f_07's `edit 1:20` ×5 (single-command repetition) and `edit 5:5` ↔ `edit 21:21` ×~55 (two-command oscillation) didn't trigger the literal rule. Refined to "any cycle length ≥ 1; mark blocked at the earliest step where any pattern hits its third iteration" (general § 6).

2. **`f_02` (509 steps)** forced the **tool-response-loop variant**. The agent issued ~250 different `find_file` keywords; every response was identical "No matches found". The command-loop rule didn't trigger because the commands varied. Added § 6(b): "when the agent's tool responses are identical for ≥3 consecutive steps regardless of query variation, mark `blocked` at the third such response."

Two further refinements landed earlier in the pilot, after `f_03`'s 113-step walk: SWE-agent addendum pitfalls **#6** (harness-forced termination is not an agent submit) and **#7** (`final_diff.patch` is a state diff, not an action diff — investigation residue may appear there).

One framework-naming gap also landed: C3's `eval_output.txt` was renamed to the framework-standard `test_output.txt` (sourced from upstream `eval_logs`), establishing the "importer maps `<upstream-field-name>` → `<framework-artifact-name>`" rule.

**Were schema changes needed?** Yes — three real protocol refinements (§ 6 stuck-loop generalization to any cycle length; § 6(b) tool-response-loop variant; addendum pitfalls #6 and #7) and one framework-artifact-naming change (`test_output.txt`). The remaining 18 of 20 pilots fit the post-refinement protocol cleanly without further changes.

## 5. Common evidence gaps

19 evidence-gap citations across 20 pilots, concentrated in three patterns:

1. **Submitted without in-trace validation** — `f_01`, `f_04`, `s_04`. Agent edits, submits, never runs tests. Validation leaf left at `not_started`. The post-hoc `test_output.txt` is not used as evidence per general § 4.4.
2. **Hidden-work gap surfaced by the trace but not acted on** — `f_01` (test mock at `tests/core/functions_test.py` named by step-7 grep, never opened), `f_06` (repro at step 7 returned success but the agent didn't notice), `f_02` (agent never tried `ls /pyupgrade` despite hundreds of `find_file` failures).
3. **Mid-edit harness-forced termination** — `f_02`, `f_03`, `f_05`, `f_07`, `f_08`, `f_10`. `final_diff.patch` reflects a state the agent didn't endorse; no in-trace validation; no agent-issued submit.

## 6. Category distribution across all leaves

| Category        | Total leaves |
|-----------------|--------------|
| INVESTIGATION   | 34           |
| PRODUCT         | 24           |
| VALIDATION      | 18           |
| ARTIFACT        | 14           |
| ENVIRONMENT     | 0            |
| DOCUMENTATION   | 0            |

INVESTIGATION dominates because most pilots had multiple investigation steps (locate, reproduce, search) before product changes. The absence of ENVIRONMENT and DOCUMENTATION reflects the pilot's bug-fix-style task type — neither dependency setup nor doc updates were demanded by any of the 20 issues.

**Status totals across all 90 leaves:**
- complete: 78
- blocked: 7
- in_progress: 2
- not_started: 3
- invalidated: 0

## 7. Annotation uncertainty distribution

- 6 of 20 pilots flagged at least one "uncertain decision" in `run_notes.md` § 4.
- 7 uncertain events total across the pilot.

Uncertain decisions, by type:

- **Test-edits-as-PRODUCT vs silence-the-failure** (s_01, f_05, f_09). Resolved by reading the issue text in each case. s_01's call was explicitly user-validated; that decision is now memorized at `feedback_test_edit_classification.md`. f_05 and f_09 remain on the "default to PRODUCT but flag" side.
- **Earliest stuck-pattern threshold** (f_03 step 22, f_07 step 40, f_10 step 28). When multiple loop patterns nest, picking the earliest "third iteration" is the locked-in rule, but the call is judgment-shaped at the boundary.
- **Whether to model intermediate validations as one leaf or many** (f_08). Chose one leaf with multiple evidence cites.
- **`reproduce.py` cleanup as ARTIFACT vs hygiene** (s_09). Treated as ARTIFACT evidence (cleaning the patch before submit is artifact-shaping work).

## 8. Honest qualitative judgment — observation or narrative?

**~85% observation, ~15% narrative-reconstruction risk.**

Most of the pilot felt strongly observation-shaped:

- The stuck-loop pilots (`f_02`, `f_03`, `f_05`, `f_07`, `f_10`) almost annotated themselves once the refined § 6 rules were in place. The pattern was visible; the call fell out.
- The reopen runs (`s_03`, `s_05`, `f_09`) had clean trace evidence for the dip — a failed re-run, an out-of-order edit. The non-monotonicity wasn't manufactured.
- The clean successes (`s_02`, `s_06`, `s_07`, `s_08`, `s_09`, `s_10`) were nearly mechanical: locate, edit, validate, submit, evidence cited at each step.

The 15% narrative-reconstruction risk concentrates in two judgment-shaped places:

1. **Test-edits-as-PRODUCT calls** (`f_05`, `f_09`). The issue text didn't explicitly justify the test edits; the trace alone is ambiguous between "patching to align with the issue's API change" (PRODUCT) and "silencing the failure" (anti-pattern). I defaulted to PRODUCT with prominent uncertain-decision notes, but a second annotator could plausibly call these the other way.

2. **Validation-as-implicit-discovered-work** (`f_01`, `s_04`, `f_04`). The protocol's strict reading is "discovered work requires the trace to surface it"; for these three the trace doesn't. I added the validation leaf anyway because the framework's purpose (surfacing skipped-validation as a process anomaly) requires it. This is a real protocol gap, captured in memory and flagged for Workstream H rather than changed without a second annotator.

**Was `final_success` ever consulted as evidence during a walk?** No. The upstream label was used twice in run_notes — once in `f_06` to corroborate the hidden-work-gap interpretation, once in `f_05` to explain why an over-broad +12-line edit might have failed. In neither case did it shape a ledger event. Per the locked-in `feedback_progress_vs_outcome_decoupling.md` rule.

## 9. What this summary tells us about the framework

1. **The progress signal discriminates between agent failure modes.** Failures span 0.50 → 1.00 in interpretable ways that map to identifiable behaviors (stuck loops at low end, validation gaps at mid, hidden-work misses at top). The signal is genuinely independent of `final_success`: `f_06` is a failure at 1.00, `s_04` is a success at 0.67, and the ledger shape says exactly why in each case.

2. **The protocol survived four trace-length stress points** (43, 113, 183, 509 steps) with three real refinements. None of the refinements broke single-task semantics; all are additive.

3. **Single-annotator caveat is real.** All 20 annotations are mine. The framework's claim "two annotators converge" is unproven empirically. Workstream H should walk at least 2 pilots independently — `f_06` (high-progress failure shape) and `s_03` or `f_09` (REOPEN runs) are the highest-signal targets.

4. **Sources beyond SWE-agent are untested.** The "general protocol + thin source addendum" claim is asserted, not verified. Workstream P (added during E1) plans the cross-source test.

5. **Set-of-tasks is unimplemented.** The progress signal exists per single trace; aggregating across a project / set of tasks is Workstream T (sketch landed during E1; implementation deferred until after this pilot's downstream consumption).

## 10. Pointers

- Annotation specs: `annotations/swe_agent_pilot/swe_agent_pilot_*.json`
- Annotation prose: `annotations/swe_agent_pilot/swe_agent_pilot_*.notes.md`
- Materialized ledgers: `runs/swe_agent_pilot/swe_agent_pilot_*/ledger.jsonl` (gitignored)
- Driver: `scripts/annotate_pilots_from_spec.py`
- Observation dataset: `datasets/swe_agent_pilot_zero/observations*.csv`
- Audit: `datasets/swe_agent_pilot_zero/audit.{md,json}`
- Memory of locked-in judgment calls: `~/.claude/projects/.../memory/feedback_*.md` (paths in `MEMORY.md`)
