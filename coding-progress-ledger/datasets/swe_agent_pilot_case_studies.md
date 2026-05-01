# SWE-agent pilot case studies (G3)

This document satisfies `TASKS.md` § Workstream G, task **G3**. Four
case studies, one per archetype:

1. Successful high-progress normal run — `s_01`
2. Successful non-monotonic run — `s_03`
3. Failed high-progress run — `f_06`
4. Failed low-progress / stuck run — `f_03`

Every claim cites a ledger event or trace step. Annotation files are
under `annotations/swe_agent_pilot/`; materialized run dirs (gitignored)
are under `runs/swe_agent_pilot/<pilot_id>/`.

---

## Case 1 — Successful high-progress normal run: `swe_agent_pilot_s_01`

- Instance: `Melevir__cognitive_complexity-15`
- Trajectory length: 43 steps
- Upstream `final_success`: `True`
- Final coding-progress: **1.00**
- Ledger leaves: 6 (all complete)

**Task summary.** Cognitive Complexity calculator over-counts a
multi-line `if` containing a sequence of binary logical operators
(should be +2 not +4). Fix is in
`cognitive_complexity/utils/ast.py`; the issue text also explicitly
asks to update the existing `test_real_function`'s expected value.

**Final outcome.** Submitted at step 42; upstream eval reports
passing.

**Progress curve summary.** Saw-tooth, monotonically rising:
`add → +1.0` at step 23 (S1 complete), drop to 0.5 at step 24
(S2 added), back to 1.0 at step 25 (S2 complete), drop and recover
again at steps 26-27 (S3 validation), 28-39 (S4 fixture update),
40-41 (S5 validation), 42 (S6 submit). Final 1.00.

**Key ledger events.**

- Step 23: `S1` (INVESTIGATION) `complete` — agent has
  `process_node_itself` in view at `utils/ast.py:88`.
- Step 25: `S2` (PRODUCT) `complete` — `edit 88:88` ack'd.
- Step 27: `S3` (VALIDATION) `complete` — pytest run produces
  output that triggers the test-fixture-update branch (cited by
  the next agent action).
- Step 39: `S4` (PRODUCT) `complete` — fixture edits at lines
  125 and 147 land after one rejected syntax-error attempt at
  step 28.
- Step 41: `S5` (VALIDATION) `complete` — pytest re-run.
- Step 42: `S6` (ARTIFACT) `complete` — submit issued.

**Evidence gaps.** None.

**Why it matters.** This is the framework's reference shape: every
discovered subtask is completed, every `complete` event has at
least one in-trace evidence citation, and the saw-tooth pattern
(visible drops when new leaves are added) is preserved through to
1.00. The annotator-uncertainty around test-edit-as-PRODUCT (the
fixture update at step 28-39) was resolved from the issue text;
the call was reviewed and confirmed. This case is what every other
pilot is measured against.

---

## Case 2 — Successful non-monotonic run: `swe_agent_pilot_s_03`

- Instance: `hsahovic__poke-env-68`
- Trajectory length: 37 steps
- Upstream `final_success`: `True`
- Final coding-progress: **1.00**
- Ledger leaves: 6 (all complete; one `S3` `REOPEN` event at step 22)

**Task summary.** `ConstantTeambuilder(team)` raises
`UnboundLocalError` when a showdown-format team description has no
`item:` lines. The fix turns out to span two files
(`player/__init__.py` and `teambuilder/teambuilder.py`).

**Final outcome.** Submitted at step 36; upstream eval reports
passing.

**Progress curve summary.** Two visible dips, both real:

- Dip at step 22 (REOPEN of `S3`): the agent's first
  `__init__.py:11-18` edit at step 18 is marked `complete` at
  step 19. At step 21, the repro re-run still raises Traceback —
  so `S3` is reopened with reason "step 21 repro still emits
  Traceback; first __init__.py edit insufficient". A second edit
  at step 27 (`__init__.py:10`) lands.
- Subsequent investigation steps 28-29 surface that another file
  needs editing (`teambuilder.py`); add `S4` PRODUCT, complete at
  step 33; validation `S5` runs at step 35 and is silent
  (success); submit at step 36.

The progress curve dips when `S3` is reopened (canonical
non-monotonic event per general protocol § 7).

**Key ledger events.**

- Step 9: `S1` (INVESTIGATION) `complete` — repro emits Traceback,
  bug confirmed.
- Step 19: `S3` (PRODUCT) `complete` — first `__init__.py` edit.
- Step 22: `S3` `reopen` — reason cites step 21's repro.
- Step 27: `S3` `complete` (after reopen) — second
  `__init__.py:10` edit.
- Step 33: `S4` (PRODUCT) `complete` — `teambuilder.py:91`
  edit (+5 lines).
- Step 35: `S5` (VALIDATION) `complete` — repro silent.
- Step 36: `S6` (ARTIFACT) `complete` — submit.

**Evidence gaps.** None. The dataset audit's "large
native/resolved divergence" warning concerns this run's coding-
progress around the step-22 REOPEN — a downstream-audit signal
worth investigating but not an integrity failure.

**Why it matters.** Real traces produce reopens because real fixes
sometimes need extending. The framework preserves the dip rather
than smoothing it. A modeler that only sees toy/live "monotonic
success" data wouldn't learn that `progress went down then back up`
is a valid in-progress shape.

---

## Case 3 — Failed high-progress run: `swe_agent_pilot_f_06`

- Instance: `googleapis__python-spanner-317`
- Trajectory length: 33 steps
- Upstream `final_success`: **`False`**
- Final coding-progress: **1.00**
- Ledger leaves: 5 (all complete)

**Task summary.** Spanner client doesn't map Python `decimal.Decimal`
to NUMERIC fields; the fix should add the mapping in
`google/cloud/spanner_dbapi/parse_utils.py`.

**Final outcome.** Submitted at step 32; upstream eval reports
**failure**. The agent's submitted patch does not resolve the
issue.

**Progress curve summary.** Clean saw-tooth rising to 1.00.
Indistinguishable in shape from a successful trace.

**Key ledger events.**

- Step 7: `S1` (INVESTIGATION) `complete` — `python reproduce.py`
  outputs **"Script completed successfully, no errors."**
  Recorded as evidence that the repro ran; **but** for a runtime-
  failure issue, this output is *prima facie* inconsistent with
  reproducing the bug. The annotator records this as a hidden-work
  signal in `run_notes.md` § 6 but does **not** retro-fit a
  discovered subtask for it (per general protocol § 2).
- Step 25: `S2` (INVESTIGATION) `complete` — multi-file
  localization to the `parse_utils.py` helper.
- Step 27: `S3` (PRODUCT) `complete` — `edit 526:528` ack'd.
- Step 29: `S4` (VALIDATION) `complete` — repro re-run emits the
  same uninformative "Script completed successfully" output as
  step 7. The agent infers success.
- Step 32: `S5` (ARTIFACT) `complete` — submit issued.

**Evidence gaps.**

- **The repro never actually triggered the bug** (steps 7, 29).
  An honest observer reading the trace can name this — "Script
  completed successfully" should not happen on a TypeError-style
  mapping bug — but the agent did not surface it as discovered
  work.
- The trace contains every shape of evidence one would want for a
  successful run, **except** the repro actually demonstrating the
  bug. The framework's ledger correctly says "all discovered work
  done"; the failure sits entirely in undiscovered hidden work.

**Why it matters.** This is the protocol's load-bearing
counter-example: a 1.00-progress failure that exists naturally in
real traces. If the framework forced this to read as `progress<1.00`,
it would be re-encoding `final_success`. Instead the ledger
preserves the shape and run_notes records the gap. A predictor
consuming progress alone cannot distinguish `f_06` from a clean
success — and that's the right answer: from a *process* standpoint,
the agent did everything they were supposed to do. The failure
mode is "the repro was wrong", which is upstream of the work the
ledger tracks.

The smoke predictor `progress_only` assigns `f_06` probability
0.45 — its highest probability for any trace in the dataset —
exactly because progress = 1.00 looks like every other 1.00-progress
run. That is the correct behavior, not a bug.

---

## Case 4 — Failed low-progress / stuck run: `swe_agent_pilot_f_03`

- Instance: `asottile__setup-cfg-fmt-132`
- Trajectory length: 113 steps
- Upstream `final_success`: `False`
- Final coding-progress: **0.50**
- Ledger leaves: 2 (1 complete, 1 BLOCKED)

**Task summary.** `configparser` downcases keys in unrelated
sections; `DJANGO_SETTINGS_MODULE` becomes lowercased even when
inside `[tool:pytest]`. Fix lives in `setup_cfg_fmt.py`.

**Final outcome.** Harness-forced termination at step 113
(`exit_status='submitted (exit_context)'` — context exhaustion).
Agent's last in-trace command is `goto 156`, **not** `submit`.

**Progress curve summary.** Reaches 1.0 at step 7 (S1 reproduce
complete), drops to 0.5 at step 10 (S2 added), stays at 0.5
through step 22 when `S2` is `BLOCKED`. The remaining ~90 steps
of stuck-loop activity do not move the curve.

**Key ledger events.**

- Step 7: `S1` (INVESTIGATION) `complete` — `python reproduce.py`
  succeeded in setting up the test scenario; agent moves on.
- Step 10: `S2` (INVESTIGATION) added — locate configparser
  configuration in `setup_cfg_fmt.py`.
- Steps 18-21: agent has `setup_cfg_fmt.py:156` in view; this is
  the right region of the right file.
- Steps 22-112: **stuck loop**. The agent issues the same
  4-command cycle 24× verbatim:
  `search_file 'configparser'` → `goto 156` →
  `search_file 'test.ini'` → `'No matches found'`.
  No query variation; no edits; no recovery.
- Step 22: `S2` `block` — third iteration of the cycle begins per
  general protocol § 6 stuck-loop rule. Cited reason mentions
  the cycle and the latest visible repetition (step 112).
- No PRODUCT, VALIDATION, or ARTIFACT leaf added — the stuck
  state never produces discovered work in those categories.

**Evidence gaps.**

- The agent never tried any non-`search_file` strategy (no `ls`,
  no `find`, no `grep -r` outside the initial steps), and never
  pivoted the query to look for `configparser` configuration vs
  the `test.ini` artifact. The agent's mental model is wrong but
  doesn't get corrected.
- `final_diff.patch` (560 chars) reflects only `reproduce.py`
  (created in steps 4-5) — investigation residue per SWE-agent
  addendum § 5 pitfall #7. Not PRODUCT evidence.

**Why it matters.** This is the protocol's most important
discrimination from the previous case: `f_06` ends at 1.00
because the agent did everything they could see; `f_03` ends at
0.50 because the agent visibly got stuck and never even reached a
PRODUCT leaf. Both are upstream `False`, but the ledger shapes
are radically different. A 1.00-progress failure (`f_06`) has the
agent doing all surface-visible work, whereas a 0.50-progress
failure (`f_03`) has the agent visibly stuck.

This case also forced the protocol's "earliest-pattern-wins"
clarification (general § 6) and the SWE-agent addendum's pitfall
#6 (harness-forced termination is not an agent submit) and
pitfall #7 (`final_diff.patch` is a state diff, not an action
diff).

---

## Cross-reference summary

| Case | Pilot | Upstream | Progress | Shape archetype                                          |
|------|-------|----------|----------|----------------------------------------------------------|
| 1    | s_01  | True     | 1.00     | Clean success — full INV→PRODUCT→VAL→ARTIFACT pipeline  |
| 2    | s_03  | True     | 1.00     | Non-monotonic success — REOPEN forced by failed re-run  |
| 3    | f_06  | False    | 1.00     | High-progress failure — hidden-work gap in repro        |
| 4    | f_03  | False    | 0.50     | Stuck-loop failure — INVESTIGATION blocked, no PRODUCT  |

These four cases together populate all four success/progress
quadrants. They are the empirical anchors for the framework's
claim that progress is decoupled from outcome — the load-bearing
test (`tests/test_swe_agent_pilot_annotations.py`) verifies that
at least one of each shape exists in the corpus.
