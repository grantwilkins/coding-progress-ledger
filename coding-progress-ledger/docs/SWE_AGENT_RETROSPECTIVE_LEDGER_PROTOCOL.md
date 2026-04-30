# SWE-agent retrospective ledger annotation protocol (D1)

This document constrains how a human (or a future automated annotator
following the same rules) converts one normalized SWE-agent trace into
a `ledger.jsonl` for the retrospective pilot. It satisfies `TASKS.md`
§ Workstream D, task **D1**. It is binding for D4 (pilot-zero, N=2)
and E1 (full N=20). It is NOT a planning document, NOT a controller
spec, and NOT an annotator-friendliness pass — D2 owns the template
and D3 owns optional helpers.

All § 0 project rules apply. The most load-bearing of those for
annotation are restated here:

> **Annotate only visible trace evidence.**
> **Do not use `final_success` to decide intermediate completion.**
> **Do not force monotonicity.**

Ground truth for the type system is the in-repo source, not this doc:

- `ledger_progress/core.py:Status` — six statuses
- `ledger_progress/core.py:EventType` — eight event types
- `ledger_progress/core.py:SubtaskCategory` — six categories
- `ledger_progress/queries.py:CODING_CATEGORIES` — the
  `(PRODUCT, VALIDATION, INVESTIGATION)` triple used to compute the
  "coding" sub-progress curve

If this protocol drifts from those files, **the files win.**

## 1. What you are doing

You are reading a normalized trace (the `events` list in
`<run_dir>/normalized_trace.json`) and writing a sequence of ledger
events that describe *what work an outside observer can see the agent
discovering and finishing*, step by step. You are not retelling the
story. You are not inferring intent. You are not predicting success.

The output is one `ledger.jsonl` per run, replayable by
`ledger_progress.replay()`, plus an extended `run_notes.md` that
records evidence citations and any places you were unsure.

## 2. Discovered work vs hidden work

This distinction is the single most important idea in the protocol.

- **Discovered work** = a subtask the trace shows the agent (or, by
  retrospective annotation, the observer) can name. If the agent never
  navigates to a file, the subtask "modify that file" is not yet
  discovered work.
- **Hidden (true) work** = the union of all subtasks that *would*
  resolve the issue. Some hidden work is never discovered in a given
  trace; that gap is exactly what makes failure failure.

The ledger tracks **discovered work only.** This is by design. The
score `complete_weight / active_weight` is a lower bound on hidden
progress, not an estimate of it. A ledger that reaches 1.0 means
*every subtask the agent surfaced was completed*, not that the
underlying issue was solved.

You may add subtasks the agent itself never named, **iff** the trace
makes their existence visible to an honest observer (e.g. a stack
trace explicitly names a missing function the agent has not yet
opened). When you do this, cite the step index and quote the line in
`run_notes.md`. When you cannot, leave the work hidden — that absence
is a real datum.

## 3. Progress is not success probability

`score(ledger).progress` is `complete_weight / active_weight` over
non-INVALIDATED, non-DELETED leaves. It rises when discovered work is
finished and stays the same (or falls, when subtasks are split or
reopened) when nothing visible was finished. It has no Bayesian
interpretation.

In particular:

- A **failed** trace can legitimately end at progress = 1.0 if every
  subtask the observer discovered was finished — the failure can sit
  entirely in the *undiscovered* hidden work.
- A **successful** trace can legitimately end at progress < 1.0 if
  artifact / documentation leaves remain unchecked.
- Two traces that end at the same progress can have wildly different
  `final_success`. That is a *feature* of an observation channel, not
  a bug — Workstream P/Q investigate the relationship; D does not
  compress it.

If you find yourself adjusting events so that progress ≈ "how confident
am I the agent succeeded", **stop and re-read § 2.**

## 4. Hard rules

- **Annotate only visible trace evidence.** Every `complete` event
  must cite either (a) a step index range in the normalized trace,
  (b) a line in `final_diff.patch`, or (c) a line in `eval_output.txt`.
  No external knowledge of the repo, the issue tracker, or how the
  task "should" be done.
- **Do not use `final_success` to decide intermediate completion.**
  `final_success` is a stratification label; per § 0 of `TASKS.md` it
  is never a feature. You may consult `eval_output.txt` only as
  evidence for a *validation* subtask the agent itself surfaced (e.g.
  the agent ran `pytest`); you may not use the upstream label to
  retroactively complete or reopen subtasks.
- **Do not force monotonicity.** If the agent edits a file, then later
  reopens that work because the patch was wrong, emit a
  `REOPEN_SUBTASK` event. Progress will dip; that is the correct
  shape.
- **Use the final eval only as final-validation evidence.** If the
  agent ran tests in-trace, cite the relevant tool turn. If the eval
  log only exists post-hoc (the agent submitted without running
  anything), then validation was *not done* by the agent, no matter
  what the eval log says about the patch. Record this in
  `run_notes.md` and leave validation in `not_started` or `in_progress`.
- **Preserve uncertainty.** When you are not sure whether to mark
  something complete, don't mark it complete. Note the ambiguity in
  `run_notes.md` under "Uncertain decisions". One annotator's
  uncertainty is the next annotator's evidence gap.

## 5. Categories — what counts as what for SWE-agent

The six categories live in `ledger_progress/core.py:SubtaskCategory`.
For SWE-agent traces specifically:

| Category        | Use for                                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------------------|
| `INVESTIGATION` | Reading the issue; `find_file`, `search_dir`, `search_file`, `grep`, `ls`, `open`, `goto` — anything that locates code or surfaces facts. |
| `PRODUCT`       | `edit`, `create` of source files; behavioral changes to implementation **or to tests required by the task** (e.g. fixing an explicitly broken test fixture cited in the issue). |
| `VALIDATION`    | `pytest`, `tox`, ad-hoc reproduction scripts (`python repro.py`), running the bug-repro from the issue, interpreting failures, reading the post-edit eval log when the agent itself opens it. |
| `ENVIRONMENT`   | `pip install`, virtualenv setup, fixing missing modules, `apt`, container-shape errors that block product work without being part of it. Rare in this corpus; the SWE-agent harness pre-installs. |
| `ARTIFACT`      | The `submit` action; producing a final patch dump; any explicit "write this to file as the answer". |
| `DOCUMENTATION` | When (and only when) the issue text asks for a doc change, or the agent updates docs as part of the fix. Default to PRODUCT for code+docstring edits. |

If you genuinely cannot decide, default to `PRODUCT` and note the
ambiguity. Do not invent new categories.

## 6. Statuses — when to transition

Canonical names live in `ledger_progress/core.py:Status`.

- `not_started` — discovered, no observable action yet. Use for
  subtasks added by the annotator from issue evidence the agent has
  not yet acted on.
- `in_progress` — partial action visible in the trace (`edit` issued,
  but the patch is not yet shown to be the right one).
- `blocked` — agent is stalled on something it cannot resolve from the
  visible context (e.g. a tool returns "directory not found" and the
  next several turns are unproductive variants of the same query).
  Pair `blocked` with a `reason` payload citing the step index.
- `complete` — supported by concrete trace evidence (see § 4). REQUIRED:
  at least one evidence string in the event payload.
- `invalidated` — the subtask is no longer load-bearing (e.g. the
  agent split a vague subtask, the parent disappears from the leaf
  set). Prefer `invalidate` over `delete`; history matters.
- `deleted` — almost never. Reserve for events that should have been
  rejected at write time; in retrospective annotation this is rare
  enough that you should ask in `run_notes.md` before using it.

## 7. Event types — when to use which

Canonical list lives in `ledger_progress/core.py:EventType`.

- `INIT` — emitted automatically by `new_ledger(...)` /
  `LedgerSession(...)`. Do not write by hand.
- `ADD_SUBTASK` — when discovered work first becomes nameable.
- `UPDATE_STATUS` — every status transition except `INIT` and the
  three event-typed transitions below.
- `ADD_EVIDENCE` — when new evidence accumulates on an already-known
  subtask without changing its status (e.g. a second confirmation from
  a later test run).
- `SPLIT_SUBTASK` — when a vague subtask becomes several leaf
  subtasks. The parent is invalidated automatically by the replay
  engine; you do not also issue an `INVALIDATE_SUBTASK`.
- `REOPEN_SUBTASK` — when previously-complete work is shown
  incomplete. **This is the canonical non-monotonic event** — use it
  whenever the trace contradicts an earlier completion, even if it
  makes the progress curve drop. That drop is the point.
- `INVALIDATE_SUBTASK` — when a subtask remains in history but should
  no longer count as active work (e.g. an approach the agent
  abandoned). Prefer this over `DELETE_SUBTASK`.
- `DELETE_SUBTASK` — see "almost never" in § 6.

## 8. Step numbering and citation

Each ledger event carries an integer `step`. For retrospective
annotation, **set `step` to the normalized-trace `step_index` of the
turn that justifies the event**. This makes evidence traceable: a
reader can open `normalized_trace.json`, jump to that step, and see
why you wrote the event.

Two consequences:

- Multiple ledger events can share a `step` (e.g. one tool turn
  surfaces evidence for two subtasks). That is fine.
- `step` need not be dense. Long stretches of trace with no annotated
  events leave gaps in the `step` axis. That is also fine.

Evidence strings in `ADD_EVIDENCE` / `UPDATE_STATUS` payloads SHOULD
include the cited step index (e.g. `"step 24: edit utils/ast.py:88
inserted parens fix"`). One short sentence per evidence string. No
multi-paragraph quotes.

## 9. Procedure (per run)

Recommended order, optimized for not lying to yourself:

1. Read `task.md` once. Note the issue's stated goal in
   `run_notes.md` under "Initial reading".
2. Skim `trajectory_summary.md` end-to-end before opening
   `normalized_trace.json`. Build a rough mental model of the run's
   shape (investigation-heavy? straight to edit? lots of failed
   commands?).
3. Open `normalized_trace.json`. Walk it linearly, top to bottom. Do
   not read `final_diff.patch` or `eval_output.txt` yet.
4. Use `LedgerSession`. For each step where new discovered work
   appears, emit `add(...)`. For each step that finishes work, emit
   `complete(...)` with the cited step index.
5. After the trace ends, read `final_diff.patch` and
   `eval_output.txt`. Use them only to (a) confirm a `validation`
   subtask the agent itself surfaced, (b) note in `run_notes.md` if
   the patch shows work the trace did not surface (a hidden-work gap).
6. Export with `session.export_jsonl(...)`. Run
   `ledger-run check-run <run_dir>` and address any missing
   artifacts. `progress.csv` and `summary_by_category.json` come from
   `ledger-run export-run`; do not write them by hand.
7. Fill in the rest of `run_notes.md` per the D2 template. Record
   uncertain decisions, evidence gaps, and whether you ever felt
   tempted to use `final_success` (you didn't, but write it down if
   the temptation surfaced).

## 10. Worked examples (good × 2, bad × 2)

The two examples below are real pilot runs from the cache. Both
trajectories are short enough to walk end-to-end here. The good and
bad annotations are deliberately of the *same* runs so the contrast
is clean.

### Example A — `swe_agent_pilot_s_01` / `Melevir__cognitive_complexity-15`

Shape (43 steps, success): the agent reads the issue (binary-logical-
operator counting bug), localizes `cognitive_complexity/utils/ast.py`
with several `find_file` / `search_dir` / `open` / `goto` calls, edits
line 88, runs `pytest`, and submits. Final eval passes.

#### A-good (right way)

```python
session = LedgerSession("Fix incorrect counting for binary logical operators")
inv = session.add(
    "Localize file controlling complexity increments for boolean ops",
    step=2, category=SubtaskCategory.INVESTIGATION,
)
session.complete(
    inv, "step 23: open cognitive_complexity/utils/ast.py shows process_node_itself", step=23,
)
prod = session.add(
    "Edit utils/ast.py to drop the nesting increment for B-op sequences",
    step=24, category=SubtaskCategory.PRODUCT,
)
session.complete(prod, "step 24: edit 88:88 inserted the +2 fix", step=24)
val = session.add(
    "Run pytest and confirm test_real_function expectation",
    step=26, category=SubtaskCategory.VALIDATION,
)
session.complete(val, "step ~30: pytest output shows test_real_function passes", step=30)
art = session.add("Submit final patch", step=42, category=SubtaskCategory.ARTIFACT)
session.complete(art, "step 42: submit issued", step=42)
```

Why this is right:

- Each leaf cites a concrete step index. Evidence is short, traceable.
- Validation is marked complete because the agent itself ran `pytest`
  in-trace — `eval_output.txt` is corroborating evidence, not the
  primary justification.
- The investigation leaf collapses ~10 navigation steps into one
  outcome ("found the right file"); we do not invent ten subtasks.
  The trace is the witness, the leaf is the unit of *discovered work*.

#### A-bad (wrong way) — DO NOT DO THIS

```python
session = LedgerSession("Fix incorrect counting for binary logical operators")
for desc in [
    "Understand issue", "Find the right file", "Edit the file",
    "Run tests", "Submit",
]:
    s = session.add(desc, step=1, category=SubtaskCategory.PRODUCT)
    session.complete(s, "final_success=True", step=1)
```

What's wrong:

- Every leaf marked complete at step 1: violates "annotate visible
  trace evidence" — none of the work is visible at step 1.
- Evidence string `"final_success=True"`: violates the hard rule
  against using the upstream label as completion evidence.
- All five leaves filed under `PRODUCT`: the category set is not
  decorative; this destroys the `(INVESTIGATION, PRODUCT, VALIDATION)`
  decomposition that downstream queries depend on.
- Front-loading produces a flat progress curve from step 1 onward
  with no shape — the observation channel reduces to "agent
  succeeded", which is just `final_success` re-encoded.

### Example B — `swe_agent_pilot_f_01` / `WIPACrepo__iceprod-339`

Shape (17 steps, **failure**): the agent reads the issue (remove a
`getip.php` lookup pointing at a soon-to-be-decommissioned server),
runs `ls` / `find_file` / `grep`, opens `iceprod/core/functions.py`,
edits line 274, and **submits without running any tests**. The
upstream eval (post-submission) records the patch as not resolving the
issue.

#### B-good (right way)

```python
session = LedgerSession("Remove getip.php request")
inv = session.add(
    "Locate getip.php usage in the repo",
    step=2, category=SubtaskCategory.INVESTIGATION,
)
session.complete(
    inv, "step 11: search_file finds getip.php in iceprod/core/functions.py", step=11,
)
prod = session.add(
    "Replace getip.php lookup at functions.py:274",
    step=12, category=SubtaskCategory.PRODUCT,
)
session.complete(prod, "step 14: edit 274:274 issued", step=14)
val = session.add(
    "Verify replacement does not break tests",
    step=14, category=SubtaskCategory.VALIDATION,
)
# DELIBERATELY NOT COMPLETED.
art = session.add("Submit final patch", step=16, category=SubtaskCategory.ARTIFACT)
session.complete(art, "step 16: submit issued", step=16)
```

…and `run_notes.md` records, under "Known missing evidence":

> The agent submitted at step 16 without running any tests in-trace.
> `eval_output.txt` exists (4030 chars, post-hoc) and shows the patch
> did not resolve the issue, but no in-trace pytest / repro / eval-log
> read happened. The `Verify replacement…` validation leaf is left at
> `not_started`. Final progress will therefore be < 1.0 even though
> the agent submitted, which is the correct shape: validation as
> *discovered* work was never performed.

Why this is right:

- Validation leaf left at `not_started` because no validation step is
  visible in the trace. Progress drops as a direct consequence — that
  drop *is* the observation we want.
- `final_success=False` is mentioned only in `run_notes.md`, never as
  ledger evidence. The ledger doesn't change shape based on it.
- Hidden-work gap (the agent never noticed there's a test fixture for
  the same lookup at `tests/core/functions_test.py:…`) noted in
  `run_notes.md` rather than retro-fitted as a discovered subtask.

#### B-bad (wrong way) — DO NOT DO THIS

```python
session = LedgerSession("Remove getip.php request")
inv = session.add("Locate the file", step=2, category=SubtaskCategory.INVESTIGATION)
session.complete(inv, "evidence: agent found it", step=11)
prod = session.add("Edit the file", step=12, category=SubtaskCategory.PRODUCT)
session.complete(prod, "evidence: agent submitted", step=14)
val = session.add("Run tests", step=14, category=SubtaskCategory.VALIDATION)
session.complete(val, "eval_output.txt shows the patch was wrong", step=16)
session.reopen(prod, reason="patch incorrect per eval_output.txt", step=16)
```

What's wrong:

- Validation marked complete because `eval_output.txt` exists. But
  the agent never *ran* validation in-trace; the eval log is post-hoc.
  This violates "use the final eval only as final validation evidence
  for a validation subtask the agent itself surfaced".
- `prod` reopened "because the patch was wrong per eval_output.txt"
  — this uses the post-hoc upstream label to retroactively change
  intermediate completion. Hard rule violation.
- Evidence strings ("evidence: agent found it") have no step
  citation, so the next annotator cannot audit the call.
- Net effect: the ledger now mirrors `final_success` (validation
  reached then immediately torn down), so the observation channel
  reduces to a noisy proxy for the upstream label. This is exactly
  the failure mode the protocol exists to prevent.

## 11. Common pitfalls (failure mode catalogue)

For each pitfall: a short statement of the trap, then the rule it
violates.

1. **"The eval log shows the patch fails, so I'll mark validation
   incomplete."** Violates § 4.4 — the eval log is post-hoc; if the
   agent never ran tests, the validation leaf was never started in
   the first place, regardless of the patch's correctness.
2. **"The agent issued `submit` after edit, so artifact = product
   complete."** Violates § 7 — `submit` is `ARTIFACT`, separate from
   `PRODUCT`. Don't merge them; the category split is what makes the
   per-category curves useful.
3. **"This trace is short, I'll just call it one big PRODUCT leaf."**
   Violates the discovered-work principle. Even a 17-step trace
   surfaces investigation work (issue read, grep, open) that the
   ledger should reflect.
4. **"Lots of repeated `find_file`s — I'll mark `blocked`."**
   Sometimes correct, sometimes not. `blocked` requires the agent to
   be visibly stuck on a *condition* (missing tool, wrong env). A
   long search that eventually succeeds is `in_progress`, not
   `blocked`.
5. **"The progress curve dipped at step 24 and that looks bad — let
   me drop the reopen."** Violates § 4.3. The dip is the
   observation. Smoothing it out reduces the channel to a monotone
   proxy, which is what the framework explicitly is not.
6. **"I split a vague leaf into three children, then also marked the
   parent invalidated."** Redundant — `SPLIT_SUBTASK` invalidates the
   parent automatically (`ledger_progress/core.py` replay rules). Do
   not also emit `INVALIDATE_SUBTASK`.
7. **"I'll add a leaf for work the agent never named because it's
   *obviously* needed."** Allowed only if a stack trace, error
   message, or issue-text quote in the visible trace makes that work
   nameable to an honest observer (§ 2). Cite the step. If you can't,
   leave it hidden — that gap is a real datum.

## 12. What this protocol does NOT decide

- **Annotator helper tooling.** D3 covers that; default is "no
  helper unless D4 reveals concrete friction."
- **`run_notes.md` template shape.** D2 owns that.
- **Inter-annotator agreement metric.** Workstream H decides this;
  for D1 + D4 a single annotator is sufficient.
- **Cross-model comparison.** Workstream P; pilot is single-model
  (`swe-agent-llama-70b`).
- **Predictive modeling.** Workstream Q; § 0 prohibits using
  `final_success` as a feature, and that prohibition extends here.

## 13. Open questions / known caveats

1. The trace's `assistant` `text` block sometimes carries thought-only
   turns with no command (parse warning `no_fenced_block`). The
   protocol treats these as zero-evidence, zero-status-change steps.
   If a future fallback model produces many such turns, we may need
   a "thought-only assistant" annotation convention; for the 70b
   pilot, this is rare.
2. The `submit` action is filed under `ARTIFACT`. If the corpus turns
   out to contain `submit` actions that *also* perform validation
   (some SWE-agent harnesses run tests on `submit`), the protocol
   may need a "submit-as-validation" rider. Not observed in pilot
   so far; revisit if seen.
3. For traces where the agent itself reads the eval log mid-run (e.g.
   a `pytest` followed by inspecting failures and editing again), the
   eval-log read is in-trace evidence and the validation leaf can
   transition normally. For traces where the eval log appears only
   after the trace ends, see § 4.4.
4. We do not require a `BLOCKED` event for every long search loop;
   long unproductive stretches are simply low-progress regions of the
   curve. If they become common enough to drown out signal,
   Workstream K's evidence audit will surface that, and we revise.
