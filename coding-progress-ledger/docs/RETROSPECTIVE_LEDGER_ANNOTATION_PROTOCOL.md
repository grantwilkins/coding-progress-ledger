# Retrospective ledger annotation protocol

This document defines the rules for converting **any** agent trace
(SWE-agent, future scaffolds, hand-written human runs, anything that
yields an ordered sequence of steps) into a `ledger.jsonl` that
replays under `ledger_progress.replay()`. It is binding for every
retrospective annotation pass anywhere in this repo. Source-specific
addenda (e.g. `docs/SWE_AGENT_TRACE_ANNOTATION_ADDENDUM.md`) refine
how it lands on a particular trace shape; they do not override.

This protocol is NOT a planning document, NOT a controller spec, and
NOT an annotator-friendliness pass — D2 owns the `run_notes.md`
template, D3 owns optional helpers.

All § 0 project rules apply (`TASKS.md`). The most load-bearing for
annotation are restated here:

> **Annotate only visible trace evidence.**
> **Do not use the upstream success label to decide intermediate completion.**
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

You read an ordered, normalized trace of agent steps and write a
sequence of ledger events that describe *what work an outside observer
can see the agent discovering and finishing*, step by step. You are
not retelling the story. You are not inferring intent. You are not
predicting success.

The output is one `ledger.jsonl` per run, replayable by
`ledger_progress.replay()`, plus an extended `run_notes.md` that
records evidence citations and any places you were unsure.

The exact filenames of the input artifacts (`normalized_trace.json`,
`task.md`, etc.) and the meaning of the upstream success label are
defined by the source-specific addendum, not here.

## 2. Discovered work vs hidden work

This distinction is the single most important idea in the protocol.

- **Discovered work** = a subtask the trace shows the agent (or, by
  retrospective annotation, the observer) can name. If the agent
  never navigates to a file, "modify that file" is not yet
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
  upstream success labels. That is a *feature* of an observation
  channel, not a bug — Workstream P/Q investigates the relationship;
  retrospective annotation does not compress it.

If you find yourself adjusting events so that progress ≈ "how
confident am I the agent succeeded", **stop and re-read § 2.**

## 4. Hard rules

- **Annotate only visible trace evidence.** Every `complete` event
  must cite either (a) a step index range in the normalized trace,
  (b) a line in a final-state artifact (e.g. patch, diff, output
  file), or (c) a line in an in-trace tool return. No external
  knowledge of the project, the issue tracker, or how the task
  "should" be done.
- **Do not use the upstream success label to decide intermediate
  completion.** The upstream label is a stratification field; per
  § 0 of `TASKS.md` it is never a feature. You may consult final-state
  artifacts only as evidence for a *validation* subtask the agent
  itself surfaced; you may not use the upstream label to retroactively
  complete or reopen subtasks.
- **Do not force monotonicity.** If the agent edits a file, then
  later reopens that work because the patch was wrong, emit a
  `REOPEN_SUBTASK` event. Progress will dip; that is the correct
  shape.
- **Use the final eval / final-state artifact only as final-validation
  evidence.** If the agent ran tests / inspected logs in-trace, cite
  the relevant tool turn. If the final-state artifact only exists
  post-hoc (the agent submitted without inspecting anything), then
  validation was *not done* by the agent, no matter what the artifact
  says about correctness. Record this in `run_notes.md` and leave
  validation in `not_started` or `in_progress`.
- **Preserve uncertainty.** When you are not sure whether to mark
  something complete, don't mark it complete. Note the ambiguity in
  `run_notes.md` under "Uncertain decisions". One annotator's
  uncertainty is the next annotator's evidence gap.

## 5. Categories

The six categories live in `ledger_progress/core.py:SubtaskCategory`.
General usage:

| Category        | Use for                                                                                       |
|-----------------|-----------------------------------------------------------------------------------------------|
| `INVESTIGATION` | Reading the task; locating relevant code or facts; navigating, searching, reading artifacts.  |
| `PRODUCT`       | Behavioral changes to implementation, including fixes to tests when the task itself requires them. |
| `VALIDATION`    | Running tests / repro scripts / checks; interpreting failures; reading in-trace test or eval output. |
| `ENVIRONMENT`   | Setup work that blocks product work without being part of it (dependency installs, missing modules, container/path issues). |
| `ARTIFACT`      | Final-output emission: patch dump, submit, "write the answer to file." |
| `DOCUMENTATION` | When (and only when) the task itself requires a doc change, or the agent updates docs as part of the fix. Default code+docstring edits to `PRODUCT`. |

If you genuinely cannot decide, default to `PRODUCT` and note the
ambiguity. Do not invent new categories.

Source-specific addenda may map domain-specific actions
(shell commands, tool calls, GUI events) onto these categories — but
the category set itself is fixed by `core.py`.

## 6. Statuses — when to transition

Canonical names live in `ledger_progress/core.py:Status`.

- `not_started` — discovered, no observable action yet. Use for
  subtasks added by the annotator from issue evidence the agent has
  not yet acted on.
- `in_progress` — partial action visible in the trace.
- `blocked` — agent is stalled on something it cannot resolve from
  the visible context. Pair `blocked` with a `reason` payload citing
  the step index.
- `complete` — supported by concrete trace evidence (see § 4).
  REQUIRED: at least one evidence string in the event payload.
- `invalidated` — the subtask is no longer load-bearing (e.g.
  `SPLIT_SUBTASK` makes the parent disappear from the leaf set).
  Prefer `invalidate` over `delete`; history matters.
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
- `ADD_EVIDENCE` — when new evidence accumulates on an
  already-known subtask without changing its status.
- `SPLIT_SUBTASK` — when a vague subtask becomes several leaf
  subtasks. The parent is invalidated automatically by the replay
  engine; do not also issue `INVALIDATE_SUBTASK`.
- `REOPEN_SUBTASK` — when previously-complete work is shown
  incomplete. **This is the canonical non-monotonic event** — use it
  whenever the trace contradicts an earlier completion, even if it
  makes the progress curve drop. That drop is the point.
- `INVALIDATE_SUBTASK` — when a subtask remains in history but
  should no longer count as active work. Prefer this over
  `DELETE_SUBTASK`.
- `DELETE_SUBTASK` — see "almost never" in § 6.

## 8. Step numbering and citation

Each ledger event carries an integer `step`. For retrospective
annotation, **set `step` to the normalized-trace step index of the
turn that justifies the event**. This makes evidence traceable: a
reader can open the normalized trace, jump to that step, and see why
you wrote the event.

Two consequences:

- Multiple ledger events can share a `step` (e.g. one tool turn
  surfaces evidence for two subtasks). That is fine.
- `step` need not be dense. Long stretches of trace with no
  annotated events leave gaps in the `step` axis. That is also fine.

Evidence strings in `ADD_EVIDENCE` / `UPDATE_STATUS` payloads SHOULD
include the cited step index. One short sentence per evidence string.
No multi-paragraph quotes.

## 9. Procedure (per run)

Recommended order, optimized for not lying to yourself:

1. Read the task description once. Note the stated goal in
   `run_notes.md` under "Initial reading".
2. Skim the trajectory summary end-to-end before opening the
   normalized trace. Build a rough mental model of the run's shape
   (investigation-heavy? straight to edit? lots of failed commands?).
3. Open the normalized trace. Walk it linearly, top to bottom. Do
   not read any final-state artifacts (patches, eval logs) yet.
4. Use `LedgerSession`. For each step where new discovered work
   appears, emit `add(...)`. For each step that finishes work, emit
   `complete(...)` with the cited step index.
5. After the trace ends, read the final-state artifacts. Use them
   only to (a) confirm a `validation` subtask the agent itself
   surfaced, (b) note in `run_notes.md` if a final artifact shows work
   the trace did not surface (a hidden-work gap).
6. Export with `session.export_jsonl(...)`. Run
   `ledger-run check-run <run_dir>` and address any missing
   artifacts. `progress.csv` and `summary_by_category.json` come from
   `ledger-run export-run`; do not write them by hand.
7. Fill in the rest of `run_notes.md` per the D2 template. Record
   uncertain decisions, evidence gaps, and whether you ever felt
   tempted to use the upstream success label (you didn't, but write
   it down if the temptation surfaced).

## 10. Common pitfalls (failure mode catalogue)

For each pitfall: a short statement of the trap, then the rule it
violates. These are general — domain-specific pitfalls (e.g. "lots
of repeated `find_file`s") belong in source-specific addenda.

1. **"The final-state artifact shows the patch fails, so I'll mark
   validation incomplete."** Violates § 4.4 — if the agent never ran
   validation in-trace, the validation leaf was never started in the
   first place, regardless of correctness.
2. **"The agent emitted the final answer, so artifact = product
   complete."** Violates § 5/§ 7 — `ARTIFACT` is a separate
   category. Don't merge them; the category split is what makes the
   per-category curves useful.
3. **"This trace is short, I'll just call it one big PRODUCT leaf."**
   Violates the discovered-work principle. Even short traces surface
   investigation work that the ledger should reflect.
4. **"The agent retried the same query a lot — I'll mark `blocked`."**
   Sometimes correct, sometimes not. `blocked` requires the agent to
   be visibly stuck on a *condition* (missing tool, wrong env). A
   long search that eventually succeeds is `in_progress`, not
   `blocked`.
5. **"The progress curve dipped at step N and that looks bad — let
   me drop the reopen."** Violates § 4.3. The dip is the
   observation. Smoothing it out reduces the channel to a monotone
   proxy, which is what the framework explicitly is not.
6. **"I split a vague leaf into three children, then also marked
   the parent invalidated."** Redundant — `SPLIT_SUBTASK`
   invalidates the parent automatically (`ledger_progress/core.py`
   replay rules). Do not also emit `INVALIDATE_SUBTASK`.
7. **"I'll add a leaf for work the agent never named because it's
   *obviously* needed."** Allowed only if the visible trace makes
   that work nameable to an honest observer (§ 2). Cite the step. If
   you can't, leave it hidden — that gap is a real datum.

## 11. What this protocol does NOT decide

- **Annotator helper tooling.** D3 covers that.
- **`run_notes.md` template shape.** D2 owns that.
- **Inter-annotator agreement metric.** Workstream H decides this.
- **Cross-source / cross-model comparison.** Workstream P.
- **Predictive modeling.** Workstream Q; § 0 prohibits using the
  upstream success label as a feature, and that prohibition extends
  here.
- **Source-specific role / action mappings.** Source-specific addenda
  (e.g. SWE-agent shell vocabulary) refine § 5 for one trace shape.
