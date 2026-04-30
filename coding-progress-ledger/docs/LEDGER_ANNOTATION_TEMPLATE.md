# Ledger annotation template (`run_notes.md`)

Copy this file's body into `<run_dir>/run_notes.md` at the start of
each retrospective annotation pass. The template is source-agnostic;
source-specific addenda (e.g.
`docs/SWE_AGENT_RETROSPECTIVE_LEDGER_PROTOCOL.md`) refine *what*
counts as evidence, but the shape below is the same for every trace.

The protocol that binds annotation is
`docs/RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md`. This file does not
restate it; it just structures the annotator's notes so they read
the same across runs.

`LedgerSession` API methods referenced below (so notes link back to
events explicitly):

- `session.add(description, step, category=...)` — `ADD_SUBTASK`
- `session.start(subtask_id, step)` — `UPDATE_STATUS → in_progress`
- `session.complete(subtask_id, evidence, step)` — `UPDATE_STATUS → complete`
- `session.block(subtask_id, reason, step)` — `UPDATE_STATUS → blocked`
- `session.split(parent_id, children, step, reason=...)` — `SPLIT_SUBTASK`
- `session.reopen(subtask_id, reason, step)` — `REOPEN_SUBTASK`
- `session.invalidate(subtask_id, reason, step)` — `INVALIDATE_SUBTASK`
- `session.score()` — read-only; never used to decide a transition.

Replace every `{{ ... }}` placeholder. Delete sections that are
genuinely empty (e.g. no `Uncertain decisions`); do not leave
"N/A" filler — that pattern hides the absence of thought.

---

## Run notes — `{{pilot_id}}` (`{{instance_id}}`)

- annotator: `{{name_or_handle}}`
- annotation pass: `{{pilot-zero | full-pilot | re-annotation}}`
- protocol version: `RETROSPECTIVE_LEDGER_ANNOTATION_PROTOCOL.md` @ commit `{{sha}}`
- source addendum: `{{path/to/addendum.md}}` (if any)
- upstream success label (NOT a feature): `{{True | False | missing}}`

### 1. Initial reading

What does the task ask for, in your own words, after one pass over
`task.md` and the trajectory summary? Two or three sentences. If the
task description is ambiguous, name the ambiguity here — annotation
proceeds against this reading, not against an unstated alternative.

> {{one paragraph}}

### 2. Initial ledger proposal

The list of subtasks you expect to discover, written **before**
walking the normalized trace. This is calibration: rereading it after
the walk tells you which assumptions the trace contradicted.

```text
- {{INVESTIGATION}} {{short description}}
- {{PRODUCT}}       {{short description}}
- {{VALIDATION}}    {{short description}}
- {{ARTIFACT}}      {{short description}}
```

### 3. Checkpoint notes

A few sentences per natural break in the trace (file localized,
first edit, first test run, retry, submit). This is for the human
reading later, not the replay engine.

- step `{{N}}`: {{what changed}}
- step `{{N}}`: {{what changed}}
- step `{{N}}`: {{what changed}}

### 4. Uncertain decisions

Every place you considered two interpretations and picked one. State
both, the choice, and the cite.

- **{{short summary}}** — alternatives: A `{{...}}` vs B `{{...}}`.
  Chose A because step `{{N}}` shows `{{evidence}}`. Re-evaluate if
  D5 audit flags this.

If you weren't uncertain anywhere, that's allowed — but read § 4.5
("Preserve uncertainty") of the general protocol once more and
confirm.

### 5. Evidence citations

The map from each `complete`-status subtask to the step(s) that
justify it. If you used `ADD_EVIDENCE` to corroborate, cite both
events.

| subtask id | category | completed at step | evidence step(s) | one-line citation |
|------------|----------|-------------------|------------------|-------------------|
| `S1`       | `INVESTIGATION` | `{{N}}` | `{{N}}`              | `{{...}}`        |

### 6. Known missing evidence

Subtasks left at `not_started` / `in_progress` / `blocked` at run
end, and *why*. The "why" is the part that earns its place in the
notes.

- `{{subtask}}` left at `not_started`: the trace never showed
  `{{condition}}`. Final-state artifact `{{artifact}}` exists but is
  post-hoc (general § 4.4).

If the `final_diff.patch` shows work the trace did not surface (a
hidden-work gap), state that here. Do **not** retro-fit it as a
discovered subtask.

### 7. Final scope closure

What does the ledger look like at the last step?

- total leaves: `{{N}}`
- complete: `{{N}}` · in_progress: `{{N}}` · blocked: `{{N}}` · not_started: `{{N}}` · invalidated: `{{N}}`
- progress (overall): `{{X.YY}}`
- progress (CODING_CATEGORIES = product+validation+investigation): `{{X.YY}}`

Was anyone tempted to use the upstream success label as evidence at
any point during the walk? Write `yes` or `no` and one sentence —
this seeds the D5 quality field
`whether_final_success_used_only_at_end`.

### 8. Schema gaps observed

Anything the protocol or schema didn't have a clean home for.

- {{e.g. "thought-only assistant turn — no fenced block. Treated as
  zero-evidence per addendum § 5.2; if these become common, may need
  a dedicated convention."}}

If none, write "none observed" and move on.
