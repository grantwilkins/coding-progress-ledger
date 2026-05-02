# Q1 — Channel-native prediction targets

Five binary targets defined per **checkpoint row** of the W3 estimator
table (`datasets/swe_agent_estimator_checkpoints.csv`). Each target is
derived from events strictly *after* the checkpoint step, joined by
`(run_id, step)`. The features at the checkpoint never see any of
these labels.

These targets predict the *visible-work-frontier dynamics* of the
ledger: drops, reopens, validation surprises, stuck states, and
submit-without-validation. They do **not** predict `final_success`;
final-outcome prediction stays deferred until Q1–Q5 show the channel
features are coherent (per § Workstream Q in `TASKS.md`, and the
locked memory note that progress is decoupled from outcome by design).

## Window convention

Default look-ahead: **`H = 5` steps** (override with `--horizon-steps`).
"Next window" means the half-open interval `(S, S + H]` of step indices.
"Terminal" means evaluated at the run's last event step. The same `H`
is used for every horizon-dependent target so reports compare cleanly.

## Targets

### 1. `future_progress_drop`
Definition. `1` iff there exists a checkpoint at step `S' ∈ (S, S+H]`
in the same run with `coding_progress[S'] < coding_progress[S]`.
Else `0`. Terminal-only rows (no `S' ≤ S+H` exists) get `0`.

Source signal. W3 column `coding_progress` at later checkpoints in the
same run.

What it asks. *In the next 5 steps, will visible coding progress
regress?* Captures retrospective REOPEN / INVALIDATE / SPLIT-driven
denominator growth that pulls the scalar back.

Non-leakage. The current row's features (`coding_progress`,
`largest_progress_drop_so_far`, `num_reopens_so_far`, …) are computed
*at or before* step `S`. The label looks at steps `> S` in the same
run. No feature column is derived from the future window.

### 2. `product_reopened_after_completion`
Definition. `1` iff `num_reopens_so_far[S+H] > num_reopens_so_far[S]`,
restricted to reopens whose subtask category is PRODUCT. Computed by
walking events in `(S, S+H]` and counting `REOPEN_SUBTASK` events on
PRODUCT-category subtasks. Else `0`.

Source signal. Per-run `ledger.jsonl` events of type
`REOPEN_SUBTASK`; subtask category resolved via current ledger state
at the reopen event.

What it asks. *In the next 5 steps, will the agent walk back a
PRODUCT subtask it had already marked complete?* This is the
canonical "I claimed done, but it wasn't" channel signal.

Non-leakage. Features only count reopens at-or-before `S`
(`num_reopens_so_far`); label counts reopens strictly after `S`.

### 3. `validation_exposes_new_work`
Definition. `1` iff inside `(S, S+H]` the run has both:
- a VALIDATION-category event of type `UPDATE_STATUS` whose new
  status is `complete` *or* `blocked` (validation actually executed
  in the window), and
- at least one `ADD_SUBTASK` or `REOPEN_SUBTASK` event after that
  validation event whose subtask category is in
  `{PRODUCT, INVESTIGATION}`.
Else `0`.

Source signal. Per-run `ledger.jsonl` events; categories resolved at
event time.

What it asks. *Will running tests / reviewing artifacts in the next 5
steps reveal hidden work that has to be added or reopened?* This is
the K2 "hidden-work-gap" signal moved into the predictive frame.

Non-leakage. The current row carries `validation_started` /
`validation_complete` / `validation_failed` flags that summarize the
past; the label requires a fresh validation event *and* a subsequent
discovery, both inside the future window.

### 4. `stuck_loop_next_window`
Definition. `1` iff inside `(S, S+H]` an `UPDATE_STATUS` event sets a
subtask to `blocked` with reason text containing `"loop"` or
`"stuck"` (mirroring the W3 `repeated_observation_loop_flag` rule).
Else `0`. Rows where `repeated_observation_loop_flag` is already
`true` at step `S` get `0` (the loop has already been observed —
nothing to predict). The cycle-length-agnostic OBSERVATION-payload
sub-rule from the D1 protocol applies only to the source trace, not
to the normalized ledger; we do not look at source-trace events here.

Source signal. Per-run `ledger.jsonl` events.

What it asks. *Will the agent enter a loop within the next 5 steps?*
Distinct from the W3 feature `repeated_observation_loop_flag`, which
is monotone-non-decreasing past state.

Non-leakage. Feature reflects "loop already seen at-or-before `S`";
label reflects "loop emerges strictly after `S`". Rows already in a
loop are masked to `0` so the model cannot learn the trivial
"already true → stays true" rule.

### 5. `submit_without_validation_state` *(terminal)*
Definition. `1` iff at the run's terminal step the W3 column
`submit_without_validation` is `true`. Same value for every
checkpoint in the run. Else `0`.

Source signal. W3 column `submit_without_validation` at the last
checkpoint in the run.

What it asks. *Will this run submit without ever running validation
to completion?* Terminal rather than horizon-bounded because
"submit-without-validation" is a run-level shape, not a
window-localized event.

Non-leakage. The label is constant per run, so a model trained on
this label is predicting "given what I see at step `S`, will this
run end in submit-without-validation?". No future feature leaks into
the row's features — the row's `submit_without_validation` feature
column reflects state *at or before* `S`. The terminal label may
equal the row's feature when the no-validation path was already
locked in early; that is a property of the data, not leakage.

## What is NOT a target (yet)

- `final_success` — explicitly deferred to Q6, after Q1–Q5 show the
  channel-native targets are coherent. The project rules forbid
  using final outcome as a feature; it remains a label only and will
  be revisited only once the easier channel-shape predictions are
  baselined.
- Wall-clock-deadline targets from V1 / V2 — the retrospective pilot
  carries no real timestamps (per N6 caveat), so deadline-aware
  labels would be derived from synthetic step-time. Defer to a live
  N=20 round.

## Frontier-policy caveat

The retrospective pilot and the live N=20 batch use slightly
different validation semantics (per `runs/swe_agent_live/PARITY_REPORT.md`).
Q targets are defined against the **retrospective pilot** because
that is the dataset W3 was built from. When a live-N=20 checkpoint
table is built (tracked elsewhere), targets 3 and 5 may need
re-evaluation under the live frontier policy; targets 1, 2, 4 are
agnostic to the policy and should transfer.

## Files

- `scripts/build_q_labels.py` — emits
  `datasets/swe_agent_q_labels.csv` keyed by `(run_id, step)` with
  one column per target plus a `horizon_steps` column for
  reproducibility.
- `tests/test_q_labels.py` — locks per-target semantics on known
  pilots (`f_06`, `s_03`, `f_02`, `s_04`).
- `tests/test_q_no_leakage.py` — asserts that no Q1 target column
  appears in W3 feature columns; asserts that label generation reads
  only events with `step > S` for horizon-dependent targets.
