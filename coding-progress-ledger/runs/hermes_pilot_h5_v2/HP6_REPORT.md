# HP6 — softened BLOCKED rule + reproducibility pin

Framing: HP5 shipped a heuristic auto-annotator that BLOCKED on the
**first non-zero error response** in a leaf. That single-error rule
distorted the ledger as a measurement instrument: 22/30 pilots tagged
`stuck_loop`, the `coding_progress` distribution stretched to
[0.50, 1.00] because any transient error truncated the leaf at non-1.0,
and `stuck_loop_next_window` was suppressed by the W3 mask (the loop
flag was already set at the same checkpoint where the next-window
target would otherwise fire). HP6 replaces the rule with **3+
consecutive identical errors** before BLOCKING — the conservative
reading of "stuck loop" that requires the agent to actually be looping
on the same failure, not merely to hit a single recoverable error.

The observation channel is supposed to be a replayable, time-indexed
trace of *what the agent did*. BLOCKED on first error compresses
"hit-and-recover" into the same bucket as "looped on the same
exception three times" — it interprets transient evidence as terminal.
The softened rule keeps the recovery story visible: the leaf stays
ACTIVE through the error, then closes COMPLETE on the recovery step.

## TL;DR

| metric | HP5 | HP6 | Δ |
|---|---:|---:|---:|
| `stuck_loop` shape-tag (per pilot) | 22/30 | **1/30** | −21 |
| BLOCKED leaves                     | 48    | **2**   | −46 |
| COMPLETE leaves                    | 122   | **146** | +24 |
| IN_PROGRESS leaves (trace ends mid-attempt) | 0  | **22**  | +22 |
| NOT_STARTED leaves (channel-state bug under HP5) | 0 | **0** | — |
| `coding_progress` median           | 0.789 | **1.000** | +0.21 |
| `coding_progress` mean             | 0.783 | **0.902** | +0.12 |
| pilots at coding_progress = 1.000  | 8/30  | **18/30** | +10 |
| `Q:future_progress_drop` positives | 151/370 | 136/348 | −15 |
| `Q:validation_exposes_new_work`    | 93/370  | 50/348  | −43 |
| `repeated_observation_loop_flag` checkpoints | 0/370 | **9/348** | +9 |
| Q checkpoint rows                  | 370 | 348 | −22 |

All four downstream pipelines run **unchanged** on the v2 tree; the
only edits are in `scripts/auto_annotate_hermes.py` (the rule) and
`tests/test_auto_annotate_hermes.py` (invariants).

## What changed in code

`scripts/auto_annotate_hermes.py`:

- Added constant `ERROR_STREAK_BLOCK_THRESHOLD = 3`.
- Replaced the "BLOCK on first error response" branch with a
  per-leaf `error_streak` accumulator. The leaf BLOCKs only when
  `len(set(error_streak[-3:])) == 1` and the streak is ≥ 3. Any
  non-error response (or a different error body) resets the streak.
- Pitfall H3 (3+ identical observation bodies regardless of error)
  is preserved — it still catches non-error stuck loops.
- `last_complete_step` now only advances on *non-error* responses,
  so a leaf with `error → success` completes at the success step,
  while a leaf whose last response is a transient error remains
  IN_PROGRESS. This is the desired behavior: the channel reports
  the actual state, not a forced terminal verdict.
- Emit `s.start(sid, step=call_step)` on the first paired response
  observed for the leaf, anchored at the assistant *call* step (not
  the response step) so the channel does not backdate the agent's
  transition to information it had not yet observed. This separates
  **NOT_STARTED** (agent never invoked the tool) from **IN_PROGRESS**
  (agent attempted, did not recover). Under HP5 with BLOCK-on-first-
  error this distinction was unreachable; HP6's softened rule made
  it observable, so the channel must now name the state honestly.
  Surfaced by the research-test-creator pass
  (`test_lone_error_leaf_is_in_progress…`); call-step anchor pinned
  by `test_in_progress_event_anchored_at_call_step_not_response_step`.
- BLOCK reason strings now include the substring `stuck loop:` so
  `build_estimator_checkpoints.py:184` (`if "loop" in reason or "stuck"
  in reason: state.repeated_loop_flag = True`) actually fires. Pinned
  by `test_block_reason_contains_loop_keyword_for_downstream_flag`.

`tests/test_auto_annotate_hermes.py` adds HP6-specific cases:

Initial pass (rule mechanics):

- `test_softened_rule_constants`
- `test_single_error_does_not_block`
- `test_three_identical_errors_block`
- `test_three_distinct_errors_do_not_block`
- `test_error_then_success_completes`
- `test_streak_resets_on_success`
- `test_softened_rule_does_not_increase_blocked_count` (per-pilot
  upper-bound regression vs the old rule on the HP4 traces)

research-test-creator pass (semantic claims):

- `test_lone_error_leaf_is_in_progress_not_complete_not_blocked` —
  failed initially; surfaced the NOT_STARTED bug (the channel was
  leaving lone-error leaves at NOT_STARTED rather than IN_PROGRESS,
  conflating "never tried" with "tried and stuck").
- `test_two_identical_errors_do_not_block_boundary` (off-by-one)
- `test_error_streak_does_not_leak_across_leaves` (per-leaf state)
- `test_recovered_leaf_contributes_full_progress_credit` (channel-
  vs-outcome decoupling: recovery = 1.0, no penalty)
- `test_recovered_leaf_evidence_cites_success_step_not_error_step`
- `test_blocked_leaf_contributes_zero_progress_credit`

Critic-pass adds (D1, D6, D9):

- `test_in_progress_event_anchored_at_call_step_not_response_step`
  pins D1 (call-step anchor).
- `test_block_reason_contains_loop_keyword_for_downstream_flag`
  pins D6 (block-reason → `repeated_observation_loop_flag` wiring).
- `test_streak_resets_on_success` was rewritten to a 4-step
  `err,err,ok,err` trace that *actually* fails if the reset is
  deleted (the original 5-step `err,err,ok,err,err` passed
  vacuously since both halves were length-2 streaks; flagged by D9).

All 53 parametrized auto-annotator tests pass; full suite at 585/585.

## Acceptance gate

| Gate | HP5 | HP6 |
|---|---|---|
| `scripts/build_ledger_observation_dataset.py` runs unchanged | PASS | PASS |
| `scripts/build_estimator_checkpoints.py` runs unchanged | PASS | PASS |
| `scripts/build_q_labels.py` runs unchanged | PASS | PASS |
| `scripts/label_observation_shapes.py` runs unchanged | PASS | PASS |
| `ledger-run check-run` passes on all 30 pilots | PASS | **PASS (30/30)** |
| stuck_loop shape-tag below HP4 frontier-policy expectation | FAIL (22/30) | **PASS (1/30)** |
| `coding_progress` mean rises toward 1.0 | n/a | **PASS (0.78 → 0.90)** |
| 30 raw HP5 rows pinned in-tree | FAIL | **PASS** |
| Heuristic vs human overlap ≥ 50% on HP4 pilots | PASS | **PASS** |

## Status × category — where the 46 unblocks went

HP5 (BLOCKED-on-first-error):

| category      | complete | blocked | in_progress |
|---------------|---------:|--------:|------------:|
| product       | 40       | 6       | 0           |
| validation    | 37       | 21      | 0           |
| investigation | 33       | 15      | 0           |
| environment   | 8        | 1       | 0           |
| artifact      | 4        | 5       | 0           |

HP6 (3+ identical errors):

| category      | complete | blocked | in_progress |
|---------------|---------:|--------:|------------:|
| product       | 42       | 1       | 3           |
| validation    | 42       | 1       | 15          |
| investigation | 46       | 0       | 2           |
| environment   | 8        | 0       | 1           |
| artifact      | 8        | 0       | 1           |

Two leaves still BLOCK under HP6 — both legitimate stuck loops where
the agent retried with the same error body three times in a row. The
22 leaves that became `in_progress` under HP6 are the channel's
honest report: the agent saw an error and stopped before producing
a clean recovery, so the leaf is left ACTIVE. The estimator's job is
to learn that an `in_progress` leaf at the trailing edge of a trace
is a different signal than a `BLOCKED` leaf — that distinction was
collapsed under HP5.

## Coding-progress distribution

```
HP5 (n=30):  [1.00]= 8  (0.75,1.00)= 8  (0.50,0.75]=11  (0.25,0.50]=3  [0,0.25]=0
HP6 (n=30):  [1.00]=18  (0.75,1.00)= 5  (0.50,0.75]= 6  (0.25,0.50]=1  [0,0.25]=0
```

The 10 pilots that moved into the [1.00] bucket are the ones whose
*only* HP5 BLOCK was a single transient error in an otherwise clean
recovery. Under HP6 those leaves close COMPLETE and the pilot's
final progress hits 1.000.

This is *not* a claim that the agent population is 90% successful —
it is a claim that **the heuristic stops mistaking a recoverable
error for a terminal failure**. The estimator now sees a cleaner
"recovery" pattern that it can learn to distinguish from genuine
stuck loops.

## stuck_loop flips (HP5 true → HP6 false)

21 pilots flipped from `stuck_loop=true` to `stuck_loop=false`:

```
003 004 005 008 009 010 012 013 014 015 017
018 019 020 022 023 024 025 027 029 030
```

The single remaining `stuck_loop=true` pilot (one of the 30) is the
case where 3+ identical errors actually fired the new rule.

## Q1 channel-native targets

| target                                  | HP5 | HP6 |
|-----------------------------------------|----:|----:|
| `future_progress_drop`                  | 151/370 | 136/348 |
| `product_reopened_after_completion`     | 0   | 0   |
| `validation_exposes_new_work`           | 93/370  | 50/348  |
| `stuck_loop_next_window`                | 0   | 0   (W3 mask, see below) |
| `submit_without_validation_state`       | 0   | 0   |
| `repeated_observation_loop_flag`        | 0/370 | 9/348 |

`future_progress_drop` falls from 151 → 136 because fewer transient
errors trigger the BLOCK→re-add path that HP5 used to manufacture
progress drops. `validation_exposes_new_work` drops from 93 → 50
because the heuristic no longer slices a single VALIDATION subtask
into multiple BLOCKED fragments separated by category boundaries —
fewer category transitions immediately after a VALIDATION close
means the W3 detector fires less. This is an improvement: HP5 was
*over-counting* both patterns by manufacturing transitions out of
transient errors.

Critic-pass D6 fix: HP6's first rollout had `BLOCK` reasons reading
"3+ consecutive identical errors" / "3+ identical tool responses
(Pitfall H3)" — neither contained the substring `loop` or `stuck`,
which is what `build_estimator_checkpoints.py` keys on to set
`repeated_observation_loop_flag`. The shape-tag `stuck_loop` fired
at 1/30 but the upstream checkpoint flag stayed 0 — the channel and
the estimator were silently desynchronized. The fix prepends
"stuck loop: " to both BLOCK reasons so the flag now fires
(9/348 checkpoint rows). `stuck_loop_next_window` remains 0 by
*W3-mask suppression* (the flag is set on the same checkpoint where
the next-window target would otherwise fire) — this is now
verifiable from the data, not just claimed.

The two remaining structural zeros (`product_reopened_after_completion`,
`submit_without_validation_state`) stay zero for the same reasons
documented in HP5: the heuristic emits no REOPEN events, and the
heuristic always closes a VALIDATION leaf before ARTIFACT.

The total checkpoint count fell from 370 to 348 because the BLOCKED
state-machine fires fewer times (W3 emits one fewer checkpoint per
non-blocked-anymore leaf). No pilot was dropped. The IN_PROGRESS
event added by HP6 anchors at the assistant call step (same step as
ADD_SUBTASK) so it does not create a new checkpoint row.

## Reproducibility — pinned

`external_data/hermes/pilot_cache_h5/` now commits the 30 raw HF
dataset rows used by HP5/HP6 (~3.4 MB total: 18 glm-5.1 + 12 kimi).
`.gitignore` was extended with two patterns:

```text
!external_data/hermes/pilot_cache_h5/
!external_data/hermes/pilot_cache_h5/**
```

A clean-clone reproducer no longer needs to re-fetch via the HF
datasets-server (which previously returned HTTP 429 above 1,500 kimi
rows during the HP5 build). The manifest at
`external_data/hermes/manifests/hermes_pilot_h5_sample.csv` and the
pinned cache together fully reproduce the v2 tree.

## What HP6 supports

- The heuristic auto-annotator no longer over-flags `stuck_loop` (1/30
  vs 22/30); the channel's BLOCKED/COMPLETE/IN_PROGRESS distinction
  now reflects the actual trace shape, not a first-error mask.
- `coding_progress` per pilot is honest about recovery: median 1.00,
  mean 0.90 across N=30; the [0.50, 0.75] band is now a real
  population of "agent stopped before finishing" rather than a
  heuristic artifact.
- The 30 raw rows are pinned in-tree, closing the principal HP5
  reproducibility gap.

## What HP6 still does NOT support

- REOPEN-mediated Q1 targets remain at zero positives — the heuristic
  cannot infer reopens from tool transcripts alone. A human-annotated
  pass on a sub-sample (or a future heuristic that detects "agent
  re-edits a file already marked PRODUCT-complete") would unlock
  those targets.
- ARTIFACT leaf bookkeeping is unchanged from HP5 — the
  `skill_manage`-as-ARTIFACT convention still produces the
  per-retry-leaf split flagged in `HERMES_H5_REPORT.md`.
- Outcome prediction (Q6) — Hermes ships no `final_success` field;
  this is unchanged structural N/A.

## Known limitations (carried from HP5; pinned by critic pass)

- **Body equality is verbatim.** `_response_body` returns
  `json.dumps(obj, sort_keys=True)` for dict observations, so two
  errors that differ only by a timestamp, PID, or tempfile path
  are considered distinct and never form a 3-streak. Real stuck
  loops with non-canonical bodies are false-negatives. A future
  iteration could canonicalize tempfile paths / numeric IDs before
  comparing.
- **Streak is per-leaf.** A genuine stuck-loop where 3 identical
  errors span two adjacent groups (e.g., write-then-test-then-write
  on the same failure) is structurally invisible to the rule. The
  test `test_error_streak_does_not_leak_across_leaves` pins this
  *as the design choice*, not as an unintended bug.
- **Rule attribution.** When both rules could fire (e.g., 3 identical
  error bodies satisfy both the error-streak rule and Pitfall H3),
  control flow ordering means the error-streak rule wins and the
  H3 reason string is never emitted. This is benign but means the
  BLOCKED-reason histogram is biased toward the error-streak label.

## Reproducer

```bash
# 1. Re-materialize pre-annotation artifacts from the pinned cache.
uv run python scripts/import_hermes_trace.py \
  --sample-csv external_data/hermes/manifests/hermes_pilot_h5_sample.csv \
  --runs-dir runs/hermes_pilot_h5_v2 \
  --raw-cache-dir external_data/hermes/pilot_cache_h5

# 2. HP6 softened auto-annotation.
uv run python scripts/auto_annotate_hermes.py --runs-dir runs/hermes_pilot_h5_v2

# 3. Four parity pipelines (unchanged).
uv run python scripts/build_ledger_observation_dataset.py \
  --runs-dir runs/hermes_pilot_h5_v2 \
  --output-csv datasets/hermes_pilot_h5_v2_observations_event.csv \
  --output-event-csv datasets/hermes_pilot_h5_v2_observations_event.csv \
  --output-step-csv datasets/hermes_pilot_h5_v2_observations_step.csv \
  --summary-md datasets/hermes_pilot_h5_v2_observations_summary.md

uv run python scripts/label_observation_shapes.py runs/hermes_pilot_h5_v2 \
  --csv datasets/hermes_pilot_h5_v2_shape_labels.csv \
  --report datasets/hermes_pilot_h5_v2_shape_report.md

uv run python scripts/build_estimator_checkpoints.py \
  --runs-dir runs/hermes_pilot_h5_v2 \
  --step-csv datasets/hermes_pilot_h5_v2_observations_step.csv \
  --shape-labels datasets/hermes_pilot_h5_v2_shape_labels.csv \
  --out-csv datasets/hermes_pilot_h5_v2_estimator_checkpoints.csv \
  --out-summary datasets/hermes_pilot_h5_v2_estimator_checkpoints_summary.md

uv run python scripts/build_q_labels.py \
  --runs-dir runs/hermes_pilot_h5_v2 \
  --checkpoint-csv datasets/hermes_pilot_h5_v2_estimator_checkpoints.csv \
  --out-csv datasets/hermes_pilot_h5_v2_q_labels.csv
```

## Files

| Artifact | Path |
|----------|------|
| Softened auto-annotator      | `scripts/auto_annotate_hermes.py` |
| HP6 invariants               | `tests/test_auto_annotate_hermes.py` |
| Pinned raw cache             | `external_data/hermes/pilot_cache_h5/` |
| Run dirs                     | `runs/hermes_pilot_h5_v2/hermes_pilot_h5_{001..030}/` |
| Observation CSVs             | `datasets/hermes_pilot_h5_v2_observations_*.csv` |
| Estimator checkpoints        | `datasets/hermes_pilot_h5_v2_estimator_checkpoints.csv` |
| Q labels                     | `datasets/hermes_pilot_h5_v2_q_labels.csv` |
| Shape labels                 | `datasets/hermes_pilot_h5_v2_shape_labels.csv` |
| This report                  | `runs/hermes_pilot_h5_v2/HP6_REPORT.md` |
