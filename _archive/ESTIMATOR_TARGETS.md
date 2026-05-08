# Estimator targets (v0)

The v0 estimator ships **four** prediction targets. Everything else is
deferred to § B2.bis until the dataset budget can support honest
evaluation. The machine-readable form lives in
`coding_estimator/labels/registry.py` and is validated against
`schemas/label_schema.json`.

## v0 headline targets

### 1. `y_success_eventual` — terminal, run-constant
Binary; 1 iff the run's final verdict is success. Replicated across every
checkpoint of the run. Run-constant; **most likely to be uninformative
beyond elapsed-time**. Its job is to anchor the no-regression gate.

- Source: `run_manifest.json::final_success`
- Mask rule: never mask.
- Upstream Q1 id: none.
- Estimated base rate: ~0.50 (mixed sources).

### 2. `y_future_progress_drop_h5` — horizon = 5 steps
Binary; 1 iff there exists `s ∈ (t, t+5]` where overall progress decreases
relative to its value at `s-1`. Equivalent to upstream Q1
`future_progress_drop`. Highest positive rate of the Q1 family
(~0.30 at N=20) — the **best-supported v0 target**.

- Source: ledger.jsonl progress trajectory.
- Mask rule: mask if `t + 5 > finish_step` or `is_terminal_checkpoint`.
- Upstream Q1 id: `future_progress_drop`.

### 3. `y_validation_new_work_h5` — horizon = 5 steps
Binary; 1 iff a validation event in `(t, t+5]` introduces a new product
leaf or reopens a completed one. Equivalent to upstream Q1
`validation_exposes_new_work`. Low positive rate (~0.02) but tied to the
validation pillar — **kept as diagnostic**.

- Source: ledger.jsonl validation + add/reopen events.
- Mask rule: mask if `t + 5 > finish_step` or `is_terminal_checkpoint`.
- Upstream Q1 id: `validation_exposes_new_work`.

### 4. `y_submit_without_validation` — terminal, run-constant
Binary; 1 iff the run terminated with a submitted artifact but never had
any validation events. Equivalent to upstream Q1
`submit_without_validation_state`. **EXPLICITLY a calibration sanity
target**: it is run-constant, so any high score at non-terminal `t` is a
data property of the run distribution, not estimator skill.

- Source: ledger.jsonl + run_manifest.json.
- Mask rule: never mask.
- Upstream Q1 id: `submit_without_validation_state`.
- Run-constant: yes.

## Shape labels are NOT prediction targets in v0

Upstream `label_observation_shapes.py` produces post-hoc, run-level
descriptors. Predicting them at non-terminal `t` is a leakage hazard
(essentially predicting `final_success` with extra steps). In v0 they are
used for profiling, evaluation slicing, and case studies only.

## Deferred (B2.bis) — re-evaluate at N > 100 runs

Listed here so they aren't lost. Full enumeration in
`coding_estimator.labels.registry.DEFERRED_TARGETS`.

```
y_success_by_h_steps_{5,10,25,50}, y_success_by_timeout
y_success_by_h_seconds_{300,900,1800,runtimeout}
y_remaining_steps_if_success, y_remaining_seconds_if_success
y_finish_step, y_finish_seconds
y_product_reopen_h5, y_stuck_loop_h5
y_blocked_within_h5, y_new_scope_within_h5, y_validation_failure_within_h5
```
