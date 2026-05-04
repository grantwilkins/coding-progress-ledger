# Checkpoint policy

A *checkpoint* is one row in the estimator dataset, indexed by
`(run_id, t)` where `t` is a step or wall-clock seconds. The policy
decides which `t` values get a row.

## v0 default: `P_step`

One checkpoint per distinct ledger step, contiguous from the run's
first step through its terminal step. This mirrors upstream
`build_estimator_checkpoints.py` exactly so the v0 dataset is parity-
checkable against upstream.

## Documented alternatives (not implemented in v0)

| name | description | when to revisit |
|---|---|---|
| `P_event` | one checkpoint per event (multiple per step possible) | if intra-step ordering becomes load-bearing |
| `P_kstep` | every k-th step | if step density varies wildly across sources and we want a uniform stride |
| `P_wallclock_grid` | every Δ seconds of wallclock | when wallclock is real on most sources (currently only `tb_live`) |
| `P_terminal_only` | one checkpoint at the terminal step | as a sanity-check baseline |

The enum lives in `coding_estimator.checkpoints.policy.CheckpointPolicy`.
Calling `checkpoint_steps(run, policy)` with any non-`P_step` policy
raises `NotImplementedError` in v0.

## Per-run guarantees for `P_step`

- The returned step list is **strictly increasing**.
- The list is contiguous: `[s_min, s_min+1, ..., s_max]`.
- The terminal step (`max(e.step for e in run.events)`) is always present
  as the final element. The first element is the smallest step in the
  run (typically `0`).
- A run with zero events raises `ValueError`. We do not invent a
  checkpoint where no run history exists.
