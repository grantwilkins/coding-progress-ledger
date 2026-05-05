# coding-estimator

Belief-state layer over `../coding-progress-ledger`. Reads append-only ledger
histories at checkpoint `t` and outputs calibrated probabilities for eventual
success, success-by-horizon, remaining time given success, and near-future
progress-dynamics events.

## v0 boundary (current scientific state)

v0 has established a measurement boundary, not a deployable estimator.

- **Positive result.** Prefix-only ledger features predict near-future
  process dynamics — in particular, progress drops within a short horizon
  (`y_future_progress_drop_h5`) substantially better than elapsed time
  (G4 Brier 0.039 vs G2 0.142 on `swe_agent_pilot` LORO; AUROC 0.977).
- **Negative result.** Prefix-only ledger features do **not** yet improve
  terminal-success prediction (`y_success_eventual`) over elapsed time at
  the current sample size.
- **Stamp.** `not_safe_for_control = true` remains on the v0 estimator.
  The v0 verdict is `indeterminate`, driven by data gaps — not leakage or
  code defects.

The next phase is **targeted data work**, not modeling: annotate the
existing Hermes retrospective corpus, collect an outcome-diverse
`tb_live_v2` (≥100 runs, ≥25 failures), then re-test completion risk.

Full evidence:
[reports/V0_FINDINGS.md](reports/V0_FINDINGS.md) ·
[reports/REVIEWER_BRIEFING.md](reports/REVIEWER_BRIEFING.md) ·
[reports/ESTIMATOR_GO_NO_GO.md](reports/ESTIMATOR_GO_NO_GO.md) ·
[reports/NOT_READY_FOR_SCHEDULING.md](reports/NOT_READY_FOR_SCHEDULING.md).

## What this is
- A **read-only consumer** of ledger artifacts (`runs/<source>/<run_id>/ledger.jsonl`).
- A schema layer + small calibrated model ladder (logreg, GBM later).
- A leakage- and calibration-first project.
- A measurement instrument for the observation channel; **headline target
  is process dynamics**, not terminal success.

## What this is not
- Not a redefinition of progress (that lives upstream).
- Not a controller: no `pause`, `stop`, `throttle`, no policy outputs.
- Not a large neural sequence model in v0.
- Not safe for scheduling, throttling, cost control, or any consumer that
  reads completion probabilities as actionable risk estimates.

## Quickstart
```bash
uv sync
uv run pytest -q
uv run python -c "import coding_estimator"
```

See [TASKS.md](TASKS.md) for the v1 backlog (Workstreams S–Y). The
immediate priority is Workstream T (Hermes annotation unblock) and
Workstream U (`tb_live_v2` collection).
