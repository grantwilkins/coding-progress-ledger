# coding-estimator

Belief-state layer over `../coding-progress-ledger`. Reads append-only ledger
histories at checkpoint `t` and outputs calibrated probabilities for eventual
success, success-by-horizon, remaining time given success, and near-future
progress-dynamics events.

## What this is
- A **read-only consumer** of ledger artifacts (`runs/<source>/<run_id>/ledger.jsonl`).
- A schema layer + small calibrated model ladder (logreg, GBM later).
- A leakage- and calibration-first project.

## What this is not
- Not a redefinition of progress (that lives upstream).
- Not a controller: no `pause`, `stop`, `throttle`, no policy outputs.
- Not a large neural sequence model in v0.

## Quickstart
```bash
uv sync
uv run pytest -q
uv run python -c "import coding_estimator"
```

The smoke pipeline lands once Workstreams C–G are implemented (see TASKS.md).
Until then, schema validation tests are the only end-to-end signal.
