# Convex allocation experiment

Fractional allocation model for deciding whether in-flight session classes should
move by prompt replay, KV-state transfer, or stay during a source load-shed
event.

CVXPY is the relaxed oracle used for result claims. Mirror descent with scalar
bisection is a preliminary first-order method that exploits the per-class
simplex structure; its objective gaps are reported only for feasible iterates
and are accompanied by shed, capacity, and replay/state diagnostics.

Run tests:

```bash
uv run pytest
```

Run the sweep:

```bash
cd convex-allocation
uv run python experiments/run_catalog_sweep.py
```

Outputs are written to `convex-allocation/outputs/sweep/`:

- `headline_action_mix.png`
- `allocation_heatmap_per_scenario.png`
- `utilization_vs_policy.png`
- `objective_vs_policy.png`
- `convergence_one_scenario.png`
- `crossover_recovery.png`
- `summary.csv`

`summary.csv` marks infeasible baselines as `INFEASIBLE` and includes shed
target, shed violation, excess shed, capacity feasibility, mirror-descent
objective gap, selected scalar shed multiplier `alpha`, and bisection count.

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
