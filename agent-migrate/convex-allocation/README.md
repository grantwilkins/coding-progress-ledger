# Convex allocation experiment

Fractional allocation model for deciding whether in-flight session classes should
move by prompt replay, KV-state transfer, or stay during a source load-shed
event.

CVXPY is the oracle used for result claims. Mirror descent is a preliminary
first-order diagnostic; its objective gaps are reported only for feasible
iterates and are accompanied by shed and capacity diagnostics.

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
- `convergence_grid.png`
- `convergence_grid.pdf`
- `convergence_semilog_feasibility_gap.png`
- `convergence_semilog_feasibility_gap.pdf`
- `crossover_recovery.png`
- `summary.csv`

`summary.csv` marks infeasible baselines as `INFEASIBLE` and includes shed
target, shed violation, excess shed, capacity feasibility, mirror-descent step
sizes, and final dual value.

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
