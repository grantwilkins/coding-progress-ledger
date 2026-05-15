# Convex allocation experiment

Fractional allocation model for deciding whether in-flight session classes should
move by prompt replay, KV-state transfer, or stay during a source load-shed
event.

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
- `crossover_recovery.png`
- `summary.csv`

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
