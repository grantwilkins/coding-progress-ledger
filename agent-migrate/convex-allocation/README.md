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

Run the safe-shed frontier sweep:

```bash
uv run python experiments/run_safe_shed_frontier.py
```

Diagnose tight-slack queue failures and rounded local repair:

```bash
uv run python experiments/run_queue_failure_diagnostics.py
```

Outputs are written to `convex-allocation/outputs/sweep/`:

- `headline_action_mix.png`
- `allocation_heatmap_per_scenario.png`
- `utilization_vs_policy.png`
- `objective_vs_policy.png`
- `convergence_one_scenario.png`
- `crossover_recovery.png`
- `summary.csv`
- `transition_coupled_policy_table.csv`
- `transition_coupled_allocation_summary.csv`
- `transition_coupled_queue_table.csv`
- `shed_slack_sweep.csv`
- `safe_shed_frontier.csv`
- `transition_coupled_queue_failure_breakdown.csv`
- `transition_coupled_repaired_queue_table.csv`

`summary.csv` marks infeasible baselines as `INFEASIBLE` and includes shed
target, shed violation, excess shed, capacity feasibility, mirror-descent
objective gap, selected scalar shed multiplier `alpha`, and bisection count.
The transition-coupled CSVs compare CVXPY, mirror descent, crossover-greedy,
mixed-greedy, replay-only, and state-only on the GLM-5 4/6/9 Gbps stress case.
The queue table rounds fractional allocations into requests and reports static
network-then-prefill EDF reconstruction delay metrics.
The safe-shed frontier sweep uses the same GLM-5 transition-coupled scenario and
marks a rounded policy safe only when it meets the shed target, has deadline miss
rate at most 1%, and has p95 delay divided by class slack at most 1.0.
The failure diagnostic adds per-request queue tracing, missed-request breakdowns
by class, destination, and action, and a one-request rounded local repair for the
tight 0.25x and 0.5x slack settings.

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
