# Convex allocation experiment

Fractional allocation model for deciding whether in-flight session classes should
move by prompt replay, KV-state transfer, or stay during a source relief event.
Relief is source prefill seconds evacuated by migration; deadline slack is now
represented as explicit soft deadline debt rather than hidden in `R0 / slack`.

CVXPY provides the fixed-relief relaxed oracle. The soft-deadline CVXPY policy
keeps physical capacity hard and penalizes deadline debt. A second CVXPY LP
maximizes relief under hard per-destination deadline-capacity constraints at
deadline margins 0.8 and 1.0. Mirror descent with scalar bisection is a
preliminary first-order method for the fixed-relief objective.

Run tests:

```bash
uv run pytest
```

Run the sweep:

```bash
cd convex-allocation
uv run python experiments/run_catalog_sweep.py
```

Run the generated workload sweep:

```bash
uv run python experiments/run_catalog_sweep.py --workload-source generated --workload-seed 7
```

Run the safe-shed frontier sweep:

```bash
uv run python experiments/run_safe_shed_frontier.py
```

Diagnose tight-slack queue failures and rounded local repair:

```bash
uv run python experiments/run_queue_failure_diagnostics.py
```

Plot queue-centered results after the safe-shed and queue-failure CSVs exist:

```bash
uv run python experiments/plot_queue_centered.py
```

Plot the simple network-bandwidth pressure relationship:

```bash
uv run python experiments/run_relief_pressure_tradeoff.py
```

Outputs are written to `convex-allocation/outputs/sweep/`:
generated workload runs write to a labeled subdirectory such as
`outputs/sweep/generated_seed7_jobs1000_classes12/`.

- `summary.csv`
- `transition_coupled_policy_table.csv`
- `transition_coupled_allocation_summary.csv`
- `transition_coupled_queue_table.csv`
- `safe_shed_frontier_lines.pdf`
- `miss_rate_frontier_lines.pdf`
- `delay_cdf_hard_case.pdf`
- `queue_depth_hard_case.pdf`
- `resource_pressure_scatter.pdf`
- `relief_pressure_tradeoff.csv`
- `relief_pressure_tradeoff.pdf`
- `shed_slack_sweep.csv`
- `safe_shed_frontier.csv`
- `transition_coupled_queue_failure_breakdown.csv`
- `transition_coupled_repaired_queue_table.csv`
- `repair_summary.csv`
- `repair_move_breakdown.csv`
- `repair_budget_frontier.csv`

`summary.csv` marks infeasible baselines as `INFEASIBLE` and includes relief
target, relief violation, excess relief, capacity feasibility, deadline debt,
mirror-descent objective gap, selected scalar relief multiplier `alpha`, and
bisection count. The transition-coupled CSVs compare fixed-relief CVXPY,
soft-deadline CVXPY, deadline-aware CVXPY, mirror descent, crossover-greedy,
mixed-greedy, replay-only, and state-only on the GLM-5 4/6/9 Gbps stress case.
`make_problem(..., workload_source="fixed")` keeps the default six-row workload.
`workload_source="generated"` opts into a seeded relief-event batch generator
with long-context tails, deadline variation, and fixed destination cache-locality
snapshots. It aggregates jobs into capped classes and keeps `ProblemData`
unchanged.
The queue table rounds fractional allocations into requests and reports static
network-then-prefill EDF reconstruction delay metrics.
The safe-shed frontier sweep uses the same GLM-5 transition-coupled scenario and
marks a rounded policy safe only when it meets the relief target, has deadline
miss rate at most 1%, and has p95 delay divided by class deadline at most 1.0.
It also reports max network/prefill busy windows, replay/state relief shares,
and soft-deadline debt at the frontier. The local repair oracle is included only
as a post-hoc upper-bound diagnostic.
The failure diagnostic adds per-request queue tracing, missed-request breakdowns
by class, destination, and action, and a one-request rounded local repair for the
tight 0.25x and 0.5x deadline settings.
It also reports how much repair changes the rounded convex allocation, the move
patterns used by repair, and 5%, 10%, and 20% repair-budget frontiers.
The relief-pressure tradeoff sweeps network bandwidth scale and plots the
largest queue-safe relief fraction, evacuated request fraction, and max
network/prefill waiting queue depth.

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
