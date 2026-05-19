# Convex allocation experiment

Fractional allocation model for deciding whether in-flight session classes should
move by prompt replay, KV-state transfer, or stay during a source load-reduction
event. Source reduction is measured in source-prefill seconds removed from the
overloaded source. The grid drain clock controls when jobs are selected for
migration; each selected request gets its own release-relative reconstruction
deadline.

CVXPY provides the fixed-load relaxed oracle. The deadline-penalty CVXPY policy
keeps physical capacity hard and penalizes deadline overrun. A second CVXPY LP
maximizes source-prefill moved under hard per-destination deadline-capacity
constraints at deadline margins 0.8 and 1.0. Mirror descent with scalar
bisection is a preliminary first-order method for the fixed-load objective.

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

Run the source-prefill frontier sweep:

```bash
uv run python experiments/run_source_load_frontier.py
```

Diagnose tight-deadline queue failures and rounded local repair:

```bash
uv run python experiments/run_queue_failure_diagnostics.py
```

Plot queue-centered results after the source-prefill frontier CSV exists:

```bash
uv run python experiments/plot_queue_centered.py
```

Plot the simple network-bandwidth pressure relationship:

```bash
uv run python experiments/run_network_bandwidth_tradeoff.py
```

Outputs are written to `convex-allocation/outputs/sweep/`:
generated workload runs write to a labeled subdirectory such as
`outputs/sweep/generated_seed7_jobs1000_classes12/`.

- `summary.csv`
- `transition_coupled_policy_table.csv`
- `transition_coupled_allocation_summary.csv`
- `transition_coupled_queue_table.csv`
- `source_load_frontier.pdf`
- `deadline_miss_frontier.pdf`
- `deadline_delay_cdf.pdf`
- `queue_depth_example.pdf`
- `network_prefill_busy_scatter.pdf`
- `network_bandwidth_tradeoff.csv`
- `network_bandwidth_tradeoff.pdf`
- `source_load_deadline_sweep.csv`
- `source_load_frontier.csv`
- `transition_coupled_queue_failure_breakdown.csv`
- `transition_coupled_repaired_queue_table.csv`
- `repair_summary.csv`
- `repair_move_breakdown.csv`
- `repair_budget_frontier.csv`

`summary.csv` marks infeasible baselines as `INFEASIBLE` and includes source
prefill target, source prefill moved, capacity feasibility, deadline overrun,
mirror-descent objective gap, selected scalar load multiplier `alpha`, and
bisection count. The transition-coupled CSVs compare fixed-load CVXPY,
deadline-penalty CVXPY, deadline-aware CVXPY, mirror descent, crossover-greedy,
mixed-greedy, replay-only, and state-only on the GLM-5 4/6/9 Gbps stress case.
`make_problem(..., workload_source="fixed")` keeps the default six-row workload.
`workload_source="generated"` opts into a seeded load-reduction batch generator
with long-context tails, deadline variation, and fixed destination cache-locality
snapshots. It aggregates jobs into capped classes and keeps `ProblemData`
unchanged.
The queue table rounds fractional allocations into requests and reports
network-then-prefill EDF reconstruction delay metrics under the default 30m
drain.
The source-prefill frontier sweep uses the same GLM-5 transition-coupled scenario
and marks a rounded policy safe only when it meets the source-prefill target, has
deadline miss rate at most 1%, and has p95 delay divided by class deadline at
most 1.0. It drains moved requests with deterministic EDF pacing over 30m for
main plots and also writes burst/15m/60m sensitivity rows. It reports deadline
scale, source-prefill fraction, network/prefill capacity pressure,
replay/state-transfer source-prefill shares, drain completion time, and deadline
overrun at the frontier. Main plots show CVXPY, deadline-penalty CVXPY, mirror
descent, crossover-greedy, replay-only, and state-transfer-only. The hard
deadline-cap LPs and local repair oracle stay in CSVs as reference diagnostics.
The failure diagnostic adds per-request queue tracing, missed-request breakdowns
by class, destination, and action, and a one-request rounded local repair for the
tight 0.25x and 0.5x deadline settings.
It also reports how much repair changes the rounded convex allocation, the move
patterns used by repair, and 5%, 10%, and 20% repair-budget frontiers.
The network-bandwidth tradeoff sweeps network bandwidth and plots the largest
queue-safe source-prefill fraction, request migration fraction, and max
network/prefill queue depth.

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
