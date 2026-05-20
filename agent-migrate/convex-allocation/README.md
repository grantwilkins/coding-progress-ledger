# Convex allocation experiment

Fractional allocation model for deciding whether in-flight session classes should
move by prompt replay, KV-state transfer, or stay during a retained-state
evacuation event. The main generated workload models active stateful sessions,
not stateless requests. The evacuation target is retained reconstruction work in
prefill seconds; CSVs and plots also report resident state in TB and fractions
of an NVIDIA GB200 NVL72 13.4 TB HBM rack.

The deadline-penalty CVXPY policy is the main rounded queue policy: it keeps
physical capacity hard and penalizes deadline overrun. Plain CVXPY remains the
fixed-load relaxed resource-cost oracle. A second CVXPY LP
maximizes retained prefill under hard per-destination deadline-capacity
constraints at deadline margins 0.8 and 1.0. Mirror descent with scalar
bisection is a preliminary first-order method for the fixed-load objective.

The retained-session workload is motivated by public stateful-serving evidence:
Mooncake's KVCache-centric serving design, vLLM/Mooncake agentic traces with
long multi-turn contexts, KVCache-in-the-Wild reuse traces, Continuum-style KV
retention across agent pauses, and NVIDIA's GB200 NVL72 HBM reference:
https://arxiv.org/abs/2407.00079,
https://vllm.ai/blog/2026-05-06-mooncake-store,
https://arxiv.org/abs/2506.02634,
https://arxiv.org/abs/2511.02230,
https://www.nvidia.com/en-sg/data-center/gb200-nvl72/.

Run tests:

```bash
uv run pytest
```

Run the sweep:

```bash
cd convex-allocation
uv run python experiments/run_catalog_sweep.py
```

Run the fixed six-class smoke sweep:

```bash
uv run python experiments/run_catalog_sweep.py --workload-source fixed
```

Run the retained-state frontier sweep:

```bash
uv run python experiments/run_retained_state_frontier.py
```

Diagnose tight-deadline queue failures and rounded local repair:

```bash
uv run python experiments/run_queue_failure_diagnostics.py
```

Plot queue-centered results after the retained-state frontier CSV exists:

```bash
uv run python experiments/plot_queue_centered.py
```

Plot the simple network-bandwidth pressure relationship:

```bash
uv run python experiments/run_network_bandwidth_tradeoff.py
```

Outputs are written to `convex-allocation/outputs/sweep/`:
generated workload runs write to a labeled subdirectory such as
`outputs/sweep/generated_seed7_sessions10000_classes48/`.

- `summary.csv`
- `transition_coupled_policy_table.csv`
- `transition_coupled_allocation_summary.csv`
- `transition_coupled_queue_table.csv`
- `retained_state_frontier.pdf`
- `deadline_miss_frontier.pdf`
- `deadline_delay_cdf.pdf`
- `queue_depth_example.pdf`
- `network_prefill_busy_scatter.pdf`
- `network_bandwidth_tradeoff.csv`
- `network_bandwidth_tradeoff.pdf`
- `retained_state_deadline_sweep.csv`
- `retained_state_frontier.csv`
- `transition_coupled_queue_failure_breakdown.csv`
- `transition_coupled_repaired_queue_table.csv`
- `repair_summary.csv`
- `repair_move_breakdown.csv`
- `repair_budget_frontier.csv`

`summary.csv` marks infeasible baselines as `INFEASIBLE` and includes retained
prefill target, retained prefill moved, resident-state TB, NVL72 HBM fraction,
capacity feasibility, deadline overrun, mirror-descent objective gap, selected
scalar load multiplier `alpha`, and bisection count. The transition-coupled CSVs compare fixed-load CVXPY,
deadline-penalty CVXPY, mirror descent, crossover-greedy, mixed-greedy,
replay-only, and state-only on the GLM-5 4/6/9 Gbps stress case.
`make_problem(...)` defaults to the generated 10k active-session workload
aggregated into 48 classes. `workload_source="fixed"` keeps the six-row smoke
workload.
The queue table rounds fractional allocations into requests and reports
network-then-prefill EDF reconstruction delay metrics under the default 30m
drain.
The retained-state frontier sweep marks a rounded policy safe only when it meets
the retained prefill target, has deadline miss rate at most 1%, and has p95 delay
divided by class deadline at most 1.0. It drains moved sessions with
deterministic EDF pacing over 30m for main plots and also writes burst/15m/60m
sensitivity rows. It reports retained-prefill fraction, average-equivalent
state target TB, actual evacuated state TB, network/prefill capacity pressure,
replay/state-transfer retained-prefill shares, drain completion time, and
deadline overrun. Main plots lead with deadline-penalty CVXPY and include plain
CVXPY as the resource-cost oracle, plus mirror descent, crossover-greedy,
replay-only, and state-transfer-only.
The failure diagnostic adds per-request queue tracing, missed-request breakdowns
by class, destination, and action, and a one-request rounded local repair for the
tight 0.25x and 0.5x deadline settings.
It also reports how much repair changes the rounded convex allocation, the move
patterns used by repair, and 5%, 10%, and 20% repair-budget frontiers.
The network-bandwidth tradeoff sweeps network bandwidth and plots the largest
queue-safe retained-prefill fraction, request migration fraction, actual
evacuated state TB, and max network/prefill queue depth.

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
