# Convex allocation experiment

Fractional allocation model for deciding whether in-flight session classes should
move by prompt replay, KV-state transfer, or stay during a retained-state
evacuation event. The main generated workload models active stateful sessions,
not stateless requests. The evacuation target is retained reconstruction work in
prefill seconds; CSVs and plots also report resident state in TB and fractions
of an NVIDIA GB200 NVL72 13.4 TB HBM rack.

The deadline-penalty CVXPY policy is the main rounded queue policy: the solver
keeps physical capacity hard and penalizes deadline overrun, while frontier
safety also requires rounded queue pressure and drain completion to stay within
the tested window. Plain CVXPY remains the fixed-load relaxed resource-cost
oracle. A second CVXPY LP maximizes retained prefill under hard
per-destination deadline-capacity constraints while enforcing the requested
retained-prefill target. `SolverResult.objective` is always the fixed-load
resource objective; solver-specific values live in diagnostics. Mirror descent
with scalar bisection is a preliminary first-order method for the fixed-load
objective.

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

Run the retained-state drain frontier sweep:

```bash
uv run python experiments/run_retained_state_frontier.py
```

Diagnose tight-deadline queue failures and rounded local repair:

```bash
uv run python experiments/run_queue_failure_diagnostics.py
```

Plot queue-centered results after the retained-state drain CSV exists:

```bash
uv run python experiments/plot_queue_centered.py
```

Plot the simple network-bandwidth pressure relationship:

```bash
uv run python experiments/run_network_bandwidth_tradeoff.py
```

Run tiny exhaustive integer comparisons:

```bash
uv run python experiments/run_integer_optimality_cases.py
```

Run the bounded H1 integer feasibility oracle:

```bash
uv run python experiments/run_h1_integer_oracle.py
```

Run the report-facing claim, rounding-gap, sensitivity, architecture, and
adversarial queue tables:

```bash
uv run python experiments/run_report_experiments.py
```

Long CVXPY sweeps run independent grid points in a shared process pool. Set
`CONVEX_ALLOCATION_WORKERS=1` for a sequential run, or to another positive
integer to cap worker processes.
Rounded queue metric paths use counted scheduling; trace-producing paths still
return per-request records for diagnostics and plots.

Outputs are written to `convex-allocation/outputs/sweep/`:
generated workload runs write the same current report/network artifacts to a
labeled subdirectory such as
`outputs/sweep/generated_seed7_sessions10000_classes48/`.

- `h1_fixed_target_stress.csv`
- `h1_integer_oracle.csv`
- `h1_integer_oracle_summary.csv`
- `h2_safe_frontier.csv`
- `h2_safe_frontier.pdf`
- `h2_delay_cdf.pdf`
- `h3_action_mix_by_model.csv`
- `h3_action_mix_by_model.pdf`
- `h4_state_manifest_heatmap.pdf`
- `retained_state_drain_sweep.csv`
- `retained_state_drain_frontier.csv`
- `network_bandwidth_tradeoff.csv`
- `network_bandwidth_tradeoff.pdf`
- `claim_table.csv`
- `rounding_gap_study.csv`
- `rounding_gap_summary.csv`
- `deadline_weight_sensitivity.csv`
- `model_architecture_sweep.csv`
- `adversarial_queue_case.csv`

`make_problem(...)` defaults to the generated 10k active-session workload
aggregated into 48 classes with seed 7. `workload_source="fixed"` keeps the
six-row smoke workload.
The retained-state drain frontier writes `retained_state_drain_sweep.csv` and
`retained_state_drain_frontier.csv`. It sweeps a dense drain-window grid from
10s to 3600s, uses that window as the resource-capacity budget, and reports the
max safe retained-prefill fraction evacuated using absolute event-start
deadline safety under EDF release order. Generated-workload runs use workload
seeds 0-15; H2 writes and plots independent drain-window frontier points for
all allocation policies with workload-seed mean and standard-deviation error
bars. Release ordering is applied as event-start service priority at counted
class/action block granularity for speed, which can make frontier cliffs sharper
than per-request ordering. Report plotting removes its owned H1-H4 artifacts
before regeneration so failed runs do not leave stale report outputs in place.
The network-bandwidth tradeoff sweeps network bandwidth and plots the largest
tested source-state fraction that can be safely evacuated, request migration
fraction, actual evacuated state TB, and max network/prefill queue depth. A
zero-retained baseline is included so fully unsafe generated sweeps report an
explicit zero frontier instead of all-NaN rows.
The H3 action-mix CSV and plot evaluate the single-request replay/state
crossover from context bytes per token, KV bytes per token, prefill rate,
network throughput, context length, and request count for each catalog model.
The report-facing experiment driver writes machine-checkable comparison rows,
rounding-gap rows for relaxed, exact, rounded, and repaired tiny cases,
deadline-weight sensitivity rows, a model architecture frontier table, and a
small adversarial rounding case.
Queue diagnostics bound the local repair search and reuse prefixes of the full
repair path for budget rows so generated-workload repairs finish predictably.
Legacy scripts still write their own diagnostic CSVs when run. Generated
diagnostic artifacts under `outputs/sweep/generated_*` are committed and should
be regenerated with the matching workload label before report use.

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
