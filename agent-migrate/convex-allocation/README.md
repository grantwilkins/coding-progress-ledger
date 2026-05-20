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
uv run python experiments/plot_queue_centered.py --workload-source fixed
```

Plot the simple network-bandwidth pressure relationship:

```bash
uv run python experiments/run_network_bandwidth_tradeoff.py
```

Run tiny exhaustive integer comparisons:

```bash
uv run python experiments/run_integer_optimality_cases.py
```

Run the report-facing claim, rounding-gap, sensitivity, architecture, and
adversarial queue tables:

```bash
uv run python experiments/run_report_experiments.py
```

Long CVXPY sweeps run independent grid points in a process pool. Set
`CONVEX_ALLOCATION_WORKERS=1` for a sequential run, or to another positive
integer to cap worker processes.

Outputs are written to `convex-allocation/outputs/sweep/`:
generated workload runs write the same current report/network artifacts to a
labeled subdirectory such as
`outputs/sweep/generated_seed7_sessions10000_classes48/`.

- `h1_resource_pressure.pdf`
- `h2_safe_frontier.pdf`
- `h2_delay_cdf.pdf`
- `h3_action_mix_by_model.pdf`
- `h4_state_manifest_heatmap.pdf`
- `network_bandwidth_tradeoff.csv`
- `network_bandwidth_tradeoff.pdf`
- `claim_table.csv`
- `rounding_gap_study.csv`
- `rounding_gap_summary.csv`
- `deadline_weight_sensitivity.csv`
- `model_architecture_sweep.csv`
- `adversarial_queue_case.csv`

`make_problem(...)` defaults to the generated 10k active-session workload
aggregated into 48 classes. `workload_source="fixed"` keeps the six-row smoke
workload.
The network-bandwidth tradeoff sweeps network bandwidth and plots the largest
tested source-state fraction that can be safely evacuated, request migration
fraction, actual evacuated state TB, and max network/prefill queue depth.
The report-facing experiment driver writes machine-checkable claim rows,
rounding-gap rows for relaxed, exact, rounded, and repaired tiny cases,
deadline-weight sensitivity rows, a model architecture frontier table, and a
small adversarial rounding case.
Legacy scripts still write their own diagnostic CSVs when run, but those stale
generated CSV artifacts are no longer committed.

The catalog is local and hard-coded from `kv-transfer-early-experiment/FINDINGS.md`.
It does not import that directory.
