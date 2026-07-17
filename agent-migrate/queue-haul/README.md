# Queue-Haul

Queue-Haul models and measures session migration under a local source-site
power limit. The active path is:

```text
profiles.py → planner.py → simulate.py → power_drain_experiment.py
```

`stage1*.py` collects and reduces model, service, migration, and power data.
`profiles/*.json` records the measured range and uncertainty used by the
simulator. Run all commands from `agent-migrate`:

```bash
uv run pytest
uv run python queue-haul/power_drain_experiment.py \
  --workload-profile queue-haul/profiles/agentic_tool_loop.json \
  --sessions 6 --seed 3 --power-limit 500 --deadline 5 --end 5 \
  --solver load_only --workers 2 --out queue-haul/outputs/profile_smoke
uv run python queue-haul/plot_simulator_validation.py
uv run python queue-haul/plot_simulator_evaluation.py
uv run python queue-haul/plot_scaling_results.py
```

The model profile uses exact measured KV bytes. KV loading overlaps network
transfer, so serial KV time is setup plus the slower of network transfer and
destination KV loading, followed by synchronization and route switching.
Destination KV copies enter a FIFO per destination before moving bytes;
`queues.csv` records arrival, start, completion, depth, bytes, observed wait,
and whether a copy is still pending at the simulation cutoff.

For active sessions with `--final-state awake`, `node_drain` ranks source nodes
by exact power reduction to idle divided by predicted drain time. It then ranks
sessions within each node by power reduction per resource use and reserves the
same source, network, destination replay/KV, compute, KV residency, and trailing
power-window capacities as the LP. It empties a node when possible and otherwise
takes only the sessions that fit. Cold-session, sleep, and shutdown plans retain
the simpler whole-node ordering until those transitions are measured.

`--solver lp` jointly selects replay and KV transfer under source-instance,
network, destination replay, destination KV, compute, residency, and source
power limits. It minimizes power shortfall, then peak resource use, then total
migration work. The current LP scope is active sessions, one destination pool,
the central profile, and `--final-state awake`; unsupported cases hard-fail.
The fractional plan is rounded to whole sessions and accepted only after the
existing discrete-event simulator checks its commits and trailing-window power.
The exact equations and conservative concave-power bound are in
`queue-haul/formulation.md`.
Action power is stored as total added power for each measured concurrency, not
as power per session. Fit the serial coding data with repeats 0–1 and evaluate
repeat 2 with:

```bash
uv run python queue-haul/stage1c_profile_fit.py \
  --run-root queue-haul/outputs/coding-run \
  --profile queue-haul/profiles/gpt_oss_20b_a100_tp1.json
```

The checked profile remains `estimated`. It has not been validated for larger
catch-up work, interactive or agentic jobs, eight-session drains, sleep or
shutdown, paired method and bandwidth comparisons, or parallel KV connections.
The new coding fit and the earlier live replay points occupy separate token
ranges in the profile; the earlier range retains its 30% error bound.

Stage 1C reduction reports measured prompt, processed, and new tokens; initial
KV payload bytes; catch-up cache hits; exact proxy KV-route bytes; request
timing; and power relative to a measured idle baseline. It does not group or
plot by requested context size.
`initial_time`, `throughput`, `concurrency_scaling`, `service_effects`,
`power_energy`, and `model_check` show the direct relationships. Concurrency
comparisons are paired; method and bandwidth observations are not paired and
must not be connected or interpreted as isolated effects.

`--workers` runs independent workload, power-limit, deadline, and solver groups
in separate processes while preserving serial result order. It defaults to one
so batch allocations are never oversubscribed implicitly. Planner predictions
skip audit records; experiment executions retain complete evidence tables.

Sessions have two states. Active sessions have GPU-resident KV and use eager
replay or KV transfer. Cold sessions have no retained KV, consume no serving
load, and use replay on request. The planner and simulator reject mismatched
methods. Legacy `idle` inputs are loaded as cold.

Serving instances are sized by both measured compute load (`max_ell`) and
engine-reported resident KV-token capacity. Active sessions count against both;
cold sessions count against neither. The GPT-OSS-20B A100 profile uses the
smaller source/sink vLLM capacity of 1,214,544 tokens per TP=1 instance.

The earlier additive model is frozen in `_archive/queue-haul-additive-v0`.

`outputs/simulator_validation.{csv,png,pdf}` compares a two-session simulation
with exact transfer, route-switch, and source-power calculations. It checks that
equal transfers share the link and source power falls only after both routes
switch. It does not validate A100 timing or power calibration. Generation
hard-fails if any checked value differs.

`outputs/simulator_evaluation.{csv,png,pdf}` shows requested versus simulated
source-power reduction, route-switch completion, and request wait for a
controlled 50-session sweep.

`outputs/scaling_1_to_100k_20260716/scaling_summary.{png,pdf}` compares
solver choices, deadline completion, simulated power reduction, migration
completion, and planning time on the paired coding sweep.
