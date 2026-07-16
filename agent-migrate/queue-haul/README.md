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
```

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
