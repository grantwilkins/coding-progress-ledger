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
  --solver load_only --out queue-haul/outputs/profile_smoke
```

The earlier additive model is frozen in `_archive/queue-haul-additive-v0`.
