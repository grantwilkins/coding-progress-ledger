# agent-migrate-agent

Research experiments for LLM state migration, evacuation, and power-shed planning.

The active thread is `queue-haul`: node-aware power shedding for active LLM jobs. The current claim is that modeled source-node power shed must be evaluated through the ramp-then-plateau node curve, not only through additive per-job active-work certificates.

## Layout

- `queue-haul/` - current node-knee power-shed model, tests, findings, and canonical plots.
- `evacuation/` - staged evacuation and fairness experiments with an EE364B write-up.
- `kv-transfer-early-experiment/` - early context replay versus KV-transfer calculations and plots.

`convex-allocation/` has been removed from the active tree.

## Run

```bash
uv sync
uv run pytest
```

## Queue-Haul Plots

Run from `queue-haul/`:

```bash
uv run python plot_node_knee_target_sweep.py
uv run python plot_node_knee_deadline_sweep.py
uv run python plot_node_knee_execution_validation.py
uv run python plot_node_knee_agentic_des_sweep.py
uv run python plot_node_knee_scale_workload_sweep.py
uv run python plot_node_knee_kappa_sweep.py
```

Canonical outputs are:

- `outputs/node_knee_target_sweep.{csv,pdf,png}`
- `outputs/node_knee_deadline_sweep.{csv,pdf,png}`
- `outputs/node_knee_execution_validation.{csv,pdf,png}`
- `outputs/node_knee_fixed_plan_replay.{csv,pdf,png}`
- `outputs/node_knee_agentic_des_sweep.{csv,pdf,png}`
- `outputs/node_knee_scale_workload_sweep.{csv,pdf,png}`
- `outputs/node_knee_kappa_sweep.{csv,pdf,png}`

See `queue-haul/FINDINGS.md` for the current result summary.

## Queue-Haul Stage 1a Curves

Stage 1a uses `powertrace-sim`'s existing vLLM probe stack. The local wrapper only
builds a small runbook for decode, prefill, and mixed-grid curve probes:

```bash
uv run python queue-haul/stage1_curves.py \
  --model openai/gpt-oss-120b \
  --tp 8 \
  --max-model-len 65536 \
  --execute \
  -- --trust-remote-code
```

Without `--execute`, it writes `queue-haul/runs/stage1/<run_id>/commands.sh`
without launching GPUs.
