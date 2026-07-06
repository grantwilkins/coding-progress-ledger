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

Stage 1a uses `powertrace-sim`'s vLLM probe stack. For the single-A100
gpt-oss-20b TP=1 collection, write the runbook with:

```bash
uv run python queue-haul/stage1_curves.py \
  --model openai/gpt-oss-20b \
  --hardware A100 \
  --tp 1 \
  --gpus-per-node 1 \
  --max-model-len 32768 \
  --prefill-lens 256 1024 4096 16384 \
  --run-id gpt-oss-20b-a100-tp1 \
  -- --async-scheduling
```

Without `--execute`, it writes `queue-haul/runs/stage1/<run_id>/commands.sh`
without launching GPUs. Run it with `APP='apptainer exec --nv --bind $SCRATCH
<sandbox>'` when the vLLM Apptainer image is needed.

After collection, refresh the powertrace fit outputs and the Queue-Haul
`ell`-vs-power plot:

```bash
(cd ../../powertrace-sim && uv run python scripts/eval/two_price_fit.py --configs gpt-oss-20b-a100 && uv run python scripts/eval/saturating_fit.py)
uv run python queue-haul/stage1_profile.py
```

The Queue-Haul reducer writes:

- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_curve.csv`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_constants.csv`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1.{pdf,png}`
