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

On the A100 cluster node used for the source-sink proof, `uv` is not on PATH;
load the local runtime and run pytest directly:

```bash
module load gcc/14.2.0 openblas/0.3.28
.venv/bin/python -m pytest
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
`ell`-vs-power plot. The reducer recomputes `ell = f/F + g/G` from the raw
window rates and writes a sampled concave node-power curve from the saturating
fit constants.

```bash
(cd ../../powertrace-sim && uv run python scripts/eval/two_price_fit.py --configs gpt-oss-20b-a100 && uv run python scripts/eval/saturating_fit.py)
uv run python queue-haul/stage1_profile.py
uv run python queue-haul/stage1_window_sensitivity.py
```

The Queue-Haul reducer writes:

- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_curve.csv`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_power_curve.csv`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_constants.csv`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1.{pdf,png}`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_window_sensitivity_summary.csv`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_window_sensitivity_binned.csv`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_window_sensitivity.{pdf,png}`

## Queue-Haul Stage 1a Service Surface

To measure `rho(T)`, context-dependent decode `G(T)`, and mixed prefill/decode
interference on one A100 node, write the MVP runbook with:

```bash
uv run python queue-haul/stage1_service_surface.py \
  --run-id gpt-oss-20b-a100-tp1-service-surface \
  -- --async-scheduling
```

It writes `queue-haul/runs/stage1_service_surface/<run_id>/commands.sh`. Execute
that script on the A100 node, using the same optional `APP='apptainer exec ...'`
wrapper as above. After collection, reduce the emitted bundles with:

```bash
uv run python queue-haul/stage1_service_reduce.py \
  --run-dir queue-haul/runs/stage1_service_surface/gpt-oss-20b-a100-tp1-service-surface/bundles
```

The reducer writes:

- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_prefill_rho.{csv,pdf,png}`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_decode_context.{csv,pdf,png}`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_mixed_surface.{csv,pdf,png}`
- `queue-haul/outputs/stage1_gpt_oss_20b_a100_tp1_service_scale.csv`

## Queue-Haul Stage 1b/1c Source-Sink Proof

Load the host controller runtime before Queue-Haul commands that import NumPy or
CVXPY:

```bash
module load gcc/14.2.0 openblas/0.3.28
PY=.venv/bin/python
```

Stage 1b uses the validated old vLLM Apptainer sandbox with LMCache and a
stdlib user-space proxy instead of Docker or privileged kernel `tc`. The proxy
applies one shared 1Gbps source-egress bucket to API replay bytes and KV-transfer
bytes. On a two-GPU A100 node, run:

```bash
$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
$PY queue-haul/stage1b_drain_sink.py smoke2-live --mbps 1000 --run-root /tmp/qh-smoke2-live
```

Stage 1c keeps the fixture proof as the fast source/sink controller check:

```bash
$PY queue-haul/stage1c_controller.py plan
$PY queue-haul/stage1c_controller.py proof --mbps 1000 --run-root /tmp/qh-proof-live
$PY queue-haul/stage1c_controller.py check --run-root /tmp/qh-proof-live
```

For the production-shaped live controller, first build a local TraceLab-shaped
session manifest from a pinned JSONL/JSONL.gz trace, then run the 4 Hz power trace
and policy-ranked drain:

```bash
$PY queue-haul/stage1c_controller.py make-manifest \
  --source tracelab \
  --input /path/to/syfi_coding_trace.jsonl.gz \
  --out queue-haul/outputs/stage1c_live-sessions.json \
  --sessions 8 \
  --seed 0
$PY queue-haul/stage1c_controller.py live-drain \
  --manifest queue-haul/outputs/stage1c_live-sessions.json \
  --mbps 1000 \
  --nvsmi-ms 250 \
  --run-root queue-haul/outputs/stage1c_live
$PY queue-haul/stage1c_controller.py check-live --run-root queue-haul/outputs/stage1c_live
$PY queue-haul/stage1c_controller.py plot-live --run-root queue-haul/outputs/stage1c_live
$PY queue-haul/stage1c_controller.py live-grid \
  --manifest queue-haul/outputs/stage1c_live-sessions.json \
  --mbps 1000 \
  --run-root queue-haul/outputs/stage1c_grid
```

`live-drain` keeps source vLLM on GPU 0 and sink vLLM on GPU 1 up together,
runs Poisson turn loops from synthetic TraceLab-sized rolling transcripts, moves
one session at a time by replay or KV transfer, and writes `gpu_power.csv`,
`events.jsonl`, `controller_manifest.json`, `power_summary.csv`,
`power_trace.png`, `source_power.png`, `sink_power.png`, `delay_summary.csv`,
`delay_summary.png`, `ell_power5s.csv`, `ell_power5s.png`, and `request_counts.csv`. `live-grid` also writes `scenario_summary.csv`, `grid_power_drop.png`, and `grid_delay.png`.
