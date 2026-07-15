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

## Queue-Haul live migration profiling

Load the host controller runtime before Queue-Haul commands that import NumPy or
CVXPY:

```bash
module load gcc/14.2.0 openblas/0.3.28
PY=.venv/bin/python
```

Stage 1b runs source vLLM on GPU 0, destination vLLM on GPU 1, a shared remote
LMCache server, and one user-space bandwidth limit shared by replay and KV
traffic. Each vLLM process has its private LMCache CPU tier disabled while
retaining the 4 GB pinned staging allocator required by the remote connector. On a
two-GPU A100 node, check the pinned vLLM `0.10.1.1` and LMCache `0.3.3` setup:

```bash
$PY queue-haul/stage1b_drain_sink.py preflight --required-gpus 2
$PY queue-haul/stage1b_drain_sink.py smoke2-live --mbps 1000 --run-root /tmp/qh-smoke2-live
```

Stage 1c profiles migration mechanisms; it does not choose a power policy or
declare power/deadline acceptance. Build a deterministic manifest from a pinned
TraceLab JSONL or JSONL.gz file. Rows without usable timing or token counts are
excluded, and manifest creation fails if too few eligible sessions remain. Trace timing, token counts,
turn boundaries,
job class, and the source hash are retained. Message text is generated because
the trace does not contain it:

```bash
$PY queue-haul/stage1c_controller.py make-manifest \
  --input /path/to/trace.jsonl.gz \
  --out queue-haul/outputs/coding-manifest.json \
  --workload coding \
  --sessions 8 \
  --seed 0
```

Expand the manifest into randomized migration scenarios and matched
no-migration controls. Generated profiling scenarios use one method at a time;
the plan schema also permits moves with different methods:

```bash
$PY queue-haul/stage1c_controller.py make-plan \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out queue-haul/outputs/coding-plan.json \
  --context-sizes 2048,8192,16384 \
  --concurrency 1,2,4 \
  --bandwidth-mbps 250,1000,10000 \
  --methods replay,kv_transfer \
  --activity none,one_turn \
  --repeats 3 \
  --seed 0
```

For each move, the source keeps serving during the initial destination copy.
The controller then pauses that session, waits for its active request, copies
the complete updated conversation once if it changed, verifies the destination,
switches the route, and resumes queued work there. One total concurrency limit
covers replay and KV moves from initial copy through route switch. A failure
before the switch resumes the source route.

Before each scenario, the controller verifies that the remote cache is empty
and requires both vLLM logs to report a successful prefix-cache reset; HTTP 200
alone is insufficient. Replay sessions are warmed before the remote cache is
cleared. KV sessions are warmed afterward. Replay requests must report zero KV
hit tokens. KV requests must hit exactly the complete 256-token chunks. A reset
failure restarts the testbed and retries once, then stops the run.

Run and reduce separately. Formal runs require a clean worktree; `--allow-dirty`
is available for development. Resume requires the same commit, plan, manifest,
and settings. A failed scenario is saved, the testbed is restarted, independent
scenarios continue, and the final command exits nonzero. The Slurm launcher
always invokes reduction and preserves the run status:

```bash
$PY queue-haul/stage1c_controller.py run \
  --plan queue-haul/outputs/coding-plan.json \
  --run-root queue-haul/outputs/coding-run
$PY queue-haul/stage1c_controller.py reduce \
  --run-root queue-haul/outputs/coding-run
# On Sherlock:
PLAN=queue-haul/outputs/coding-plan.json \
RUN_ROOT=queue-haul/outputs/coding-run \
sbatch queue-haul/stage1c_benchmark.sbatch
```

Each scenario records controller events, every streamed response chunk, prompt
and output token totals, structured cache operations, 250 ms link totals,
per-connection byte totals, and 250 ms GPU power/utilization/memory samples.
Per-session KV work uses exact complete chunks and bytes derived from the logged
KV layout; concurrent shared-wire bytes remain scenario totals. Source sleep is
attempted only after every active session commits, and its timing is separate
from migration time.

`reduce` validates the raw run and writes `migrations.csv`, `scenarios.csv`, and
`benchmark_summary.csv`. Each scenario gets `migration_timeline.png` and
`resource_trace.png`. Cross-scenario copy-time, concurrency-scaling, and service
effect plots are written as PNG and PDF. Power and deadline fields are
diagnostics, not acceptance criteria. Cells keep job class, context size, link,
method, and activity separate; p95 appears only with at least 20 samples and a
bootstrap median interval only with at least 10.

The separate offline 10,000-session power-drain experiment is unchanged:

```bash
uv run python queue-haul/power_drain_experiment.py \
  --out queue-haul/outputs/power_drain_offline
```

It writes `scale_results.csv`, `scale_policy_comparison.png`, and
`scale_network_sensitivity.png`.

Raw live run directories and scheduler logs are generated and ignored.
