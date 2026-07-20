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
  --link-bytes-per-s 125000000 --intra-dc-bytes-per-s 12500000000 \
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

The network simulator is a fixed-path fluid-capacity model, not a TCP model.
Active transfers share every named bottleneck with work-conserving max-min
rates. The default route crosses a source-node fabric link, one shared source
site egress, one shared WAN allocation, one shared destination-site ingress,
and a destination-node fabric link. `--link-bytes-per-s` controls all three
shared inter-site cuts; `--intra-dc-bytes-per-s` controls the nonbinding
per-node fabric tier. Adding nodes therefore does not multiply WAN capacity.
Treat these as sensitivity inputs, not calibrated physical-site claims. Published
[A100 GPUDirect measurements](https://developer.nvidia.com/blog/accelerating-io-in-the-modern-data-center-network-io/)
give 24 GB/s per 200 Gbps RDMA rail, while
[Jupiter](https://research.google.com/pubs/archive/43837.pdf) motivates
full/half-bisection fabrics and sensitivity around shared external cuts. The
shaped 1/10 Gbps WAN allocations remain scenario inputs rather than claims
about the cluster. Workload profile v2 fixes durable logs at `source_dc`, so
replay traffic crosses the same site egress and WAN as KV traffic.

Deadline pacing reserves KV setup and initial-completion time, includes the
configured expected-growth envelope, and rejects rates above physical route or
endpoint capacity. It caps background preparation only; paused final catch-up
uses the available shared transport. Planner validation materializes expected
growth at quiescence without exposing sampled future requests.

For active sessions with `--final-state awake`, `node_drain` ranks source nodes
by exact power reduction to idle divided by predicted drain time. It then ranks
sessions within each node by power reduction per resource use and reserves the
same source, network, destination replay/KV, compute, KV residency, and trailing
power-window capacities as the LP. It empties a node when possible and otherwise
takes only the sessions that fit. Cold-session, sleep, and shutdown plans retain
the simpler whole-node ordering until those transitions are measured.

`--solver lp` jointly selects replay and KV transfer under source-instance,
network, destination replay, destination KV, compute, residency, and source
power limits. Its CVXPY model uses CLARABEL and restores the earlier Queue-Haul
objective: meet the requested power reduction with minimum total migration
work, or maximize power reduction when the target is infeasible. The
`lp_peak_first` and `lp_work_first` solvers retain the two three-stage objective
orders for direct comparison. The current LP scope is active
sessions, one destination pool, and `--final-state awake`;
unsupported cases hard-fail. The fractional plan is rounded to whole sessions
and accepted only when the discrete-event simulator meets trailing-window power
and every migration commits by the migration deadline. The exact equations and
conservative concave-power bound are in `queue-haul/formulation.md`.
Action power is stored as total added power for each measured concurrency, not
as power per session. The simulator updates these totals when concurrency
changes, and the planner reuses a route-resource summary only when the complete
set of route paths matches. Fit the serial coding data with repeats 0–1 and
evaluate repeat 2 with:

```bash
uv run python queue-haul/stage1c_profile_fit.py \
  --serial-root queue-haul/outputs/serial-power-run-2 \
  --catch-up-root queue-haul/outputs/append-catch-up-run-2 \
  --parallel-root queue-haul/outputs/parallel-kv-gate-run-2 \
  --base-profile queue-haul/profiles/gpt_oss_20b_a100_tp1.json \
  --out-profile /tmp/gpt_oss_20b_a100_tp1.json
```

The checked profile remains `estimated`. It incorporates the paired serial,
append-only catch-up, parallel-gate concurrency, and GPU-only sleep results,
but has not been validated for interactive or agentic jobs, eight-session
drains, shutdown, or exclusive whole-node power.

The completed `serial-power-run-2` pins the same session and turn across
methods and bandwidths and shares controls across those comparisons. All 30
scenarios completed within deadline. The older `coding-run` predates paired
planning, so only its observations remain unpaired.
Two paired 60-second windows found that source level-1 sleep released GPU
memory but left A100 board power unchanged at about 84.9 W. The run did not
collect exclusive whole-node power.

Stage 1C reduction reports measured prompt, processed, and new tokens; initial
KV payload bytes; catch-up cache hits; connection-attributed proxy bytes;
initial and catch-up wire windows; request timing; and power relative to a
measured idle baseline. Active runs also write `catch_up.csv` with measured
prompt/output separation, KV growth, effective copy service, final pause, and
the resulting convergence test. It does not group or plot by requested context
size.
`initial_time`, `throughput`, `concurrency_scaling`, `service_effects`,
`power_energy`, and `model_check` show the direct relationships.

New plans record migration and serving concurrency separately and default to
`final_state: awake`; source sleep occurs only when a plan explicitly requests
it. Version-2 plans and results remain readable, while new artifacts use schema
version 3. The migration controller also preserves ordered append-stage
snapshots and compares final catch-up against the last prepared stage.
`copy_policy: after_each_request` pipelines the next controlled source turn
with the current destination write, applies one serving-concurrency gate across
sessions, and reduces exact key-attributed stage bytes to
`migration_stages.csv`. Request schedules start after reset and warm-up;
`service_requests.csv` records scheduled delay, TTFT, service time, token
growth, route, and success for migrations and controls. Version-2 reduction
does not require the new connection-attribution evidence.

Reproduce the completed 30-scenario serial crossover plan with:

```bash
uv run python queue-haul/stage1c_controller.py make-plan \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out queue-haul/outputs/serial-power-plan.json \
  --context-sizes 10896,24292 --concurrency 1 \
  --bandwidth-mbps 1000,10000 --methods kv_transfer,replay \
  --activity none --repeats 3 --seed 0 \
  --session-ids codex:e381cc89-38ef-e67e-79b9-4b800369b4f5
```

`stage1c_benchmark.sbatch` profiles two 60-second empty-awake/sleep pairs once
before running that plan. It requests two GPUs. The source and destination use
Slurm's first and second assigned GPUs. The primary 250 ms GPU power samples
come from `nvidia-smi`; migration energy is time-weighted over those samples.
Raw GPU telemetry, state windows, transition times, wake probes, and
`summary.csv` are stored in `RUN_ROOT/power_states`.

Reproduce the two completed targeted jobs without repeating power profiling:

```bash
sbatch queue-haul/stage1d_parallel_gate.sbatch
sbatch queue-haul/stage1e_catch_up.sbatch
```

Generate and submit the bounded 105-scenario hardware campaign with:

```bash
uv run python queue-haul/stage1c_controller.py make-campaign \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out queue-haul/outputs/bounded-hardware-campaign-plan.json --seed 0
sbatch queue-haul/stage1f_campaign.sbatch
```

The plan contains 63 parallel-surface and 42 staged-append scenarios. It runs
the 4k/1-Gbps/concurrency-4 smoke first and aborts on its failure, randomizes
the remainder, resumes only against the same hashed plan, and runs
`check-campaign` after all 105 results complete. The workload-class and mixed
held-out extension remains gated on pre-staged complete SWE-chat traces; it is
not synthesized from the coding manifest.

Stage 1G is the opt-in LMCache multiprocess path; the legacy vLLM 0.10.1.1 /
LMCache 0.3.3 path remains the default. It uses the verified immutable image
`/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif`
(SHA-256 `50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab`),
vLLM 0.22.0+cu129, LMCache 0.5.1's shipped `LMCacheMPConnector`, two CPU-only
engine-driven MP servers,
and Redis L2. The shared 10-Gbps proxy parses RESP and attributes successful
GET response bodies to source SET keys, so remote wire bytes exclude source
context growth. Run it on two A100s with:

```bash
sbatch queue-haul/stage1g_mp_campaign.sbatch
```

The completed `mp-campaign-run-10-20260719` ran four distinct approximately
16K-token sessions at concurrency 1/2/4 and one four-stage append-only session,
all with three repeats. Its concurrency and accounting gates passed, but its
append stages fetched 48/53/58/64 remote blocks and therefore did not establish
incremental wire transfer. Median aggregate KV throughput
was 591 MB/s at concurrency 2 and 1.206 GB/s at concurrency 4, versus a 111
MB/s serialized ceiling; at least two key-attributed session bodies overlapped.
At the final 16K append stage, vLLM reported 16,384 cached tokens, decomposed
exactly into 14,848 vLLM-local and 1,536 LMCache-retrieved tokens. All
continuations, RESP wire/body equations, and repeat counts passed.

Stage 1H warm-prefetches each complete 12K/13.6K/15K/16K snapshot into
destination L1 before the real vLLM lookup. It hard-fails unless WAN key counts
are exactly 48/5/5/6 with no duplicate prefix keys, token and wire equations
are exact, the final conversational turn preserves the state code, and one
concurrency-four repeat exceeds 1 GB/s with two sessions overlapping:

```bash
sbatch queue-haul/stage1h_mp_incremental.sbatch
```

Stage 1D completed all 12 fixed two-session scenarios at 1 Gbps within
deadline. All six migrations passed cache, continuation, exact aggregate byte,
and independent large-body connection checks; each used 95 connections with up
to four overlapping windows. Stage 1E completed all 24 scenarios within
deadline: one fixed session, 32/128/512/2,048-token controlled appends, 1/10
Gbps, two repeats, and matched controls. All 16 migrations overlapped generation
with the initial copy, transferred positive incremental catch-up bytes, retained
every appended token, continued correctly, and measured copy service faster
than KV growth. The evidence is stored in `parallel-kv-gate-run-2` and
`append-catch-up-run-2`. The shared runner reduces partial evidence but executes
the hard gate only after every scenario completes.

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

`outputs/scaling_1_to_100k_20260717/scaling_summary.{png,pdf}` compares the
updated capacity-aware greedy, CLARABEL LP, and node-aware baseline on solver
choices, deadline completion, simulated power reduction, migration completion,
and planning time in the paired coding sweep.
`outputs/scaling_1_to_100k_15min_20260717/scaling_summary.{png,pdf}` repeats
the sweep with a 15-minute deadline and 22.5-minute observation window.
`outputs/lp_objective_comparison_15min_20260717/scaling_summary.{png,pdf}`
compares the restored LP with peak-before-work and work-before-peak. All three
choose nearly identical plans; at 100,000 sessions they move 94,956 sessions
and achieve about 189% of the requested reduction. The restored LP is faster,
so the remaining over-selection comes from the conservative linear power bound,
not objective order.
