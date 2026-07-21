# Queue-Haul

`destination_campaign.py` builds content-free, revision-pinned trace manifests
through the exact GPT-OSS tokenizer endpoint. It hard-fails unless each workload
has disjoint 12/6/6 fit/tuning/final splits. `audit-evidence` lists the five GPU
measurements; WAN/KV arithmetic, trace growth, horizons, and architecture sweeps
remain offline derivations, while older hardware results are priors only.
General coding uses Trace Commons, interactive coding uses uniformly sampled,
opt-in English multi-turn WildChat rows passing a high-precision code filter,
and agentic work uses permissively sourced NVIDIA SWE-Hero trajectories.
`make-plan` freezes five 12-hour shards plus one conditional repeat shard (72
A100-pair-hours total); `check` stops dependent shards on failed preflight,
frontier, loaded-migration, or validation gates. `reduce` emits paired central
and conservative measured A100 profile fragments.

Download public trace rows before allocating GPUs, then submit checksum-pinned
job files through the single parameterized Slurm script:

```bash
uv run python queue-haul/destination_campaign.py fetch-traces --out-dir /scratch/$USER/qh-traces
uv run python queue-haul/destination_campaign.py make-plan --out queue-haul/outputs/destination-plan.json
uv run python queue-haul/destination_campaign.py submit-next --plan queue-haul/outputs/destination-plan.json --job-dir queue-haul/outputs/destination-jobs
```

Each `shard-N.sh` needs a sibling `shard-N.sh.sha256`; shards 1–5 submit with
fail-fast `afterok` dependencies and shard 6 is submitted only with
`--include-reserve`. Write `SHA256SUMS` remotely, then retrieve full raw and
reduced artifacts with `QH_REMOTE`, `QH_REMOTE_ROOT`, and `sync`; rsync is
resumable and every received file is verified.

Queue-Haul optionally accepts a versioned `DestinationArchitecture` from
`destination.py`. It describes compatibility, context-conditioned service work,
nested normal/emergency/stable envelopes, per-replica baseline service and KV
state, pool routes, and conservative loaded-migration coefficients. Omitting it
keeps the legacy scalar destination model unchanged.

Pass it as `plan(..., destination=architecture)` for pool-aware LP or greedy
admission, or to `execute(..., destination=architecture)` for independent stable
envelope validation. Results report admission mode, shortfall/failure, packing
repairs, and predicted migration makespan; `target_unmet` is valid best effort,
not successful curtailment.

`destination_evaluation.py` reduces three-or-more independent runs into central
and conservative envelope/migration inputs and provides the fixed 36-cell
`rho × H × pool-count` grid, exact facet headroom, integer replica allocation,
and paired scalar/LP/greedy sweep records.

Queue-Haul models and measures session migration under a local source-site
power limit. The active path is:

```text
profiles.py → planner.py → simulate.py → power_drain_experiment.py
```

The role-named `*_runner.py`, `*_reduce.py`, `migration_*.py`, and
`lmcache_*.py` programs collect and reduce service, migration, and power data.
`profiles/*.json` records the measured range and uncertainty used by the
simulator. Run all commands from `agent-migrate`:

```bash
uv run pytest
uv run python queue-haul/power_drain_experiment.py \
  --workload-profile queue-haul/profiles/agentic_tool_loop.json \
  --sessions 6 --seed 3 --power-limit 500 --deadline 5 --end 5 \
  --link-bytes-per-s 125000000 --intra-dc-bytes-per-s 12500000000 \
  --solver greedy --workers 2 --out queue-haul/outputs/profile_smoke
uv run python queue-haul/plot_simulator_validation.py
uv run python queue-haul/plot_simulator_evaluation.py
uv run python queue-haul/plot_scaling_results.py
```

The model profile moves only complete immutable KV blocks; an unsealed tail is
reconstructed during final preparation. KV loading overlaps network transfer,
so serial KV time is setup plus the slower of network transfer and destination
KV loading, followed by synchronization and route switching.
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

Deadline pacing reserves KV setup, endpoint completion, final fixed catch-up,
partial-tail reconstruction, and the configured expected-growth envelope. It
rejects rates above physical route or endpoint capacity and caps background
preparation only; paused final catch-up uses the available shared transport.
Replay execution and LP capacity also include measured replay completion time.
Expected replay WAN bytes grow with the materialized durable log. Planner
validation materializes expected growth at quiescence without exposing sampled
future requests.

For active sessions with `--final-state awake`, `greedy` scores each replay
and KV-transfer option by conservative marginal source-power reduction per unit
of priced resource use. Prices rise with normalized demand from one initially
cheapest action per session, providing a scalable approximation to LP resource
duals without counting mutually exclusive actions twice. It sorts once, chooses
the highest-scoring action that still fits, and reserves the same source, network,
destination service, compute, KV-residency, and trailing power-window capacities
as the LP. This remains approximately `O(N log N)`.

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
uv run python queue-haul/migration_profile_fit.py \
  --serial-root queue-haul/outputs/serial-power-run-2 \
  --catch-up-root queue-haul/outputs/append-catch-up-run-2 \
  --parallel-root queue-haul/outputs/parallel-kv-gate-run-2 \
  --base-profile queue-haul/profiles/gpt_oss_20b_a100_tp1.json \
  --out-profile /tmp/gpt_oss_20b_a100_tp1.json
```

After `check-campaign` passes, `--parallel-root` may point to the bounded
campaign; the fitter then consumes its concurrency gate and measured action
power. The one-off MP reports remain mechanism proofs, not profile-fit inputs.

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

Migration reduction reports measured prompt, processed, and new tokens; initial
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
uv run python queue-haul/migration_profiler.py make-plan \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out queue-haul/outputs/serial-power-plan.json \
  --context-sizes 10896,24292 --concurrency 1 \
  --bandwidth-mbps 1000,10000 --methods kv_transfer,replay \
  --activity none --repeats 3 --seed 0 \
  --session-ids codex:e381cc89-38ef-e67e-79b9-4b800369b4f5
```

`migration_profile.sbatch` profiles two 60-second empty-awake/sleep pairs once
before running that plan. It requests two GPUs. The source and destination use
Slurm's first and second assigned GPUs. The primary 250 ms GPU power samples
come from `nvidia-smi`; migration energy is time-weighted over those samples.
Raw GPU telemetry, state windows, transition times, wake probes, and
`summary.csv` are stored in `RUN_ROOT/power_states`.

Reproduce the two completed targeted jobs without repeating power profiling:

```bash
sbatch queue-haul/parallel_kv_gate.sbatch
sbatch queue-haul/append_catch_up.sbatch
```

Generate and submit the bounded 105-scenario hardware campaign with:

```bash
uv run python queue-haul/migration_profiler.py make-campaign \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out queue-haul/outputs/bounded-hardware-campaign-plan.json --seed 0
sbatch queue-haul/bounded_hardware_campaign.sbatch
```

The plan contains 63 parallel-surface and 42 staged-append scenarios. It runs
the 4k/1-Gbps/concurrency-4 smoke first and aborts on its failure, randomizes
the remainder, resumes only against the same hashed plan, and runs
`check-campaign` after all 105 results complete. This launcher pins the
validated LMCache MP image and uses explicit warm-prefetch before inference;
the gate requires distinct-session body overlap, exact missing-block bytes,
L1-only readiness, and valid continuations. The workload-class and mixed
held-out extension remains gated on pre-staged complete SWE-chat traces; it is
not synthesized from the coding manifest.

Standalone legacy plans still default to vLLM 0.10.1.1 / LMCache 0.3.3; the
bounded campaign and MP-specific jobs opt into the verified immutable image
`/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif`
(SHA-256 `50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab`),
vLLM 0.22.0+cu129, LMCache 0.5.1's shipped `LMCacheMPConnector`, two CPU-only
engine-driven MP servers,
and Redis L2. The shared 10-Gbps proxy parses RESP and attributes successful
GET response bodies to source SET keys, so remote wire bytes exclude source
context growth. Run it on two A100s with:

```bash
sbatch queue-haul/lmcache_mp_campaign.sbatch
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

The incremental-prefetch campaign (`mp-incremental-run-3-20260720`) proved
incremental MP staging:
warm-prefetching each complete 12K/13.6K/15K/16K snapshot into retained
destination L1 transferred exactly 48/5/5/6 new blocks with no repeated prefix
keys. Every token, state, and wire gate passed, including a real second turn.
The concurrency-four confirmation overlapped two attributed sessions at 1.116
GB/s. This is an LMCache implementation of the simulator's generic contract:
move missing sealed blocks, retain them at the destination until commit, and
report readiness only after residency. Another transport needs new measured
block geometry and timing, not LMCache concepts in the simulator. Reproduce it
with:

```bash
sbatch queue-haul/lmcache_incremental_prefetch.sbatch
```

The parallel KV gate completed all 12 fixed two-session scenarios at 1 Gbps within
deadline. All six migrations passed cache, continuation, exact aggregate byte,
and independent large-body connection checks; each used 95 connections with up
to four overlapping windows. The append catch-up campaign completed all 24 scenarios within
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

Scaling plots compare random selection, the greedy solver, and CLARABEL LP on
solver choices, deadline completion, simulated power reduction, migration
completion, and planning time. Random lines are means over ten seeds and shaded
bands show the observed minimum and maximum.
`outputs/scaling_1_to_100k_15min_20260717/scaling_summary.{png,pdf}` repeats
the sweep with a 15-minute deadline and 22.5-minute observation window.
`outputs/lp_objective_comparison_15min_20260717/scaling_summary.{png,pdf}`
compares the restored LP with peak-before-work and work-before-peak. All three
choose nearly identical plans; at 100,000 sessions they move 94,956 sessions
and achieve about 189% of the requested reduction. The restored LP is faster,
so the remaining over-selection comes from the conservative linear power bound,
not objective order.

Greedy planning batches homogeneous seeded random choices without changing
their sequence and reuses static expected scenarios. Exact simulation also
drops inactive links from fair-share calculations.
