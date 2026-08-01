# Queue-Haul

Queue-Haul plans request-boundary migration of stateful LLM sessions to reduce
source accelerator power before a deadline. It jointly chooses sessions,
replay or full-KV transfer, and compatible destination serving pools.

The planner emits a fixed plan naming each migrated session, replay or KV
transfer, destination pool and replica, route, and order. Execution adds phase
timing, commit, observed first-token timing, source transitions, resource use,
debt, recovery, achieved shed, and unmet shed. A requirement frontier summarizes
plans across source-power targets. Destination capacity is an advertised pool
contract, not an inferred GPU inventory.

## Current evidence

The repository contains:

- measured GPT-OSS-20B/A100 power curves;
- working replay and compatible KV handoff on two A100s, with 24/24 serial and
  90/90 bounded-campaign migrations completing by their deadlines;
- 105/105 passing bounded-campaign gates and 6/6 passing parallel-KV gates;
- conservative replay and KV duration fits;
- exact full and incremental KV block/wire accounting;
- measured request-boundary replay and KV Gantt charts through the first
  destination token;
- a one-pool requirement-frontier solver;
- LP, static greedy, experimental bundle and prefix greedies, and simulator-only
  price-coupled greedy with an exact width-8 action oracle and cached recovery;
- pool-aware planning and internal packing checks; and
- a deterministic migration, network, request, queue, and power simulator.

The completed planner-driven policy campaign executes mixed replay/KV choices
with ordered eager-parallel launch. The archived destination service campaign
does not provide an accepted shared-load capacity boundary, so simulator
service headroom remains a sensitivity. That boundary is not required for the
dedicated two-A100 migration claim.

## System boundary

A handoff prepares state in the background, quiesces at a request boundary,
performs final catch-up, switches routing, and succeeds when the destination
returns the first token. Mid-token migration, return migration, cold model
placement, unrelated destination arrivals, provider fleet policy, and facility
power are out of scope.
Same-source migrations may overlap and share the bandwidth of every link on
their route; there is no fixed per-source migration-count limit.

The public candidate is:

```text
(session, replay-or-KV, compatible destination pool)
```

The planner lowers the logical pool choice to a deterministic replica assignment.
A V1 pool advertises ongoing and stable service envelopes, temporary queued-work
allowance, live-KV blocks, route identity, allowed methods, compatibility, and
evidence status. The scenario supplies link rates. The pool planner enforces
ongoing event capacity and conservative replica-second debt and reports required
recovery.

## Evidence flow

```text
archived raw logs
  -> checksum-pinned reduced measurements
  -> versioned model/workload/pool inputs
  -> result tables with provenance
  -> deterministic figures
```

Every input and result must say whether it is measured, fitted, assumed, or
simulated. Assumed values are sensitivities, never admission guarantees.
The conversation workload pins ShareGPT artifact revision
`192ab2185289094fc556ec8ce5ce1e8e587154ca` and stores only token/turn shapes.

The normative formulation and executable-contract mapping are in:

- `formulation_nsdi.md`;
- `PROPOSED_DESTINATION_ARCH.md`.

## Main commands

Run commands from the parent `agent-migrate` directory:

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
uv run python queue-haul/plot_testbed_kv_timeline.py
uv run python queue-haul/plot_testbed_kv_timeline.py --method replay
uv run python queue-haul/mechanism_validation_campaign.py prepare \
  --out queue-haul/outputs/mechanism-validation-plan
# Run the matched current-stack KV/replay campaign on one 2xA100 node:
sbatch queue-haul/outputs/mechanism-validation-plan/run.sbatch
uv run python queue-haul/plot_migration_ttft_cdf.py
uv run python queue-haul/plot_fixed_contract_residuals.py
uv run python queue-haul/policy_hardware_campaign.py prepare \
  --out queue-haul/outputs/policy-hardware-width8-pilot-plan
QH_POLICY_RUN_ROOT=/scratch/users/$USER/qh-policy-run-width8-pilot \
  bash queue-haul/outputs/policy-hardware-width8-pilot-plan/run.sh
# Or submit the resumable two-A100 Slurm job:
sbatch queue-haul/outputs/policy-hardware-width8-pilot-plan/run.sbatch
uv run python queue-haul/migration_profiler.py make-crossover \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out queue-haul/outputs/policy-hardware-crossover-plan/plan.json \
  --context-sizes 2048,4096,8192,16384,24576,32768 \
  --bandwidth-mbps 1000,2500,5000,10000 --repeats 3 --seed 1
uv run python queue-haul/policy_hardware_campaign.py plot-reduced \
  --out queue-haul/outputs/policy-hardware-width8-frontier-20260730
uv run python queue-haul/canonical_simulator_campaign.py
uv run python queue-haul/dual_lagrangian_experiment.py
uv run python queue-haul/simulated_pareto_campaign.py
uv run python queue-haul/paper_evaluation.py \
  --out queue-haul/outputs/paper-evaluation
```

`requirement_frontier.py` computes destination requirements without constructing
a destination inventory. `pool_planner.py` compares those requirements with
concrete pool contracts and emits physical use/capacity rows. Pool admission
shares route capacity across methods but tracks replay and KV aggregate work
separately, matching their distinct measured throughput caps. `simulate.py`
independently schedules routes, reconstruction endpoints, requests, commits,
and power. It separately traces declared pool-service demand and debt from
realized replay and commit times. `evaluation_config.py` is the canonical
source for assumed paper operating points and their replacement evidence.
The default pool LP remains the Clarabel implementation. The experimental
`lp_highs` solver runs the same relaxation and rounder through SciPy/HiGHS;
resource rows are assembled directly from candidate nonzeros.
`lp_column_generation` is a Phase-I/Phase-II prototype with a reported
primal-dual certificate. It materializes the full candidate table and rebuilds
restricted masters, so it is a correctness reference rather than the
million-session implementation.
`lp_column_generation_persistent` keeps one native HiGHS master and basis while
adding session rows and priced columns, and reuses one column-oriented resource
matrix across insertion batches. Candidate construction uses compact
immutable records and reuses identical session physics across equivalent pool
type/route signatures while retaining pool-specific capacity rows.
`destination_bench.py --pool-counts`
splits fixed replica inventory across pools to vary alternatives without adding
hardware or route capacity. All column-generation solvers remain experimental;
`lp_column_generation_lazy` instead streams the complete implicit action
universe, retains only generated master columns, stops on a global certified
gap, and consults the oracle again during integral completion. It preserves the
same LP but currently regenerates Python candidate physics on every pricing
sweep. The native-pricing boundary factors these candidates into float64
session/signature features, packed per-session feasibility masks, and distinct
pool/method sparse templates; the initial kernel is limited to 16 options.
Native sweep execution, indexed replica placement, and a compact execution
verifier are still required for million-session operation.
`dual_lagrangian_experiment.py` evaluates the simulator-only `greedy_prefix`
policy as a dual-Lagrangian prefix method against the unchanged static greedy
baseline and other experimental pool policies on paired trace-derived targets;
infeasible plans receive zero validated shed. The archived exploratory run under
`outputs/dual-lagrangian-*` finds lower migration work and better target
completion in packed coding traces. The scale-oriented policy uses a small
prefix-length frontier, incremental aggregate recovery, one final packing pass,
and a broader frontier at 75% requested shed. Recovery represents each visited
combination by an exact mixed-radix source-prefix ID instead of copying the
global selected set. This bounded rewrite uses
roughly 6–44% more migration work than its exhaustive-prefix experimental
precursor in the archived comparable cases, while remaining below static
greedy. It remains experimental and does not change the hardware controller.
In a single-seed coding 10% sensitivity, prefix versus static greedy takes
65/92 seconds at 250K, 200/290 seconds at 500K, and 694/1,236 seconds at 1M,
while selecting 55–60% less migration work. At 1M it uses 5.02 versus 4.63 GB
peak RSS and finishes migration at 111 versus 9 seconds against a 115-second
horizon. Two additional 100K workloads also pass exact validation with lower
prefix time and work. The 75% branch passes at 100K but takes 229 versus 174
seconds and 1.78 versus 0.78 GB at 74.9%. These are scale sensitivities, not
confidence intervals or a production-scale claim.
`paper_evaluation.py` writes the legacy Q1–Q9 result/plot registry while the
paper evaluation is reorganized into mechanism validation, fixed-contract
coordination, multi-pool contracts, and planner quality/scale. It rejects
tables with missing provenance.
`mechanism_validation_campaign.py` replaces the mechanism Gantts with matched
current-stack 28K-context, 10-Gb/s, concurrency-one KV and replay measurements.
It runs five repetitions of the same four-turn source workload and plots the
trace nearest each mechanism's median request-boundary wait.
Timestamped arrivals execute independently of migration. Quiescence drains only
requests admitted before the pause; later arrivals wait for the destination
route. The committed `fix1` charts predate this correction and must be replaced
by the fresh `quiesce` run before using their gray intervals as drain evidence.
`plot_fixed_contract_residuals.py` caches the canonical 100K-session,
120-second fixed-contract requirement sweep and compares mixed greedy,
GPU-work-first, replay-only, and KV-only resource headroom at common requested
source-power shed levels. Use `--refresh` when its pinned inputs change.
`policy_hardware_campaign.py` creates a resumable paired idle-session campaign
that launches every session concurrently for Queue-Haul, greedy, KV-only, and
replay-only. Its default truncated grid uses coding, interactive-coding, and
agentic-tool-loop context-length profiles; uniform-over-support and
uniform-over-range token distributions, or named exact context packs; configurable
bandwidths and deadlines; and full-episode migration width. The requirement and
cell bandwidth are passed to the Queue-Haul planner. The runner remains eager and
does not execute planner pacing or measure
planning latency. A policy-infeasible deadline retains its admitted prefix and
appends an explicitly marked independently-fastest tail so runtime width remains
the episode size. Failed episodes remain in denominators. Reduction writes
timing CDFs and a power-attainment CDF; attainment is trailing-five-second
average modeled source-power shed divided by the 100% source-power target. MP
runs require bounded RESP quiescence between scenarios so late cache writes
cannot cross scenario boundaries. Run from a clean committed checkout with two
A100 80GB GPUs. The pinned frontier plan under
`outputs/policy-hardware-width8-frontier-plan/` uses seed 1, three episodes per
cell, eight sessions and moves, 5/10 Gbit/s, and 19/30-second requirements for
360 scenarios. Its fresh default root is
`/scratch/users/$USER/qh-policy-run-width8-frontier`; 1 Gbit/s is excluded.
Its completed checksum-pinned reduced bundle is retained under
`outputs/policy-hardware-width8-frontier-20260730/`, including move-admission
provenance and raw GPU samples. `plot-reduced` writes a pooled
migration-to-destination-first-token CDF and median modeled source-power shed
over elapsed time with an interquartile band, plus paired attainment–completion
points and a CDF of measured session downtime per modeled watt shed. This idle
evidence also includes an episode migration-makespan-per-modeled-watt CDF and
supports timing and projected, not realized, power attainment.
The separate live power-drain evidence in
`outputs/power_drain_live_20260714/` includes planned and measured source-power
reductions; `plot_migration_results.py` writes their shared-axis parity plot.
The parity plot compares Queue-Haul LP with greedy only.
`migration_profiler.py make-crossover` creates paired single-session replay/KV
measurements for each nominal context, bandwidth, and repeat. The synthetic body
reserves 192 tokens for message overhead, and the first 32K replay is a fail-fast
model-limit smoke. Bandwidths remain contiguous to avoid unnecessary MP-stack
restarts. Its 6-context, 4-bandwidth, 3-repeat grid contains
144 migrations. Use the reduced measurements to establish the method crossover
before freezing the 2K–16K width-8 packing plan; the legacy replay profile
starts at 3,473 tokens and must not be extrapolated to 2K. The completed 144/144
crossover bundle is checksum-pinned under
`outputs/policy-hardware-crossover-20260730/`. Replay is faster across the tested
range at 1/2.5 Gbit/s, through 8K at 5 Gbit/s, and through 4K at 10 Gbit/s; the
8K 10-Gbit/s cell is effectively tied. The derived profile
`profiles/gpt_oss_20b_a100_tp1_crossover.json` replaces serial replay rate,
replay/KV completion, KV ingestion lower bound, and route-switch timing while
preserving the existing catch-up, power, and capacity evidence. The pinned
`outputs/policy-hardware-width8-packing-plan/` runs three paired width-8 episodes
for Tiny, Small, Medium, Mixed, and Large packs at 1/2.5/5/10 Gbit/s and 19/30-s
requirements: 600 scenarios in total. Job 36822272 completed all 600 scenarios
without failures in 8:08:22. Its checksum-pinned reduced bundle is under
`outputs/policy-hardware-width8-packing-20260730/`; compressed results, GPU
samples, proxy byte counters, and RESP transfer records retain the raw evidence
without runtime debug logs. Its bandwidth-faceted destination-TTFT CDF pools
all five workloads, both deadlines, and three episodes within each bandwidth;
the companion pooled CDF combines all four bandwidths. Regenerate that CDF
with the earlier trace-sampled 5/10-Gbit/s frontier rows included at raw-sample
weight using `plot-reduced --out <packing-results> --pooled-with
<frontier-results>`; this pooling also applies to both per-watt CDFs.
`simulated_pareto_campaign.py` evaluates the same five fixed context packs with
the calibrated crossover profile and adds paired random and price-coupled
greedy baselines. Coupled greedy uses the same single destination and link; its
zero-background-load service envelope is explicitly a sensitivity. The Pareto
CSV and plot label 12K/14K contexts as interpolated, concurrent action power as
extrapolated, and commit-derived power attainment as modeled.
Parallel launch and replay/KV aggregate-throughput caps are anchored by the
hardware traces; KV uses serial measurements at 1/2.5 Gbit/s and width-8
measurements at 5/10 Gbit/s. The figure pairs deadline-normalized and raw-second
clouds over 30/40/50/60/75-s budgets, with one point per result. Every policy is
replanned for each budget against the same aggregate caps and executes only its
deadline-admitted actions; cleanup moves are excluded from both axes. The cloud
crosses five fixed packing anchors and 18 observed width-8 workload mixes with
every bandwidth and budget; no display jitter is applied.
`canonical_simulator_campaign.py` runs a four-target paired 10K-session
Queue-Haul, greedy, per-session-fastest, replay-only, and KV-only comparison
under one assumed dedicated-pool contract. Its compact 10K/100K/1M scale check
uses the Queue-Haul greedy planner, an equivalent pooled-destination topology,
10 Gbps per 10K sessions, and summary-only prediction. Sampled future requests
are disabled; measured two-A100 results provide continuation first-token
evidence.
Override `QH_APPTAINER_IMAGE` if the pinned LMCache image is not at the default
scratch path; set `QH_RESUME_FROM_GIT_SHA` when resuming after a code change.
MP catch-up completeness uses exact rendered-token prefix chunks; observed key
sets may also contain a generated-tail chunk.
Stacked MP runs retain per-scenario byte, connection, and RESP traces.
Re-submit after a time limit; the stable run root reuses its original port
offset and completed scenarios. Set `QH_RESUME_FROM_GIT_SHA` after code changes.

## Measurement programs

- `power_window_sensitivity.py` and `power_profile_reduce.py`: source power.
- `migration_profiler.py` and `migration_profile_fit.py`: replay/KV handoff.
- `destination_campaign.py` and `destination_runner.py`: targeted destination
  service and loaded-migration evidence.
- `service_surface_runner.py` and `service_profile_reduce.py`: isolated and
  mixed service profiles.

The verified 2026-07-23 destination bundle is retained under
`outputs/destination-v7-20260722/`. Do not treat its service rows as an accepted
capacity profile. See `FINDINGS.md` and `DATA_TO_COLLECT.md` for the forensic
audit and retained evidence.

## Development rules

- Keep implementation and documentation small.
- Hard-fail unsupported domains and invalid evidence.
- Add semantic tests for every source change.
- Run `uv run pytest` after every change.
- Commit each completed task separately with a descriptive message.
