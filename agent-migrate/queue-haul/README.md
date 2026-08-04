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
- LP, static `greedy`, and dual-priced `greedy_lagrangian` optimizers;
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

## Three-region Azure A100 campaign

The implemented campaign powers down the Sweden Central source GPU and
reconstructs sessions in East US 2 and West Europe. It uses private IPs over Global VNet
Peering; Azure routes peered-VNet traffic over the Microsoft backbone, not the
public Internet. No Azure CLI access is needed on the VMs. The relevant Azure
contracts are [Global VNet Peering](https://learn.microsoft.com/en-us/azure/networking/design-guide/cross-region),
[Linux PTP/chrony](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/time-sync),
and [Spot Scheduled Events](https://learn.microsoft.com/en-us/azure/virtual-machines/windows/scheduled-events).

The frozen node map in `queue-haul/azure_network_cluster.json` is:

| role | region | private IP |
|---|---|---|
| source/power-down | Sweden Central | `10.0.0.4` |
| destination | East US 2 | `10.1.0.4` |
| destination | West Europe | `10.2.0.4` |

`check` compares every entry with Azure IMDS and hard-fails before calibration
if the actual West/Sweden address assignment differs. In that case, correct the
two destination records; do not bypass the check.

### One-time portal work for the Azure account owner

The account owner must complete these items. The experiment operator does not
need `az` permissions.

1. Use three `Standard_NC24ads_A100_v4` Spot VMs with one visible A100 each,
   Azure Linux 3.0, persistent `/datadrive`, eviction policy `Deallocate`, and
   no delete-on-eviction data disk.
2. Configure bidirectional Global VNet Peering between the Sweden Central VNet and
   each destination VNet. Both peerings must show `Connected`; address spaces
   must not overlap. Destination-to-destination peering is unnecessary.
3. Do not add a public data-plane address, NAT gateway, VPN, load balancer, or
   TLS terminator. SSH can use the existing private access path. Private Azure
   backbone traffic is not application-layer encryption; that is acceptable for
   this measurement-only, private-VNet deployment.
4. Restrict NSGs to these experiment flows. Source egress goes only to each
   destination on TCP `22,5201,8081,8200` and ICMP. East US 2 permits source
   `10.0.0.4/32` on those ports; West Europe does the same. Sweden Central
   permits TCP `8301` from East `10.1.0.4/32` and TCP `8302` from West
   `10.2.0.4/32`. Ports `5556,5557,5655,8080,8100,8401,8402` remain
   host-local. Do not expose any experiment port to `0.0.0.0/0`.
5. Confirm all three VMs have the repository at
   `/home/azureuser/coding-progress-ledger/agent-migrate`, the same commit, and
   the source has `~/.ssh/azrs` plus verified host keys for both destinations.

### Install all three hosts

Run from the `agent-migrate` repository on every VM as `azureuser`, not root:

```bash
bash queue-haul/setup.sh
source ~/.bashrc
```

This installs Valkey, `chrony`, and `iperf3`, configures chrony against Azure's
stable `/dev/ptp_hyperv` device, waits for synchronization, installs the pinned
Python 3.12/vLLM 0.22.0/LMCache 0.5.1 CUDA 12.9 runtime, and stores the pinned
GPT-OSS-20B model and caches under `/datadrive`. Setup hard-fails without the
A100, persistent data mount, PTP device, or pinned runtime.

From the Sweden Central source, establish and verify SSH host keys once, then confirm
that the same commit is checked out everywhere:

```bash
ssh -i ~/.ssh/azrs azureuser@10.1.0.4 true
ssh -i ~/.ssh/azrs azureuser@10.2.0.4 true
git rev-parse HEAD
ssh -i ~/.ssh/azrs azureuser@10.1.0.4 'cd /home/azureuser/coding-progress-ledger/agent-migrate && git rev-parse HEAD'
ssh -i ~/.ssh/azrs azureuser@10.2.0.4 'cd /home/azureuser/coding-progress-ledger/agent-migrate && git rev-parse HEAD'
```

### Calibrate, smoke-test, and run

Run every command below from the Sweden Central `agent-migrate` directory. Do not
start a formal run with a dirty tracked worktree.

```bash
source ~/.bashrc
mkdir -p /datadrive/queue-haul-network/control

uv run python queue-haul/network_campaign.py check \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs

uv run python queue-haul/network_campaign.py calibrate \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --out /datadrive/queue-haul-network/control/calibration.json
```

Formal calibration takes three 60-second repeats. It records 200 RTT samples per
path, isolated one- and eight-stream `iperf3`, simultaneous eight-stream
receiver goodput to both destinations, all raw iperf JSON, host fingerprints,
and clock uncertainty. Controlled 40% and 80% rates come from simultaneous—not
isolated—receiver goodput, with an aggregate source-NIC cap. Clock uncertainty
above 2 ms is a hard failure.

Run both an unshaped and shaped end-to-end gate before planning:

```bash
uv run python queue-haul/network_campaign.py smoke \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --calibration /datadrive/queue-haul-network/control/calibration.json \
  --bandwidth natural \
  --run-root /datadrive/queue-haul-network/smoke-natural

uv run python queue-haul/network_campaign.py smoke \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --calibration /datadrive/queue-haul-network/control/calibration.json \
  --bandwidth controlled_40 \
  --run-root /datadrive/queue-haul-network/smoke-controlled-40
```

Each smoke must prove nonzero KV wire bytes and cached tokens at both remote
destinations, then sleep and wake the source GPU. Use new smoke directories;
existing directories are rejected.

Prepare and run the targeted campaign:

```bash
uv run python queue-haul/network_campaign.py prepare \
  --cluster queue-haul/azure_network_cluster.json \
  --calibration /datadrive/queue-haul-network/control/calibration.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/plan.json

uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --current-calibration /datadrive/queue-haul-network/control/calibration.json \
  --plan /datadrive/queue-haul-network/control/plan.json \
  --run-root /datadrive/queue-haul-network/formal-001
```

The design is 7 targeted one-factor conditions x 3 repeats x 6 policies = 126
physical policy scenarios. The policies are Queue-Haul, greedy, Lagrangian
greedy, KV-only, replay-only, and seeded feasible random. The anchor is
interactive coding, controlled 80% bandwidth, idle sink, and 30-second
deadline; one cell each changes workload to coding or agentic, bandwidth to 40%
or natural, sink load to historical-throughput `rho=0.8`, or deadline to 19 s.
This replaces the 648-run full matrix, which adds interactions we do not need to
answer the present one-factor questions. Every policy reaches both destinations.
When a destination is unavailable, the same design can run against a pinned
one-destination cluster and calibration; every policy then reaches that node.

The runner checkpoints each scenario result atomically and fsyncs it before
continuing. `progress.json` records completed scenario IDs and counts after every
attempt. Rerun the same command and run root to skip every completed scenario;
an interrupted or failed `attempt-NNNN` remains intact and resume uses the next
attempt number. After any Spot deallocation, restart the VMs, rerun `setup.sh`
and `check`, write a fresh formal
calibration file, and resume with that file as `--current-calibration`. Resume
hard-fails if RTT or simultaneous goodput drifts more than 10%, or if the plan,
commit, node identity, model, or runtime changed. Azure Scheduled Events are
logged on all three hosts and an active Spot event fails the attempt.

The Azure campaign profile uses the Sweden A100 PCIe 300 W calibration retained
under `/datadrive/queue-haul-network/control/power-cal-300w-*`: 98.1 W
model-resident idle and 140.2 W at the profile's maximum expected load. Bare GPU
idle is outside this curve.
Reconstruction requests end with an explicit state-code probe and reserve 32
output tokens; a successful HTTP response without the code hard-fails the attempt.

Reduction runs automatically and can also be repeated without hardware:

```bash
uv run python queue-haul/network_campaign.py reduce \
  --plan /datadrive/queue-haul-network/control/plan.json \
  --run-root /datadrive/queue-haul-network/formal-001
sha256sum -c /datadrive/queue-haul-network/formal-001/artifacts.sha256
```

`summary.json` is valid only with all 126 latest attempts complete. `results.csv`
contains status, deadline, migration time, API/log bytes, KV bytes, TCP RTT, and
retransmissions per scenario. Every stack retains 250-ms directional byte logs,
per-connection duration/bytes/RTT/RTT variance/congestion window/retransmits,
per-RESP-transfer payload and wire bytes, source and destination GPU power,
service logs, scheduled events, and loaded-sink request traces. Scenario results
retain streaming chunks, prompt/cached token counts, TTFT timestamps, source
sleep timestamps, state-code validation, and every attempt. Power scope is GPU,
not whole-node. These files become evidence only after the hardware run passes;
their presence in the implementation is not a measurement claim.

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
uv run python queue-haul/greedy_lagrangian_experiment.py
uv run python queue-haul/simulated_pareto_campaign.py prepare
for shard in {0..63}; do
  uv run python queue-haul/simulated_pareto_campaign.py run-shard --shard "$shard"
done
uv run python queue-haul/simulated_pareto_campaign.py reduce
uv run python queue-haul/paper_evaluation.py \
  --out queue-haul/outputs/paper-evaluation
```

The exact 10K Pareto campaign includes Queue-Haul, greedy, isolated-fastest,
feasible-random, replay-only, and KV-only. The combinatorial
`greedy_lagrangian` recovery is excluded and remains a separate scale-limited
experiment. Trace-derived context anchors stop at the measured 31,562-token
prefill/decode boundary. `pareto-hero.png` shows one explicitly scoped example:
interactive-coding seed 1 at 10 Gb/s. Repeated identical frontier points are
collapsed, and the endpoint shared by all four frontier policies is labeled.

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
resource rows are assembled directly from candidate nonzeros, and maximum-gain
fallback uses a scale-relative normalized feasibility margin.
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
same LP but regenerates Python candidate physics on every pricing sweep.
`lp_column_generation_native` runs the identical Phase-I/Phase-II master and
certificate with a long-lived Rust pricing oracle while keeping HiGHS in Python.
The native boundary factors candidates into float64 session/signature features,
packed per-session feasibility masks, and distinct pool/method sparse templates;
bounded chunks load directly into Rust-owned storage. It is limited to 16
options and hard-fails outside that scope. `uv sync` installs it for development;
rustup Cargo must precede any system Cargo, and the toolchain is pinned under
`native/`. Pools reuse admission physics only when type, route, replica count,
baseline, bounds, and methods are exactly equal; variables and capacity rows
remain pool-specific. Indexed replica placement and a compact execution verifier are still
required for million-session operation.
`outputs/native-lp-scale-20260801/one-million.json` records the post-optimization
one-seed 1M-session LP/rounding sensitivity and its hashes; it explicitly excludes
replica packing, DES, prediction, and execution validation.
`greedy_lagrangian_experiment.py` compares the two supported greedies on paired
trace-derived targets; infeasible plans receive zero validated shed. Static
`greedy` fixes one scarcity price and one global candidate order.
`greedy_lagrangian` iterates aggregate-resource prices, retains a bounded set of
exact nonlinear source prefixes, performs target-capped recovery, and packs the
final set. Recovery caches sparse candidate columns and accumulates each retained
prefix without constructing sparse submatrices, and reuses those statistics in
packing fallback. Equal normal/emergency pool bounds reuse one admission solve.
Feasible-random groups each session's candidates in one pass, so its
setup is linear in candidate count. Immutable single-policy scale runs under `outputs/dual-lagrangian-*`
record the former `greedy_prefix` name; mixed bundles containing retired
optimizers were removed rather than rewritten.
Replica-packing repair considers actions from best to worst normalized resource
per watt, keeps feasible placements fixed, and rejects actions that cannot fit.
It reports rejected-action count and complete repair-pass time.
The campaign also computes a fractional source-chord LP lower bound on migration
work and plots each feasible greedy's excess work over that bound. The bound is
valid for the concave GPU-scoped power profile and relaxes integrality, replica
packing, and exact timing; it is not the unknown integral optimum.
The plot reports completion over every case and compares work only on the paired
common-feasible cohort to avoid survivor bias.
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
Warm prefetch permits concurrent destination L1 fills but requires a full L1
hit before inference.
Source-store completion uses unique RESP keys, not LMCache summary token counts.
Its expected key count comes from the request's retained-cache miss.
Traces with no catch-up plot a zero-length catch-up at the measured idle boundary.
Destination inference forbids L2 reads but may recompute an unstorable boundary chunk.
`plot_fixed_contract_residuals.py` caches the canonical 100K-session,
120-second fixed-contract requirement sweep and compares mixed greedy,
GPU-work-first, replay-only, and KV-only resource headroom at common requested
source-power shed levels. Use `--refresh` when its pinned inputs change.
`policy_hardware_campaign.py` creates a resumable paired idle-session campaign
that launches every session concurrently for Queue-Haul, both greedies,
KV-only, and replay-only. Static greedy retains its established hardware
planning path. Lagrangian greedy uses a one-pool idle dedicated-sink adapter,
then emits the same frozen move schema. Its default truncated grid uses coding,
interactive-coding, and
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
The pinned 2026-07-30 bundles predate `greedy_lagrangian`; they do not constitute
hardware evidence for it. A new two-A100 run is required for that claim.
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
`simulated_pareto_campaign.py` creates 64 deterministic shards for 14 exact
10K-session idle snapshots: three trace seeds for each workload and five
trace-derived context anchors. It compares Queue-Haul, static and Lagrangian
greedy, isolated-fastest, feasible-random, replay-only, and KV-only over fixed
1/2.5/5/10-Gbit/s site links and 30-second through four-hour deadlines. Replay
is a divisible destination-fleet service fitted from successful width-8 10G
episodes; width 8 is calibration evidence, not an execution cap. KV bytes use
both fixed WAN and per-replica ingest capacity. All methods use identical idle
destination packing and report simulated sensitivity, trailing-five-second
target attainment, last commit, censoring, source hashes, and dirty Git state.
Reduction hard-fails missing or duplicate shards and emits separate trace and
anchor small multiples. Conservative and optimistic fits run on sentinel cells;
the full grid uses the central fit. The pinned stage-span evidence fits
p25/median/p75 replay capacity factors of 0.963/0.984/1.026. These are used
without clamping: the lower two represent a small effective slowdown, while
width 8 remains only the upper validity bound.
`canonical_simulator_campaign.py` runs a four-target paired 10K-session
Queue-Haul, both greedies, per-session-fastest, replay-only, and KV-only comparison
under one assumed dedicated-pool contract. Its compact 10K/100K/1M scale check
uses both greedies, an equivalent pooled-destination topology,
10 Gbps per 10K sessions, and summary-only prediction. Sampled future requests
are disabled; measured two-A100 results provide continuation first-token
evidence.

Software results are authoritative for optimizer intent, aggregate feasibility,
and fleet-scale sensitivities. Hardware results are authoritative for measured
execution and timing of the frozen plan. A mismatch fails validation; neither
result silently overwrites the other.
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
