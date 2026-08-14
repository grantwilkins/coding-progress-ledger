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

- a measured GPT-OSS-20B/H100 NVL TP=1 occupancy and GPU-power curve;
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

`service_holdout_analysis.py` reproduces the prefill/decode staircase audit,
leaky context-bundle retrospective diagnostic, and matched-work
request-simulation falsification. It keeps pooled-token ITL separate from
per-request decode duration and reports why those traces do not yet support a
TTFT/ITL or decode-hold admission guarantee. Supplying `--powertrace-root`
adds the raw disaggregated GPT-OSS empirical sensitivity reference; it is
explicitly not used as a Queue-Haul latency model.

```bash
uv run python service_holdout_analysis.py \
  --powertrace-root ~/powertrace-sim \
  --out outputs/service-holdout-20260814/summary.json
```

The audit's primary profile is the A100 staircase profile; its default H100
comparison input is `profiles/gpt_oss_20b_h100_tp1.json`. The latter's 2026-08-11
H100 NVL measurements give `F=11415.78` prefill tok/s, `G=451.32` decode tok/s,
1,205,376 production KV-cache tokens, and a concave GPU-power envelope reaching 168.39 W
and measured through offered load `ell=12.566`. Admission remains bounded at
`ell=0.96647`. Raw benchmark and power samples are under
`outputs/h100-profile-20260811/`. Replay, KV-transfer, and transition timings
remain clearly marked A100-derived estimates until rerun on H100.

The separate boundary-aligned H100 power campaign retains 111 measured cells
(90 discovery, 18 unseen confirmation, and three idle anchors) with no cached
tokens or counter/window violations. Its frozen rational model is
`P=P0+A z/(1+z)`, `z=alpha f+beta g`, without a cross term. The unseen cells
have 2.37 W MAE and 4.34 W p90 absolute error; the fit remains explicitly
`holdout_failed` because its `R^2=0.942` misses the declared 0.95 gate. The
pooled parity view also applies that frozen fit to 38 valid cells from the prior
physical H100. That external-device cohort has 5.17 W MAE and `R^2=0.840`, so
device transfer is visibly weaker than the unseen same-device result.
Interrupted and offered-work-attributed sweeps are excluded.

```bash
uv run python queue-haul/plot_h100_power_parity.py \
  --run-root /datadrive/queue-haul-power/h100-realized-20260814-005 \
  --history-run-root /datadrive/queue-haul-power/h100-realized-20260814-004 \
  --out queue-haul/outputs/h100_power_model_parity
```

The completed 72-scenario H100 hardware-gap campaign has no failed or missing
runs. It scales the constrained East KV reserve to 96% of the measured
1,205,376-token capacity, reserves 65% of each destination's migration window
for KV, and requests 41.4% of removable source power. All three deadline-blind
controls missed 45 seconds and reached the target in 55.5--59.5 seconds.
The reduced evidence and raw scenario attempts are retained in
`outputs/east-germany-hardware-gap-h100-20260812/`, copied from node run root
`/datadrive/queue-haul-network/hardware-gap-h100-002`. Its artifact manifest
also addresses the uncommitted reusable stack logs.

## System boundary

A handoff prepares state in the background, quiesces at a request boundary,
performs final catch-up, switches routing, and succeeds when the destination
returns the first token. Mid-token migration, return migration, cold model
placement, unrelated destination arrivals, provider fleet policy, and facility
power are out of scope.
Destinations use a 32 GB bidirectional LMCache L1 so resumed sessions retain
and write back KV instead of remaining read-only migration consumers.
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

`repair_controller.py` is an optional in-memory feasibility latch; fixed planning
remains the default. Progress, route rate, replay rate, and observed per-pool
prefill capacity update the versioned ledger. Two consecutive deadline-miss
forecasts request one residual repair, while hard failures request one
immediately. `repair_destination` lands committed work on its concrete replica,
keeps explicitly locked running attempts fixed, prices their measured remainder
with the regional timing components, and minimizes changes to repairable work.
It proposes a diff only when that residual schedule restores the target;
otherwise it reports the attainable shed and leaves execution unchanged.

`repair_shadow_campaign.py` is the narrow RAMR validation: three seeded repeats
each of a sustained 10-to-1 Gbps cut, replay load from rho 0 to 0.8, and both.
It requests two A100-SXM4-80GB GPUs, records but does not apply proposed diffs,
and validates the trigger policy rather than live redirection or performance.

The artifact in `outputs/repair-plan-shift-sim-20260812/` is superseded: that
campaign independently solved degraded snapshots and cannot be interpreted as
within-plan repair evidence. `repair_plan_shift_campaign.py` now starts from the
passing regional A100 timing plan, runs one four-worker schedule to 25% aggregate
planned work, sends two observations through the real repair ledger, and applies
only target-restoring pending-work diffs. Its 16 cells cross bandwidth and
prefill-capacity locations in `{none, east, germany, both}` at 0.1x. Bandwidth-cut
cells remain explicitly labeled sensitivity evidence until the live 0.1x timing
gate passes. The replacement output is in
`outputs/repair-scheduled-sim-20260814/`.

`repair_hardware_campaign.py` prepares the three-region Azure run. A source-side
proxy changes live API/KV route rates without restarting connections, while a
destination gateway imposes an aggregate uncached-prefill completion cap on
background and replay requests. The job first runs 36 regional timing checks
(two destinations, two migration methods, three contexts, three repeats) at
0.1x bandwidth. It launches the 48 repair episodes only if median and p90
relative error are at most 15%, p90 absolute error is at most one second, and KV
reuse is verified. Every proposed diff is shadow-checked before application;
active work cannot be redirected, changes cannot move work toward an impaired
destination, and applied repairs must meet the measured hardware target.

The pinned launch bundle is `outputs/repair-scheduled-hardware-20260814/`:

```bash
export QH_AZURE_SSH_KEY=/path/to/azure-key
export QH_REPAIR_RUN_ROOT=/datadrive/queue-haul-repair-20260814
outputs/repair-scheduled-hardware-20260814/run.sh
```

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

The H100 path uses West US 3 (`10.11.0.4`) as source and Australia East
(`10.12.0.4`) plus South Central US (`10.13.0.4`) as destinations via
`azure_network_cluster_australia_southcentral.json`. Set
`QH_MODEL_PROFILE=gpt_oss_20b_h100_tp1.json`; the A100 profile remains the
network campaign default so archived plans retain their original meaning.

The node map across the provided cluster files is:

| role | region | private IP |
|---|---|---|
| source/power-down | Sweden Central | `10.0.0.4` |
| destination | East US 2 | `10.1.0.4` |
| destination | West Europe | `10.2.0.4` |
| optional destination | Germany West Central | `10.3.0.4` |

Use `azure_network_cluster_germany.json` for an isolated Germany run when West
Europe is unavailable, or `azure_network_cluster_east_germany.json` for the
two-destination East US 2 and Germany campaign.

`check` compares every selected entry with Azure IMDS and hard-fails before
calibration if an address assignment differs. Correct the selected node record;
do not bypass the check.

### One-time portal work for the Azure account owner

The account owner must complete these items. The experiment operator does not
need `az` permissions.

1. Use `Standard_NC24ads_A100_v4` Spot VMs with one visible A100 each,
   Azure Linux 3.0, persistent `/datadrive`, eviction policy `Deallocate`, and
   no delete-on-eviction data disk.
2. Configure bidirectional Global VNet Peering between the Sweden Central VNet and
   each selected destination VNet. Each peering must show `Connected`; address spaces
   must not overlap. Destination-to-destination peering is unnecessary.
3. Do not add a public data-plane address, NAT gateway, VPN, load balancer, or
   TLS terminator. SSH can use the existing private access path. Private Azure
   backbone traffic is not application-layer encryption; that is acceptable for
   this measurement-only, private-VNet deployment.
4. Restrict NSGs to these experiment flows. Source egress goes only to each
   destination on TCP `22,5201,8081,8200` and ICMP. Each selected destination
   permits source `10.0.0.4/32` on those ports. Sweden Central
   permits TCP `8301` from East `10.1.0.4/32`, West `10.2.0.4/32`, or Germany
   `10.3.0.4/32`, according to the selected isolated cluster. The joint cluster
   also uses TCP `8302` from West. Ports `5556,5557,5655,8080,8100,8401,8402` remain
   host-local. Do not expose any experiment port to `0.0.0.0/0`.
5. Confirm all selected VMs have the repository at
   `/home/azureuser/coding-progress-ledger/agent-migrate`, the same commit, and
   the source has `~/.ssh/azrs` plus verified host keys for each destination.

### Install each selected host

Run from the `agent-migrate` repository on every VM as `azureuser`, not root:

```bash
bash queue-haul/setup.sh
source ~/.bashrc
```

This installs Valkey, `chrony`, and `iperf3`, configures chrony against Azure's
stable `/dev/ptp_hyperv` device, waits for synchronization, installs the pinned
Python 3.12/vLLM 0.22.0/LMCache 0.5.1 CUDA 12.9 runtime, and stores the pinned
GPT-OSS-20B model and caches under `/datadrive`. Setup hard-fails without
`nvidia-smi`, the persistent data mount, PTP device, or pinned runtime.

From the Sweden Central source, establish and verify SSH host keys once, then confirm
that the same commit is checked out everywhere:

```bash
ssh -i ~/.ssh/azrs azureuser@10.1.0.4 true
ssh -i ~/.ssh/azrs azureuser@10.2.0.4 true
ssh -i ~/.ssh/azrs azureuser@10.3.0.4 true
git rev-parse HEAD
ssh -i ~/.ssh/azrs azureuser@10.1.0.4 'cd /home/azureuser/coding-progress-ledger/agent-migrate && git rev-parse HEAD'
ssh -i ~/.ssh/azrs azureuser@10.2.0.4 'cd /home/azureuser/coding-progress-ledger/agent-migrate && git rev-parse HEAD'
ssh -i ~/.ssh/azrs azureuser@10.3.0.4 'cd /home/azureuser/coding-progress-ledger/agent-migrate && git rev-parse HEAD'
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
# Validate one route first with 54 paired replay/KV migrations.
uv run python queue-haul/network_campaign.py prepare \
  --design isolated \
  --cluster queue-haul/azure_network_cluster_east.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-post-west-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/plan-east-validation.json

uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east.json \
  --ssh-key ~/.ssh/azrs \
  --current-calibration /datadrive/queue-haul-network/control/calibration-east-post-west-001.json \
  --plan /datadrive/queue-haul-network/control/plan-east-validation.json \
  --run-root /datadrive/queue-haul-network/validation-east-001

# Prepare the joint campaign after both routes validate.
uv run python queue-haul/network_campaign.py prepare \
  --design joint \
  --cluster queue-haul/azure_network_cluster.json \
  --calibration /datadrive/queue-haul-network/control/calibration.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/plan-joint.json

uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster.json \
  --ssh-key ~/.ssh/azrs \
  --current-calibration /datadrive/queue-haul-network/control/calibration.json \
  --plan /datadrive/queue-haul-network/control/plan-joint.json \
  --run-root /datadrive/queue-haul-network/formal-001
```

An isolated plan pairs replay and KV at 2K, 8K, and 32K contexts, route-relative
40%, 80%, and natural bandwidth, and three repeats: 54 migrations per site. The
West run uses `azure_network_cluster_west.json` with its one-path calibration.
Multi-site destination services start concurrently for each bandwidth stack.
Resume permits a synchronized commit update while keeping all other run identity
fields pinned and retaining both commits in the metadata checks.
joint design is 7 targeted agentic-trace conditions x 3 repeats x 6 policies =
126 physical scenarios. The policies are Queue-Haul, greedy, Lagrangian greedy,
KV-only, replay-only, and seeded feasible random. Its 20%-compute/20%-KV anchor
uses controlled 80% bandwidth and a 30-second deadline. The other cells are
idle destinations; crossed 20/40 and 40/20 compute/KV pressure in both
directions; controlled 40%; natural bandwidth; and a 19-second deadline.

Joint scenarios do not preassign a destination. After five seconds of seeded
background warmup and five one-second vLLM metric samples, the selected policy
chooses destination, replay or KV, and order for every session. Declared work
comes from pinned agentic turn rates and prefill/decode tokens; measured live KV
usage is also a planner input. Trace demand is normalized to the campaign's
existing total source-load contract of `ell=0.4`. A live-state deviation over five
percentage points or any waiting request is retained as a warning, not a failed
measurement. Missing metrics, invalid reconstruction, or missing KV evidence
remain hard failures.

The runner retries one malformed state-code probe, then hard-fails with its
response excerpt. It checkpoints each decision and scenario result atomically and fsyncs
them. `progress.json` records completed scenario IDs and counts after every
attempt. Rerun the same command and run root to skip every completed scenario;
an interrupted or failed `attempt-NNNN` remains intact and resume uses the next
attempt number. A failed attempt stops immediately instead of repeatedly cold
loading the models. After any Spot deallocation, restart the VMs, rerun `setup.sh`
and `check`, write a fresh formal
calibration file, and resume with that file as `--current-calibration`. Resume
hard-fails if RTT or simultaneous goodput drifts more than 10%, or if the plan,
node identity, model, or runtime changed. A synchronized commit update remains
visible in the audit checks. Azure Scheduled Events are
logged on all three hosts and an active Spot event fails the attempt.

The handoff accepts any validated two-destination cluster. For East US 2 and
Germany West Central, create one fresh calibration and matched plan after East
comes online, then pass the same repeat to Queue-Haul, KV-only, and replay-only:

```bash
uv run python queue-haul/network_campaign.py check \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --ssh-key ~/.ssh/azrs
uv run python queue-haul/network_campaign.py calibrate \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --ssh-key ~/.ssh/azrs \
  --out /datadrive/queue-haul-network/control/calibration-east-germany-001.json
uv run python queue-haul/network_campaign.py prepare --design joint \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/plan-east-germany-001.json
uv run python queue-haul/network_campaign.py smoke \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --ssh-key ~/.ssh/azrs \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --bandwidth natural \
  --run-root /datadrive/queue-haul-network/smoke-east-germany-natural-001

for policy in queue_haul kv_only replay_only; do
  uv run python queue-haul/network_campaign.py handoff \
    --cluster queue-haul/azure_network_cluster_east_germany.json \
    --ssh-key ~/.ssh/azrs \
    --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
    --plan /datadrive/queue-haul-network/control/plan-east-germany-001.json \
    --manifest queue-haul/outputs/coding-manifest.json \
    --policy "$policy" --repeat 0 \
    --run-root "/datadrive/queue-haul-network/handoff-east-germany-$policy-001"
  uv run python queue-haul/plot_handoff_power.py \
    --run-root "/datadrive/queue-haul-network/handoff-east-germany-$policy-001"
done
```

The East/Germany frontier campaign uses measured natural bandwidth only. Its
185-episode pilot is six movement packs (4x16K, 8x16K, 16x16K, 8x8K, 8x24K,
and 8x31K), destination loads 0, 0.5, 0.85, 0.9, and 0.95, plus the seven-cell
8x16K asymmetric slice. Every matched cell runs Queue-Haul LP, greedy,
replay-only, KV-only, and power-blind Queue-Haul against a 30-second deadline
and an 80% modeled removable-power target. The source load is 80%; replay
requests explicitly bypass LMCache and KV requests require positive cache
evidence only as a warning.

The H100 frontier keeps the 4x16K, 8x16K, and 16x16K width bridge, expands the
8K, 24K, and 31K packs to width 16, and adds one 32x31K red-zone tail. Its 304
matched scenarios cover Queue-Haul, greedy, Lagrangian greedy,
isolated-fastest, KV-only, replay-only, power-blind, and deadline-blind using
only the two measured natural WAN paths; no bandwidth cap or fixed destination
split is applied. H100 destination load is normalized to the measured
604-prefill/64-decode service rates used by the live background generator.

```bash
uv run python queue-haul/network_campaign.py prepare --design frontier \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/frontier-pilot.json
uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --current-calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --plan /datadrive/queue-haul-network/control/frontier-pilot.json \
  --run-root /datadrive/queue-haul-network/frontier-pilot-001
uv run python queue-haul/network_campaign.py refine \
  --plan /datadrive/queue-haul-network/control/frontier-pilot.json \
  --run-root /datadrive/queue-haul-network/frontier-pilot-001 \
  --out /datadrive/queue-haul-network/control/frontier-refinement.json
uv run python queue-haul/network_campaign.py deadline-blind \
  --plan /datadrive/queue-haul-network/control/frontier-pilot.json \
  --plan /datadrive/queue-haul-network/control/frontier-refinement.json \
  --out /datadrive/queue-haul-network/control/frontier-deadline-blind.json
```

Reduction writes raw episode CSV plus PNG/PDF prefill--network mechanism and
target-attainment figures. Refinement adds 0.875 or 0.925 load midpoints where
an action or attainment boundary appears and caps the first adaptive stage at
65 matched episodes, for 250 pilot-plus-refinement episodes total. New midpoint
cells receive repeats first. A second refinement takes unstable cells to ten
repeats when the action changes or the 95% shed-width exceeds 10 W. Deadline
misses, target misses, individual request failures, load drift, queueing, and
missing secondary telemetry remain observations. Invalid identity or inputs,
unusable primary outcomes, and failure of more than half of the planned
episodes stop the campaign.

The deadline-blind plan selects one Queue-Haul ablation for every unique
condition/repeat block in the pilot and capped refinement. It plans against a
nonbinding 600-second horizon while execution and attainment retain the measured
30-second deadline.

`plot_network_power_attainment_cdf.py` combines the pilot, refinement, and
deadline-blind phases into matched 50-episode-per-policy ECDFs. Each event is the
earliest common-epoch completion at which nonlinear modeled source-power shed
reaches the 80% target; late events remain and unattained targets are missing
mass. Run `uv run python queue-haul/plot_network_power_attainment_cdf.py`; it
writes CSV, PNG, and PDF outputs to the frontier campaign root using Tab10
policy colors and the longest observed episode runtime as the plotting horizon.

The constrained East/Germany diagnostic is one frozen 24-episode run, not a
sweep. It uses the measured simultaneous natural paths (2.280 Gb/s East and
8.733 Gb/s Germany), 80% source load, and the original generator's 50% East
and 95% Germany source-rate-normalized load labels. Those labels are not
destination service utilization, so this campaign remains a migration-window
capacity diagnostic rather than evidence that destination service binds. Its
19-second cell has 22 exact recorded-support contexts selected with disclosed
seed 15 (513,650 tokens); its 30-second cell has 28 contexts from seed 8
(648,131 tokens); and its 60-second cell has eight copies of each trace at
14,042 tokens (898,688 tokens). Every cell requests the full 61.86 W removable
pack power, so the attained result is a deadline-constrained capacity point
rather than an easy target. The fourth cell reuses the exact 30-second pack
while limiting Germany replay to 25% of its migration window, or 6.25
replica-seconds. This quota is an operator counterfactual, not a measured
load-slowdown coefficient.

Every cell runs exact maximum-shed Queue-Haul, Queue-Haul greedy, KV-only,
replay-only, per-session-fastest, and power-blind Queue-Haul once. The exact
binary solver jointly chooses sessions, methods, and destinations to maximize
removed single-source load, then minimizes migration work at that optimum. In
the pinned simulation at `outputs/east-germany-constraint-20260808/`, it sheds
49.25, 55.92, 60.48, and 51.69 W at 19, 30, 60, and quota-constrained 30
seconds. Greedy reaches 46.99, 51.69, 60.48, and 51.69 W. The full-pack request
is intentionally unattainable in every cell. Under the replay quota,
Queue-Haul chooses 5 KV→Germany, 7 replay→East, 3 KV→East, and 1
replay→Germany. All four method--destination migration windows have positive
Phase-I additive-surrogate prices; direct route and destination-service prices
are zero. Exact nonlinear bundle shed is recomputed after integral packing.

```bash
uv run python queue-haul/network_campaign.py prepare --design constraint \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/constraint.json
uv run python queue-haul/network_campaign.py simulate-constraint \
  --plan /datadrive/queue-haul-network/control/constraint.json \
  --out /datadrive/queue-haul-network/constraint-simulation
uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --current-calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --plan /datadrive/queue-haul-network/control/constraint.json \
  --run-root /datadrive/queue-haul-network/constraint-001
```

The runner keeps collecting after an episode failure but hard-fails after final
reduction unless all 24 episodes complete and meet their deadline, every
intentionally oversized target remains unmet, and no episode contains a
request, KV-evidence, load-drift, or queueing warning.
Migration timing ends when parallel reconstruction finishes; draining background
load happens afterward and is excluded from `migration_s` and `deadline_met`.
Each background generator caps pending work at its eight request workers, so
overload cannot create a stale client-side queue that lengthens episode cleanup.
Reduction writes matched Tab10 attainment and action-composition figures; the
simulator additionally writes the Phase-I dual table and figure. There is no
adaptive refinement or CDF for this single-block diagnostic.

The separation campaign is the service-binding follow-up. It pins the same
28-session recorded-support pack (648,131 tokens, seed 8) in every matched
cell, uses measured destination prefill and decode rates to generate exact
604-prompt/64-output-token background requests, warms that traffic for 30
seconds, and then launches all selected migrations concurrently. Each policy
plans inside 30 seconds and is measured against a 45-second hardware deadline.
Background prompts stay fixed while unique cache salts prevent reuse, and
forced-length decoding makes every service-normalized request exactly 604/64.
Attainment credits only reconstructions ending by the common migration
start plus 45 seconds; reduction recomputes this value from raw request
timestamps, including for results written by an earlier runner revision.

| cell | paths | East/Germany service load | target | Queue-Haul | greedy | strongest losing baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Germany service | natural | 25% / 95% | 37.11 W | 47.72 W | 41.77 W | 32.51 W |
| East slow path | natural | 90% / 25% | 47.63 W | 55.92 W | 53.72 W | 41.88 W |
| joint shaped | 0.9 / 3.4 Gb/s | 50% / 85% | 33.40 W | 51.06 W | 37.22 W | 29.72 W |

The 63 episodes are three cells by three repeats by seven policies: exact
maximum-shed Queue-Haul, target-aware greedy, KV-only, replay-only,
per-session-fastest, power-blind, and deadline-blind Queue-Haul. The simulator
hard-fails unless Queue-Haul and greedy both use at least two KV and two replay
actions, span both destinations, and attain at least 110% of target; each of
the five ablations must remain at or below 90%. Deadline-blind must claim feasibility at
600 seconds and then fail the 45-second evaluation. At least three Queue-Haul
resources per cell must reach 89--98% utilization, including Germany service
at 98%, all four action windows in the East cell at 92--98%, and both services
plus all four action windows in the joint cell at 89--97%. Their Phase-I duals
must also be positive, except for degenerate East-service pricing. Thus the
certified separation is at least 20 percentage points, with 50% extra hardware
time beyond the planning window; the hardware reducer applies the same attainment gates and also
requires clean requests, cache evidence, destination load, and queue telemetry.

```bash
uv run python queue-haul/network_campaign.py prepare --design separation \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --manifest queue-haul/outputs/coding-manifest.json \
  --out /datadrive/queue-haul-network/control/separation.json
uv run python queue-haul/network_campaign.py simulate-separation \
  --plan /datadrive/queue-haul-network/control/separation.json \
  --out /datadrive/queue-haul-network/separation-simulation
uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --current-calibration /datadrive/queue-haul-network/control/calibration-east-germany-001.json \
  --plan /datadrive/queue-haul-network/control/separation.json \
  --run-root /datadrive/queue-haul-network/separation-001
```

The frozen local certificate is in
`outputs/east-germany-separation-20260809/`. Simulation proves the selected
model points have the intended separation. The completed 63-episode A100
hardware run had no request or load failures. Across Germany-service,
East-slow-path, and joint-shaped cells, Queue-Haul's median attainment was
128.6%, 117.4%, and 152.9%; greedy reached 112.5%, 112.8%, and 111.4%.
KV-only, replay-only, isolated-fastest, and power-blind stayed below 90% in
every cell. Deadline-blind finished the full move set after 45 seconds in all
three cells, but the joint-shaped cell already crossed its power target, so it
is not a clean deadline-blind negative control. One greedy repeat also lacked
clean KV evidence. Consequently all 63 executions completed, but the strict
certificate is invalid in nine episodes. The measured run further confirms
that destination KV occupancy was near zero, leaving KV capacity untested.
The compact reduction, plots, run identity, and verified 438-entry raw-artifact
manifest are retained in `outputs/east-germany-separation-hardware-20260809/`;
the large per-request and power traces remain in the hardware run root.

`simulate-oracle-stale` is the no-new-campaign constraint and stale-information
certificate. It reuses that exact 28-session recorded pack, measured East and
Germany paths, measured destination service profile, and the measured 75%
service-load support point. The constructed all-bind corner reserves 90% of
East's profiled A100 KV capacity, applies 75% Germany service load, and uses
40%-controlled caps derived from the measured paths (0.9/3.4 Gbit/s). The
request is 40.21 W, or 65% of removable pack power. Exact restricted max-shed
oracles all use forced normal admission and change only the allowed action or
destination: Queue-Haul reaches 49.13 W, versus 27.31 W for KV only, 30.90 W
for replay only, 22.73 W for East only, and 24.31 W for Germany only. East KV
and Germany service reach 95% and 98% utilization and both have positive
Phase-I duals.

The eight corners independently release East KV, Germany service, and path
bandwidth. A single exact worst-corner plan reaches 49.13 W by the deadline in
all eight, while fresh replanning reaches 49.13--55.92 W. The all-release plan
is capacity-invalid at the all-bind corner because it exceeds both East KV and
Germany service. As a negative control, Germany-only reaches 41.88 W once all
constraints are released. Deadline-aware Queue-Haul crosses 40.21 W at 23.33
seconds; the 90-second deadline-blind ablation eventually reaches 54.20 W but
has only 31.37 W at 45 seconds and first crosses the same target at 58.19
seconds. Generated Tab10 figures, exact moves, nonlinear attainment curves,
resource use, duals, and checksums are in
`outputs/east-germany-oracle-stale-20260809/`.

```bash
uv run python queue-haul/network_campaign.py simulate-oracle-stale \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --out queue-haul/outputs/east-germany-oracle-stale-20260809
```

The 72-episode hardware-gap follow-up turns those missing mechanisms into five
matched operating points using the same 28-session recorded pack, profiled
destination service, source-power model, and measured routes. Its all-bind
point physically limits East to 10% of profiled vLLM KV blocks, loads Germany
to 75% service utilization, and applies the measured 40% route caps. It tests
Queue-Haul against greedy, four exact restricted oracles, isolated-fastest,
power-blind, deadline-blind, and a frozen all-release plan. One-at-a-time
release controls restore East KV, Germany service, or path bandwidth; the
all-release control admits the previously stale plan. Every block has three
matched repeats, forced normal admission, and concurrent migrations. The fixed
worst-corner robust plan also runs in every released state, so its cross-state
guarantee is hardware-tested rather than inferred only from simulation.

The frozen simulation requests 44.54 W, or 72% of removable pack power.
All-bind Queue-Haul reaches 49.13 W while every losing baseline reaches at most
39.60 W; the stale plan is rejected because East KV and Germany service both
overflow. Deadline-blind reaches enough eventual power only after 59.31
seconds. Releasing KV, service, and bandwidth expands the corresponding exact
restricted oracle by 4.58, 12.91, and 7.44 W. The runner parses each vLLM
engine's reported KV-token capacity and hard-fails unless it matches the
planned fraction within one percentage point, preventing a labeled-only quota.
Planning additionally reserves 668 KV tokens per destination for one continuing
background request. Eventual attainment stops at the frozen 90-second horizon;
responses after it cannot satisfy the deadline-blind control.

```bash
uv run python queue-haul/network_campaign.py hardware-gap \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --oracle-plans queue-haul/outputs/east-germany-oracle-stale-20260809/plans.json \
  --out queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json
uv run python queue-haul/network_campaign.py simulate-hardware-gap \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-hardware-gap-20260809/simulation
uv run python queue-haul/network_campaign.py run \
  --cluster queue-haul/azure_network_cluster_east_germany.json \
  --current-calibration queue-haul/outputs/east-germany-frontier-20260808/control/calibration-east-germany-frontier-001.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --run-root /datadrive/queue-haul-network/hardware-gap-001
```

The completed 72-episode A100 hardware run had no failed or missing scenarios,
and all 585 retained raw artifacts verify. At all-bind, robust Queue-Haul shed
49.13 W (110.3% of target) in 24.56 seconds, while greedy shed 39.60 W
(88.9%). KV-only, replay-only, East-only, Germany-only, isolated-fastest, and
power-blind all stayed below target. Releasing KV, service, and bandwidth
increased their corresponding restricted oracle by 4.58, 12.91, and 7.43 W;
the robust plan continued to shed 49.13 W in every state. The stale plan was
rejected at all-bind and reached 55.92 W at all-release. Deadline-blind was not
a valid negative control: all three repeats crossed the target in 38.89--43.29
seconds, before the 45-second cutoff, although their full migrations ended in
54.37--56.67 seconds. The strict reducer therefore reports three invalid
episodes. The compact reduction, plots, run identity, and verified raw-artifact
manifest are retained in
`outputs/east-germany-hardware-gap-hardware-20260810/`; large traces remain in
the raw run root.

`plot_hardware_constraint_timeline.py` reconstructs the all-bind repeat-0
Queue-Haul, power-blind, and deadline-blind resource accounting. KV and service
curves are residual headroom consumed after measured cutover; migration curves
are modeled work charged to Queue-Haul's 30-second planning budget in measured
completion order, not sampled instantaneous utilization.

```bash
uv run python queue-haul/plot_hardware_constraint_timeline.py \
  --raw-root /datadrive/queue-haul-network/hardware-gap-001 \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-hardware-gap-hardware-20260810/constraint_timeline
```

The requested-shed frontier is a model sweep over that frozen all-bind hardware
scenario, not additional hardware observations. It retains raw overshed above
the requested-equals-attained diagonal and carries the last safe attainment
forward when a larger request is unsafe. Its second panel reports Queue-Haul
LP's modeled resource pressure and action/destination mix; those are diagnostic
rather than the primary comparison.

```bash
uv run python queue-haul/plot_hardware_shed_frontier.py \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-hardware-gap-frontier-20260810/shed_frontier
```

The pooled publication view removes those diagnostic panels and standardizes
twelve constraint, separation, and hardware-gap operating points to a common
30-second cutoff. Requested and attained shed are normalized by each case's
removable power before equal-weight pooling; raw overshed remains visible above
the diagonal. Lines are medians and ribbons are the interquartile spread across
designed cases, not repeated-run confidence intervals. Deadline-blind plans
against 90 seconds but receives credit only for shed attained by the common
30-second cutoff. The x-axis ends at 80%, just beyond the nonzero intersection
of Queue-Haul LP's median frontier with requested-equals-attained parity; the
y-axis retains the full 0--100% removable-power range.

The companion attainment-time ECDF uses the same twelve cases at the common
67% stress point. Each event is the first modeled target crossing plus the
profile's power-window delay; misses retain missing CDF mass. The 30-second
line is the common evaluation deadline, while the 90-second horizon exposes
late deadline-blind attainment:

```bash
uv run python queue-haul/plot_pooled_attainment_cdf.py \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-pooled-shed-frontier-20260810/pooled_attainment_cdf
```

The H100 counterpart pools the five completed hardware-gap states using the
measured H100 profile. Its star is the median of 15 matched Queue-Haul robust
hardware runs at the common 41.4% request; the curves remain modeled 30-second
frontiers rather than interpolated hardware measurements:

```bash
QH_MODEL_PROFILE=gpt_oss_20b_h100_tp1.json uv run python \
  queue-haul/plot_pooled_shed_frontier.py \
  --plan queue-haul/outputs/east-germany-hardware-gap-h100-20260812/plan.json \
  --hardware-results queue-haul/outputs/east-germany-hardware-gap-h100-20260812/results.csv \
  --out queue-haul/outputs/east-germany-pooled-shed-frontier-h100-20260812/pooled_shed_frontier
```

```bash
uv run python queue-haul/plot_pooled_shed_frontier.py \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-pooled-shed-frontier-20260810/pooled_shed_frontier
```

The companion hardware target-attainment plot retains uncapped overshed for the
six deadline-safe policies in the 63 separation episodes. It divides realized
shed by requested shed within each episode before pooling the three repeats, so
values above 100% explicitly show target overshoot rather than extra physical
efficiency. Deadline-blind is omitted because its recorded shed is eventual,
not shed attained by the deadline.

The matched A100 migration-timing parity view compares the pre-run modeled
episode makespan with the measured hardware migration duration for Queue-Haul
LP and Queue-Haul Greedy across the nine separation conditions/repeats:

```bash
uv run python queue-haul/plot_migration_timing_parity.py \
  --predictions queue-haul/outputs/east-germany-separation-20260809/simulation/separation_predictions.csv \
  --measurements queue-haul/outputs/east-germany-separation-hardware-20260809/results.csv \
  --out queue-haul/outputs/east-germany-migration-timing-parity-20260813/migration_timing_parity
```

```bash
uv run python queue-haul/plot_hardware_target_attainment.py \
  --results queue-haul/outputs/east-germany-separation-hardware-20260809/results.csv \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --out queue-haul/outputs/east-germany-hardware-target-attainment-20260810/target_attainment
```

The companion resource-pressure view compares the same cases at two-thirds of
removable power, where Queue-Haul usually succeeds and the restricted policies
usually fail. Its four facets sum physical use and capacity across destinations
for prefill service, KV headroom, replay time, and bandwidth-sensitive KV
transfer time. Points are designed cases, filled when the target is met by 30
seconds; diamonds and whiskers are the equal-case mean and 95% case-bootstrap
interval.

```bash
uv run python queue-haul/plot_pooled_resource_pressure.py \
  --cases queue-haul/outputs/east-germany-pooled-shed-frontier-20260810/pooled_shed_frontier_cases.csv \
  --out queue-haul/outputs/east-germany-pooled-resource-pressure-20260810/resource_pressure
```

The action-adaptation views use the same equal-case sweep. The primary chart
shows Queue-Haul's total replay/KV composition under all eight combinations of
HBM, bandwidth, and prefill constraints at a common 67% target. Gray reports
sessions left at the source, so every 100%-stacked bar accounts for the same
28-session pack; destination identities are intentionally omitted.

```bash
uv run python queue-haul/plot_pooled_action_adaptation.py \
  --cases queue-haul/outputs/east-germany-pooled-shed-frontier-20260810/pooled_shed_frontier_cases.csv \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out-dir queue-haul/outputs/east-germany-action-adaptation-20260811
```

`bootstrap_action_adaptation.py` repeats those eight matched cases under 1,000
paired calibration draws. It stratifies timing by destination, method,
bandwidth, and context and samples each fitted phase-power tuple jointly. This
is a modeled calibration-sensitivity distribution for the fixed 28-session
pack, not 8,000 independent hardware observations. The stacked bars show joint-
bootstrap mean shares; black whiskers mark the 5--95% Replay and total-moved
boundaries, with a three-facet interval companion for the full composition.

```bash
uv run python queue-haul/bootstrap_action_adaptation.py \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out-dir queue-haul/outputs/east-germany-action-adaptation-20260811
```

The time-to-binding view uses the same two-thirds stress point. It estimates
completion-ordered slack for VRAM, network-transfer, and prefill constraints;
each class reports its tightest component without exposing destination names.
Thin step curves are the twelve cases, thick curves are policy medians, and a
cross marks the first time a case reaches at most 5% residual slack. The
deadline-blind trajectory likewise shows only its first 30 seconds.

```bash
uv run python queue-haul/plot_pooled_resource_slack.py \
  --plan queue-haul/outputs/east-germany-constraint-20260808/plan.json \
  --plan queue-haul/outputs/east-germany-separation-20260809/plan.json \
  --plan queue-haul/outputs/east-germany-hardware-gap-20260809/plan.json \
  --out queue-haul/outputs/east-germany-pooled-resource-slack-20260810/resource_slack
```

In the separate standard handoff experiment, each policy uses the same eight
pinned agentic sessions. An independent 80%-load stream serves on Sweden while
both destinations sustain 50% background inference.
The 30-second handoff clock includes live metrics, policy planning, and parallel
reconstruction, and hard-fails unless all sessions are admitted and complete.
Destination background service never pauses. At the traffic switch, destination
service starts immediately while source admissions stop and Sweden drains and
sleeps concurrently. Defaults retain five-minute pre/post windows and 100-ms
power sampling.

Handoff processes pin `kv_both`, 33 GB L1 pools, a 32 GB Redis cap, and disabled
vLLM prefix caching. Unrelated source and destination loads bypass LMCache so
they cannot evict migration state. The reducer shades migration, switch, and
source power-fall windows, writes phase queue depth, and requires Sweden to
start at or above 200 W and shed at least 50 W. The measured Sweden window
itself populates the migrating KV state; there is no separate warmup before it.

Only the Azure campaign uses
`profiles/gpt_oss_20b_a100_tp1_azure_300w.json`, calibrated on an Azure
NVIDIA A100 80GB PCIe whose `nvidia-smi` power limit is 300 W. The generic
`gpt_oss_20b_a100_tp1.json` retains the original non-Azure A100 calibration.
The Azure profile uses the cache-cold fixed-rate sweep at
`/datadrive/queue-haul-network/control/power-cal-300w-rate-001`: 18 rates from
0.25 to 20 requests/s, 20-second windows, a 1,100-word synthetic prompt body
with a unique leading hash, 64-token outputs, and zero cached prompt tokens. Its
conservative concave envelope reaches 300.24 W at `ell=10.0543`; the 14--20
requests/s deep-queue points independently remain near 299 W. The 98.1 W
model-resident idle anchor comes from
`power-cal-300w-002`; bare GPU idle is outside this curve. Reproduce and reduce
the sweep with the commands below. The runner hard-fails unless `nvidia-smi`
reports exactly one NVIDIA A100 80GB PCIe with a 300 W power limit.

```bash
uv run python power_rate_sweep.py --out PATH --window-s 20 --warmup-s 5 --workers 512 --rates 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 12 14 16 20
uv run python power_rate_sweep.py --out PATH --reduce-only --prefill-capacity-tps 1448.32 --decode-capacity-tps 1260.38 --idle-power-w 98.11623555 --curve-max-rate 12
```

The scalar Azure curve remains estimated evidence. The promotion campaign uses
`phase_power_calibration.py` to measure five prefill/decode mixtures at six load
levels and three repeats, fit `z = a f + b g`, and validate by holding out whole
mixtures. A v5 profile keeps destination service load (`f/F + g/G`) separate
from phase-aware power load and hard-fails outside the calibrated `(f,g)` hull.

```bash
uv run python phase_power_calibration.py prepare --out /datadrive/queue-haul-network/phase-power-v1
uv run python phase_power_calibration.py run --plan /datadrive/queue-haul-network/phase-power-v1/plan.json --profile profiles/gpt_oss_20b_a100_tp1_azure_300w.json --out /datadrive/queue-haul-network/phase-power-v1/run
uv run python phase_power_calibration.py fit --base-profile profiles/gpt_oss_20b_a100_tp1_azure_300w.json --measurements /datadrive/queue-haul-network/phase-power-v1/run/measurements.csv --idle-power-w 98.11623555 --out-profile profiles/gpt_oss_20b_a100_tp1_azure_300w_phase.json --summary outputs/azure-calibration/power-summary.json
uv run python testbed_calibration_campaign.py prepare --parent outputs/east-germany-separation-20260809/plan.json --out outputs/azure-calibration/testbed-plan.json
```

`evidence_catalog.py` writes an immutable-raw, checksum-bound sidecar at
`/datadrive/queue-haul-network/evidence-catalog.json`. Existing
`realized_shed_w` hardware-gap fields are cataloged as `model_credited`;
`trailing-power` separately derives direct five-second Sweden power windows.
The final `stress_frontier_campaign.py` plan contains 40 equal-weight states and
runs Queue-Haul, six baselines, and the exact modeled MILP optimum independently
at 10--60 second deadlines. Reduction uses the fifth-smallest of 40 values and
automatically retains the title “Modeled stress-suite sensitivity” until every
power, timing, correctness, provenance, transition, and hardware-window gate
passes.

```bash
uv run python stress_frontier_campaign.py prepare --parent outputs/east-germany-separation-20260809/plan.json --profile profiles/gpt_oss_20b_a100_tp1_azure_300w_phase.json --out outputs/azure-calibration/stress-plan.json
uv run python stress_frontier_campaign.py run --plan outputs/azure-calibration/stress-plan.json --shard 0 --shards 40 --out outputs/azure-calibration/stress-00.csv
uv run python stress_frontier_campaign.py reduce --results outputs/azure-calibration/stress-*.csv --profile profiles/gpt_oss_20b_a100_tp1_azure_300w_phase.json --power-summary outputs/azure-calibration/power-summary.json --destination-summary outputs/azure-calibration/destination-summary.json --trailing-power /datadrive/queue-haul-network/hardware-gap-001/trailing_power.csv --catalog /datadrive/queue-haul-network/evidence-catalog.json --out outputs/azure-calibration/frontier.json
```

`outputs/network-campaign-20260805` retains the complete 54/54 East and West
single-link campaigns plus the successful `handoff-009` and bidirectional-cache
`handoff-010` three-node evidence, including raw 100-ms power, request,
transfer, decision, plot, and checksum artifacts. It also retains
`joint-queue-002-partial-086`: 86/126 completed joint scenarios, three raw stack
captures, and interrupted-attempt audit evidence. The first 17 scenarios used
consumer-only destination caches; the following 69 used bidirectional caches.
Reconstruction requests end with an explicit state-code probe and reserve 128
output tokens; a successful HTTP response without the code hard-fails the attempt.
The 600-second HTTP timeout is independent of the measured scenario deadline;
slow reconstruction completes with `deadline_met=false` instead of restarting.

`plot_network_action_breakdown.py` writes stacked bars for the completed joint
planner's Queue-Haul LP, Queue-Haul Greedy, KV-only, and replay-only recorded
destination-method selections. Run
`uv run python queue-haul/plot_network_action_breakdown.py`.

`plot_hardware_power_parity.py` audits the 840 matched Queue-Haul LP, Queue-Haul
Greedy, True Greedy, KV-only, replay-only, power-blind, and deadline-blind raw
traces. These runs do not provide the required settled pre-migration window:
the median gap from the final source request to migration is 0.075 s, and only
3/840 episodes cover a one-second window plus a one-second settling guard. The
script therefore hard-fails direct parity reduction instead of averaging
migration warm-up power. `--audit-only --out PATH` writes the episode-level
window audit without claiming measured shed. `--raw-delta --out PATH` writes an
explicitly `warmup_contaminated_immediate_pre` exploratory delta from the final
second before migration to the one-second post-switch window after a one-second
guard. A new hardware run with an explicit pre-migration hold is required for a
settled-state parity claim.

`power_parity_experiment.py` supplies that focused run without campaign controls:
50 matched random migration sizes per policy (350 runs total), a five-second
settled source window at load 0.4, and a five-second settled window at the
remaining source load. It reduces the direct GPU-0 means against the pinned
RAMR A100 power curve and writes the CSV plus y=x PNG/PDF.

```bash
uv run python queue-haul/power_parity_experiment.py prepare \
  --source-plan queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json \
  --out queue-haul/outputs/power-parity-random-plan
sbatch queue-haul/outputs/power-parity-random-plan/run.sbatch
```

The completed 350-scenario run is retained in scratch. The descriptive
phase-aware refit under `outputs/power-parity-phase-aware-20260813/` uses the
same observations for fitting and parity, so it is not held-out evidence. Its
shed regression has a 0.997 through-origin slope, effectively zero aggregate
bias, and 8.66 W RMSE; grouped five-fold episode cross-validation retains a
0.999 slope, 0.11 W bias, and 8.78 W RMSE. The CSV preserves all policy repeats.
The parity x-axis reports shed from the fitted
`P0 + delta_p * z / (1 + z)` model with `z = a*f + b*g`.
The publication plot shows Queue-Haul LP and Queue-Haul Greedy; its CSV retains
all seven measured policy arms.

Reduction runs automatically and can also be repeated without hardware:

```bash
uv run python queue-haul/network_campaign.py reduce \
  --plan /datadrive/queue-haul-network/control/plan.json \
  --run-root /datadrive/queue-haul-network/formal-001
sha256sum -c /datadrive/queue-haul-network/formal-001/artifacts.sha256
```

`summary.json` is valid only with all planned latest attempts complete. `results.csv`
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
timing CDFs, including a 30-second full-target attainment CDF whose event time
includes the trailing five-second power window; missing mass is deadline
failure. The same 30-second cohort also produces standalone bandwidth plots for
episode attainment and Queue-Haul's deadline-admitted replay/KV action mix.
Power attainment is trailing-five-second average modeled source-power
shed divided by the 100% source-power target. MP
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
migration-to-destination-first-token CDF, CDFs of the slowest session in each
complete episode before and after normalization by watts shed, and median
modeled source-power shed over
elapsed time with an interquartile band, plus paired attainment–completion
points and a CDF of measured session downtime per modeled watt shed. This idle
evidence also includes an episode migration-makespan-per-modeled-watt CDF and
supports timing and projected, not realized, power attainment.
`plot-reduced --pooled-with` adds supplied reduced campaigns to every pooled,
bandwidth, condition, attainment, power, and Pareto plot.
The pinned 2026-07-30 bundles predate `greedy_lagrangian`; they do not constitute
hardware evidence for it. A new two-A100 run is required for that claim.
Matched reruns of either reduced bundle use its frozen plan as the cohort source.
The three network baselines use resource-aware per-session fastest, uniform-gain
power-blind LP, and a 600-second deadline-blind planning horizon while retaining
the source plan's 19/30-second scoring deadlines:

```bash
uv run python queue-haul/policy_hardware_campaign.py prepare-baselines \
  --source-plan queue-haul/outputs/policy-hardware-width8-frontier-20260730/plan.json \
  --model-profile queue-haul/profiles/gpt_oss_20b_a100_tp1_20260730.json \
  --out queue-haul/outputs/policy-hardware-width8-frontier-network-baselines-plan
uv run python queue-haul/policy_hardware_campaign.py prepare-baselines \
  --source-plan queue-haul/outputs/policy-hardware-width8-packing-20260730/plan.json \
  --out queue-haul/outputs/policy-hardware-width8-packing-network-baselines-plan
uv run python queue-haul/policy_hardware_campaign.py prepare-baselines \
  --source-plan queue-haul/outputs/policy-hardware-width8-packing-20260730/plan.json \
  --policies queue_haul greedy isolated_fastest queue_haul_power_blind queue_haul_deadline_blind \
  --out queue-haul/outputs/policy-hardware-width8-packing-contemporaneous-plan
for shard in 0 1; do uv run python queue-haul/policy_hardware_campaign.py prepare-baselines \
  --source-plan queue-haul/outputs/policy-hardware-width8-packing-20260730/plan.json \
  --policies queue_haul greedy isolated_fastest queue_haul_power_blind queue_haul_deadline_blind \
  --condition-shard "$shard" 2 \
  --out queue-haul/outputs/policy-hardware-width8-packing-contemporaneous-shard${shard}-plan; done
```

The 720-scenario contemporaneous packing rerun completed as RAMR jobs 38607705
and 38607709. Both 360-scenario condition shards validate independently; their
checksum-pinned reduced evidence is under
`outputs/policy-hardware-width8-packing-contemporaneous-20260811/`, with pooled
graphs at the top level and shard-1 provenance nested under `shard1/`. The
pooled migration CDF compares Queue-Haul, greedy, isolated-fastest, power-blind,
and deadline-blind on this single contemporaneous cohort. Rebuild the pooled
and per-condition graphs with:

```bash
uv run python queue-haul/policy_hardware_campaign.py plot-reduced \
  --out queue-haul/outputs/policy-hardware-width8-packing-contemporaneous-20260811 \
  --model-profile queue-haul/profiles/gpt_oss_20b_a100_tp1_crossover.json \
  --pooled-with queue-haul/outputs/policy-hardware-width8-packing-contemporaneous-20260811/shard1 \
  --cdf-policies queue_haul isolated_fastest queue_haul_power_blind queue_haul_deadline_blind
```

The completed reduced baselines are checksum-pinned under each 2026-07-30
bundle in `network-baselines-20260811/`; the parent graphs pool both campaigns.
The packing bundle's 30-second full-attainment CDF pools packing and frontier
parent and baseline episodes. It uses an 8-by-4-inch plot, 17-point axis text,
Okabe–Ito colors, distinct line styles, and a collision-free legend inside the
lower-right axes. The deadline is labeled vertically in italics on its line.

The canonical output style is `plot_style.py`: 8-by-5 inches, 15-point titles,
labels, and ticks, 11-point legends and annotations, 3-point lines, and 220 DPI.
Plot-specific layouts may use the shared compact size. New and modified plot
producers must inherit it. Policy identities are:

| Internal name | Display name | Okabe–Ito | Line |
|---|---|---:|---|
| `queue_haul` | Queue-Haul LP | `#0072B2` | solid |
| `greedy` | Queue-Haul Greedy | `#E69F00` | dashed |
| `greedy_lagrangian` | Queue-Haul Lagrangian Greedy | `#F0E442` | dash-dot-dot |
| `isolated_fastest` | True Greedy | `#D55E00` | long dash |
| `kv_only` | KV Migrate Only | `#56B4E9` | dash-dot |
| `replay_only` | Replay Context Only | `#CC79A7` | dotted |
| `queue_haul_power_blind` | Queue-Haul Power Blind | `#009E73` | short dash |
| `queue_haul_deadline_blind` | Queue-Haul Deadline Blind | `#000000` | fine dotted |

Rebuild them with:

```bash
uv run python queue-haul/policy_hardware_campaign.py plot-reduced --out queue-haul/outputs/policy-hardware-width8-frontier-20260730 --model-profile queue-haul/profiles/gpt_oss_20b_a100_tp1_20260730.json --pooled-with queue-haul/outputs/policy-hardware-width8-frontier-20260730/network-baselines-20260811
uv run python queue-haul/policy_hardware_campaign.py plot-reduced --out queue-haul/outputs/policy-hardware-width8-packing-20260730 --model-profile queue-haul/profiles/gpt_oss_20b_a100_tp1_crossover.json --pooled-with queue-haul/outputs/policy-hardware-width8-packing-20260730/network-baselines-20260811 queue-haul/outputs/policy-hardware-width8-frontier-20260730 queue-haul/outputs/policy-hardware-width8-frontier-20260730/network-baselines-20260811
```

capacity_sweep_campaign.py keeps the completed two-point load run as a
pilot and builds the publication load curve at
0,.25,.50,.65,.75,.80,.85,.875,.90,.925,.95,.975. Normalized offered load is
scheduled prefill GPU-seconds plus decode GPU-seconds per second, using the
checksum-pinned independent destination calibration; it is not achieved
throughput or RPS. Each load has ten randomized, paired repeats with the same
pre-generated deterministic trace for Queue-Haul LP, static greedy,
replay-only, and KV-only. All eight source sessions are offered to each policy,
the destination warms for 30 seconds before migration, and arrivals continue
through the full 30-second deadline. A session earns shed credit only when its
route commits and first destination continuation token both arrive by the
deadline. The live figure uses trace-derived load, median shed, bootstrap 95%
confidence bands, and the 147.2 W requested-shed line; LP and Greedy component
panels report replay, KV, and unmet watts. After the dense base run, add
the midpoint of every adjacent pair where any policy's measured median shed
changes by more than 5 W. The empirical LP knee is the last base load whose
median reaches requested shed (within 1e-6 W) and the following load; failure
to bracket it is a hard error. Phase 2a runs repeats 10--19 for both knee loads
and every base load where any policy's 95% interval is wider than 5 W, plus
repeats 0--9 for the midpoints. Phase 2b runs repeats 20--29 only where the
combined base and phase-2a interval remains wider than 5 W. Every selected
load/repeat remains a four-policy common-trace cell with all eight source
sessions. Plans, hardware runs, and the final merged bundle use separate roots;
plans record the exact repeat map, selection reasons, and prior-plan hashes.
All 480 base episodes are checkpointed independently and completed scenario IDs
are skipped on resume.

```
uv run python queue-haul/capacity_sweep_campaign.py load --out queue-haul/outputs/capacity-load-publication-20260807 --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json --run-root /scratch/users/$USER/qh-capacity-load-publication-20260807
uv run python queue-haul/capacity_sweep_campaign.py load --adaptive-stage phase2a --prior-run-root /scratch/users/$USER/qh-capacity-load-publication-20260807 --out queue-haul/outputs/capacity-load-publication-20260807-phase2a --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json --run-root /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2a
uv run python queue-haul/capacity_sweep_campaign.py load --adaptive-stage phase2b --prior-run-root /scratch/users/$USER/qh-capacity-load-publication-20260807 /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2a --out queue-haul/outputs/capacity-load-publication-20260807-phase2b --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json --run-root /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2b
uv run python queue-haul/capacity_sweep_campaign.py load --merge-run-root /scratch/users/$USER/qh-capacity-load-publication-20260807 /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2a /scratch/users/$USER/qh-capacity-load-publication-20260807-phase2b --out queue-haul/outputs/capacity-load-publication-20260807-final
uv run python queue-haul/capacity_sweep_campaign.py goodput --out queue-haul/outputs/capacity-goodput-20260806 --calibration goodput-calibration.json --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json
```

If phase 2b selects no cells, omit its run root from the merge command. Base and
phase plan roots keep `plan.json`, `modeled_capacity.csv`, `live_plan.json`, and
`summary.json`; hardware and final roots add `live_capacity.csv`,
`live_summary.json`, and the live PNG/PDF figures.

`plot_capacity_operating_curves.py` compares the completed 2xA100 load campaign,
balanced full-drain bandwidth blocks 0--1, and width-8 bandwidth scheduler
campaign. It writes scheduler-colored time-to-full-power and deadline-shed
curves plus Queue-Haul replay/KV/not-moved action shares. Full-drain bandwidth
figures split by destination load; width-8 figures split by context profile and
19/30-second campaign. The primary width-8 ECDFs pool the balanced context
profiles at equal frequency within each bandwidth panel (15 episodes per
scheduler); the faceted curves and CSV retain the context-specific results.
The combined action heatmap shows the observed L-shaped 30-second design:
zero-prefill bandwidth cells and positive-prefill 10-Gbit/s load cells, with
unmeasured combinations left gray rather than interpolated. Prefill throughput
uses the load campaign's pinned service-rate calibration.
Coincident scheduler curves are horizontally offset slightly for visibility;
the CSV retains exact operating points. Run
`uv run python queue-haul/plot_capacity_operating_curves.py`; outputs go to
`queue-haul/outputs/capacity-operating-curves-20260808/`.

The forced-full-drain appendix crosses
.85,.875,.8875,.90,.9125,.925,.9375,.95,.9625,.975 offered load with
1, 2.5, 5, and 10 Gbit/s. Five repeat-block shards each contain two repeats at
all four bandwidths (160 episodes), preventing bandwidth from being confounded
with allocation time. Bandwidth block order rotates from a deterministic random
base, and each block randomizes its ten loads and two policies. Arrival traces
are identical across policy and bandwidth for a load/repeat. Every episode
attempts all eight sessions, credits shed at 30 seconds, records the last route
commit and continuation token, and keeps arrivals active until the drain
completes, up to the 180-second timeout. Normalized load still uses only the
30-second measurement window.

The retained 2026-08-08 hardware evidence includes the complete standalone
10-Gbit/s appendix (200 episodes and all 1,600 sessions credited by 30 seconds)
and balanced blocks 0--1 (320 episodes and four repeats per exact
load/bandwidth/policy cell). Across the balanced blocks, all 40 replay-only
episodes at 1 Gbit/s drain in 32.919--33.501 seconds and miss the all-session
deadline, while all 40 KV-only episodes drain in 27.256--28.267 seconds and
meet it. Every episode at 2.5, 5, and 10 Gbit/s meets the deadline. These two
blocks support the qualitative bandwidth boundary but not a ten-repeat
confidence claim; they are retained separately rather than presented as the
five-block merged result.

Run blocks 0 through 4 from the same commit, one per allocation, then merge the
five roots:

```
uv run python queue-haul/capacity_sweep_campaign.py full-drain --repeat-block 0 --out queue-haul/outputs/capacity-full-drain-block0 --live-template queue-haul/outputs/policy-hardware-width8-packing-plan/plan.json --run-root /scratch/users/$USER/qh-capacity-full-drain-block0
uv run python queue-haul/capacity_sweep_campaign.py full-drain --merge-run-root /scratch/users/$USER/qh-capacity-full-drain-block0 /scratch/users/$USER/qh-capacity-full-drain-block1 /scratch/users/$USER/qh-capacity-full-drain-block2 /scratch/users/$USER/qh-capacity-full-drain-block3 /scratch/users/$USER/qh-capacity-full-drain-block4 --out queue-haul/outputs/capacity-full-drain-final
```

Replace `0` in the first command with each block number. Reusing its plan and
run root resumes a block only when the commit and run metadata still match; an
audited code change additionally requires `--resume-from-git-sha OLD_SHA`.
`--bandwidth-mbps 10000` remains available for the standalone ten-repeat knee
appendix. Each shard writes `full_drain_capacity.csv` and its PNG/PDF figure;
the merge accepts exactly five complete blocks, checks profile, calibration,
manifest, context, and trace provenance, and emits the 800-row final result.
The separate live power-drain evidence in
`outputs/power_drain_live_20260714/` includes planned and measured source-power
reductions. The contemporaneous packing traces retain raw `power.csv`,
`result.json`, and plans, but their warm-up-to-migration timing is insufficient
for the same direct measurement.
`outputs/live-power-shed/` retains the 2026-08-06 two-A100 seamless full-shed
run. The Queue-Haul LP arm moved all eight sessions under continuous 4 rps
source and 1 rps destination agentic load with `kv_both` and 33 GB L1 pools;
there is no pre-migration flush, pause, or drain. After 300 s with both sites
serving, the parallel migration took 4.656 s, the traffic switch took 70 us,
and the source drained naturally in 3.002 s before a stable five-minute
destination hold. Mean source power fell from 255.8 W to 85.8 W while mean
destination power rose from 128.2 W to 283.4 W. The three replay moves
recomputed 10,662 tokens; the five KV moves computed 713 tokens while reusing
2.72 GB of KV. The plot shows the 300–400 s handoff region as fixed 500 ms mean
power in a compact, outlined view. The retained paired 10 Hz samples have 95.6%
coverage and the bundle includes phase-level engine queue depth and
checksum-pinned raw data.
`outputs/live-power-handoff-east-germany-20260807/` retains the three-A100
Sweden-to-East-US-2/West-Germany campaign with 80% source and 50% load at each
destination. Queue Haul admitted all eight sessions in 29.669 s and KV-only in
25.159 s; replay-only admitted six before the fixed 30-second deadline. The
bundle includes raw `power.csv`, load and transfer telemetry, and separate 500
ms mean regional-power plots cropped from session-state preparation through GPU
sleep. Each trace reports percent of the 300 W per-GPU TDP and marks Migration,
Switch, Barrier, and Sleep. The bundle also retains the exact plan and composed
non-formal calibration used.
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
preserving the existing catch-up, power, and capacity evidence. `--policies isolated_fastest` enables the per-session-fastest policy without
changing the default policy set. The pinned
`outputs/policy-hardware-width8-packing-plan/` runs three paired width-8 episodes
for Tiny, Small, Medium, Mixed, and Large packs at 1/2.5/5/10 Gbit/s and 19/30-s
requirements: 600 scenarios in total. Job 36822272 completed all 600 scenarios
without failures in 8:08:22. Its checksum-pinned reduced bundle is under
`outputs/policy-hardware-width8-packing-20260730/`; compressed results, GPU
samples, proxy byte counters, and RESP transfer records retain the raw evidence
without runtime debug logs. The matched 120-episode, 240-scenario baseline plan is pinned under
`outputs/policy-hardware-width8-isolated-fastest-plan/`. Its bandwidth-grouped
order has four bandwidth blocks instead of 93 bandwidth runs; the 30-scenario
hygiene rotation yields eight model-stack starts. Its launcher reduces all
episodes, hard-validates three complete repetitions per condition and clean
plan/profile provenance, writes checksums, and rebuilds the comparison from the
common packing cohort only (120 observations per method). The comparison uses
±5 percentage points relative to Queue-Haul LP for better/similar/worse and
shows every sample count; frontier and partial-network rows are excluded. The
completed allocation 37874352 produced 240/240 scenarios with no failures and
120 validated attainment rows across 40 conditions with three repetitions each.
Its reduced, checksum-pinned bundle is under
`outputs/policy-hardware-width8-isolated-fastest-20260806/`, including the
common-cohort chart and compressed result, power, proxy-byte, and RESP-transfer
evidence.
Its bandwidth-faceted destination-TTFT CDF pools
all five workloads, both deadlines, and three episodes within each bandwidth;
the companion pooled CDF combines all four bandwidths. Regenerate that CDF
with the earlier trace-sampled 5/10-Gbit/s frontier rows included at raw-sample
weight using `plot-reduced --out <packing-results> --pooled-with
<frontier-results>`; this pooling also applies to the maximum-session,
maximum-session-per-watt, and existing per-watt CDFs.
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
