# What we can defend, what we can't, and the experiments that close the gap

Plain-language companion to `PROPOSED_DESTINATION_ARCH.md`. Every number was read
from the code or recomputed from the checked-in profile and manifest on
2026-07-25. Where a claim is someone else's measurement, it is cited.

---

## 0. Correction to the previous version of this document

The earlier draft claimed the source and destination use the **same** load
coordinate, and that the destination bound is therefore 5.481× too strict. **That
is wrong.** They are two different coordinates:

- The source packs with the **scalars** `F = 1448.32`, `G = 1260.38`
  (`destination_bench.py:120`, `power_model.py:15`).
- The destination scores work with the **context-conditioned curves**
  (`destination.py:146`): prefill 4,655–5,872 tok/s, decode 1,180 tok/s at 4K
  falling to 191 tok/s at 24K.

Measured over the actual seed-0 sessions, destination work ÷ source load is:

| Workload | aggregate | per-session p10 / p50 / p90 | spread |
|---|---:|---|---:|
| Interactive coding | 0.856 | 0.59 / 0.77 / 1.30 | 3.9× |
| Coding | 0.888 | 0.28 / 0.41 / 2.57 | 65× |
| Agentic | 0.701 | 0.28 / 0.43 / 1.44 | 32× |

So the destination coordinate is on average slightly **cheaper**, and varies by up
to 65× per session. The "393.5% at 0.096953 vs 71.8% at 0.531358" table in the
old draft was that column scaled by 5.481 and is invalid. Delete it.

**The real objection to `SERVICE_BOUND = 0.096953` is simpler and survives:** it is
a probe that passed, and `FINDINGS.md` states there is **no failure anywhere in
the dataset**. It is a lower bound on capacity being used as an upper bound. There
is even a passing observation at 0.114063 (`FINDINGS.md:86`) that we ignore.

---

## 1. The next modeling result

The primary result is **required destination residual resources versus source
watts shed**. It requires a pinned destination class, not a concrete destination
inventory. The candidate set contains replay and KV transfer together; the
solver may select both methods across sessions, with at most one action per
session.

Destination power caps and pricing are deliberately out of scope. Measured
action power may remain provenance, but it does not choose the method or define
feasibility. A later operator can compare a concrete destination's residual
service, KV, migration, and route budgets against the reported frontier.

---

## 2. The honest ledger

### Measured, defensible

| Quantity | Value | Error |
|---|---|---|
| Source power vs load | 12-point curve, 67.12 W idle → 248.89 W at ℓ=0.531 | 5% |
| Prefill / decode work coordinates | `F` = 1448.32, `G` = 1260.38 tok/s | 25% |
| KV capacity | 963,152 tokens/replica | 0% (readback) |
| KV bytes/token | 49,152 B = 48 KiB | exact — see below |
| Replay time | `log_B/rate + 0.5867·(tokens/replay_tps + 1.340·(1+growth)) + switch` | **9.6%** held-out at 24K, never underpredicts |
| KV time | `sealed_B/rate + 1.1338 + catch-up` | **7.8%** held-out at 24K, never underpredicts |
| Migration action power | replay 189.84 W dst; KV 11.84 W dst | measured, outside this formulation |
| Replay foreground cost | +1.084 s TTFT, +3.45 ms/tok TPOT | n=1, n=5 |
| KV foreground cost | +4.7 ms TTFT, +0.42 ms/tok TPOT | n=1, n=5 |

The 48 KiB/token checks out exactly from the architecture:
2 × 24 layers × 8 KV heads × 64 head-dim × 2 bytes = 49,152 B. And
963,152 × 48 KiB = 44.1 GiB, leaving 11.8 GiB of the 55.9 GiB budget for weights
— which matches gpt-oss-20b's MXFP4 expert weights. Back-solving predicts the
measured KV capacity to **0.0065%**. The `precision: bf16` field means the KV and
activation dtype, not the MoE weight dtype; that label should be split.

The migration timing models are the strongest thing we have: two parameters,
physically structured, held out to a different context **and** bandwidth,
conservative by construction. They should anchor the paper.

### Assumed, and load-bearing

| Assumption | Where | Consequence if wrong |
|---|---|---|
| Source streams | sensitivity | sweep 1/2/4/8 and report which resources change |
| Service and migration are **independent** resources | `pool_planner.py:218-226` | overlap is allowed when both budgets fit |
| Migration concurrency 1 per sink replica | `pool_planner.py` | untested |
| One request rate, 1/180 req/s/session | `destination_bench.py:46` | — |
| WAN is one logical route | requirement input | 5 Gbps central; 1/5/10 Gbps sensitivity; one fixed P50 RTT per action |
| `normal = emergency = stable = 0.096953` | `destination_bench.py:187` | the mode machinery is inert |

### Broken or inert, found by audit

- **`predict` *is* the simulator** (`simulate.py:1010-1026`), differing only by a
  logging flag. There is no analytic model to cross-check it against, so
  "our model matches our simulator" has nothing behind it.
- **`outputs/simulator_validation.csv` is a 2-session, 100-byte hand-arithmetic
  unit test rendered as a PDF.** The file itself says "not hardware measurements."
- **`outputs/simulator_evaluation.csv` and 4 of 5 scaling runs use deleted
  solvers** (`node_aware`, `node_drain`, `load_only`) and cannot be regenerated.
- **Largest real-GPU run migrates 4 concurrent sessions**, 105 scenarios, no
  power deadline, no sleep or shutdown, `final_state` hardcoded to `awake`.
- **The one sim-vs-hardware comparison** (`model_check`) is an analytic timing
  equation, concurrency-1 only, 72 rows, **median +78% error against TTFT**
  (+7.7% against copy wall time), with no error metric computed and no test.
- **The 10,000-session bench runs zero inference requests.** `sample_sessions`
  never sets `SimSession.requests` and `_expected_scenario` strips them anyway.
  We measured foreground interference and then evaluated with no foreground.
- **`workload_prefill_fraction_range = (0,1)`**, so the affinity gate never fires.
  **`LoadedCoefficients`** is required and dead. **The stable-envelope check is a
  tautology** — it re-tests a bound the packer already enforced.
- **Fixed:** the pool KV path now applies
  `max(route, bytes/ingest_rate) + residual + catch_up` (`ef435092`).
- **Idle floor disagreement:** the curve says 67.12 W at ℓ=0; `README.md:273`
  reports a measured 84.9 W after sleep. `power_limit_w` is defined as every GPU
  at the floor, so a 26% floor error propagates into every reported shortfall.
  Resolve which configuration each number came from.
- **n=1.** One seed, one profile case (`central` is hard-required), despite
  declared input errors of 25% (service), 30% (replay), 48% (KV), 100%
  (transitions). No error bars anywhere.

---

## 3. Where the result actually comes from

### Migration has a hard ceiling, set by GPUs sitting awake

| Workload | Source power | Idle floor | Max sheddable by migration |
|---|---:|---:|---:|
| Interactive coding | 15,433 W | 4,833 W | **68.7%** |
| Coding | 19,877 W | 12,350 W | **37.9%** |
| Agentic | 21,394 W | 11,813 W | **44.8%** |

Measured GPU sleep saves **0.0158 W**, so sleep is useless. Only powering nodes
off recovers the floor, `shutdown_s` is `null`, and all three planners hard-require
`final_state == "awake"`. Independent corroboration: DynamoLLM measures 8 idle
H100s at **550 W** — our A100 node floor is 8 × 67.12 = 537 W.

### Node power-off hinges on one unmeasured constant

Exact per-replica drain time against the 115 s window:

| Workload | Drainable at 1 stream | at 2 streams |
|---|---:|---:|
| Interactive coding | 1 of 65 | **65 of 65** |
| Coding | 23 of 179 | **179 of 179** |
| Agentic | 14 of 172 | **172 of 172** |

The median replica needs 1.17–1.37 streams. This is a razor edge, and
`max_source_streams = 1` has **no measurement behind it**. Our own README already
contradicts it: `mp-campaign-run-10` measured **591 MB/s at concurrency 2 and
1.206 GB/s at concurrency 4 against a 111 MB/s serialized ceiling**, and
`bounded-hardware-campaign-run` completed 105 scenarios at concurrency 1/2/4.

Confirmed at the other end of the scale: at a 6-hour window the legacy path drains
**1,328–1,410 of 2,975 nodes**; at 115 s the destination path drains **0–1 of 23**.

### The fleet is memory-parked, not compute-loaded

| Workload | Source service used | Source KV used |
|---|---:|---:|
| Interactive coding | 83.9% | 94.5% |
| Coding | **14.4%** | **99.4%** |
| Agentic | **19.3%** | **99.5%** |

Coding needs ~26 replicas for its compute and 178 for its KV. Each GPU sits near
113 W, of which 67 W is idle floor. **What we are migrating is memory residency,
not load** — which is exactly why powering nodes off, not moving compute, is where
the watts are. It also means the obvious local baseline (consolidate + offload KV
to DRAM/SSD, à la LMCache/Mooncake) must be measured before we claim the WAN hop
is necessary.

### The solver contributes almost nothing in 2 of 3 workloads

Coding: LP 7,215.45 W vs greedy 7,204.36 W (**0.15%**). Agentic: 9,004.43 vs
8,980.01 (**0.27%**). The LP reaches 95.9% of the trivial upper bound. The only
place it wins is interactive coding (2,753 vs 1,964 W) — against our own
single-shot-pricing greedy, which prices once and never reprices. At 1M sessions
the LP is actively worse: **1.89× overshoot, 949,031 moves where 524,241 suffice.**

---

## 4. The claim to make

Stop claiming "these N sessions will be served acceptably at the destination." We
have no fleet telemetry and don't want per-request TTFT modelling. Claim:

> For each source-power target, here is the destination service, live KV,
> migration occupancy, source-stream occupancy, WAN bytes, and makespan required.
> A concrete destination may later compare its residual vector against this
> frontier.

The headline figure becomes **required destination residual vs. watts shed**, with
the reader's own bound as a horizontal line. That plot is invariant to the
0.096953 argument. It is Skyplane's posture with profiled bandwidth grids and
Llumnix's with reported freeness.

The frontier already computes cleanly:

| Cut demanded | Coding shed | preserved | Interactive shed | preserved |
|---:|---:|---:|---:|---:|
| 30% | 2,774 W ✓ | 575 | 2,753 W ✗ | 3,792 |
| 70% | 6,263 W ✓ | 3,758 | 2,753 W ✗ | 3,792 |
| 90% | 7,215 W ✓ | **7,746** | 2,753 W ✗ | 3,792 |
| 100% | 7,215 W ✗ | 7,746 | 2,753 W ✗ | 3,792 |

Coding meets every cut to 90% preserving 77% of sessions. Interactive **saturates
at a 20% cut** and never moves again — the sink-service row pins at 1.0000.

---

## 5. Experiment plan

### Phase A — fix what makes the current numbers meaningless (days, no GPU)

**A1. Add the requirement-only frontier.** For source targets at
10/25/50/75/90/100% of maximum shed, jointly select replay and KV candidates and
report the raw residual-resource vector without constructing a destination.

**A2. Use one logical WAN route.** Charge effective bandwidth plus exactly one
`route_rtt_s` P50 RTT per action. Use 5 Gbps centrally, bandwidth sensitivity
1/5/10 Gbps, and RTT classes 10/60/90/150/240 ms. Do not model TCP or require a
multi-edge topology.

**A3. Sweep `max_source_streams ∈ {1,2,4,8}`** and report landed sessions, watts,
resource totals, method mix, and makespan. Separate fixed-plan invariants from
changes caused by reoptimization. Pack indivisible durations into per-source
stream bins; keep the single WAN byte row as a fluid relaxation and label its
makespan a lower bound until event execution validates it.

**A4. Allow `final_state ∈ {sleep, off}`** in the destination planner and add a
node-emptying term to the objective. Report nodes drained as a first-class metric
— the column already exists in the 1M artifact.

**A5. Done: restore the ingest floor** in
`pool_planner._destination_duration` (`ef435092`).

**A6. Report the LP triple** (fractional bound / rounded / packed) plus an exact
integer optimum on 100–500-session instances, with the watt gap.

**A7. Ten seeds with bands; `faster` and `slower` profile cases.** Everything
quoted today is n=1 with declared input errors of 25–100%.

**A8. Put real requests in the bench** so the deadline and independent service
budget checks exercise foreground work.

**A9. Replace `bottleneck` with a binding set** and drive the pressure search
with watts, not `all_sessions_landed`.

### Phase B — baselines (days, no GPU). Without these there is no paper.

Plot watts shed against a **quality** axis (sessions lost, TTFT/TPOT damage):

1. Do nothing.
2. Drop/drain sessions until under cap.
3. **Local consolidation + node power-off with KV offload** — the strongest
   competitor, since our fleet is at 14.4% compute.
4. GPU power capping / DVFS.
5. Migration (ours).

Migration must win on quality at equal watts. On the power-capping baseline we
already have the answer and it is favourable: [Ma et al.](https://arxiv.org/abs/2605.11999)
measured decode drawing **137–300 W against a 700 W TDP, with the lowest 280 W cap
never engaging** — the driver holds ~1830 MHz because memory-bound decode saturates
HBM, not compute. Splitwise Fig. 9 independently shows capping 700→350 W costs
decode almost nothing. **The standard lever is inert for the phase that dominates
serving.** Our own curve agrees: 67.12 → 248.89 W on a 400 W part.

### Phase C — the measurements (~2 GPU-days)

**C1. Migration concurrency 2 and 4 per source replica.** The harness, campaign,
and 105 completed scenarios already exist. This one constant decides whether nodes
can be emptied at all. It is the highest-value GPU-hour in the plan.

**C2. The service-frontier rerun** `FINDINGS.md` specifies — safe forced tokens,
APC reset, hard-fail on missing work — targeting a **Mooncake-style relative SLO**
(p90 TTFT = k× isolated; Mooncake uses TTFT_P90 = 10×, TBT_P90 = 5×). Start at
0.096953 and **expand until failure is bracketed**. A bound with no failure in the
dataset is not a bound.

**C3. Migration timing at 4–8K contexts**, where interactive coding actually lives
and where **100% of its moves are currently extrapolated** (`in_domain_fraction`
is 0.000 / 0.339 / 0.347 across the three workloads).

### Phase D — the figure that carries the paper (~2 GPU-days, 16 GPUs)

Two 8-GPU A100 nodes per site, our stack, **live open-loop traffic on both sites**,
a shaped WAN cut, a synthetic curtailment signal. Measure with the 250 ms sampler
we already have:

- source **and destination** wall power through the whole event;
- source crossing under the cap by the deadline in the 5 s trailing window;
- destination p90 TTFT/TPOT for **pre-existing** destination sessions during the
  migration burst — the number our model asserts and has never measured;
- the same baselines from Phase B on the same hardware.

Then show the simulator reproduces that power trace within its stated error at 16
GPUs, before extrapolating to 358. **That single figure is the paper.**

### Phase E — generality (~10 GPU-hours)

The reviewer attack is not "one GPU" — `DestinationType` is already parameterized
per type with its own curves, KV capacity, `synthetic` flag, and `evidence_status`.
The dangerous attack is internal: *changing destination hardware is exactly the
operation that decouples prefill and decode, which is the regime where a single
`f/F + g/G` facet fails.*

The cheapest defence is two points that move the prefill:decode ratio in **opposite**
directions:

- **H100, TP=1** (~4–6 GPU-h). Pre-register: KV capacity ratio 1.01× from the
  closed form; prefill uplift near the **1.64× bandwidth** ratio rather than the
  1.85–1.95× Splitwise saw on dense Llama-70B, because our MoE model runs at only
  **10.7–17.6% prefill MFU**; decode 1.4–1.6×. Predicting a *lower*-than-published
  uplift from our own MFU is a stronger result than the measurement.
- **A100 TP=2** (~4 GPU-h, hardware we own). TP moves the ratio the other way and
  changes KV capacity ~2×, exercising the closed form on a second axis.

Two free moves: (i) **print the existing cross-hardware table** — the power law
holds at R² 0.91–0.99 across **25 node types** (7 models × {A100, H100} × TP 1–8)
with the decode:prefill energy ratio stable at 5–25×; the ℓ coordinate already
demonstrably transfers for *power*, and only the service facet is single-type.
(ii) **Reframe the facet as a conservative inner approximation**: `f/F + g/G ≤ 1`
is the simplex inscribed in the box `{f ≤ F} ∩ {g ≤ G}`, so it under-admits, never
over-admits. State the geometry and "why one facet" stops being an attack.

Also state the per-type profiling protocol and its cost — **4–6 GPU-hours** (KV by
closed form, 5-point prefill sweep, 5-point decode sweep, 16-point mixed grid).
Mélange claims "<1 hour" without context curves; Helix, Splitwise, Vidur and
AIBrix publish no number at all. **Quantifying this is a contribution.**

---

## 4b. We model the landing, not the stay — and the stay is where it breaks

A real curtailment event is **about two hours long** (Duke/Norris: the average
curtailment event lasts ~2 hours). Our residency horizon is
`max(180, migration_s + 5)` = **180 seconds**, and `destination_bench.scenario()`
zeroes `expected_growth_tokens_per_s` at placement, so sessions are frozen
snapshots that never grow after they land.

Let them keep growing at their traced rate:

| Residency | Interactive coding | Coding | Agentic |
|---|---:|---:|---:|
| 180 s (**current**) | 94.5% | 99.4% | 99.5% |
| 15 min | 139.6% | 107.6% | 110.4% |
| 1 hour | 308.7% | 138.2% | 151.5% |
| 2 hours | **534.2%** | **179.0%** | **206.3%** |

**The destination runs out of KV memory 217–289 seconds after the migration
completes.** The evacuation succeeds for about four minutes and then the sink is
full.

This is not a tuning problem, it is a missing dimension. All five resource rows we
model are either transition-time rows (streams, routes, migration slots) or
instantaneous-occupancy rows (service, KV at t=0). **None of them is a sustained
constraint.** A grid event you must ride for two hours cannot be served by a
destination that fills in four minutes.

Three ways out, and the paper must pick one and say so: model session churn
(arrivals and departures at the destination), model progressive eviction or
tier-offload during the stay, or model return migration when the event ends. The
alternative is to state plainly that we solve the *landing* problem and that
sustained occupancy is out of scope — which is defensible, but only if said.

## 4c. Grid grounding: what the event actually looks like

**The failure mode we prevent is already happening, at our exact scale.** NERC's
incident review of **July 10, 2024** (*Incident Review: Considering Simultaneous
Voltage-Sensitive Load Reductions*, published 2025-01-08): a lightning-arrestor
failure on a 230 kV line, with auto-reclosing configured for three attempts at
each end, produced **six faults in 82 seconds**, each 42–66 ms, dropping voltage
to 0.25–0.40 p.u. **About 1,500 MW of load disconnected — exclusively data-centre
load.** No utility equipment tripped it; **the customer's own protection
transferred the load to backup power.** About **1,260 MW dropped at the third
depression and did not return for hours.** NERC interviewed the operators and
found the rule: *"three voltage disturbances within one minute will result in data
centers... transferring their load off the grid and staying off until they
manually transfer back."*

That is our baseline, and it is brutal: **an instantaneous, uncontrolled,
hours-long disconnect of every session.** Graceful migration is not competing with
"do nothing" — it is competing with a hard drop that has already happened twice at
gigawatt scale (a second ~1,500 MW event occurred in ERCOT).

**Regulators are actively writing the requirement we would satisfy.** NERC's
Level 3 Alert (2026-05-04) proposes registering a **"Computational Load Entity"**
— loads **≥20 MW at ≥60 kV with >1 MW of IT load** — and requires each to report
its **"Expected Ramp Rate (MW/min), down-ramp and up-ramp."** FERC (RD26-7-000,
2026-07-16) directed NERC to file mandatory computational-load Reliability
Standards by **2026-12-31**. NERC's May 2026 Reliability Guideline names the
failure mode **CILR (customer-initiated load reduction)**, notes that both existing
NERC load-loss definitions **explicitly exclude it**, and recommends *"establishing
large load curtailment as a System Operating Limit."*

**A ramp rate in MW/min is exactly what our simulator computes.** That is the
output format to report.

**The market has already priced this.** ERCOT's demand-response contribution to
resource adequacy jumped from **2.7 GW (2024 LTRA) to 13.3 GW for Summer 2026,
rising to 53.1 GW by 2030**, attributed directly to new large-load curtailability
under Texas SB6. Duke/Norris model **76 GW of new load integrable at 0.25% annual
curtailment, 98 GW at 0.5%, 126 GW at 1.0%.** LBNL puts US data centres at 176 TWh
in 2023 (4.4% of national electricity), reaching 6.7–12.0% by 2028.

**And NERC names a modelling gap we are positioned to fill:** of 33,282 MW of data
centre load in submitted dynamic model files, **25,504 MW has no dynamic model
representation in stability software at all.**

Two things to be careful about. NERC's white paper says stability events are
*"low-probability, high-impact"* — **not** "high likelihood." And the July 2024
report is a **NERC-only** review that says "Eastern Interconnection," never
"Northern Virginia"; that attribution is trade press.

**Still missing:** the ERCOT/PJM ancillary-service response-time table (RRS, ECRS,
Synchronized Reserve, Regulation D). Two agents died on session limits before
retrieving it. We still cannot say which product a 120-second response maps onto.
What we *can* say is that response is measured in **seconds** (NERC: *"these events
can transpire in a matter of seconds"*; POLCA: a **10 s** UPS deadline) while the
event lasts **hours** — which is precisely the mismatch §4b exposes.

## 4d. The constants, and why the formulation outlives them

An NSDI reviewer will ask two separate questions and we should answer them
separately: *what did you assume, and where did it come from?* and *what happens
when the hardware changes?*

### Every constant, with its provenance

| Constant | Value | Where it came from |
|---|---|---|
| KV bytes per token | **49,152 B (48 KiB)** | closed form 2 × 24 layers × 8 KV heads × 64 head-dim × 2 bytes; predicts the measured capacity to **0.0065%** |
| KV capacity per replica | **963,152 tokens** (44.1 GiB) | vLLM 0.22.0 readback at 0.75 GPU-memory utilisation |
| KV ingest rate | **620.8 MB/s = 12,630 tok/s** | measured, CPU-mediated LMCache path |
| Prefill throughput | **4,655–7,634 tok/s** over 256–31,562 tokens | measured; **10.7–17.6% MFU** against a 43,333 tok/s roofline (3.6B active params × 2 FLOP on 312 TFLOPS) |
| Decode throughput | **3,774 → 77.9 tok/s** across 256 → 31,562 tokens (48.5× collapse) | measured; `1/G = a + b·T` holds to 4% below 16K, then breaks (19% over at 24.5K, 138% over at 31.5K) |
| Tail replay rate | 919.4 tok/s | measured |
| Replay time | `log_B/rate + 0.5867·(tokens/replay_tps + 1.340·(1+growth)) + switch` | fit on six 16K rows, **9.6%** held-out at 24K, never underpredicts |
| KV transfer time | `sealed_B/rate + 1.1338 + catch-up` | fit on six 16K rows, **7.8%** held-out at 24K, never underpredicts |
| Migration action power | replay **189.84 W**, KV **11.84 W** at the destination | measured |
| Power vs load | 67.12 W idle → 248.89 W at ℓ=0.531 | measured, 5% error |
| Request rate | 1/180 req/s/session = **0.31–0.86 rps/replica** | assumed; Llumnix's evaluation runs 0.42–1.9 req/s per instance, so we sit in the published range |
| WAN route | **5 Gbps central; 1/5/10 Gbps; P50 RTT 10/60/90/150/240 ms** | explicit sensitivity; one logical route and one RTT per action |
| Source migration streams | **1/2/4/8** | sensitivity; our own `mp-campaign-run-10` measured 591 MB/s at concurrency 2 and 1.206 GB/s at concurrency 4 against a 111 MB/s serialized ceiling |
| Service bound | 0.096953 | a probe that passed; **no failure exists in the dataset** |

The bottom four are the honest weak points, and three of them we can close with
measurements we already know how to run.

### The constants move at very different speeds — and that is the result

| Constant | Trajectory | Rate |
|---|---|---|
| Prefill throughput | compute-scaling **and** 5.7× of software headroom to the roofline; A100→H100 measured 1.85–1.95× | **fast** |
| KV capacity | 80 → 141 → 192 GB, times FP8 (2×) or INT4 (4×), times MLA-style compression (~3.6×) | **fast** (10–25× plausible) |
| KV ingest | 620.8 MB/s is a CPU-mediated path; Splitwise moves KV over RDMA in 5–8 ms | **fast** (~100× available today) |
| Decode throughput | HBM-bandwidth-bound: 2,039 → 3,352 → 4,800 GB/s, ~1.64× per generation | **slow** |
| WAN egress | 5 Gbps per VM on AWS — a *policy* cap, not physics; flat for years | **slowest** |
| Migration concurrency | unmeasured | unknown |

This gives the paper a claim that gets *stronger* with time rather than expiring:

> Compute and memory improve by roughly 10× per generation while wide-area
> bandwidth stays flat. So the binding constraint on cross-site evacuation
> migrates away from the destination's steady-state capacity and toward the
> transition — source streams, migration slots, and the network.

Our reference run already shows two of three workloads in that regime (source
streams saturated in 178 of 179 replicas, migration slots at 0.987, WAN at 0.993),
and interactive coding — the one still limited by destination service — is exactly
the case that a faster GPU moves into it. **The structural finding is not "a 2026
A100 can absorb N sessions." It is "this problem is transition-limited, and
becomes more so."**

### What would actually break the formulation

The model is `Ax ≤ 1, Ux ≤ 1, 0 ≤ x ≤ 1`. `A` is session incidence and depends on
no measurement at all. Every entry of `U` is (a measured consumption) ÷ (a measured
capacity). Changing any constant changes **numbers inside `U`** and nothing else —
not the number of rows, not the columns, not the sparsity pattern, not the
objective, not the solver, not the packer, not the validator.

Exactly three things would break it, and the architecture already names all three:

1. **A new kind of consumable appears** — staging memory, facility power, a
   licence — and needs its own row. The fix is to add a row; no other change.
2. **A resource stops being additive.** Cross-session prefix sharing is the live
   example: KV would become a block-union rather than a sum, which concrete
   packing cannot express today.
3. **A resource stops being linear in the candidate** — interference that grows
   faster than the sum of parts — which needs another facet, and the rule is to
   add one only when held-out mixed-load data rejects the single facet.

Being able to state the complete list of what would break it is a much stronger
claim than "the model is general."

### End to end, only one stage is hardware-specific

| Stage | Depends on the constants? |
|---|---|
| 1. Measure the type (4–6 GPU-hours: KV by closed form, 5-point prefill sweep, 5-point decode sweep, 16-point mixed grid) | **yes — this is the only one** |
| 2. Eligibility predicates (model, tokenizer, KV ABI, warmness) | no — boolean |
| 3. Build the five resource rows | no — same rows, new numbers |
| 4. Solve | no |
| 5. Pack to concrete replicas | no |
| 6. Validate by execution simulation | no |

Adding a destination site means adding pools, per-replica baselines, candidate
columns, and a logical-route input. Adding a *hardware type* means running stage 1 once.
Neither introduces a new abstraction. That is the robustness claim, and §4e tests
it rather than asserting it.

## 4e. The invariance test — run, not asserted

Scale the measured constants by 10× and 100× and re-solve. Same code, same rows,
same solver; only numbers inside `U` change. 10,000 sessions, seed 0, reference
pressure. Binding set = every row family at or above 0.95.

**Interactive coding** (65 replicas)

| Streams | KV × | Prefill × | Landed | Watts | Binding |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 3,808 | 2,016 | service 1.00, source 1.00 |
| 1 | **10** | 1 | 3,808 | 2,016 | service 1.00, source 1.00 |
| 1 | **100** | 1 | 3,808 | 2,016 | service 1.00, source 1.00 |
| 1 | 1 | **100** | 4,246 | 2,354 | service 1.00, source 1.00 |
| **2** | 1 | 1 | 4,022 | 2,227 | service 1.00, source 0.97 |
| **2** | 100 | **100** | 4,874 | 2,677 | service 1.00, source 0.97 |

**Coding** (179 replicas)

| Streams | KV × | Prefill × | Landed | Watts | Binding |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 7,746 | 6,100 | source 1.00, route 0.99, migration 0.99 |
| 1 | **100** | **100** | 7,746 | 6,100 | source 1.00, route 0.99, migration 0.99 |
| **2** | 1 | 1 | 8,295 | 6,192 | **migration 1.00, route 0.99** |
| **2** | **100** | **100** | 8,295 | 6,192 | **migration 1.00, route 0.99** |

Three results, and the second is the one to lead with.

**1. The formulation is invariant.** Every cell above ran through unmodified code.
No new row, no new abstraction, no solver change. Only entries in `U` moved.

**2. Multiplying KV capacity by 100 changes nothing — not one session, not one
watt, in any workload.** This is the direct answer to "what if we 10× the packing
capability?" Nothing happens, because KV was never the binding row (0.34 and 0.77
at reference). The formulation does not merely survive the change; it explains why
the change is irrelevant.

**3. Multiplying prefill throughput by 100 buys 11.5% on the service-bound
workload and 0% on the transition-bound ones.** The reason is exact: the service
coordinate is `f/F + g/G`, and interactive coding's demand splits 3.60 prefill
against 21.20 decode. Prefill is 15% of the load, so even infinite prefill
throughput removes at most 15%, and the row stays pinned at 1.00. **Decode, not
prefill, is what holds the destination.**

That matters because decode throughput is the **slowest-improving** constant in
the table (HBM-bandwidth-bound, ~1.64× per GPU generation), while KV capacity and
prefill throughput are the fastest. The sweep confirms the trajectory argument
empirically rather than by assertion: **the constants that are improving quickly
are the ones that do not bind, and the ones that bind are improving slowly or are
policy-capped.**

**4. The one knob that moves coding is migration concurrency, and it relocates the
bottleneck.** Going from 1 to 2 source streams lands 7.1% more sessions and pushes
the source-stream row off the binding set entirely — the constraint moves to the
**destination migration slot and the WAN**. That is the predicted migration of the
bottleneck from source to transition, observed.

Two honest caveats. This scales only the *sink* constants, holding source packing
fixed, which isolates the destination question by design. And 100× prefill is
unphysical — it exceeds the roofline by 15× — so it is a limit probe establishing
insensitivity along that ray, not a forecast. Per Floyd & Paxson, a flat sweep
does not prove global insensitivity; it proves it along the axis swept, with the
other factors held at the values stated.

**What to print in the paper:** this table, plus the trajectory table from §4d.
Together they say the thing that survives the hardware treadmill — *here is which
constraint binds, here is where it moves as each constant improves, and here is
the code that did not change while we found out.*

## 5b. The simulator-credibility bar, and the thing we are underselling

### Nobody else has a validated dynamic power model for LLM serving

This is the most important thing the literature review turned up. **No published
LLM-serving simulator models dynamic GPU power.**

- **Splitwise** measured power but scopes it out of the simulator: *"We only
  consider the provisioned power, and not the dynamic power utilization, in our
  study."*
- **Vidur** has no power model: *"We plan to extend these to also capture the
  cluster's energy consumption."*
- The one paper that bolts a power model onto Vidur ([arXiv 2507.11417](https://arxiv.org/abs/2507.11417))
  calibrates from **datasheet TDPs** and states in its own limitations paragraph
  that it *"has not been validated against profiling tools such as NVML."*

Our `power-concave-curve.md` — a Michaelis–Menten law at **R² 0.91–0.99 across 25
node types** (7 models × {A100, H100} × TP 1–8, ~80k 5-second windows), with a
two-price decomposition showing decode tokens cost **5–25× more energy** than
prefill tokens, stable across every configuration — is stronger than anything
published in this space. **We have been treating it as a methods detail. It is a
headline contribution.**

Two things to do with it: (i) print the 25-node-type table, since it is also the
answer to "you only measured one GPU"; (ii) **refit on a held-out split and report
MAPE**, which is the one thing it currently lacks — see C.3 below.

### The accepted fidelity bar

| Layer | Accepted error | Evidence |
|---|---|---|
| Fitted component model, held out | **MAPE < 3%** | Splitwise, 80:20 split |
| End-to-end request metric | **2–9%, ~5% conventional** | AlpaServe <2%, DistServe <2%, Lucid <4.6%, Helix <5%, Sia <5%, Synergy <5%, Gavel <8%, **Vidur <9%** |
| Heterogeneous serving | 9–15% | LLMServingSim 14.7% |
| GPU power model | 3–13% | EnergAIzer 6.7–12.7% |

Physical validation clusters in accepted papers are **small**: Vidur used a single
4-GPU node; DistServe 32; Sia 44; Gavel 48; Pollux 64; Splitwise 4 VMs. Under 10%
error on an end-to-end request metric, on whatever you actually own, is a passing
grade.

### Our extrapolation ratio is the problem, and it is nameable

Published ratios run from 1× (Pollux) to 65× (Lucid). Ours is **2 GPUs validated →
358 GPUs simulated ≈ 179×**, which exceeds everything in the literature. And
extrapolation error does not stay flat: SimAI (NSDI'25) measured ASTRA-sim going
from **45.9% error at 128 GPUs to 530.2% at 512**.

Phase D at 16 GPUs brings us to **22×** — comparable to Synergy (16×) and well
under Lucid. **State the ratio ourselves rather than letting a reviewer compute it.**

### Copy Vidur's five experiments, and Helix's shape

**Helix is our structural template, not Vidur:** validate on the one cluster you
can actually build (they used 24 nodes, 3 GPU types, 10 Gb/s), report <5% on
throughput and both latencies **plotted in the same figure as the real results**,
then simulate exactly the configurations you cannot build — geo-distributed and
high-heterogeneity — with inter-cluster bandwidth grounded in a *separate* iperf3
measurement. That is precisely our shape.

Vidur's fidelity section is the format to copy:

1. **Static fidelity** — all requests at t=0, report **median and P95** as paired
   real-vs-simulated bars with the signed % error annotated on each.
2. **Dynamic fidelity** — Poisson arrivals at **85% of measured capacity**, with
   the operating point justified (lower is idle, higher is a queueing tipping
   point). Same paired-bar format.
3. **Fidelity vs operating point** — sweep 0.75×–0.95× of capacity and show where
   it degrades. Vidur honestly reports 12.65% at 95%.
4. **Cost of the alternative** — GPU-hours the measurement would have taken vs
   simulated wall-clock. Vidur: 42K GPU-hours / $218K vs $125.
5. **What-if analysis only after fidelity is established.**

### Four things that will get us rejected, and the fixes

- **Heiser C.3 — calibrating and validating on the same data.** This is the most
  likely kill-shot for a paper selling "grounded in our own measurements." Fix:
  hold out a split for the power law and every timing fit, and report MAPE. The
  migration timing models already do this correctly (fit repeats 0–1, evaluate on
  repeat 2); the power law does not.
- **Heiser C.1 / "evaluating a model against itself."** `predict` *is* the
  simulator; there is no independent check. Fix: either build a genuinely
  independent analytic bound, or stop implying a cross-check exists.
- **Heiser A.3 — selective data range.** *"The interesting data range starts where
  the graph ends."* Our measured domain ends at 24,576 tokens; the simulator runs
  to 31,562 and 100% of interactive-coding moves are extrapolated. Fix: Phase C3.
- **Claiming a flat sweep proves insensitivity.** Floyd & Paxson: *"Finding that
  the simulation results do not change as the parameter is varied does not provide
  a definitive result."* Fix: sweep one factor at a time and say what was held.

Sweeping an unmeasured parameter is accepted in exactly four forms, all with
precedent: bound the endpoints by a physical limit and say which (SimAI stops at
400 Gbps because H100 PCIe is 512 Gbps); ground the swept value in a separate
measurement (Helix's iperf3 matrix); sweep it and show the conclusion is invariant
(Pollux injects synthetic network interference across a range); or state concretely
why measuring was infeasible (DistServe: no high cross-node bandwidth on the
testbed).

**And publish a disagreement if we find one.** Sia reports that Pollux performed
significantly *worse* on the physical cluster than its simulator predicted, and
diagnoses why. That is the single highest-credibility move in the corpus, and our
`model_check` median +78% error against TTFT is exactly such a disagreement — we
should diagnose and publish it rather than leave it uncomputed.

## 6. What the literature settles

**The motivation is citable, and the gap is named for us.** TAPAS (ASPLOS'25,
Microsoft Azure Research): *"Currently, live migration of GPU VMs is unsupported
due to the complexities of GPU memory management, but this capability would
enhance performance if implemented."*

**The timescale is real.** POLCA (ASPLOS'24) documents a **10 s deadline that UPSes
impose on power-capping response** against **out-of-band GPU control taking up to
40 s**. TAPAS's power emergency is an **immediate cut to 75% of capacity sustained
over 5 minutes**, where the baseline (uniform 35% frequency cap) costs **−35% IaaS
and −28% SaaS performance**. Utility scheduling intervals are 5–15 min. Our 120 s
deadline sits sensibly inside this.

**Nobody responds to a utility curtailment signal, and nobody moves work between
sites.** POLCA responds to a PDU breaker threshold; TAPAS to a UPS/AHU failure;
Power Stabilization (Microsoft+OpenAI+NVIDIA 2025) to a grid frequency spec.
Cross-region LLM work — SkyLB, GORGO, "AI Inference as Relocatable Electricity
Demand" — routes **new** requests and explicitly does not migrate live KV state.
Llumnix migrates, but intra-datacenter only, at 64 Gb/s, with no power objective.
Splitwise already moves KV machine-to-machine in **5–8 ms over InfiniBand** — the
transport primitive is published; it has just never been used for power.

**Llumnix justifies our mechanics:** migration is serialized per instance with a
pre-alloc/ACK/stage/commit handshake; downtime is one decode iteration and constant
in length while copy time scales with length; the cluster metric is free memory ÷
batch size.

**Skyplane grounds the effective-WAN sensitivity:** its cloud results put 5 Gbps
in a plausible central range and show route-dependent throughput. V1 represents
that evidence with one effective bandwidth and one fixed P50 RTT, not a TCP or
multi-edge network model. Skyplane also **relaxes, solves, and rounds down,
reporting ≤1% from optimal**: a direct precedent for our LP-guided rounding.

**Citation hygiene.** POLCA (arXiv 2308.12908) and "Characterizing Power Management
Opportunities for LLMs in the Cloud" (ASPLOS'24) are **the same paper** retitled.
Splitwise's headline "20% lower" is **cost**, not power. The 0.2–3 Hz power
oscillation spectrum is a **training** phenomenon — POLCA's own production data
shows inference is diurnal with 9% spikes at 2 s.

---

## 7. Open items I could not settle

- Whether the 84.9 W measured sleep floor and the 67.12 W curve floor come from
  the same configuration. This changes every reported shortfall.
- Whether `max_source_streams` is a vLLM limit, an LMCache limit, or a choice;
  the 1/2/4/8 sensitivity prevents that uncertainty from becoming a hidden
  constant.
- arXiv 2602.02987 (multiclass queueing with prefill/decode contention) may be the
  nearest prior art to the `f/F + g/G` coordinate. Abstract only; read it before
  claiming that coordinate is novel.
