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

## 1. The finding that decides whether this paper works

**We optimize half of a power ledger.** The profile measures what a migration
costs the destination, and we never put it in the objective.

Measured, from `profiles/gpt_oss_20b_a100_tp1.json` → `action_power_w`:

| Method | Source | **Destination** |
|---|---:|---:|
| Replay | 2.06 W | **189.84 W** |
| KV transfer | 3.18 W | **11.84 W** |

An A100's entire dynamic range in our own power curve is 248.89 − 67.12 =
**181.77 W**. A replica doing replay draws **1.04× the full dynamic range of a
loaded GPU**. Recomputing the reference run:

| Workload | Destination power added over the 115 s window | Source power relieved | Net |
|---|---:|---:|---:|
| Interactive coding | 6.70 kW | 2.75 kW | **+3.95 kW** |
| Coding | 33.00 kW | 7.22 kW | **+25.78 kW** |
| Agentic | 31.69 kW | 9.00 kW | **+22.68 kW** |

**Across two sites, power goes up by 2.4–4.6× what the source sheds, during
exactly the 115 seconds of the grid event.**

This is structural, not a bug. `PROPOSED_DESTINATION_ARCH.md:167` puts destination
facility power out of scope, `PlanResult` has no destination-power field,
`evaluate()` records only `initial_source_w` and `source_w_shed`, and
`simulate.py:481` computes destination node power and discards it. The optimizer
has an unpriced resource, so it always prefers replay (189.84 W, ~35 KB) over KV
(11.84 W, ~850 MB). It does, **98.7% of the time**.

Read the other way, this is the paper's most interesting axis and we have it
half-modeled: **replay and KV transfer are a power-versus-bandwidth trade.** Replay
costs 16× the destination power and almost no network; KV costs 16× less power
and ~850 MB. Pricing both is what makes the choice meaningful.

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
| Migration action power | replay 189.84 W dst; KV 11.84 W dst | measured, **unused** |
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
| `max_source_streams = 1` | profile | decides everything — see §3 |
| Service and migration are **independent** resources | `pool_planner.py:218-226` | a replica can be 98.7% migrating and 66.7% serving at once |
| Destination power is free | objective | §1 |
| Migration concurrency 1 per sink replica | `pool_planner.py` | untested |
| One request rate, 1/180 req/s/session | `destination_bench.py:46` | — |
| WAN is one scalar rate on three identical links | `destination_bench.py:232` | see §6 |
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
- **The pool path drops the destination ingest floor.** `formulation.md:363` and
  `planner.py:145` both specify `max(route, bytes/ingest_rate)`;
  `pool_planner.py:91` computes `route + residual` only. At 10 Gbps that
  underestimates a 17K-token transfer by **27%**; in the pressure search, which
  runs at 1000 Gbps, by **2.2×** — so the bandwidth axis is meaningless above
  ~5 Gbps.
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

> Given a destination that reports its residual capacity, this evacuation either
> fits or it does not, and here is exactly the residual it requires — and exactly
> what the transition costs in watts, bytes, and time at both sites.

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

**A1. Price destination power.** Add it to every table and add a destination power
cap as a constraint row. Re-run. Expect the method mix to flip toward KV transfer,
and expect the paper's real contribution to become the replay-vs-KV
power/bandwidth trade. *This is the highest-information change available.*

**A2. Couple service and migration** into one per-replica resource, or gate replay
to drained replicas as `PROPOSED_DESTINATION_ARCH.md:366` already requires.
Re-run; expect coding/agentic landed counts to fall.

**A3. Sweep `max_source_streams ∈ {1,2,4,8}`** and report landed sessions, watts,
and nodes drained. Cite `bounded-hardware-campaign-run`, not Llumnix.

**A4. Allow `final_state ∈ {sleep, off}`** in the destination planner and add a
node-emptying term to the objective. Report nodes drained as a first-class metric
— the column already exists in the 1M artifact.

**A5. Restore the ingest floor** in `pool_planner._destination_duration`.

**A6. Report the LP triple** (fractional bound / rounded / packed) plus an exact
integer optimum on 100–500-session instances, with the watt gap.

**A7. Ten seeds with bands; `faster` and `slower` profile cases.** Everything
quoted today is n=1 with declared input errors of 25–100%.

**A8. Put real requests in the bench** so the deadline check and the foreground
interference we measured actually bind.

**A9. Replace `bottleneck` with a binding set**; collapse the three identical route
rows into one; drive the pressure search with watts, not `all_sessions_landed`.

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

**Skyplane replaces our WAN placeholder:** AWS throttles **all egress to 5 Gbps per
VM**, GCP to 3 Gbps per flow and 7 Gbps total; reaching those needs **up to 64
parallel TCP connections**; throughput decays with RTT but is **stable over 18
hours**, which validates a static profiled grid. Their constraint structure —
per-link cap, per-VM egress × #VMs, per-VM ingress × #VMs — is what we should
adopt. They also **relax, solve, and round down, reporting ≤1% from optimal**: a
direct precedent for our LP-guided rounding.

**Citation hygiene.** POLCA (arXiv 2308.12908) and "Characterizing Power Management
Opportunities for LLMs in the Cloud" (ASPLOS'24) are **the same paper** retitled.
Splitwise's headline "20% lower" is **cost**, not power. The 0.2–3 Hz power
oscillation spectrum is a **training** phenomenon — POLCA's own production data
shows inference is diurnal with 9% spikes at 2 s.

---

## 7. Open items I could not settle

- Whether the 84.9 W measured sleep floor and the 67.12 W curve floor come from
  the same configuration. This changes every reported shortfall.
- Whether `max_source_streams` is a vLLM limit, an LMCache limit, or a choice.
- Whether a destination replica can serve and ingest simultaneously at all — we
  have one paired observation per method, not a percentile.
- arXiv 2602.02987 (multiclass queueing with prefill/decode contention) may be the
  nearest prior art to the `f/F + g/G` coordinate. Abstract only; read it before
  claiming that coordinate is novel.
