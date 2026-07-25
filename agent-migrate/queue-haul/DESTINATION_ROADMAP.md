# What the destination model actually is, and what to fix

Plain-language companion to `PROPOSED_DESTINATION_ARCH.md`. No new notation.
Every number here was read out of the code or recomputed from the checked-in
profile and manifest on 2026-07-25.

## The short version

A destination site is five numbers, not a "capacity":

| # | Question | Row in the solver | Unit |
|---|---|---|---|
| 1 | Room to **hold** | `kv:<pool>` | KV blocks free |
| 2 | Room to **serve** | `service:<pool>` | service headroom |
| 3 | Room to **accept** | `migration:<pool>` | replica-seconds |
| 4 | Room to **receive** | `route:<link>` | bytes |
| 5 | (source side) Room to **send** | `source:<replica>` | stream-seconds |

That is the whole surface. Everything else in the architecture doc is either a
yes/no eligibility test before the solve, or a check after it.

**The one real problem:** we let the source fill a GPU to 0.531 and the
destination fill an identical GPU to 0.097. Same GPU, same model, same engine,
same coordinate. That 5.5× gap is not physics, and it is why interactive coding
"cannot fit." Section 3 covers it. Everything else is bookkeeping by comparison.

## 1. What we can model, and how well

These come from our own measurements. This is the part we should defend hard.

| Quantity | Value | Provenance | Error |
|---|---|---|---|
| Source power vs load | 12-point curve, 67.1 W idle → 248.9 W at ℓ=0.531 | measured | 5% |
| Prefill work coordinate `F` | 1448.32 tok/s | measured | 25% |
| Decode work coordinate `G` | 1260.38 tok/s | measured | 25% |
| KV capacity | 963,152 tokens/replica | vLLM 0.22.0 readback @ 0.75 util | 0% |
| KV block | 16 tokens | pinned | — |
| Sealed KV bytes | 12,582,912 B per 256 tokens (48 KiB/token) | measured | — |
| Replay time | `log_B/rate + 0.5867·(tokens/replay_tps + 1.340·(1+growth)) + switch` | fit on six 16K rows | 9.6% held-out median at 24K, never underpredicts |
| KV time | `sealed_B/rate + 1.1338 + catch-up` | fit on six 16K rows | 7.8% held-out median at 24K, never underpredicts |
| Replay foreground cost | +1.084 s TTFT to one arriving request; +3.45 ms/tok median TPOT | n=1 and n=5 | observation, not a percentile |
| KV foreground cost | +4.7 ms TTFT; +0.42 ms/tok median TPOT | n=1 and n=5 | observation, not a percentile |
| Measured migration domain | context 16,384–24,576 tok; link 5–10 Gbps | — | hard boundary |

The migration timing is genuinely good: two-parameter physical models, held out
to a different context and bandwidth, conservative by construction. That is a
publishable result on its own.

## 2. What we cannot model, and what we substitute

| Gap | What we do instead | Where | Honest label |
|---|---|---|---|
| Destination service capacity | fixed constant 0.096953, identical for normal/emergency/stable | `destination_bench.py:47` | **guess, and the wrong direction** |
| Cache-conditioned prefill | `f/F(T)` — appended tokens charged at the cold full-context rate | `destination.py:144` | normalization coordinate, not a physical bound |
| Cross-session prefix sharing | none; every session charged its full history | `pool_planner.py:222` | conservative by an unknown factor |
| Destination background load | knob `pressure.service ∈ [0,1]` scaling the bound | `destination_bench.py:66` | sensitivity axis, not a measurement |
| Concurrency > 1 | all timing from concurrency-one schedules | `_destination_duration` | out of domain |
| WAN | one scalar bytes/s, three identical links, no latency term | `destination_bench.py:232` | placeholder |
| Migration parallelism | 1 stream/source replica, 1 slot/sink replica | profile + `pool_planner.py` | assumed — but see §4, Llumnix backs it |
| Context beyond 24,576 | replay curve extended flat at its slowest rate | `extrapolate_replay` | flagged `unsupported_extrapolation`; **in-domain fraction is 0.00 for interactive coding, 0.34 for coding/agentic** |
| Failures, leases, rollback, cold model load | absent | — | out of scope, say so |

## 3. The 5.48× problem

This is the headline issue and it is simple.

Both bounds live on the **same** coordinate ℓ = f/F + g/G, with the **same**
F=1448.32 and G=1260.38, from the **same** profile file, for the **same**
A100/GPT-OSS-20B/TP=1 configuration.

| Side | Bound | What was actually observed |
|---|---|---|
| Source (`max_ell`) | **0.531358** | the last point of the power sweep — GPU at 248.9 W |
| Destination (`SERVICE_BOUND`) | **0.096953** | a v7 probe that passed TTFT/TPOT/stability |

Ratio: **5.481**.

`FINDINGS.md` is explicit that there is **no private-prefix-consistent failure
anywhere in the dataset**. Nothing was ever pushed until it broke. So 0.096953
is a *lower* bound on what the destination can do, and we are using it as an
*upper* bound. On the power curve, 0.096953 corresponds to an A100 drawing about
125 W of a measured 67–249 W range. That is not a full GPU.

What it costs us:

| Workload | Required sink service, at 0.096953 | at 0.531358 |
|---|---|---|
| Interactive coding | **393.5%** | **71.8%** |
| Coding | 70.1% | 12.8% |
| Agentic tool loop | 74.0% | 13.5% |

The entire "interactive coding cannot steady-host the workload" result is this
one constant. Under a matched bound it fits with room to spare.

Note this does *not* trivialize the paper. It relocates the contribution to the
transition — source streams, migration window, WAN — which is exactly the part
we actually measured well (§1). Steady state then becomes a clean sensitivity
axis ("what if the sink is already X% busy") instead of a disguised assertion.

**Recommendation.** Rename the constants to say what they are —
`SOURCE_SWEEP_MAX = 0.531358` and `SINK_PROBED_SAFE = 0.096953` — and report
every headline number as a band across the two. Then take the one measurement
that collapses the band (§6, Phase 2).

## 4. What the literature buys us

### Llumnix (OSDI '24) — justifies our migration mechanics

- Migration is **serialized per instance** with a pre-alloc → ACK → stage → commit
  handshake; the source "migrates requests to the destination continuously" one
  at a time. This is a real citation for `max_source_streams=1` and one migration
  slot per sink replica — currently our two most naked assumptions.
- **Downtime ≠ transfer time.** Downtime is one decode iteration and is constant
  in sequence length; total copy time scales with length. Our migration-occupancy
  row is copy time, which is the right choice.
- Their cluster-level metric is **freeness** `F = (M − ΣV)/B`: free memory over
  batch size. Memory is the primary residual; speed enters only as a divisor.
  Supports making KV the first-class destination number.
- **They never cross a WAN.** 16 GPUs, 4 VMs, 64 Gb/s, one datacenter. Nobody has
  measured cross-site live migration — that gap is our contribution, but it also
  means we cannot borrow a network number from them.
- vLLM KV blocks are small and non-contiguous (128 KB per 16 tokens for 7B/16-bit;
  1k tokens = 4k blocks), so they stage GPU→CPU into one buffer before sending.
  We have no staging row; the architecture doc already admits this.

### Skyplane (NSDI '23) — replaces our WAN placeholder

- **AWS throttles all egress to 5 Gbps per VM** (≤32 cores). **GCP: 3 Gbps per
  flow, 7 Gbps total egress.** Azure reaches ~16 Gbps NIC. Our single 10 Gbps
  pipe shared by 179 replicas is not any real deployment.
- Reaching those rates needs **up to 64 parallel TCP connections per VM**; one
  connection gets far less. Bandwidth is *provisioned*, not constant.
- Throughput **decays with RTT** — a geographic term we do not have at all.
- Throughput is **stable over 18 hours**, so a static profiled bandwidth grid is
  legitimate for a simulator. This validates our static-snapshot choice.
- Their constraint structure is exactly what we should copy: per-link cap,
  per-VM **egress** cap × #VMs, per-VM **ingress** cap × #VMs, per-region VM cap.
- They **relax the integers, solve the LP, and round down**, reporting ≤1% from
  optimal. That is a direct precedent for our LP-guided rounding — cite it and
  stop treating the heuristic as a weakness.

### Mooncake (ToS '25) — replaces our service bound

- SLO is **relative**: TTFT_P90 = 10× and TBT_P90 = 5× the latency of the same
  request running alone without interference. This is the principled definition
  of an admissible radius, and it is what our frontier rerun should target.
  (For reference, our source knee convention — within 25% of best median ITL —
  is far *stricter* than production practice.)
- **Goodput**: only requests that fully complete under SLO count. Matches our
  "landed" semantics.
- **Prediction-based early rejection**: a real destination refuses admission.
  Our eligibility predicate is the right shape.
- Production prefix cache hit ratio is **0.30–0.51** (512-token blocks, saturating
  near 0.5); >50% of blocks are never reused while some are hit tens of thousands
  of times. Our zero-sharing-credit KV charge is likely ~2× conservative.

## 5. What we are optimizing, and the change worth making

**Today.** `destination_bench.scenario` sets `power_limit_w` to the source power
with *every* session moved. So the target is always full evacuation, the LP
always returns `target_unmet` in `emergency` mode, and the search signal is the
binary `all_sessions_landed`. We then read off `sessions_landed` after the fact.
The one continuous quantity we care about — watts shed — is computed and
discarded.

**The change.** Stop claiming "these N sessions will be served acceptably at the
destination." We cannot back that; we have no fleet telemetry and we do not want
to model TTFT per request. Claim the inverse instead:

> Given a destination that reports its residual vector, this evacuation either
> fits or it does not, and here is the exact residual it requires.

Then the headline figure is **required destination residual vs. watts shed** —
one curve per workload, with the reader's own bound drawn as a horizontal line.
That plot is *invariant* to the 0.097-vs-0.531 argument: the reader supplies the
bound and reads off the answer. It removes the single biggest reviewer objection
instead of arguing with it, and it is exactly the posture Skyplane takes with its
profiled grid and Llumnix takes with reported freeness.

## 6. Roadmap

### Phase 0 — make the current run honest (days, no GPU time)

1. Split the service constant into `SOURCE_SWEEP_MAX` / `SINK_PROBED_SAFE`; run
   every headline at both and report a band.
2. Replace the `bottleneck` field (argmax of one row) with a **binding set**:
   every row ≥ 0.95, plus a count of source rows ≥ 0.95. One comma-separated
   column. Today it says "source constrained" while hiding two rows above 98%.
3. Collapse the three route links into one path-pressure row. They are identical
   by construction (`source-egress`, `wan`, `destination-ingress` all get the
   same rate), so three bars imply three independent bottlenecks that do not exist.
4. Report the **LP triple**: fractional bound / rounded / post-packing. We already
   solve the relaxation and throw the bound away.
5. Drive `pressure_search` with **watts shed**, not `all_sessions_landed`.
6. Make `emergency` differ from `normal`, or delete it. Right now all three modes
   get the same bound, so the label carries no meaning.

### Phase 1 — the residual vector and a real network (1–2 weeks, no GPU time)

7. Emit the five numbers per site per solve as first-class output.
8. Add Skyplane's network structure: per-replica egress cap, per-replica ingress
   cap, shared inter-site link. Default egress to the measured 5 Gbps/VM.
9. Add an RTT term to route time.
10. Give cross-session prefix sharing a credit knob defaulted to 0, with
    Mooncake's 0.30–0.51 as the sensitivity range.

### Phase 2 — the one measurement that settles it (~1 GPU-day)

11. The service-frontier rerun `FINDINGS.md` already specifies (safe forced
    tokens, APC reset, hard-fail on missing work), but targeting a **Mooncake-style
    relative SLO**: find the radius where p90 TTFT hits k× the isolated-request
    TTFT. Start at 0.096953, expand until failure is bracketed, three runs at the
    boundary. This converts the largest sensitivity axis into a measured number.

    Note this contradicts `FINDINGS.md`'s current "no rerun is needed." That is
    true for *sensitivity* modelling; it is not true for an NSDI claim about
    destination capacity.
12. Optional: a concurrency-2 migration probe, to test whether one stream per
    replica is a real limit or a configuration choice.

### Phase 3 — many sites (after Phase 1)

13. One source, 1–8 sinks, identical A100 pools first. Two topologies:
    independent paths, and a shared source-egress cut. Skew the sink baselines.
    Synthetic non-A100 profiles only after that, and visibly marked.
    No new "site capacity" abstraction — duplicate candidate columns and add rows.

### Phase 4 — solver evaluation

14. Exact integer optimum on small instances; LP fractional bound at all sizes;
    LP-rounded; post-packing; greedy. Report watt-gap to the bound, runtime,
    memory, and scaling in site count. Then, and only then, consider the
    residual-aware primal-dual greedy.

## 7. What the 10,000-session run actually shows

Recomputed at seed 0, reference pressure (10 Gbps, 115 s window, no background
load). Row usage is a fraction of capacity; values are the LP-rounded selection
before packing repair.

| | Interactive coding | Coding | Agentic tool loop |
|---|---:|---:|---:|
| Source replicas packed | 65 | 179 | 172 |
| Sessions per replica | 154 | 56 | 58 |
| Source service used | 83.9% | 14.4% | 19.3% |
| Source KV used | 94.5% | 99.4% | 99.5% |
| Median context | 5,751 | 17,249 | 16,537 |
| **Landed (LP / greedy)** | **3,808 / 1,687** | **7,746 / 7,731** | **7,829 / 7,812** |
| Sink service | **1.0000** | 0.667 | 0.705 |
| Sink KV | 0.339 | 0.772 | 0.780 |
| Sink migration occupancy | 0.543 | **0.987** | **0.987** |
| Shared path (all 3 links) | 0.0003 | **0.993** | **0.999** |
| Source streams ≥0.95 | 34 of 65 | 178 of 179 | 171 of 172 |
| Packing drops | 16 | 0 | 0 |
| Method split (LP) | 3,808 replay, 0 KV | 7,646 replay, 100 KV | 7,725 replay, 104 KV |

Three things the current plot hides:

- **Nothing is bound by one thing.** Interactive is service *and* stream bound.
  Coding and agentic are bound by streams, sink migration slots, *and* the WAN,
  all within 1.3% of full, simultaneously.
- **The "WAN at 99%" is not a migration budget, it is a KV budget.** The 7,646
  replay migrations use ~0.27 GB of the 143.75 GB window. The 100 KV migrations
  use ~140 GB, about 1.3 GB each. The LP buys back scarce source-stream seconds
  by spending WAN on a handful of sessions. Greedy makes 18 KV moves and leaves
  the WAN at 17% — and lands 15 fewer sessions.
- **The source packing decides the destination question.** Coding and agentic
  pack to 99.4% KV and only 14–19% service, so they arrive KV-shaped. Interactive
  packs to 84% service, so it arrives service-shaped. The workload is not choosing
  the bottleneck; the source packer is.

Greedy is only badly beaten in the one case where a resource genuinely saturates:
interactive coding, where it lands 1,687 vs the LP's 3,808 (2.3× more service per
admitted session) because it prices resources once from total offered demand and
never reprices. Where nothing saturates hard, it is within 0.2%.

## 8. The three decisions to confirm

1. **First scaling experiment stays one source → many sinks.** Yes. Multi-source
   routing adds a whole shared-cut problem we have no evidence for.
2. **Watts shed primary, sessions landed secondary.** Yes — and more strongly
   than the framing suggests: `sessions_landed` is not even monotone in resources,
   because a bigger budget lets the LP admit heavier sessions. Watts is what the
   paper claims.
3. **Non-A100 sites stay visibly synthetic until measured.** Yes.

## 9. What we should stop saying

- "Matching hardware." We match replica *count*, not capacity. Say so.
- "LP." It is an LP-guided rounded heuristic with packing repair. Skyplane does
  the same thing and calls it that; so should we.
- "Bottleneck." Report the binding set.
- "Unrounded KV" in `PROPOSED_DESTINATION_ARCH.md` — the code block-rounds now
  (`pool_planner.py:62`, `:152`). That documentation is stale.
- "Feasible" without qualification. Feasibility here means *source power and
  modeled deadline*; evidence status is a separate field and 0–34% of moves are
  outside the measured context domain.
