# Queue-Haul findings

## Decision

The recovered 2026-07-23 archive changes the diagnosis but still does not
justify an accepted destination profile.

- Do not launch the current reserve; its task list is ignored and it would
  repeat the full campaign.
- Do not collect more migration timing for the measured low-concurrency case.
  The raw records support empirical-max replay and KV timing envelopes and a
  method-affinity rule for live traffic.
- Before any service rerun, replace the six invalid forced-token signatures and
  prevent repeated appended prompts from surviving in APC across cells. If a
  measured boundary is required, rerun only the unresolved frontier.

All 767 artifacts listed in `SHA256SUMS` match; the archive contains those
artifacts plus the manifest itself. The local `data/` copy is ignored by Git
and is not part of the evidence commit.

## Prefill, decode, and holding audit

The 2026-08-14 audit rejects a new decode-hold constraint for now. It also
rejects predicting numeric TTFT or TBT changes from `ell`, service slack, or
simulator queue debt. Keep these quantities separate:

| Quantity | Unit | Current status |
|---|---|---|
| prefill work `P` | context-conditioned prefill service-seconds per horizon | represented by the first destination-work coordinate |
| decode work `D` | context-conditioned decode service-seconds per horizon | represented by the second destination-work coordinate |
| resident state `K` | block-rounded KV tokens or bytes | represented separately as stock |
| active decode `N(t)` | scheduled sequence occupancy or sequence-seconds | not identified and not a session count |
| backlog `Q(t)` | offered requests/work waiting for admission | observable state, not a holding reservation |

The existing two-dimensional destination halfspaces can express independent or
coupled `P,D` budgets. Power remains a separate phase-rate model. Add an
independent, context-conditioned `N` row only if a matched hardware campaign
shows that it removes held-out false-feasible decisions beyond `P,D,K`. Do not
put `P,D,N,K`, transition debt, or power into one scalar.

### Complete serving-data inventory

| Corpus | Scale | What it establishes | What it does not establish |
|---|---:|---|---|
| A100 stage-1 prefill | 6 levels, 324 requests, one physical bundle | intrinsic concurrency-one TTFT versus prompt length | bursts, mixed load, independent uncertainty |
| A100 stage-1 decode | 27 levels, 3,577 requests, three physical context bundles | a context/concurrency latency cliff and saturation throughput | an independently repeated SLO boundary |
| v7 anchors | 18 runs across 4K/16K/24K and phase | context means, queue and running-width diagnostics | tail ITL; histogram labels were discarded |
| v7 mixed service | 113 executions, 9,181 requests | five descriptive cache-correct anchors | a boundary: 47 runs are missing-work invalid, 60 append-hot, one under-hit |
| v7 loaded migration | 27 executions, 36 foreground requests | matched qualitative replay-versus-KV interference | a percentile interference surface |
| H100 profile | one 16-request prefill burst and one 128-request decode burst | aggregate saturation anchors | production SLO capacity or context curves |
| stored network/frontier/handoff streams | 154,009 requests | first-network-response and inter-SSE stall diagnostics | true TTFT/ITL or open-loop stability |
| live-power-oneoff | one 1,816-request output-one storm | queued prefills can drive p90 TTFT to 27--103 s | decode behavior |
| live-power-shed | one mixed-load run, 7,166 engine samples | running/waiting occupancy across a handoff | request-level tail latency |
| continuation probes | 12,606 requests | post-migration continuation response | active-decode capacity |
| capacity reductions | 1,048 episode rows | migration deadline and drain outcomes | latency fitting; raw request/engine traces were not retained |
| migration microcampaigns | 819 scenarios | exact KV bytes, catch-up, concurrency gates, and migration timing | service hold or an eviction/residency SLO |
| phase-power calibration | 14 mixture measurements | phase direction can matter to source power | a validated joint power/SLO promise; grouped-CV RMSE is 7.99 W and the gate fails |

The stored stream corpus comprises 9,972 old network, 79,940 A100 frontier,
63,012 H100 hardware-gap, and 1,085 live-handoff requests. Its load generator
blocks after eight futures, records no scheduled arrival, and therefore becomes
closed/concurrency-capped under pressure. `first_byte_ns` equals request end for
8.1%, 44.4%, 99.9%, and 45.0% of those four groups because the client waits for
visible `delta.content`; the first SSE event is available separately. An SSE
event is not necessarily one token. The prefill-only pacing formula also uses
512 tokens for a measured 604-token prompt, understating offered prefill by
18%. These records are descriptive interference evidence only.

The evidence catalog is not complete for serving analysis: it omits
`requests.json`, `engine.csv`, `service_requests.csv`, H100 `prefill.json` and
`decode.json`, compressed episode/engine files, and west sink streams. Raw
serving arrays and all histogram labels must be checksum-cataloged before any
model promotion.

Power cannot repair the latency gap. The phase-aware source-power form is a
reasonable separate coordinate, but its held-out gate and the packed-power
gate both fail. Power coefficients are not service-facet weights, and service
debt cannot be converted into watts or milliseconds. A future paired service
campaign may sample power concurrently, but power and latency require separate
acceptance gates.

### Measured prefill/decode separation

The isolated A100 prefill staircase holds concurrency at one. Its p95 TTFT
rises from 34.321 ms at 256 input tokens to 65.838 ms at 1K, 283.493 ms at 4K,
589.069 ms at 8K, 1.954 s at 16K, and 4.618 s at 28K while throughput remains
2,430--2,981 input tok/s. This is an execution-length curve, not a prefill-storm
curve.

For the decode staircase, the following uses descriptive sample-p95 thresholds
of TTFT <=2 s across requests and ITL <=100 ms across pooled emitted token
gaps. The latter is token-weighted: it is neither per-request TPOT nor a
percentile over per-request gap-tail summaries. The repository's legacy policy
is p90, so this is an intentionally stricter diagnostic, not an SLO or
confidence bound.

| Context | Max-throughput cell | Pooled-ITL-safe cell | Joint diagnostic cell |
|---:|---|---|---|
| 256 | N=256, 3,774 tok/s, 1.681 s / 48.0 ms | same | same |
| 4,096 | N=256, 1,377 tok/s, 43.417 s / 629.1 ms | N=64, 812 tok/s, 37.7 ms ITL | N=4, 78 tok/s, 1.269 s / 23.0 ms |
| 8,192 | N=128, 630 tok/s, 56.112 s / 857.3 ms | N=64, 514 tok/s, 45.9 ms ITL | N=2, 42 tok/s, 1.801 s / 22.8 ms |

Here `N` is configured client concurrency, not measured active-decode
occupancy. Each request emits exactly 512 output tokens.

Each diagnostic cell is the threshold-passing tested concurrency with maximum
achieved output throughput, not a proven monotone boundary. The 4K and 8K joint
cells contain only 8 and 4 requests respectively. Exact completion would imply
4,088 and 2,044 token gaps, but the reducer did not retain the number of finite
ITLs it actually quantiled. Their request p95 TTFT estimates therefore have no
useful tail-confidence claim.

`service_profile_reduce.py` selects the maximum aggregate decode throughput
and ignores both latency columns. The profile then stores that aggregate rate
under concurrency key `1`, and `simulate.py` consumes it as a single-request
rate. At 256/4K/8K, the resulting deterministic decode durations per output
token are 0.265/0.848/1.587 ms versus measured concurrency-one pooled-token p95
ITL 22.206/22.484/22.016 ms, ratios of 84x/27x/14x. These are different
statistics, and the profile is not a calibrated tail model. The H100 profile
also reuses the A100 context tables despite having different H100 saturation
`F,G`. The simulator emits no token timestamps, so it has no simulated ITL/TBT.

The H100 runs are saturation anchors for another serving class. The prefill
run reports mean/p99 TTFT 5.458/11.464 s. The decode run reports 451.318 output
tok/s, mean TPOT 261.986 ms, p99 ITL 942.994 ms, and p99 TTFT 18.022 s. It used
GPU-memory utilization 0.9 and raw KV capacity 2,472,995, while the migration
stack uses 0.75 and advertises 1,205,376. They cannot share an SLO envelope.

### A/B and retrospective diagnostics

`outputs/service-holdout-20260814/summary.json` contains the reproducible
analysis. Four simulated traces have exactly 4,096 prefill and 8,192 decode
tokens over 30 seconds, identical `(p,d)=(0.09427,0.21665)`, and identical
modeled power. Smooth arrivals give p95 TTFT 45.6 ms; synchronous arrivals give
7.066 s. With the same arrivals and decode-token total, putting four long
decodes first versus last changes p95 TTFT from 7.750 to 5.012 s. This proves
that mean work does not determine this simulator's queueing result.

It does not validate a serving model. The simulator FCFS-serializes whole
requests, always reports peak active decode one, and gives the same p95
per-request modeled mean decode duration of 0.873 ms/output-token for every
shape. This is not TBT. It contains no token iterations, continuous
batching, chunked-prefill interference, sequence slowdown, preemption, or
output-length uncertainty.

A retrospective leave-one-context-bundle-out diagnostic on the 27 correlated
decode staircase cells compares profile-normalized achieved work with the
client-concurrency/throughput proxy
`1000 * concurrency / achieved_output_tps`. Work alone has 111.4 ms MAE,
80.6% MAPE, and four of 27 false-feasible cells at each 100 and 250 ms. The
proxy has 84.7 ms MAE, 39.0% MAPE, one of 27 false-feasible cells at 100 ms,
and three of 27 at 250 ms. Both use achieved throughput, and the profile
normalizer itself includes the held-context maxima. Client concurrency is not
active decode occupancy. This is a leaky retrospective diagnostic over only
three physical bundles, with no confidence interval or promotion value.

At the common v7 radius 0.096953, the four cache-correct runs have p95 active
decode occupancy from one to four while p90 per-request mean TPOT remains
22.646--23.687 ms. The fifth anchor at radius 0.114063 has p95 occupancy eight
and p90 mean TPOT 24.706 ms. Scalar work therefore does not identify hold, but
all five points pass and true token gaps are absent, so the data do not show
that hold is needed to predict an SLO boundary.

### SLO semantics

Freeze the target population, threshold, quantile, and window before the
validation campaign. Observations can establish feasibility and headroom. If
they help select a product tier, those runs become tuning data and fresh runs
must validate it. Report raw completed-request TTFT/true-ITL distributions,
service-failure counts, and attainment/goodput against the frozen SLO.

For request `i`, define `TTFT_i` over requests and `ITL_ij` over emitted token
gaps. Per-request mean TPOT is a third statistic. Freeze one decode statistic
`L_i` before collection, then define
`good_i = exact_completion_i AND TTFT_i <= tau_F AND L_i <= tau_D` and request
attainment as `sum(good_i) / N_offered`. Request goodput is separately
`sum(good_i) / measurement_seconds`; service-failure rate is
`N_service_failures / N_offered`. The offered-request denominator includes
timeouts, incomplete output, rejections, OOMs, and service crashes as misses.

The legacy classifier instead requires exact completion of every request and
checks the two marginal conditions `p90(TTFT_i) <= 2 s` and
`p90(mean_TPOT_i) <= 100 ms`. These marginal p90 conditions do not mean 90% of
requests meet both; without further information their joint lower bound is
80%. Label this a run-level legacy rule, not request goodput. A run label must
also include the frozen stability/drain gate. Physical process/cache-reset
runs are the independent units; requests and token gaps are correlated within
a run.

For a session SLO, first declare whether every eligible turn or a fixed fraction
of turns must be good, then divide good sessions by all offered sessions. Do not
condition any attainment denominator on successful completion.

Mean TPOT is not tail ITL/TBT. A product-facing decode statistic and threshold
must be selected and then frozen. The current honest output is binary membership
and slack in an advertised sensitivity envelope, not a predicted millisecond
change.

## Service evidence

The 47 summaries classified infeasible contain 50 requests with HTTP status
200, no recorded error, and zero prompt and output usage. The client requested
a forced token with `ignore_eos=true`, so these are not legitimate early model
stops. The stream ended without the required work or usage record. The other
9,131 requests reported their planned completion-token count. The archive did
not retain returned token identity, finish reason, or whether `[DONE]` arrived.
Consequently, it supports a forensic cache-geometry audit but cannot satisfy
the stricter completion-evidence contract now enforced for new service runs.

Every empty response is one of six deterministic
`(session, request_index, forced_token)` signatures. Their forced token IDs are
200110–200952; no successful request uses a forced ID at or above 200000. This
is a harness/token-eligibility defect, not evidence of overload. The
descriptive incidence is 11/470 agentic requests, 18/5,065 coding requests, and
21/3,646 interactive-coding requests.

This separates two questions that the old reduction conflated:

1. **Measurement validity:** an empty forced-token stream invalidates the
   probe; it does not establish production availability.
2. **Consumable capacity:** a run with missing work is not a capacity-boundary
   observation.

The 66 complete-work runs all pass normal, emergency, and stability, but
request completeness is not sufficient for service-capacity evidence. The
runner prewarms only each session's historical prefix, while repeated request
indices produce the same appended prompts across cells and vLLM APC is not
reset. Later cells can therefore reuse nominally future append blocks.

For 16-token blocks, a request is consistent with the intended private-prefix
state only when

```text
cached_tokens <= floor((prompt_tokens - input_tokens) / 16) * 16
```

The physical execution is the contamination unit: one append-hot request
excludes the whole execution. These are not statistically independent
replications because the campaign retained one vLLM process. The audit-only
`archived_cache_state` geometry reduction gives:

| Run state | Runs | Meaning |
|---|---:|---|
| measurement-invalid | 47 | at least one missing-work response |
| append-hot | 60 | at least one cached block extends into the new append |
| private-prefix consistent | 5 | every recorded cache count matches the intended warmed prefix |
| prefix under-hit | 1 | no append reuse, but one request lost intended prefix blocks |

Across all 9,181 requests, the corresponding request counts are 50 invalid,
8,020 append-hot, 1,110 private-prefix-consistent, and one prefix under-hit. Request
counts inside a contaminated run are descriptive only; they are not additional
independent evidence.

The under-hit is not private-prefix capacity evidence. Its request had zero
cached tokens instead of the intended 9,616, and its unarchived prewarm used
forced token 200740 in the same range where no measured request succeeded.
Ordinary eviction is implausible at that point in the sequential prewarm, but
the missing prewarm record prevents a causal claim. Exclude it.

Only five executions are private-prefix-consistent descriptive observations:

| Affinity | Split/cell | Radius | Requests | Cache state |
|---|---|---:|---:|---|
| interactive coding | fit/emergency | 0.114063 | 83 | private-prefix consistent |
| coding | tune/normal | 0.096953 | 78 | private-prefix consistent |
| agentic tool loop | tune/normal | 0.096953 | 27 | private-prefix consistent |
| interactive coding | validation/normal | 0.096953 | 48 | private-prefix consistent |
| agentic tool loop | validation/normal | 0.096953 | 11 | private-prefix consistent |

The recorded usage-based summaries for all five pass every policy. Their worst
p90 TTFT is 0.383 s, worst p90 mean TPOT is 0.0248 s/token, and largest
queue-drift upper bound is 0.00163 requests/s. The common descriptive point is
0.096953. Interactive coding also has one fit-only point at 0.114063. Because
stream completion is unobservable, these are sensitivity anchors, not strict
service evidence or an accepted envelope. Coding has one such execution, the
other affinities have two, and there is no private-prefix-consistent failure.

The earlier 0.108677–0.182805 affinity maxima came from append-hot runs and are
withdrawn as private-prefix capacity evidence. The data do not justify more
service facets or a learned cache-work model. For current analysis, keep
measured cold `F(T)`, `G(T)`, and KV capacity, and report:

- **descriptive anchor:** a legacy cache-geometry-consistent cell whose
  completion evidence is unobservable;
- **possible/sensitivity:** dependent on a descriptive anchor, assumed service
  envelope, monotonicity, or interpolation; or
- **unsupported:** outside the measured domain or based on an invalid or
  append-hot probe.

## Migration timing

Twelve of 18 migration treatments overlap a foreground request. Ten have one
request active at migration start, and two start a request during migration.
All 36 control and treatment foreground requests reuse cached prefixes; the 20
treatment requests reuse 2,608–3,168 cached tokens. Twenty-six of 27 first
engine samples record the intended 264,699 prompt tokens, 18 generated tokens,
and 18 successful prewarm requests. One control contains additional prior state
and is not cache-state-identical to its treatments. The data proves reuse, not
exact idle residency: the GPU cache gauge reports zero for evictable cached
blocks. The intended 264,699-token prewarm is 21.79% of declared capacity but
must not be modeled as measured resident occupancy. Six treatments have no
foreground overlap; four complete no foreground request at all.

The recorded `achieved_rho` remains the wrong timing covariate. It is a
preceding 30-second average divided by the rejected service bound, and it
counts cached prompt tokens as new compute. Subtracting cached prompts gives
prewindow compute rho from 0 to 0.898. Exact migration-interval foreground work
is not recoverable: aggregate request totals have no per-token timestamps,
engine counters include migration work, and 8/18 samplers end 39–222 ms before
route switch. The archive identifies overlap topology, not a load curve.

The smallest reliable duration models remain method-specific:

- Keep the reused replay compute-plus-completion context curve. On the six 16K
  rows, central and conservative calibration factors are 0.564571 and
  0.586660 after separating route bytes. Applied to the untouched 24K curve,
  they have 5.5% and 9.6% held-out median error and never underpredict the
  three held-out runs. One fitting context cannot identify a separate replay
  intercept.
- Model KV duration as `sealed_bytes / route_bytes_per_s + c`. The six 16K
  rows give `c=0.961186 s` centrally and `c=1.133822 s` conservatively. The
  24K held-out median errors are 2.4% and 7.8%; the conservative form never
  underpredicts.

A categorical overlap split improves central replay held-out median error from
5.5% to 4.2%, but its 16K idle fit contains one row. The analogous KV split
underpredicts two held-out rows. That is insufficient evidence for another
planner coefficient; the global conservative envelopes already cover both
idle and observed busy executions.

Exact route time must remain outside runtime calibration. The current
`LoadedCoefficients` multiplies the complete migration duration and can predict
less than `bytes/link_rate`, so it cannot safely encode the KV model. Replay
log transfer and switch time likewise remain physical components.

## Foreground impact and affinity

Control and treatment runs provide ten exactly matched foreground requests per
method, six of which overlap migration.

- For five already-active replay requests, paired TPOT increased by a median
  3.45 ms/token and at most 6.79 ms/token. The five KV counterparts increased
  by a median 0.42 ms/token and at most 0.67 ms/token.
- The one request arriving during replay reconstruction incurred 1.084 s
  additional TTFT. The same scheduled request arriving during KV transfer
  incurred 4.7 ms additional TTFT.

These are observations, not percentile bounds. They support a simple
eligibility rule: prefer KV transfer for latency-sensitive busy destinations;
allow replay only on an idle/drained destination or when explicit TTFT slack
covers the empirical replay penalty plus uncertainty. Foreground impact should
not be hidden inside a service-capacity facet.

## Remaining evidence

No additional migration campaign is needed for component timing or method
ranking on the recorded concurrency-one schedules in the measured
16K/10-Gbps and 24K/5-Gbps cells. Higher concurrency, a new foreground
schedule, continuous load claims, or an interference percentile requires new
evidence.

There is no strictly admissible service point or boundary. Retain the five
private-prefix-consistent executions as descriptive sensitivity anchors and
exclude the under-hit and append-hot cells. No rerun is needed for explicit
sensitivity modeling at 0.096953. Any accepted service evidence requires a
targeted rerun: select safe forced tokens, reset APC or make appended prompts
unique across cells, and hard-fail missing work, incomplete streams, or cache
reuse beyond the warmed prefix. Start near 0.096953, expand until a failure is
bracketed, and collect three independent runs only around each affinity's
boundary. Compute normal, emergency, and stability from every physical run.

The current vLLM 0.22.0 MP source and sink each report 963,152 KV tokens at
`gpu_memory_utilization=0.75`. The earlier checked-in 1,214,544-token value came
from a vLLM 0.10.1.1 configuration and is not capacity evidence for v7.
