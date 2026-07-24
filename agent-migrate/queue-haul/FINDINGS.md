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
